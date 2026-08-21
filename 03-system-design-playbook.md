# System Design Playbook — LLM Serving Interviews

Eleven worked scenarios covering what serving/platform loops ask. Scenarios 1–6 are the classics; 7–11 are the 2026 ones — MoE, reasoning models, million-token context, constrained decoding at scale, and mixed-regime platforms. Each follows the same five-step method — internalize the METHOD; the scenarios are reps.

**The method (35–40 min round):**
1. **Requirements (5 min):** traffic (req/s, prompt/output length distributions), SLOs (TTFT, ITL, availability), model(s), budget posture, growth. Interviewers grade the questions you ask.
2. **Capacity math (5 min):** weights, KV per request, per-GPU throughput ceiling, GPU count. Doing arithmetic unprompted is the single biggest differentiator.
3. **Architecture (15 min):** boxes and arrows — gateway → router → engine replicas; where batching, caching, autoscaling live.
4. **Deep-dive (10 min):** they pick a component; go deep with trade-offs.
5. **Failure modes & metrics (5 min):** volunteer these before being asked.

---

## Scenario 1 — "Serve a 70B model with sub-200ms TTFT at 1,000 req/s"

**Requirements to extract:** prompt length distribution (say median 1k, p99 8k), output ~300 tokens, streaming OK, TTFT 200ms p95, ITL ~40ms.

**Capacity math:** 70B fp16 = 140 GB → TP=4 on H100s per replica (320 GB: weights + KV headroom). Prefill of 1k tokens on 4×H100 ≈ tens of ms — fine; the 200ms budget is really a QUEUEING budget. Sanity-check any per-replica number against prefill FLOPs before you say it: at fp16, 1k-token prompts cost 2×70e9×1000 ≈ 140 TFLOPs each, and 4×H100 peak ≈ 4 PFLOPS — so ~10–15 req/s per replica is the honest fp16 figure, which puts 1,000 req/s at 300+ H100s. Getting to ~100 H100s requires saying the assumptions out loud: FP8 (≈2× compute, halved weights) plus a real prefix-cache hit rate on the shared system prompt. An interviewer who does this arithmetic is checking whether you do it too.

**Architecture:** gateway (auth, rate limits, admission) → router (least-KV-load + prefix affinity) → TP=4 vLLM replicas with continuous batching + chunked prefill + prefix caching → SSE streaming back. FP8 quantization halves the footprint and roughly doubles per-replica throughput — say this early, it changes the math.

**Deep-dive bait:** why TP=4 not PP (latency: TP splits bandwidth per token; PP doesn't); why chunked prefill (p99 prompts would stall decodes); prefix cache hit-rate assumptions (shared system prompt → TTFT for cached fraction collapses).

**Failure modes:** replica loss (drain + router health), burst > capacity (queue with TTFT-budget-aware shedding), long-prompt abuse (max length + admission control).
**Metrics:** TTFT p50/p95/p99 split into queue vs prefill, ITL, KV utilization, preemption count, cache hit rate, goodput.

---

## Scenario 2 — "Design GPU autoscaling that handles bursty traffic without thrashing"

**Frame immediately:** the crux is cold starts (2–10 min: provision + image pull + load 140 GB of weights) vs bursts that arrive in seconds. Autoscaling can't outrun a burst; the design absorbs bursts with buffers and pre-position capacity.

**Design:** (1) Scale signal: queue depth and KV utilization and TTFT SLO burn — explicitly NOT CPU% or GPU-util%. (2) Burst absorption: bounded queue sized to TTFT budget, admission control, overflow → 429/degrade to smaller model. (3) Warm pool: N idle loaded replicas (cost = insurance premium; size = burst p99 minus autoscale reaction). (4) Fast provisioning path: weights streamed from regional object store, pre-pulled images, snapshot/restore if available. (5) Predictive layer on diurnal shape. (6) Anti-thrash: asymmetric policy — scale up fast, scale down slow (e.g., 10-min sustained low), hysteresis band between thresholds, per-decision cooldown.

**Trade-off to state:** warm pool cost vs SLO violations cost — make it a number ("one idle TP=4 replica ≈ $X/hr vs revenue impact of Y% TTFT violations").
**Failure modes:** scale-down killing in-flight streams (drain gracefully), oscillation at threshold (hysteresis), regional capacity stockouts (multi-zone/multi-region spill).

---

## Scenario 3 — "p99 latency tripled, p50 unchanged. Walk me through it." (debugging round)

Run it as incident response, out loud:
1. **Scope (2 min):** which metric (TTFT vs total vs ITL), when did it start, what shipped (deploy? traffic mix? new customer?), which replicas/regions.
2. **Read the tail:** sample slow requests — what do they share? Long prompts? One tenant? One replica? Cache misses?
3. **Hypothesis ladder:** (a) long-prefill stragglers stalling batches → fix chunked prefill; (b) KV preemption (check counter) → memory pressure from longer contexts or bigger batch → raise headroom/admission; (c) queue spikes from bursty arrivals → autoscale/shed; (d) prefix cache hit-rate drop after deploy (cache cleared / prompt template changed by a customer); (e) one degraded replica in a TP group (stragglers serialize allreduce) → drain it; (f) gateway-side: Python GC, connection pool exhaustion, DNS.
4. **Verify with one graph each,** fix the top hypothesis, add a regression alert.

What's graded: structure, instrument-first instinct, and inference-specific hypotheses (preemption, prefill interference, cache miss storms) that a generic backend engineer wouldn't know.

---

## Scenario 4 — "Serve 200 fine-tuned variants of one 8B base model"

**The trap:** 200 full copies = 200 × 16 GB = 3.2 TB of VRAM. Don't fall in.

**Answer: multi-LoRA serving.** Fine-tunes as LoRA adapters (~10–200 MB each) over ONE shared base: base weights loaded once per replica; adapters hot-loaded to GPU on demand with an LRU adapter cache; per-request adapter ID; batched execution across DIFFERENT adapters in one forward pass (S-LoRA/Punica-style grouped GEMM — vLLM supports this natively). One TP=1 replica on an 80 GB card serves the base + dozens of hot adapters.

**Router layer:** adapter-aware routing (requests for hot adapters go where they're resident), cold-adapter load path (object store → host → GPU, ~100ms-class), per-tenant quotas.
**Deep-dive bait:** what if some tenants need full fine-tunes (separate dedicated pool); adapter eviction storms (pin top-K by traffic); quality isolation between tenants sharing a batch (none needed — math is independent per request).
**Extension they may add:** "some variants get 100× the traffic" → tiered: dedicated replicas for whales, shared multi-LoRA pool for the long tail.

---

## Scenario 5 — "Offline batch pipeline: summarize 50M documents/day as cheaply as possible"

**Reframe first:** no latency SLO → optimize tokens/sec/dollar. Everything flips vs chat serving.

**Design:** (1) Max batch: crank concurrency until KV is the binding constraint; no chunked prefill needed (nothing to protect), long scheduler queues are fine. (2) Quantize aggressively: FP8/int4 — validate quality once on a sample with task-specific evals. (3) Cheap capacity: spot GPUs with checkpointed progress (queue of doc IDs, idempotent writes — spot preemption just requeues), off-peak scheduling, older GPU generations if $/token wins. (4) Prefill-heavy workload (long docs in, short summaries out) → input-token throughput dominates; a compute-strong/cheaper card may beat H100 on $/token — say you'd benchmark $/1M tokens per GPU type. (5) Pipeline shape: object store → sharded work queue → stateless worker pods (vLLM, huge batch) → results store; scale by adding workers; dedupe/near-dupe docs first (free tokens). (6) Consider smaller model: distill or use 8B if evals pass — the single biggest cost lever.

**Metrics:** $/1M tokens, docs/hour, spot preemption rate, quality drift on a sampled eval.
**The senior move:** ask "do all 50M docs need the big model?" → cascade: small model first, escalate low-confidence ones.

---

## Scenario 6 — "RAG-heavy enterprise assistant: 10-turn conversations over big shared documents"

**Workload shape:** huge prefills (system prompt + retrieved chunks + history), short outputs; heavy prefix overlap (same system prompt org-wide; same doc chunks across users; growing per-session history).

**Design around the KV cache:** (1) Prefix caching as the load-bearing feature: stable prompt template ordering — system prompt first, then docs, THEN user-specific content, so shared prefixes actually match (cache keys are exact-prefix). (2) Session affinity routing so turn N hits turn N−1's KV blocks; multi-turn TTFT then scales with the NEW tokens only. (3) For org-wide hot documents: replica-set-level cache warming, or a Mooncake-style shared KV tier if scale justifies. (4) Chunked prefill mandatory (20k-token cache-miss prefills would wreck ITL for everyone). (5) Long-context memory math out loud: 32k context on a GQA 8B ≈ 4 GB KV per sequence — session limits, eviction policy for idle sessions, maybe fp8 KV. (6) The retrieval system itself is out of scope but SAY that its chunk stability affects cache hit rate — reordering retrieved chunks between turns silently kills caching.

**Metrics:** cache hit rate (the north star here), TTFT by cache-hit vs miss, KV utilization, per-session memory, ITL p99.

---

## Scenario 7 — "Serve a DeepSeek-V3-class MoE model"

**The reframe that earns the round:** total params set memory, active params set compute. ~671B total / ~37B active means you need the memory of a giant and the FLOPs of a mid-size model — and every intuition from dense serving needs re-deriving.

**Capacity math:** FP8 weights ≈ 671 GB. An 80 GB H100 gives ~70 GB usable → 10–12 GPUs minimum for weights alone, realistically 16+ with KV and headroom. That's two NVLink-connected nodes with InfiniBand between them. Note out loud that those GPUs are memory-holders more than FLOP-providers — which is what drives the rest of the design.

**Architecture:** expert parallelism as the primary axis (all-to-all dispatch and combine replacing TP's allreduce), plus TP within the attention layers. Then disaggregate: prefill routes many tokens through every expert and is compute-bound; decode routes one token per sequence and is all-to-all-bound. They want different EP widths and batch shapes, so give them separate pools.

**Deep-dive bait:** expert load imbalance — routing is learned, not uniform, so hot experts stall the all-to-all barrier and set your step time; fix with redundant hot-expert replicas and periodic re-placement from observed routing stats. Also: why large batch is *mandatory* here (expert utilisation), and why that fights your latency SLO.

**Failure modes:** one slow GPU stalls every step (all-to-all is a barrier); routing distribution shift after a model update invalidates your placement; capacity-factor overflow silently dropping tokens.
**Metrics:** per-expert token counts, all-to-all time as a fraction of step time, EP imbalance ratio, tokens/sec/GPU.

---

## Scenario 8 — "Serve a reasoning model, p50 8k output tokens"

**Everything from Scenario 1 inverts.** Say that first, then show it.

**Capacity math:** at 300 output tokens the fleet is prefill-heavy; at 8k it is overwhelmingly decode-bound. Redo the arithmetic: decode time now dominates total latency, so ITL is the SLO that matters and TTFT becomes almost irrelevant. KV grows throughout a single request — a sequence cheap at admission is expensive 20k tokens later, which breaks admission control that only looks at prompt length.

**Design consequences:** (1) preemption becomes very costly — recomputing a 20k-token prefix is enormous, so swap-to-CPU starts beating recompute, and you need to bias the scheduler against preempting deep sequences; (2) prefix caching loses value *per token* because the shared prefix is a shrinking fraction of the total; (3) speculative decoding gets MORE attractive — long low-entropy reasoning chains give high acceptance rates; (4) admission control must predict or bound output length, or long sequences starve short ones.

**Product lever worth naming:** reasoning-effort / thinking-budget controls. Cost per request is now mostly a product decision, not an infrastructure one.
**Metrics:** ITL p99 above all, KV growth rate per sequence, preemption depth distribution, tokens generated per dollar.

---

## Scenario 9 — "1M-context code assistant"

**Capacity math first, because it's brutal.** KV at 1M tokens on a GQA 8B: 128 KB/token × 1M ≈ 128 GB for ONE sequence. That doesn't fit an H100. So the honest opening is "single-GPU is off the table; here's what I'd actually do."

**Levers, ranked:** context parallelism to shard the sequence across GPUs for prefill; KV quantisation to fp8 (halves it immediately); architectural help if you get to choose the model — sliding-window or hybrid attention, attention sinks, MLA-style latent KV; KV eviction (H2O/SnapKV-style) accepting quality loss on evicted spans; and aggressive prefix caching, since a code assistant re-sends nearly the same repo context every turn.

**The insight to volunteer:** for this workload, cache hit rate is the whole ballgame. The user's context barely changes between turns — if you route them to the pod holding their KV and never evict it mid-session, you turn a 1M-token prefill into a few thousand new tokens. Session affinity stops being an optimisation and becomes the architecture.

**Failure modes:** one long session pinning an entire GPU's memory; eviction mid-session forcing a catastrophic re-prefill; prefill of a cold 1M context blocking every other request (chunked prefill mandatory).

---

## Scenario 10 — "100% valid JSON at 500 req/s"

**Requirements to extract:** how complex is the schema (flat vs deeply nested), is it fixed or per-request, and what's the latency budget. A fixed flat schema and a per-request nested one are different systems.

**Mechanism:** compile the schema to an FSM (pushdown automaton if nested) over the vocabulary, mask logits each step so only valid tokens survive. Precompute and cache the compiled masks — compilation per request at 500 req/s is a non-starter, so key the cache by schema hash.

**Where the cost lands:** mask application is CPU work per request per step. At high batch this can become the decode bottleneck while the GPU sits idle — the interesting half of the answer. Mitigations: overlap masking with GPU compute, batch mask application, and pre-warm the compile cache for known schemas.

**Interactions (this is what separates the answer):** constrained decoding vs speculative decoding — naive drafting craters acceptance because draft tokens violate the grammar; the fix is applying the same mask to the draft (grammar-aligned drafting), which production stacks do; vs prefix caching — fine, they're orthogonal; vs batching — requests with different grammars can't share mask work. Also tokenizer boundaries: a valid JSON *string* isn't always a valid *token sequence*, which is where naive implementations produce invalid output.

**Metrics:** schema-validity rate (should be 100% — if not, your FSM is wrong), mask time as a fraction of step time, compile-cache hit rate.

---

## Scenario 11 — "One platform serving LLMs, embeddings and rerankers"

**The point of this question:** three completely different regimes on one fleet and one on-call rota. Name the differences before designing anything.

**The regimes:** LLMs are autoregressive, KV-cached, latency measured in TTFT/ITL. Embeddings are single-forward-pass, no KV cache at all, throughput-bound — and under *naive* padded batching, dominated by padding waste; the fix is length bucketing or varlen/packed attention, and naming both marks you as someone who has actually run one. Rerankers are cross-encoders: one forward pass per (query, document) pair, so latency scales with candidate count and the budget is usually tens of milliseconds inside a RAG request.

**Architecture:** don't force one engine to do all three. Separate pools with a shared gateway, shared auth, shared metrics, shared deploy pipeline. Bin-pack the small embedding and reranker models onto fewer GPUs (they're tiny); give LLMs dedicated cards. One control plane, three data planes.

**Deep-dive bait:** why length bucketing matters so much for embeddings (padding waste is the dominant inefficiency); why reranker latency budgets force you to cap candidate count; and how a RAG request's total budget gets divided across retrieval, rerank and generation.

**Metrics per regime:** LLM — TTFT/ITL/goodput. Embeddings — sequences/sec and padding efficiency. Reranker — p99 for a fixed candidate count.

---

## Rapid-fire variants (one-paragraph answers to prepare)

- **Multi-region serving:** route by geo + data residency; weights replicated per region; no cross-region KV (latency kills it) — sessions pin to a region; global anycast gateway; per-region autoscaling with cross-region spill only for full outage.
- **Streaming protocol choice:** SSE (simple, HTTP-native, proxy-friendly) vs WebSocket (bidirectional — needed only for interruption/voice); always send heartbeats; cancel generation server-side on disconnect (wasted decode = wasted money).
- **Model router / cascade:** classify request difficulty → small model for easy, big for hard; grade on cost saved at iso-quality; risks: classifier drift, added TTFT (keep the classifier tiny or use logprob-based early exit).
- **Capacity-plan N req/s on model M:** always the same skeleton — per-request KV, concurrent sequences per GPU from KV budget, tokens/sec ceiling from bandwidth, replicas = demand/ceiling × headroom. Practice until it's 3 minutes flat.

---

## Whiteboard drill schedule

One scenario per sitting, 35 minutes, actually drawing and talking. Cycle scenarios 1–6 until fluent, then 7–11 — and cap any one scenario at ~4 passes; past that you're memorising the scenario, not the method. Then have someone (or a recording of yourself) play interruption-heavy interviewer — the skill under pressure is returning to the method's spine after each tangent.
