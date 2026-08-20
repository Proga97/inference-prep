# System Design Playbook — LLM Serving Interviews

Six worked scenarios covering ~90% of what serving/platform loops ask. Each follows the same five-step method — internalize the METHOD; the scenarios are reps.

**The method (35–40 min round):**
1. **Requirements (5 min):** traffic (req/s, prompt/output length distributions), SLOs (TTFT, ITL, availability), model(s), budget posture, growth. Interviewers grade the questions you ask.
2. **Capacity math (5 min):** weights, KV per request, per-GPU throughput ceiling, GPU count. Doing arithmetic unprompted is the single biggest differentiator.
3. **Architecture (15 min):** boxes and arrows — gateway → router → engine replicas; where batching, caching, autoscaling live.
4. **Deep-dive (10 min):** they pick a component; go deep with trade-offs.
5. **Failure modes & metrics (5 min):** volunteer these before being asked.

---

## Scenario 1 — "Serve a 70B model with sub-200ms TTFT at 1,000 req/s"

**Requirements to extract:** prompt length distribution (say median 1k, p99 8k), output ~300 tokens, streaming OK, TTFT 200ms p95, ITL ~40ms.

**Capacity math:** 70B fp16 = 140 GB → TP=4 on H100s per replica (320 GB: weights + KV headroom). Prefill of 1k tokens on 4×H100 ≈ tens of ms — fine; the 200ms budget is really a QUEUEING budget. Each replica sustains some X req/s (state you'd measure with a benchmark harness — e.g., ~50 req/s at these lengths); 1,000 req/s → ~20–25 replicas ≈ 100 H100s, then add 20% headroom.

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

## Rapid-fire variants (one-paragraph answers to prepare)

- **Multi-region serving:** route by geo + data residency; weights replicated per region; no cross-region KV (latency kills it) — sessions pin to a region; global anycast gateway; per-region autoscaling with cross-region spill only for full outage.
- **Streaming protocol choice:** SSE (simple, HTTP-native, proxy-friendly) vs WebSocket (bidirectional — needed only for interruption/voice); always send heartbeats; cancel generation server-side on disconnect (wasted decode = wasted money).
- **Model router / cascade:** classify request difficulty → small model for easy, big for hard; grade on cost saved at iso-quality; risks: classifier drift, added TTFT (keep the classifier tiny or use logprob-based early exit).
- **Capacity-plan N req/s on model M:** always the same skeleton — per-request KV, concurrent sequences per GPU from KV budget, tokens/sec ceiling from bandwidth, replicas = demand/ceiling × headroom. Practice until it's 3 minutes flat.

---

## Whiteboard drill schedule

One scenario per sitting, 35 minutes, actually drawing and talking. Weeks 12–16: cycle all six twice, then have someone (or a recording of yourself) play interruption-heavy interviewer — the skill under pressure is returning to the method's spine after each tangent.
