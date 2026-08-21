# Question Bank — Worked Answers

51 questions with model answers, grouped the way interviewers group them. Sections I and J are the 2026 additions — MoE serving, constrained decoding, multimodal, determinism, reasoning workloads, KV offload, benchmarking methodology and regression testing. Method: read the question, answer OUT LOUD first, then compare. An answer you can only recognize, not produce, is not yet yours. Numbers are part of the answer — inference interviews reward candidates who compute.

---

## A. Inference fundamentals

**A1. Why is decode memory-bandwidth-bound while prefill is compute-bound?**
Every decode step for one sequence processes ONE token, but must read every model weight from HBM to compute it. Arithmetic intensity ≈ 1 FLOP per byte read at fp16 (2 FLOPs per weight ÷ 2 bytes per weight; ~2 FLOPs/byte only at fp8) — far below the GPU's ridge point (an H100 needs ~300 FLOPs/byte to saturate compute), so the GPU idles waiting on memory. Prefill processes all N prompt tokens in one pass: the same weights are reused across N tokens, so intensity scales with N and quickly crosses into compute-bound. Consequences: decode speed ≈ bandwidth/model-bytes; batching raises decode intensity (same weights amortized over B sequences); quantization speeds decode (fewer bytes) but barely helps prefill.

**A2. Estimate batch-1 decode tokens/sec for Llama-3-8B fp16 on an H100.**
Weights: 8B × 2 bytes = 16 GB. H100 SXM HBM bandwidth ≈ 3.35 TB/s (PCIe variant is 2.0 — say which you mean). Ceiling ≈ 3350/16 ≈ ~210 tok/s; real systems get 60–80% of that (KV reads, kernel overhead). The method matters more than the exact number: *bytes moved per token / bandwidth*.

**A3. What are TTFT, TPOT/ITL, and goodput? Which do you optimize for chat vs batch?**
TTFT = time to first token (dominated by queueing + prefill). TPOT/ITL = time per output token / inter-token latency (dominated by decode). Total latency = TTFT + TPOT × output_len. Goodput = throughput that meets SLO (throughput excluding requests that violated latency targets) — the honest production metric. Chat: TTFT (< ~500ms feels instant) and steady ITL (~30–50ms reads faster than humans). Offline batch: pure tokens/sec/GPU = cost; latency irrelevant, so max out batch size.

**A4. FLOPs per generated token?**
≈ 2 × parameter count (one multiply + one add per weight), plus attention FLOPs which grow with context. So Llama-3-8B ≈ 16 GFLOPs per token. Used for: prefill time estimates (N tokens × 2P FLOPs / GPU FLOPS) and MFU calculations.

**A5. Why does throughput-vs-latency form a curve, and how do you pick an operating point?**
Bigger batches amortize weight reads → more tokens/sec/GPU (cheaper), but each request's tokens come slower and queueing adds TTFT. Small batches → snappy but expensive. You pick the largest batch whose p99 still meets the SLO — that's the cost-optimal point. Being able to SAY "I'd plot the curve and pick the knee relative to the SLO" is the expected answer shape.

**A6. Walk through temperature, top-k, top-p, min-p. What do you use for code vs creative writing?**
Temperature divides logits before softmax: <1 sharpens (more deterministic), >1 flattens. Top-k keeps only k highest-probability tokens. Top-p (nucleus) keeps the smallest set with cumulative probability ≥ p — adapts to the distribution's shape where top-k doesn't. Min-p keeps tokens above a fraction of the max token's probability. Code/math: low temp (0–0.3), maybe greedy — correctness has few forms. Creative: temp 0.7–1.0 + top-p 0.9. Also know: pure greedy tends to loop/repeat; repetition penalties exist; modern engines sample on-GPU — the per-step CPU cost story belongs to constrained-decoding mask computation, not vanilla sampling.

---

## B. KV cache

**B1. Why does the KV cache exist and what would decode cost without it?**
Attention for the new token needs K,V of ALL previous tokens. Without cache you recompute every layer's K,V for the whole prefix each step: token t costs O(t) recompute → O(n²) total FLOPs in extra work. With cache: compute K,V once per token, store, and each step is O(1) new work plus O(t) cache *reads*. It trades memory for compute — and creates the central memory-management problem of serving.

**B2. Derive KV cache size. Compute for Llama-3-8B at 8k context, batch 32, fp16.**
Per token per layer: 2 (K and V) × n_kv_heads × head_dim × bytes. Llama-3-8B: 32 layers, 8 KV heads (GQA), head_dim 128 → 2×32×8×128×2 = **128 KB/token**. × 8192 tokens = **1 GB per sequence**. × 32 = **32 GB** — double the 16 GB of weights. Punchline: at long context, KV, not weights, is what you're managing; it caps batch size and thus throughput.

**B3. MQA vs GQA vs MLA?**
All shrink KV. MQA: 1 shared KV head — max savings (n_heads×), some quality cost. GQA: query heads share grouped KV heads (Llama-3-8B: 32Q/8KV = 4× smaller) — the industry default, near-MHA quality. MLA (DeepSeek V2/V3): stores a low-rank latent instead of full K,V, decompressing on the fly — ~90%+ reduction, needs architecture co-design. Effect chain: smaller KV → bigger batch → more throughput per GPU.

**B4. What is PagedAttention and what problem does it solve?**
Classic systems preallocated contiguous KV for max_len per request → internal fragmentation (short outputs waste the tail), external fragmentation, no sharing; measured waste 60–80%. vLLM applies OS paging: fixed-size KV blocks (e.g., 16 tokens), a block table per sequence maps logical→physical, blocks allocated on demand; waste <4%, and identical prefixes share blocks copy-on-write. Effective batch size jumps → the headline 2–4× (the '24×' figure is the 2023 vLLM-vs-HF-Transformers comparison — quote it as history, not a current engine gap). Cost: block-table indirection in the attention kernel, and paged layouts complicate kernel design — a fair "downsides?" answer.

**B5. What is prefix caching and when does it pay off?**
Persist KV blocks of common prefixes (hash- or radix-tree-keyed, cf. SGLang's RadixAttention) so a request hitting the same prefix skips that prefill. Pays off with shared system prompts, few-shot templates, multi-turn chat (prior turns are the prefix), RAG with repeated document contexts. TTFT drops roughly proportional to cached fraction. Requires: eviction policy (LRU over blocks), and routing that sends same-prefix requests to the same replica — session affinity — or a distributed KV layer (Mooncake-style) shared across replicas.

**B6. vLLM runs out of KV blocks mid-decode. What happens?**
Preemption. The scheduler evicts a running sequence: either **recompute** (drop its blocks, re-prefill later — cheap to store, costs compute) or **swap** to CPU RAM (costs PCIe transfer + host memory). vLLM v1 defaults to recompute. Symptom of heavy preemption: p99 collapse while p50 looks fine — worth volunteering in debugging questions; the counter is exposed in vLLM metrics.

---

## C. Batching & scheduling

**C1. Static vs continuous batching — why is continuous strictly better for LLMs?**
Static: batch formed at admission, held until ALL finish — short outputs wait for the longest (head-of-line blocking), GPU slots idle as sequences finish. Continuous (Orca's iteration-level scheduling): after EVERY decode step, finished sequences exit and queued ones join. GPU stays full; the published ~20× figure is vs *naive* (no-batching) serving in the 2023 Anyscale writeup — the honest current claim is that every serious engine does this and static batching is simply not a competitor. Every modern engine (vLLM, SGLang, TensorRT-LLM, TGI) does this. Know the term "iteration-level scheduling."

**C2. What is chunked prefill and what does it fix?**
Problem: a 20k-token prefill entering a mixed batch stalls every ongoing decode for hundreds of ms → ITL spikes for everyone. Chunked prefill splits long prefills into chunks (e.g., 512 tokens) batched alongside decode steps: decode ITL stays smooth, prefill compute fills leftover capacity. Trade-off: that request's own TTFT rises slightly. This is the standard fix for "decode stutters when big prompts arrive" — recognize that symptom.

**C3. How would you implement a dynamic batcher (the coding-round version)?**
Queue + flush policy: flush when batch reaches max_size OR oldest item waited max_wait_ms. Asyncio: requests put (input, future) on a queue; a scheduler loop drains up to max_size with a timeout, runs the model, resolves futures. Tune max_wait against the TTFT SLO (it's pure added latency at low traffic). Practice this — it's a real Baseten-style prompt and it's basically mini-vLLM's core loop.

**C4. One GPU, mixed traffic: interactive chat + a nightly batch summarization job. Options?**
Priority scheduling with preemption (chat preempts batch); separate queues with token-budget shares per step; chunked prefill so batch jobs' long prefills don't spike chat ITL; KV reservation so batch can't starve chat of blocks; or time-slice (batch only off-peak). Best single-GPU answer: priority + chunked prefill + per-class KV quota, with goodput per class as the metric. If allowed two pools, physically separate them — isolation beats scheduling cleverness.

---

## D. Quantization

**D1. Weight-only vs weight+activation quantization — mechanics and when each wins.**
Weight-only (GPTQ, AWQ, llama.cpp K-quants): weights stored int4/int8, dequantized to fp16 in the kernel; compute stays fp16. Wins where decode is memory-bound — 4× fewer weight bytes ≈ up to 4× faster decode + smaller VRAM. Doesn't accelerate compute-bound prefill. Weight+activation (INT8 SmoothQuant, FP8): both sides quantized → uses int8/fp8 tensor cores → speeds up compute too (prefill AND large-batch decode); needs calibration for activation scales. Rule of thumb: latency-sensitive single-stream → weight-only int4; high-throughput serving on H100 → FP8 everything.

**D2. What are activation outliers and why do they break naive INT8?**
LLMs (>~6B) develop channels with activation magnitudes ~100× typical. Per-tensor INT8 scales to the max → normal values get crushed into few levels → accuracy collapses. LLM.int8: keep outlier channels in fp16, quantize the rest. SmoothQuant: migrate the scale from activations into weights offline. AWQ's related insight: protect the ~1% of weights aligned with large activations by scaling. Naming "outlier channels" unprompted is a strong signal.

**D3. Expected quality by scheme?**
INT8 weight-only: ≈ lossless. FP8 (E4M3 weights/activations): ≈ lossless with calibration — the H100/H200-era serving default. Below that, 4-bit float formats (NVFP4/MXFP4 on Blackwell) are no longer niche — models now ship natively in MXFP4 — so keep the 'below 4-bit is niche' claim for INT formats only. INT4 GPTQ/AWQ: small but real regression, worse on math/code/long reasoning chains; fine for chat. Below 4-bit: noticeable damage, niche. Always caveat: "I'd verify on OUR evals, not perplexity — perplexity hides task-specific regressions." That sentence alone marks you as production-minded.

**D4. Quantizing the KV cache — what and why?**
KV cache can be stored fp8 (or int8) independently of weights: halves cache size vs fp16 → doubles feasible batch/context; decode reads the cache every step, so it saves bandwidth too. Quality impact small but nonzero (attention becomes slightly noisy); supported in vLLM via `kv_cache_dtype=fp8`. Nice differentiator answer because most candidates only think of weights.

---

## E. Speculative decoding

**E1. Explain speculative decoding end to end. Why is the output distribution exactly preserved?**
Small draft model autoregressively proposes γ tokens (cheap). Target model runs ONE forward pass over all γ positions in parallel (like prefill — cheap per token). Each draft token accepted with probability min(1, p_target/p_draft); on first rejection, the rest are discarded and a corrected token is sampled from the residual distribution max(0, p_target−p_draft) (normalized). Rejection sampling makes accepted+corrected tokens *provably distributed exactly as target-only sampling* — the Leviathan et al. theorem. It converts memory-bound serial steps into parallel verification, spending idle FLOPs to buy latency.

**E2. The math: expected tokens per target pass, and when it's a net loss.**
With per-token acceptance rate α and draft length γ: E[tokens] = (1−α^(γ+1))/(1−α). (An idealisation — real acceptance is content- and position-dependent, so say 'assuming i.i.d. acceptance' and you're ahead of the question.) α=0.8, γ=4: (1−0.328)/0.2 ≈ 3.36 tokens per target pass vs 1 without. Net loss when: α low (unaligned draft — wasted draft compute + discarded verification), batch already large (decode is compute-bound; there are no idle FLOPs to spend), or draft too slow relative to target. Optimal γ grows with α; production systems tune γ dynamically.

**E3. Draft-model vs Medusa vs EAGLE vs n-gram lookup?**
Draft-model: separate small LM — flexible, but needs a well-aligned model and doubles ops burden. Medusa: extra decoding heads on the target predicting several future tokens — no second model, tree attention verifies candidates. EAGLE: lightweight autoregressive head over the target's hidden states — best-in-class acceptance; note also MTP (multi-token-prediction) heads, which DeepSeek-class models ship natively and which are the default draft for those models in 2026. N-gram / prompt-lookup: propose by copying matching spans from the prompt — free, shines in RAG/summarization where output quotes input. Ranking trade-offs across these is a senior-signal answer.

---

## F. Parallelism & distributed serving

**F1. Llama-3-70B fp16 — minimum and typical H100 deployment?**
Weights: 140 GB → doesn't fit one 80 GB H100. TP=2 gives 160 GB total = only 20 GB left for KV (tight). Typical: **TP=4** (320 GB: 140 weights + ~180 for KV/activations) or TP=8 for latency (more aggregate bandwidth → faster decode). Or quantize: FP8 halves weights to 70 GB — TP=2 becomes comfortable on H100s; note that a single 141 GB H200 fits 70B-FP8 with KV room, which changes the answer if you're allowed newer hardware. Show the arithmetic; that's the point of the question.

**F2. Tensor vs pipeline parallelism for inference.**
TP shards each matmul across GPUs; every layer ends in an allreduce (2 per transformer block) → needs NVLink-class interconnect, effectively intra-node (≤8 GPUs); splits the per-token bandwidth load → REDUCES latency. PP assigns contiguous layers to stages; only activations cross stage boundaries → tolerates slower links, scales across nodes; per-token latency ≈ unchanged (still traverses all layers) and bubbles hurt unless batches keep stages fed → raises THROUGHPUT/capacity, not speed. Default mental model: TP within a node, PP across nodes, both only as forced by memory or latency targets.

**F3. Why disaggregate prefill and decode (DistServe/Mooncake)? What moves between pools?**
Interference: prefill is bursty compute that stalls decode ITL; the two phases also want different parallelism and batch shapes. Disaggregation: prefill pool optimizes TTFT, decode pool optimizes ITL/throughput, scaled independently (traffic shifts the prefill:decode ratio). Cost: the prompt's KV cache must ship prefill→decode — B2's math says ~128 KB/token for 8B-class, ~2.5× that for 70B-class (80 layers → 320 KB/token): a 4k prompt ≈ 0.5–1+ GB per request → needs NVLink/InfiniBand/RDMA and layer-wise streaming overlap. Mooncake generalizes this into a cluster-wide KV store (prefix reuse across machines). Cite the KV-transfer number — it's what makes the answer real.

**F4. When does a single replica stop being enough, and what does the fleet look like?**
Single replica caps at (KV VRAM / per-request KV) concurrent sequences within SLO. Beyond: replicate the engine behind a router. Router concerns: least-loaded routing on queue depth/KV utilization (not round-robin); session/prefix affinity to preserve cache hits; draining on deploys; per-replica health from engine metrics. Multi-model fleets add model routing and bin-packing small models per GPU (MIG or MPS for hard/soft isolation).

---

## G. Production operations & debugging

**G1. p99 latency spiked, p50 flat. Debug it.**
Structure first: "p50 flat means the median path is healthy; something affects a subset." Then hypotheses in order: (1) long-prompt stragglers — check request-length distribution vs latency correlation; fix: chunked prefill, length-based routing. (2) KV preemption under memory pressure — check preemption/eviction counters; fix: more KV headroom, admission control. (3) Queueing bursts — p99 wait time vs arrival spikes; fix: autoscaling headroom, load shedding. (4) Prefix-cache miss storms after deploy/eviction (TTFT bimodality). (5) Infra tail: one bad replica in TP (stragglers serialize the group), thermal throttling, noisy neighbor, Python GC in the gateway. Name the observability you'd want — per-stage timing (queue/prefill/decode), preemption count, cache hit rate — and you've passed.

**G2. Design GPU autoscaling for bursty LLM traffic.**
Signals: queue depth, KV utilization, tokens-in-flight, TTFT SLO burn — never CPU%, and GPU-util% is also misleading (a memory-bound decode shows high util at low goodput). Cold start is the crux: pull image + load 16–140 GB of weights = 2–10 min. Mitigations: warm pools (loaded, idle — costs money, saves SLO), streaming weight load from object store (Run:ai-style streamer, tuned image layout), container/GPU snapshotting, over-provisioned headroom (e.g., +20%), predictive scaling on daily traffic shape. Scale-down: slow drain with connection close, hysteresis window to prevent thrash (thrash = paying cold-start cost repeatedly at the oscillation frequency). Load-shed/queue with 429s when scaling can't catch up — degraded honestly beats violated silently.

**G3. Your API is 'slow,' says a customer. First 15 minutes?**
Clarify: which metric (TTFT? total? throughput?), which percentile, since when, which model/endpoint/region. Then read the request lifecycle stages from metrics: client→gateway (network/TLS), gateway queue, admission, prefill, decode stream, detokenization. Compare each stage p50/p99 against last week's baseline. Common finds: prompt lengths grew (customer changed their app → prefill up), cache hit rate fell after a deploy, a quota pushed them to a slower model tier, or throughput expectations vs streaming perception. The winning shape: *instrument-first, hypothesis-second*, and say "baseline" out loud.

**G4. Measure and improve cost per million tokens.**
Cost/1M tok = GPU-$/hr ÷ (tokens/sec × 3600) × 10⁶, computed separately for input and output tokens (prefill tokens are far cheaper — batched compute). Levers in ROI order: (1) raise batch/utilization until SLO knee (often the biggest win — idle GPUs are the #1 cost bug), (2) quantize (FP8/int4), (3) prefix caching for template-heavy traffic, (4) speculative decoding at low batch, (5) right-size the model (distill/smaller model for easy requests + router), (6) cheaper capacity (spot/reserved, off-peak batch). Frame everything as goodput per dollar.

**G5. Multi-tenant GPU platform — how do you isolate tenants?**
Layers: (1) scheduling quotas — per-tenant token budgets per step, weighted fair queueing; (2) KV quotas so one tenant can't hog blocks and force others' preemption; (3) admission/rate limits per tenant at the gateway; (4) hard isolation where required — MIG partitions (compliance-grade, wastes capacity) vs shared engine with soft quotas (efficient, noisy-neighbor risk) — state the trade-off; (5) per-tenant metrics/billing (tokens in/out). The phrase "noisy neighbor shows up as ITL jitter, and KV quota is the fix" lands well.

**G6. How do you deploy a new model version without hurting the fleet?**
Shadow traffic first (mirror requests, compare latency/quality offline); canary a small % with automatic rollback on SLO burn or eval regression; blue-green for the engine itself (vLLM upgrades change perf characteristics — re-benchmark, never assume); warm the new pool BEFORE shifting (cold caches → TTFT spike, see G1-4); version-pin tokenizer+weights+sampler config together. Bonus: keep a locked benchmark suite (fixed prompts, fixed seeds) run on every engine/model change — that's what "performance regression CI" means for inference teams.

---

## I. MoE serving (2026's default deep-dive opener)

**I1. How does serving a mixture-of-experts model differ from a dense one?**
Total parameters set your MEMORY; active parameters set your COMPUTE. DeepSeek-V3 is ~671B total but ~37B active per token — you need the memory footprint of a giant and the FLOPs of a mid-size model. Consequences: (a) you cannot size it from "2 × params FLOPs per token" the way you would a dense model; (b) decode arithmetic intensity gets *worse*, because you read a large expert's weights to serve very few tokens routed to it; (c) parallelism changes shape — expert parallelism replaces tensor parallelism's allreduce with **all-to-all** dispatch and combine, which is latency-sensitive and now your bottleneck; (d) batching behaves differently — large batches make experts efficient but scatter reads across *all* experts, so memory traffic stops being predictable.

**I2. What is expert load imbalance and why does it matter in production?**
Routing is learned, not uniform, so real traffic makes some experts hot. With expert parallelism, one overloaded GPU stalls the entire all-to-all, so your slowest expert sets your step time. Mitigations: redundant copies of hot experts placed on multiple devices (EPLB-style rebalancing), periodic re-placement based on observed routing statistics, and capacity factors that drop or reroute overflow tokens. Say the shape of the problem — "load imbalance turns into tail latency through the all-to-all barrier" — and you've answered it.

**I3. Why does MoE push you toward disaggregated prefill/decode?**
Prefill routes many tokens through every expert (high utilisation, compute-bound); decode routes one token per sequence (low utilisation, memory-bound, all-to-all dominated). The two phases want completely different expert-parallel widths and batch shapes. Splitting them lets you run wide EP for decode across many GPUs while keeping prefill dense and local. This is why the 2026 large-scale deployments are disaggregated rather than because of dense-model interference alone.

**I4. Size DeepSeek-V3 in FP8 on H100s.**
~671B params at FP8 ≈ 671 GB of weights, plus KV and activations. An 80 GB H100 holds ~70 GB usable → minimum ~10–12 GPUs just for weights, realistically 16+ with KV and headroom, i.e. two nodes with NVLink inside each and InfiniBand between. Then note the interesting part: the *compute* is only 37B-active-worth per token, so those GPUs are memory-holders more than they are FLOP providers — which is exactly the economics that make wide expert parallelism and high batch essential.

---

## J. Structured output, multimodal, and the rest of 2026

**J1. How does constrained/guided decoding work, and what does it cost?**
Compile the grammar (JSON schema, regex, EBNF) into a finite-state machine — or a pushdown automaton for nesting — over the token vocabulary. At each step, mask the logits so only tokens that keep the string valid can be sampled. Cost: mask computation per step, on CPU, per request. At high batch this can become the decode bottleneck, which is why XGrammar-style implementations precompute and cache masks and overlap them with GPU work. Complications worth naming: tokenizer boundaries (a valid string may not be a valid token sequence), batching requests with *different* grammars, and the interaction with speculative decoding — with naive drafting, draft tokens that violate the grammar get rejected and acceptance craters; production stacks apply the same mask to the draft model (grammar-aligned drafting), which largely preserves it. Knowing both halves is the differentiator.

**J2. What changes when you serve a vision-language model?**
One image becomes hundreds to thousands of tokens depending on resolution and tiling, so "prompt length" is no longer what the user typed — prefill cost explodes. You also add a vision tower whose cost is real compute nobody budgets for, plus CPU-side image decode and resize that frequently stalls before the GPU does. Practical answers: cache encoded image embeddings (mm cache) keyed by image hash, cap or negotiate resolution, and preprocess off the critical path. Say the token-blowup number from your own measurement.

**J3. Why does temperature 0 still give different outputs across runs?**
Because reduction order in floating-point matmuls depends on how work is tiled, and the tiling depends on batch shape. A request batched with different neighbours takes a different reduction order, gets slightly different logits, and can tip an argmax. So your output depends on your batchmates — which is fine for chat and a serious problem for evals, caching and reproducibility. Fixes: batch-invariant kernels, fixed batch shapes, or accepting nondeterminism and testing distributionally rather than by exact match.

**J4. How do you serve a reasoning model, and what breaks?**
Outputs go from ~300 tokens to 5k–30k. Everything inverts: the fleet becomes decode-bound rather than prefill-bound, so ITL matters far more than TTFT; KV grows throughout a single request, so a sequence that was cheap at admission is expensive 20k tokens later; preemption becomes very costly because recomputing a 20k-token prefix is enormous; and prefix caching loses value per token because the shared prefix is a smaller fraction of the total. Speculative decoding gets *more* attractive — long, low-entropy reasoning chains give high acceptance rates. Add: reasoning-effort or thinking-budget controls as a product-level lever on cost.

**J5. When would you offload KV to CPU or NVMe instead of recomputing it?**
Compare the two costs directly: recomputing a prefix costs roughly 2 × params × prefix_tokens FLOPs; fetching it costs prefix KV bytes over your PCIe or NVMe bandwidth. For a 4k-token prefix on an 8B model: recompute ≈ 2 × 8e9 × 4096 ≈ **65 TFLOPs** ≈ 65–130 ms on one H100 at realistic MFU; the KV is ~0.5 GB ≈ 8–16 ms over PCIe Gen5/Gen4, ~70 ms from one NVMe drive. So on a fast link transfer wins by roughly an order of magnitude, and wins by more as the prefix grows. Systems like LMCache and vLLM's KV-connector API productise exactly this. Then name the caveat: it only pays if hit rates are high, and it adds a whole tier to evict and invalidate.

**J6. How do you benchmark a serving system honestly?**
Open-loop (Poisson arrivals) not closed-loop (fixed concurrency): a closed-loop client can never demonstrate queueing collapse because it only sends a new request when one finishes, so your throughput-vs-latency curve will be wrong in exactly the interesting region. Also: warm up before measuring, count tokens with the model's own tokenizer, report percentiles not means, run three times and report the median, and quote **MBU** (model bandwidth utilisation) alongside MFU — MBU is the honest metric for a memory-bound decode phase. Stating methodology unprompted is a strong signal; most candidates report a number with no protocol.

**J7. Your prefix cache is shared across tenants. What's the risk?**
A timing side channel. If tenant B's request is unusually fast, that reveals tenant A already sent a prompt with the same prefix — enough to confirm guesses about confidential prompts. Mitigation: salt cache keys per tenant (or per trust boundary) so entries are never shared across them, accepting the hit-rate loss; or share only across an explicitly consenting group. Volunteering this about your *own* cluster is a strong senior signal.

**J8. How do you prove a quantisation or engine upgrade didn't regress the model?**
A locked harness, run as one command in CI: a fixed golden prompt set with fixed seeds, greedy-output diff against an fp16 reference, logprob KL divergence on a sample, and at least one real task eval (lm-eval-harness or equivalent) with enough samples that the difference is meaningful. Perplexity alone hides task-specific damage. State what you'd gate on — "no greedy-output changes on the golden set, task eval within noise" — and you sound like someone who has shipped a model change rather than read about one.

---

## H. Coding-round classics (know the shape cold)

**H1. LRU cache** — dict + doubly-linked list, O(1) get/put. Say the bonus line: "this is exactly what KV block eviction uses."
**H2. Rate limiter** — token bucket (burst-friendly) vs sliding window (strict); per-key state; asyncio-safe.
**H3. Dynamic batcher** — C3 above; futures + queue + (max_size | max_wait) flush.
**H4. Consistent-hash ring** — sorted ring of virtual node hashes, bisect for lookup, minimal remap on node change. It's your Phase-3 router.
**H5. SSE streaming endpoint** — FastAPI async generator, heartbeats, client-disconnect cleanup (and cancel the underlying generation!).
**H6. Producer/consumer with backpressure** — bounded asyncio.Queue; explain WHY bounded (unbounded queue = hidden latency + OOM; backpressure surfaces overload early).

---

## Self-scoring rubric

For each question: 3 = fluent with numbers, could survive two follow-ups; 2 = right idea, fumbled details; 1 = recognized, couldn't produce; 0 = blank. Interview-ready = section averages ≥ 2.5 with no zeros. Re-test weekly; track in a sheet.
