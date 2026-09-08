# Study Guide — Inference Engineering, Module by Module

Companion to `inference-engineer-roadmap.md` (the schedule). This is the *what and from where*. Every module lists several independent sources — pick ONE primary per module and move on; the others are backups for when an explanation doesn't click. Depth of understanding beats breadth of resources.

How to use each module: read/watch the primary source → do the exercises (non-negotiable — exercises are where learning happens) → check yourself against "You're done when." Then close it and move on.

**Worked walkthroughs.** Every exercise below is worked step by step — terms defined from zero, the real code to read first, fully commented code, the arithmetic shown, expected results — in `10-study-walkthroughs.md`. Exercises that are satisfied by a project milestone point to the project walkthrough (`06-p1-walkthrough.md`, `07-p2-walkthrough.md`, `08-p3-walkthrough.md`, `09-p4-walkthrough.md`). This file stays the map; that one is the route.

---

## Module 1 — Transformer internals (Week 1, Mon–Tue)

> Walkthrough: `10-study-walkthroughs.md` § Module 1 · exercises 1–2 are Project 1 M1–M2 (`06-p1-walkthrough.md`).

**Why interviewers care:** every inference question bottoms out in the forward pass. If you can't sketch what happens between input tokens and logits, nothing downstream makes sense.

**Sources (pick one primary):**
- Karpathy — "Let's build GPT: from scratch, in code" (YouTube, ~2h) — best if you like one intense sitting.
- Vizuara — "Build LLMs from scratch" 43-lecture series (YouTube, free) — same ground, granular lectures; better if you prefer 20-min chunks.
- Sebastian Raschka — *Build a Large Language Model (From Scratch)* (book + free YouTube walkthroughs) — best written treatment.
- Jay Alammar — Illustrated Transformer + Illustrated GPT-2 — visual pre-read before any of the above.

**Exercises:**
1. Implement multi-head self-attention in PyTorch from memory (no peeking). Q/K/V projections, causal mask, softmax, output projection.
2. Load real GPT-2 or Qwen2.5-0.5B weights and run a greedy generation loop you wrote yourself.
3. On paper: count the parameters of one transformer block given d_model, n_heads, d_ff. Verify against the real config.

**You're done when:** you can draw the full decoder block from memory and explain, for one generated token, every matmul that happens and its shapes.

---

## Module 2 — Inference arithmetic & the GPU mental model (Week 1)

> Walkthrough: `10-study-walkthroughs.md` § Module 2 — the roofline derivation, the 2060 ceiling, prefill FLOPs, the full model×GPU table.

**Why interviewers care:** "prefill is compute-bound, decode is memory-bound" plus the supporting arithmetic is the #1 domain filter question. Numbers, not vibes.

**Sources:**
- kipply — "Transformer Inference Arithmetic" (blog) — the classic; primary.
- JAX scaling book — "How to Think About GPUs" + inference chapter (jax-ml.github.io/scaling-book) — deeper, excellent.
- Modal — GPU Glossary (modal.com/gpu-glossary) — reference, not a read-through.
- Baseten — "A guide to LLM inference and performance" (blog).

**Key ideas to master:** arithmetic intensity = FLOPs/byte moved; roofline model; why decode at batch 1 reads every weight once per token → tokens/sec ≈ bandwidth / model bytes; why batching moves decode toward compute-bound; FLOPs per token ≈ 2 × params.

**Exercises:**
1. Your RTX 2060: 336 GB/s, 6 GB VRAM. Compute the theoretical decode ceiling for Qwen2.5-0.5B in fp16 (~1 GB of weights → ~336 tok/s at batch 1). Measure your Phase-1 engine against it; explain the gap.
2. Compute prefill FLOPs for a 512-token prompt on the same model, and show why prefill saturates compute while decode doesn't.
3. Fill in a table: for Llama-3-8B, Llama-3-70B, Qwen2.5-0.5B — weights in GB (fp16/int8/int4), min VRAM to serve, batch-1 decode ceiling on an H100 (3.35 TB/s).

**You're done when:** given any (model size, GPU) pair you can estimate tokens/sec on a whiteboard in under two minutes.

---

## Module 3 — KV cache & attention variants (Week 1, Wed–Thu)

> Walkthrough: `10-study-walkthroughs.md` § Module 3 · exercise 1 is Project 1 M3 (`06-p1-walkthrough.md`).

**Why interviewers care:** KV cache is THE central object of inference engineering. Memory math on it appears in nearly every loop.

**Sources:**
- Your own Phase-1 implementation — primary. Build it before reading about it.
- Vizuara — "KV Caching: Speeding up LLM Inference" (free YouTube lecture) + their Substack MLA deep-dive series.
- GQA paper (Ainslie et al.) — short read.
- vLLM docs — conceptual pages on KV cache management.

**Key ideas:** why caching K/V makes decode O(n) instead of O(n²) per token; size formula `2 × layers × kv_heads × head_dim × seq_len × bytes × batch`; MQA (1 KV head) vs GQA (grouped, e.g. Llama-3's 32 query/8 KV heads = 4× cache reduction) vs MLA (DeepSeek's latent compression, ~90%+ reduction); why KV cache, not weights, limits batch size at long context.

**Exercises:**
1. Add a KV cache to your Module-1 generation loop; measure tokens/sec before vs after at 1k context.
2. Memorize and re-derive the size formula. Compute it for Llama-3-8B @ 8k context, batch 32, fp16 (answer: ~32 GB — check yourself).
3. Recompute exercise 2 as if the model used MHA (32 KV heads) instead of GQA. Feel the 4×.

**You're done when:** you can do KV math for any published model from its config.json without notes.

---

## Module 4 — Serving engines: vLLM & SGLang internals (Week 1 hands-on, Weeks 5–7 deep)

> Walkthrough: `10-study-walkthroughs.md` § Module 4 — serving Qwen2.5-1.5B on the 2060, the concurrency sweep, the prefix-cache TTFT experiment, the vLLM v1 reading path · exercise 3 is Project 2 (`07-p2-walkthrough.md`).

**Why interviewers care:** this is the job. Postings literally name vLLM/SGLang/TensorRT-LLM proficiency.

**Sources:**
- vLLM blog — "Inside vLLM: Anatomy of a High-Throughput LLM Inference System" — primary; read it twice.
- DeepLearning.AI × vLLM short course (free, 2026).
- PagedAttention paper (Kwon et al., SOSP '23) + Orca paper (Yu et al., OSDI '22).
- SGLang docs — RadixAttention section.
- vLLM source code — after the blog post, trace one request through `v1/engine`: scheduler → block manager → model runner. Reading real engine code is what separates you from candidates who only read blogs.

**Key ideas:** continuous (iteration-level) batching; paged KV with block tables, fragmentation <4% vs 60–80% wasted in contiguous allocation; preemption (recompute vs swap); chunked prefill; prefix caching / radix-tree reuse; scheduler policy; CUDA graphs for decode; the API-server/engine split.

**Exercises:**
1. Serve Qwen2.5-1.5B on your 2060 with vLLM. Run its benchmark script; sweep max concurrent requests; plot throughput vs p50/p99 latency curves.
2. Enable prefix caching with a 1k-token shared system prompt; measure TTFT hit vs miss.
3. Build mini-vLLM (see project spec 2) — the capstone of this module.

**You're done when:** you can narrate the life of a request inside vLLM — arrival, scheduling, prefill, block allocation, decode steps, preemption, streaming, completion — for five minutes without stopping.

---

## Module 5 — Quantization (Week 1 alongside Project 4, then Week 6)

> Walkthrough: `10-study-walkthroughs.md` § Module 5 · exercise 2 is Project 4 (`09-p4-walkthrough.md`).

**Why interviewers care:** the standard "trade-offs" question — every serving team quantizes something, and wants to know you understand what breaks.

**Sources:**
- HuggingFace quantization docs + blog posts (bitsandbytes, GPTQ, AWQ overviews) — primary survey.
- LLM.int8 paper (Dettmers) — for the activation-outliers insight; AWQ paper — for activation-aware scaling; skim GPTQ.
- llama.cpp quantization docs (the K-quants) — practical grounding, and directly relevant to your Qwen-VL resume bullet.

**Key ideas:** weight-only (GPTQ/AWQ/K-quants — helps memory-bound decode, weights dequantized on the fly) vs weight+activation (INT8/FP8 — helps compute too, needs calibration); activation outlier channels and why naive INT8 activations fail; FP8 on Hopper+; typical quality: int8 ≈ lossless, good int4 ≈ small loss, naive int4 ≈ visible loss; KV cache quantization (fp8 KV) as a separate lever.

**Exercises:**
1. Run the same model at fp16 vs AWQ-int4 in vLLM: measure tokens/sec, VRAM, and quality on 20 fixed prompts.
2. llama.cpp sweep on your 2060: Q4_K_M vs Q5_K_M vs Q8_0 — speed/memory/quality table (this IS project 4's core).
3. Explain in writing (one page): why does weight-only quantization speed up decode but barely help prefill?

**You're done when:** given a deployment scenario (latency-critical chat vs cheap offline batch) you can name a quantization scheme and defend it.

---

## Module 6 — Speculative decoding (Week 7)

> Walkthrough: `10-study-walkthroughs.md` § Module 6 — a complete toy implementation with the rejection-sampling rule explained, plus the speedup arithmetic worked.

**Why interviewers care:** favorite "do they actually understand it or just name-drop it" topic. The exactness proof and acceptance-rate math are the differentiators.

**Sources:**
- Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" — primary; work through the rejection-sampling argument.
- vLLM docs/blog on spec decode; EAGLE and Medusa papers at skim level (know they exist and the one-line idea).

**Key ideas:** draft proposes γ tokens; target verifies all in ONE forward pass (parallel, like prefill); accept/reject via rejection sampling → output distribution provably identical to target-only decoding; expected tokens per target pass = (1−α^(γ+1))/(1−α) for acceptance rate α; wins when decode is memory-bound and draft is cheap+aligned; loses at high batch (decode already compute-bound) or low α.

**Exercises:**
1. Implement toy speculative decoding with Qwen2.5-0.5B drafting for Qwen2.5-1.5B on your 2060. Measure acceptance rate and speedup.
2. Compute expected speedup for α=0.8, γ=4, draft cost 10% of target. Then find the α below which it's a net loss.

**You're done when:** you can state *why the math guarantees the same output distribution* in three sentences.

---

## Module 7 — Distributed inference & disaggregation (Weeks 2, 8)

> Walkthrough: `10-study-walkthroughs.md` § Module 7 — the 70B sizing arithmetic and the KV-transfer one-pager with numbers.

**Why interviewers care:** anything ≥70B forces multi-GPU; senior-leaning question territory, and Together's posting names Mooncake explicitly.

**Sources:**
- vLLM docs — distributed serving (TP/PP) pages — primary.
- DistServe paper + Mooncake paper (skim for architecture, not proofs).
- Anyscale/Baseten blog posts on multi-GPU serving; "How to Think About GPUs" chapter on collectives.

**Key ideas:** when multi-GPU is forced (weights + KV > VRAM); tensor parallelism = split matmuls, allreduce per layer, needs NVLink, cuts latency; pipeline parallelism = layer stages, bubbles, cross-node friendly, throughput not latency; disaggregated prefill/decode: prefill and decode interfere (batched decode stalls behind long prefills), so split onto separate pools and ship KV; expert parallelism for MoE at awareness level.

**Exercises:**
1. Whiteboard: serve Llama-3-70B fp16 — how many H100s minimum, and TP or PP? Show the arithmetic (140 GB weights + KV headroom → TP=4 minimum, TP=8 typical).
2. Write one page: what exactly is transferred in disaggregated serving, how big is it (use Module 3 math), and over what link?

**You're done when:** you can size a multi-GPU deployment for any model and justify TP vs PP in under three minutes.

---

## Module 8 — Production serving ops (Weeks 3–4, then 8 — overlaps the rebuild)

> Walkthrough: `10-study-walkthroughs.md` § Module 8 — concrete SLO numbers with justifications; the TTFT instrumentation plan · exercise 2 is Project 3 M6–M7 (`08-p3-walkthrough.md`).

**Why interviewers care:** this is the Baseten/Modal/Anyscale system-design round: SLOs, autoscaling, debugging, cost.

**Sources:**
- BentoML — LLM Inference Handbook (free, bentoml.com/llm) — primary for vocabulary and metrics.
- Baseten engineering blog (autoscaling, cold starts, speculative deployment posts); Modal blog (cold starts, GPU utilization); Anyscale blog (continuous batching numbers).
- Your own Phase-3 rebuild — the real teacher here.

**Key ideas:** metrics — TTFT, TPOT/ITL, goodput, tokens/sec/GPU, cost per 1M tokens; SLOs per product type; autoscaling signals (queue depth, KV utilization, concurrency — never CPU%); cold-start anatomy (image pull → weight load → warmup) and mitigations (warm pools, streaming weight load, snapshots); observability (per-stage latency breakdown, preemption count, cache hit rate); multi-tenancy and isolation; request routing (session affinity for prefix cache).

**Exercises:**
1. Define numeric SLOs for: consumer chat, coding copilot, offline summarization. Justify each.
2. For your Phase-3 KV cache cluster: instrument TTFT breakdown (queue / route / prefill / first token) and write the p99 story of one real spike you observe.

**You're done when:** you can run the "p99 spiked, p50 flat — debug it" drill cold (see playbook doc).

---

## Module 9 — Coding fitness (every week)

> Walkthrough: `10-study-walkthroughs.md` § Module 9 — for each practice item: the interface, the data structure, the trap, and the test that defines "done"; the 45-minute routine.

**Format seen in loops:** Python, practical problems > algorithm trivia (Baseten explicitly), plus one classic medium round at some companies (Fireworks).

**Practice set (build each in <45 min, clean, tested):** LRU cache from scratch (dict + doubly-linked list — also *explains KV eviction*); token-bucket and sliding-window rate limiter; bounded producer/consumer with asyncio; a request batcher that flushes on max-size OR max-wait (this is literally dynamic batching); SSE streaming endpoint (FastAPI); consistent-hash ring with virtual nodes (literally your Phase-3 router); heap-based scheduler; top-k frequent items; interval merging.

**No LeetCode curriculum.** You have ~300 problems behind you; grinding patterns buys less than building the things these loops actually ask about. Two timed builds a week (Mon, Wed) from the twenty-exercise practical list, each of which is also a component of one of your projects. If a company confirms a classic algorithms screen, spend two evenings on it then — targeted, not speculative.

**You're done when:** never — this stays weekly through December. But you're interview-safe when you can do any of the practice set cold while talking.

---

## Module 11 — Production toolchain (Weeks 6, 8–9)

> Walkthrough: `10-study-walkthroughs.md` § Module 11 — exact profiler commands, the three-engine comparison plan, the Prometheus/Grafana compose with panels and alerts, the 2×GPU rental checklist, the cold-start measurement method.

**Why interviewers care:** platform roles hire people who have *run* things. Theory about bottlenecks loses to "I profiled it and found X." This module is the difference between a candidate who has read about serving and one who has operated it.

**Profiling.** `torch.profiler` with a short trace window (never the whole run — you'll drown), `py-spy top` / `py-spy record` attached to the live server, and one Nsight Systems trace to see the timeline properly. The question to answer with data: in a single decode step, how much is GPU compute and how much is Python and scheduling overhead? Keep a flame graph for the README.

**Engine comparison.** Run the same model, same prompts, same arrival trace through vLLM, TensorRT-LLM and SGLang. Record throughput, TTFT, memory, build/deploy friction. TensorRT-LLM's build-time compilation is the whole trade-off — upfront cost for speed, at the price of runtime flexibility; use the NGC container rather than building from source. SGLang gives you RadixAttention to compare against the prefix cache you wrote yourself. Be honest when an engine doesn't win on your hardware: knowing *when* each one wins is the actual interview answer, and most candidates have used exactly one.

**Observability.** Prometheus scraping vLLM's `/metrics` and your own, Grafana on top. Build one dashboard you'd genuinely want at 3am: TTFT percentiles, ITL, throughput, KV utilisation, queue depth, preemptions, cache hit rate. Add two alerts and be able to justify both thresholds.

**Multi-GPU, actually run.** A few hours on rented 2×A100 or 2×L40S — the cost of lunch. Prepare every script before the clock starts. Run `--tensor-parallel-size 2`, measure scaling efficiency against one GPU, and note the gap: that gap is allreduce, and it's what gets probed. Then show TP *costing* you latency on a model that fits on one card — the more interesting result.

**Managed serving.** Deploy behind Ray Serve or KServe and compare it honestly to the gateway you built. You will be asked "why did you build your own instead of using X," and this is where you earn a good answer rather than a defensive one.

**Cold starts.** Measure it in three parts — image pull, weight load, first token — then attack each: baked weights vs volume vs streamed from object storage, layer ordering, smaller base image. Cold start is *the* production question for serving platforms.

**You're done when:** you can answer "how would you find the bottleneck" with tools and a story from your own system, name which engine you'd pick for a given workload and why, and quote your own TP scaling number.

---

## Module 10 — Story bank & behavioral (Week 10+, alongside interviews)

> Walkthrough: `10-study-walkthroughs.md` § Module 10 — the STAR template, a worked first-person pivot story, the five-whys drill.

Write STAR stories (half a page each), then compress each to 90 seconds spoken:
1. Ordermatic — solo-built multi-tenant SaaS: architecture choices, a production incident, what you'd redo.
2. GymClan — scaling to 500k users in a 2-person team; mentoring juniors.
3. Hexion — building eval mechanisms for GenAI outputs; working with non-technical stakeholders.
4. Purdue×Microsoft — fine-tuning LLaMA with qLoRA, diagnosing overfitting, A/B evaluation design.
5. The pivot story — "why inference?": production backend engineer who kept getting pulled toward the hardest serving problems; rebuilt his team's KV cache project solo to learn the domain properly (tellable after Sep 18 — and it's a GREAT story because it shows intellectual honesty).
6. For each target company: one paragraph on why THEM, citing a specific engineering blog post of theirs.

**Resume rule:** after resume v2 (Sep 18), every bullet survives five consecutive "why?"s. Rehearse the two rebuilt-project walkthroughs as 5-minute whiteboard talks with numbers.

---

## Weekly self-test ritual (Sundays, 30 min)

Pick 5 random questions from the question bank (doc 02), answer out loud, grade yourself brutally. Pick 1 design prompt from the playbook (doc 03) every other week and whiteboard it in 35 minutes. Log scores in a sheet — the trend line is your real readiness signal, not gut feel.
