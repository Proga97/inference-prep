# Project Specs — The Four Portfolio Builds

These four projects ARE the preparation. Reading produces recognition; building produces the fluency interviews test. Each spec has milestones (do them in order — each is demo-able), acceptance criteria, and the interview story the project buys you. All fit your RTX 2060 (6 GB) except one optional cloud benchmark.

Rules that apply to all four: public GitHub repo; README written like an engineering blog post (problem → design → benchmark graphs → what surprised you); commit as you go (a real history reads as real work); every number in the README reproducible by a script in the repo.

**Detailed walkthroughs.** Each project has a companion document with every milestone worked step by step — concepts from zero, the real code to read first, fully commented code, tests, expected numbers, and failure tables: `06-p1-walkthrough.md` (Project 1), `07-p2-walkthrough.md` (Project 2), `08-p3-walkthrough.md` (Project 3), `09-p4-walkthrough.md` (Project 4). The study-guide exercises are worked the same way in `10-study-walkthroughs.md`. This file stays the spec — the *what* and the acceptance bar; the walkthroughs are the *how*.

---

## Project 1 — `inference-from-scratch` (Week 1 · Aug 24–30)

> **Walkthrough:** `06-p1-walkthrough.md` — Step 0 housekeeping, M1 model + parity, M2 naive loop, M3 KV cache, M4 sampling, M5 bench.py.

**One-liner:** a plain-PyTorch LLM inference engine, built up from a naive generation loop to a KV-cached, batched, sampled engine — with measurements at every step.

**Milestones:**
1. Load Qwen2.5-0.5B (or GPT-2) weights into your OWN module code — no `model.generate()`, no HF pipeline. Verify logits match HF's to ~1e-4.
2. Naive greedy loop: full-sequence forward pass per token. Measure tok/s vs context length (watch it degrade quadratically). Plot it.
3. Add KV cache. Re-measure: the before/after plot is the money graph. Explain the gap to theoretical bandwidth ceiling (Module 2 math) in the README.
4. Sampling: temperature, top-k, top-p, and a seed-reproducible RNG. Show greedy-repetition-loop vs sampled output examples.
5. Benchmark harness: `bench.py` producing every graph in the README from scratch.
6. *(Deferred to Project 2, where it belongs:* batched generation with left-padding and masks. Week 1 is short — ship milestones 1–5 and let mini-vLLM handle batching.*)*

**Acceptance:** you can rebuild milestone 3 (KV cache) from a blank file in 30 minutes; README has ≥3 graphs; logit-parity test passes.
**Interview story bought:** every question in bank sections A–B becomes a thing you *did*, not read. "I measured 22 tok/s naive vs 118 with my KV cache on a 2060, ~65% of the bandwidth ceiling, and here's where the rest went" is an answer nobody can fake.

---

## Project 2 — `mini-vllm` (Weeks 4–6 · Sep 14 – Oct 4) — built LAST, after both resume fixes

> **Walkthrough:** `07-p2-walkthrough.md` — sizing the 2060, the interface contract with Project 1, M1–M6, design divergences from real vLLM.

**One-liner:** a continuous-batching inference server over the Project-1 engine: paged KV blocks, iteration-level scheduling, SSE streaming, live metrics. ~600–900 lines of Python.

**Milestones:**
1. FastAPI server, single-request `/generate` with SSE token streaming (heartbeats, disconnect cancels generation).
2. Request queue + step-loop scheduler: each iteration, admit waiting requests into the running batch if capacity allows; retire finished ones. (Static batch → continuous batching is THE milestone; benchmark the difference under Poisson arrivals.)
3. Block-based KV manager: fixed-size blocks (e.g., 16 tokens), free-list allocator, per-sequence block tables. Simplification allowed: gather blocks into contiguous tensors for attention (real paged attention needs custom kernels — out of scope, and knowing exactly WHY it's out of scope is itself interview material).
4. Preemption: when blocks run out, evict lowest-priority running sequence (recompute-on-resume). Expose a preemption counter.
5. Prefix reuse (stretch): hash-based block sharing for common prompt prefixes + LRU eviction of cached blocks.
6. `/metrics`: TTFT/ITL histograms, queue depth, KV utilization, preemption count. Load-test with Poisson arrivals; produce the throughput-vs-p99 curve and mark the knee.

**Acceptance:** sustains ≥16 concurrent streams on the 2060 with 0.5B model; continuous-vs-static benchmark shows the win; you can explain every design divergence from real vLLM (that conversation IS a great interview).
**Interview story bought:** "I built a toy vLLM" turns Module 4 and question bank C into first-person knowledge, and it's your best differentiator as a portfolio piece.

---

## Project 3 — `distributed-kv-cache` rebuild (Weeks 2–3 · Aug 31 – Sep 13) — THE BIG RESUME FIX

> **Walkthrough:** `08-p3-walkthrough.md` — kind cluster, llama-server workers, consistent-hash coordinator, SSE gateway, LRU eviction, the affinity-on-vs-off benchmark, KEDA/chaos/GKE.

**One-liner:** solo rebuild of the resume project: a Kubernetes cluster serving LLM inference through a KV-cache-aware routing layer — gateway, coordinator, workers — with consistent-hash prefix routing, LRU eviction, autoscaling, and honest benchmarks.

**Why it runs early:** this bullet sits on your resume with numbers you can't currently defend, and every inference interviewer will drill it. It's also mostly distributed-systems work — your existing strength — so it doesn't need serving-engine internals first. Two weeks of foundation (a hand-built KV cache, plus seeing vLLM's prefix caching with your own eyes) is enough. Scope v1 to what the bullet actually claims — prefix-affinity routing, LRU eviction, SSE gateway, and the affinity-on-vs-off benchmark — and it fits two weeks: **resume v2 ships Fri Sep 4**. Milestones 7–8 (HPA, chaos test, Terraform, optional GKE) are week 3, hardening a project whose numbers are already published; re-run the benchmark afterwards and upgrade the bullet if it improved.

**Milestones (1–6 by Sep 4; 7–8 in week 3):**
1. Local-first: `kind` cluster on your machine; workers run llama.cpp or your mini-vllm with a small model (CPU workers are FINE — the system being built is the routing/caching layer, not the kernel).
2. Worker: wraps the engine, holds per-prefix KV/session state, reports load + cache stats.
3. Coordinator + consistent-hash ring (virtual nodes) keyed on prompt-prefix hash → requests sharing a prefix land on the worker already holding its KV. Handle worker join/leave with minimal remap.
4. Gateway: single entry, SSE streaming passthrough, per-client rate limiting, request logging.
5. Eviction & memory management: per-worker KV budget, LRU over cached prefixes, metrics for hit/miss/eviction.
6. Benchmarks (the whole point): TTFT with routing-affinity ON vs OFF (random routing) at several concurrency levels; throughput under a workload with realistic prefix reuse (e.g., 20 system-prompt templates). Report YOUR measured deltas — whatever they are. If your TTFT improvement is 42% not 37%, the resume says 42%.
7. StatefulSets + HPA on custom metrics (queue depth); kill a worker mid-load and document recovery behavior.
8. Optional single GKE run (one small GPU node pool, few hours, inside free credits) for a "and on real GPUs" section.

**Acceptance:** `terraform apply` / `kind` script reproduces everything; affinity-on-vs-off graph in README; you can whiteboard the architecture and defend consistent hashing vs alternatives (central routing table, rendezvous hashing) for five minutes.
**Then:** rewrite the resume bullet in first person with your measured numbers, and retire the old ones.

---

## Project 4 — `quantization-tradeoffs` (Week 1 · Aug 24–30, ~4 days) — THE QWEN-VL FIX, done FIRST

> **Walkthrough:** `09-p4-walkthrough.md` — llama.cpp CUDA build, GGUF conversion and K-quants, llama-bench harness, GSM8K/perplexity eval, the tradeoff table and resume bullet.

**One-liner:** llama.cpp quantization sweep of Qwen2-VL (or Qwen2.5 text model) on your own hardware: speed/memory/quality across Q4_K_M / Q5_K_M / Q8_0 / fp16, with a small task-specific eval.

**Milestones:**
1. Build llama.cpp with CUDA; quantize the same base checkpoint to 4 levels.
2. Benchmark each: tokens/sec (prefill and decode separately), VRAM, load time — on the 2060 and on CPU-only (two hardware points = a real tradeoff table; the Snapdragon column becomes "hardware I'd extend this to").
3. Quality eval: 50-prompt fixed set scored consistently (for VL: image-description accuracy vs fp16 reference; for text: exact-match QA set). Report degradation per level.
4. README centerpiece: the speed-vs-quality-vs-memory table + "which level would I ship for chat vs batch vs edge, and why."

**Acceptance:** every number regenerable by script; you can answer all of question bank section D from your own data.
**Then:** rewrite the Qwen-VL resume bullet around what you measured.

---

## Sequencing & fallbacks

Build order is **1 + 4 (week 1, in parallel) → 3 (weeks 2–4) → 2 (weeks 5–7)**: the cheapest resume fix and the foundation first, then the big rebuild, then the differentiator. Project 4 runs in week 1's lighter slots because it is mechanical benchmarking — you are waiting on quantization jobs, not thinking hard. Priority if time compresses: **3 > 4 > 1 > 2.** (3 and 4 are the resume; 1 is the foundation everything rests on but its later milestones can fold into 2; 2 is the best portfolio differentiator and survives scope cuts — prefix reuse and preemption are droppable.)

If GPU trouble on the 2060 blocks something: everything in projects 2 and 3 works on CPU with small models — the systems you're building are schedulers and routers, and interviewers know that. Never let hardware stall the plan; note the limitation and keep moving.

After each project: 30-minute mock walkthrough out loud — problem, design, one hard bug, the graphs, what you'd do differently. That rehearsal, not the code, is what shows up in the interview.
