# Inference Engineer Roadmap — Aug 20 → Dec 13, 2026

Prepared for Pranay Chimmani. Goal: **an offer in hand by December 1** (serving/platform track).

**The Dec 1 calendar, counted backwards — this governs everything:** an offer by Dec 1 means final decisions by ~Nov 21, because Thanksgiving week (Nov 23–29) freezes hiring for ten days. Onsites therefore happen Oct 27 – Nov 14. A loop takes 4–6 weeks end to end, so **loops must START by mid-October**, which means screens in late September, which means the September application-and-referral push is the single most decisive fortnight of the plan. Everything before Sep 4 exists to make that push land with a defensible resume. December is overtime and January is the second window — planned, not feared — but the aim point is Dec 1.

**This is the strategy.** The rest of the kit — and the interactive tracker — sit beside it: `01-study-guide.md` (what to learn, from which sources, with exercises), `02-question-bank.md` (40 questions with worked answers), `03-system-design-playbook.md` (6 worked design scenarios), `04-project-specs.md` (detailed specs for the four portfolio builds), `05-daily-plan.md` (day-by-day tasks, Aug 20 → Dec 13 — heavy Mon–Thu, lighter Fri–Sun).

---

## 1. Honest starting point

**Your real edge:** you are a production backend engineer. You built a multi-tenant SaaS solo (Ordermatic), scaled a 500k-user app (GymClan), and you already touch Kubernetes, GCP, Terraform, Docker, and Python daily. Most candidates for inference roles come from the ML side and have never run a real production system. Serving-side inference engineering is 60% distributed backend engineering + 40% LLM-specific domain knowledge. You already have the 60%. This plan closes the 40%.

**Your real gaps:**

- Transformer/inference internals — you've fine-tuned with qLoRA and used llama.cpp outputs, but you can't yet explain a forward pass, KV cache math, or why decode is memory-bound.
- The serving stack — vLLM/SGLang internals: PagedAttention, continuous batching, speculative decoding, quantization trade-offs.
- **The resume liability:** the two most inference-relevant lines on your resume — the Distributed Inference KV Cache (GKE/Terraform, "TTFT −37%") and the quantized Qwen-VL on Snapdragon — were team projects you didn't personally build. Inference interviewers will drill exactly these bullets, with follow-ups ("why consistent hashing? what was the eviction policy? why 37%?"). An undefendable bullet at these companies is a near-automatic reject.

**The fix for the liability is also your best prep:** rebuild both projects yourself, solo, during this plan (Phase 3). The distributed KV cache is *literally the ideal* inference-platform portfolio project. Once you've rebuilt it, every number on that bullet is yours. Until then, if asked in an interview, be precise: "that was a team project — my part was X; I've since rebuilt the whole thing solo" (only claimable after Phase 3).

**Also fix on the resume this week:** it says "Expected Aug 2026" — update to December 2026.

**Track choice (already decided, and it's right for you):** platform/serving roles (Python-heavy: vLLM/SGLang deployment, batching, autoscaling, K8s, benchmarking) — NOT kernel roles (CUDA/C++, much harder bar, needs C++ you don't have). NVIDIA's new-grad inference posting accepts "Python **or** C++"; Baseten/Modal/Anyscale platform loops are Python-first. C++ can come later on the job.

---

## 2. What the interviews actually test

From current job postings (NVIDIA new-grad AI Inference Performance Engineer, Together AI, Fireworks, Baseten) and interview guides, the loop for serving/platform roles is consistently four pillars:

### Pillar 1 — Practical coding (Python)
Not algorithm-trivia. Baseten-style loops use "practical engineering work instead of LeetCode trivia": rate limiters, request queues, LRU caches, token buckets, producer-consumer with asyncio, streaming parsers. Fireworks adds one traditional medium-hard algorithms round, and big-tech loops (AWS, Databricks, Google-class) impose 1–2 classic DS&A rounds regardless of team. Be honest with yourself about that tier: two evenings will not de-rust 300 problems from a year ago to a Databricks bar. If a big-tech loop gets scheduled, start maintenance reps 2–3 weeks before it, not the night before — and if you skip that tier entirely, skip the reps guilt-free. You need clean, fast Python with good structure under time pressure, explained out loud while you type.

### Pillar 2 — Inference domain knowledge (the deep-dive round)
The canonical question set: prefill vs decode (compute-bound vs memory-bound — interviewers call distinguishing these "the fastest way to show you understand the domain"), KV cache memory math, PagedAttention and fragmentation, continuous batching vs static, speculative decoding and why it's mathematically exact, quantization trade-offs (INT8/INT4/FP8, GPTQ vs AWQ, activation outliers), sampling parameters, GQA/MQA, prefix caching, disaggregated prefill/decode.

### Pillar 3 — System design for LLM serving
Real prompts from current loops: "Design the fastest serving path for a 70B model with sub-200ms first-token latency." "Design GPU autoscaling that avoids thrashing while handling cold starts." "Debug p99 latency spikes while p50 stays flat." "Serve LLMs behind low-latency APIs under bursty traffic." Plus: multi-tenant GPU isolation, batching strategy trade-offs, streaming/timeouts/retries/versioning. Your backend background shines here once you add the GPU-specific vocabulary.

### Pillar 4 — Resume deep-dive + behavioral
They walk your resume line by line. Every project must survive five levels of "why?". Baseten explicitly wants "a real backend project you can defend end-to-end, including failure handling and observability under load" — you have Ordermatic for this. Also prepare a crisp answer for "why inference infrastructure?" (yours is good: production backend engineer who wants to work on the hardest serving problem of this decade).

---

## 3. The 16-week plan

Today is Thu Aug 20, and the plan starts today. Applications start **now**, not after you're "ready" — new-grad AI infra reqs (NVIDIA's is live) are posted Aug–Oct and close early. Early interviews are diagnostic; expect to bomb one or two and learn the question distribution firsthand.

**Day-by-day detail lives in `05-daily-plan.md` and the tracker (`inference-tracker.html`).** This is the shape.

| Phase | Weeks | Dates | What ships |
|---|---|---|---|
| Kickoff | — | Aug 20–23 | Environment + llama.cpp running, job tracker live, SDE applications out, learning spine started |
| **1 · Fundamentals + Qwen-VL fix** | 1 | Aug 24–30 | Resume Day (Aug 24) · KV cache built by hand · vLLM hands-on · `inference-from-scratch` and `quantization-tradeoffs` published → **Qwen-VL bullet rewritten Aug 30** |
| **2 · Resume rebuild** | 2–3 | Aug 31 – Sep 13 | `distributed-kv-cache` core rebuilt solo → **resume v2 Fri Sep 4**; week 3 hardens it. Breadth passes + **first mock Sep 12** |
| **3 · Serving stack + toolchain** | 4–9 | Sep 14 – Oct 25 | `mini-vllm` published · MoE, structured decoding, multimodal · profiling · three-engine comparison · real multi-GPU · Prometheus/Grafana · Ray Serve · cold starts · OSS PRs |
| **4 · Interview mode** | 10–16 | Oct 26 – Dec 13 | Drills, design scenarios, mocks 4–7, live loops |
| **5 · Close the offer** | 17–19 | Dec 14–31 | Interviews, follow-ups, negotiation, holiday-aware push and restart |

**Mocks run Sep 12, Sep 19, Oct 3, Oct 17, Oct 28, Nov 4, Nov 11, Nov 18** — deliberately early, because applications go out from day one and screens land in September. The most common way this plan could fail is doing all the reps after your best companies have already interviewed you.

### Why the resume work comes first — and how Sep 4 is possible

Your two weakest bullets are the two an inference interviewer will drill hardest, so they get fixed before anything else, in the order of how cheaply each can be made honest.

The **Qwen-VL bullet is four days of mechanical work** — build llama.cpp, quantize four ways, benchmark speed/memory/quality, write it up. No prerequisites, so it runs in week 1's lighter slots and is done **Aug 30**.

The **KV cache cluster gets a hard two-week deadline** by scoping v1 to what the bullet actually claims: prefix-affinity routing over a consistent-hash ring, per-worker KV budget with LRU eviction, an SSE gateway, and the affinity-on-vs-off benchmark. HPA, chaos testing and Terraform are week 3 — hardening a project whose numbers are already published.

Attribution matters as much as the work: the team project stays in Experience with an honest scoped bullet; the solo rebuild goes in Projects with the hardware stated inline. Volunteered, it's a strength. Extracted under questioning, it's a disqualifier.

### Ongoing tracks, every week from now to December

- **Applications: 12–15/week from day one.** Track in a sheet. Referrals > cold applies — message Purdue alumni at target companies. On Sep 4, re-engage everyone you've already applied to with resume v2 and repo links.
- **Coding practice: two timed builds a week, Mon and Wed.** No LeetCode. Twenty ML-systems exercises — byte-budget LRU cache, token bucket, dynamic batcher, consistent-hash ring, SSE with cancellation, admission controller, incremental detokeniser — each of which is also a component of one of your projects.
- **Revision queue.** Everything you fumble goes in it (↻ in the tracker) and comes back at 2 / 7 / 21 days. Three clean passes = mastered.
- **Write-ups.** Every project gets a README with benchmark graphs. Recruiters and interviewers actually read these; they're the difference between claims and evidence.

---

## 4. The question bank — answers you must own

Work through these until each is a 2-minute confident answer with numbers:

1. Why is the decode phase memory-bandwidth-bound while prefill is compute-bound? What's arithmetic intensity, and how does batch size change it?
2. Derive the KV cache size for Llama-3-8B (32 layers, 8 KV heads, head_dim 128) at 8k context, fp16, batch 32. What fraction of an 80GB H100 is that?
3. What problem does PagedAttention solve? (Contiguous preallocation wastes 60–80% of KV memory to fragmentation; paging gets waste under ~4%, so effective batch size and throughput jump.)
4. Continuous batching vs static batching — why does throughput improve? What's iteration-level scheduling (Orca)?
5. TTFT vs TPOT/ITL vs total latency — which does chunked prefill help and why? What SLO would you set for a chat product vs a batch summarization pipeline?
6. How does speculative decoding work, and why is the output distribution provably unchanged? What's the throughput formula in terms of acceptance rate? When does it hurt?
7. GPTQ vs AWQ vs FP8 vs INT8 — weight-only vs weight+activation, where activation outliers come from, what quality loss to expect at each level, and what you'd pick for (a) a latency-critical chat app, (b) cheap batch offline inference.
8. What do MQA/GQA change, and why did every modern model adopt GQA? (KV cache shrinks by n_heads/n_kv_heads → bigger batches.)
9. Prefix caching / RadixAttention — when does it help (shared system prompts, RAG templates, multi-turn chat) and what's the hit-rate math?
10. Why disaggregate prefill and decode onto separate GPU pools (DistServe/Mooncake)? What's the interference problem, and what's the cost of KV transfer?
11. Tensor parallelism vs pipeline parallelism for inference — when is each right, and when do you need multi-GPU at all? (Model bytes + KV cache vs VRAM; TP for latency within a node, PP across nodes.)
12. Temperature / top-k / top-p / min-p — what would you use for code generation vs creative writing, and why does greedy decoding loop?
13. p99 latency spiked, p50 flat — walk your debugging process. (Queueing, long-prompt stragglers hogging batch slots, preemption/recompute, prefill interference, cache eviction, GC…)
14. Design GPU autoscaling for bursty traffic — what signals (queue depth, KV utilization — not CPU%), how do you handle 2–5 min cold starts (warm pools, model streaming), and how do you avoid thrash?
15. A customer says your API is "slow." What do you measure first, and what's your mental model of the request lifecycle from HTTP ingress to last token?

---

## 5. Resources (ordered, no filler)

**Foundations:** Karpathy "Let's build GPT" + "Let's reproduce GPT-2" (YouTube); Jay Alammar's Illustrated Transformer & Illustrated GPT-2; kipply's "Transformer Inference Arithmetic."

**Serving:** vLLM blog "Inside vLLM: Anatomy of a High-Throughput LLM Inference System"; DeepLearning.AI × vLLM course (2026); vLLM docs (esp. paged attention, chunked prefill, prefix caching, spec decode pages); BentoML's "LLM Inference Handbook"; SGLang docs (RadixAttention).

**Papers (concept level):** Orca (OSDI '22) · PagedAttention (SOSP '23) · Leviathan et al. speculative decoding · LLM.int8 · GPTQ · AWQ · GQA · FlashAttention 1–2 · DistServe · Mooncake.

**GPU mental model (no CUDA required):** Modal's GPU Glossary; "How to Think About GPUs" (JAX scaling-book chapter). Skip PMPP/CUDA courses — wrong track for you this cycle.

**Interview-specific:** techinterview.org company guides (Baseten, Fireworks); StackScholar LLM inference question sets; company engineering blogs of wherever you interview (Baseten, Anyscale, Fireworks, Modal all publish excellent serving posts — reading them before the loop is a cheat code).

**Vizuara AI Labs — free content (skip the paid workshop):** their paid workshop's syllabus validates this plan (it covers the same Phase 1–2 topics: roofline, KV cache, GQA/MQA/MLA, PagedAttention, quantization, continuous batching, speculative decoding, vLLM/SGLang internals), but the paid cohort isn't needed — use their free material instead. Free assets: the 43-lecture "Build LLMs from scratch" YouTube series (transformer architecture, GPT internals, BPE tokenization, attention, full Python implementations — a more granular alternative to Karpathy if you prefer shorter lectures; use one or the other as your Phase 1 spine, not both); standalone free lectures like "KV Caching: Speeding up LLM Inference"; and the free Vizuara Substack deep-dives (e.g., the Multi-Head Latent Attention series — MLA is the DeepSeek-era KV-cache answer and a fresh interview topic). Total cost of this entire roadmap: ~$0 — the only possible spend is a short GKE benchmark run in Phase 3, which GCP's free-trial credits cover.

---

## 6. Where to apply (serving/platform track)

- **Inference clouds (best fit for your profile):** Together AI, Fireworks, Baseten, Modal, Anyscale, Replicate, RunPod, Lambda, CoreWeave, Nebius, Groq, Cerebras, SambaNova.
- **NVIDIA:** the "AI Inference Performance Engineer — New College Grad 2026" req is live and matches you (Python accepted; vLLM/SGLang/TensorRT-LLM benchmarking). Apply this week.
- **vLLM commercial ecosystem:** Red Hat (owns vLLM's commercial arm, llm-d project), IBM, Hugging Face.
- **Big-co model serving teams:** Databricks, Snowflake, LinkedIn, Apple (AIML), Microsoft/Azure AI, AWS (Bedrock/SageMaker), Oracle OCI GenAI, Salesforce, Uber/DoorDash ML platform.
- **Frontier labs (apply, expect a high bar):** Anthropic, OpenAI, xAI, Google DeepMind — inference/performance teams.
- **AI products with heavy self-hosted inference:** Perplexity, Cursor, Harvey, Glean, Character.AI.

Filter postings on: vLLM/SGLang/TensorRT-LLM, "model serving," "inference platform," "LLM performance," "GPU infrastructure." Titles vary wildly (Inference Engineer, ML Platform Engineer, LLM Serving Engineer, AI Infra Engineer) — search by keyword, not title.

---

## 7. Weekly rhythm

Mon–Thu (~6.5h): build 2.5h · study 1.5h · coding 1h · applications 1h · close 0.5h. Fri (~4h): outreach, applications, write-ups. Sat (~4h): build catch-up + one design drill. Sun (~2.5h): self-test and plan, then stop — rest is part of the plan.

**Definition of ready (you'll be there by late October):** you can implement a KV cache from scratch in 30 minutes; answer the question bank out loud with numbers; whiteboard the 70B-serving design in 35 minutes; and defend every resume line five whys deep.
