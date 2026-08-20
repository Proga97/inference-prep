# Daily Plan — Thu Aug 20 → Sun Dec 13, 2026

Generated from the same source as `inference-tracker.html`, so the two never drift. Tick days there; read context here.

**The rules.** Mon–Thu are heavy (~6.5h). Fri (~4h), Sat (~4h), Sun (~2.5h) are lighter by design. Coding slots are *revision and speed* — you've done ~300 problems, so these are timed reps mixed with the practical builds these loops actually ask (rate limiter, dynamic batcher, consistent-hash ring). Applications run 3/day Mon–Thu plus Friday outreach ≈ 12–15/week, from day one — postings close, so you never wait to be "ready." If a day slips, drop the study slot first, never the build slot: projects compound, reading compresses.

**Why the resume work comes first.** Your two weakest resume bullets are the two an inference interviewer will drill hardest, so they get fixed in the order of how cheaply each can be made honest. The Qwen-VL bullet is four days of mechanical benchmarking — it runs in week 1's lighter slots and is **rewritten Aug 30**. The KV cache cluster gets a hard two-week deadline: **resume v2 ships Friday Sep 4**, built to the scope the bullet actually needs — prefix-affinity routing, LRU eviction, and the affinity-on-vs-off benchmark. HPA, chaos testing and Terraform come in week 3, hardening a project whose claim is already published. Everything after that — mini-vLLM, open source, seven weeks of drilling — happens on top of a resume you can defend line by line.

**Heavy-day shape (Mon–Thu):** Build 2.5h · Study 1.5h · Code 1h · Apply 1h · Close 0.5h.
**Fri:** outreach + applications 2h · redo missed problems 1h · write-up/commit 1h.
**Sat:** build catch-up 2.5h · design drill or study 1.5h.
**Sun:** self-test (5 random question-bank Qs, scored) 1h · review and plan 1h · then stop. Rest is part of the plan.

**Revision.** Anything you fumble — a question, a paper, a debugged bug — hit ↻ on it in the tracker (or type it into the queue). It comes back in 2 days, then 7, then 21, and after three clean passes it's mastered. Clearing the due queue is the first thing you do in the Close slot.

---


## Kickoff


### Kickoff weekend · Aug 20 → Aug 23

**Thu Aug 20** *(6h 25m)* — *Study (1h 45m):* Pick your spine — Karpathy 'Let's build GPT' or Vizuara's series — and get through the first two hours tonight · *Setup (50m):* Environment: PyTorch + CUDA on the 2060, HF account, pull Qwen2.5-0.5B, verify a forward pass · *Design (50m):* Learn the 5-step method: requirements → capacity maths → architecture → deep-dive → failure modes. Write it on a card you keep · *Code (40m):* Revise: arrays + hashmap — 2 timed mediums · *Setup (50m):* Install llama.cpp + download a Qwen2-VL GGUF — quantization jobs start Monday · *Setup (50m):* Set up your job tracker — duplicate the Notion template, add tier-1 / tier-2 columns · *Apply (40m):* Apply to SDE / backend roles with your current resume — 5 today, keep the pipeline warm

**Fri Aug 21** *(4h 5m)* — *Study (1h 5m):* Continue the spine through self-attention — type the code alongside, don't just watch · *Apply (40m):* 5 more SDE / backend applications · *Design (50m):* Capacity drill: how many H100s to serve Llama-3-70B in fp16? Do the arithmetic out loud, then again in FP8 · *Code (40m):* Revise: two pointers / sliding window — 2 timed · *Setup (50m):* Tidy LinkedIn and GitHub — clean, current, nothing pinned yet

**Sat Aug 22** *(3h 45m)* — *Study (1h 15m):* Continue the spine: multi-head attention and the full block — type it, don't watch it · *Code (45m):* 2 timed LeetCode mediums — rust check · *Design (1h):* Capacity drill: concurrent sequences per 80GB card at 8k context on a GQA 8B — derive it from the KV formula · *Apply (45m):* 5 SDE applications · note which job titles keep appearing on the inference side

**Sun Aug 23** *(2h 30m)* — *Study (50m):* Finish the spine + skim Illustrated Transformer · *Plan (30m):* Read the Project 1/3/4 specs + the design playbook's 5-step method — you use it Saturday · *Design (40m):* SLO drill: write numeric TTFT / ITL / availability targets for chat, a coding copilot, and batch summarisation — defend each · *Code (30m):* Revise: heap and top-k — 2 timed


## Phase 1 · Fundamentals + Qwen-VL fix


### Week 1 · Aug 24 → Aug 30

**Mon Aug 24** *(6h 35m)* — *Build (2h):* P1-M1: write your own model module; load Qwen2.5-0.5B weights · *Build (1h 40m):* RESUME DAY: build the full inference resume — every project, done and planned, written once · *Design (50m):* Component: draw the request lifecycle from HTTP ingress to last token, and name every stage you would instrument · *Code (35m):* Revise: intervals — merge, insert, minimum rooms · *Apply (1h 15m):* Search TODAY's inference postings, tier them, then send NVIDIA new-grad + 2 tier-2 — the true resume · *Close (15m):* Kick off P4-M1 in the background: quantize the checkpoint 4 ways

**Tue Aug 25** *(6h 30m)* — *Build (2h 30m):* P1-M1 parity test vs HuggingFace (~1e-4) · P1-M2 naive greedy loop · *Study (1h 15m):* kipply 'Transformer Inference Arithmetic' · compute your 2060's decode ceiling · *Design (1h):* Component: static vs continuous batching — draw both schedulers and explain the throughput delta with numbers · *Code (45m):* Practical: LRU cache from scratch — the structure you'll use for KV eviction · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Check the quantization jobs · log notes

**Wed Aug 26** *(6h 30m)* — *Build (2h 30m):* P1-M3: implement the KV cache — the single most important exercise here · *Study (1h 15m):* Serve Qwen2.5-1.5B in vLLM on the 2060; benchmark it against your own loop · *Design (1h):* Component: KV memory budgeting — given a VRAM budget, how many concurrent requests, and what do you evict first? · *Code (45m):* LeetCode ×2 (heap / two-pointer) · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Thu Aug 27** *(6h 30m)* — *Build (2h 30m):* KV cache before/after benchmark + sampling (temp / top-k / top-p) · *Study (1h 15m):* vLLM prefix-caching experiment: 1k shared prompt, TTFT hit vs miss — the exact effect your cluster will exploit · *Design (1h):* Component: design the streaming API — SSE vs WebSocket, heartbeats, cancellation, what happens on disconnect · *Code (45m):* Practical: consistent-hash ring with virtual nodes — you need it Monday · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Fri Aug 28** *(4h)* — *Build (1h 30m):* P4-M2: quantization sweep — tok/s, VRAM, load time across Q4_K_M / Q5 / Q8 / fp16, GPU and CPU · *Outreach (45m):* Message 10 Purdue alumni at target companies · *Design (35m):* Component: prefix-cache-aware routing — how do you keep a session on the pod already holding its KV? · *Code (25m):* Revise: binary search, including search-on-answer · *Study (45m):* Quantization survey + LLM.int8 activation outliers

**Sat Aug 29** *(4h)* — *Build (2h):* P4-M3/M4: quality eval on 50 prompts · trade-off table · publish quantization-tradeoffs · *Write-up (35m):* Publish inference-from-scratch: README with your KV cache graphs · *Design (50m):* System design drill #1: Scenario 1 — 70B model, sub-200ms TTFT (35 min, whiteboard) · *Code (35m):* Revise: BFS / DFS on grids

**Sun Aug 30** *(2h 25m)* — *Build (1h 15m):* REWRITE THE QWEN-VL BULLET with your measured numbers — one liability down · *Plan (20m):* Sketch the cluster on paper: gateway → coordinator → workers, and what you measure · *Design (30m):* Component: admission control and load shedding — where exactly does the 429 come from, and at what queue depth? · *Code (20m):* Revise: graphs — topological sort and union-find


## Phase 2 · Resume rebuild


### Week 2 · Aug 31 → Sep 6

**Mon Aug 31** *(6h 30m)* — *Build (2h 30m):* P3-M1: kind cluster up; workers (vLLM or llama.cpp) deployed and serving · *Study (1h 15m):* vLLM distributed docs · prefix caching + RadixAttention · *Design (1h):* Component: chunked prefill — draw the batch timeline with and without it, mark where ITL spikes · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Tue Sep 1** *(6h 30m)* — *Build (2h 30m):* P3-M2 + M3: worker wrapper reporting load/cache stats · coordinator + consistent-hash ring on prefix hash · *Study (1h 15m):* Mooncake skim — a cluster-wide KV store is exactly what you're building · *Design (1h):* Component: the metrics endpoint — which numbers, at which percentiles, and which one is your north star? · *Code (45m):* Practical: worker protocol (REST/gRPC) + health checks · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Wed Sep 2** *(6h 30m)* — *Build (2h 30m):* P3-M4 + M5: gateway with SSE passthrough · per-worker KV budget + LRU eviction + hit/miss metrics · *Study (1h 15m):* BentoML handbook: TTFT, ITL, goodput — the metrics you're about to report · *Design (1h):* Capacity drill: 1,000 req/s at 1k prompt / 300 output — how many replicas, and what is your headroom policy? · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Thu Sep 3** *(6h 30m)* — *Build (2h 30m):* P3-M6: affinity-ON vs OFF TTFT benchmark at several concurrency levels — THE headline number · *Study (1h 15m):* Cost per million tokens · what your numbers actually mean · *Design (1h):* Component: speculative decoding — when do you turn it ON in production, and what tells you to turn it off? · *Code (45m):* Practical: streaming p99 tracker · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Fri Sep 4** *(4h)* — *Build (1h 25m):* RESUME V2: README with graphs, then rewrite the KV cache bullet with YOUR numbers · full resume pass · *Apply (25m):* Re-engage every company you've applied to: new resume + repo links · *Design (35m):* Component: quantisation choice — pick a scheme for a latency-critical chat product and defend it to a sceptic · *Code (25m):* Revise: trees — DFS, level order, lowest common ancestor · *Outreach (45m):* LinkedIn + GitHub pinned repos updated, then outreach ×10 with the artifacts · *Write-up (25m):* Publish: 'What I learned rebuilding a distributed KV cache solo'

**Sat Sep 5** *(4h 5m)* — *Study (1h 5m):* Re-read your own KV cache code and explain every line out loud — the parts you skim are the parts you don't know · *Catch-up (1h 30m):* Recorded 30-min walkthroughs: the cluster and the quantization study · *Design (50m):* System design drill #2: Scenario 2 — GPU autoscaling without thrash · *Code (40m):* Revise: monotonic stack and expression parsing

**Sun Sep 6** *(2h 30m)* — *Self-test (35m):* Full self-test across sections A–D — mid-plan baseline · *Plan (35m):* Apply to your top-10 dream list with resume v2 · plan the hardening week · *Design (45m):* Debug drill: p99 tripled, p50 flat — run it as an incident, out loud, hypothesis ladder first · *Code (35m):* Revise: linked lists — reverse, detect cycle, merge k


### Week 3 · Sep 7 → Sep 13

**Mon Sep 7** *(6h 30m)* — *Build (2h 30m):* P3-M7: StatefulSets + HPA on queue depth · *Study (1h 15m):* Autoscaling signals: queue depth and KV utilisation, never CPU% · *Design (1h):* Debug drill: TTFT is fine but ITL is terrible — what is your first graph, and your first three hypotheses? · *Code (45m):* LeetCode ×2 · *Apply (45m):* Application blitz: 5 applications with resume v2, not 3 · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Tue Sep 8** *(6h 30m)* — *Build (2h 30m):* P3-M3 hardening: worker join/leave with minimal remap · write the test · *Study (1h 15m):* Consistent hashing vs rendezvous vs a central routing table — trade-offs cold · *Design (1h):* Debug drill: throughput halved right after a deploy — walk the diff and the metrics together · *Code (45m):* Practical: retry/timeout wrapper with jitter · *Apply (45m):* 5 applications · nudge anyone quiet · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Wed Sep 9** *(6h 30m)* — *Build (2h 30m):* Chaos test: kill a worker mid-load, document the recovery behaviour · *Study (1h 15m):* Question bank section G — the debugging answers · *Design (1h):* Debug drill: one tenant's traffic destroyed the fleet's p99 — find it, then design so it cannot recur · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Thu Sep 10** *(6h 30m)* — *Build (2h 30m):* Terraform the cluster · optional single GKE GPU run inside free credits · *Study (1h 15m):* Re-run the benchmarks — upgrade the resume number if hardening improved it · *Design (1h):* Debug drill: cache hit rate collapsed overnight and TTFT doubled — what changed? · *Code (45m):* Practical: idempotent work queue · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Fri Sep 11** *(4h 5m)* — *Study (1h 5m):* Question bank section A — answer out loud, score yourself, queue anything under 3 · *Outreach (50m):* Outreach ×10 — lead with the write-up and the affinity graph · *Design (50m):* Variant: multi-region serving with data residency — routing, replication, and what never crosses a border · *Code (40m):* Revise: 1-D DP — climbing, house robber, coin change · *Write-up (40m):* Weekly write-up + commit/push everything

**Sat Sep 12** *(4h 5m)* — *Study (1h 5m):* Question bank section B (KV cache) — redo every calculation on paper · *Design (50m):* System design drill #3: Scenario 3 — p99 tripled, p50 flat (incident response) · *Catch-up (1h 30m):* Cluster README final pass · reproducibility script · *Code (40m):* Revise: 2-D DP — grid paths, edit distance, LCS

**Sun Sep 13** *(2h 30m)* — *Self-test (35m):* Self-test: 5 Qs from sections C + G · *Plan (35m):* Plan Phase 3 — the differentiator build · *Design (45m):* Variant: model cascade — cheap model first, escalate on low confidence. Where does the added latency come from? · *Code (35m):* Revise: greedy — scheduling and jump game


## Phase 3 · Serving stack + OSS


### Week 4 · Sep 14 → Sep 20

**Mon Sep 14** *(6h 30m)* — *Build (2h 30m):* P2-M1: FastAPI server, /generate with SSE streaming · *Study (1h 15m):* PagedAttention paper §1–4 · *Design (1h):* Variant: zero-downtime model version rollout — shadow, canary, rollback triggers · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Tue Sep 15** *(6h 30m)* — *Build (2h 30m):* P2-M2: request queue + step-loop scheduler skeleton · *Study (1h 15m):* Orca paper — iteration-level scheduling · *Design (1h):* Variant: eliminate cold starts for a 70B model — warm pools vs streamed weights vs snapshots, with costs · *Code (45m):* Practical: dynamic batcher (max_size OR max_wait) · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Wed Sep 16** *(6h 30m)* — *Build (2h 30m):* P2-M2: admit/retire logic working end to end · *Study (1h 15m):* Finish the PagedAttention paper · *Design (1h):* Variant: cut serving cost 50% without breaking the SLO — rank every lever by ROI · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Thu Sep 17** *(6h 30m)* — *Build (2h 30m):* P2-M2: Poisson load generator · static-vs-continuous benchmark · *Study (1h 15m):* Chunked prefill docs — why long prompts wreck ITL · *Design (1h):* Variant: multi-tenant isolation — MIG vs shared engine with quotas, and how noisy neighbours show up in metrics · *Code (45m):* Practical: bounded queue with backpressure · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Fri Sep 18** *(4h 5m)* — *Study (1h 5m):* Question bank section C (batching and scheduling) — out loud, timed · *Outreach (50m):* Outreach ×10 · *Design (50m):* Variant: disaggregated prefill and decode — what moves between pools, how big is it, over what link? · *Code (40m):* Revise: strings — anagrams, palindromes, tokenising · *Write-up (40m):* Weekly write-up + commit/push everything

**Sat Sep 19** *(4h 5m)* — *Study (1h 5m):* Question bank section D (quantisation) — answer from YOUR measured numbers · *Design (50m):* System design drill #4: Scenario 4 — 200 fine-tuned variants (multi-LoRA) · *Catch-up (1h 30m):* Build catch-up on mini-vllm · *Code (40m):* Revise: prefix sums and counting

**Sun Sep 20** *(2h 30m)* — *Self-test (35m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (35m):* Review the week · plan the next one · *Design (45m):* Variant: serving 100 LoRA adapters on one base model — adapter cache, routing, eviction storms · *Code (35m):* Revise: matrix — rotate, spiral, search a sorted matrix


### Week 5 · Sep 21 → Sep 27

**Mon Sep 21** *(6h 30m)* — *Build (2h 30m):* P2-M3: block allocator + per-sequence block tables · *Study (1h 15m):* Quantization deep: weight-only vs weight+activation · *Design (1h):* Variant: a spot-instance batch pipeline — checkpointing, idempotency, what happens on preemption · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Tue Sep 22** *(6h 30m)* — *Build (2h 30m):* P2-M3: wire blocks into the attention path · *Study (1h 15m):* FP8 / int4 trade-offs — answer section D from your own data · *Design (1h):* Variant: design the benchmark harness you would trust before signing off a model swap · *Code (45m):* Practical: LFU cache variant · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Wed Sep 23** *(6h 30m)* — *Build (2h 30m):* P2-M4: preemption — evict + recompute on resume · *Study (1h 15m):* Read vLLM's preemption path in source · *Design (1h):* Rebuild from memory: Scenario 1 (70B, sub-200ms TTFT) — no notes, 35 minutes · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Thu Sep 24** *(6h 30m)* — *Build (2h 30m):* P2-M4: expose a preemption counter in metrics · *Study (1h 15m):* Question bank sections D + F review · *Design (1h):* Rebuild from memory: Scenario 2 (autoscaling) — no notes, 35 minutes · *Code (45m):* Practical of choice · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Fri Sep 25** *(4h 5m)* — *Study (1h 5m):* Question bank section E (speculative decoding) — including the exactness argument · *Outreach (50m):* Outreach ×10 · *Design (50m):* Rebuild from memory: Scenario 4 (multi-LoRA) — no notes, 35 minutes · *Code (40m):* Revise: backtracking — subsets, permutations, word search · *Write-up (40m):* Weekly write-up + commit/push everything

**Sat Sep 26** *(4h 5m)* — *Study (1h 5m):* Question bank section F (parallelism) — size three deployments live · *Design (50m):* System design drill #5: Scenario 5 — 50M docs/day offline batch, cheapest · *Catch-up (1h 30m):* Build catch-up · *Code (40m):* Revise: design questions — LRU, min-stack, hit counter

**Sun Sep 27** *(2h 30m)* — *Self-test (35m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (35m):* Review the week · plan the next one · *Design (45m):* Rebuild from memory: Scenario 5 (offline batch) — no notes, 35 minutes · *Code (35m):* Revise: custom comparators and sort-based problems


### Week 6 · Sep 28 → Oct 4

**Mon Sep 28** *(6h 30m)* — *Build (2h 30m):* P2-M5: prefix reuse + LRU block cache (your cluster, one level down) · *Study (1h 15m):* Leviathan speculative decoding paper · *Design (1h):* Rebuild from memory: Scenario 6 (RAG serving) — no notes, 35 minutes · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Tue Sep 29** *(6h 30m)* — *Build (2h 30m):* P2-M6: /metrics — TTFT & ITL histograms, queue depth, KV utilisation · *Study (1h 15m):* Spec-decode acceptance math: α=0.8, γ=4 → expected tokens/pass · *Design (1h):* Explain your own cluster as a design interview — requirements, maths, architecture, trade-offs, 20 minutes · *Code (45m):* Practical: SSE + backpressure · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Wed Sep 30** *(6h 30m)* — *Build (2h 30m):* Load test · throughput-vs-p99 curve · mark the knee · *Study (1h 15m):* Toy spec decode: 0.5B drafts for 1.5B — measure acceptance rate · *Design (1h):* Explain mini-vLLM as a design interview — and name every divergence from real vLLM · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Thu Oct 1** *(6h 30m)* — *Build (2h 30m):* Publish mini-vllm: README with every graph · 30-min recorded walkthrough · *Study (1h 15m):* Skim EAGLE + Medusa — one-line idea each · *Design (1h):* Rapid-fire: five capacity questions in 15 minutes, numbers only · *Code (45m):* Practical of choice · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Fri Oct 2** *(4h 5m)* — *Study (1h 5m):* Question bank section G (production debugging) — the hypothesis ladders · *Outreach (50m):* Outreach ×10 + post mini-vllm on LinkedIn · *Design (50m):* Rapid-fire: five 'when would you NOT do this?' questions — batching, spec decode, disaggregation, quantisation, caching · *Code (40m):* Revise: your two weakest patterns, from the log · *Write-up (40m):* Weekly write-up + commit/push everything

**Sat Oct 3** *(4h 5m)* — *Study (1h 5m):* Skim the vLLM release notes since your version — what changed, and why would it change your benchmarks? · *Design (50m):* System design drill #6: Scenario 6 — RAG assistant, 10-turn, big shared docs · *Catch-up (1h 30m):* Repo polish across all four projects · *Code (40m):* Revise: arrays + hashmap — 2 timed mediums

**Sun Oct 4** *(2h 30m)* — *Self-test (35m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (35m):* Review the week · plan the next one · *Design (45m):* Interruption practice: pick any scenario, have someone derail you twice, return to the method both times · *Code (35m):* Revise: two pointers / sliding window — 2 timed


### Week 7 · Oct 5 → Oct 11

**Mon Oct 5** *(6h 30m)* — *Build (2h 30m):* OSS: claim 2–3 good-first-issues in vLLM or SGLang · *Study (1h 15m):* Tensor vs pipeline parallelism — size a 70B deployment out loud · *Design (1h):* Learn the 5-step method: requirements → capacity maths → architecture → deep-dive → failure modes. Write it on a card you keep · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Tue Oct 6** *(6h 30m)* — *Build (2h 30m):* OSS: first pull request submitted · *Study (1h 15m):* DistServe + Mooncake — disaggregation and the KV transfer math · *Design (1h):* Capacity drill: how many H100s to serve Llama-3-70B in fp16? Do the arithmetic out loud, then again in FP8 · *Code (45m):* Practical of choice · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Wed Oct 7** *(6h 30m)* — *Build (2h 30m):* OSS: address review comments · *Study (1h 15m):* Module 8 ops: SLOs, autoscaling signals, cold starts, observability · *Design (1h):* Capacity drill: concurrent sequences per 80GB card at 8k context on a GQA 8B — derive it from the KV formula · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Thu Oct 8** *(6h 30m)* — *Build (2h 30m):* Second PR, or deepen the first · *Study (1h 15m):* Question bank sections F + G, out loud · *Design (1h):* SLO drill: write numeric TTFT / ITL / availability targets for chat, a coding copilot, and batch summarisation — defend each · *Code (45m):* Practical of choice · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Fri Oct 9** *(4h 5m)* — *Study (1h 5m):* Read one inference-cloud engineering post (Baseten / Modal / Fireworks / Anyscale) and extract one number you didn't know · *Outreach (50m):* Outreach ×10 — mention the merged/open PR · *Design (50m):* Component: draw the request lifecycle from HTTP ingress to last token, and name every stage you would instrument · *Code (40m):* Revise: heap and top-k — 2 timed · *Write-up (40m):* Weekly write-up + commit/push everything

**Sat Oct 10** *(4h 5m)* — *Study (1h 5m):* Re-derive the KV cache size formula and apply it to three models from their config.json · *Design (50m):* System design drill #7: Scenario 1 again — timed, and have someone interrupt you · *Catch-up (1h 30m):* Rebuild one component from memory · *Code (40m):* Revise: intervals — merge, insert, minimum rooms

**Sun Oct 11** *(2h 30m)* — *Self-test (35m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (35m):* Review the week · plan the next one · *Design (45m):* Component: static vs continuous batching — draw both schedulers and explain the throughput delta with numbers · *Code (35m):* Revise: binary search, including search-on-answer


### Week 8 · Oct 12 → Oct 18

**Mon Oct 12** *(6h 30m)* — *Build (2h 30m):* Extend mini-vllm: chunked prefill, or spec decode wired in · *Study (1h 15m):* Re-read 'Inside vLLM' now that you've built one — it reads differently · *Design (1h):* Component: KV memory budgeting — given a VRAM budget, how many concurrent requests, and what do you evict first? · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Tue Oct 13** *(6h 30m)* — *Build (2h 30m):* Walkthrough rehearsals: all four projects, 5 minutes each with numbers · *Study (1h 15m):* Module 7: distributed inference recap · *Design (1h):* Component: design the streaming API — SSE vs WebSocket, heartbeats, cancellation, what happens on disconnect · *Code (45m):* Practical of choice · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Wed Oct 14** *(6h 30m)* — *Build (2h 30m):* Fix whatever the rehearsals exposed — usually a missing graph or a shaky number · *Study (1h 15m):* Question bank sections A–C, out loud · *Design (1h):* Component: prefix-cache-aware routing — how do you keep a session on the pod already holding its KV? · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Thu Oct 15** *(6h 30m)* — *Build (2h 30m):* Buffer: finish anything unfinished across all four projects · *Study (1h 15m):* Behavioral: draft 5 STAR stories · *Design (1h):* Component: admission control and load shedding — where exactly does the 429 come from, and at what queue depth? · *Code (45m):* Practical of choice · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Fri Oct 16** *(4h 5m)* — *Study (1h 5m):* Trace one request through vLLM's source — scheduler, block manager, model runner · *Outreach (50m):* Outreach ×10 · nudge every live application · *Design (50m):* Component: chunked prefill — draw the batch timeline with and without it, mark where ITL spikes · *Code (40m):* Revise: BFS / DFS on grids · *Write-up (40m):* Weekly write-up + commit/push everything

**Sat Oct 17** *(4h 5m)* — *Study (1h 5m):* Read the SGLang RadixAttention docs and compare it to what your own prefix cache does · *Catch-up (1h 30m):* Rebuild a second component from memory · *Design (50m):* System design drill #8: rapid-fire variants — multi-region, model cascade, capacity planning · *Code (40m):* Revise: graphs — topological sort and union-find

**Sun Oct 18** *(2h 30m)* — *Self-test (35m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (35m):* Review the week · plan the next one · *Design (45m):* Component: the metrics endpoint — which numbers, at which percentiles, and which one is your north star? · *Code (35m):* Revise: trees — DFS, level order, lowest common ancestor


### Week 9 · Oct 19 → Oct 25

**Mon Oct 19** *(6h 30m)* — *Build (2h 30m):* Portfolio polish: 4 consistent READMEs, pinned repos, LinkedIn featured section · *Study (1h 15m):* Re-read your own write-ups as an interviewer would · *Design (1h):* Capacity drill: 1,000 req/s at 1k prompt / 300 output — how many replicas, and what is your headroom policy? · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Tue Oct 20** *(6h 30m)* — *Build (2h 30m):* Write one public post: 'What I learned building an inference stack from scratch' · *Study (1h 15m):* Company research: eng blogs for your live pipeline · *Design (1h):* Component: speculative decoding — when do you turn it ON in production, and what tells you to turn it off? · *Code (45m):* Practical of choice · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Wed Oct 21** *(6h 30m)* — *Build (2h 30m):* Address anything weak in the portfolio · final graph pass · *Study (1h 15m):* Question bank sections D–F, out loud · *Design (1h):* Component: quantisation choice — pick a scheme for a latency-critical chat product and defend it to a sceptic · *Code (45m):* LeetCode ×2 · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Thu Oct 22** *(6h 30m)* — *Build (2h 30m):* Buffer + rest of the STAR stories · *Study (1h 15m):* Behavioral rehearsal: the pivot story, 90 seconds · *Design (1h):* Debug drill: p99 tripled, p50 flat — run it as an incident, out loud, hypothesis ladder first · *Code (45m):* Practical of choice · *Apply (45m):* Send 3 applications + log them in the job tracker · *Close (15m):* Log notes · say 2 question-bank answers out loud

**Fri Oct 23** *(4h 5m)* — *Study (1h 5m):* Arithmetic hour: prefill vs decode FLOPs and bytes for three model sizes, no notes · *Outreach (50m):* Outreach ×10 · confirm interview slots · *Design (50m):* Debug drill: TTFT is fine but ITL is terrible — what is your first graph, and your first three hypotheses? · *Code (40m):* Revise: monotonic stack and expression parsing · *Write-up (40m):* Weekly write-up + commit/push everything

**Sat Oct 24** *(3h)* — *Design (1h):* System design drill #9: Scenarios 2 and 5 back to back, no break · *Study (1h 15m):* Book the four mocks (peers / interviewing.io) · clear the revision queue · *Code (45m):* Revise: linked lists — reverse, detect cycle, merge k

**Sun Oct 25** *(2h 30m)* — *Self-test (35m):* FULL self-test across every section — Phase-4 entry score · *Plan (35m):* Plan interview mode against your actual interview calendar · *Design (45m):* Debug drill: throughput halved right after a deploy — walk the diff and the metrics together · *Code (35m):* Revise: 1-D DP — climbing, house robber, coin change


## Phase 4 · Interview mode


### Week 10 · Oct 26 → Nov 1

**Mon Oct 26** *(4h 15m)* — *Drill (1h 30m):* Drill: question bank A–B out loud, rubric-scored · *Design (1h):* Whiteboard: Scenario 1, timed 35 min · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: 2-D DP — grid paths, edit distance, LCS · *Close (15m):* Queue your weakest answer for revision

**Tue Oct 27** *(4h 15m)* — *Drill (1h 30m):* Drill: sections C–D · *Design (1h):* Whiteboard: Scenario 2 · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: greedy — scheduling and jump game · *Close (15m):* Queue your weakest answer for revision

**Wed Oct 28** *(5h 45m)* — *Mock (2h):* MOCK #1 — system design round (peer or interviewing.io) · *Study (1h 15m):* Watch the recording back · queue every fumble for revision · *Design (1h):* Debug drill: one tenant's traffic destroyed the fleet's p99 — find it, then design so it cannot recur · *Code (45m):* Revise: strings — anagrams, palindromes, tokenising · *Apply (45m):* Applications · nudges

**Thu Oct 29** *(4h 15m)* — *Drill (1h 30m):* Drill: everything Mock #1 exposed · *Design (1h):* Whiteboard: Scenario 3 (debugging) · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: prefix sums and counting · *Close (15m):* Queue your weakest answer for revision

**Fri Oct 30** *(4h 10m)* — *Study (55m):* Watch a recorded inference talk (GTC / Ray Summit / vLLM meetup) and write five bullets · *Outreach (45m):* Outreach ×10 · *Design (45m):* Debug drill: cache hit rate collapsed overnight and TTFT doubled — what changed? · *Code (35m):* Revise: matrix — rotate, spiral, search a sorted matrix · *Redo (35m):* Redo missed problems · *Write-up (35m):* Write up the mock lessons

**Sat Oct 31** *(4h 5m)* — *Catch-up (1h 30m):* Project walkthrough rehearsal: Project 3, five-whys deep · *Study (1h 5m):* Revision queue: clear everything due · *Design (50m):* Variant: multi-region serving with data residency — routing, replication, and what never crosses a border · *Code (40m):* Revise: backtracking — subsets, permutations, word search

**Sun Nov 1** *(2h 30m)* — *Self-test (35m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (35m):* Review the week · plan the next one · *Design (45m):* Variant: model cascade — cheap model first, escalate on low confidence. Where does the added latency come from? · *Code (35m):* Revise: design questions — LRU, min-stack, hit counter


### Week 11 · Nov 2 → Nov 8

**Mon Nov 2** *(4h 15m)* — *Drill (1h 30m):* Drill: sections E–F · *Design (1h):* Whiteboard: Scenario 4 (multi-LoRA) · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: custom comparators and sort-based problems · *Close (15m):* Queue your weakest answer for revision

**Tue Nov 3** *(4h 15m)* — *Drill (1h 30m):* Drill: sections G–H · *Design (1h):* Whiteboard: Scenario 5 (offline batch) · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: your two weakest patterns, from the log · *Close (15m):* Queue your weakest answer for revision

**Wed Nov 4** *(5h 45m)* — *Mock (2h):* MOCK #2 — inference deep-dive; have them grill both rebuilt projects · *Study (1h 15m):* Recording review · queue the gaps · *Design (1h):* Variant: zero-downtime model version rollout — shadow, canary, rollback triggers · *Code (45m):* Revise: arrays + hashmap — 2 timed mediums · *Apply (45m):* Applications · nudges

**Thu Nov 5** *(4h 15m)* — *Drill (1h 30m):* Drill: resume walkthroughs — 5 minutes each, with numbers · *Design (1h):* Whiteboard: Scenario 6 (RAG serving) · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: two pointers / sliding window — 2 timed · *Close (15m):* Queue your weakest answer for revision

**Fri Nov 6** *(4h 5m)* — *Study (1h 5m):* Re-read the PagedAttention paper's evaluation section — what baselines, what wins, what's hidden · *Outreach (50m):* Outreach ×10 · *Design (50m):* Variant: eliminate cold starts for a 70B model — warm pools vs streamed weights vs snapshots, with costs · *Code (40m):* Revise: heap and top-k — 2 timed · *Write-up (40m):* Weekly write-up + commit/push everything

**Sat Nov 7** *(4h 5m)* — *Catch-up (1h 30m):* Behavioral: write 3 STAR stories in full, then cut each to 90 seconds · *Study (1h 5m):* Revision queue: clear everything due · *Design (50m):* Variant: cut serving cost 50% without breaking the SLO — rank every lever by ROI · *Code (40m):* Revise: intervals — merge, insert, minimum rooms

**Sun Nov 8** *(2h 30m)* — *Self-test (35m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (35m):* Review the week · plan the next one · *Design (45m):* Variant: multi-tenant isolation — MIG vs shared engine with quotas, and how noisy neighbours show up in metrics · *Code (35m):* Revise: binary search, including search-on-answer


### Week 12 · Nov 9 → Nov 15

**Mon Nov 9** *(4h 15m)* — *Drill (1h 30m):* Drill: full sweep A–D · *Design (1h):* Whiteboard: your lowest-scoring scenario · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: BFS / DFS on grids · *Close (15m):* Queue your weakest answer for revision

**Tue Nov 10** *(4h 15m)* — *Drill (1h 30m):* Drill: full sweep E–H · *Design (1h):* Whiteboard: capacity planning until it's 3 minutes flat · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: graphs — topological sort and union-find · *Close (15m):* Queue your weakest answer for revision

**Wed Nov 11** *(5h 45m)* — *Mock (2h):* MOCK #3 — coding round + applied ML-systems problem · *Study (1h 15m):* Recording review · queue the gaps · *Design (1h):* Variant: disaggregated prefill and decode — what moves between pools, how big is it, over what link? · *Code (45m):* Revise: trees — DFS, level order, lowest common ancestor · *Apply (45m):* Applications · nudges

**Thu Nov 12** *(4h 15m)* — *Drill (1h 30m):* Drill: the Mock #3 fix list · *Design (1h):* Whiteboard: interviewer-interruption practice · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: monotonic stack and expression parsing · *Close (15m):* Queue your weakest answer for revision

**Fri Nov 13** *(4h 5m)* — *Study (1h 5m):* Read about FP8 serving on Hopper and decide whether it would change your quantisation recommendation · *Outreach (50m):* Outreach ×10 · *Design (50m):* Variant: serving 100 LoRA adapters on one base model — adapter cache, routing, eviction storms · *Code (40m):* Revise: linked lists — reverse, detect cycle, merge k · *Write-up (40m):* Weekly write-up + commit/push everything

**Sat Nov 14** *(4h 5m)* — *Catch-up (1h 30m):* Rebuild a component from memory · *Study (1h 5m):* Revision queue: clear everything due · *Design (50m):* Variant: a spot-instance batch pipeline — checkpointing, idempotency, what happens on preemption · *Code (40m):* Revise: 1-D DP — climbing, house robber, coin change

**Sun Nov 15** *(2h 30m)* — *Self-test (35m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (35m):* Review the week · plan the next one · *Design (45m):* Variant: design the benchmark harness you would trust before signing off a model swap · *Code (35m):* Revise: 2-D DP — grid paths, edit distance, LCS


### Week 13 · Nov 16 → Nov 22

**Mon Nov 16** *(4h 15m)* — *Drill (1h 30m):* Drill: weakest 15 questions · *Design (1h):* Whiteboard: Scenario 1 + 2 back to back · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: greedy — scheduling and jump game · *Close (15m):* Queue your weakest answer for revision

**Tue Nov 17** *(4h 15m)* — *Drill (1h 30m):* Drill: numbers only — every calculation in the bank · *Design (1h):* Whiteboard: Scenario 3 + 4 · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: strings — anagrams, palindromes, tokenising · *Close (15m):* Queue your weakest answer for revision

**Wed Nov 18** *(5h 45m)* — *Mock (2h):* MOCK #4 — full loop simulation: coding + design + deep-dive · *Study (1h 15m):* Recording review · final fix list · *Design (1h):* Rebuild from memory: Scenario 1 (70B, sub-200ms TTFT) — no notes, 35 minutes · *Code (45m):* Revise: prefix sums and counting · *Apply (45m):* Applications · nudges

**Thu Nov 19** *(4h 15m)* — *Drill (1h 30m):* Drill: the full-loop fix list · *Design (1h):* Whiteboard: Scenario 5 + 6 · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: matrix — rotate, spiral, search a sorted matrix · *Close (15m):* Queue your weakest answer for revision

**Fri Nov 20** *(4h 5m)* — *Study (1h 5m):* Compare your mini-vLLM scheduler to vLLM's line by line and list every divergence · *Outreach (50m):* Outreach ×10 · *Design (50m):* Rebuild from memory: Scenario 2 (autoscaling) — no notes, 35 minutes · *Code (40m):* Revise: backtracking — subsets, permutations, word search · *Write-up (40m):* Weekly write-up + commit/push everything

**Sat Nov 21** *(4h 5m)* — *Catch-up (1h 30m):* Company research: eng blogs of your live interview pipeline · *Study (1h 5m):* Revision queue: clear due · *Design (50m):* Rebuild from memory: Scenario 4 (multi-LoRA) — no notes, 35 minutes · *Code (40m):* Revise: design questions — LRU, min-stack, hit counter

**Sun Nov 22** *(2h 30m)* — *Self-test (35m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (35m):* Review the week · plan the next one · *Design (45m):* Rebuild from memory: Scenario 5 (offline batch) — no notes, 35 minutes · *Code (35m):* Revise: custom comparators and sort-based problems


### Week 14 · Nov 23 → Nov 29

**Mon Nov 23** *(4h 15m)* — *Drill (1h 30m):* Drill: 10 random questions · *Design (1h):* Whiteboard: pick your weakest scenario · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: your two weakest patterns, from the log · *Close (15m):* Queue your weakest answer for revision

**Tue Nov 24** *(4h 15m)* — *Drill (1h 30m):* Drill: 10 random questions · *Design (1h):* Whiteboard: one scenario, relaxed · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: arrays + hashmap — 2 timed mediums · *Close (15m):* Queue your weakest answer for revision

**Wed Nov 25** *(3h 45m)* — *Study (1h 15m):* Light: revision queue only · *Apply (45m):* Applications — December postings are live · *Design (1h):* Rebuild from memory: Scenario 6 (RAG serving) — no notes, 35 minutes · *Code (45m):* Revise: two pointers / sliding window — 2 timed

**Thu Nov 26** *(3h 45m)* — *Study (1h 15m):* Read one production post-mortem from any infra company — map its failure mode onto LLM serving · *Plan (45m):* Off. Actually off. · *Design (1h):* Explain your own cluster as a design interview — requirements, maths, architecture, trade-offs, 20 minutes · *Code (45m):* Revise: heap and top-k — 2 timed

**Fri Nov 27** *(3h)* — *Study (1h 15m):* Light: revision queue only · *Design (1h):* Explain mini-vLLM as a design interview — and name every divergence from real vLLM · *Code (45m):* Revise: intervals — merge, insert, minimum rooms

**Sat Nov 28** *(3h 15m)* — *Drill (1h 30m):* One design drill, no pressure · *Design (1h):* Rapid-fire: five capacity questions in 15 minutes, numbers only · *Code (45m):* Revise: binary search, including search-on-answer

**Sun Nov 29** *(2h 30m)* — *Self-test (35m):* Self-test: 5 Qs · *Plan (35m):* Plan the final two weeks against your interview calendar · *Design (45m):* Rapid-fire: five 'when would you NOT do this?' questions — batching, spec decode, disaggregation, quantisation, caching · *Code (35m):* Revise: BFS / DFS on grids


### Week 15 · Nov 30 → Dec 6

**Mon Nov 30** *(4h 15m)* — *Drill (1h 30m):* Drill: everything still in the revision queue · *Design (1h):* Whiteboard: your two strongest scenarios, polished · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: graphs — topological sort and union-find · *Close (15m):* Queue your weakest answer for revision

**Tue Dec 1** *(4h 15m)* — *Drill (1h 30m):* Drill: rapid-fire — 20 questions in 40 minutes · *Design (1h):* Whiteboard: the one you avoid · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: trees — DFS, level order, lowest common ancestor · *Close (15m):* Queue your weakest answer for revision

**Wed Dec 2** *(5h 15m)* — *Drill (1h 30m):* Interviews / targeted revision · *Study (1h 15m):* Company research for the live pipeline · *Design (1h):* Interruption practice: pick any scenario, have someone derail you twice, return to the method both times · *Code (45m):* Revise: monotonic stack and expression parsing · *Apply (45m):* Follow-ups · thank-you notes

**Thu Dec 3** *(4h 15m)* — *Drill (1h 30m):* Drill: project walkthroughs, both rebuilds, cold · *Design (1h):* Whiteboard: free choice · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: linked lists — reverse, detect cycle, merge k · *Close (15m):* Queue your weakest answer for revision

**Fri Dec 4** *(4h 5m)* — *Study (1h 5m):* Review every number currently on your resume and reproduce two of them from scratch · *Outreach (50m):* Outreach ×10 — December postings exist; keep applying · *Design (50m):* Learn the 5-step method: requirements → capacity maths → architecture → deep-dive → failure modes. Write it on a card you keep · *Code (40m):* Revise: 1-D DP — climbing, house robber, coin change · *Write-up (40m):* Weekly write-up + commit/push everything

**Sat Dec 5** *(4h 5m)* — *Catch-up (1h 30m):* Behavioral stories to 90 seconds each, out loud · *Study (1h 5m):* Revision queue: clear everything · *Design (50m):* Capacity drill: how many H100s to serve Llama-3-70B in fp16? Do the arithmetic out loud, then again in FP8 · *Code (40m):* Revise: 2-D DP — grid paths, edit distance, LCS

**Sun Dec 6** *(2h 30m)* — *Self-test (35m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (35m):* Review the week · plan the next one · *Design (45m):* Capacity drill: concurrent sequences per 80GB card at 8k context on a GQA 8B — derive it from the KV formula · *Code (35m):* Revise: greedy — scheduling and jump game


### Week 16 · Dec 7 → Dec 13

**Mon Dec 7** *(5h 15m)* — *Drill (1h 30m):* Interview prep for what's on the calendar · targeted revision only · *Study (1h 15m):* Day-before ritual: their eng blog + your walkthroughs. Nothing new. · *Design (1h):* SLO drill: write numeric TTFT / ITL / availability targets for chat, a coding copilot, and batch summarisation — defend each · *Code (45m):* Revise: strings — anagrams, palindromes, tokenising · *Apply (45m):* Follow-ups · thank-you notes

**Tue Dec 8** *(5h 15m)* — *Drill (1h 30m):* Interviews / targeted revision · *Study (1h 15m):* Company research · *Design (1h):* Component: draw the request lifecycle from HTTP ingress to last token, and name every stage you would instrument · *Code (45m):* Revise: prefix sums and counting · *Apply (45m):* Follow-ups

**Wed Dec 9** *(5h 15m)* — *Drill (1h 30m):* Interviews / targeted revision · *Study (1h 15m):* Company research · *Design (1h):* Component: static vs continuous batching — draw both schedulers and explain the throughput delta with numbers · *Code (45m):* Revise: matrix — rotate, spiral, search a sorted matrix · *Apply (45m):* Follow-ups

**Thu Dec 10** *(5h 15m)* — *Drill (1h 30m):* Interviews / targeted revision · *Study (1h 15m):* Company research · *Design (1h):* Component: KV memory budgeting — given a VRAM budget, how many concurrent requests, and what do you evict first? · *Code (45m):* Revise: backtracking — subsets, permutations, word search · *Apply (45m):* Follow-ups

**Fri Dec 11** *(4h 10m)* — *Study (55m):* Re-read your own KV cache code and explain every line out loud — the parts you skim are the parts you don't know · *Outreach (45m):* Outreach + follow-ups · *Design (45m):* Component: design the streaming API — SSE vs WebSocket, heartbeats, cancellation, what happens on disconnect · *Code (35m):* Revise: design questions — LRU, min-stack, hit counter · *Redo (35m):* Light revision of weak spots · *Write-up (35m):* Negotiate with data if offers land

**Sat Dec 12** *(4h)* — *Drill (1h 30m):* Final polish: whichever scenario is next in your loop · *Plan (45m):* Rest. You did the work. · *Design (1h):* Component: prefix-cache-aware routing — how do you keep a session on the pod already holding its KV? · *Code (45m):* Revise: custom comparators and sort-based problems

**Sun Dec 13** *(2h 30m)* — *Study (50m):* Question bank section A — answer out loud, score yourself, queue anything under 3 · *Plan (30m):* Review the whole run · write down what you learned · graduate 🎓 · *Design (40m):* Component: admission control and load shedding — where exactly does the 429 come from, and at what queue depth? · *Code (30m):* Revise: your two weakest patterns, from the log


---

## Tracking

The tracker does the counting: XP, level, streak, days complete, applications, revision queue, and a consistency heatmap. Rules that matter more than the numbers: a day counts toward the streak at 60% — a day where life happens but the build slot got done still keeps the chain. Miss one day, fine. Never miss two.
