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

**Thu Aug 20** *(6h)* — *Study (2h):* Pick your spine — Karpathy 'Let's build GPT' or Vizuara's series — and get through the first two hours tonight · *Setup (1h):* Environment: PyTorch + CUDA on the 2060, HF account, pull Qwen2.5-0.5B, verify a forward pass · *Setup (1h):* Install llama.cpp + download a Qwen2-VL GGUF — quantization jobs start Monday · *Setup (1h):* Set up your job tracker — duplicate the Notion template, add tier-1 / tier-2 columns · *Apply (45m):* Apply to SDE / backend roles with your current resume — 5 today, keep the pipeline warm · *Small thing (15m):* Fix the resume: graduation date → December 2026

**Fri Aug 21** *(3h)* — *Study (1h):* Continue the spine through self-attention — type the code alongside, don't just watch · *Apply (45m):* 5 more SDE / backend applications · *Setup (1h):* Tidy LinkedIn and GitHub — clean, current, nothing pinned yet · *Small thing (15m):* Rename the resume file to Pranay_Chimmani_Inference_Engineer.pdf — recruiters see the filename

**Sat Aug 22** *(2h 45m)* — *Study (1h):* Continue the spine: multi-head attention and the full block — type it, don't watch it · *Code (45m):* 2 timed LeetCode mediums — rust check · *Apply (45m):* 5 SDE applications · note which job titles keep appearing on the inference side · *Small thing (15m):* Check your resume PDF has selectable text, not an image — copy-paste it into a text editor and see what an ATS sees

**Sun Aug 23** *(2h)* — *Study (1h):* Finish the spine + skim Illustrated Transformer · *Plan (45m):* Read the Project 1/3/4 specs + the design playbook's 5-step method — you use it Saturday · *Small thing (15m):* Save a plain-text version of the resume for ATS forms that mangle PDFs


## Phase 1 · Fundamentals + Qwen-VL fix


### Week 1 · Aug 24 → Aug 30

**Mon Aug 24** *(6h 30m)* — *Build (2h 30m):* P1-M1: write your own model module; load Qwen2.5-0.5B weights · *Build (2h):* RESUME DAY: build the full inference resume — every project, done and planned, written once · *Apply (1h 30m):* Search TODAY's inference postings, tier them, then send NVIDIA new-grad + 2 tier-2 — the true resume · *Close (15m):* Kick off P4-M1 in the background: quantize the checkpoint 4 ways · *Small thing (15m):* Turn on LinkedIn 'Open to work', recruiters-only — set the titles to inference / ML infra, not 'software engineer'

**Tue Aug 25** *(6h 30m)* — *Build (2h 30m):* P1-M1 parity test vs HuggingFace (~1e-4) · P1-M2 naive greedy loop · *Study (1h):* kipply 'Transformer Inference Arithmetic' · compute your 2060's decode ceiling · *Design (45m):* Learn the 5-step method: requirements → capacity maths → architecture → deep-dive → failure modes. Write it on a card you keep · *Code (45m):* Practical: LRU cache from scratch — the structure you'll use for KV eviction · *Career (1h):* Applications ×4 · then message 2 Purdue alumni at companies on your tier-2 list · *Close (15m):* Check the quantization jobs · log notes · *Small thing (15m):* Add the inference keywords to your LinkedIn Skills section — that is what recruiter search actually queries

**Wed Aug 26** *(6h 30m)* — *Build (2h 30m):* P1-M3: implement the KV cache — the single most important exercise here · *Study (1h):* Serve Qwen2.5-1.5B in vLLM on the 2060; benchmark it against your own loop · *Design (45m):* Capacity drill: how many H100s to serve Llama-3-70B in fp16? Do the arithmetic out loud, then again in FP8 · *Code (45m):* LeetCode ×2 (heap / two-pointer) · *Career (1h):* Applications ×4 · follow up with everyone who has gone quiet for 7+ days · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Verify your GitHub commit email so your contributions actually show on your profile

**Thu Aug 27** *(6h 30m)* — *Build (2h 30m):* KV cache before/after benchmark + sampling (temp / top-k / top-p) · *Study (1h):* vLLM prefix-caching experiment: 1k shared prompt, TTFT hit vs miss — the exact effect your cluster will exploit · *Design (45m):* Capacity drill: concurrent sequences per 80GB card at 8k context on a GQA 8B — derive it from the KV formula · *Code (45m):* Practical: consistent-hash ring with virtual nodes — you need it Monday · *Career (1h):* Applications ×3 · refresh the resume with anything that shipped this week — numbers only, no rewrites · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Add a photo and your real name to GitHub — an anonymous profile reads as a throwaway

**Fri Aug 28** *(4h)* — *Build (2h 5m):* P4-M2: quantization sweep — tok/s, VRAM, load time across Q4_K_M / Q5 / Q8 / fp16, GPU and CPU · *Outreach (1h 5m):* Message 10 Purdue alumni at target companies · *Study (50m):* Quantization survey + LLM.int8 activation outliers

**Sat Aug 29** *(4h)* — *Build (2h 30m):* P4-M3/M4: quality eval on 50 prompts · trade-off table · publish quantization-tradeoffs · *Write-up (45m):* Publish inference-from-scratch: README with your KV cache graphs · *Design (45m):* System design drill #1: Scenario 1 — 70B model, sub-200ms TTFT (35 min, whiteboard)

**Sun Aug 30** *(2h 30m)* — *Build (1h 55m):* REWRITE THE QWEN-VL BULLET with your measured numbers — one liability down · *Plan (35m):* Sketch the cluster on paper: gateway → coordinator → workers, and what you measure


## Phase 2 · Resume rebuild


### Week 2 · Aug 31 → Sep 6

**Mon Aug 31** *(6h 30m)* — *Build (2h 30m):* P3-M1: kind cluster up; workers (vLLM or llama.cpp) deployed and serving · *Study (1h):* vLLM distributed docs · prefix caching + RadixAttention · *Design (45m):* SLO drill: write numeric TTFT / ITL / availability targets for chat, a coding copilot, and batch summarisation — defend each · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×4 · find one engineer whose post you actually read and send a specific question · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Write a GitHub profile README: one paragraph on what you build, links to the projects

**Tue Sep 1** *(6h 30m)* — *Build (2h 30m):* P3-M2 + M3: worker wrapper reporting load/cache stats · coordinator + consistent-hash ring on prefix hash · *Study (1h):* Mooncake skim — a cluster-wide KV store is exactly what you're building · *Design (45m):* Component: draw the request lifecycle from HTTP ingress to last token, and name every stage you would instrument · *Code (45m):* Practical: worker protocol (REST/gRPC) + health checks · *Career (1h):* Applications ×4 · tracker hygiene: every row has a status and a next-action date, or it gets closed · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Check the phone number on your resume can receive US calls, and that voicemail is set up and professional

**Wed Sep 2** *(6h 30m)* — *Build (2h 30m):* P3-M4 + M5: gateway with SSE passthrough · per-worker KV budget + LRU eviction + hit/miss metrics · *Study (1h):* BentoML handbook: TTFT, ITL, goodput — the metrics you're about to report · *Design (45m):* Component: static vs continuous batching — draw both schedulers and explain the throughput delta with numbers · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×3 · write one STAR story in full, then cut it to 90 seconds out loud · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Make your email signature: name, one-line title, LinkedIn, GitHub. Every outreach message carries it

**Thu Sep 3** *(6h 30m)* — *Build (2h 30m):* P3-M6: affinity-ON vs OFF TTFT benchmark at several concurrency levels — THE headline number · *Study (1h):* Cost per million tokens · what your numbers actually mean · *Design (45m):* Component: KV memory budgeting — given a VRAM budget, how many concurrent requests, and what do you evict first? · *Code (45m):* Practical: streaming p99 tracker · *Career (1h):* Applications ×4 · research two companies in your pipeline: read an eng blog post each, note one question to ask them · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Bookmark the careers pages of your top 10 targets directly — aggregators miss half the postings

**Fri Sep 4** *(4h)* — *Build (1h 55m):* RESUME V2: README with graphs, then rewrite the KV cache bullet with YOUR numbers · full resume pass · *Apply (35m):* Re-engage every company you've applied to: new resume + repo links · *Outreach (55m):* LinkedIn + GitHub pinned repos updated, then outreach ×10 with the artifacts · *Write-up (35m):* Publish: 'What I learned rebuilding a distributed KV cache solo'

**Sat Sep 5** *(3h 45m)* — *Study (1h):* Re-read your own KV cache code and explain every line out loud — the parts you skim are the parts you don't know · *Catch-up (1h 45m):* Recorded 30-min walkthroughs: the cluster and the quantization study · *Design (45m):* System design drill #2: Scenario 2 — GPU autoscaling without thrash · *Small thing (15m):* Set a weekly calendar reminder to check those 10 pages

**Sun Sep 6** *(1h 45m)* — *Self-test (45m):* Full self-test across sections A–D — mid-plan baseline · *Plan (45m):* Apply to your top-10 dream list with resume v2 · plan the hardening week · *Small thing (15m):* Standardise every date format on the resume — inconsistency reads as carelessness


### Week 3 · Sep 7 → Sep 13

**Mon Sep 7** *(6h 30m)* — *Build (2h 30m):* P3-M7: StatefulSets + HPA on queue depth · *Study (1h):* Autoscaling signals: queue depth and KV utilisation, never CPU% · *Design (45m):* Component: design the streaming API — SSE vs WebSocket, heartbeats, cancellation, what happens on disconnect · *Code (45m):* LeetCode ×2 · *Career (1h):* Application blitz: 5 applications with resume v2, not 3 · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Create a folder for tailored resume versions so you never overwrite the master

**Tue Sep 8** *(6h 30m)* — *Build (2h 30m):* P3-M3 hardening: worker join/leave with minimal remap · write the test · *Study (1h):* Consistent hashing vs rendezvous vs a central routing table — trade-offs cold · *Design (45m):* Component: prefix-cache-aware routing — how do you keep a session on the pod already holding its KV? · *Code (45m):* Practical: retry/timeout wrapper with jitter · *Career (1h):* 5 applications · nudge anyone quiet · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Write the two-sentence 'what I'm looking for' blurb you paste into application forms

**Wed Sep 9** *(6h 30m)* — *Build (2h 30m):* Chaos test: kill a worker mid-load, document the recovery behaviour · *Study (1h):* Question bank section G — the debugging answers · *Design (45m):* Component: admission control and load shedding — where exactly does the 429 come from, and at what queue depth? · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×4 · LinkedIn: one post or one comment on someone's inference work. Visibility compounds · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Whitelist recruiter domains and check your spam folder — people have missed interviews this way

**Thu Sep 10** *(6h 30m)* — *Build (2h 30m):* Terraform the cluster · optional single GKE GPU run inside free credits · *Study (1h):* Re-run the benchmarks — upgrade the resume number if hardening improved it · *Design (45m):* Component: chunked prefill — draw the batch timeline with and without it, mark where ITL spikes · *Code (45m):* Practical: idempotent work queue · *Career (1h):* Applications ×3 · re-read your resume as a hostile screener and fix the weakest line · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Add your city and work authorisation status to the resume header if it is not already clear

**Fri Sep 11** *(3h 45m)* — *Study (1h):* Question bank section A — answer out loud, score yourself, queue anything under 3 · *Outreach (1h):* Outreach ×10 — lead with the write-up and the affinity graph · *Design (45m):* Component: the metrics endpoint — which numbers, at which percentiles, and which one is your north star? · *Write-up (45m):* Weekly write-up + commit/push everything · *Small thing (15m):* Set your LinkedIn URL and GitHub URL side by side on the resume, both clickable in the PDF

**Sat Sep 12** *(3h 45m)* — *Study (1h):* Question bank section B (KV cache) — redo every calculation on paper · *Design (45m):* System design drill #3: Scenario 3 — p99 tripled, p50 flat (incident response) · *Catch-up (1h 45m):* Cluster README final pass · reproducibility script · *Small thing (15m):* Test your resume link on a phone — half of recruiters open it there first

**Sun Sep 13** *(1h 45m)* — *Self-test (45m):* Self-test: 5 Qs from sections C + G · *Plan (45m):* Plan Phase 3 — the differentiator build · *Small thing (15m):* Check that every link in your resume actually resolves; dead links are silent rejections


## Phase 3 · Serving stack + OSS


### Week 4 · Sep 14 → Sep 20

**Mon Sep 14** *(6h 30m)* — *Build (2h 30m):* P2-M1: FastAPI server, /generate with SSE streaming · *Study (1h):* PagedAttention paper §1–4 · *Design (45m):* Capacity drill: 1,000 req/s at 1k prompt / 300 output — how many replicas, and what is your headroom policy? · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×4 · ask one alum for a referral, explicitly and specifically — name the req · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Add one-line descriptions to your pinned GitHub repos with the headline number in them

**Tue Sep 15** *(6h 30m)* — *Build (2h 30m):* P2-M2: request queue + step-loop scheduler skeleton · *Study (1h):* Orca paper — iteration-level scheduling · *Design (45m):* Component: speculative decoding — when do you turn it ON in production, and what tells you to turn it off? · *Code (45m):* Practical: dynamic batcher (max_size OR max_wait) · *Career (1h):* Applications ×4 · check which of your applications went cold and why; adjust the targeting · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Set up a Calendly or equivalent so scheduling never costs you three emails

**Wed Sep 16** *(6h 30m)* — *Build (2h 30m):* P2-M2: admit/retire logic working end to end · *Study (1h):* Finish the PagedAttention paper · *Design (45m):* Component: quantisation choice — pick a scheme for a latency-critical chat product and defend it to a sceptic · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×3 · rehearse the 'why inference, why now' answer until it is 60 seconds and true · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Practise saying your name and a one-sentence intro out loud until it sounds unforced

**Thu Sep 17** *(6h 30m)* — *Build (2h 30m):* P2-M2: Poisson load generator · static-vs-continuous benchmark · *Study (1h):* Chunked prefill docs — why long prompts wreck ITL · *Design (45m):* Debug drill: p99 tripled, p50 flat — run it as an incident, out loud, hypothesis ladder first · *Code (45m):* Practical: bounded queue with backpressure · *Career (1h):* Applications ×4 · update the README of whichever project you touched most recently · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Put your interview calendar in one place and block prep time before each slot, not after

**Fri Sep 18** *(3h 45m)* — *Study (1h):* Question bank section C (batching and scheduling) — out loud, timed · *Outreach (1h):* Outreach ×10 · *Design (45m):* Debug drill: TTFT is fine but ITL is terrible — what is your first graph, and your first three hypotheses? · *Write-up (45m):* Weekly write-up + commit/push everything · *Small thing (15m):* Re-read your last five outreach messages — would you reply to them?

**Sat Sep 19** *(3h 45m)* — *Study (1h):* Question bank section D (quantisation) — answer from YOUR measured numbers · *Design (45m):* System design drill #4: Scenario 4 — 200 fine-tuned variants (multi-LoRA) · *Catch-up (1h 45m):* Build catch-up on mini-vllm · *Small thing (15m):* Add alt text or captions to the graphs in your project READMEs

**Sun Sep 20** *(1h 45m)* — *Self-test (45m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (45m):* Review the week · plan the next one · *Small thing (15m):* Check your LinkedIn headline says what you do, not what you want


### Week 5 · Sep 21 → Sep 27

**Mon Sep 21** *(6h 30m)* — *Build (2h 30m):* P2-M3: block allocator + per-sequence block tables · *Study (1h):* Quantization deep: weight-only vs weight+activation · *Design (45m):* Debug drill: throughput halved right after a deploy — walk the diff and the metrics together · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×4 · two referral messages · thank anyone who replied this week · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Make sure your resume is one consistent font and the margins are not cheating for space

**Tue Sep 22** *(6h 30m)* — *Build (2h 30m):* P2-M3: wire blocks into the attention path · *Study (1h):* FP8 / int4 trade-offs — answer section D from your own data · *Design (45m):* Debug drill: one tenant's traffic destroyed the fleet's p99 — find it, then design so it cannot recur · *Code (45m):* Practical: LFU cache variant · *Career (1h):* Applications ×3 · write the STAR story for a production incident — Ordermatic or GymClan · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Delete anything from the resume you would not want to be asked about

**Wed Sep 23** *(6h 30m)* — *Build (2h 30m):* P2-M4: preemption — evict + recompute on resume · *Study (1h):* Read vLLM's preemption path in source · *Design (45m):* Debug drill: cache hit rate collapsed overnight and TTFT doubled — what changed? · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×4 · sweep the inference clouds' career pages directly; ATS aggregators miss half of them · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Add a 'Selected Projects' line to LinkedIn Featured pointing at the strongest repo

**Thu Sep 24** *(6h 30m)* — *Build (2h 30m):* P2-M4: expose a preemption counter in metrics · *Study (1h):* Question bank sections D + F review · *Design (45m):* Variant: multi-region serving with data residency — routing, replication, and what never crosses a border · *Code (45m):* Practical of choice · *Career (1h):* Applications ×4 · pipeline review: what stage is every live conversation in, and what unblocks it? · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Confirm your Purdue email still forwards somewhere you read after graduation

**Fri Sep 25** *(3h 45m)* — *Study (1h):* Question bank section E (speculative decoding) — including the exactness argument · *Outreach (1h):* Outreach ×10 · *Design (45m):* Variant: model cascade — cheap model first, escalate on low confidence. Where does the added latency come from? · *Write-up (45m):* Weekly write-up + commit/push everything · *Small thing (15m):* Screenshot your best benchmark graph and keep it handy — you will want it in a message one day

**Sat Sep 26** *(3h 45m)* — *Study (1h):* Question bank section F (parallelism) — size three deployments live · *Design (45m):* System design drill #5: Scenario 5 — 50M docs/day offline batch, cheapest · *Catch-up (1h 45m):* Build catch-up · *Small thing (15m):* Write down the three numbers you most want an interviewer to remember about you

**Sun Sep 27** *(1h 45m)* — *Self-test (45m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (45m):* Review the week · plan the next one · *Small thing (15m):* Check that your GitHub repos have licences and a sensible default branch name


### Week 6 · Sep 28 → Oct 4

**Mon Sep 28** *(6h 30m)* — *Build (2h 30m):* P2-M5: prefix reuse + LRU block cache (your cluster, one level down) · *Study (1h):* Leviathan speculative decoding paper · *Design (45m):* Variant: zero-downtime model version rollout — shadow, canary, rollback triggers · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×3 · practise the resume walkthrough for your strongest project, five whys deep · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Update the tracker with any application you sent and forgot to log

**Tue Sep 29** *(6h 30m)* — *Build (2h 30m):* P2-M6: /metrics — TTFT & ITL histograms, queue depth, KV utilisation · *Study (1h):* Spec-decode acceptance math: α=0.8, γ=4 → expected tokens/pass · *Design (45m):* Variant: eliminate cold starts for a 70B model — warm pools vs streamed weights vs snapshots, with costs · *Code (45m):* Practical: SSE + backpressure · *Career (1h):* Applications ×4 · one cold message to a hiring manager, not a recruiter — short, specific, no ask beyond a reply · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Reply to the oldest unanswered message in your inbox, whatever it is

**Wed Sep 30** *(6h 30m)* — *Build (2h 30m):* Load test · throughput-vs-p99 curve · mark the knee · *Study (1h):* Toy spec decode: 0.5B drafts for 1.5B — measure acceptance rate · *Design (45m):* Variant: cut serving cost 50% without breaking the SLO — rank every lever by ROI · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×4 · confirm your GitHub pins and LinkedIn featured section match what you are claiming · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Back up your progress: Export from the tracker, and commit the repo

**Thu Oct 1** *(6h 30m)* — *Build (2h 30m):* Publish mini-vllm: README with every graph · 30-min recorded walkthrough · *Study (1h):* Skim EAGLE + Medusa — one-line idea each · *Design (45m):* Variant: multi-tenant isolation — MIG vs shared engine with quotas, and how noisy neighbours show up in metrics · *Code (45m):* Practical of choice · *Career (1h):* Applications ×3 · STAR story: the two-person team at GymClan — mentoring and ownership · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Fix the resume: graduation date → December 2026

**Fri Oct 2** *(3h 45m)* — *Study (1h):* Question bank section G (production debugging) — the hypothesis ladders · *Outreach (1h):* Outreach ×10 + post mini-vllm on LinkedIn · *Design (45m):* Variant: disaggregated prefill and decode — what moves between pools, how big is it, over what link? · *Write-up (45m):* Weekly write-up + commit/push everything · *Small thing (15m):* Rename the resume file to Pranay_Chimmani_Inference_Engineer.pdf — recruiters see the filename

**Sat Oct 3** *(3h 45m)* — *Study (1h):* Skim the vLLM release notes since your version — what changed, and why would it change your benchmarks? · *Design (45m):* System design drill #6: Scenario 6 — RAG assistant, 10-turn, big shared docs · *Catch-up (1h 45m):* Repo polish across all four projects · *Small thing (15m):* Check your resume PDF has selectable text, not an image — copy-paste it into a text editor and see what an ATS sees

**Sun Oct 4** *(1h 45m)* — *Self-test (45m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (45m):* Review the week · plan the next one · *Small thing (15m):* Save a plain-text version of the resume for ATS forms that mangle PDFs


### Week 7 · Oct 5 → Oct 11

**Mon Oct 5** *(6h 30m)* — *Build (2h 30m):* OSS: claim 2–3 good-first-issues in vLLM or SGLang · *Study (1h):* Tensor vs pipeline parallelism — size a 70B deployment out loud · *Design (45m):* Variant: serving 100 LoRA adapters on one base model — adapter cache, routing, eviction storms · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×4 · re-apply or nudge on the three roles you most want that have not replied · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Turn on LinkedIn 'Open to work', recruiters-only — set the titles to inference / ML infra, not 'software engineer'

**Tue Oct 6** *(6h 30m)* — *Build (2h 30m):* OSS: first pull request submitted · *Study (1h):* DistServe + Mooncake — disaggregation and the KV transfer math · *Design (45m):* Variant: a spot-instance batch pipeline — checkpointing, idempotency, what happens on preemption · *Code (45m):* Practical of choice · *Career (1h):* Applications ×4 · read one job description slowly and map every requirement to something you can show · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Add the inference keywords to your LinkedIn Skills section — that is what recruiter search actually queries

**Wed Oct 7** *(6h 30m)* — *Build (2h 30m):* OSS: address review comments · *Study (1h):* Module 8 ops: SLOs, autoscaling signals, cold starts, observability · *Design (45m):* Variant: design the benchmark harness you would trust before signing off a model swap · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×3 · STAR story: diagnosing overfitting in the Purdue×Microsoft fine-tune · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Verify your GitHub commit email so your contributions actually show on your profile

**Thu Oct 8** *(6h 30m)* — *Build (2h 30m):* Second PR, or deepen the first · *Study (1h):* Question bank sections F + G, out loud · *Design (45m):* Rebuild from memory: Scenario 1 (70B, sub-200ms TTFT) — no notes, 35 minutes · *Code (45m):* Practical of choice · *Career (1h):* Applications ×4 · ask a peer to read your resume cold and tell you what the role is · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Add a photo and your real name to GitHub — an anonymous profile reads as a throwaway

**Fri Oct 9** *(3h 45m)* — *Study (1h):* Read one inference-cloud engineering post (Baseten / Modal / Fireworks / Anyscale) and extract one number you didn't know · *Outreach (1h):* Outreach ×10 — mention the merged/open PR · *Design (45m):* Rebuild from memory: Scenario 2 (autoscaling) — no notes, 35 minutes · *Write-up (45m):* Weekly write-up + commit/push everything · *Small thing (15m):* Write a GitHub profile README: one paragraph on what you build, links to the projects

**Sat Oct 10** *(3h 45m)* — *Study (1h):* Re-derive the KV cache size formula and apply it to three models from their config.json · *Design (45m):* System design drill #7: Scenario 1 again — timed, and have someone interrupt you · *Catch-up (1h 45m):* Rebuild one component from memory · *Small thing (15m):* Check the phone number on your resume can receive US calls, and that voicemail is set up and professional

**Sun Oct 11** *(1h 45m)* — *Self-test (45m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (45m):* Review the week · plan the next one · *Small thing (15m):* Make your email signature: name, one-line title, LinkedIn, GitHub. Every outreach message carries it


### Week 8 · Oct 12 → Oct 18

**Mon Oct 12** *(6h 30m)* — *Build (2h 30m):* Extend mini-vllm: chunked prefill, or spec decode wired in · *Study (1h):* Re-read 'Inside vLLM' now that you've built one — it reads differently · *Design (45m):* Rebuild from memory: Scenario 4 (multi-LoRA) — no notes, 35 minutes · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×4 · interview logistics: book, confirm, and prep-block anything on the calendar · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Bookmark the careers pages of your top 10 targets directly — aggregators miss half the postings

**Tue Oct 13** *(6h 30m)* — *Build (2h 30m):* Walkthrough rehearsals: all four projects, 5 minutes each with numbers · *Study (1h):* Module 7: distributed inference recap · *Design (45m):* Rebuild from memory: Scenario 5 (offline batch) — no notes, 35 minutes · *Code (45m):* Practical of choice · *Career (1h):* Applications ×3 · write down your salary floor and target, with the data behind them · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Set a weekly calendar reminder to check those 10 pages

**Wed Oct 14** *(6h 30m)* — *Build (2h 30m):* Fix whatever the rehearsals exposed — usually a missing graph or a shaky number · *Study (1h):* Question bank sections A–C, out loud · *Design (45m):* Rebuild from memory: Scenario 6 (RAG serving) — no notes, 35 minutes · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×4 · send two thank-you notes and one follow-up you have been putting off · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Standardise every date format on the resume — inconsistency reads as carelessness

**Thu Oct 15** *(6h 30m)* — *Build (2h 30m):* Buffer: finish anything unfinished across all four projects · *Study (1h):* Behavioral: draft 5 STAR stories · *Design (45m):* Explain your own cluster as a design interview — requirements, maths, architecture, trade-offs, 20 minutes · *Code (45m):* Practical of choice · *Career (1h):* Applications ×4 · then message 2 Purdue alumni at companies on your tier-2 list · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Create a folder for tailored resume versions so you never overwrite the master

**Fri Oct 16** *(3h 45m)* — *Study (1h):* Trace one request through vLLM's source — scheduler, block manager, model runner · *Outreach (1h):* Outreach ×10 · nudge every live application · *Design (45m):* Explain mini-vLLM as a design interview — and name every divergence from real vLLM · *Write-up (45m):* Weekly write-up + commit/push everything · *Small thing (15m):* Write the two-sentence 'what I'm looking for' blurb you paste into application forms

**Sat Oct 17** *(3h 45m)* — *Study (1h):* Read the SGLang RadixAttention docs and compare it to what your own prefix cache does · *Catch-up (1h 45m):* Rebuild a second component from memory · *Design (45m):* System design drill #8: rapid-fire variants — multi-region, model cascade, capacity planning · *Small thing (15m):* Whitelist recruiter domains and check your spam folder — people have missed interviews this way

**Sun Oct 18** *(1h 45m)* — *Self-test (45m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (45m):* Review the week · plan the next one · *Small thing (15m):* Add your city and work authorisation status to the resume header if it is not already clear


### Week 9 · Oct 19 → Oct 25

**Mon Oct 19** *(6h 30m)* — *Build (2h 30m):* Portfolio polish: 4 consistent READMEs, pinned repos, LinkedIn featured section · *Study (1h):* Re-read your own write-ups as an interviewer would · *Design (45m):* Rapid-fire: five capacity questions in 15 minutes, numbers only · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×4 · follow up with everyone who has gone quiet for 7+ days · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Set your LinkedIn URL and GitHub URL side by side on the resume, both clickable in the PDF

**Tue Oct 20** *(6h 30m)* — *Build (2h 30m):* Write one public post: 'What I learned building an inference stack from scratch' · *Study (1h):* Company research: eng blogs for your live pipeline · *Design (45m):* Rapid-fire: five 'when would you NOT do this?' questions — batching, spec decode, disaggregation, quantisation, caching · *Code (45m):* Practical of choice · *Career (1h):* Applications ×3 · refresh the resume with anything that shipped this week — numbers only, no rewrites · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Test your resume link on a phone — half of recruiters open it there first

**Wed Oct 21** *(6h 30m)* — *Build (2h 30m):* Address anything weak in the portfolio · final graph pass · *Study (1h):* Question bank sections D–F, out loud · *Design (45m):* Interruption practice: pick any scenario, have someone derail you twice, return to the method both times · *Code (45m):* LeetCode ×2 · *Career (1h):* Applications ×4 · find one engineer whose post you actually read and send a specific question · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Check that every link in your resume actually resolves; dead links are silent rejections

**Thu Oct 22** *(6h 30m)* — *Build (2h 30m):* Buffer + rest of the STAR stories · *Study (1h):* Behavioral rehearsal: the pivot story, 90 seconds · *Design (45m):* Learn the 5-step method: requirements → capacity maths → architecture → deep-dive → failure modes. Write it on a card you keep · *Code (45m):* Practical of choice · *Career (1h):* Applications ×4 · tracker hygiene: every row has a status and a next-action date, or it gets closed · *Close (15m):* Log notes · say 2 question-bank answers out loud · *Small thing (15m):* Add one-line descriptions to your pinned GitHub repos with the headline number in them

**Fri Oct 23** *(3h 45m)* — *Study (1h):* Arithmetic hour: prefill vs decode FLOPs and bytes for three model sizes, no notes · *Outreach (1h):* Outreach ×10 · confirm interview slots · *Design (45m):* Capacity drill: how many H100s to serve Llama-3-70B in fp16? Do the arithmetic out loud, then again in FP8 · *Write-up (45m):* Weekly write-up + commit/push everything · *Small thing (15m):* Set up a Calendly or equivalent so scheduling never costs you three emails

**Sat Oct 24** *(3h)* — *Design (45m):* System design drill #9: Scenarios 2 and 5 back to back, no break · *Study (1h):* Book the four mocks (peers / interviewing.io) · clear the revision queue · *Career (1h):* Applications ×3 · write one STAR story in full, then cut it to 90 seconds out loud · *Small thing (15m):* Practise saying your name and a one-sentence intro out loud until it sounds unforced

**Sun Oct 25** *(1h 45m)* — *Self-test (45m):* FULL self-test across every section — Phase-4 entry score · *Plan (45m):* Plan interview mode against your actual interview calendar · *Small thing (15m):* Put your interview calendar in one place and block prep time before each slot, not after


## Phase 4 · Interview mode


### Week 10 · Oct 26 → Nov 1

**Mon Oct 26** *(4h 15m)* — *Drill (1h 30m):* Drill: question bank A–B out loud, rubric-scored · *Design (45m):* Whiteboard: Scenario 1, timed 35 min · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: arrays + hashmap — 2 timed mediums · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Re-read your last five outreach messages — would you reply to them?

**Tue Oct 27** *(4h 15m)* — *Drill (1h 30m):* Drill: sections C–D · *Design (45m):* Whiteboard: Scenario 2 · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: two pointers / sliding window — 2 timed · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Add alt text or captions to the graphs in your project READMEs

**Wed Oct 28** *(5h 30m)* — *Mock (2h):* MOCK #1 — system design round (peer or interviewing.io) · *Study (1h):* Watch the recording back · queue every fumble for revision · *Design (45m):* Capacity drill: concurrent sequences per 80GB card at 8k context on a GQA 8B — derive it from the KV formula · *Code (45m):* Revise: heap and top-k — 2 timed · *Apply (45m):* Applications · nudges · *Small thing (15m):* Check your LinkedIn headline says what you do, not what you want

**Thu Oct 29** *(4h 15m)* — *Drill (1h 30m):* Drill: everything Mock #1 exposed · *Design (45m):* Whiteboard: Scenario 3 (debugging) · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: intervals — merge, insert, minimum rooms · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Make sure your resume is one consistent font and the margins are not cheating for space

**Fri Oct 30** *(3h 45m)* — *Study (1h):* Watch a recorded inference talk (GTC / Ray Summit / vLLM meetup) and write five bullets · *Outreach (1h):* Outreach ×10 · *Redo (45m):* Redo missed problems · *Write-up (45m):* Write up the mock lessons · *Small thing (15m):* Delete anything from the resume you would not want to be asked about

**Sat Oct 31** *(3h 45m)* — *Catch-up (1h 45m):* Project walkthrough rehearsal: Project 3, five-whys deep · *Study (1h):* Revision queue: clear everything due · *Design (45m):* SLO drill: write numeric TTFT / ITL / availability targets for chat, a coding copilot, and batch summarisation — defend each · *Small thing (15m):* Add a 'Selected Projects' line to LinkedIn Featured pointing at the strongest repo

**Sun Nov 1** *(1h 45m)* — *Self-test (45m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (45m):* Review the week · plan the next one · *Small thing (15m):* Confirm your Purdue email still forwards somewhere you read after graduation


### Week 11 · Nov 2 → Nov 8

**Mon Nov 2** *(4h 15m)* — *Drill (1h 30m):* Drill: sections E–F · *Design (45m):* Whiteboard: Scenario 4 (multi-LoRA) · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: binary search, including search-on-answer · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Screenshot your best benchmark graph and keep it handy — you will want it in a message one day

**Tue Nov 3** *(4h 15m)* — *Drill (1h 30m):* Drill: sections G–H · *Design (45m):* Whiteboard: Scenario 5 (offline batch) · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: BFS / DFS on grids · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Write down the three numbers you most want an interviewer to remember about you

**Wed Nov 4** *(5h 30m)* — *Mock (2h):* MOCK #2 — inference deep-dive; have them grill both rebuilt projects · *Study (1h):* Recording review · queue the gaps · *Design (45m):* Component: draw the request lifecycle from HTTP ingress to last token, and name every stage you would instrument · *Code (45m):* Revise: graphs — topological sort and union-find · *Apply (45m):* Applications · nudges · *Small thing (15m):* Check that your GitHub repos have licences and a sensible default branch name

**Thu Nov 5** *(4h 15m)* — *Drill (1h 30m):* Drill: resume walkthroughs — 5 minutes each, with numbers · *Design (45m):* Whiteboard: Scenario 6 (RAG serving) · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: trees — DFS, level order, lowest common ancestor · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Update the tracker with any application you sent and forgot to log

**Fri Nov 6** *(3h 45m)* — *Study (1h):* Re-read the PagedAttention paper's evaluation section — what baselines, what wins, what's hidden · *Outreach (1h):* Outreach ×10 · *Design (45m):* Component: static vs continuous batching — draw both schedulers and explain the throughput delta with numbers · *Write-up (45m):* Weekly write-up + commit/push everything · *Small thing (15m):* Reply to the oldest unanswered message in your inbox, whatever it is

**Sat Nov 7** *(3h 45m)* — *Catch-up (1h 45m):* Behavioral: write 3 STAR stories in full, then cut each to 90 seconds · *Study (1h):* Revision queue: clear everything due · *Design (45m):* Component: KV memory budgeting — given a VRAM budget, how many concurrent requests, and what do you evict first? · *Small thing (15m):* Back up your progress: Export from the tracker, and commit the repo

**Sun Nov 8** *(1h 45m)* — *Self-test (45m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (45m):* Review the week · plan the next one · *Small thing (15m):* Fix the resume: graduation date → December 2026


### Week 12 · Nov 9 → Nov 15

**Mon Nov 9** *(4h 15m)* — *Drill (1h 30m):* Drill: full sweep A–D · *Design (45m):* Whiteboard: your lowest-scoring scenario · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: monotonic stack and expression parsing · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Rename the resume file to Pranay_Chimmani_Inference_Engineer.pdf — recruiters see the filename

**Tue Nov 10** *(4h 15m)* — *Drill (1h 30m):* Drill: full sweep E–H · *Design (45m):* Whiteboard: capacity planning until it's 3 minutes flat · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: linked lists — reverse, detect cycle, merge k · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Check your resume PDF has selectable text, not an image — copy-paste it into a text editor and see what an ATS sees

**Wed Nov 11** *(5h 30m)* — *Mock (2h):* MOCK #3 — coding round + applied ML-systems problem · *Study (1h):* Recording review · queue the gaps · *Design (45m):* Component: design the streaming API — SSE vs WebSocket, heartbeats, cancellation, what happens on disconnect · *Code (45m):* Revise: 1-D DP — climbing, house robber, coin change · *Apply (45m):* Applications · nudges · *Small thing (15m):* Save a plain-text version of the resume for ATS forms that mangle PDFs

**Thu Nov 12** *(4h 15m)* — *Drill (1h 30m):* Drill: the Mock #3 fix list · *Design (45m):* Whiteboard: interviewer-interruption practice · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: 2-D DP — grid paths, edit distance, LCS · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Turn on LinkedIn 'Open to work', recruiters-only — set the titles to inference / ML infra, not 'software engineer'

**Fri Nov 13** *(3h 45m)* — *Study (1h):* Read about FP8 serving on Hopper and decide whether it would change your quantisation recommendation · *Outreach (1h):* Outreach ×10 · *Design (45m):* Component: prefix-cache-aware routing — how do you keep a session on the pod already holding its KV? · *Write-up (45m):* Weekly write-up + commit/push everything · *Small thing (15m):* Add the inference keywords to your LinkedIn Skills section — that is what recruiter search actually queries

**Sat Nov 14** *(3h 45m)* — *Catch-up (1h 45m):* Rebuild a component from memory · *Study (1h):* Revision queue: clear everything due · *Design (45m):* Component: admission control and load shedding — where exactly does the 429 come from, and at what queue depth? · *Small thing (15m):* Verify your GitHub commit email so your contributions actually show on your profile

**Sun Nov 15** *(1h 45m)* — *Self-test (45m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (45m):* Review the week · plan the next one · *Small thing (15m):* Add a photo and your real name to GitHub — an anonymous profile reads as a throwaway


### Week 13 · Nov 16 → Nov 22

**Mon Nov 16** *(4h 15m)* — *Drill (1h 30m):* Drill: weakest 15 questions · *Design (45m):* Whiteboard: Scenario 1 + 2 back to back · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: greedy — scheduling and jump game · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Write a GitHub profile README: one paragraph on what you build, links to the projects

**Tue Nov 17** *(4h 15m)* — *Drill (1h 30m):* Drill: numbers only — every calculation in the bank · *Design (45m):* Whiteboard: Scenario 3 + 4 · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: strings — anagrams, palindromes, tokenising · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Check the phone number on your resume can receive US calls, and that voicemail is set up and professional

**Wed Nov 18** *(5h 30m)* — *Mock (2h):* MOCK #4 — full loop simulation: coding + design + deep-dive · *Study (1h):* Recording review · final fix list · *Design (45m):* Component: chunked prefill — draw the batch timeline with and without it, mark where ITL spikes · *Code (45m):* Revise: prefix sums and counting · *Apply (45m):* Applications · nudges · *Small thing (15m):* Make your email signature: name, one-line title, LinkedIn, GitHub. Every outreach message carries it

**Thu Nov 19** *(4h 15m)* — *Drill (1h 30m):* Drill: the full-loop fix list · *Design (45m):* Whiteboard: Scenario 5 + 6 · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: matrix — rotate, spiral, search a sorted matrix · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Bookmark the careers pages of your top 10 targets directly — aggregators miss half the postings

**Fri Nov 20** *(3h 45m)* — *Study (1h):* Compare your mini-vLLM scheduler to vLLM's line by line and list every divergence · *Outreach (1h):* Outreach ×10 · *Design (45m):* Component: the metrics endpoint — which numbers, at which percentiles, and which one is your north star? · *Write-up (45m):* Weekly write-up + commit/push everything · *Small thing (15m):* Set a weekly calendar reminder to check those 10 pages

**Sat Nov 21** *(3h 45m)* — *Catch-up (1h 45m):* Company research: eng blogs of your live interview pipeline · *Study (1h):* Revision queue: clear due · *Design (45m):* Capacity drill: 1,000 req/s at 1k prompt / 300 output — how many replicas, and what is your headroom policy? · *Small thing (15m):* Standardise every date format on the resume — inconsistency reads as carelessness

**Sun Nov 22** *(1h 45m)* — *Self-test (45m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (45m):* Review the week · plan the next one · *Small thing (15m):* Create a folder for tailored resume versions so you never overwrite the master


### Week 14 · Nov 23 → Nov 29

**Mon Nov 23** *(4h 15m)* — *Drill (1h 30m):* Drill: 10 random questions · *Design (45m):* Whiteboard: pick your weakest scenario · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: backtracking — subsets, permutations, word search · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Write the two-sentence 'what I'm looking for' blurb you paste into application forms

**Tue Nov 24** *(4h 15m)* — *Drill (1h 30m):* Drill: 10 random questions · *Design (45m):* Whiteboard: one scenario, relaxed · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: design questions — LRU, min-stack, hit counter · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Whitelist recruiter domains and check your spam folder — people have missed interviews this way

**Wed Nov 25** *(3h 30m)* — *Study (1h):* Light: revision queue only · *Apply (45m):* Applications — December postings are live · *Design (45m):* Component: speculative decoding — when do you turn it ON in production, and what tells you to turn it off? · *Code (45m):* Revise: custom comparators and sort-based problems · *Small thing (15m):* Add your city and work authorisation status to the resume header if it is not already clear

**Thu Nov 26** *(4h 30m)* — *Study (1h):* Read one production post-mortem from any infra company — map its failure mode onto LLM serving · *Plan (45m):* Off. Actually off. · *Design (45m):* Component: quantisation choice — pick a scheme for a latency-critical chat product and defend it to a sceptic · *Code (45m):* Revise: your two weakest patterns, from the log · *Career (1h):* Applications ×4 · research two companies in your pipeline: read an eng blog post each, note one question to ask them · *Small thing (15m):* Set your LinkedIn URL and GitHub URL side by side on the resume, both clickable in the PDF

**Fri Nov 27** *(3h 45m)* — *Study (1h):* Light: revision queue only · *Design (45m):* Debug drill: p99 tripled, p50 flat — run it as an incident, out loud, hypothesis ladder first · *Code (45m):* Revise: arrays + hashmap — 2 timed mediums · *Career (1h):* Applications ×4 · LinkedIn: one post or one comment on someone's inference work. Visibility compounds · *Small thing (15m):* Test your resume link on a phone — half of recruiters open it there first

**Sat Nov 28** *(3h 30m)* — *Drill (1h 30m):* One design drill, no pressure · *Design (45m):* Debug drill: TTFT is fine but ITL is terrible — what is your first graph, and your first three hypotheses? · *Career (1h):* Applications ×3 · re-read your resume as a hostile screener and fix the weakest line · *Small thing (15m):* Check that every link in your resume actually resolves; dead links are silent rejections

**Sun Nov 29** *(1h 45m)* — *Self-test (45m):* Self-test: 5 Qs · *Plan (45m):* Plan the final two weeks against your interview calendar · *Small thing (15m):* Add one-line descriptions to your pinned GitHub repos with the headline number in them


### Week 15 · Nov 30 → Dec 6

**Mon Nov 30** *(4h 15m)* — *Drill (1h 30m):* Drill: everything still in the revision queue · *Design (45m):* Whiteboard: your two strongest scenarios, polished · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: two pointers / sliding window — 2 timed · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Set up a Calendly or equivalent so scheduling never costs you three emails

**Tue Dec 1** *(4h 15m)* — *Drill (1h 30m):* Drill: rapid-fire — 20 questions in 40 minutes · *Design (45m):* Whiteboard: the one you avoid · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: heap and top-k — 2 timed · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Practise saying your name and a one-sentence intro out loud until it sounds unforced

**Wed Dec 2** *(5h)* — *Drill (1h 30m):* Interviews / targeted revision · *Study (1h):* Company research for the live pipeline · *Design (45m):* Debug drill: throughput halved right after a deploy — walk the diff and the metrics together · *Code (45m):* Revise: intervals — merge, insert, minimum rooms · *Apply (45m):* Follow-ups · thank-you notes · *Small thing (15m):* Put your interview calendar in one place and block prep time before each slot, not after

**Thu Dec 3** *(4h 15m)* — *Drill (1h 30m):* Drill: project walkthroughs, both rebuilds, cold · *Design (45m):* Whiteboard: free choice · *Apply (45m):* Applications · nudges · thank-you notes · *Code (45m):* Revise: binary search, including search-on-answer · *Close (15m):* Queue your weakest answer for revision · *Small thing (15m):* Re-read your last five outreach messages — would you reply to them?

**Fri Dec 4** *(3h 45m)* — *Study (1h):* Review every number currently on your resume and reproduce two of them from scratch · *Outreach (1h):* Outreach ×10 — December postings exist; keep applying · *Design (45m):* Debug drill: one tenant's traffic destroyed the fleet's p99 — find it, then design so it cannot recur · *Write-up (45m):* Weekly write-up + commit/push everything · *Small thing (15m):* Add alt text or captions to the graphs in your project READMEs

**Sat Dec 5** *(3h 45m)* — *Catch-up (1h 45m):* Behavioral stories to 90 seconds each, out loud · *Study (1h):* Revision queue: clear everything · *Design (45m):* Debug drill: cache hit rate collapsed overnight and TTFT doubled — what changed? · *Small thing (15m):* Check your LinkedIn headline says what you do, not what you want

**Sun Dec 6** *(1h 45m)* — *Self-test (45m):* Self-test: 5 random question-bank Qs out loud, scored · *Plan (45m):* Review the week · plan the next one · *Small thing (15m):* Make sure your resume is one consistent font and the margins are not cheating for space


### Week 16 · Dec 7 → Dec 13

**Mon Dec 7** *(5h)* — *Drill (1h 30m):* Interview prep for what's on the calendar · targeted revision only · *Study (1h):* Day-before ritual: their eng blog + your walkthroughs. Nothing new. · *Design (45m):* Variant: multi-region serving with data residency — routing, replication, and what never crosses a border · *Code (45m):* Revise: BFS / DFS on grids · *Apply (45m):* Follow-ups · thank-you notes · *Small thing (15m):* Delete anything from the resume you would not want to be asked about

**Tue Dec 8** *(5h)* — *Drill (1h 30m):* Interviews / targeted revision · *Study (1h):* Company research · *Design (45m):* Variant: model cascade — cheap model first, escalate on low confidence. Where does the added latency come from? · *Code (45m):* Revise: graphs — topological sort and union-find · *Apply (45m):* Follow-ups · *Small thing (15m):* Add a 'Selected Projects' line to LinkedIn Featured pointing at the strongest repo

**Wed Dec 9** *(5h)* — *Drill (1h 30m):* Interviews / targeted revision · *Study (1h):* Company research · *Design (45m):* Variant: zero-downtime model version rollout — shadow, canary, rollback triggers · *Code (45m):* Revise: trees — DFS, level order, lowest common ancestor · *Apply (45m):* Follow-ups · *Small thing (15m):* Confirm your Purdue email still forwards somewhere you read after graduation

**Thu Dec 10** *(5h)* — *Drill (1h 30m):* Interviews / targeted revision · *Study (1h):* Company research · *Design (45m):* Variant: eliminate cold starts for a 70B model — warm pools vs streamed weights vs snapshots, with costs · *Code (45m):* Revise: monotonic stack and expression parsing · *Apply (45m):* Follow-ups · *Small thing (15m):* Screenshot your best benchmark graph and keep it handy — you will want it in a message one day

**Fri Dec 11** *(3h 45m)* — *Study (1h):* Re-read your own KV cache code and explain every line out loud — the parts you skim are the parts you don't know · *Outreach (1h):* Outreach + follow-ups · *Redo (45m):* Light revision of weak spots · *Write-up (45m):* Negotiate with data if offers land · *Small thing (15m):* Write down the three numbers you most want an interviewer to remember about you

**Sat Dec 12** *(4h)* — *Drill (1h 30m):* Final polish: whichever scenario is next in your loop · *Plan (45m):* Rest. You did the work. · *Design (45m):* Variant: cut serving cost 50% without breaking the SLO — rank every lever by ROI · *Career (1h):* Applications ×4 · ask one alum for a referral, explicitly and specifically — name the req

**Sun Dec 13** *(2h)* — *Study (1h):* Question bank section A — answer out loud, score yourself, queue anything under 3 · *Plan (45m):* Review the whole run · write down what you learned · graduate 🎓 · *Small thing (15m):* Check that your GitHub repos have licences and a sensible default branch name


## Phase 5 · Close the offer


### Week 17 · Dec 14 → Dec 20

**Mon Dec 14** *(5h 45m)* — *Study (1h):* Question bank section B (KV cache) — redo every calculation on paper · *Career (2h):* Interview day / prep for what's on the calendar — the loop is the job now · *Career (1h):* Apply to January-start reqs — they post now and decide in January · *Code (45m):* Revise: linked lists — reverse, detect cycle, merge k · *Design (45m):* Rehearse the design scenario your next loop is most likely to ask · *Small thing (15m):* Update the tracker with any application you sent and forgot to log

**Tue Dec 15** *(6h 15m)* — *Career (2h):* Interviews · same-day thank-you notes naming something specific they said · *Drill (1h 30m):* Drill whatever the last loop exposed — within 24 hours, while it stings · *Design (45m):* Variant: multi-tenant isolation — MIG vs shared engine with quotas, and how noisy neighbours show up in metrics · *Code (45m):* Revise: 1-D DP — climbing, house robber, coin change · *Career (1h):* 5 applications · nudge every silent thread · *Small thing (15m):* Reply to the oldest unanswered message in your inbox, whatever it is

**Wed Dec 16** *(6h 30m)* — *Career (2h):* Interviews · follow-ups · *Career (1h 30m):* Negotiation prep: know your number, your floor, and the market band for new-grad inference roles · *Design (45m):* Variant: disaggregated prefill and decode — what moves between pools, how big is it, over what link? · *Code (45m):* Revise: 2-D DP — grid paths, edit distance, LCS · *Drill (1h 30m):* Project walkthroughs, cold, both rebuilds

**Thu Dec 17** *(5h 45m)* — *Study (1h):* Question bank section C (batching and scheduling) — out loud, timed · *Career (2h):* Interviews · follow-ups · *Career (1h):* Ask every live company for a decision timeline — you need them comparable, not sequential · *Code (45m):* Revise: greedy — scheduling and jump game · *Design (45m):* One design drill, timed · *Small thing (15m):* Back up your progress: Export from the tracker, and commit the repo

**Fri Dec 18** *(4h)* — *Study (1h):* Question bank section D (quantisation) — answer from YOUR measured numbers · *Outreach (1h):* Outreach ×10 — last push before the holiday freeze · *Career (1h):* Chase every application older than two weeks with a short, specific nudge · *Write-up (45m):* Write up the year: what you built, in one post · *Small thing (15m):* Fix the resume: graduation date → December 2026

**Sat Dec 19** *(4h)* — *Career (1h 30m):* Tidy the pipeline: every row in the tracker has a status and a next action with a date · *Drill (1h 30m):* Weak-spot drill of your choosing · *Design (45m):* Variant: serving 100 LoRA adapters on one base model — adapter cache, routing, eviction storms · *Small thing (15m):* Rename the resume file to Pranay_Chimmani_Inference_Engineer.pdf — recruiters see the filename

**Sun Dec 20** *(1h 45m)* — *Self-test (45m):* Self-test: 10 questions, scored — where are you actually? · *Plan (45m):* Plan the quiet week honestly: what gets done, what gets rested · *Small thing (15m):* Check your resume PDF has selectable text, not an image — copy-paste it into a text editor and see what an ATS sees


### Week 18 · Dec 21 → Dec 27

**Mon Dec 21** *(4h 15m)* — *Career (1h):* Any live interviews · otherwise: rest. Hiring is frozen this week and pushing does nothing · *Drill (1h 30m):* Light drill — keep the edge, don't grind · *Design (45m):* Variant: a spot-instance batch pipeline — checkpointing, idempotency, what happens on preemption · *Code (45m):* Revise: strings — anagrams, palindromes, tokenising · *Small thing (15m):* Save a plain-text version of the resume for ATS forms that mangle PDFs

**Tue Dec 22** *(3h 45m)* — *Career (1h):* Follow-ups only · keep threads warm with something useful, not a check-in · *Study (1h):* Read something in the field you actually want to read · *Design (45m):* Variant: design the benchmark harness you would trust before signing off a model swap · *Code (45m):* Revise: prefix sums and counting · *Small thing (15m):* Turn on LinkedIn 'Open to work', recruiters-only — set the titles to inference / ML infra, not 'software engineer'

**Wed Dec 23** *(3h 45m)* — *Small thing (15m):* Back up everything: export tracker progress, push all repos · *Study (1h):* Rest or read. Your choice, no guilt either way · *Design (45m):* Rebuild from memory: Scenario 1 (70B, sub-200ms TTFT) — no notes, 35 minutes · *Code (45m):* Revise: matrix — rotate, spiral, search a sorted matrix · *Career (1h):* Applications ×4 · check which of your applications went cold and why; adjust the targeting

**Thu Dec 24** *(4h 30m)* — *Study (1h):* Question bank section E (speculative decoding) — including the exactness argument · *Plan (45m):* Off. · *Design (45m):* Rebuild from memory: Scenario 2 (autoscaling) — no notes, 35 minutes · *Code (45m):* Revise: backtracking — subsets, permutations, word search · *Career (1h):* Applications ×3 · rehearse the 'why inference, why now' answer until it is 60 seconds and true · *Small thing (15m):* Add the inference keywords to your LinkedIn Skills section — that is what recruiter search actually queries

**Fri Dec 25** *(3h 30m)* — *Study (1h):* Question bank section F (parallelism) — size three deployments live · *Plan (45m):* Off. · *Design (45m):* Rebuild from memory: Scenario 4 (multi-LoRA) — no notes, 35 minutes · *Code (45m):* Revise: design questions — LRU, min-stack, hit counter · *Small thing (15m):* Verify your GitHub commit email so your contributions actually show on your profile

**Sat Dec 26** *(3h)* — *Study (1h):* Optional: one paper you've been meaning to read · *Design (45m):* Rebuild from memory: Scenario 5 (offline batch) — no notes, 35 minutes · *Career (1h):* Applications ×4 · update the README of whichever project you touched most recently · *Small thing (15m):* Add a photo and your real name to GitHub — an anonymous profile reads as a throwaway

**Sun Dec 27** *(2h 30m)* — *Study (50m):* Question bank section G (production debugging) — the hypothesis ladders · *Plan (50m):* Plan the January restart: reqs reopen, recruiters return, decisions land the first two weeks · *Career (50m):* Draft the January outreach batch now so it sends on the 2nd


### Week 19 · Dec 28 → Dec 31

**Mon Dec 28** *(4h 15m)* — *Career (1h):* Applications restart: 5 today — January hiring cycles open before people return · *Drill (1h 30m):* Back to drilling: sections you scored lowest on · *Design (45m):* Design drill — rebuild your strongest scenario from memory · *Code (45m):* Revise: custom comparators and sort-based problems · *Small thing (15m):* Write a GitHub profile README: one paragraph on what you build, links to the projects

**Tue Dec 29** *(4h 15m)* — *Career (1h):* 5 applications · re-engage every company that went quiet in December · *Drill (1h 30m):* Drill · question bank, out loud · *Design (45m):* Design drill · *Code (45m):* Revise: your two weakest patterns, from the log · *Small thing (15m):* Check the phone number on your resume can receive US calls, and that voicemail is set up and professional

**Wed Dec 30** *(4h 15m)* — *Career (1h):* 5 applications · confirm January interview slots · *Drill (1h 30m):* Drill · project walkthroughs · *Design (45m):* Rebuild from memory: Scenario 6 (RAG serving) — no notes, 35 minutes · *Code (45m):* Revise: arrays + hashmap — 2 timed mediums · *Small thing (15m):* Update the resume with anything shipped in December

**Thu Dec 31** *(5h 45m)* — *Study (1h):* Skim the vLLM release notes since your version — what changed, and why would it change your benchmarks? · *Plan (1h):* Honest review: where is the pipeline, what worked, what didn't · *Design (45m):* Explain your own cluster as a design interview — requirements, maths, architecture, trade-offs, 20 minutes · *Code (45m):* Revise: two pointers / sliding window — 2 timed · *Career (1h):* Applications ×4 · two referral messages · thank anyone who replied this week · *Plan (1h):* Set the January target with dates — offer or no offer, the work continues on the 2nd · *Small thing (15m):* Make your email signature: name, one-line title, LinkedIn, GitHub. Every outreach message carries it


---

## Tracking

The tracker does the counting: XP, level, streak, days complete, applications, revision queue, and a consistency heatmap. Rules that matter more than the numbers: a day counts toward the streak at 60% — a day where life happens but the build slot got done still keeps the chain. Miss one day, fine. Never miss two.
