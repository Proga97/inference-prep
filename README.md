# Inference Prep

A personal study tracker for a 16-week run at LLM inference and serving engineering
(Aug 20 – Dec 13, 2026).

`index.html` is a single self-contained page — no build step, no dependencies, no network
calls. Open it directly, or serve it with GitHub Pages.

- **116 days** of tasks across four tracks: inference projects, system design, coding
  revision, and the job search.
- Every task expands (▸) into how to do it, the resource to use, and a done-when check.
- Eleven reference documents are embedded in the page and readable from the **Files** panel:
  roadmap, study guide, question bank, system-design playbook, project specs, daily plan,
  and the five walkthroughs below. Doc names mentioned inside a doc are clickable.
- A spaced-repetition queue (2 → 7 → 21 days) for anything that needs a second look.

## Walkthroughs

Step-by-step companions to the specs and the study guide — concepts from zero, the real
code to read first, fully commented code, tests, expected numbers, and failure tables:

- `06-p1-walkthrough.md` — Project 1, `inference-from-scratch` (model, parity, naive loop, KV cache, sampling, bench)
- `07-p2-walkthrough.md` — Project 2, `mini-vllm` (SSE server, continuous batching, paged KV, preemption, prefix cache, metrics)
- `08-p3-walkthrough.md` — Project 3, `distributed-kv-cache` rebuild (kind, workers, consistent-hash routing, gateway, LRU, benchmarks, KEDA, GKE)
- `09-p4-walkthrough.md` — Project 4, `quantization-tradeoffs` (llama.cpp build, GGUF/K-quants, llama-bench, eval, tradeoff table)
- `10-study-walkthroughs.md` — every exercise in study-guide Modules 1–11, worked

## Progress

Progress is kept in the browser's `localStorage`, keyed to the origin — so redeploying the
page never touches it. It does not sync across devices or browsers; use the **Export** and
**Import** buttons to move it deliberately.

## Regenerating

The page is generated from a plan definition plus per-task briefs. Editing the plan and
rebuilding is safe: task identity is stable, so reordering or rewording tasks does not
disturb saved progress.
