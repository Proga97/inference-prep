# Project 3 — `distributed-kv-cache`: the solo rebuild, milestone by milestone

This is the project that fixes the resume bullet ("Distributed Inference KV Cache on GKE/Terraform, TTFT −37%"). You built the team version; you cannot currently defend every number in it. By the end of this document you will have a version where every number is yours, every component is under 300 lines, and every graph regenerates from one script.

Work top to bottom. Every milestone ends in a test that prints a number or a table. Do not move on until it does.

**How to read this document.** It assumes you are a strong backend engineer (Python, FastAPI, Kubernetes, GCP) who has *not* worked inside an LLM inference engine before. So:

- The section **"Concepts you need before starting"** defines, from scratch, every term this document relies on. Read it once now, then come back to it whenever a word in a milestone feels fuzzy. Inside each milestone, every new term is also defined in one sentence the first time it appears.
- Every milestone has a **"Look first"** step before its code. That step tells you exactly what to curl, print, open, or `kubectl explain` in the *real* system (llama-server, Starlette, a real hash-ring library, the Kubernetes API) and what to notice. Do it before writing your version — the code you write afterwards will make sense because you have seen the thing it wraps.
- Every code block is commented densely enough to read cold in three months. The comments say what each line does *and why it is there*. Do not strip them when you copy the code into the repo; they are the README you will not have time to write later.

Cross-references: the KV-cache and prefill material lives in Module 3 of `01-study-guide.md`; the project spec is in `04-project-specs.md`; the engine you may later swap in is Project 2 (`07-p2-walkthrough.md`); the profiling and metrics vocabulary (Prometheus, histograms) is Module 8 of the study guide and Project 4 (`09-p4-walkthrough.md`).

---

## Intro — what you are building and why it looks like this

### The architecture

```
client ──SSE──▶ gateway ──/route──▶ coordinator (hash ring + worker stats)
                  │                         │ polls /stats
                  │  /generate (SSE)        ▼
                  └────────────────▶ worker-0 ┐ each: FastAPI wrapper + llama-server
                                     worker-1 ├─ (its own KV cache, N slots)
                                     worker-2 ┘
```

Three services, one job each:

- **gateway** — the only thing a client talks to. Terminates the client connection, rate-limits per client, asks the coordinator where the request should go, streams the answer back as SSE (Server-Sent Events — a plain HTTP response that stays open and delivers many small `data:` lines over time, defined below), measures TTFT (time to first token), writes one structured log line per request.
- **coordinator** — owns the routing decision. Keeps a consistent-hash ring of live workers (defined below: a way of mapping a key to a server so that adding or removing a server moves as few keys as possible) and a fresh-ish copy of each worker's load. Answers "which worker for this prefix?" in microseconds. Stateless except for the ring, which it rebuilds from DNS.
- **worker** — one inference engine (llama.cpp's `llama-server`) plus a thin FastAPI wrapper that proxies to it, decides which engine *slot* a request uses (a slot is one of the engine's parallel request lanes, each with its own cache), enforces a KV budget with LRU eviction (least-recently-used — throw away the entry that was touched longest ago), and reports `/stats`.

### What "KV-cache-aware routing" means, concretely

When a transformer processes a prompt, the expensive part (**prefill** — running the whole prompt through the model in one pass before any output token exists) produces a K and V tensor per layer for every prompt token — the **KV cache** (Module 3 of `01-study-guide.md`, and defined again in the concepts section below). If a second request begins with the *same tokens*, the KV for that shared prefix is identical and can be reused; the engine only has to prefill the tokens after the divergence point. vLLM calls this prefix caching; llama.cpp calls it prompt caching. Either way it only works if the second request lands on the *same engine process* that computed the prefix — KV lives in one process's memory.

That is the whole idea of this project: hash the prefix, route every request with that prefix to the same worker, and the shared part of prefill is skipped. Prefill is what you wait for before the first token, so skipping most of it lowers **TTFT** (time to first token). With a 500-token system prompt and a 40-token user turn, a cache hit prefills 40 tokens instead of 540 — roughly 13× less prefill work. Random routing throws that away most of the time, which is the "affinity OFF" baseline you will measure against.

The engine is llama.cpp's `llama-server` with `Qwen2.5-0.5B-Instruct` in Q8_0 (~530 MB). (Q8_0 is a **quantization** format: each weight stored as an 8-bit integer plus a per-block scale instead of a 16-bit float, halving the file and the memory traffic; "0.5B" means about half a billion weights.) Reasons: it runs on CPU at a usable speed, it has prompt/KV caching built in (`cache_prompt`, per-slot caches, slot save/restore to disk), it reports how many prompt tokens were actually reused in every response (so hit/miss is *measured*, not inferred), and it speaks SSE. The alternative is to put your own `mini-vllm` (Project 2, `07-p2-walkthrough.md`) behind the same wrapper later — the wrapper's interface (`/generate`, `/stats`) is engine-agnostic on purpose, and swapping engines is a one-afternoon README section once Project 2 exists. Do not build both at once.

CPU workers are fine. The system under test is the routing and caching layer; a 0.5B model on CPU makes prefill *slow* (seconds for 500 tokens), which makes the effect of skipping it *large and easy to see*. Section M8 covers what changes on a real GPU.

### Repo layout

```
distributed-kv-cache/
├── gateway/        app.py, Dockerfile
├── coordinator/    app.py, ring.py, Dockerfile, tests/test_ring.py
├── worker/         app.py, Dockerfile, Dockerfile.llama
├── k8s/            kind-config.yaml, worker.yaml, coordinator.yaml, gateway.yaml, keda.yaml
├── bench/          loadgen.py, summarize.py, plot.py, run.sh, results/
├── scripts/        kind-up.sh, chaos.sh
├── terraform/      (M8, optional) main.tf, variables.tf
└── README.md
```

### Two-week scoping

M1–M6 first (ship by Fri Sep 4 — the benchmark is what the resume needs). M7–M8 in week 3 harden a project whose numbers are already published; re-run the benchmark afterwards and update the bullet only if it changed.

### Windows / WSL2 setup (once)

Docker Desktop with the WSL2 backend; do everything from an Ubuntu WSL2 shell. Keep the repo under `~/` in the Linux filesystem, not `/mnt/c/...` — bind-mount I/O across the boundary is 5–10× slower and `kind load` will crawl. Give WSL enough resources in `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=12GB       ; RAM ceiling for the whole WSL VM. 3 workers × ~1 GB model+KV, plus kind's
                  ; own containers, plus Docker builds: 12 GB leaves headroom; 8 GB is the floor.
processors=8      ; CPU cores visible inside WSL. Three llama-servers at -t 2 each = 6 threads,
                  ; plus the wrapper/gateway/coordinator/kind control plane on the remaining 2.
```

Then `wsl --shutdown` from PowerShell and reopen. Install `kind`, `kubectl`, `helm` inside WSL (standard Linux instructions). Everything below runs unchanged on any Linux box.

One thing that bites on a single host: three llama-servers share the same physical cores. Give each worker a fixed thread count (`-t 2` on an 8-core machine) and matching CPU limits, or they will oversubscribe (more runnable threads than cores, so the OS time-slices them and every one runs slower and less predictably) and every benchmark number will be noise.

---

## Concepts you need before starting

Read this once, top to bottom. Each entry is: the plain-words definition, an analogy where one helps, and why this project cares. Nothing here assumes you have seen the term before.

### Tokens and the tokenizer

A **token** is the unit a language model reads and writes — usually a word fragment of 3–5 characters ("assist", "ant", " the"). The **tokenizer** is a deterministic function from text to a list of integer token ids and back. *Analogy:* a model reads a sentence the way you'd read a sentence cut into syllable cards. Why you care: the KV cache is indexed per token, prefill cost is per token, and "500-token system prompt" is what makes the numbers in this project.

### Prefill and decode

Generating text has two phases.

- **Prefill**: the model reads the entire prompt in one big parallel pass. Every prompt token is processed at once, so it is compute-heavy and its cost grows with prompt length. Nothing is shown to the user during prefill.
- **Decode**: the model produces output tokens one at a time. Each step reads all the weights once to produce one token, so it is limited by how fast memory can be read (see "bound" below), not by arithmetic.

*Analogy:* prefill is reading the whole question before answering; decode is speaking the answer one word at a time. Why you care: TTFT is basically prefill time plus queueing; the entire project exists to skip repeated prefill.

### KV cache

Inside every transformer layer, attention computes for each token a **key** vector K and a **value** vector V. Later tokens attend to earlier ones by looking at those stored K and V. Rather than recompute them every step, the engine keeps them in memory: that store is the **KV cache**. Its shape for one sequence is

```
K: [n_layers, seq_len, n_kv_heads, head_dim]     # one key vector per layer per token per KV head
V: [n_layers, seq_len, n_kv_heads, head_dim]     # same for values
```

so its size in bytes is `2 (K and V) × n_layers × seq_len × n_kv_heads × head_dim × bytes_per_element`. For Qwen2.5-0.5B (24 layers, 2 KV heads, head_dim 64, fp16 = 2 bytes) that is `2 × 24 × 2 × 64 × 2 = 12,288 bytes ≈ 12 KB per token`. For Llama-3-8B (32 layers, 8 KV heads, head_dim 128) it is 128 KB per token. *Analogy:* the notes you keep while reading a long document so you don't re-read the start every time you turn a page. Why you care: the cache lives in one process's RAM. It is *not* shared across workers, which is why routing decides whether it is reused.

### Why a cached prefix skips prefill (prefix / prompt caching)

The K and V for token *i* depend only on tokens `0..i` (causal attention: a token never looks forward). So if two prompts share their first *P* tokens, the first *P* rows of the KV cache are byte-identical. An engine that still has those rows in a slot can skip straight to token *P* and prefill only the remainder. llama-server does this per slot: on a new request it finds the longest common prefix (at the *token* level) between the new prompt and what the slot already holds, keeps that, and prefills the rest. vLLM does the same at block granularity and calls it *prefix caching*. Why you care: `timings.prompt_n` in llama-server's response is "how many tokens I actually had to prefill"; `tokens_evaluated` is "how many tokens were in the prompt". The difference is the reuse — that is how this project *measures* hits instead of guessing.

### TTFT, ITL, latency vs throughput

- **TTFT** (time to first token): from the client sending the request to the client receiving the first output token. ≈ queue wait + routing + prefill + one decode step.
- **ITL** (inter-token latency): time between consecutive output tokens during decode; on CPU here it is ~30–100 ms.
- **Latency** is how long one request takes; **throughput** is how many requests (or tokens) per second the system completes. They trade off: batching more requests together raises throughput but each request waits longer. *Analogy:* a bus (high throughput, you wait for it) versus a taxi (low latency, fewer people moved per hour).

Why you care: the resume claim is about TTFT, measured at the gateway, and the graph's x-axis is throughput offered to the system.

### Percentiles: p50, p95, p99; histograms

Sort all measured latencies. **p50** (the median) is the value half the requests were under; **p95** is the value 95% were under; **p99** likewise. p50 tells you the typical experience, p99 the worst 1% — and the tail is usually caused by a *different mechanism* than the median (here: cache misses and spills, versus hits). A **histogram** is the same data as counts per bucket ("how many requests took 100–200 ms") — Prometheus stores latencies that way, and percentiles are estimated from the buckets. Why you care: you report p50 and p95 for each mode, and you must be able to say *why* they differ.

### Batching, slots, and context size

**Batching** means the engine processes several requests in the same forward pass to use the hardware better. llama-server implements this with **slots** (`-np N`): N independent lanes, each holding its own prompt and its own KV cache, that the engine steps together. `-c` is the total **context** (token capacity) split evenly across slots. *Analogy:* N checkout lanes, each with its own conveyor belt; `-c` is the total belt length shared out. Why you care: one slot = one warm prefix, so `slots × workers` is how many prefixes the cluster can keep warm — the number M6's design hinges on.

### Quantization, Q8_0, GGUF

**Quantization** stores weights in fewer bits (8, 4, …) than the 16-bit floats they were trained in, with a small per-block scale factor to recover the range. **Q8_0** is llama.cpp's 8-bit-per-weight format (blocks of 32 weights with one fp16 scale, ≈8.5 bits per weight). **GGUF** is llama.cpp's single-file model format (weights + tokenizer + metadata). Why you care: fewer bytes per weight means fewer bytes to read per decode step, which on CPU is the whole speed story; Q8_0 is close to lossless for this size.

### "Bound": compute-bound vs memory-bandwidth-bound

Every step of a model does some arithmetic and moves some bytes. If the arithmetic takes longer, the step is **compute-bound**; if moving the bytes takes longer, it is **memory-bandwidth-bound**. Prefill (many tokens, lots of matrix math per weight read) is compute-bound; decode (one token, every weight read once) is memory-bound. *Analogy:* a chef limited by chopping speed versus one limited by how fast ingredients arrive from the pantry. Why you care: it predicts the numbers you will see — prefill throughput in tok/s depends on your CPU's vector units; decode tok/s depends on your RAM bandwidth.

### Hashing, from zero — and why not `hash(key) % N`

A **hash function** turns any input (a string) into a fixed-size number that looks random but is deterministic: same input, same number, every time, on every machine. `blake2b` is a fast, well-distributed one in Python's standard library. Python's built-in `hash()` is *not* suitable: it is randomized per process (so two coordinators disagree) and poorly spread on short strings.

The obvious way to assign a key to one of N servers is `server = hash(key) % N`. It is even and deterministic — until N changes. Going from 3 servers to 4 changes `% 3` to `% 4`, and roughly *three quarters* of all keys land on a different server. Every moved key is a cache miss on its new owner. *Analogy:* renumbering every locker in a school because one new locker was installed. Why you care: workers join and leave (crashes, autoscaling), and each remap is a wave of cold prefills; you want the remap to be ≈1/N of keys, not ≈(N−1)/N.

### Consistent hashing and virtual nodes

**Consistent hashing** fixes the remap problem. Picture the whole range of hash values bent into a circle (the **ring**). Each server is placed at `hash(server_name)` on the circle. A key sits at `hash(key)` and belongs to the first server clockwise from it. Removing a server hands only *its* keys to the next server clockwise; adding one takes only the keys in the arc it lands in. Roughly 1/N of keys move either way.

With one point per server, arcs are of random size and one server can own half the circle. **Virtual nodes** (vnodes) fix that: place each server at many points (`hash(f"{name}#{i}")` for i in 0..149), so every server owns 150 small random arcs whose total evens out, and a departing server's keys spread to *all* its neighbours instead of dumping on one. Lookup is a binary search over the sorted vnode positions. *Analogy:* instead of each person guarding one long stretch of fence, each guards 150 short stretches scattered around it. **Rendezvous (HRW) hashing** is the main alternative: score every server with `hash(key, server)` and pick the highest — same minimal-remap property, no ring, O(N) per lookup.

### LRU (least recently used)

An **LRU cache** has a fixed capacity and, when full, evicts the entry that was used longest ago. In Python an `OrderedDict` does it in a few lines: move an entry to the end on every use, pop from the front to evict. Its known worst case: cycling through *capacity + 1* items in order misses every time. *Analogy:* a desk with room for five books — you put the one you just read on top and shove the bottom one back on the shelf. Why you care: each worker keeps at most `N_SLOTS` warm prefixes under a token budget, and LRU is the policy that decides which one to drop.

### Event loop, coroutines, semaphores, cancellation

Python's `asyncio` runs many tasks on one thread by switching between them at every `await`, where a task is waiting on I/O. That scheduler is the **event loop**; each `async def` is a **coroutine**. An **`asyncio.Semaphore(N)`** is a counter that lets at most N tasks past `async with` at once; the rest wait in line — and *that line is the queue depth* this project measures. **Cancellation**: when a client disconnects, the server cancels the task serving it by raising `CancelledError` at its current `await`; `finally` blocks and `async with` exits still run, which is how sockets get closed on the way out. Why you care: the whole "client hangs up → engine slot is freed" chain is built from these three pieces.

### HTTP streaming and SSE

Normally an HTTP response is one body sent when it is complete. With **chunked streaming** the server keeps the connection open and sends pieces as they are ready. **SSE** (Server-Sent Events) is a small text convention on top of that: `content-type: text/event-stream`, and the body is a sequence of events, each `data: <payload>\n\n` (optionally preceded by `event: <name>\n`). Browsers have a built-in `EventSource` client for it; `curl -N` shows the lines as they arrive. Why you care: every token goes out as one `data:` line, and TTFT is "when the first one leaves".

### Backpressure

**Backpressure** is when a slow consumer slows the producer instead of the producer piling up unbounded data in between. In an async generator, `yield`ing a chunk and only then `await`ing the next one from upstream gives you this for free: a slow client makes the gateway pull from the worker more slowly. *Analogy:* a bucket brigade — nobody fills a bucket until the next person has taken the previous one.

### Token bucket rate limiting

A **token bucket** holds up to `burst` tokens and refills at `rate` per second. Each request takes one token; if the bucket is empty the request is rejected (HTTP 429). It permits short bursts up to `burst` while bounding the long-run average to `rate`. *Analogy:* a prepaid allowance that tops up steadily but can't be saved beyond a cap. Unrelated to LLM tokens — an unfortunate name clash.

### Sidecar, headless Service, DNS discovery, readiness probe

- A **sidecar** is a second container in the same pod that shares its network namespace (`localhost`) and lifecycle. Here the FastAPI wrapper is a sidecar to llama-server.
- A normal Kubernetes **Service** gives you one virtual IP that load-balances across pods. A **headless Service** (`clusterIP: None`) instead makes DNS return the IP of *every ready pod* — that is how the coordinator discovers workers with no Kubernetes API client.
- A **readiness probe** is a periodic HTTP check; a pod that fails it is dropped from Service endpoints (and therefore from headless DNS) until it passes again. Readiness doubles as membership here.

### kind vs a real cluster

**kind** ("Kubernetes in Docker") runs a full Kubernetes cluster where each *node* is a Docker container on your laptop. It is real Kubernetes — same API, same manifests — with two practical differences: there is no cloud load balancer (so you use `NodePort` plus a port mapping to reach services) and images must be pushed into the nodes with `kind load docker-image` (there is no registry). *Analogy:* a flight simulator with the real cockpit. GKE (M8) is the real aircraft; the manifests are the same.

### Deployment vs StatefulSet

A **Deployment** manages N interchangeable pods with random suffixes (`worker-6d9f-xk2p1`); a replaced pod gets a new name. A **StatefulSet** manages N pods with stable ordinal names (`worker-0`, `worker-1`, …) and stable DNS entries; a replaced pod comes back with the *same* name. Why you care: the hash ring positions are computed from the pod name, so a Deployment restart moves keys twice (away, then to the stranger) while a StatefulSet restart moves them away and back.

### HPA, custom metrics, KEDA

The **HPA** (Horizontal Pod Autoscaler) is a Kubernetes controller that sets the replica count of a Deployment/StatefulSet from a metric. Out of the box it knows CPU and memory. For anything else (like "queued requests per worker") something must serve that metric to the HPA through the custom/external metrics API. **KEDA** is an operator that does exactly that: you write a `ScaledObject` with a Prometheus query and a threshold, and KEDA creates and feeds the HPA. Why you care: CPU is the wrong signal for an inference worker; queue depth is right.

### Prometheus scrape, gauges, counters, PromQL

**Prometheus** is a metrics database that *pulls*: every N seconds it does an HTTP GET (a **scrape**) of each target's `/metrics` endpoint, which returns plain text lines like `worker_queue_depth{worker="worker-1"} 3`. A **gauge** is a value that goes up and down (queue depth); a **counter** only goes up (total hits). **PromQL** is its query language — `sum(worker_queue_depth)` adds the gauge across all workers. Why you care: the KEDA trigger is one PromQL expression over metrics your wrapper emits.

### Poisson arrivals, open vs closed loop, Little's law, Zipf

- **Poisson arrivals**: requests arrive independently at an average rate λ, so the gaps between them are exponentially distributed (`random.expovariate(λ)`). It is the standard model for many independent users and produces realistic bursts.
- **Open-loop** load generation fires requests on that schedule regardless of whether earlier ones finished; **closed-loop** ("N concurrent clients") waits for a response before sending the next. Closed loop hides overload because clients slow down with the server; open loop exposes it as growing queues.
- **Little's law**: `concurrency = arrival rate × mean latency`. It links the two views: at 2 req/s with 1.5 s mean latency, 3 requests are in flight on average.
- **Zipf** popularity: item *i* is requested with probability ∝ 1/i^s — a few items get most of the traffic. Real system prompts are Zipf-ish; it is what makes load-aware spill matter.

### Chaos test

A **chaos test** deliberately kills a component during load and records, in time order, what the system does. The goal is a timeline you can narrate, not a pass/fail.

### Terraform, kustomize, helm (one line each)

**Terraform** declares cloud resources (cluster, node pool, budget) in `.tf` files and creates/destroys them with `apply`/`destroy`. **kustomize** (built into `kubectl -k`) layers small patches over base manifests so the GKE variant reuses the kind manifests. **helm** installs packaged Kubernetes applications (Prometheus, KEDA) from charts.

### Structured logs and request ids

A **structured log** line is one JSON object per event rather than free text, so you can grep and aggregate fields. A **request id** is a unique string minted at the edge and passed on every hop so one request's lines can be joined across services.

---

## M1 — kind cluster + llama-server workers (half a day)

### What it is

A local Kubernetes cluster with one control-plane and three worker nodes, and a `worker` Deployment running `llama-server` with the model baked into the image. No routing yet — the goal is to see prompt caching work with your own eyes using nothing but `curl`, so that every later milestone is building on a behavior you have observed.

Why bake the model into the image rather than download at startup: on kind, `kind load docker-image` pushes the image to every node once, so pod restarts are instant. On GKE (M8) you flip to an init container (a container that runs to completion before the main ones start) that pulls from GCS; the manifest change is five lines.

Why `-np 8 -c 8192`: `llama-server` divides its context (`-c`, the total number of tokens of KV it allocates) among `-np` parallel *slots*; each slot has its own KV cache and remembers the prompt it last processed. Eight slots of 1024 tokens fit a 500-token system prompt, a user turn and 128 output tokens with room to spare. Eight slots per worker × 3 workers = 24 warm prefixes cluster-wide, which is more than the 20 templates in the benchmark — that matters in M6. KV for this model is tiny (2 × 24 layers × 2 KV heads × 64 dims × 2 bytes ≈ 12 KB per token, so 8192 tokens ≈ 100 MB); the real constraint on CPU is compute, not memory.

### Look first

Before writing a single manifest, look at what llama-server actually exposes. Every later milestone depends on the JSON you will see here.

1. **Read the flags you are about to use.** `docker run --rm ghcr.io/ggml-org/llama.cpp:server --help | less`. Find `-c`/`--ctx-size`, `-np`/`--parallel`, `-t`/`--threads`, `--cache-prompt`, `--slots`, `--slot-save-path`, `--metrics`, `--host`, `--port`. Notice which of them say "(default: …)" — those are the ones a 2025 build may reject as explicit flags.
2. **Read the server README.** Open `tools/server/README.md` in the llama.cpp repo (in older checkouts it is `examples/server/README.md`). Read the sections on `POST /completion` (note the `cache_prompt`, `id_slot`, `n_predict`, `stream`, `stop` fields), `GET /slots`, `POST /slots/{id}?action=save|restore|erase`, `GET /props`, `GET /metrics`, and `GET /health`. This is the contract the worker wrapper (M2) and the slot LRU (M5) are written against.
3. **Run one server locally and curl each endpoint** (download the GGUF first — the command is in the Code section):

   ```bash
   # One llama-server in Docker, model directory bind-mounted, 4 threads. Runs in the background (&).
   docker run --rm -p 8080:8080 -v $PWD/worker/models:/models ghcr.io/ggml-org/llama.cpp:server \
     -m /models/qwen2.5-0.5b-instruct-q8_0.gguf --host 0.0.0.0 --port 8080 -c 8192 -np 8 -t 4 &
   sleep 5                                       # give it a few seconds to load the model
   curl -s localhost:8080/health                 # {"status":"ok"} once the model is loaded
   curl -s localhost:8080/props | python3 -m json.tool    # model metadata: n_ctx per slot, total_slots, chat template
   curl -s localhost:8080/slots | python3 -m json.tool    # one object per slot: id, is_processing, n_ctx, last prompt
   curl -s localhost:8080/metrics                # Prometheus text: prompt_tokens_total, kv_cache_usage_ratio, ...
   # A real completion, non-streaming, so the whole response JSON is one document you can read:
   curl -s localhost:8080/completion -d '{"prompt":"Hello, my name is","n_predict":8,"cache_prompt":true}' \
     | python3 -m json.tool
   ```

   In the `/completion` output, find and write down: `tokens_evaluated` (prompt length in tokens), `tokens_predicted` (output length), `tokens_cached`, and inside `timings`: `prompt_n` (tokens actually prefilled), `prompt_ms` (prefill time), `predicted_n`, `predicted_ms`, `predicted_per_second`. Run the same curl a second time and watch `prompt_n` and `prompt_ms` collapse. Then curl `/slots` again and notice that one slot now records the prompt it holds — that is the cache you are about to route around.
4. **Streaming shape.** Repeat the completion with `"stream":true` and `curl -N`: you will see `data: {...}` lines, one per token, and a final one with `"stop":true` that carries the `timings` object. The M2 wrapper parses exactly that final frame.
5. **Kubernetes fields you will use.** `kubectl explain deployment.spec.template.spec.topologySpreadConstraints` (what `maxSkew` and `whenUnsatisfiable` mean), `kubectl explain pod.spec.containers.readinessProbe` (note `periodSeconds`, `failureThreshold`), `kubectl explain pod.spec.containers.resources` (requests vs limits). And `kind create cluster --help` to see `--config` and `--name`.

Stop the local server (`kill %1`) before the cluster test below, or the port-forward will collide on 8080.

### Code

`k8s/kind-config.yaml`:

```yaml
kind: Cluster                          # a kind (Kubernetes-in-Docker) cluster definition, not a Kubernetes object
apiVersion: kind.x-k8s.io/v1alpha4     # kind's own config schema version
nodes:
  - role: control-plane                # runs the API server, scheduler, etcd; also schedulable in kind
    extraPortMappings:                 # Docker-level port publish: host:30080 -> this node container:30080
      - containerPort: 30080           # the gateway Service's NodePort (M4) — reachable as http://localhost:30080
        hostPort: 30080                # from WSL *and* from Windows, because Docker Desktop forwards it
  - role: worker                       # three worker nodes so the three llama-server pods spread out
  - role: worker                       #   (topologySpreadConstraints below puts one per node)
  - role: worker                       # each "node" is one Docker container running kubelet + containerd
```

`worker/Dockerfile.llama` (download the GGUF once into `worker/models/` first):

```dockerfile
# Base image: llama.cpp's prebuilt CPU server. Its ENTRYPOINT is /app/llama-server, so the
# container runs llama-server directly and the pod spec's `args` become its command-line flags.
FROM ghcr.io/ggml-org/llama.cpp:server

# Bake the model into the image. ~530 MB. Rationale: `kind load` pushes the image to every node
# once, so pod restarts need no download. On GKE (M8) this becomes an init container instead.
COPY models/qwen2.5-0.5b-instruct-q8_0.gguf /models/model.gguf

# No CMD/ENTRYPOINT override: the base image's ENTRYPOINT is /app/llama-server; args come from the pod spec.
```

```bash
mkdir -p worker/models                                   # keep the model next to the Dockerfile (Docker build context)
curl -L -o worker/models/qwen2.5-0.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
# -L follows the redirect Hugging Face returns for /resolve/ URLs; -o names the output file.
# Add worker/models/ to .gitignore — the GGUF must never be committed.
```

`k8s/worker.yaml` (M1 version — one container, plain Service):

```yaml
apiVersion: apps/v1
kind: Deployment                       # M1 uses a Deployment; M7 switches to a StatefulSet for stable names
metadata: { name: worker }
spec:
  replicas: 3                          # three engines = three independent KV caches to route between
  selector: { matchLabels: { app: worker } }    # which pods this Deployment owns: those labelled app=worker
  template:                            # the pod template; every replica is stamped from this
    metadata: { labels: { app: worker } }
    spec:
      topologySpreadConstraints:       # ask the scheduler to spread pods across nodes
        - maxSkew: 1                   # no node may have more than 1 more worker pod than any other
          topologyKey: kubernetes.io/hostname   # "node" is the unit of spreading
          whenUnsatisfiable: ScheduleAnyway     # soft constraint: if it can't be met, still schedule
          labelSelector: { matchLabels: { app: worker } }   # spread is computed over pods with this label
      containers:
        - name: llama
          image: dkv/worker-llama:dev  # the image built from Dockerfile.llama and pushed with `kind load`
          imagePullPolicy: IfNotPresent   # never try to pull from a registry; the image is only on the nodes
          args:                        # these become llama-server's flags (the image ENTRYPOINT is llama-server)
            ["-m", "/models/model.gguf",        # model path inside the image
             "--host", "0.0.0.0",               # listen on all interfaces so the pod IP is reachable
             "--port", "8080",                  # engine port; the M2 wrapper talks to 127.0.0.1:8080
             "-c", "8192",                      # total context tokens, split evenly across slots
             "-np", "8",                        # 8 parallel slots -> 1024 tokens of KV each
             "-t", "2",                         # 2 CPU threads per engine; 3 engines x 2 = 6 of 8 cores
             "--cache-prompt",                  # reuse KV for the longest shared token prefix (default in new builds)
             "--metrics",                       # expose GET /metrics in Prometheus text format
             "--slots",                         # expose GET /slots (per-slot state); needed for M5's Look first
             "--slot-save-path", "/tmp/slots"]  # enables POST /slots/{id}?action=save|restore (M5 stretch goal)
          ports: [{ containerPort: 8080 }]      # documentation + Service targetPort; does not open anything
          readinessProbe:              # pod is "Ready" (and in Service DNS) only while this returns 200
            { httpGet: { path: /health, port: 8080 }, periodSeconds: 2 }   # /health is 503 until the model is loaded
          resources:
            requests: { cpu: "1", memory: "1Gi" }   # what the scheduler reserves on the node
            limits:   { cpu: "2", memory: "2Gi" }   # hard caps: cpu=2 matches -t 2; memory > model (530 MB) + KV (100 MB) + overhead
---
apiVersion: v1
kind: Service                          # M1 only: a normal ClusterIP Service so `port-forward svc/worker` works
metadata: { name: worker }
spec:
  selector: { app: worker }            # send traffic to pods with this label
  ports: [{ port: 8080, targetPort: 8080 }]   # Service port -> container port
```

If your `llama-server` build rejects a flag (`--cache-prompt` and `--slots` became defaults in 2025 builds and may or may not still be accepted), check with `docker run --rm ghcr.io/ggml-org/llama.cpp:server --help | grep -E 'cache-prompt|slots'` and drop the ones it doesn't list.

`scripts/kind-up.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail                      # -e: exit on error; -u: unset var is an error; -o pipefail: a failing pipe stage fails the pipe
cd "$(dirname "$0")/.."                # run from the repo root regardless of where the script was invoked

# Create the cluster only if it does not already exist (idempotent: safe to re-run after every milestone).
kind get clusters | grep -q '^dkv$' || kind create cluster --name dkv --config k8s/kind-config.yaml

# Build the engine image (model baked in) and push it into every kind node's container runtime.
docker build -t dkv/worker-llama:dev -f worker/Dockerfile.llama worker   # last arg = build context dir
kind load docker-image --name dkv dkv/worker-llama:dev                   # no registry on kind: copy the image to the nodes

# Apply the manifest and block until all 3 replicas report Ready (or fail after 180 s).
kubectl apply -f k8s/worker.yaml
kubectl rollout status deploy/worker --timeout=180s
```

### Test

```bash
chmod +x scripts/kind-up.sh && scripts/kind-up.sh
kubectl get pods -o wide            # 3 workers, one per kind node (the NODE column should show three different names)
kubectl port-forward svc/worker 8080:8080 &    # tunnel localhost:8080 -> ONE worker pod (port-forward picks a single pod)

# 1. Health + slots
curl -s localhost:8080/health       # {"status":"ok"} once the model is loaded
curl -s localhost:8080/slots | python3 -c 'import sys,json; s=json.load(sys.stdin); print(len(s), "slots")'   # expect "8 slots"

# 2. Same prompt twice via the native endpoint; watch prompt_n (tokens actually prefilled)
# Build a ~380-token ChatML prompt: the system message is one sentence repeated 60x, then a short user turn.
P=$(python3 -c 'print("<|im_start|>system\n" + "You are a careful assistant. " * 60 + "<|im_end|>\n<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n")')
for i in 1 2; do
  # -d: JSON body built by Python so quoting is safe; n_predict=8 keeps decode short; cache_prompt=true asks for prefix reuse
  curl -s localhost:8080/completion -d "$(python3 -c "import json;print(json.dumps({'prompt':'''$P''','n_predict':8,'cache_prompt':True}))")" \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); print("evaluated", r["tokens_evaluated"], "processed", r["timings"]["prompt_n"], "prompt_ms", round(r["timings"]["prompt_ms"]))'
    # evaluated = prompt length in tokens; processed = tokens actually prefilled; prompt_ms = prefill wall time
done
```

Port-forward pins you to one pod, so both calls hit the same llama-server — that is what you want for this test.

### Expected

Pods ready in 10–40 s after image load (model load itself is 1–3 s; the rest is scheduling and the 2 s readiness period). The first `/completion` call reports `processed` ≈ `evaluated` (~380 tokens) and `prompt_ms` somewhere between 800 and 4000 ms — prefill throughput of a 0.5B Q8_0 model on 2 CPU threads is roughly 100–400 tok/s depending on your CPU's AVX2/AVX-512 support (the CPU's vector instruction sets; wider vectors = more multiply-adds per cycle) and whether WSL is fighting Windows for the cores. The second call reports `processed` of 0–2 tokens and `prompt_ms` of a few milliseconds. That gap is the entire project in one line of output. Write those two numbers down; they are the theoretical ceiling for what routing can save on this hardware.

The `/completion` response keys (`tokens_evaluated`, `timings.prompt_n`, `timings.prompt_ms`) are the ones the worker wrapper depends on in M2; if your build names them differently, `curl ... | python3 -m json.tool` once and adjust there.

### Reading a failure

| Symptom | Cause |
|---|---|
| Pod `CrashLoopBackOff`, log says unknown argument | A flag your llama-server build doesn't have; see the `--help` note above |
| Pod `OOMKilled` | `-c` too large for the memory limit, or the image was built with a bigger GGUF than you think; check `kubectl describe pod` |
| Readiness never passes, log stuck at "loading model" | `kind load` didn't happen and the node pulled a stale image, or CPU limit so low the load takes minutes — raise limits |
| Second call still shows `processed` ≈ `evaluated` | `cache_prompt` not sent, or the port-forward reconnected to a different pod (`kubectl port-forward` prints which pod it picked) |
| `prompt_ms` wildly different run to run | Thread oversubscription: three servers × `-t` > physical cores, or Windows Defender scanning the WSL disk |
| `curl: (52) Empty reply` on `/slots` | Slots endpoint disabled; add `--slots` or drop `--no-slots` |

### Close the milestone

Commit: `M1: kind cluster with 3 llama-server workers, prompt caching verified via /completion timings`.
README section: "Local cluster" — the kind config, the two-call `prompt_ms` output verbatim, and one sentence on why CPU workers are the right substrate for testing a router.

---

## M2 — the worker wrapper (1 day)

### What it is

A FastAPI **sidecar** (a second container in the same pod, sharing `localhost` with llama-server) in the same pod as llama-server. Everything upstream talks to the wrapper, never to the engine. It does four things:

1. **Proxies generation.** Accepts `{system, user, max_tokens, prefix_hash}`, renders the ChatML prompt (Qwen's chat template — the `<|im_start|>role\n...<|im_end|>` markup that turns a system/user pair into the exact text the model was trained on) itself so the token sequence is byte-identical for the same system prompt, calls llama-server's native `/completion` with `cache_prompt: true`, and re-emits tokens as SSE.
2. **Gates concurrency.** An `asyncio.Semaphore(N_SLOTS)` (a counter that admits at most N tasks at once; see the concepts section) means at most one in-flight request per engine slot. Requests beyond that wait in the wrapper, and *that waiting is the queue depth* the coordinator and autoscaler will read. llama-server has its own internal queue, but you cannot observe it cheaply from outside, and controlling admission yourself is what lets M5 pin slots.
3. **Observes cache state.** Every completed request tells you `tokens_evaluated` (prompt length) and `timings.prompt_n` (tokens actually prefilled). Their difference is the number of tokens reused from the slot's KV. A hit is a request where most of the prompt was reused. In M2 the wrapper merely *records* which prefix hashes have been seen warm; in M5 it takes control of which slot holds what.
4. **Reports `/stats` and `/healthz`.** Stats is JSON for the coordinator; the same numbers go out in Prometheus text format (one `name{labels} value` line per metric, the format a Prometheus scrape expects) on `/metrics` for M7.

Why a sidecar rather than a fork of llama-server or a single fat image: the wrapper is engine-agnostic. When Project 2 exists you swap the container image and the `/completion` call; nothing else in the cluster knows.

### Look first

The wrapper is glue between three real things: llama-server's streaming `/completion`, Starlette's `StreamingResponse`, and `httpx`'s streaming client. Look at each before gluing.

1. **The raw stream you will parse.** With the local llama-server from M1 running:

   ```bash
   # -N disables curl's output buffering so lines print as they arrive; note the 'data: ' prefix and blank-line separators
   curl -sN localhost:8080/completion -d '{"prompt":"<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n","n_predict":6,"stream":true,"cache_prompt":true}'
   ```

   Notice: every frame is `data: {"content": "...", "stop": false, ...}`; the last frame has `"stop": true` and carries `timings`, `tokens_evaluated`, `tokens_predicted`. The wrapper's parser is nothing more than "strip `data: `, `json.loads`, forward `content`, act on `stop`".

2. **How Starlette streams and cancels.** Open the installed source and read two things:

   ```python
   import inspect, starlette.responses, starlette.requests
   print(inspect.getsource(starlette.responses.StreamingResponse))   # __call__: runs stream_response and listen_for_disconnect concurrently
   print(inspect.getsource(starlette.requests.Request.is_disconnected))   # polls the ASGI receive channel for http.disconnect
   ```

   In `StreamingResponse.__call__`, find the task group that runs `stream_response` alongside `listen_for_disconnect`: when the client goes away, the disconnect listener cancels the group, which is what raises `CancelledError` inside your generator at its current `yield`/`await`. That is the mechanism the disconnect test below relies on. Also note `media_type` becomes the `content-type` header, and that each `yield`ed chunk is sent immediately (no buffering inside Starlette).

3. **How httpx streams.** `print(inspect.getsource(httpx.AsyncClient.stream))` — it is a context manager around `send(..., stream=True)`; leaving the `async with` block closes the response and the underlying socket, which is what tells llama-server the client is gone. `aiter_lines()` splits the body on newlines for you.

4. **The primitives.** `print(inspect.getsource(asyncio.Semaphore))` — read `acquire`: it appends a Future to a deque of waiters when the counter is 0 — *that deque is your queue*. And `help(collections.OrderedDict.move_to_end)`.

5. **Slot state after a request.** `curl -s localhost:8080/slots | python3 -m json.tool` after the stream above: find the slot whose `prompt` (or `n_past`/`tokens` in newer builds) reflects the prompt you sent. That is the engine's own record of what is warm — the thing the wrapper's `warm` table mirrors.

### Code

`worker/app.py`:

```python
import asyncio, json, os, time                    # asyncio: semaphore + async streaming; json: SSE payloads; os: env; time: timestamps
from collections import OrderedDict              # OrderedDict = dict that remembers insertion order -> a 5-line LRU

import httpx                                      # async HTTP client used to talk to llama-server on localhost
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse   # StreamingResponse: send chunks as an async generator yields them
from pydantic import BaseModel                    # request-body validation

# --- configuration, all from the environment so the same image runs locally and in the pod ---
LLAMA = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")   # engine address; a sidecar shares the pod's localhost
N_SLOTS = int(os.environ.get("N_SLOTS", "8"))                   # MUST equal llama-server's -np; one semaphore permit per engine slot
NAME = os.environ.get("POD_NAME", os.uname().nodename)          # worker identity reported in /stats; the pod name in k8s (downward API)

app = FastAPI()
slots = asyncio.Semaphore(N_SLOTS)                # admission control: at most N_SLOTS requests inside the engine at once
# Counters reported by /stats. "waiting" = tasks blocked on the semaphore = the queue depth the coordinator/HPA read.
state = {"in_flight": 0, "waiting": 0, "hits": 0, "misses": 0, "evictions": 0}
# Which prefixes this engine has recently prefilled (so should be warm in some slot). Ordered oldest -> newest.
warm: "OrderedDict[str, dict]" = OrderedDict()          # prefix_hash -> {"tokens": n, "last": ts}
# One shared client (connection pool) to the engine. connect timeout 5 s; read timeout 300 s because a
# streaming generation can legitimately take minutes on CPU and the read clock resets on every chunk.
client = httpx.AsyncClient(base_url=LLAMA, timeout=httpx.Timeout(5.0, read=300.0))


class GenReq(BaseModel):
    system: str                                   # the shared template portion — what the cache key is derived from
    user: str                                     # the per-request turn
    max_tokens: int = 128                         # decode budget; llama-server calls it n_predict
    prefix_hash: str                              # computed ONCE by the gateway (M4); the worker never recomputes it


def render(system: str, user: str) -> str:
    """Render Qwen's ChatML template by hand.

    Doing it here (not in the gateway, not via /v1/chat/completions) guarantees the exact same bytes
    -> the exact same token ids for the same system prompt, which is what makes the KV prefix match.
    Any stray whitespace difference would silently turn every request into a cache miss.
    """
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")   # ends where the model should start writing


def record_cache(prefix_hash: str, total: int, reused: int) -> bool:
    """Classify one finished request as hit/miss from MEASURED reuse and update the warm table.

    total  = prompt tokens (tokens_evaluated); reused = tokens the engine did not have to prefill.
    A 'hit' means at least half the prompt came from cache: the system prompt dominates, so a warm
    template gives reuse/total ~ 0.9; a cold one gives ~0.
    """
    hit = total > 0 and reused / total >= 0.5
    state["hits" if hit else "misses"] += 1
    warm[prefix_hash] = {"tokens": total, "last": time.time()}   # (re)insert the entry with fresh metadata
    warm.move_to_end(prefix_hash)                # LRU bookkeeping: most recently used goes to the end
    while len(warm) > N_SLOTS:                 # llama-server can hold at most one prompt per slot
        warm.popitem(last=False)               # pop from the FRONT = evict the least recently used prefix
        state["evictions"] += 1
    return hit


@app.post("/generate")
async def generate(req: GenReq):
    async def stream():                           # async generator: each `yield` is one SSE chunk sent to the caller
        t_arrive = time.perf_counter()            # monotonic clock, for queue-wait measurement
        state["waiting"] += 1                     # we are now in line for a slot (visible in /stats as queue_depth)
        async with slots:                         # blocks here until one of the N_SLOTS permits is free
            state["waiting"] -= 1                 # admitted: leave the queue...
            state["in_flight"] += 1               # ...and count as running inside the engine
            queue_ms = (time.perf_counter() - t_arrive) * 1000   # how long this request waited for a slot
            payload = {"prompt": render(req.system, req.user),   # exact ChatML text -> deterministic token ids
                       "stream": True,                            # llama-server emits SSE frames, one per token
                       "cache_prompt": True,                      # reuse the slot's KV for the longest common token prefix
                       "n_predict": req.max_tokens,               # max output tokens
                       "stop": ["<|im_end|>"],                    # stop when the model closes its turn
                       "temperature": 0.7}                        # sampling randomness; irrelevant to caching
            try:
                # Open a streaming POST to the engine. Leaving this block (normally or via cancellation) closes the
                # socket, which is what makes llama-server abort generation and free the slot on client disconnect.
                async with client.stream("POST", "/completion", json=payload) as r:
                    r.raise_for_status()          # non-2xx from the engine -> httpx.HTTPStatusError -> 500 upstream
                    async for line in r.aiter_lines():           # engine frames: "data: {...}" separated by blank lines
                        if not line.startswith("data: "):
                            continue                             # skip blank separators and any non-data lines
                        ev = json.loads(line[6:])                # strip the 6-char "data: " prefix and parse the frame
                        if ev.get("content"):                    # a token (or a few merged characters)
                            # Re-emit as our own SSE frame. "\n\n" terminates an SSE event.
                            yield f"data: {json.dumps({'token': ev['content']})}\n\n"
                        if ev.get("stop"):                       # final frame: carries timings for the whole request
                            tm = ev.get("timings", {})
                            total = ev.get("tokens_evaluated", 0)                 # prompt length in tokens
                            reused = max(total - tm.get("prompt_n", total), 0)    # prompt tokens NOT prefilled = served from KV
                            hit = record_cache(req.prefix_hash, total, reused)    # measured hit/miss, not assumed
                            # The "done" event: everything the gateway's log line and the benchmark need, per request.
                            done = {"worker": NAME, "hit": hit, "prompt_tokens": total, "reused": reused,
                                    "prefill_ms": tm.get("prompt_ms"),           # engine-measured prefill wall time
                                    "queue_ms": round(queue_ms, 1),              # wrapper-measured wait for a slot
                                    "output_tokens": ev.get("tokens_predicted")}
                            yield f"event: done\ndata: {json.dumps(done)}\n\n"   # named SSE event so clients can tell it from tokens
            finally:
                state["in_flight"] -= 1           # runs on success, error AND CancelledError -> counters never drift
        # leaving `async with slots` here releases the permit -> the next waiter is admitted
    return StreamingResponse(stream(), media_type="text/event-stream")   # text/event-stream = SSE content type


def snapshot() -> dict:
    """The /stats document. queue_depth is the coordinator's spill signal and (via /metrics) the HPA's."""
    total = state["hits"] + state["misses"]
    return {"name": NAME, "slots": N_SLOTS, "in_flight": state["in_flight"], "queue_depth": state["waiting"],
            "cache_entries": len(warm), "hits": state["hits"], "misses": state["misses"],
            "evictions": state["evictions"], "hit_rate": round(state["hits"] / total, 3) if total else None,
            "warm": list(warm.keys())}            # which prefixes we believe are warm — the seed of "hybrid" routing


@app.get("/stats")
def stats():                                      # sync handler is fine: no I/O, must stay fast (the coordinator polls it every second)
    return snapshot()


@app.get("/healthz")
async def healthz():
    r = await client.get("/health")               # proxy the engine's health so the POD is not Ready until the model is loaded
    return PlainTextResponse("ok" if r.status_code == 200 else "engine down", status_code=r.status_code)


@app.get("/metrics")
def metrics():
    """Prometheus text exposition: one `worker_<field>{worker="name"} value` line per numeric stat."""
    s = snapshot()
    lines = [f'worker_{k}{{worker="{NAME}"}} {v}' for k, v in s.items() if isinstance(v, (int, float))]   # skip strings/lists
    return PlainTextResponse("\n".join(lines) + "\n")   # Prometheus requires a trailing newline
```

`worker/Dockerfile`:

```dockerfile
FROM python:3.11-slim                                          # small base; no compiler needed for these wheels
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic  # --no-cache-dir keeps the layer small
COPY app.py /app/app.py
WORKDIR /app                                                   # so `app:app` resolves to /app/app.py
# uvicorn = the ASGI server that runs FastAPI. --host 0.0.0.0 so the pod IP is reachable; port 8000 = wrapper port.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Manifest changes in `k8s/worker.yaml`: add the sidecar container, pass the pod name down, and replace the Service with a **headless** one on port 8000 (the coordinator will discover workers by resolving its DNS name in M3):

```yaml
        - name: wrapper                          # second container in the SAME pod: shares localhost with llama
          image: dkv/worker:dev                  # built from worker/Dockerfile
          imagePullPolicy: IfNotPresent
          env:
            - { name: LLAMA_URL, value: "http://127.0.0.1:8080" }   # sidecar reaches the engine over the pod's loopback
            - { name: N_SLOTS, value: "8" }                         # must match llama's -np above
            - name: POD_NAME                     # downward API: inject this pod's own name as an env var
              valueFrom: { fieldRef: { fieldPath: metadata.name } }   # -> NAME in app.py -> ring position in M3
          ports: [{ containerPort: 8000 }]
          readinessProbe: { httpGet: { path: /healthz, port: 8000 }, periodSeconds: 2 }   # /healthz proxies llama's /health
          resources: { requests: { cpu: "100m", memory: "128Mi" }, limits: { cpu: "500m", memory: "256Mi" } }   # it only shuffles bytes
---
apiVersion: v1
kind: Service
metadata: { name: worker-hl }                    # "-hl" = headless; the coordinator resolves this DNS name
spec:
  clusterIP: None            # headless: DNS returns one A record per READY pod (instead of one virtual IP)
  selector: { app: worker }
  ports: [{ port: 8000, targetPort: 8000 }]      # wrapper port, not the engine port: upstream never talks to llama directly
```

Add `docker build -t dkv/worker:dev worker && kind load docker-image --name dkv dkv/worker:dev` to `kind-up.sh`.

### Test

Iterate locally first — one llama-server in Docker, the wrapper under `uvicorn`:

```bash
# Engine: same as M1's Look-first, -t 4 because nothing else is competing on the laptop right now
docker run --rm -p 8080:8080 -v $PWD/worker/models:/models ghcr.io/ggml-org/llama.cpp:server \
  -m /models/qwen2.5-0.5b-instruct-q8_0.gguf --host 0.0.0.0 --port 8080 -c 8192 -np 8 -t 4 &
cd worker && uvicorn app:app --port 8000 &       # wrapper with defaults: LLAMA_URL=127.0.0.1:8080, N_SLOTS=8

SYS=$(python3 -c 'print("You are a careful assistant. " * 60)')   # ~360-token system prompt
for i in 1 2 3; do
  # Same system prompt + same prefix_hash each time, different user turn: call 1 should miss, calls 2-3 should hit
  curl -sN localhost:8000/generate -H 'content-type: application/json' \
    -d "$(python3 -c "import json;print(json.dumps({'system':'''$SYS''','user':'Say hi in $i words.','max_tokens':16,'prefix_hash':'abc'}))")" \
    | grep -A1 'event: done' | tail -1           # print only the JSON of the done event
done
curl -s localhost:8000/stats | python3 -m json.tool
```

Then the disconnect test: start a long generation and kill curl.

```bash
# timeout 1: kill curl after 1 s, mid-stream. The wrapper must notice and free the slot.
timeout 1 curl -sN localhost:8000/generate -H 'content-type: application/json' \
  -d '{"system":"x","user":"Write 500 words about rivers.","max_tokens":400,"prefix_hash":"z"}' >/dev/null
sleep 2; curl -s localhost:8000/stats | grep -o '"in_flight": [0-9]*'   # must be 0: the cancellation chain worked
```

### Expected

Call 1: `"hit": false`, `reused` 0, `prefill_ms` in the hundreds to low thousands. Calls 2–3: `"hit": true`, `reused` ≈ `prompt_tokens` − 10 or so (the user turn differs each time, and llama-server's match is a token-level longest-common-prefix, so the reused count is the shared *token* prefix, which ends a token or two before your `<|im_end|>` marker), `prefill_ms` an order of magnitude smaller. `/stats` shows `hits: 2, misses: 1, cache_entries: 1`.

After the disconnect test, `in_flight` must be 0 within ~2 s. What happens under the hood: uvicorn cancels the response task when the client goes away, `CancelledError` fires at the `yield` inside `stream()`, the `async with client.stream(...)` exits and closes the socket to llama-server, and llama-server notices the closed connection on its next token and releases the slot (its log says so). If `in_flight` stays 1 for the whole 400 tokens, the cancellation chain is broken somewhere — see the table.

### Reading a failure

| Symptom | Cause |
|---|---|
| Every call is a miss even with identical system prompt | The rendered prompt differs (trailing whitespace, `\r\n` from Windows editors) or `cache_prompt` didn't make it into the payload |
| `reused` is large but `hit` is false | Threshold: a very long user turn can push the shared fraction under 0.5; that is by design — long user turns genuinely dominate prefill |
| `KeyError: 'tokens_evaluated'` | Different llama-server version; print `ev` on the stop frame and map the names |
| `in_flight` stuck after disconnect | You are behind a proxy that buffers (some `curl` flags, Windows port-forward quirks); test from inside WSL directly, and confirm llama-server logs a slot release |
| Second concurrent request waits until the first finishes | Semaphore size 1 (env var not set) or llama-server started with `-np 1` |
| Tokens arrive in bursts, not one at a time | Something is buffering: make sure `media_type="text/event-stream"` and you are using `curl -N` |

### Close the milestone

Commit: `M2: worker wrapper — /generate SSE proxy over llama-server, slot-gated admission, /stats with measured hit/miss`.
README section: "Worker" — the pod diagram (two containers), the three JSON `done` events from the test, and the paragraph explaining that hit/miss is measured from the engine's own prefill count rather than assumed.

---

## M3 — coordinator with a consistent-hash ring (1.5 days)

### What it is

The coordinator answers one question: *given a prefix hash, which worker?* The answer must be (a) the same every time for the same prefix, so that prefix's KV keeps living in one place; (b) spread evenly across workers; (c) change as little as possible when a worker joins or leaves, because every key that moves is a cache miss on its new owner.

**Consistent hashing** gives all three. (If you skipped the concepts section: the naive `hash(key) % N` remaps ~all keys when N changes; consistent hashing remaps ~1/N.) Picture the 64-bit hash space as a circle. Each worker is placed on the circle at several points (its *virtual nodes*, here 150 per worker, at `hash(f"{name}#{i}")`). A key is placed at `hash(key)` and owned by the first vnode clockwise from it. Lookup is a binary search over the sorted vnode hashes: O(log(V·N)). When a worker leaves, only the keys that were owned by *its* vnodes move — to whoever is next clockwise — which is ≈1/N of all keys. When one joins, it takes ≈1/(N+1) of keys, spread evenly across the others. Nothing else moves.

Why virtual nodes: with one point per worker, three workers cut the circle into three arcs of random size, and one worker can easily own half the keys. With 150 points each, every worker owns 150 random arcs and the sum averages out — the test below shows max/mean load within ~10% at 150 vnodes versus 50–200% off at 1 vnode. Also, when a worker leaves, its keys fan out to *all* remaining workers (its 150 arcs have 150 different clockwise neighbors) instead of dumping onto one.

**What is "the prefix"?** This must be precise or the whole thing is undefined. The prefix is the *template portion* of the prompt — the system message, normalized (NFC — Unicode's canonical composed form, so "é" is always one code point and not "e" + combining accent — and stripped of leading/trailing whitespace) — because that is what is genuinely shared across requests in a real product: many users, one system prompt or few-shot block, different user turns. Hashing the first K tokens instead would require a tokenizer in the gateway, would make "K" a magic number, and would make routing depend on the user turn when the system prompt is short — none of which buys anything, because llama-server's cache match is token-level longest-common-prefix regardless of how you routed. The routing key only needs to be *consistent*, not token-exact: two requests that route together and share the system message will share the token prefix. If there is no system message, fall back to the first 256 characters of the first user message. The hash is `blake2b(normalized, digest_size=16)`; blake2b is in the stdlib, fast, and the digest size is irrelevant for routing (you hash the hex again to place it on the ring). `xxhash` is faster but is a dependency for no gain at this request rate.

**Alternatives you must be able to defend against** (this is a guaranteed interview thread):

- *Central routing table* (prefix → worker, stored in the coordinator or Redis). Strictly more accurate — it can know what is actually cached, not just where it *should* be. But now the coordinator is stateful: the table must survive restarts, be replicated if you run two coordinators, be expired (it grows with every prefix ever seen), and be kept consistent with worker evictions. The ring is a pure function of the membership set; a fresh coordinator rebuilds it from DNS in one poll. Production systems that go this route (SGLang's router, llm-d's prefix-aware scheduler) keep an *approximate* per-worker radix tree (a prefix tree — a trie whose nodes hold token runs, so "what prefixes does this worker hold" is one walk) of recent prefixes and treat it as a hint, not truth — that is the "hybrid" you mention as the next step, and M5's `warm` list in `/stats` is the seed of it.
- *Rendezvous (HRW) hashing*: for each worker compute `hash(key, worker)`, pick the max. Same minimal-remap property, no vnodes needed, perfectly even by construction, and the sorted list of scores gives you a natural fallback order. It is O(N) per lookup instead of O(log(V·N)) — irrelevant at N=3, relevant at N=1000. Honest answer: at this scale either is fine; the ring is the one interviewers expect to see drawn, it is the one you can whiteboard the remap argument on, and the vnode-balance tradeoff is a richer discussion. Say you'd pick rendezvous for a small fixed pool with weights, and the ring when N is large or you want cheap range ownership.

**Load-aware fallback.** Pure affinity has a failure mode: one popular prefix makes one worker hot while the others idle. So the coordinator keeps each worker's `queue_depth` from polling `/stats` every second, and if the affinity worker's queue is over a threshold it *spills* to the next distinct worker clockwise on the ring. This trades a cache hit for shorter queueing. On CPU workers where a miss costs 1–4 s of prefill, the threshold should be where expected queue wait exceeds that — a queue of 3–4 with ~1 s per request is about right, and it is an env var so you can sweep it in M6. Side effect worth writing down: the spill target now warms that prefix too, so hot prefixes get replicated organically. The stats are up to a second stale, so a burst can send several requests to a worker before the coordinator sees its queue grow; a counter of "routed since last poll" per worker is the cheap fix if you see it in the logs.

### Look first

Read a real ring before writing one, and look at the DNS answer the coordinator will consume.

1. **A real consistent-hash ring in Python.** `pip install uhashring`, then:

   ```python
   import inspect
   from uhashring import HashRing
   hr = HashRing(nodes=["w0", "w1", "w2"])            # default: many virtual nodes per node (vnodes=40 x replicas=4 in uhashring's terms)
   print(hr.get_node("prefix-7"))                     # same key -> same node, every time
   print(len(hr.get_points()))                        # total vnode points on the ring: 3 nodes x 160 points
   print(hr.get_points()[:5])                         # (hash, node) pairs, SORTED by hash: this is the ring
   print(inspect.getsource(HashRing.get_node))        # hashes the key, then...
   print(inspect.getsource(hr.runtime._get_pos))      # ...a bisect over the sorted point list, with wrap-around at the end
   print(inspect.getsource(HashRing.remove_node))     # drops that node's points only; nothing else moves
   ```

   (Exact method names differ slightly between uhashring versions — `HashRing.runtime` may be a `HashRingRuntime` or the points may live on `hr.runtime.ring`; `dir(hr)` and `dir(hr.runtime)` will show you. The thing to notice is the same in any version: a sorted list of `(hash, node)`, a hash of the key, a `bisect`, and modulo wrap.) An older, one-file alternative with the same structure is the `hash_ring` package (`hash_ring.HashRing`, `replicas=160`).

2. **Envoy's ring-hash load balancer docs.** Read the "Ring hash" section of Envoy's load-balancing docs (`config.cluster.v3.Cluster.RingHashLbConfig`): note `minimum_ring_size` (default 1024) and `maximum_ring_size` (8M), and the explanation that each host gets *ring_size / hosts* hashes — so a 1024-entry ring with 3 hosts is ~340 vnodes each — and the sentence about balance improving with ring size. That is the same vnode-count tradeoff you are about to test with 1/10/150. Also skim the "Maglev" subsection: a lookup-table alternative to the ring with O(1) lookups.

3. **The bisect you will call.** `python3 -c "import bisect; help(bisect.bisect)"` — it returns the insertion point *after* any equal entries; that index, modulo the list length, is "first vnode clockwise". Confirm with a 4-line experiment on `[10, 20, 30]` and keys 5, 25, 35.

4. **What the headless Service DNS actually returns.** From inside the cluster:

   ```bash
   kubectl run -it --rm dns --image=busybox:1.36 --restart=Never -- nslookup worker-hl.default.svc.cluster.local
   # expect: one A record per READY worker pod (three 10.244.x.y addresses), no virtual IP
   kubectl explain service.spec.clusterIP      # read the paragraph about "None" = headless
   ```

   Then `kubectl scale deploy/worker --replicas=2` and run the lookup again — two records. That is the membership protocol; the coordinator just calls `socket.getaddrinfo` on that name.

5. **The FastAPI startup hook.** `print(inspect.getsource(fastapi.FastAPI.on_event))` — it is deprecated in favour of `lifespan`; it still works and is shorter, which is why the code below uses it. Know the replacement exists.

### Code

`coordinator/ring.py`:

```python
import bisect, hashlib                          # bisect: binary search over the sorted vnode list; hashlib: blake2b


def h64(s: str) -> int:
    """Stable 64-bit hash of a string. blake2b with an 8-byte digest -> int in [0, 2^64).

    Stable across processes and machines (unlike Python's hash()), well distributed even for short
    strings like "w0#17". Every position on the ring (vnodes AND keys) comes through this function.
    """
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


class Ring:
    def __init__(self, vnodes: int = 150):
        self.vnodes = vnodes                      # points per worker on the circle; 150 gives ~±5% balance
        self._keys: list[int] = []      # sorted vnode hashes  (the "circle", stored as a sorted list)
        self._owners: list[str] = []    # parallel list: owner of each vnode (same index as _keys)
        self.nodes: set[str] = set()              # current membership; the ring is a pure function of this set

    def add(self, node: str) -> None:
        if node in self.nodes:
            return                                # idempotent: re-adding must not create duplicate points
        self.nodes.add(node)
        for i in range(self.vnodes):
            k = h64(f"{node}#{i}")                # vnode i of this node; depends on the NAME only -> same positions after a restart
            idx = bisect.bisect(self._keys, k)    # where k belongs in the sorted list
            self._keys.insert(idx, k)             # keep _keys sorted...
            self._owners.insert(idx, node)        # ...and _owners aligned with it

    def remove(self, node: str) -> None:
        if node not in self.nodes:
            return
        self.nodes.discard(node)
        # Drop ONLY this node's points. Everyone else's positions are untouched, which is exactly why only
        # this node's keys move (to whichever vnode is next clockwise from each of its former points).
        kept = [(k, o) for k, o in zip(self._keys, self._owners) if o != node]
        self._keys = [k for k, _ in kept]
        self._owners = [o for _, o in kept]

    def lookup(self, key: str, n: int = 1) -> list[str]:
        """Up to n distinct owners, walking clockwise from the key's position."""
        if not self._keys:
            return []                             # empty ring: no workers known yet
        # bisect gives the index of the first vnode with hash > h64(key) = first point clockwise.
        # % len wraps past the end of the list back to index 0: the list is a circle.
        start = bisect.bisect(self._keys, h64(key)) % len(self._keys)
        out: list[str] = []
        for j in range(len(self._keys)):          # walk clockwise, at most one full turn
            o = self._owners[(start + j) % len(self._keys)]
            if o not in out:                      # collect DISTINCT owners: out[0] is the affinity worker,
                out.append(o)                     # out[1] is the spill/retry target, and so on
                if len(out) == n:
                    break
        return out
```

`coordinator/app.py`:

```python
import asyncio, json, logging, os, random, socket, time   # socket: DNS resolution of the headless service; random: "random" routing mode

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ring import Ring

# --- configuration ---
WORKER_DNS = os.environ.get("WORKER_DNS", "worker-hl.default.svc.cluster.local")   # headless Service FQDN: <svc>.<ns>.svc.cluster.local
WORKER_PORT = int(os.environ.get("WORKER_PORT", "8000"))   # the wrapper's port
SPILL_QUEUE = int(os.environ.get("SPILL_QUEUE", "4"))      # spill to the next worker when the affinity worker has >= this many queued
POLL_S = float(os.environ.get("POLL_S", "1.0"))            # how often to re-resolve DNS and re-read /stats
VNODES = int(os.environ.get("VNODES", "150"))              # virtual nodes per worker (sweepable for the balance discussion)

logging.basicConfig(level=logging.INFO, format="%(message)s")   # bare message: we log JSON strings, one per line
log = logging.getLogger("coordinator")
app = FastAPI()
ring = Ring(vnodes=VNODES)
stats: dict[str, dict] = {}       # worker name -> last /stats (+ "url")   ; the only "state" besides the ring
fails: dict[str, int] = {}        # worker name -> consecutive failed polls; 2 in a row = gone
counters = {"routed": 0, "spilled": 0, "random": 0}   # exposed on /workers; the benchmark reads "spilled"


class RouteReq(BaseModel):
    prefix_hash: str
    mode: str = "affinity"        # "affinity" | "random"   (random = the baseline the benchmark compares against)


def resolve() -> set[str]:
    """Membership = the A records of the headless Service. Only READY pods are listed (readiness probe = membership)."""
    try:
        infos = socket.getaddrinfo(WORKER_DNS, WORKER_PORT, type=socket.SOCK_STREAM)   # one tuple per pod IP
    except socket.gaierror:                       # name does not resolve (no ready pods, or wrong namespace)
        return set()
    return {f"http://{sa[0]}:{sa[1]}" for *_, sa in infos}   # sa = (ip, port) -> base URL of each wrapper


async def poll_loop() -> None:
    """Background task: every POLL_S, refresh membership and load. This is the ONLY writer of ring/stats."""
    async with httpx.AsyncClient(timeout=1.0) as client:   # 1 s total per request: /stats must be fast (no I/O in the wrapper's handler)
        while True:
            urls = sorted(resolve())              # sorted for stable log output
            # Fetch every worker's /stats concurrently; return_exceptions=True keeps one dead worker from failing the gather
            results = await asyncio.gather(*(client.get(f"{u}/stats") for u in urls), return_exceptions=True)
            alive: dict[str, dict] = {}
            for u, r in zip(urls, results):
                if isinstance(r, Exception) or r.status_code != 200:
                    continue                      # unreachable or unhealthy this round: not in `alive`
                s = r.json(); s["url"] = u; s["seen"] = time.time()   # remember where to send requests and how fresh this is
                alive[s["name"]] = s              # keyed by the worker's self-reported POD_NAME = its ring identity
            # Departures: a known ring member missing from this poll gets a strike; two strikes (~2 s) removes it.
            for name in list(ring.nodes):
                if name in alive:
                    fails[name] = 0
                else:
                    fails[name] = fails.get(name, 0) + 1
                    if fails[name] >= 2:
                        ring.remove(name); stats.pop(name, None)
                        log.info(json.dumps({"event": "worker_left", "worker": name, "ring": sorted(ring.nodes)}))
            # Arrivals + refresh: update stats for everyone alive; add newcomers to the ring immediately.
            for name, s in alive.items():
                stats[name] = s
                if name not in ring.nodes:
                    ring.add(name); fails[name] = 0
                    log.info(json.dumps({"event": "worker_joined", "worker": name, "ring": sorted(ring.nodes)}))
            await asyncio.sleep(POLL_S)


@app.on_event("startup")
async def _start() -> None:
    asyncio.create_task(poll_loop())              # fire-and-forget background loop on the same event loop as the handlers


@app.post("/route")
def route(req: RouteReq) -> dict:
    """The routing decision. Pure in-memory work: microseconds. Sync handler so it never awaits."""
    if not ring.nodes:
        raise HTTPException(503, "no workers")    # gateway turns this into a 503 to the client
    counters["routed"] += 1
    if req.mode == "random":                      # baseline mode: ignore the prefix entirely
        counters["random"] += 1
        name = random.choice(sorted(ring.nodes))
        return {"worker": name, "url": stats[name]["url"], "spilled": False, "candidates": [name]}
    # Affinity mode: candidates = all workers in clockwise order from the key; [0] is the owner.
    candidates = ring.lookup(req.prefix_hash, n=len(ring.nodes))
    chosen = candidates[0]                        # default: the affinity owner (even if everyone is busy)
    for c in candidates:                          # load-aware spill: first candidate whose queue is under the threshold
        if stats.get(c, {}).get("queue_depth", 0) < SPILL_QUEUE:
            chosen = c
            break
    spilled = chosen != candidates[0]             # True when we traded a probable cache hit for a shorter queue
    counters["spilled"] += spilled                # bool adds as 0/1
    return {"worker": chosen, "url": stats[chosen]["url"], "spilled": spilled, "candidates": candidates}


@app.get("/workers")
def workers() -> dict:                            # debugging/benchmark view: membership, last stats, spill counters
    return {"ring": sorted(ring.nodes), "stats": stats, "counters": counters}
```

`coordinator/Dockerfile` is the worker one with `COPY app.py ring.py /app/` and port 8001. `k8s/coordinator.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: coordinator }
spec:
  replicas: 1                                    # one is enough: it is stateless apart from the ring it rebuilds from DNS in one poll
  selector: { matchLabels: { app: coordinator } }
  template:
    metadata: { labels: { app: coordinator } }
    spec:
      containers:
        - name: coordinator
          image: dkv/coordinator:dev
          imagePullPolicy: IfNotPresent
          env:
            - { name: WORKER_DNS, value: "worker-hl.default.svc.cluster.local" }   # the headless Service from M2, default namespace
            - { name: SPILL_QUEUE, value: "4" }   # spill threshold; sweep it in M6
          ports: [{ containerPort: 8001 }]
          resources: { requests: { cpu: "100m", memory: "128Mi" }, limits: { cpu: "500m", memory: "256Mi" } }   # /route is microseconds of CPU
---
apiVersion: v1
kind: Service
metadata: { name: coordinator }                  # a normal ClusterIP Service: the gateway calls http://coordinator:8001
spec:
  selector: { app: coordinator }
  ports: [{ port: 8001, targetPort: 8001 }]
```

Discovery through the headless Service's DNS is deliberate: no RBAC (Kubernetes' permission system — a service account would need rights to list pods), no Kubernetes client library, and DNS only lists *ready* pods, so readiness probes double as membership. Two consecutive failed polls (≈2 s) remove a worker; one successful poll re-adds it.

### Test

`coordinator/tests/test_ring.py`:

```python
import statistics
from collections import Counter
from ring import Ring

KEYS = [f"prefix-{i}" for i in range(100_000)]    # 100k synthetic prefix hashes: enough for stable fractions


def owners(ring):
    return {k: ring.lookup(k)[0] for k in KEYS}    # key -> affinity owner, for every key


def test_remap_fraction_on_leave():
    ring = Ring(150)
    for n in ["w0", "w1", "w2", "w3"]:
        ring.add(n)
    before = owners(ring)
    ring.remove("w2")                              # one of four workers leaves
    after = owners(ring)
    moved = [k for k in KEYS if before[k] != after[k]]   # keys whose owner changed
    assert all(before[k] == "w2" for k in moved)          # ONLY w2's keys moved
    frac = len(moved) / len(KEYS)
    print(f"moved {frac:.3f} of keys (ideal 0.250)")      # ~1/N with N=4; hash(key) % N would move ~0.75
    assert 0.21 < frac < 0.29


def test_balance_by_vnodes():
    for v in (1, 10, 150):                         # sweep vnode count to SEE why 150
        ring = Ring(v)
        for n in ["w0", "w1", "w2"]:
            ring.add(n)
        c = Counter(owners(ring).values())         # keys owned per worker
        ratio = max(c.values()) / statistics.mean(c.values())   # 1.0 = perfect balance
        print(f"vnodes={v:4d} max/mean={ratio:.2f} counts={dict(c)}")
        if v == 150:
            assert ratio < 1.15                    # within 15% of even at 150 vnodes


def test_lookup_order_is_distinct_and_complete():
    ring = Ring(150)
    for n in ["w0", "w1", "w2"]:
        ring.add(n)
    assert sorted(ring.lookup("anything", n=3)) == ["w0", "w1", "w2"]   # candidates list covers every worker exactly once
```

```bash
cd coordinator && python -m pytest -q -s tests/     # -s: show the print() lines (the numbers you want to read)
```

Then in-cluster: build/load the coordinator image, `kubectl apply -f k8s/coordinator.yaml`, and

```bash
kubectl port-forward svc/coordinator 8001:8001 &
curl -s localhost:8001/workers | python3 -m json.tool | head -20      # ring lists 3 pod names; stats per worker
for i in 1 2 3; do curl -s localhost:8001/route -d '{"prefix_hash":"tpl-7"}' -H 'content-type: application/json'; echo; done   # same worker x3
kubectl scale deploy/worker --replicas=2; sleep 5; curl -s localhost:8001/workers | grep -o '"ring": \[[^]]*\]'   # 2 names after ~2 s
kubectl scale deploy/worker --replicas=3                             # back to 3 (the new pod has a NEW name)
```

### Expected

`moved` between 0.21 and 0.29 (a run with these exact names gives 0.224) — the ideal is 0.250, and the deviation is the randomness of where w2's 150 vnodes fell: with 150 arcs per worker a single worker's share wanders by a few points, which is the residual imbalance vnodes don't fully remove. Balance at 150 vnodes: max/mean 1.03–1.12 (1.07 with these names). At 1 vnode: anywhere from 1.3 to 2.5 (1.87 here, with one worker holding 5% of keys and another 62%) — run it twice with different names and watch it swing; that swing is the argument for vnodes. Same worker returned three times for `tpl-7`; after scaling to 2 the ring lists 2 names within ~3 s, and back to 3 names after scaling up (the new pod has a new name, so it takes different ring positions than the one that left — fine for a Deployment, and the reason M7 moves to a StatefulSet).

### Reading a failure

| Symptom | Cause |
|---|---|
| `moved` ≈ 0.75 or ≈ 1.0 | `remove()` rebuilt the ring in a different order, or you are re-adding all nodes — vnode hashes must be a function of the name only |
| Some keys moved to a node that isn't the removed one's neighbor | `lookup` is not walking clockwise from `bisect` — check the modulo wrap at the end of the list |
| Balance ratio > 1.3 even at 150 vnodes | Hash is weak (using Python's `hash()` — randomized per process and poorly distributed on short strings); use blake2b |
| Ring empty in-cluster | `WORKER_DNS` wrong namespace, or Service is not headless (`clusterIP: None` missing → one virtual IP, not pod IPs) |
| Worker flaps in and out | `/stats` slower than the 1 s client timeout under load — raise timeout, and never do blocking work in the wrapper's `/stats` |
| `spilled` always true | `queue_depth` missing from stats (typo), so the `< SPILL_QUEUE` compare uses 0 — and never spills; if it's always *true*, the threshold is 0 |

### Close the milestone

Commit: `M3: coordinator — consistent-hash ring (150 vnodes, blake2b), DNS membership, load-aware spill; remap test ≈1/N`.
README section: "Routing" — the ring diagram, the prefix definition in one paragraph, the pytest output verbatim, and a short "why not a central table / rendezvous" subsection — write that one carefully, it is the interview answer.

---

## M4 — the gateway (1 day)

### What it is

The single public entry. Everything a client sees comes through here, which makes it the right place for four cross-cutting concerns:

- **SSE passthrough with backpressure.** The gateway opens a streaming request to the worker and yields each SSE line to the client as it arrives. Because it `await`s the next chunk from the worker only after the client has accepted the previous one, a slow client naturally slows the pull from the worker — no unbounded buffer in the middle (that is what backpressure means: the consumer's pace propagates upstream). When the client disconnects, uvicorn cancels the response task, the `async with client.stream(...)` block exits, the worker sees its client go away and cancels its own upstream (M2's chain), and the engine slot is freed. One cancellation propagating through three hops is the thing to test.
- **Per-client token-bucket rate limiting.** A bucket per API key (or client IP): capacity `BURST`, refilled at `RATE` tokens/s, lazily on each request (no background timer: the refill is computed from the elapsed time whenever the bucket is next touched). Reject with 429 and `Retry-After`. Token bucket rather than a fixed window because it allows short bursts while bounding the average, and it is 12 lines.
- **Request IDs and structured logs.** Honor an incoming `X-Request-ID`, else mint one; pass it to the worker; echo it in the response headers; log exactly one JSON line per request at the end with everything you'll need for the p99 story: `ttft_ms`, `route_ms`, `queue_ms`, `prefill_ms`, `hit`, `worker`, `spilled`, `status`.
- **TTFT measured at the gateway.** Clock starts when the request arrives; stops when the first `data:` line with a token leaves. This is the number the benchmark reports. It includes the coordinator hop (~1 ms), worker queueing, and prefill, so `ttft ≈ route + queue + prefill + first decode step`, and the log line has each term.

The prefix hash is computed here, once, and passed downstream — the worker needs it to key its cache table and the coordinator needs it to route. Neither should recompute it.

### Look first

1. **Cancellation across hops, in the source.** Re-open `starlette.responses.StreamingResponse.__call__` (M2's Look first) and this time trace the *outer* side: `listen_for_disconnect` awaits `receive()` until it sees `{"type": "http.disconnect"}` and then cancels the task group. Then `print(inspect.getsource(starlette.requests.Request.is_disconnected))` — the polling alternative for handlers that are not streaming. Ask yourself where in *your* generator the `CancelledError` will surface (answer: at whichever `await`/`yield` is current — usually inside `aiter_lines()`), and confirm that the `finally` in the code below runs on that path.
2. **What a client actually receives.** With the M2 wrapper running locally: `curl -sN -D - localhost:8000/generate ...` (`-D -` dumps response headers to stdout first). Notice `transfer-encoding: chunked` and `content-type: text/event-stream` — the gateway must pass exactly this shape through and add its own headers.
3. **A production token bucket.** Read nginx's `limit_req` docs (the `rate=` and `burst=` parameters and the paragraph on how excess requests are delayed or rejected) — that is a token bucket with the same two knobs as the class below. If you want Python: `pip install limits` and `print(inspect.getsource(limits.strategies.MovingWindowRateLimiter.hit))` for a sibling algorithm; compare the state each keeps (the bucket keeps two floats per client, the moving window keeps a timestamp per hit).
4. **Kubernetes NodePort.** `kubectl explain service.spec.type` and `kubectl explain service.spec.ports.nodePort` — note the default range 30000–32767, which is why kind-config.yaml maps 30080.
5. **NFC normalization.** `python3 -c "import unicodedata; print(len('é'), len(unicodedata.normalize('NFC','é')))"` — two code points become one. That is why the prefix key normalizes before hashing.

### Code

`gateway/app.py`:

```python
import asyncio, hashlib, json, logging, os, time, unicodedata, uuid   # hashlib: prefix key; unicodedata: NFC; uuid: request ids

import httpx
from fastapi import FastAPI, HTTPException, Request    # Request: access to headers and client IP
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- configuration ---
COORD = os.environ.get("COORDINATOR_URL", "http://coordinator:8001")   # the coordinator Service (M3)
RATE = float(os.environ.get("RATE_PER_S", "5"))    # token-bucket refill rate per client (requests/second, long-run average)
BURST = int(os.environ.get("BURST", "20"))         # token-bucket capacity per client (max instantaneous burst)

logging.basicConfig(level=logging.INFO, format="%(message)s")   # one JSON line per request, nothing else
log = logging.getLogger("gateway")
app = FastAPI()
# One pooled client for both the coordinator hop and the worker stream. read=300 s: streams can be long on CPU.
client = httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=300.0))


class ChatReq(BaseModel):
    system: str = ""                                # the template portion (may be empty)
    user: str
    max_tokens: int = 128


def prefix_key(req: ChatReq) -> str:
    """The routing/cache key: the normalized system prompt, hashed. Computed here ONCE, passed to both downstream hops.

    Falls back to the first 256 chars of the user turn when there is no system prompt (otherwise every
    system-less request would hash to the same key and pile onto one worker).
    """
    text = req.system.strip() or req.user[:256]
    norm = unicodedata.normalize("NFC", text).strip()        # canonical Unicode form + trim: 'é' vs 'e'+accent must hash the same
    return hashlib.blake2b(norm.encode(), digest_size=16).hexdigest()   # 32 hex chars; the ring hashes this string again


class TokenBucket:
    """Classic token bucket: `burst` capacity, refilled at `rate` per second, refill computed lazily on each take()."""

    def __init__(self, rate: float, burst: int):
        self.rate, self.burst, self.tokens, self.ts = rate, burst, float(burst), time.monotonic()   # starts full

    def take(self) -> bool:
        now = time.monotonic()                                               # monotonic: immune to wall-clock jumps
        self.tokens = min(self.burst, self.tokens + (now - self.ts) * self.rate)   # add rate*elapsed, capped at burst
        self.ts = now
        if self.tokens >= 1:                                                 # one whole token per request
            self.tokens -= 1
            return True
        return False                                                         # empty: caller returns 429


buckets: dict[str, TokenBucket] = {}                # client key -> bucket. Unbounded in a demo; in prod, expire idle keys.


@app.post("/v1/chat")
async def chat(req: ChatReq, request: Request):
    t0 = time.perf_counter()                        # TTFT clock starts the moment the request arrives at the edge
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]    # honor a caller-supplied id, else mint one
    client_key = request.headers.get("x-api-key") or request.client.host  # rate-limit identity: API key, else source IP
    if not buckets.setdefault(client_key, TokenBucket(RATE, BURST)).take():   # create-on-first-use, then take one token
        raise HTTPException(429, "rate limited", headers={"Retry-After": "1"})   # 429 Too Many Requests + hint to retry in 1 s
    mode = request.headers.get("x-routing", "affinity")   # benchmark switch: "affinity" (the system) vs "random" (baseline)
    ph = prefix_key(req)

    # Hop 1: ask the coordinator which worker. Synchronous, ~1 ms in-cluster; its cost is recorded as route_ms.
    r = await client.post(f"{COORD}/route", json={"prefix_hash": ph, "mode": mode})
    if r.status_code != 200:
        raise HTTPException(503, f"router: {r.text}")   # e.g. "no workers" while the ring is empty
    route = r.json()
    route_ms = (time.perf_counter() - t0) * 1000
    body = {"system": req.system, "user": req.user, "max_tokens": req.max_tokens, "prefix_hash": ph}   # the worker's GenReq

    async def stream():
        # The structured log record for this request; filled in as the stream progresses, written once in `finally`.
        rec = {"rid": rid, "client": client_key, "prefix": ph[:8], "mode": mode, "worker": route["worker"],
               "spilled": route["spilled"], "route_ms": round(route_ms, 1), "ttft_ms": None, "status": "ok"}
        expect_done = False                         # set when we see "event: done"; the NEXT data: line is the done JSON
        try:
            # Hop 2: stream from the worker. Leaving this block closes the upstream socket (cancellation propagates).
            async with client.stream("POST", f"{route['url']}/generate", json=body,
                                     headers={"x-request-id": rid}) as resp:
                if resp.status_code != 200:
                    rec["status"] = f"worker_http_{resp.status_code}"
                    yield f"event: error\ndata: {json.dumps({'error': rec['status'], 'rid': rid})}\n\n"   # tell the client, never hang
                    return
                async for line in resp.aiter_lines():       # one SSE line at a time, in arrival order
                    if line.startswith("event: done"):
                        expect_done = True                  # the worker's summary is coming on the next data: line
                    elif line.startswith("data:"):
                        if expect_done:
                            rec.update(json.loads(line[5:]))   # merge hit/reused/prefill_ms/queue_ms/... into the log record
                        elif rec["ttft_ms"] is None:
                            rec["ttft_ms"] = round((time.perf_counter() - t0) * 1000, 1)   # first token leaves: TTFT stops here
                    yield line + "\n"                       # pass the line through verbatim (aiter_lines stripped the newline)
                    # ^ backpressure: we only loop back to aiter_lines() after the client has accepted this chunk
        except asyncio.CancelledError:                      # client disconnected: Starlette cancelled this generator
            rec["status"] = "client_disconnected"
            raise                                           # MUST re-raise so the cancellation completes upstream
        except httpx.HTTPError as e:                        # connect failure, read timeout, etc. from the worker
            rec["status"] = f"worker_error:{type(e).__name__}"
            yield f"event: error\ndata: {json.dumps({'error': rec['status'], 'rid': rid})}\n\n"
        finally:
            rec["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            log.info(json.dumps(rec))                       # exactly one line per request, on every exit path

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"X-Request-ID": rid,                 # echo the id so clients can correlate
                                      "X-Worker": route["worker"],         # which worker served it (the affinity test reads this)
                                      "Cache-Control": "no-cache"})        # tell proxies not to buffer/cache the stream
```

`k8s/gateway.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: gateway }
spec:
  replicas: 1                                    # one is fine locally; rate-limit buckets are per replica, so >1 needs shared state
  selector: { matchLabels: { app: gateway } }
  template:
    metadata: { labels: { app: gateway } }
    spec:
      containers:
        - name: gateway
          image: dkv/gateway:dev
          imagePullPolicy: IfNotPresent
          env:
            - { name: COORDINATOR_URL, value: "http://coordinator:8001" }   # the coordinator Service name resolves in-namespace
            - { name: RATE_PER_S, value: "50" }     # generous for the benchmark; tighten per client in prod
            - { name: BURST, value: "100" }         # the bench spreads its load over 8 api keys, so 4 req/s never trips this
          ports: [{ containerPort: 8002 }]
          resources: { requests: { cpu: "200m", memory: "128Mi" }, limits: { cpu: "1", memory: "256Mi" } }   # it relays every token; give it a full core
---
apiVersion: v1
kind: Service
metadata: { name: gateway }
spec:
  type: NodePort                                 # expose on every node's IP at a fixed port (kind has no cloud LoadBalancer)
  selector: { app: gateway }
  ports: [{ port: 8002, targetPort: 8002, nodePort: 30080 }]   # 30080 is the port kind-config.yaml maps to localhost:30080
```

The Dockerfile is the worker one with port 8002. Add build/load/apply lines for gateway and coordinator to `kind-up.sh`, and `kubectl rollout status` for each.

### Test

```bash
scripts/kind-up.sh
SYS=$(python3 -c 'print("You are a support agent for Acme. Rules: " + "be brief and polite. " * 50)')   # ~340-token template
body() { python3 -c "import json;print(json.dumps({'system':'''$SYS''','user':'$1','max_tokens':32}))"; }   # helper: JSON body with user=$1

# 1. Streaming + headers   (-D - prints response headers; head -20 shows headers + the first few token frames)
curl -sN -D - localhost:30080/v1/chat -H 'content-type: application/json' -d "$(body 'Hello?')" | head -20

# 2. Affinity: same worker for the same system prompt, 5 times   (-o /dev/null discards the body; grep the X-Worker header)
for i in 1 2 3 4 5; do curl -s -o /dev/null -D - localhost:30080/v1/chat -H 'content-type: application/json' \
  -d "$(body "Question $i")" | grep -i x-worker; done

# 3. Random mode: workers vary   (x-routing: random is the benchmark's baseline switch)
for i in 1 2 3 4 5; do curl -s -o /dev/null -D - localhost:30080/v1/chat -H 'x-routing: random' \
  -H 'content-type: application/json' -d "$(body "Question $i")" | grep -i x-worker; done

# 4. Rate limit: 150 requests as fast as possible from one key   (-w prints the status code; & fires them concurrently)
for i in $(seq 150); do curl -s -o /dev/null -w '%{http_code}\n' localhost:30080/v1/chat \
  -H 'x-api-key: k1' -H 'content-type: application/json' -d "$(body 'x')" & done; wait | sort | uniq -c

# 5. Disconnect propagation   (kill the client after 1 s; every hop must release)
timeout 1 curl -sN localhost:30080/v1/chat -H 'content-type: application/json' -d "$(body 'Write 600 words about rivers.')" >/dev/null
sleep 2; kubectl port-forward svc/coordinator 8001:8001 >/dev/null 2>&1 & sleep 1
curl -s localhost:8001/workers | grep -o '"in_flight": [0-9]*'   # every worker: 0

# 6. Logs
kubectl logs deploy/gateway --tail=3
```

### Expected

Test 1: headers `X-Request-ID`, `X-Worker`, `content-type: text/event-stream`, then `data: {"token": ...}` lines arriving one at a time at CPU decode speed (10–30 tokens/s per stream on 2 threads — a 0.5B Q8_0 model is ~0.5 GB of weights read once per token; two threads of a desktop CPU move 10–25 GB/s, so 20–50 tok/s is the ceiling and you'll see under it — this is the memory-bandwidth-bound decode from the concepts section). Test 2: the same worker name five times. Test 3: at least two different names. Test 4: ~100 `200`s and ~50 `429`s (BURST=100, and at RATE=50/s a burst of 150 sent in well under a second gets roughly 100 + a handful through). Test 5: all workers `in_flight: 0`. Test 6: log lines like

```json
{"rid": "3f9c...", "client": "10.244.0.1", "prefix": "a1b2c3d4", "mode": "affinity", "worker": "worker-6d9f-xk2p1", "spilled": false, "route_ms": 1.8, "ttft_ms": 214.5, "status": "ok", "hit": true, "prompt_tokens": 341, "reused": 322, "prefill_ms": 96.2, "queue_ms": 0.0, "output_tokens": 32, "total_ms": 1830.1}
```

where `ttft_ms ≈ route_ms + queue_ms + prefill_ms + one decode step + ~20 ms of hops`. If the terms don't roughly add up, something is buffering.

### Reading a failure

| Symptom | Cause |
|---|---|
| Client gets all tokens at once at the end | Buffering: `aiter_lines` is fine, but check you are not behind an ingress that buffers, and that the worker's `yield` frames end with `\n\n` |
| `X-Worker` changes for the same system prompt in affinity mode | `prefix_key` sees different bytes (whitespace/newline differences in the shell quoting), or `spilled` is true — check the log line |
| 429 for everything | Client key resolves to the same value for all (behind kind's NodePort every client is `10.244.x.x`); use `X-API-Key` in the benchmark |
| `in_flight` stays > 0 after disconnect | Cancellation didn't propagate: an `except Exception` swallowed `CancelledError` (it is a `BaseException` in 3.8+, but check), or the worker wrapper caught it |
| `ttft_ms` null on successful requests | The first `data:` line was the `done` event (0 output tokens — `max_tokens` 0 or an immediate stop token) |
| `route_ms` > 50 ms | Coordinator is CPU-throttled, or `ring.lookup` is being called with `n=len(nodes)` on a huge ring — fine at 3 workers, revisit at 100 |

### Close the milestone

Commit: `M4: gateway — SSE passthrough with cancel propagation, per-client token bucket, request ids, TTFT logging`.
README section: "Gateway" — the request-lifecycle diagram with the four timestamps, one real log line, and the sentence "TTFT is measured here, not at the engine, so it includes routing and queueing." This is also the day to deploy everything to kind end-to-end; from here on, `scripts/kind-up.sh` is the reproduction command.

---

## M5 — eviction and the KV budget (1 day)

### What it is

In M2 the wrapper *observed* which prefixes were warm; llama-server was choosing slots by its own rule (best longest-common-prefix match, else least-recently-used slot). That is reasonable, but it is not a *budget* — you cannot say "this worker keeps at most 40k tokens of cached prefix" — and you cannot report evictions because you never see them. M5 moves slot ownership into the wrapper:

- A table `prefix_hash → slot id` kept as an `OrderedDict` used as an LRU (move-to-end on hit, pop from the front to evict).
- A **token budget** per worker (`KV_BUDGET_TOKENS`). Each warm entry accounts for its prompt length. Admitting a new prefix evicts from the LRU end until both a slot is free *and* the budget holds. For this model the slot count is the binding constraint (8 slots × ~600 tokens ≈ 5k tokens ≈ 60 MB); set the budget to something like 3000 tokens in one benchmark run to *force* eviction and show the hit-rate-vs-budget curve. The budget in bytes is `tokens × 2 × layers × kv_heads × head_dim × dtype_bytes` (the KV tensor is `[n_layers, 2 (K,V), tokens, n_kv_heads, head_dim]` of `dtype_bytes` each), which for Qwen2.5-0.5B is 12 KB/token and for Llama-3-8B is 128 KB/token — write that mapping in the README, it is the Module 3 formula applied to your own system.
- Evict *idle* slots first. A slot with a request in flight is pinned; llama-server would queue a new request that names a busy `id_slot` until it frees, so evicting a busy one just adds latency. Because the semaphore is sized to the slot count and you admit only after acquiring it, at least one slot is always idle when you look.
- Metrics: hits, misses, evictions, entries, used/budget tokens.

**How this maps onto llama-server.** Pinning `id_slot` (a `/completion` request field that names which slot must serve it) makes the wrapper's table the truth: the slot's KV *is* that prefix until the wrapper reassigns it. When a request lands on a slot with a different prompt, llama-server discards the mismatched suffix (from the first differing token) and prefills the rest — for a different prefix that is essentially a full prefill, i.e. exactly what an eviction should cost. There is a second tier available for free: `--slot-save-path` enables `POST /slots/{id}?action=save` and `?action=restore` with a filename, which writes/reads the slot's KV to disk (~12 KB/token, so ~6 MB for a 500-token prefix — milliseconds on an SSD versus seconds of CPU prefill). Wiring save-on-evict / restore-on-miss is a stretch goal with a clear graph ("miss cost: recompute vs restore from disk"); skip it until M6 is published.

### Look first

1. **`id_slot` and slot actions in the README.** In llama.cpp's `tools/server/README.md`, find `id_slot` under `POST /completion` ("Assign the completion task to a specific slot"), then the `POST /slots/{id_slot}?action=save|restore|erase` section: note the `filename` body field, that files land under `--slot-save-path`, and the response JSON (`n_saved`/`n_restored` tokens, `n_written` bytes, timings). That response is your "restore from disk" cost measurement if you do the stretch goal.
2. **Watch a pinned slot.** With the local llama-server from M2 running:

   ```bash
   # Pin the request to slot 3 and then read the slot table: slot 3 is the one that now holds this prompt
   curl -s localhost:8080/completion -d '{"prompt":"<|im_start|>system\nTemplate A. Be brief.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n","n_predict":4,"cache_prompt":true,"id_slot":3}' | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["id_slot"], r["tokens_evaluated"], r["timings"]["prompt_n"])'
   curl -s localhost:8080/slots | python3 -c 'import sys,json; [print(s["id"], s["is_processing"], s.get("n_ctx"), str(s.get("prompt",""))[:40]) for s in json.load(sys.stdin)]'
   ```

   Now send a *different* system prompt to the same `id_slot: 3` and read `prompt_n`: it is ≈ the whole prompt — the slot's old KV was discarded from the first differing token. That is the cost of an eviction, measured.
3. **Save/restore round trip** (only if `--slot-save-path` was passed):

   ```bash
   curl -s -X POST 'localhost:8080/slots/3?action=save' -d '{"filename":"a.bin"}' | python3 -m json.tool     # n_saved, n_written, timings
   curl -s -X POST 'localhost:8080/slots/3?action=erase'                                                     # slot 3 now cold
   curl -s -X POST 'localhost:8080/slots/3?action=restore' -d '{"filename":"a.bin"}' | python3 -m json.tool  # n_restored + ms: compare with prompt_ms
   ```
4. **The LRU primitive.** `python3 -c "from collections import OrderedDict; help(OrderedDict.popitem)"` — `last=False` pops the *first* (oldest) item. Then the 6-line experiment: insert a, b, c; `move_to_end('a')`; `popitem(last=False)` returns `b`. That is the entire eviction policy.

### Code

Replace the `warm`/`record_cache` parts of `worker/app.py` with a slot LRU and pin `id_slot`:

```python
KV_BUDGET = int(os.environ.get("KV_BUDGET_TOKENS", "8000"))   # per-worker cap on cached prefix tokens; 8000 > 8 slots x ~600 so slots bind first


class SlotLRU:
    """prefix_hash -> engine slot. LRU over prefixes, bounded by slot count and a token budget."""

    def __init__(self, n_slots: int, budget_tokens: int):
        self.free = list(range(n_slots))          # slot ids not currently assigned to any prefix
        self.entries: "OrderedDict[str, dict]" = OrderedDict()   # ph -> {"slot", "tokens"}   ; order = LRU (oldest first)
        self.busy: set[str] = set()               # prefixes with a request in flight: pinned, never evicted
        self.budget, self.used, self.evictions = budget_tokens, 0, 0   # used = sum of tokens over entries

    def acquire(self, ph: str, est_tokens: int) -> tuple[int, bool]:
        """Return (slot, was_warm). Evicts idle LRU entries as needed."""
        if ph in self.entries:                    # warm: this prefix already owns a slot
            self.entries.move_to_end(ph)          # LRU touch: most recently used goes to the end
            self.busy.add(ph)                     # pin while the request runs
            return self.entries[ph]["slot"], True
        # Cold: make room. Loop until a slot is free AND the budget holds (or nothing evictable remains).
        while not self.free or (self.used + est_tokens > self.budget and self.entries):
            victim = next((k for k in self.entries if k not in self.busy), None)   # oldest IDLE entry (dict order = LRU)
            if victim is None:
                break                                    # everything busy; take a free slot if any
            e = self.entries.pop(victim)
            self.free.append(e["slot"]); self.used -= e["tokens"]; self.evictions += 1   # return the slot, release its tokens
        slot = self.free.pop()                    # guaranteed non-empty: semaphore == slot count, so >= 1 slot is idle/free
        self.entries[ph] = {"slot": slot, "tokens": est_tokens}   # provisional size; corrected in release()
        self.used += est_tokens
        self.busy.add(ph)
        return slot, False

    def release(self, ph: str, actual_tokens: int) -> None:
        """Unpin after the request and replace the estimated token count with the engine-reported one."""
        self.busy.discard(ph)
        if ph in self.entries:                    # may have been evicted meanwhile only if it was idle -> it wasn't (busy); guard anyway
            self.used += actual_tokens - self.entries[ph]["tokens"]   # adjust the running total by the estimate error
            self.entries[ph]["tokens"] = actual_tokens


lru = SlotLRU(N_SLOTS, KV_BUDGET)
```

Inside `stream()` in `/generate`, after acquiring the semaphore, the flow becomes: estimate the prompt size, acquire a slot, pin it with `payload["id_slot"]`, and always release in `finally`. Here is the complete M5 version of `generate()` so the pieces are in context (everything that is not marked `# M5` is unchanged from M2):

```python
@app.post("/generate")
async def generate(req: GenReq):
    async def stream():
        t_arrive = time.perf_counter()
        state["waiting"] += 1
        async with slots:                                     # one permit per engine slot (M2)
            state["waiting"] -= 1
            state["in_flight"] += 1
            queue_ms = (time.perf_counter() - t_arrive) * 1000
            payload = {"prompt": render(req.system, req.user), "stream": True, "cache_prompt": True,
                       "n_predict": req.max_tokens, "stop": ["<|im_end|>"], "temperature": 0.7}
            est = len(req.system) // 4 + 8                    # M5: ~4 chars/token estimate of the prefix size; corrected on release
            slot, warm_slot = lru.acquire(req.prefix_hash, est)   # M5: pick (or evict for) a slot; warm_slot = our prediction of a hit
            payload["id_slot"] = slot                         # M5: pin the engine slot -> the wrapper's table is the truth
            total = 0                                         # M5: prompt tokens as reported by the engine (0 until the stop frame)
            try:
                async with client.stream("POST", "/completion", json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        ev = json.loads(line[6:])
                        if ev.get("content"):
                            yield f"data: {json.dumps({'token': ev['content']})}\n\n"
                        if ev.get("stop"):                    # final frame with timings
                            tm = ev.get("timings", {})
                            total = ev.get("tokens_evaluated", 0)
                            reused = max(total - tm.get("prompt_n", total), 0)
                            hit = total > 0 and reused / total >= 0.5    # M5: hit/miss still MEASURED from the engine's prefill count
                            state["hits" if hit else "misses"] += 1
                            done = {"worker": NAME, "hit": hit, "prompt_tokens": total, "reused": reused,
                                    "prefill_ms": tm.get("prompt_ms"), "queue_ms": round(queue_ms, 1),
                                    "output_tokens": ev.get("tokens_predicted"),
                                    "slot": slot,                          # M5: which engine slot served it
                                    "expected_warm": warm_slot}            # M5: the LRU's prediction; compare with `hit` in the bench
                            yield f"event: done\ndata: {json.dumps(done)}\n\n"
            finally:
                lru.release(req.prefix_hash, total or est)    # M5: unpin; replace the estimate with the real token count (or keep est on error)
                state["in_flight"] -= 1
    return StreamingResponse(stream(), media_type="text/event-stream")


def snapshot() -> dict:
    """M5 /stats: the LRU is now the source of truth for entries, evictions and budget use."""
    total = state["hits"] + state["misses"]
    return {"name": NAME, "slots": N_SLOTS, "in_flight": state["in_flight"], "queue_depth": state["waiting"],
            "cache_entries": len(lru.entries),                # warm prefixes currently owning a slot
            "hits": state["hits"], "misses": state["misses"],
            "evictions": lru.evictions,                       # counted by the LRU, not guessed
            "used_tokens": lru.used, "budget_tokens": lru.budget,   # KV budget occupancy (multiply by 12 KB/token for bytes)
            "hit_rate": round(state["hits"] / total, 3) if total else None,
            "warm": list(lru.entries)}                        # the worker's warm set: a hint a hybrid router could use
```

Delete `warm` and `record_cache`. Keep `expected_warm` in the done event: when it disagrees with the measured `hit`, the wrapper's model of the engine is wrong, and the benchmark's `expected_warm != hit` count is your sanity check that the LRU is real.

### Test

A local test that forces eviction with a tiny budget:

```bash
# Engine as before
docker run --rm -p 8080:8080 -v $PWD/worker/models:/models ghcr.io/ggml-org/llama.cpp:server \
  -m /models/qwen2.5-0.5b-instruct-q8_0.gguf --host 0.0.0.0 --port 8080 -c 8192 -np 8 -t 4 &
# Wrapper with a 900-token budget: room for two ~420-token prefixes, not three
cd worker && N_SLOTS=8 KV_BUDGET_TOKENS=900 uvicorn app:app --port 8000 &

# Three different ~400-token system prompts, cycled A B C A B C: budget fits 2 → A is evicted before it comes back
python3 - <<'EOF'
import json, httpx, hashlib
def sysm(i): return f"Template {i}. " + f"Rule {i}: answer precisely and briefly. " * 45    # distinct from the first word
for tag in "ABCABC":
    s = sysm(tag); ph = hashlib.blake2b(s.encode(), digest_size=16).hexdigest()           # same key derivation as the gateway
    with httpx.stream("POST", "http://localhost:8000/generate", timeout=120,
                      json={"system": s, "user": "hi", "max_tokens": 4, "prefix_hash": ph}) as r:
        for line in r.iter_lines():
            if line.startswith("data:") and '"worker"' in line:                            # the done event (the only frame with "worker")
                d = json.loads(line[5:]); print(tag, "hit" if d["hit"] else "MISS", "slot", d["slot"], "reused", d["reused"])
print(httpx.get("http://localhost:8000/stats").json())
EOF
```

### Expected

`A MISS, B MISS, C MISS, A MISS, B MISS, C MISS` with `evictions: 4` and `cache_entries: 2` — with a 900-token budget and ~420-token prompts only two fit, so cycling three is the classic LRU-worst-case pattern and every access misses. Now rerun with `KV_BUDGET_TOKENS=8000`: `A MISS, B MISS, C MISS, A hit, B hit, C hit`, `evictions: 0`, `cache_entries: 3`. The `slot` numbers should show reuse: the second `A` lands on a slot that a previous entry vacated. `reused` on a hit is the prompt length minus a few tokens, as in M2.

### Reading a failure

| Symptom | Cause |
|---|---|
| `expected_warm: true` but `hit: false` | The slot was reused by llama-server for something else — `id_slot` not in the payload, or a request without `id_slot` (health check? a stray curl) grabbed it |
| `hit: true` on `expected_warm: false` | Coincidence of LCP (two templates share a long opening) — harmless, but it means your templates are less distinct than you think; check M6's generator |
| `evictions` climbs even with a large budget | `est` far above real token count (system prompt full of long words) — the estimate only matters until `release` corrects it; check `release` runs in `finally` |
| Deadlock: requests wait forever | `free` is empty and every entry is busy — impossible if semaphore == slot count; check `N_SLOTS` matches `-np` |
| `used_tokens` drifts negative | `release` called twice or with the wrong token count on error paths |

### Close the milestone

Commit: `M5: worker owns slot assignment — LRU over prefixes with a token budget, eviction metrics, id_slot pinning`.
README section: "Eviction & memory" — the LRU-worst-case output, the tokens→bytes formula with the 0.5B and 8B numbers, and one paragraph on llama-server's own slot policy versus the explicit one and why explicit is required for a budget.

---

## M6 — the benchmark (1.5 days; this is what the resume needs)

### What it is

One workload, two routing modes, several load levels, a CSV, a plot. The claim you will make is "affinity routing lowered TTFT p50 by X% and p95 by Y% at Z req/s, with a cache hit rate of H% vs R%", and every word of it must come out of `bench/run.sh`.

**Workload.** 20 system-prompt templates of ~500 tokens each (generated deterministically so the repo doesn't need a data file), a pool of short user turns, `max_tokens` 64. Template popularity is either uniform or Zipf (`--zipf 1.1`; Zipf = a few templates get most of the traffic, probability ∝ 1/rank^1.1) — real products have a few hot prompts, and Zipf is what makes the load-aware spill matter. Arrivals are **open-loop Poisson**: inter-arrival times drawn from `expovariate(rate)` (the exponential distribution — the gap between independent random events with a given average rate), requests fired regardless of whether previous ones finished. This is the right model for a service (users don't wait for each other) and it is what exposes queueing; a closed-loop "N concurrent clients" generator hides overload because clients slow down with the server. By Little's law the two are related — concurrency ≈ rate × mean latency — so "several concurrency levels" is a sweep over `--rate`.

**Measurements**, all client-side: TTFT (first `data:` token line), total latency, output tokens, plus `hit`, `worker`, `spilled`, `queue_ms`, `prefill_ms` from the `done` event. Percentiles per (mode, rate). Throughput as completed requests/s and output tokens/s.

**Protocol.** 30 s warm-up (discarded — the first touch of every template is a miss in both modes and would flatter neither), then 120 s measured, for each rate in `{0.5, 1, 2, 4}` req/s, affinity then random, restarting nothing between runs. Fix the RNG seed. Run the whole matrix twice on different days; if the deltas move by more than ~10 points, your machine is noisy (Windows updates, thermal throttling, WSL memory reclaim) and you report the range.

**What to expect and why.** On a miss, prefill covers ~540 tokens; on a hit, ~40. Prefill time is proportional to tokens, so a hit saves roughly `500 / prefill_tok_per_s` seconds — 1.2 to 5 s on 2 CPU threads. Hit TTFT is dominated by the 40-token user turn plus one decode step: 0.15–0.5 s. So at low load, where queueing is nil, TTFT p50 in affinity mode should be 60–85% below random mode. Random routing isn't 0% hits: 3 workers × 8 slots = 24 slots and 20 templates, so each worker eventually holds ~8 of the 20, and a random pick hits ~40% of the time; affinity gets ~95%+ (the misses being spills and the warm-up tail). That difference in hit rate, times the per-hit saving, *is* the TTFT delta — and it is why the number is what it is. As rate rises, queueing time grows in both modes and dilutes the ratio; at saturation (all workers' queues non-empty most of the time) the delta compresses toward whatever fraction of total wait prefill still is. Affinity mode also saturates *later*, because each request costs less CPU — that shows up as higher throughput at the same p95, which is your second graph. Expect p95/p99 deltas to be smaller than p50 and noisier; report them honestly.

### Look first

1. **A production load generator's arrival loop.** Open vLLM's `benchmarks/benchmark_serving.py` and find `get_request` (an async generator). Read how it draws the inter-arrival gap — `np.random.gamma(shape=burstiness, scale=1/request_rate)`, which with `burstiness=1` *is* the exponential distribution — and note that it `await asyncio.sleep(interval)` and then yields the next request without waiting for the previous one's response: open loop. Then find where `ttft` is computed in the request function (first chunk with content) — the same "first data: line" rule you use. That file is the reference your `loadgen.py` is a 90-line cousin of.
2. **The exponential draw itself.** `python3 -c "import random, statistics; random.seed(7); xs=[random.expovariate(2.0) for _ in range(10000)]; print(statistics.mean(xs), statistics.median(xs))"` — mean 0.5 s (= 1/rate), median ~0.35 s: most gaps are short and a few are long, which is where the bursts come from.
3. **Percentiles.** `python3 -c "import statistics; help(statistics.quantiles)"` — the stdlib's version; note the `method` argument. The scripts below use the simpler "sorted list, index int(q·n)" rule; with n ≈ 200 the difference between methods is one sample. Look at a *histogram* of one CSV's `ttft_ms` column once (`python3 -c "..."` with `numpy.histogram` or just `sort -n | uniq -c` on rounded values) and see the two bumps: hits and misses. The percentiles summarize that bimodal shape, which is why p50 and p95 tell different stories.
4. **Coordinator counters before and after.** `curl -s localhost:8001/workers | python3 -c 'import sys,json; print(json.load(sys.stdin)["counters"])'` — you will diff this across the Zipf run to count spills.

### Code

`bench/loadgen.py`:

```python
import argparse, asyncio, csv, hashlib, json, random, time   # csv: one row per request; random: seeded arrivals + template choice
import httpx

# A small pool of short user turns (~10 tokens): they differ per request so the hit is ONLY the system-prompt prefix
USER_TURNS = ["Summarize your rules in one sentence.", "What can you help with?", "Give me a two-line example.",
              "Answer yes or no: are you an assistant?", "List three things you must not do.",
              "Rephrase rule two.", "Who are you?", "Explain your first rule briefly."]


def make_templates(n: int = 20, words: int = 380) -> list[str]:
    """~500 tokens each, deterministic, mutually distinct from the first sentence."""
    out = []
    for i in range(n):
        rng = random.Random(1000 + i)             # per-template seed: template i is identical on every run and machine
        rules = [f"Rule {j}: {rng.choice(['always', 'never', 'sometimes'])} {rng.choice(['cite', 'summarize', 'verify', 'escalate', 'translate'])} "
                 f"{rng.choice(['customer', 'invoice', 'shipment', 'ticket', 'contract'])} details before replying."
                 for j in range(1, 60)]            # 59 rules of ~9 words = plenty of text to trim to `words`
        text = f"You are assistant #{i} for department {rng.choice(['billing', 'legal', 'ops', 'sales'])}. " + " ".join(rules)
        out.append(" ".join(text.split()[:words]))   # trim to `words` words (~1.3 tokens/word -> ~500 tokens)
    return out


def phash(s: str) -> str:                         # short prefix id for logs only; the gateway computes the real key
    return hashlib.blake2b(s.strip().encode(), digest_size=16).hexdigest()[:8]


async def one(client, url, tpl_id, system, user, mode, rid, rows, t_start):
    """Send one request, stream it, record one row. Never raises: errors become a status string."""
    t0 = time.perf_counter(); ttft = None; ntok = 0; done = {}; status = "ok"
    try:
        async with client.stream("POST", url, json={"system": system, "user": user, "max_tokens": 64},
                                 headers={"x-routing": mode,                    # affinity vs random baseline
                                          "x-api-key": f"bench-{rid % 8}"}) as r:   # spread over 8 rate-limit buckets
            if r.status_code != 200:
                status = f"http_{r.status_code}"          # 429 = rate limited, 503 = no workers
            else:
                expect_done = False
                async for line in r.aiter_lines():
                    if line.startswith("event: done"):
                        expect_done = True                # next data: line is the worker's summary
                    elif line.startswith("event: error"):
                        status = "error"                  # gateway surfaced a worker failure mid-stream
                    elif line.startswith("data:"):
                        if expect_done:
                            done = json.loads(line[5:])   # hit / worker / queue_ms / prefill_ms / slot / expected_warm
                        else:
                            if ttft is None:
                                ttft = time.perf_counter() - t0   # first token frame: client-side TTFT
                            ntok += 1
    except Exception as e:                                # timeouts, connection resets: recorded, not fatal
        status = type(e).__name__
    rows.append({"mode": mode, "rid": rid, "t": round(t0 - t_start, 3), "template": tpl_id, "status": status,
                 "ttft_ms": round(ttft * 1000, 1) if ttft else None,
                 "total_ms": round((time.perf_counter() - t0) * 1000, 1), "tokens": ntok,
                 "hit": done.get("hit"), "worker": done.get("worker"), "queue_ms": done.get("queue_ms"),
                 "prefill_ms": done.get("prefill_ms")})   # t = send time relative to run start, for the warm-up cut and chaos buckets


async def run(a):
    random.seed(a.seed)                           # same arrival times and template sequence for every mode -> fair comparison
    tpls = make_templates()
    # Zipf weights: template i (0-based) gets weight 1/(i+1)^s; None -> uniform
    weights = [1 / (i + 1) ** a.zipf for i in range(len(tpls))] if a.zipf else None
    rows, tasks = [], []
    t_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=300.0)) as client:   # generous read timeout: queued requests take long at 4 req/s
        rid = 0
        while time.perf_counter() - t_start < a.warmup + a.duration:
            await asyncio.sleep(random.expovariate(a.rate))   # Poisson arrivals: exponential gaps with mean 1/rate
            i = random.choices(range(len(tpls)), weights=weights)[0]   # pick a template (uniform or Zipf)
            # OPEN LOOP: create_task fires and forgets; we do not wait for this request before scheduling the next
            tasks.append(asyncio.create_task(one(client, a.url, i, tpls[i], random.choice(USER_TURNS), a.mode, rid, rows, t_start)))
            rid += 1
        await asyncio.gather(*tasks)              # drain: let every in-flight request finish before summarizing
    measured = [r for r in rows if r["t"] >= a.warmup]   # drop the warm-up window (first touch of every template is a miss)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(measured)
    ok = [r for r in measured if r["status"] == "ok" and r["ttft_ms"]]   # only successful requests with a first token count
    ttfts = sorted(r["ttft_ms"] for r in ok)
    hits = sum(1 for r in ok if r["hit"])
    p = lambda q: ttfts[min(int(q * len(ttfts)), len(ttfts) - 1)] if ttfts else None   # percentile by sorted index
    print(f"{a.mode:8s} rate={a.rate:4.1f} n={len(ok):4d} err={len(measured) - len(ok):3d} "
          f"ttft p50={p(0.5)} p95={p(0.95)} p99={p(0.99)} hit={hits / max(len(ok), 1):.2f} "
          f"req/s={len(ok) / a.duration:.2f} tok/s={sum(r['tokens'] for r in ok) / a.duration:.1f}")   # one summary line per run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:30080/v1/chat")     # the gateway NodePort
    ap.add_argument("--mode", choices=["affinity", "random"], default="affinity")   # sent as the x-routing header
    ap.add_argument("--rate", type=float, default=1.0, help="requests/s, Poisson")   # offered load (the x-axis)
    ap.add_argument("--duration", type=float, default=120)                  # measured window, seconds
    ap.add_argument("--warmup", type=float, default=30)                     # discarded window, seconds
    ap.add_argument("--zipf", type=float, default=0.0, help="0 = uniform; 1.1 = skewed")   # template popularity skew
    ap.add_argument("--seed", type=int, default=7)                          # fixed so runs are comparable
    ap.add_argument("--out", required=True)                                 # CSV path
    asyncio.run(run(ap.parse_args()))
```

`bench/run.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"                                     # run from bench/ so relative paths work
TAG=${1:-$(date +%Y%m%d-%H%M)}; mkdir -p results/$TAG    # results/<tag>/ holds every CSV, the table and the plot for one matrix
for rate in 0.5 1 2 4; do                                # the load sweep (req/s)
  for mode in affinity random; do                        # the system vs the baseline, at the same load, back to back
    # ZIPF env var (default 0 = uniform) lets `ZIPF=1.1 bench/run.sh tag` produce the skewed matrix; tee appends every summary line
    python3 loadgen.py --mode $mode --rate $rate --zipf ${ZIPF:-0} --out results/$TAG/${mode}_r${rate}.csv | tee -a results/$TAG/summary.txt
  done
done
python3 summarize.py results/$TAG > results/$TAG/table.md   # markdown table for the README
python3 plot.py results/$TAG                                 # ttft_vs_load.png for the README
echo "results in results/$TAG"
```

`bench/summarize.py` (percentiles per file into a markdown table):

```python
import csv, glob, os, sys

def pct(xs, q):
    """Percentile by sorted index: the value below which a fraction q of samples fall."""
    xs = sorted(xs); return xs[min(int(q * len(xs)), len(xs) - 1)] if xs else float("nan")

d = sys.argv[1]; rows = {}
for f in sorted(glob.glob(os.path.join(d, "*_r*.csv"))):          # one CSV per (mode, rate), named <mode>_r<rate>.csv
    mode, rate = os.path.basename(f)[:-4].split("_r")             # parse mode and rate back out of the filename
    rs = [r for r in csv.DictReader(open(f)) if r["status"] == "ok" and r["ttft_ms"]]   # successful rows only
    t = [float(r["ttft_ms"]) for r in rs]
    rows[(float(rate), mode)] = dict(n=len(rs), p50=pct(t, .5), p95=pct(t, .95), p99=pct(t, .99),
                                     hit=sum(r["hit"] == "True" for r in rs) / max(len(rs), 1))   # CSV stores bools as text
print("| rate | mode | n | TTFT p50 | p95 | p99 | hit rate | p50 delta |\n|---|---|---|---|---|---|---|---|")
for (rate, mode), v in sorted(rows.items()):
    base = rows.get((rate, "random"), v)["p50"]                    # the random-mode p50 at the same rate is the baseline
    delta = f"{(1 - v['p50'] / base) * 100:+.0f}%" if mode == "affinity" else ""   # negative = affinity is faster; this is the resume number
    print(f"| {rate} | {mode} | {v['n']} | {v['p50']:.0f} | {v['p95']:.0f} | {v['p99']:.0f} | {v['hit']:.2f} | {delta} |")
```

`bench/plot.py`:

```python
import csv, glob, os, sys
import matplotlib; matplotlib.use("Agg")          # headless backend: write a PNG, never open a window (WSL has no display)
import matplotlib.pyplot as plt

d = sys.argv[1]; series = {}                      # mode -> list of (rate, p50, p95)
for f in glob.glob(os.path.join(d, "*_r*.csv")):
    mode, rate = os.path.basename(f)[:-4].split("_r")
    t = sorted(float(r["ttft_ms"]) for r in csv.DictReader(open(f)) if r["status"] == "ok" and r["ttft_ms"])
    series.setdefault(mode, []).append((float(rate), t[len(t) // 2], t[int(0.95 * len(t))]))   # median and p95 by index
fig, ax = plt.subplots(figsize=(7, 4))
for mode, pts in series.items():
    pts.sort()                                    # by rate, so the line is drawn left to right
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", label=f"{mode} p50")    # solid: p50
    ax.plot([p[0] for p in pts], [p[2] for p in pts], "--", label=f"{mode} p95")    # dashed: p95
ax.set_xlabel("offered load (req/s, Poisson)"); ax.set_ylabel("TTFT (ms)"); ax.set_yscale("log")   # log y: hits (~200 ms) and misses (~3 s) on one axis
ax.set_title("TTFT vs load — prefix-affinity routing ON vs OFF (3 CPU workers, Qwen2.5-0.5B Q8_0)")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(d, "ttft_vs_load.png"), dpi=130)   # the README graph
```

### Test

```bash
scripts/kind-up.sh                      # fresh cluster state; nothing warm
pip install matplotlib httpx            # bench dependencies on the host
chmod +x bench/run.sh && bench/run.sh v1   # the full 8-run matrix: ~20 minutes (8 x 150 s)
cat bench/results/v1/table.md
```

Then one skewed run for the spill discussion: `ZIPF=1.1 bench/run.sh v1-zipf` and compare `spilled` counts from the coordinator's `/workers` counters before and after.

### Expected

A table shaped like this (numbers illustrative of the *shape* only — yours will differ, and yours are the ones that go on the resume):

| rate | mode | TTFT p50 | p95 | hit rate | p50 delta |
|---|---|---|---|---|---|
| 0.5 | random | ~1500–4000 ms | higher | ~0.35–0.45 | |
| 0.5 | affinity | ~200–600 ms | ~1000–3000 (the spills + misses) | ~0.9–1.0 | −60% to −85% |
| 4 | random | queue-dominated, several s | | | |
| 4 | affinity | lower but climbing | | ~0.85–0.95 | −30% to −60% |

The p50 delta should be large at low load and shrink with load; the hit-rate columns explain it. If the affinity hit rate is under 0.85 at low load, look at the coordinator's spill counter and at `expected_warm != hit` — either spilling is too eager (raise `SPILL_QUEUE`) or something is stealing slots. `err` should be 0 at all rates; a nonzero count at 4 req/s means a worker's read timeout is being hit under queueing — that is a finding, not a bug, and belongs in the README as the saturation point.

The README graph is `ttft_vs_load.png`: x = offered load, y = TTFT on a log axis, two solid p50 lines and two dashed p95 lines. The gap between the solid lines is the resume number. Add the table under it, the exact command that produced both, and the machine spec (CPU model, cores given to WSL, `-t 2` per worker).

**Writing the bullet.** Take the p50 delta at the load level closest to where random-mode p95 is still under ~5 s (i.e., before either mode is queue-dominated) and say exactly that:

> Rebuilt the KV-cache-aware routing layer solo: gateway → consistent-hash coordinator → llama.cpp workers on Kubernetes; prefix-affinity routing cut TTFT p50 by X% (p95 by Y%) vs random routing at Z req/s on 3 CPU workers, with cache hit rate H% vs R%; LRU eviction under a per-worker KV budget; HPA on queue depth; reproducible via `bench/run.sh`.

If X is 42, write 42. If it is 71, write 71 and expect the question "why so large?" — the answer is the arithmetic above (prefill share of TTFT on CPU is huge), and the M8 section is where you show it shrinks on a GPU.

### Reading a failure

| Symptom | Cause |
|---|---|
| Both modes similar, both hit rates high | Too few templates for the slot count (random routing hits often when 24 slots ≥ 20 templates); run with `-np 4` per worker or 40 templates to see the modes separate; document both configurations |
| Both modes similar, both hit rates low | Prefix hash differs per request — a template is being regenerated non-deterministically, or `user` leaked into `prefix_key` because `system` is empty |
| Affinity hit rate high but TTFT delta small | Prefill isn't the dominant TTFT term — queueing is; lower the rate, or check `queue_ms` in the CSV |
| p99 in affinity mode *worse* than random | Hot-template pile-up on one worker and spill threshold too high; run the Zipf variant and lower `SPILL_QUEUE` |
| Numbers vary 2× between identical runs | Thermal throttling or another process; check `-t` × workers ≤ cores, close Chrome, run again at night |
| `err` > 0 at low rates | Gateway rate limit hit by the bench (`RATE_PER_S`/`BURST` too low for the sweep), showing up as `http_429` in the CSV |

### Close the milestone

Commit: `M6: benchmark harness — Poisson loadgen over 20 templates, affinity vs random sweep, table + plot; results/v1`.
README section: the top of the README — problem statement, the graph, the table, the bullet-shaped sentence with your numbers, and "what surprised me" (typical candidates: the second hit-rate dip after a worker returns, the spill/hit tradeoff under Zipf, the p95 tail being spills). Then rewrite the resume bullet and ship resume v2.

---

## M7 — StatefulSets, autoscaling on queue depth, chaos (week 3, 2 days)

### What it is

**StatefulSet.** With a Deployment, a restarted worker gets a new name and therefore new ring positions: its old keys had already remapped to its neighbors, and now they remap *again* to the newcomer. With a StatefulSet (the Kubernetes controller that gives pods stable ordinal names — `worker-0`, `worker-1`, … — and brings a replaced pod back under the same name), `worker-1` comes back as `worker-1`, its 150 vnodes land in the same places, and its keys return to it — cold, but at least the remap happens once, and scaling is ordinal (scale-down always removes the highest index, so the set of moved keys is predictable). The headless Service already exists; the StatefulSet just names it.

**HPA on a custom metric.** CPU% is the wrong signal for an inference worker: a worker with 8 requests queued and one with 0 both show ~100% CPU on this hardware. Queue depth is the signal (Module 8: queue depth, KV utilization, concurrency — never CPU). The path from "gauge in my `/metrics`" to "HPA decision" needs Prometheus to scrape it (pull it over HTTP every few seconds) and something to expose it to the HPA. Two standard options: `prometheus-adapter` (implements the custom-metrics API; more YAML, more moving parts) or **KEDA** (a `ScaledObject` with a Prometheus query; it creates the HPA for you). Use KEDA. The query is `sum(worker_queue_depth)` with `threshold: "3"`: KEDA sets desired replicas to `ceil(metric / threshold)`, i.e. "add a worker whenever average queued requests per worker would exceed 3". Scale-down gets a 5-minute stabilization window because every scale event remaps ~1/N of the prefixes and costs a wave of cold prefills — autoscaling churns your cache, and the README should say so with a number.

**Chaos test.** Kill a worker during the benchmark and describe exactly what happens, in order: in-flight streams on that worker break (the gateway emits `event: error` and closes — never a silent hang); the coordinator misses two polls (~2 s) and removes it from the ring; the ~1/3 of prefixes it owned now route to their next-clockwise workers, cold, so hit rate dips; the pod restarts (image is local, model load 1–3 s, readiness 2 s), DNS lists it, the coordinator re-adds it with the same name, the keys come *back* — cold again — and hit rate dips a second time. Two dips, not one. Recovery is complete when each of its ~7 templates has been prefilled once more.

**Retry policy.** Retry a failed worker call only if *no bytes have been sent to the client*. Before the first token the request is idempotent from the client's point of view (retrying it changes nothing the client has seen); after it, a retry would replay tokens. So: on connect failure or a non-200 before streaming, try the next candidate from the coordinator's list (it is already in the route response), at most once, and mark the log line `retried: true`. Mid-stream failures are surfaced as an SSE `error` event and left to the client.

### Look first

1. **StatefulSet vs Deployment, in the API docs.** `kubectl explain statefulset.spec` — read `serviceName` (the headless Service that provides the per-pod DNS), `podManagementPolicy` (`OrderedReady` vs `Parallel`), and `updateStrategy`. Then `kubectl explain statefulset.spec.template.metadata` to confirm the pod template is the same object as a Deployment's. Compare with `kubectl explain deployment.spec.strategy`. Note what a StatefulSet does *not* give you here: you use no `volumeClaimTemplates`, because the KV cache is deliberately ephemeral.
2. **What the HPA actually reads.** `kubectl explain hpa.spec.metrics` — the four metric source types (`Resource`, `Pods`, `Object`, `External`); KEDA feeds an `External` metric. Then `kubectl explain hpa.spec.behavior.scaleDown.stabilizationWindowSeconds` — the exact knob the ScaledObject's `advanced` block sets. After installing KEDA (below): `kubectl explain scaledobject.spec` and `kubectl explain scaledobject.spec.triggers` — the CRD (custom resource definition — a user-installed API type) is documented in-cluster like any built-in. The KEDA docs page for the Prometheus scaler lists `serverAddress`, `query`, `threshold`, `activationThreshold` — read what activation means (0 → 1 replica is separate from 1 → N).
3. **The metric path end to end.** After Prometheus is installed: `kubectl port-forward -n monitoring svc/prometheus-server 9090:80`, open `http://localhost:9090/targets` and find your three worker pods (the annotations below are what make them appear), then run `sum(worker_queue_depth)` in the query box while a bench is running. Then `kubectl get --raw /apis/external.metrics.k8s.io/v1beta1 | python3 -m json.tool` — that is the API KEDA registers and the HPA reads.
4. **What `kubectl delete pod --force` does.** `kubectl delete --help | grep -A2 -- --force`: it removes the object immediately without waiting for the kubelet to confirm; combined with `--grace-period=0` it is the closest thing to a crash. Also `kubectl explain pod.spec.terminationGracePeriodSeconds`.
5. **httpx exception classes.** `python3 -c "import httpx; print([c.__name__ for c in (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError)]); print(httpx.ConnectError.__mro__)"` — the retry loop catches exactly these three; a `ReadTimeout` after the first byte must NOT be retried, which is what `sent_bytes` guards.

### Code

`k8s/worker.yaml` becomes a StatefulSet — header change only, the pod template is unchanged:

```yaml
apiVersion: apps/v1
kind: StatefulSet                              # was Deployment: pods are now worker-0, worker-1, worker-2 and keep those names across restarts
metadata: { name: worker }
spec:
  serviceName: worker-hl                       # the headless Service (M2) that gives each pod a stable DNS entry
  replicas: 3
  podManagementPolicy: Parallel                # start/stop pods all at once; the default OrderedReady would boot them one by one
  selector: { matchLabels: { app: worker } }
  template:
    metadata:
      labels: { app: worker }
      annotations:                             # the Prometheus helm chart's default scrape config discovers pods by these
        prometheus.io/scrape: "true"           # scrape this pod
        prometheus.io/port: "8000"             # ...on the wrapper's port
        prometheus.io/path: "/metrics"         # ...at the wrapper's Prometheus text endpoint (M2)
    spec:
      ...  # exactly as in M2/M5
```

Prometheus + KEDA:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts   # chart repo for Prometheus
helm repo add kedacore https://kedacore.github.io/charts                                 # chart repo for KEDA
# Prometheus server only: no alertmanager, no pushgateway, no persistent volume (kind has no default storage; ephemeral is fine here)
helm upgrade --install prometheus prometheus-community/prometheus -n monitoring --create-namespace \
  --set alertmanager.enabled=false --set prometheus-pushgateway.enabled=false --set server.persistentVolume.enabled=false
helm upgrade --install keda kedacore/keda -n keda --create-namespace   # KEDA operator + metrics API server
```

`k8s/keda.yaml`:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject                             # KEDA CRD: "scale this target from this metric"; KEDA creates the HPA
metadata: { name: worker-queue }
spec:
  scaleTargetRef: { kind: StatefulSet, name: worker }   # what to scale
  minReplicaCount: 2                           # never below 2 (keeps the ring meaningful)
  maxReplicaCount: 4                           # hard cap: 4 x (-t 2) = 8 threads = every core on the laptop
  pollingInterval: 10                          # KEDA queries Prometheus every 10 s
  cooldownPeriod: 300                          # seconds to wait before scaling back to minReplicaCount after the metric goes quiet
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:                                # passed straight through to hpa.spec.behavior
        scaleDown: { stabilizationWindowSeconds: 300 }   # 5 min: each scale-down remaps ~1/N prefixes (cache churn), so be slow
        scaleUp:   { stabilizationWindowSeconds: 30 }    # react to a queue within 30 s
  triggers:
    - type: prometheus                         # the Prometheus scaler
      metadata:
        serverAddress: http://prometheus-server.monitoring.svc.cluster.local   # the helm chart's Service, in the monitoring namespace
        query: sum(worker_queue_depth)         # PromQL: total requests waiting for a slot across all workers (a gauge from /metrics)
        threshold: "3"                         # desired replicas = ceil(query / 3): "at most ~3 queued per worker"
```

Retry in the gateway. The coordinator's `/route` needs to return a URL per candidate, with the chosen worker first, so the gateway can fall through the list. `route()` in `coordinator/app.py` becomes:

```python
@app.post("/route")
def route(req: RouteReq) -> dict:
    if not ring.nodes:
        raise HTTPException(503, "no workers")
    counters["routed"] += 1
    if req.mode == "random":
        counters["random"] += 1
        name = random.choice(sorted(ring.nodes))
        return {"worker": name, "url": stats[name]["url"], "spilled": False,
                "candidates": [{"worker": name, "url": stats[name]["url"]}]}   # M7: candidates carry URLs
    candidates = ring.lookup(req.prefix_hash, n=len(ring.nodes))     # clockwise order; [0] = affinity owner
    chosen = candidates[0]
    for c in candidates:                                              # load-aware spill (unchanged from M3)
        if stats.get(c, {}).get("queue_depth", 0) < SPILL_QUEUE:
            chosen = c
            break
    spilled = chosen != candidates[0]
    counters["spilled"] += spilled
    # M7: the CHOSEN worker (possibly a spill target) comes first so the gateway's first attempt IS the routing
    # decision and its second attempt is the next worker clockwise. Skip anyone whose stats vanished mid-poll.
    ordered = [chosen] + [c for c in candidates if c != chosen]
    cands = [{"worker": c, "url": stats[c]["url"]} for c in ordered if c in stats]
    return {"worker": chosen, "url": stats[chosen]["url"], "spilled": spilled, "candidates": cands}
```

In `gateway/app.py`, `chat()` becomes the following — the only change from M4 is the `for attempt, cand in ...` loop around the streaming block and the `sent_bytes` guard:

```python
@app.post("/v1/chat")
async def chat(req: ChatReq, request: Request):
    t0 = time.perf_counter()
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    client_key = request.headers.get("x-api-key") or request.client.host
    if not buckets.setdefault(client_key, TokenBucket(RATE, BURST)).take():
        raise HTTPException(429, "rate limited", headers={"Retry-After": "1"})
    mode = request.headers.get("x-routing", "affinity")
    ph = prefix_key(req)

    r = await client.post(f"{COORD}/route", json={"prefix_hash": ph, "mode": mode})
    if r.status_code != 200:
        raise HTTPException(503, f"router: {r.text}")
    route = r.json()
    route_ms = (time.perf_counter() - t0) * 1000
    body = {"system": req.system, "user": req.user, "max_tokens": req.max_tokens, "prefix_hash": ph}

    async def stream():
        rec = {"rid": rid, "client": client_key, "prefix": ph[:8], "mode": mode, "worker": route["worker"],
               "spilled": route["spilled"], "route_ms": round(route_ms, 1), "ttft_ms": None, "status": "ok"}
        expect_done = False
        sent_bytes = False                                    # M7: becomes True at the first chunk sent to the client
        try:
            # M7: at most two attempts — the routed worker, then the next candidate clockwise on the ring
            for attempt, cand in enumerate(route["candidates"][:2]):
                rec["worker"] = cand["worker"]                # the worker this attempt is talking to
                try:
                    async with client.stream("POST", f"{cand['url']}/generate", json=body,
                                             headers={"x-request-id": rid}) as resp:
                        if resp.status_code != 200:           # non-200 BEFORE any bytes: treat like a connect failure -> retryable
                            raise httpx.HTTPStatusError("worker", request=resp.request, response=resp)
                        async for line in resp.aiter_lines():
                            if line.startswith("event: done"):
                                expect_done = True
                            elif line.startswith("data:"):
                                if expect_done:
                                    rec.update(json.loads(line[5:]))
                                elif rec["ttft_ms"] is None:
                                    rec["ttft_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                            sent_bytes = True                 # M7: from here on the request is NOT retryable (tokens would replay)
                            yield line + "\n"
                    return                                    # stream finished normally: leave the attempt loop
                except (httpx.ConnectError, httpx.HTTPStatusError, httpx.ReadTimeout) as e:
                    if sent_bytes or attempt == 1:            # M7: mid-stream failure, or second attempt failed too -> give up
                        rec["status"] = f"worker_error:{type(e).__name__}"
                        yield f"event: error\ndata: {json.dumps({'error': rec['status'], 'rid': rid})}\n\n"
                        return
                    rec["retried"] = True                     # M7: first attempt failed before any byte -> try the next candidate
        except asyncio.CancelledError:
            rec["status"] = "client_disconnected"
            raise
        except httpx.HTTPError as e:                          # anything else from httpx (e.g. a ReadError mid-stream)
            rec["status"] = f"worker_error:{type(e).__name__}"
            yield f"event: error\ndata: {json.dumps({'error': rec['status'], 'rid': rid})}\n\n"
        finally:
            rec["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            log.info(json.dumps(rec))

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"X-Request-ID": rid, "X-Worker": route["worker"], "Cache-Control": "no-cache"})
```

`scripts/chaos.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Background load: affinity mode, 2 req/s, 30 s warm-up + 150 s measured = 180 s total. Job %1.
python3 bench/loadgen.py --mode affinity --rate 2 --warmup 30 --duration 150 --out bench/results/chaos.csv &
sleep 70                                                       # 40 s into the measured window: caches are warm, hit rate ~0.95
echo "$(date +%T) killing worker-1"; kubectl delete pod worker-1 --grace-period=0 --force   # simulate a crash: no graceful shutdown
# Watch the pod's lifecycle (Terminating -> Pending -> Running -> Ready) with timestamps, in the background. Job %2.
kubectl get pod worker-1 -w --no-headers | awk '{print strftime("%T"), $0; fflush()}' &
wait %1; kill %2 2>/dev/null || true                           # wait for the bench to finish, then stop the watcher
# Bucket the CSV into 10-second windows of MEASURED time (t - warmup) and print errors, hit rate and p50 per bucket
python3 - <<'EOF'
import csv
rows = [r for r in csv.DictReader(open("bench/results/chaos.csv"))]
for b in range(0, 150, 10):                                    # 15 buckets of 10 s
    w = [r for r in rows if b <= float(r["t"]) - 30 < b + 10]  # requests SENT in this bucket (t is relative to run start; 30 = warmup)
    ok = [r for r in w if r["status"] == "ok" and r["ttft_ms"]]
    t = sorted(float(r["ttft_ms"]) for r in ok)
    print(f"t={b:3d}s n={len(w):3d} err={len(w)-len(ok):2d} hit={sum(r['hit']=='True' for r in ok)/max(len(ok),1):.2f} "
          f"p50={t[len(t)//2] if t else 0:.0f}ms")             # the kill lands in the t=40s bucket
EOF
```

### Test

```bash
kubectl delete deploy worker 2>/dev/null; kubectl apply -f k8s/worker.yaml -f k8s/keda.yaml   # replace the Deployment with the StatefulSet + ScaledObject
kubectl get hpa                                          # KEDA created keda-hpa-worker-queue
python3 bench/loadgen.py --mode affinity --rate 6 --duration 240 --out /dev/null &   # overload: 6 req/s is beyond 3 CPU workers
watch -n5 'kubectl get hpa; kubectl get pods -l app=worker'   # TARGETS column climbs; REPLICAS goes 3 -> 4
scripts/chaos.sh
```

### Expected

Under the 6 req/s overload, `sum(worker_queue_depth)` climbs past 9 within a minute and the StatefulSet goes to 4 replicas; after the run, 5+ minutes of stabilization and then back to 2. In the chaos output: the 10-s bucket containing the kill shows `err` equal to the number of streams in flight on worker-1 (typically 2–6 at 2 req/s), then hit rate drops from ~0.95 to roughly 0.6–0.75 for one or two buckets (the ~7 remapped templates each miss once), recovers, then dips again 20–40 s later when worker-1 is back and reclaims its keys, then recovers to ~0.95. TTFT p50 follows the hit rate. No bucket after the kill should show errors — the coordinator drops the dead worker within ~2 s, and the retry catches the connect failures in that window.

The autoscaling section of the README should record the cost of a scale event too: how many templates remapped (≈ 20/N) and the p50 bump in the bucket after the scale.

### Reading a failure

| Symptom | Cause |
|---|---|
| HPA shows `<unknown>` metric | Prometheus isn't scraping the pods — check the annotations and `kubectl port-forward -n monitoring svc/prometheus-server 9090:80` → Targets |
| Scales up, never down | `cooldownPeriod` / stabilization; or the queue gauge is stuck because `waiting` was never decremented on an error path |
| Errors continue for 10+ s after the kill | Coordinator still routing to the dead IP: poll timeout too long, or `fails` threshold too high; also confirm the retry loop runs (`retried: true` in logs) |
| Only one hit-rate dip | The pod came back with a new name (it's still a Deployment) or DNS TTL/caching in the coordinator's resolver |
| Pod stuck `Terminating` | `--force` missing; llama-server ignores SIGTERM for a moment — set `terminationGracePeriodSeconds: 5` |
| Retried request produced duplicated tokens | Retry after `sent_bytes` — the guard is wrong or set too late |

### Close the milestone

Commit: `M7: StatefulSet workers, KEDA autoscaling on queue depth, gateway retry-before-first-byte, chaos test with recovery timeline`.
README section: "Operations" — the chaos timeline table, the two-dip explanation, the retry rule in one sentence, and the autoscaling-churns-cache paragraph with your remap numbers.

---

## M8 — optional: one GKE run with Terraform (week 3, half a day + a few hours of cloud time)

### What it is

A single small GPU node pool, the same manifests with a kustomize overlay (a patch layered over the kind manifests), the same `bench/run.sh`, and a README section titled "and on a real GPU". The point is not to show a bigger delta — it is to show you understand *why the delta changes*, and that the deploy is reproducible with `terraform apply`.

What changes on a GPU: prefill of 500 tokens for a 0.5B model on a T4 takes tens of milliseconds, not seconds, so the absolute saving per hit shrinks ~30–50×, and the fixed costs (routing hop, HTTP, tokenization, the first decode step) become a large share of TTFT. The *relative* delta will therefore be much smaller with the same model. To show a GPU-meaningful effect you scale the two things the saving is proportional to: model size and prefix length. A Qwen2.5-3B-Instruct Q4_K_M (~2 GB; Q4_K_M = llama.cpp's 4-bit "K-quant", ~4.5 bits per weight) with 2k-token templates gives ~1–2 s of prefill per miss on a T4 (prefill throughput for a 3B model on a T4 is on the order of 1–3k tok/s), which is the same regime as the CPU experiment. Say this in the README; it is the single most "senior" observation in the project.

Three workers on one GPU: Kubernetes GPU resources are integers, so by default one pod owns the whole T4. GKE's GPU time-sharing config lets 3 pods share it (`max_shared_clients_per_gpu = 3`); they time-slice, which is fine because the experiment is about routing, not raw throughput. The alternative — three GPU nodes — triples the bill for no additional insight.

Cost guardrails: `max_node_count = 1`, spot VMs (preemptible capacity at a steep discount that Google can reclaim with 30 s notice), a billing budget with an alert, and `terraform destroy` in the same sitting. A T4 node pool of one `n1-standard-4` + T4 is on the order of $0.50–0.60/hour on-demand and about a third of that spot (check the pricing page the day you run it); the control plane is free for one zonal cluster. A three-hour session should cost under $3.

### Look first

1. **The Terraform resource you are about to write.** Open the Terraform Registry page for `google_container_node_pool` (hashicorp/google provider). Read, in this order: the `autoscaling` block (`min_node_count`, `max_node_count`, and the note about `total_*` variants for regional pools), `node_config.spot` vs the older `preemptible`, `node_config.guest_accelerator` (`type`, `count`, `gpu_driver_installation_config`, `gpu_sharing_config` with `gpu_sharing_strategy = "TIME_SHARING"` and `max_shared_clients_per_gpu`). Then `google_container_cluster`: `remove_default_node_pool` + `initial_node_count = 1` (the documented idiom for "no default pool") and `deletion_protection` (must be `false` or `destroy` refuses). Then `google_billing_budget`: `amount.specified_amount` and `threshold_rules`.
2. **Provider version and plan output.** After `terraform init`, `terraform providers` shows the resolved google provider version; `terraform plan` prints every attribute it will create — read the node pool's `guest_accelerator` block in the plan and confirm the sharing config is present *before* you apply (it cannot be added to an existing pool).
3. **GPU availability and the driver DaemonSet.** `gcloud compute accelerator-types list --filter="name=nvidia-tesla-t4"` for zones; after the cluster is up, `kubectl get ds -n kube-system | grep nvidia` — the driver installer runs as a DaemonSet and the node only advertises `nvidia.com/gpu` once it finishes.
4. **The overlay before you apply it.** `kubectl explain pod.spec.initContainers` (they run to completion, in order, before the main containers), `kubectl explain pod.spec.nodeSelector`, `kubectl explain pod.spec.tolerations`. Then `kubectl kustomize k8s/overlays/gke | less` — the fully rendered manifest, so you can see the patch merged into the StatefulSet before it hits the cluster.
5. **The CUDA build's flags.** `docker run --rm ghcr.io/ggml-org/llama.cpp:server-cuda --help | grep -E 'n-gpu-layers|ngl'` — `-ngl 99` offloads all layers to the GPU; without it the CUDA image still runs on CPU.

### Code

`terraform/main.tf`:

```hcl
terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 6.0" }   # pin the major version: block attributes below are 6.x names
  }
}
provider "google" {
  project = var.project                    # from variables.tf / terraform.tfvars
  region  = var.region
  zone    = var.zone
}

resource "google_container_cluster" "dkv" {
  name                     = "dkv"
  location                 = var.zone                 # zonal: free control plane for one cluster (regional is not free)
  remove_default_node_pool = true                     # we manage node pools explicitly below
  initial_node_count       = 1                        # required with remove_default_node_pool; the pool is deleted right after creation
  deletion_protection      = false                    # otherwise `terraform destroy` refuses to delete the cluster
}

resource "google_container_node_pool" "system" {      # small CPU pool for gateway, coordinator, kube-system add-ons
  name       = "system"
  cluster    = google_container_cluster.dkv.name
  location   = var.zone
  node_count = 1                                      # fixed size, no autoscaling
  node_config {
    machine_type = "e2-standard-2"                    # 2 vCPU / 8 GB: enough for the two small services + monitoring
    disk_size_gb = 50
  }
}

resource "google_container_node_pool" "gpu" {
  name     = "gpu-t4"
  cluster  = google_container_cluster.dkv.name
  location = var.zone
  autoscaling {
    min_node_count = 0                                # scale to zero when no GPU pods are pending
    max_node_count = 1                                # hard cap: never more than one GPU node
  }
  node_config {
    machine_type = "n1-standard-4"                    # T4s attach to N1 machines; 4 vCPU / 15 GB
    spot         = true                               # spot pricing (~1/3 of on-demand); may be preempted mid-run
    disk_size_gb = 100                                # room for the CUDA image (~3 GB) + the 3B GGUF (~2 GB) x 3 pods
    guest_accelerator {
      type  = "nvidia-tesla-t4"                       # one T4 (16 GB)
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"                # GKE installs the driver DaemonSet for you
      }
      gpu_sharing_config {
        gpu_sharing_strategy       = "TIME_SHARING"   # let several pods time-slice the one GPU
        max_shared_clients_per_gpu = 3                # exactly our three workers; each requests nvidia.com/gpu: 1
      }
    }
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]   # lets nodes pull from Artifact Registry
  }
}

resource "google_billing_budget" "cap" {              # not a hard stop (GCP has none) — an email at 50% and 100% of $20
  billing_account = var.billing_account
  display_name    = "dkv-cap"
  amount {
    specified_amount {
      currency_code = "USD"
      units         = "20"
    }
  }
  threshold_rules { threshold_percent = 0.5 }          # alert at $10
  threshold_rules { threshold_percent = 1.0 }          # alert at $20
}
```

`terraform/variables.tf`: `project`, `region` (`us-central1`), `zone` (`us-central1-a`), `billing_account`. Check T4 availability in your zone first (`gcloud compute accelerator-types list --filter="name=nvidia-tesla-t4"`).

`k8s/overlays/gke/worker-patch.yaml` (kustomize strategic-merge patch on the StatefulSet — only the fields listed here change; containers are matched by `name`):

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata: { name: worker }                       # must match the base object to be patched
spec:
  template:
    spec:
      nodeSelector: { cloud.google.com/gke-accelerator: nvidia-tesla-t4 }   # only schedule on the T4 node
      tolerations: [{ key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }]   # GPU nodes carry this taint; tolerate it
      initContainers:                            # runs to completion BEFORE the llama container starts
        - name: fetch-model
          image: curlimages/curl
          args: ["-L", "-o", "/models/model.gguf",   # download the 3B Q4_K_M GGUF into the shared volume
                 "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"]
          volumeMounts: [{ name: models, mountPath: /models }]
      containers:
        - name: llama                            # matches the base container by name -> these fields override
          image: ghcr.io/ggml-org/llama.cpp:server-cuda   # the CUDA build (the plain :server tag has no GPU support)
          args: ["-m", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080",
                 "-c", "32768",                  # 32k context / 8 slots = 4k per slot: fits a 2k template + turn + output
                 "-np", "8",
                 "-ngl", "99",                   # offload all layers to the GPU (99 > the model's layer count)
                 "--metrics", "--slots"]
          resources: { limits: { nvidia.com/gpu: 1 } }        # with time-sharing, 3 pods each get "1"
          volumeMounts: [{ name: models, mountPath: /models }]
      volumes: [{ name: models, emptyDir: {} }]  # pod-local scratch disk shared by the init container and llama; gone with the pod
```

Images for the gateway/coordinator/wrapper go to Artifact Registry (`gcloud auth configure-docker us-central1-docker.pkg.dev`, tag, push, and set the image names in the overlay). The bench runs against the gateway through `kubectl port-forward` — no public LoadBalancer needed, which is one less thing to leave running.

### Test

```bash
cd terraform && terraform init && terraform apply            # init: download the provider; apply: show the plan, ask yes/no, create
gcloud container clusters get-credentials dkv --zone us-central1-a   # write the kubeconfig entry for the new cluster
kubectl apply -k k8s/overlays/gke && kubectl rollout status sts/worker --timeout=600s   # -k: kustomize overlay; 10 min for driver + download
kubectl port-forward svc/gateway 30080:8002 &                # same localhost:30080 the bench defaults to
# templates ~2k tokens for this run: edit make_templates(words=1500) or add a --words flag
bench/run.sh gke-t4
cd terraform && terraform destroy          # same sitting. Verify: gcloud container clusters list → empty
```

### Expected

Node pool up in 3–6 minutes (driver install is most of it); the 3B download ~1–2 minutes per pod. With 2k-token templates and the 3B model, miss TTFT of roughly 1–2.5 s and hit TTFT of 100–300 ms at low load, so a p50 delta in the same 60–85% range as CPU — but from a different mechanism balance, which is your point. With the *0.5B* model and 500-token templates on the same GPU, expect the delta to collapse to 10–40%: prefill is no longer where TTFT goes. Report both if you have the time; the pair of numbers is the story.

### Reading a failure

| Symptom | Cause |
|---|---|
| Pods `Pending`, "insufficient nvidia.com/gpu" | Time-sharing not applied (needs the node pool recreated), or the driver DaemonSet hasn't finished — `kubectl get pods -n kube-system | grep nvidia` |
| Zone has no T4 capacity / spot preempted | Try `us-central1-b/c/f` or L4 on `g2-standard-4`; spot preemption mid-benchmark shows up as a M7-style chaos event — rerun |
| llama-server logs "no CUDA device" | Wrong image tag (`server` instead of `server-cuda`) or `-ngl` missing |
| Delta tiny | You ran the 0.5B/500-token config; that is the expected result — say so, then run the 3B/2k config |
| Bill higher than expected | Leaked cluster; `terraform destroy` again and check `gcloud compute disks list` for orphaned disks |

### Close the milestone

Commit: `M8: Terraform GKE (1× T4, time-shared, spot, budget alert), kustomize overlay, GPU benchmark results`.
README section: "On a real GPU" — the two result tables, the paragraph on prefill share of TTFT, the cost of the run to the cent, and `terraform apply`/`destroy` as the reproduction commands.

---

## Closing — how this project performs in an interview

### The 5-minute whiteboard

Draw the three boxes and the ring, then narrate in this order; each sentence is roughly 30 seconds.

1. *Problem.* "Every request that shares a system prompt recomputes the same prefill unless it lands on the engine that already holds that prefix's KV. TTFT is prefill-bound, so routing decides TTFT."
2. *Key.* "The routing key is the normalized template portion of the prompt, hashed. Token-level cache matching happens in the engine; the router only has to be consistent."
3. *Placement.* "A consistent-hash ring with 150 virtual nodes per worker. Same key → same worker; a worker leaving moves ≈1/N of keys, only its own, spread over everyone; vnodes give ±5% balance."
4. *Load.* "Pure affinity makes hot prefixes hot workers. The coordinator polls queue depth and spills to the next worker on the ring above a threshold — trades a miss for less queueing; the spill target warms the prefix, so hot prefixes replicate."
5. *Worker.* "The wrapper owns slot assignment: an LRU over prefixes under a token budget, `id_slot` pinning, and hit/miss measured from the engine's own prefill count, not assumed."
6. *Gateway.* "Single entry, SSE passthrough with backpressure, disconnect cancels three hops down, per-client token bucket, one structured log line with TTFT broken into route/queue/prefill."
7. *Numbers.* "Affinity vs random at the same Poisson load: p50 down X%, hit rate H% vs R%; the delta is hit-rate difference × per-hit prefill saving, which is why it shrinks under queueing and on faster hardware."
8. *Ops.* "StatefulSet so ring positions survive restarts; KEDA on queue depth not CPU; chaos test showed two hit-rate dips — leave and return — and retries only before the first byte."
9. *What I'd change.* "Hybrid routing: ring for placement plus worker-reported warm sets as a hint; KV save/restore as a second tier; and disaggregated prefill if prefixes get long enough that transferring KV beats recomputing it."

### The five-whys drill for the bullet

- **Why consistent hashing?** Because placement must be deterministic without shared state; a central table is more precise but stateful, and the ring rebuilds from membership alone. Rendezvous hashing is the equivalent alternative; at N=3 either is fine, and I can say when I'd switch.
- **Why virtual nodes?** With one point per worker the arcs are random sizes and one worker can own half the keys; 150 points average that out to within ~10%, and a departing worker's keys fan out to everyone instead of one neighbor. The test in the repo shows the max/mean ratio at 1, 10 and 150.
- **Why LRU?** The workload's popularity is skewed and recency predicts reuse; LRU is what the engine approximates anyway, and it is the policy whose worst case (cyclic access larger than capacity) I demonstrated. LFU or a cost-aware policy (evict the cheapest-to-recompute prefix) is the improvement, and I can say what data would justify it.
- **Why this prefix definition?** The template portion is what is genuinely shared across users; first-K-tokens needs a tokenizer in the gateway and makes the key depend on the user turn when the template is short. Consistency is all routing needs; the engine does token-level matching.
- **Why is the number what it is?** Delta ≈ (hit rate gain) × (prefill time saved per hit) / (TTFT under random). On CPU, prefill is seconds and the saving is ~90% of a miss's TTFT, so a hit-rate gain of ~55 points gives a large p50 delta; queueing dilutes it at load, and a GPU shrinks the per-hit saving unless the model or the prefix grows. I measured it on both.

### Attribution wording

The team project stays in **Experience**, scoped to what you actually did there, and the rebuild goes in **Projects** with your own numbers. Two bullets, no overlap in claims:

Experience (team, past employer/course):
> Contributed to a team-built KV-cache-aware inference routing layer on GKE (Terraform, Kubernetes); owned [the specific piece you owned — e.g. the worker stats/metrics path and the load test].

Projects (solo, this repo):
> `distributed-kv-cache` — solo rebuild of a KV-cache-aware LLM serving layer: gateway → consistent-hash coordinator (150 vnodes, load-aware spill) → llama.cpp workers with LRU-managed prefix KV under a token budget; on 3 CPU workers, prefix-affinity routing cut TTFT p50 by X% (p95 Y%) vs random routing at Z req/s, hit rate H% vs R%; KEDA autoscaling on queue depth, chaos-tested recovery; every number reproducible via `bench/run.sh`. Also run on a single T4 on GKE via Terraform.

Retire the old "TTFT −37%" the day resume v2 ships. If an interviewer asks about the team number, the honest and strong answer is: "that was the team's figure on their setup; I rebuilt the system alone to understand it and measured X% on mine — here is why the two differ."
