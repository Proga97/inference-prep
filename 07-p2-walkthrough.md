# Project 2 · `mini-vllm` — Milestones 1–6, step by step

Project 1 gave you an engine: `MyQwen` in plain PyTorch, a KV cache, sampling, and numbers you measured yourself. Project 2 wraps that engine in the thing people actually deploy: a server that keeps a GPU busy with many requests at once. When you are done you will have a FastAPI process that accepts requests over HTTP, streams tokens back over SSE, runs an iteration-level scheduler that admits new requests into the running batch every step, stores KV in fixed-size blocks with a block table per sequence, preempts and recomputes when blocks run out, optionally shares prompt-prefix blocks between requests, and exposes Prometheus metrics — a toy vLLM, structured the way vLLM is structured, with every simplification named.

Work top to bottom. Every milestone ends in a script that prints numbers. Commit after each one.

**How each milestone is laid out.** *What it is* explains the idea and defines every new word the first time it appears. *Look first* tells you exactly what to open, print, or curl in the real system (vLLM v1 source, Hugging Face `transformers`, Starlette, `prometheus_client`) before you write your own version — read the real thing, understand it, then build the small version. *Code* is fully commented so you can read it cold in three months. *Test*, *Expected*, and *Reading a failure* are the same shape as Project 1. Clone vLLM once so the "Look first" paths resolve: `git clone --depth 1 https://github.com/vllm-project/vllm ~/src/vllm` — you never need to install it, only read it. Files are named as `vllm/v1/...` relative to that checkout. vLLM's layout moves between releases, so if a path is missing, `grep -rn "def allocate_slots" ~/src/vllm/vllm/v1` finds where it went.

## The finished system

```
mini-vllm/
  engine.py        ModelRunner (wraps P1 MyQwen), sampling, padded-batch builders, Detokenizer
  scheduler.py     Request dataclass + Scheduler.step(): admit → prefill → decode → retire
  kv_manager.py    SlotKV (M2), BlockKV (M3), PrefixCachingKV (M5)
  server.py        FastAPI: /generate (SSE), /stats, /metrics; engine loop as a background task
  metrics.py       prometheus_client instruments (M6)
  bench/
    loadgen.py     Poisson-arrival load generator → JSON with throughput and p50/p99
    sweep.sh       loadgen at several arrival rates
    plot.py        throughput-vs-p99 curve
    kv_waste.py    samples /stats during a run → KV utilization (M3)
    prefix_test.py 20 shared system prompts, cold vs warm (M5)
  test_batch_parity.py   batched/padded/paged decode == plain P1 decode, token for token
```

Core code (engine + scheduler + kv_manager + server + metrics) lands at ~670 lines; with `bench/` and the test, ~900.

Dependencies: Python 3.11+ (`asyncio.timeout` is used), `torch` with CUDA, `transformers` (tokenizer + config only), `safetensors`, `huggingface_hub`, `fastapi`, `uvicorn`, `pydantic`, `httpx` (load generator), `prometheus_client`, `matplotlib`. Your `qwen_from_scratch.py` from Project 1 sits on `PYTHONPATH` (symlink it into the repo or `pip install -e` P1).

On the Windows box, do this inside **WSL2** (Ubuntu). The Windows NVIDIA driver exposes CUDA to WSL2 with no extra install; you get a Linux `uvicorn`, real `asyncio` selector semantics, and `bench/sweep.sh` just works. Run the load generator inside the same WSL2 instance — going Windows→WSL2 over the virtual NIC adds jitter you would otherwise be measuring. If you must stay native, everything here runs on Windows too, but `hash()` seeding, shell scripts, and process signals are the things to watch.

---

## Concepts you need before starting

Every term below is used in this document as if you know it. Read this section once now, and come back to it whenever a word in a milestone feels unfamiliar. Terms are grouped by where they come from; each one gets a plain definition and, where it helps, a one-line analogy.

### The model side (from Project 1, restated)

- **Token.** The unit the model reads and writes: a chunk of text (a word, part of a word, or a byte) that the tokenizer maps to an integer id. Qwen's vocabulary has 151,936 of them. "Generate 128 tokens" ≈ generate ~100 English words.
- **Tokenizer / byte-level BPE.** The function text → ids and back. Qwen's is *byte-level* BPE: the base alphabet is the 256 byte values, so a single non-ASCII character (an emoji, a Chinese character) is 2–4 bytes and can be split across several tokens. Consequence: decoding one token alone can give half a character.
- **Chat template.** The fixed wrapper text (`<|im_start|>system ... <|im_start|>user ... <|im_start|>assistant`) that an instruct-tuned model expects around a conversation. `tokenizer.apply_chat_template` builds it. Every request through this server starts with the same ~20 template tokens, which M5 exploits.
- **Logits.** The model's raw output: one score per vocabulary entry for "which token comes next", shape `[batch, vocab]`. Higher = more likely. Not probabilities until you apply softmax.
- **Softmax.** Turns a vector of scores into a probability distribution (all positive, summing to 1) by exponentiating and normalizing. Used twice here: inside attention (over keys) and on logits (over the vocabulary).
- **Sampling / greedy / temperature.** Choosing the next token from the logits. *Greedy* = take the argmax (deterministic). *Temperature* divides the logits before softmax: 0 → greedy, 1 → sample from the model's own distribution, > 1 → flatter, more random. This project supports temperature only; top-k/top-p from P1 can be plugged in.
- **Attention, Q/K/V.** The mechanism by which a token gathers information from earlier tokens. Each token produces a *query* (what am I looking for), a *key* (what do I contain), and a *value* (what I hand over). Scores = Q·Kᵀ; softmax over the scores gives weights; the output is the weighted sum of V. Analogy: a lookup where every past token is a labelled drawer and the query decides how much to take from each.
- **Causal mask.** In a decoder-only model, a token may only attend to itself and earlier positions, never later ones (it is predicting the future). Implemented by setting the scores of "later" positions to a huge negative number before softmax so their weight becomes 0.
- **GQA (grouped-query attention).** Qwen2.5-0.5B has 14 query heads but only 2 key/value heads; each K/V head is shared by 7 query heads (`repeat_kv` copies them out at compute time). It exists purely to shrink the KV cache — 7× here.
- **RoPE (rotary position embedding).** How the model knows *where* a token is: Q and K are rotated by an angle proportional to the token's position. Hence *K depends on position*: the same token at position 5 and position 500 has different K. This is why cached K/V can only be shared between requests whose tokens match from position 0.
- **KV cache.** During generation the model needs the K and V of every previous token at every layer. Recomputing them each step would be quadratic; instead they are computed once and *cached*. The cache is the biggest dynamic memory consumer in serving — for this model, 12 KiB per token (see Step 0). Analogy: notes you keep from every page you've read so you do not reread the book each time you turn a page.
- **Prefill vs decode.** *Prefill* is the first forward pass over the whole prompt: `T` tokens go in at once, K/V for all of them are written to the cache, and the first output token is sampled. *Decode* is every step after that: one new token goes in, attends to the cache, one token comes out. Prefill is a big matrix multiply (compute-heavy); decode reads the whole model's weights to produce one token per sequence (memory-heavy). Almost every serving decision comes from this asymmetry.
- **fp16 / bf16 / fp32.** Floating-point formats with 16, 16, and 32 bits. fp16 has more precision but a small range (max 65,504 — easy to overflow); bf16 has fp32's range with less precision; fp32 is the safe reference. The RTX 2060 (Turing) has fast fp16 tensor cores and no bf16, so this project runs fp16 and falls back to fp32 for parity tests.
- **`torch.no_grad()`.** Tells PyTorch not to build the graph needed for backpropagation. Inference never needs gradients; forgetting this doubles memory and slows every step.
- **Advanced indexing.** `tensor[index_tensor]` where the index is itself a tensor of ids. It *always copies* (unlike a slice, which is a view). The KV managers use it to gather blocks, and the copy is a cost you will measure.

### Hardware and performance

- **FLOP.** One floating-point operation (a multiply or an add). A 0.5B-parameter model does roughly 2 × 0.5B = 1 GFLOP per token in the matrix multiplies. The 2060 can do ~10–20 TFLOP/s in fp16 in practice.
- **Memory bandwidth.** How many bytes per second the GPU can read from its memory: 336 GB/s on the 2060. Reading 1 GB of weights therefore takes ≥ 3 ms no matter how little math you do with them.
- **Compute-bound vs memory-bound ("bound").** Whichever resource is saturated is what *bounds* the step time. Decode at small batch reads all weights for a handful of tokens' worth of math → memory-bound. Prefill of a long prompt does lots of math per byte read → compute-bound. Analogy: a kitchen is *oven-bound* if the oven is full and the cooks are idle, *cook-bound* if the opposite.
- **Batching, and why GPUs want it.** Running several sequences through the model in one forward pass. For decode, the weights are read from memory once either way, so 16 sequences cost barely more per step than 1 — 16× the throughput for nearly free. Analogy: a bus route where the fuel cost is the same whether 1 or 16 people ride. The catch is that different sequences have different lengths, which is what padding and masks solve.
- **Kernel launch.** A *kernel* is one GPU function (a matmul, a softmax). Each launch costs ~5–10 µs of CPU-side overhead before any work happens. A decode step of this model launches ~400 of them, so launch overhead is a real fraction of a 10 ms step; CUDA graphs (mentioned in the divergences section) exist to remove it.
- **Throughput vs latency.** *Throughput* = work per unit time (tokens/s, requests/s) — what the operator cares about. *Latency* = time a single request waits — what the user cares about. They fight: batching raises throughput but each request shares the GPU so its per-token time rises. Serving is choosing a point on that trade-off.
- **TTFT (time to first token).** Latency from request arrival to the first output token: queue wait + prefill time. The "did it start responding" number.
- **ITL (inter-token latency).** Time between consecutive output tokens once streaming has begun; also called TPOT. Its median is your decode step time; its tail shows stalls.
- **e2e (end-to-end) latency.** Arrival to last token. ≈ TTFT + (tokens − 1) × ITL.
- **p50 / p99 (percentiles).** Sort all measured latencies; p50 is the median (half were faster), p99 is the value 99% were faster than. p99 is what SLOs are written against, because the slow 1% is the user who complains. Analogy: p50 is the typical commute, p99 is the commute on the day of the accident.
- **Histogram.** A count of observations per *bucket* (e.g. "how many TTFTs were between 25 and 50 ms"). Prometheus histograms are cumulative buckets plus a sum and count; percentiles are estimated from them. Cheap to keep in-process, aggregatable across servers — which is why serving metrics are histograms, not lists.
- **Offered load / λ (lambda) / arrival rate.** How many requests per second the clients *send*, independent of whether the server keeps up. The x-axis of every serving benchmark.
- **Poisson arrivals.** A model of "independent users showing up at random" at mean rate λ: the gap between consecutive arrivals is drawn from an exponential distribution with mean 1/λ. It produces realistic clumps and lulls; firing N requests at once does not. In Python: `await asyncio.sleep(random.expovariate(rate))`.
- **The knee.** On a throughput-vs-p99 plot, the point where the system stops absorbing load and latency starts climbing without bound. Left of it, you are adding useful work; right of it, you are only adding queue. The knee is your capacity per GPU.
- **Utilization / fragmentation.** *Utilization* = memory actually holding useful data ÷ memory reserved. *Fragmentation* = the reserved-but-unused part. Contiguous per-sequence reservation has terrible utilization because it reserves for the worst case; paging fixes it.

### Serving-system vocabulary

- **Request / sequence.** A request is one HTTP call. A sequence is the token stream the engine tracks for it (prompt + generated so far). One request = one sequence in this project (no `n > 1`).
- **Scheduler.** The component that decides, every step, which sequences the GPU works on. Ours has `admit → prefill → decode → retire`.
- **Static batching.** Collect a batch, run it until every sequence finishes, then start the next batch. Simple, wasteful.
- **Continuous (iteration-level) batching.** Re-decide the batch at *every step*: finished sequences leave immediately, waiting ones join immediately. The single biggest throughput idea in LLM serving (Orca, 2022), and the point of M2.
- **FCFS / head-of-line blocking.** First-come, first-served admission. If the request at the head of the queue cannot be admitted (no memory), everything behind it waits even if it would fit — that is head-of-line blocking.
- **Block / page, block table, free list.** *Block*: a fixed-size chunk of KV storage (16 tokens here). *Block table*: per sequence, the list of physical block ids holding its tokens in order — logical position `p` lives in block `table[p // 16]`, slot `p % 16`. *Free list*: the block ids nobody owns. Straight from operating-system virtual memory, hence "PagedAttention".
- **Preemption.** Evicting a running sequence to free memory for others, and resuming it later. *Recompute* rebuilds its cache by re-running prefill; *swap* copies the cache to CPU RAM and back.
- **Reference count (ref count).** A per-block counter of how many sequences use it. A block is truly free only when the count reaches 0. Needed once blocks can be shared.
- **LRU (least recently used).** An eviction policy: when you must throw something away, throw away the thing untouched for the longest time. Implemented with an ordered dictionary where each use moves the key to the end and eviction pops from the front.
- **Prefix caching / content addressing / hash chain.** If two prompts start with the same tokens, their first K/V blocks are identical and can be shared. A block is found by a *hash* (a fixed-size fingerprint of its content); *chaining* means block *i*'s hash includes block *i−1*'s hash, so the fingerprint encodes "these 16 tokens *after exactly this prefix*", which is required because RoPE makes K position-dependent.
- **Sentinel.** A special value (here `None`) pushed into a queue to mean "no more items" so the consumer knows to stop.

### Python async and HTTP

- **Event loop.** A single thread running a scheduler of its own: it keeps a list of *coroutines* (functions declared `async def`) and, whenever one hits `await` on something not ready, switches to another that is ready. Nothing runs in parallel; everything takes turns. Consequence: any synchronous work (a 10 ms GPU step) *blocks the whole loop* — no HTTP request is accepted until it returns. Analogy: one chef with many pots, stirring whichever is ready; a pot that needs constant stirring for 10 seconds stalls all the others.
- **Coroutine, task, `await`.** `async def f()` defines a coroutine; calling it makes a coroutine object that does nothing until driven. `asyncio.create_task(f())` hands it to the loop to run concurrently. `await x` pauses this coroutine until `x` is done and lets others run. `await asyncio.sleep(0)` pauses for zero time — its only purpose is to give every other ready task one turn.
- **`asyncio.Queue`, `asyncio.Event`.** Loop-friendly hand-off primitives. `queue.get()` awaits until an item is put; `queue.put_nowait(x)` adds without waiting. An `Event` is a flag that awaiting tasks sleep on until someone calls `.set()`. They are how the HTTP handler and the engine task talk without sharing locks.
- **Cancellation.** The loop can inject a `CancelledError` into a task at its next `await`. A task that is never suspended at an await (because the thing it awaits is always already done) cannot be cancelled that way — the bug M1 works around.
- **ASGI, Starlette, FastAPI, uvicorn.** *ASGI* is the interface between an async Python web app and a server. *Starlette* implements requests/responses on top of it; *FastAPI* adds routing and pydantic validation on top of Starlette; *uvicorn* is the server process that speaks HTTP and drives the ASGI app. When this document says "Starlette's `StreamingResponse`", that is the class FastAPI re-exports.
- **`lifespan`.** FastAPI's startup/shutdown hook: code before `yield` runs once when the server starts (load the model, start the engine task), code after runs at shutdown.
- **SSE (Server-Sent Events).** The simplest streaming protocol over plain HTTP: the server keeps one response open with `Content-Type: text/event-stream` and writes messages of the form `data: <text>\n\n`. Lines beginning with `:` are comments the client ignores — used as heartbeats. Browsers read it with `EventSource`; curl needs `-N` (no buffering). It is what ChatGPT-style token streaming uses.
- **Heartbeat.** A no-content message sent when nothing else has been sent for a while, so proxies and load balancers do not close an "idle" connection.
- **Disconnect.** The client closed the socket. The server is *not* automatically told; something must check. If nobody checks, the GPU keeps generating tokens into the void.
- **Prometheus, scrape, counter/gauge/histogram.** Prometheus is a metrics database that *scrapes* — periodically HTTP-GETs — a `/metrics` endpoint that returns current values in a text format. A *counter* only goes up (tokens generated); a *gauge* goes up and down (queue depth); a *histogram* is bucket counts (TTFT). The `prometheus_client` library keeps them in-process and renders the text.
- **Load generator.** A client program that sends requests on a schedule (Poisson here), measures per-request TTFT/ITL/e2e, and summarizes. Yours is `bench/loadgen.py`; vLLM's is `vllm bench serve`.

---

### The request lifecycle in one paragraph

A request arrives at the FastAPI handler, is tokenized (chat template applied), becomes a `Request` object with an `asyncio.Queue` for output tokens, and is appended to the scheduler's **waiting** queue. The handler then does nothing but read from that queue and write SSE frames. Separately, one background task runs the **engine loop**: each iteration calls `scheduler.step()`, which (1) **admits** waiting requests into the running set if KV capacity allows and runs one **prefill** forward pass over the admitted batch, producing each one's first token; (2) runs one **decode** forward pass over every running sequence, producing one more token each; (3) **retires** finished sequences, freeing their KV and pushing a `None` sentinel so the handler closes the stream. A request therefore spends its life as: arrive → queue → admitted (prefill step, TTFT clock stops) → decode steps (ITL clock ticks) → finish/stream close. This is real vLLM's shape: an API server (`AsyncLLM`) that only moves bytes, and an engine core whose `step()` is the unit of scheduling. vLLM puts the engine core in a separate process behind ZMQ; you keep it as an asyncio task in the same process, and you will be able to say exactly what that costs.

---

## Step 0 — Sizing the 2060 before you write anything

Get the arithmetic on paper first, because it decides every default in the code.

### Look first

Print the numbers you are about to multiply instead of trusting this document's copy of them.

```python
from transformers import AutoConfig
import torch

cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# Every number in the KV arithmetic below comes from these four fields.
print(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.num_attention_heads,
      cfg.hidden_size // cfg.num_attention_heads)          # expect 24 2 14 64
print(cfg.tie_word_embeddings, cfg.vocab_size)             # True 151936 — no separate lm_head
free, total = torch.cuda.mem_get_info()                    # bytes free / total on the GPU right now
print(f"GPU free {free/2**30:.2f} GiB of {total/2**30:.2f} GiB")   # what is actually left for KV
```

Then read how vLLM does the same sizing: `vllm/v1/core/kv_cache_utils.py`, functions `get_kv_cache_config` and, inside it, the line that divides `available_memory` by the bytes-per-block to get `num_blocks` (search for `num_blocks =`). Notice that vLLM does not take a KV budget in MB — it takes `--gpu-memory-utilization` (default 0.9), *profiles* one worst-case forward pass to learn peak activation memory (`vllm/v1/worker/gpu_worker.py`, `determine_available_memory`), and gives the remainder to KV. That profiling run is the honest version of the "where the rest of the 6 GB goes" paragraph below. Also notice `page_size_bytes` on `FullAttentionSpec` in `vllm/v1/kv_cache_interface.py`: it is `2 * block_size * num_kv_heads * head_size * dtype_size` — the same product you compute next.

### The arithmetic

**Weights.** Qwen2.5-0.5B is ~494M parameters (136M of them the tied embedding). fp16 → ~0.99 GB. The 2060 is Turing (compute 7.5): fp16 tensor cores yes, bf16 no. Use `torch.float16`. If you ever see NaN/garbage in fp16 that is fine in fp32, the model is overflowing in an fp16 activation; fp32 weights are 2 GB and still fit for debugging.

**KV per token.** From `config.json`: 24 layers, 2 KV heads (GQA: 14 query heads share them), head_dim 64, fp16:

```
2 (K and V) × 24 layers × 2 kv_heads × 64 head_dim × 2 bytes = 12,288 bytes = 12 KiB per token
```

Note the GQA factor: with 14 KV heads it would be 84 KiB. This is why you must cache K/V *before* `repeat_kv`, never after.

**Blocks.** With block size 16, one block is 192 KiB. A KV budget of 3 GiB is 3 × 2³⁰ / 12,288 = **262,144 tokens = 16,384 blocks**; 1 GiB is 87,381 tokens = 5,461 blocks.

**What "≥ 16 concurrent streams" implies.** 16 streams × (512 prompt + 256 output) = 12,288 tokens = 768 blocks = 144 MiB. KV memory is *not* what limits a 0.5B model on a 6 GB card — per-step time is. That has two consequences: default the KV budget to 1 GiB (plenty), and when you want to *see* preemption and eviction you must starve the pool deliberately (`MV_KV_MB=8` → 682 tokens → 42 blocks).

**Where the rest of the 6 GB goes.** CUDA context and cuBLAS workspace ~300–500 MB; weights 1 GB; prefill activations — the attention scores for a batch are `[B, 14, T, T]` in fp32, i.e. 56·T² bytes per row, so a 2048-token row costs 235 MB and four of them nearly 1 GB; the allocator's cached slack. A 1–2 GiB KV budget plus `max_prefill_tokens=2048` leaves comfortable headroom. 3 GiB KV is reachable only if you cap prefill smaller.

**Per-step time (what to expect, not what to promise).** Decode at batch 1 reads all 1 GB of weights once: 1 GB / 336 GB/s ≈ 3 ms floor. Your P1 number (~100–120 tok/s ⇒ 8–10 ms/step) shows the other 5–7 ms is kernel-launch overhead (~24 layers × ~15 kernels) plus Python. Batching to 16 barely changes the weight read; it adds the KV gather (16 seqs × ~600 tokens × 12 KiB ≈ 115 MB moved ≈ 0.7 ms), a wider attention, and ~1 ms of Python bookkeeping. Expect 10–16 ms per decode step at batch 16, i.e. roughly 1,000–1,600 tokens/s of decode capacity, less once prefills interleave. Measure it; the README number is whatever you measure.

---

## Interface contract with Project 1

P2 imports `MyQwen` from `qwen_from_scratch.py` and needs exactly this from it. If your P1-M3 KV cache ended up shaped differently (e.g. cache held inside the modules), write a thin adapter in P1 that presents this signature — P2 never reaches inside the model.

### Look first

The contract below is a stripped-down version of what Hugging Face's own `Qwen2` implementation does. Read HF's before writing yours so every argument has a known counterpart.

```python
import inspect
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2 as mq
from transformers.cache_utils import DynamicCache

hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="float16").cuda()

# 1. The attention forward: find where past K/V are concatenated and where repeat_kv is applied.
print(inspect.getsource(mq.Qwen2Attention.forward))
# 2. repeat_kv itself: [B, kv_heads, L, hd] -> [B, kv_heads*n_rep, L, hd]. Notice it is an expand+reshape,
#    i.e. a real copy — done AFTER the cache update, which is why the cache stays at 2 heads.
print(inspect.getsource(mq.repeat_kv))
# 3. The cache: DynamicCache.update is a torch.cat along dim=-2 (the sequence axis) per layer.
#    In transformers >= 4.54 it delegates to DynamicLayer.update in the same file — read that one too.
print(inspect.getsource(DynamicCache.update))
```

What to notice: (a) in `Qwen2Attention.forward`, `past_key_values.update(key_states, value_states, self.layer_idx, ...)` is called *before* `repeat_kv` (inside the attention function) — the cache holds `[B, 2, L, 64]`, not `[B, 14, L, 64]`; (b) `DynamicCache.update` grows the cache with `torch.cat`, which allocates a fresh `[B, 2, L+T, 64]` tensor and copies every step — O(L) per step per layer. That copy is what M3's fixed blocks eliminate; (c) `position_ids` flow into the rotary embedding (`Qwen2RotaryEmbedding.forward(x, position_ids)`) and are per-row `[B, T]` — HF already supports the "different rows at different positions" case you need for left padding. Now run one HF generation with a cache and inspect the shapes:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()   # [1, T] token ids
out = hf(ids, use_cache=True)                                                   # one prefill pass; use_cache -> returns the cache
pkv = out.past_key_values                                                       # a DynamicCache object
# Layer-0 K tensor. The attribute name changed across transformers versions: new = .layers[i].keys, old = .key_cache[i].
k0 = pkv.layers[0].keys if hasattr(pkv, "layers") else pkv.key_cache[0]
print(type(pkv).__name__, k0.shape)          # expect DynamicCache torch.Size([1, 2, T, 64]) — 2 heads, not 14
```

### The contract

```python
logits, new_kv = model(input_ids, position_ids, kv_cache=None, attention_mask=None, last_only=False)
```

| Argument / return | Shape and meaning |
|---|---|
| `input_ids` | `[B, T]` long. `T` = prompt length in prefill, `1` in decode. |
| `position_ids` | `[B, T]` long, **per row**. Row *i*'s positions are `past_len_i .. past_len_i+T-1`; with left padding, pad columns hold 0. RoPE must use these, not `arange(T)`. |
| `kv_cache` | `None`, or a list of `num_layers` tuples `(K, V)`, each `[B, kv_heads, L, head_dim]` — K/V of the *past*, stored **before** `repeat_kv` (2 heads, not 14). Rows may contain garbage beyond their true length; the mask hides it. |
| `attention_mask` | `None`, or an **additive** float mask `[B, 1, T, L+T]`: `0` = attend, `torch.finfo.min` = masked. When given, the model uses it **instead of** building its own causal mask (P2 builds causal+padding in one tensor). Add it to scores in fp32. |
| `last_only` | `True` → logits `[B, vocab]` for the last position only. Prefill logits for all positions are `[B, T, 151936]` fp16 — 622 MB for 2048 tokens — and P2 never needs them. |
| returns | `(logits, new_kv)`; `new_kv` is the same per-layer list with the new tokens appended: `[B, kv_heads, L+T, head_dim]`. |

An **additive mask** means a float tensor you *add* to the attention scores: 0 where attention is allowed, a huge negative number where it is not, so that after softmax the forbidden positions get weight ≈ 0. (The other convention, a boolean mask, is what P1 used internally; additive is what HF and vLLM's reference paths use because it composes: causal + padding is just the elementwise minimum.)

The pieces of P1 that implement it — this is what your `Attention.forward` tail, `DecoderLayer`, `QwenModel`, and `MyQwen` should look like after P1-M3 plus this contract:

```python
class Attention(nn.Module):
    ...
    def forward(self, x, position_ids, past_kv=None, attention_mask=None):
        # x            : [B, T, hidden]      hidden = 896 for Qwen2.5-0.5B
        # position_ids : [B, T]              absolute position of every token, PER ROW (left padding shifts them)
        # past_kv      : None or (K, V), each [B, kv_heads, L, head_dim] = [B, 2, L, 64]  (the cache, pre-repeat_kv)
        # attention_mask: None or additive float [B, 1, T, L+T]; None means "build a plain causal mask yourself"
        B, T, _ = x.shape
        # Project to Q/K/V and split heads. .view splits the last dim into (heads, head_dim);
        # .transpose(1, 2) moves heads before the sequence axis so matmul batches over (B, heads).
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # [B, 14, T, 64]
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B, 2, T, 64]  (GQA: 2 heads)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B, 2, T, 64]
        # RoPE tables computed from the per-row positions — NOT from arange(T). With left padding, row i's
        # real tokens sit at positions past_len_i.., and pad columns hold 0 (harmless: they are masked out).
        cos, sin = self.rotary_emb(x, position_ids)             # each [B, T, head_dim]
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)           # [B, 1, T, head_dim] — broadcast over heads
        q = q * cos + rotate_half(q) * sin                      # rotate Q by its position  [B, 14, T, 64]
        k = k * cos + rotate_half(k) * sin                      # rotate K by its position  [B, 2, T, 64] — K now encodes position
        if past_kv is not None:
            # Prepend the cached past along the sequence axis (dim 2): [B, 2, L, 64] ++ [B, 2, T, 64] -> [B, 2, L+T, 64].
            # This is exactly HF's DynamicCache.update — and exactly the O(L) copy that paged blocks avoid.
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)                                          # cache BEFORE repeat_kv: 2 heads, 12 KiB/token, not 84
        # Expand the 2 KV heads to 14 so each query head has a matching key head: [B, 2, L+T, 64] -> [B, 14, L+T, 64].
        k = repeat_kv(k, self.num_key_value_groups)              # num_key_value_groups = 14 // 2 = 7
        v = repeat_kv(v, self.num_key_value_groups)
        # Scores: every query position against every key position, scaled by 1/sqrt(head_dim).
        # Upcast to fp32 BEFORE adding the mask: adding -65504 to an fp16 score can round to -inf, and a row of -inf
        # softmaxes to NaN, which then poisons every later layer.
        scores = torch.matmul(q, k.transpose(-2, -1)).float() * self.scaling   # [B, 14, T, L+T], fp32
        if attention_mask is None:                               # batch-1 path (P1 tests, M1): plain causal mask
            past = k.shape[2] - T                                # how many key columns belong to the past
            # tril(past): query t may see key columns 0..past+t (all of the past + itself + earlier new tokens).
            causal = torch.ones(T, past + T, dtype=torch.bool, device=x.device).tril(past)   # [T, L+T] bool
            scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        else:                                                    # P2 hands in causal + padding together
            scores = scores + attention_mask                     # broadcast [B, 1, T, L+T] over the 14 heads
        weights = torch.softmax(scores, dim=-1).to(q.dtype)      # [B, 14, T, L+T]; back to fp16 for the matmul
        # Weighted sum of values, then merge heads back: [B, 14, T, 64] -> [B, T, 14, 64] -> [B, T, 896].
        out = torch.matmul(weights, v).transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out), new_kv                          # ([B, T, hidden], (K, V) each [B, 2, L+T, 64])


class DecoderLayer(nn.Module):
    ...
    def forward(self, x, position_ids, past_kv=None, attention_mask=None):
        # Pre-norm residual block, unchanged from P1 except the cache/mask are threaded through.
        h, new_kv = self.self_attn(self.input_layernorm(x), position_ids, past_kv, attention_mask)   # h: [B, T, hidden]
        x = x + h                                                # residual add (attention sub-block)
        x = x + self.mlp(self.post_attention_layernorm(x))       # residual add (MLP sub-block)
        return x, new_kv                                         # ([B, T, hidden], this layer's (K, V))


class QwenModel(nn.Module):
    ...
    def forward(self, input_ids, position_ids, kv_cache=None, attention_mask=None):
        h = self.embed_tokens(input_ids)                         # [B, T] ids -> [B, T, hidden]
        new_cache = []                                           # one (K, V) per layer, collected as we go
        for i, layer in enumerate(self.layers):                  # 24 layers
            # Hand layer i ITS slice of the cache (or None on a fresh prefill).
            h, kv = layer(h, position_ids, None if kv_cache is None else kv_cache[i], attention_mask)
            new_cache.append(kv)
        return self.norm(h), new_cache                           # final RMSNorm; cache is a list of 24 (K, V) tuples


class MyQwen(nn.Module):
    ...
    def forward(self, input_ids, position_ids, kv_cache=None, attention_mask=None, last_only=False):
        h, new_cache = self.model(input_ids, position_ids, kv_cache, attention_mask)   # h: [B, T, hidden]
        if last_only:
            h = h[:, -1]                                          # [B, hidden] — only the last position's state
        # Tied head: logits = h @ E^T, E the embedding matrix [vocab, hidden]. With last_only: [B, vocab];
        # without: [B, T, vocab] — 622 MB in fp16 for T=2048, which is why P2 always passes last_only=True.
        logits = h @ self.model.embed_tokens.weight.T
        return logits, new_cache
```

Three things in there are load-bearing and will come up in interviews: the cache is taken before `repeat_kv` (7× smaller); the scores are upcast to fp32 *before* the mask is added (adding `-65504` to an fp16 score can round to `-inf`, and a fully masked row then softmaxes to NaN, which poisons every later layer through the padded keys); and the mask, when provided, is the whole mask — the model does not add its own causal term on top.

Quick check that P1 satisfies the contract (batch 1, no mask), before touching P2:

```python
import torch
from qwen_from_scratch import MyQwen
# ... load model + tokenizer as in P1 ...
ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(device)   # [1, T]
pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)                             # [1, T] positions 0..T-1
with torch.no_grad():
    # Path A: the whole prompt in one shot -> logits for the last position.            full: [1, vocab]
    full, _ = model(ids, pos, last_only=True)
    # Path B: everything but the last token (fills the cache)...                        kv: 24 × ([1, 2, T-1, 64], same)
    l0, kv = model(ids[:, :-1], pos[:, :-1], last_only=True)
    # ...then the last token alone, attending to the cache. Must give the same logits.  inc: [1, vocab]
    inc, kv = model(ids[:, -1:], pos[:, -1:], kv_cache=kv, last_only=True)
print("cache parity:", (full - inc).abs().max().item(), "| next:", tokenizer.decode(inc[0].argmax()))
```

**Expected:** difference ≲ 1e-3 in fp16 (≲ 1e-5 in fp32) and ` Paris`. `kv[0][0].shape` must be `[1, 2, T, 64]` with `T` the prompt length — if the second dim is 14, you cached after `repeat_kv`.

---

## Milestone 1 — SSE streaming server over the P1 engine

### What it is

**Server-Sent Events** (SSE) is the simplest way to stream tokens over HTTP: one long-lived response with `Content-Type: text/event-stream`, each message a `data: ...\n\n` frame, and lines starting with `:` are comments the client ignores — which is exactly what a heartbeat needs. FastAPI's `StreamingResponse` takes an **async generator** (an `async def` that `yield`s values); every `yield` becomes a chunk on the wire. The client sees text appear token by token instead of waiting for the whole answer.

The design decisions that matter, and that M2 depends on:

**The engine runs in a background task, not in the handler.** A *handler* is the function FastAPI calls for one HTTP request. If the handler called the model directly, the event loop would be blocked for the duration of every GPU step, and a second concurrent request could not even be *accepted* (recall: the loop is one thread taking turns; a 10 ms synchronous GPU call is 10 ms during which no other coroutine runs). Instead `lifespan` starts one `asyncio.Task` that owns the GPU; handlers push a job into a queue and read tokens back from a per-request `asyncio.Queue`. This is the API-server/engine split in miniature. The engine yields to the loop (`await asyncio.sleep(0)`) between steps so handlers get a turn.

**Heartbeats.** Proxies and load balancers kill idle connections (often at 30–60 s); a long prefill or a queue wait can exceed that. Every `HEARTBEAT_S` seconds without a token, emit `: ping\n\n`.

**Disconnect must cancel generation.** A client that closes its tab must not keep burning GPU steps. You would expect the framework to cancel your generator when the socket closes — *do not rely on it*. With the engine loop feeding one token per event-loop iteration, anyio's cancellation delivery (which only cancels a task whose awaited future is still pending) keeps finding the generator's queue-future already resolved, and the generator runs to `max_tokens` sending into the void. This was measured while writing this walkthrough (Starlette 1.0, uvicorn 0.46): 2,000 tokens generated after the client had gone. The fix is to own it: poll `request.is_disconnected()` on every heartbeat timeout **and** at least every `DISCONNECT_POLL_S` while tokens are flowing, and set `job.cancelled` in a `finally` so whichever path ends the generator also stops the engine. vLLM's API server does the same — it polls `is_disconnected()` in its streaming loops and races non-streaming handlers against a disconnect listener — for the same reason.

**Detokenize with a rolling window, never token by token.** *Detokenizing* is turning ids back into text. Qwen's tokenizer is byte-level BPE: a token can be *part* of a multi-byte UTF-8 character (any emoji, most CJK). `tokenizer.decode([tok])` on such a token returns `�` (the Unicode replacement character, meaning "invalid byte sequence"). The correct incremental algorithm (HF `TextStreamer`, vLLM's detokenizer) keeps two offsets: decode `ids[prefix:]`, compare with decode `ids[prefix:read]`, emit only the text that grew, and hold it back while it ends in `�`. Also skip stop tokens before they reach the detokenizer, and always `skip_special_tokens=True`.

### Look first

Four real implementations to read before writing yours. Each one is short.

**1. How Starlette streams a response and when it looks for a disconnect.** Open `starlette/responses.py`, class `StreamingResponse`. Find `stream_response` (the `async for chunk in self.body_iterator:` loop that sends `{"type": "http.response.body", "body": chunk, "more_body": True}` per chunk — that is what one `yield` becomes) and `__call__`, where it runs `stream_response` and `listen_for_disconnect` together in an anyio task group and cancels the group when either finishes. That cancellation is the thing the paragraph above says not to rely on: it can only land at an `await` that is actually suspended.

```python
import inspect, starlette.responses as sr, starlette.requests as srq
print(inspect.getsource(sr.StreamingResponse))          # note __call__ and listen_for_disconnect
print(inspect.getsource(srq.Request.is_disconnected))   # the poll you will call yourself
```

In `Request.is_disconnected`, notice it does a *non-blocking peek* at the ASGI receive channel (a `CancelScope` that is cancelled immediately, so `receive()` returns only if a message is already waiting) and looks for `{"type": "http.disconnect"}`. Two consequences: it is cheap to call often, and it only reports a disconnect *after uvicorn has noticed one* — which typically happens when a write (a token or a heartbeat) fails on the closed socket. That is why the heartbeat and the disconnect poll belong together.

**2. How vLLM's OpenAI server streams and cancels.** `vllm/entrypoints/openai/serving_chat.py`, method `chat_completion_stream_generator`: scroll to the `yield f"data: {data}\n\n"` lines and the final `yield "data: [DONE]\n\n"` — the same SSE framing you will write. Then `vllm/entrypoints/utils.py`: `listen_for_disconnect` (loops on `request.receive()` until it sees `http.disconnect`) and the `with_cancellation` decorator that races the handler against it. Finally `vllm/v1/engine/async_llm.py`, method `generate`: the `except asyncio.CancelledError` / `GeneratorExit` branch calls `self.abort(request_id)` — vLLM's version of `job.cancelled = True` in a `finally`.

**3. How HF detokenizes incrementally.** `transformers/generation/streamers.py`, class `TextStreamer`, method `put`: it keeps `self.token_cache` and `self.print_len`, decodes the whole cache each time, and emits `text[self.print_len:]` — but only up to the last space, or when the text ends in a full character. vLLM's equivalent is `vllm/v1/engine/detokenizer.py` (`IncrementalDetokenizer`; the slow path `SlowIncrementalDetokenizer.decode_next` shows the `prefix_offset` / `read_offset` pair your `Detokenizer` copies; the fast path uses the Rust tokenizer's `DecodeStream`). Then prove the problem exists:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ids = tok.encode("你好 🚀")                                   # a few tokens
print(ids, [tok.decode([i]) for i in ids])                   # per-token decode: expect at least one '�'
print(tok.decode(ids))                                       # whole-sequence decode: intact text
```

**4. What SSE looks like on the wire.** Run any SSE endpoint — vLLM's if you have one (`vllm serve Qwen/Qwen2.5-0.5B-Instruct` then `curl -N localhost:8000/v1/chat/completions -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"hi"}],"stream":true}' -H 'content-type: application/json'`) — and look at the raw frames: `data: {...}\n\n` per token, `data: [DONE]` at the end. Your `/generate` will produce the same shape with a simpler JSON body.

### Code

`engine.py` — everything M1 needs; M2 adds the batch builders to this same file.

```python
"""engine.py — wraps the Project 1 model: weight loading, forward, sampling, incremental detokenization.

Nothing in here knows about HTTP or scheduling. ModelRunner = "one forward pass";
sample = "logits -> token ids"; Detokenizer = "token ids -> text, safely".
"""
import torch
import torch.nn.functional as F


def load_qwen(model_name, device, dtype):
    """Build MyQwen from the HF checkpoint WITHOUT instantiating the HF model.

    Loading the safetensors file straight into our module halves peak startup memory
    (no second copy of the weights living in an HF model object)."""
    from transformers import AutoConfig, AutoTokenizer        # config.json + tokenizer only; no HF model class
    from huggingface_hub import hf_hub_download               # fetch one file from the Hub cache
    from safetensors.torch import load_file                   # zero-copy loader for .safetensors
    from qwen_from_scratch import MyQwen                      # Project 1 — your own model class

    config = AutoConfig.from_pretrained(model_name)           # num_hidden_layers, num_key_value_heads, ... (Step 0 numbers)
    tokenizer = AutoTokenizer.from_pretrained(model_name)     # byte-level BPE + chat template
    model = MyQwen(config)                                    # random weights, fp32, on CPU for now
    sd = load_file(hf_hub_download(model_name, "model.safetensors"))   # dict: HF parameter name -> tensor
    # Qwen2.5-0.5B ties lm_head to the embedding, so the checkpoint may carry an lm_head.* entry that
    # MyQwen has no parameter for. Drop it; strict=True then guarantees every OTHER key matched exactly.
    sd = {k: v for k, v in sd.items() if not k.startswith("lm_head.")}
    model.load_state_dict(sd, strict=True)                    # raises if any name differs — your naming bug, spelled out
    # Move to GPU and cast to fp16 in one call; .eval() disables dropout-style training behaviour.
    return model.to(device=device, dtype=dtype).eval(), tokenizer, config


class ModelRunner:
    """One forward pass = one engine step. Owns the model, tokenizer and stop ids.

    The scheduler (M2) calls runner.forward(...) exactly once per prefill and once per decode step."""

    def __init__(self, model, tokenizer, config, device, dtype):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.device, self.dtype = device, dtype
        # Pad id fills the left side of short rows in a batch (M2). Qwen defines one; fall back to 0 if not.
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # Two stop ids: Qwen-Instruct ends a turn with <|im_end|> (= eos_token_id, 151645) but can also
        # emit <|endoftext|> (151643). Either one means "this sequence is finished".
        self.stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")}
        # The three numbers the KV managers need to size storage: 24, 2, 64 for Qwen2.5-0.5B.
        self.num_layers = config.num_hidden_layers
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

    @classmethod
    def load(cls, model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cuda", dtype=torch.float16):
        """Convenience constructor: download/load weights, then wrap."""
        model, tok, cfg = load_qwen(model_name, device, dtype)
        return cls(model, tok, cfg, device, dtype)

    @torch.no_grad()                                           # inference only: no autograd graph, half the memory
    def forward(self, input_ids, position_ids, kv_cache=None, attention_mask=None):
        """input_ids/position_ids [B, T]; kv_cache list of 24 (K, V) [B, 2, L, 64] or None;
        attention_mask additive [B, 1, T, L+T] or None. Returns (logits [B, vocab] fp32, new_kv)."""
        logits, new_kv = self.model(input_ids, position_ids, kv_cache=kv_cache,
                                    attention_mask=attention_mask, last_only=True)   # last_only: never [B, T, vocab]
        # Sampling divides by temperature and softmaxes; do that in fp32 so small temperatures do not overflow fp16.
        return logits.float(), new_kv                      # logits: [B, vocab]

    def tokens_for_prompt(self, prompt):
        """User text -> chat-templated token ids. add_generation_prompt appends '<|im_start|>assistant\\n'
        so the model's first generated token is the start of the answer, not more of the user turn."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        # add_special_tokens=False: the template already contains every special token; do not add a BOS on top.
        return self.tokenizer.encode(text, add_special_tokens=False)


def sample(logits, temperatures, generator=None):
    """logits [B, vocab] fp32, temperatures [B]. temperature 0 → greedy for that row.

    Batched from day one so M2's scheduler can sample a whole step in one call."""
    greedy = logits.argmax(dim=-1)                              # [B] — the highest-scoring token per row
    # Avoid dividing by 0: clamp temperature to a tiny positive; rows with t == 0 are overridden below anyway.
    t = temperatures.clamp(min=1e-5).unsqueeze(1)               # [B, 1] so it broadcasts over the vocab axis
    probs = torch.softmax(logits / t, dim=-1)                   # [B, vocab] — a proper distribution per row
    # multinomial draws one index per row according to probs. `generator` lets tests fix the randomness.
    sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)   # [B]
    # Per-row select: greedy where temperature == 0, sampled elsewhere. Still [B].
    return torch.where(temperatures == 0, greedy, sampled)


class Detokenizer:
    """Incremental decoding for byte-level BPE. Never decode a single token in isolation:
    a token can be half of a multi-byte UTF-8 character. Decode a rolling window and emit
    only the text that grew — and hold it back while it ends in the replacement char.

    Same two-offset scheme as vLLM's SlowIncrementalDetokenizer (prefix_offset / read_offset)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids = []             # every generated id so far (stop ids are filtered out before push)
        self.prefix_offset = 0    # start of the window we decode (a few tokens before read_offset, for context)
        self.read_offset = 0      # everything before this has been emitted

    def push(self, token_id):
        """Add one id; return the new text it completes (possibly '' if it is only half a character)."""
        self.ids.append(token_id)
        # Text of the window up to what we already emitted...
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=True)
        # ...and text of the window including the new token. Decoding both from the same start means
        # BPE merges across the boundary are handled identically in both strings.
        full = self.tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=True)
        # Emit only if something grew AND it does not end in '�' (an incomplete multi-byte character).
        if len(full) > len(prefix) and not full.endswith("�"):
            new_text = full[len(prefix):]                       # exactly the characters that appeared
            self.prefix_offset = self.read_offset               # slide the window forward
            self.read_offset = len(self.ids)                    # everything up to here is now emitted
            return new_text
        return ""                                               # hold: wait for the rest of the character
```

Loading from `model.safetensors` directly (instead of instantiating an HF model and copying) halves peak memory at startup; the keys already match because your `MyQwen` mirrors HF's `model.*` layout. `sample` is batched from day one (temperature per row) so M2 needs no change; plug P1's top-k/top-p in here if you want them. The stop set has two ids because Qwen-Instruct ends turns with `<|im_end|>` (`eos_token_id`) but can also emit `<|endoftext|>`.

`server.py` — M1 version. `SerialEngine` is throwaway (M2 replaces it with the scheduler); everything else survives.

```python
"""server.py (M1) — FastAPI + SSE streaming over a one-request-at-a-time engine loop.

Two roles in one process:
  - HTTP handlers (many, one per in-flight request): tokenize, enqueue a Job, forward tokens as SSE frames.
  - The engine task (exactly one): owns the GPU, pops Jobs, generates, pushes token ids into each Job's queue.
They only ever touch each other through asyncio.Queue objects.
"""
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager                      # turns an async generator into a lifespan hook

import torch
from fastapi import FastAPI, HTTPException, Request as HttpRequest   # renamed: our own `Request` arrives in M2
from fastapi.responses import StreamingResponse                 # Starlette's StreamingResponse, re-exported
from pydantic import BaseModel                                  # request-body validation

from engine import Detokenizer, ModelRunner, sample

HEARTBEAT_S = 5.0            # emit ': ping' after this long without a token (keeps proxies from closing us)
DISCONNECT_POLL_S = 0.5      # while tokens flow, check for a gone client at least this often


class GenRequest(BaseModel):
    """JSON body of POST /generate. pydantic rejects wrong types with a 422 before our code runs."""
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0


class Job:
    """One generation request as the engine sees it."""

    def __init__(self, prompt_ids, max_tokens, temperature):
        self.prompt_ids, self.max_tokens, self.temperature = prompt_ids, max_tokens, temperature
        self.out = asyncio.Queue()          # token ids → handler; None = finished (the sentinel)
        self.cancelled = False              # set by the handler when the client is gone; engine checks it every step


class SerialEngine:
    """M1 only: runs jobs one after another, one decode step per loop iteration."""

    def __init__(self, runner):
        self.runner = runner
        self.queue = asyncio.Queue()        # jobs from handlers, in arrival order

    async def run(self):
        """The background task. Lives for the life of the process."""
        while True:
            job = await self.queue.get()    # suspends here (loop free) when there is nothing to do
            if not job.cancelled:           # client may have left while the job sat in the queue
                await self._generate(job)

    async def _generate(self, job):
        r = self.runner
        ids = torch.tensor([job.prompt_ids], device=r.device)                       # [1, T] prompt ids
        pos = torch.arange(len(job.prompt_ids), device=r.device).unsqueeze(0)       # [1, T] positions 0..T-1
        temps = torch.tensor([job.temperature], device=r.device)                    # [1] per-row temperature
        logits, kv = r.forward(ids, pos, None, None)                  # PREFILL: logits [1, vocab], kv 24 × [1, 2, T, 64]
        for _ in range(job.max_tokens):
            tok = sample(logits, temps).item()                        # .item() = GPU sync; one per token, fine at batch 1
            job.out.put_nowait(tok)                                   # hand the id to the handler immediately
            if tok in r.stop_ids or job.cancelled:                    # finished, or the client left
                break
            await asyncio.sleep(0)                                    # let the event loop breathe: handlers run here
            n = kv[0][0].shape[2]                                     # tokens in the cache so far = position of the new token
            logits, kv = r.forward(torch.tensor([[tok]], device=r.device),          # [1, 1] the token just produced
                                   torch.tensor([[n]], device=r.device), kv, None)   # [1, 1] its position; DECODE step
        job.out.put_nowait(None)                                      # sentinel: stream is over


@asynccontextmanager
async def lifespan(app):
    """Runs once at startup (before yield) and once at shutdown (after yield)."""
    runner = ModelRunner.load(os.environ.get("MV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),   # MV_MODEL: which checkpoint
                              device=os.environ.get("MV_DEVICE", "cuda"))                 # MV_DEVICE: cuda or cpu
    app.state.runner = runner                                     # app.state is FastAPI's per-process bag for shared objects
    app.state.engine = SerialEngine(runner)
    task = asyncio.create_task(app.state.engine.run())            # the ONE task that owns the GPU
    yield                                                         # server runs while suspended here
    task.cancel()                                                 # shutdown: stop the engine task


app = FastAPI(lifespan=lifespan)


def sse(event: dict) -> str:
    """One SSE frame: 'data: <json>' followed by a blank line."""
    return f"data: {json.dumps(event)}\n\n"


@app.post("/generate")
async def generate(body: GenRequest, http: HttpRequest):
    runner, engine = app.state.runner, app.state.engine
    if not 1 <= body.max_tokens <= 2048:
        raise HTTPException(400, "max_tokens out of range")
    job = Job(runner.tokens_for_prompt(body.prompt), body.max_tokens, body.temperature)   # tokenize on the loop (fast)
    await engine.queue.put(job)                                   # enqueue; the engine task will pick it up

    async def stream():
        """Async generator: each `yield` is one chunk on the wire. Runs concurrently with the engine task."""
        detok = Detokenizer(runner.tokenizer)
        t0, n = time.perf_counter(), 0                            # for the final done frame: elapsed + token count
        last_check = time.perf_counter()                          # last time we polled is_disconnected()
        try:
            while True:
                try:
                    # Wait for the next token id, but at most HEARTBEAT_S. asyncio.timeout (3.11+) is a context
                    # manager that cancels the awaited get() and raises TimeoutError when the time is up.
                    async with asyncio.timeout(HEARTBEAT_S):
                        tok = await job.out.get()
                except TimeoutError:
                    if await http.is_disconnected():              # quiet AND gone: stop
                        return
                    yield ": ping\n\n"                          # SSE comment = heartbeat; clients ignore it
                    continue
                now = time.perf_counter()
                if now - last_check > DISCONNECT_POLL_S:      # do NOT rely on Starlette cancelling us
                    last_check = now
                    if await http.is_disconnected():          # cheap non-blocking peek at the ASGI receive channel
                        return
                if tok is None:                                   # sentinel from the engine: generation finished
                    yield sse({"done": True, "tokens": n, "seconds": round(time.perf_counter() - t0, 3)})
                    return
                n += 1
                if tok in runner.stop_ids:                        # never let <|im_end|> reach the detokenizer
                    continue
                text = detok.push(tok)                            # '' while a multi-byte character is incomplete
                if text:
                    yield sse({"text": text})
        finally:
            job.cancelled = True          # reached on normal end, client disconnect, or error — engine stops next step

    # media_type sets Content-Type: text/event-stream. Cache-Control stops caches from holding the stream;
    # X-Accel-Buffering: no tells nginx-style proxies to pass chunks through instead of buffering them.
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

`asyncio.timeout` rather than `asyncio.wait_for`: in 3.11 `wait_for` wraps the awaitable in an inner task and can swallow an outer cancellation when the inner completes at the same moment — precisely the situation a token queue creates. `X-Accel-Buffering: no` tells nginx-style proxies not to buffer the stream.

### Test

```bash
# Start the server. `server:app` = module server.py, object `app`. --port 8000 = listen port (default anyway).
uvicorn server:app --port 8000
# in another terminal:
# -N  : no output buffering, so curl prints each SSE frame as it arrives (without it you see nothing until the end)
# -X POST, -H content-type, -d body: a JSON POST to /generate
curl -N -X POST localhost:8000/generate -H 'content-type: application/json' \
     -d '{"prompt": "Write two sentences about GPUs, then say 你好 and add a 🚀.", "max_tokens": 80}'
```

Then the two things curl cannot show you. Heartbeats: temporarily set `HEARTBEAT_S = 0.2`, send a 400-token prompt, and watch `: ping` lines appear before the first token. Disconnect: send `max_tokens: 2000`, hit Ctrl-C after a few tokens, and immediately send a 5-token request — it must answer within a second, not after the abandoned request finishes.

### Expected

Text arrives word by word; the Chinese characters and the emoji come out intact (each is 1–3 tokens; a naive per-token decode would show `�`). The `done` frame reports `tokens` and elapsed seconds; tokens/seconds should be within ~10% of your P1 KV-cached tok/s — the SSE layer costs almost nothing. The post-disconnect request answers in well under a second: `SerialEngine` saw `job.cancelled` at its next step.

### Reading a failure

| Symptom | Where the bug is |
|---|---|
| `�` appears in the stream | Decoding token-by-token, or `Detokenizer.push` returning text that ends in `�` |
| Whole response arrives at once at the end | A buffering proxy, or the client (`curl` without `-N`; browsers need `EventSource` or `fetch` streaming) |
| Second request waits for the abandoned one | `job.cancelled` never set — `finally` not reached because the generator never resumed; you removed the periodic `is_disconnected()` poll |
| Server unresponsive to `/docs` while generating | The engine step runs inside the handler, or the `await asyncio.sleep(0)` is missing from the engine loop |
| Output is `<|im_end|>` text or generation never stops | Stop ids wrong: print `runner.stop_ids`, expect `{151645, 151643}` |
| Response starts with "You are Qwen…" | You are decoding the prompt; the detokenizer must only see generated ids |

### Close the milestone

```
git commit -m "M1: FastAPI /generate with SSE streaming, heartbeats, disconnect cancellation over the P1 engine"
```

README: an architecture sketch (handler ↔ queue ↔ engine task), a curl transcript showing multibyte characters arriving intact, and one paragraph on why disconnect handling is your responsibility (cite the measurement: tokens generated after disconnect, before and after the fix).

---

## Milestone 2 — Request queue + step-loop scheduler (continuous batching)

### What it is

**Static batching** collects a batch, runs it to completion, then takes the next batch. Two costs: a request arriving one step after a batch started waits for the whole batch (TTFT ≈ batch duration), and a sequence that finishes early leaves its slot idle until the last one is done. **Continuous (iteration-level) batching** — Orca's idea, vLLM's default — reschedules at every step: finished sequences leave, waiting ones join, and the GPU sees a full batch as often as capacity allows. It is the single largest throughput win in serving, and it is the milestone of this project.

The step loop is:

```
step():
  admit()   waiting → running while capacity allows; one PREFILL forward over the admitted batch
  decode()  one DECODE forward over every running sequence (one new token each)
  retire()  free finished sequences, close their streams
```

Prefill and decode are **separate forward passes** in each iteration. Prefill is compute-bound (T tokens per row, big matmuls); decode is memory-bound (one token per row). Real vLLM puts both in *one* forward pass and, with **chunked prefill** (splitting a long prompt into fixed-size pieces processed over several steps, so no single step stalls the running decodes for more than a token budget), those are refinements of this loop, not different loops; the cost of not having them is visible in your ITL p99 (M6) and you will be able to point at it.

**Variable-length batches need padding and masks.** *Padding* means filling short rows with a dummy token so every row has the same length `T` and the batch is a rectangular `[B, T]` tensor; the *attention mask* then tells the model which columns are real. Prompts in one prefill batch have different lengths; the model wants `[B, T]`. Use **left** padding (dummies at the start of the row): real tokens end at column `T-1` for every row, so `logits[:, -1]` (what `last_only=True` returns) is each row's next-token logit without any gather. Three tensors follow from that choice: `position_ids[i] = arange(len_i)` shifted right past the pad (pads get 0), so RoPE sees true positions; a mask that is causal over real tokens and forbids real tokens from attending to pad columns (pad *query* rows are allowed to see the pad columns so no row is fully masked — a fully masked row softmaxes to NaN); and, when writing K/V into storage, skip the first `pad_i` positions of row *i*. The KV store itself is **left-aligned** (position 0 at index 0), which makes decode trivial: gather every row's past to the batch maximum `L`, mask out `[len_i, L)`, and place the new token at index `L` with `position_ids[i] = len_i`.

**Capacity in M2 is a slot.** `SlotKV` preallocates `num_slots` contiguous regions of `max_len` tokens; an admitted request takes one slot for its whole life. This is how everyone served before PagedAttention, and it is the baseline M3 measures against: a request that will generate 40 tokens reserves 2048.

**Poisson arrivals.** Benchmarking with "fire N requests at once" tells you nothing about a serving system; with Poisson arrivals (exponential inter-arrival times at rate λ) you can sweep offered load and watch latency respond. `bench/loadgen.py` does this and reports throughput, p50/p99 TTFT, end-to-end latency, and ITL.

### Look first

**1. vLLM's scheduler — the thing you are miniaturizing.** Open `vllm/v1/core/sched/scheduler.py`, class `Scheduler`, method `schedule()`. Read it top to bottom once (it is long; skip the multimodal/encoder and speculative-decoding branches). What to notice:

- There is no "prefill phase" and "decode phase". One list `self.running`, and for each request `num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens` — a decode request has 1 new token, a half-prefilled request has many. `num_computed_tokens` is exactly your `num_cached`, and the invariant "everything up to `num_computed_tokens` is in the cache" is the one this milestone builds on.
- `token_budget = self.max_num_scheduled_tokens` is decremented as requests are scheduled; `num_new_tokens = min(num_new_tokens, token_budget)` is chunked prefill in one line. Your `max_prefill_tokens` is the same budget applied only to prefill.
- Running requests are scheduled *before* waiting ones (decodes are protected; new prompts only fill what is left) — the opposite order from your `admit → decode`, and the reason vLLM's ITL does not spike on admission.
- The waiting loop calls `self.kv_cache_manager.get_computed_blocks(request)` (prefix caching, M5) and `allocate_slots(...)`; on success `request.status = RequestStatus.RUNNING` and `self.running.append(request)`. If allocation fails it `break`s — FCFS head-of-line blocking, same as yours.
- The return value is a `SchedulerOutput` (`vllm/v1/core/sched/output.py`): lists of new and cached requests plus `num_scheduled_tokens` per request. The model runner never sees `Request` objects.

Then `update_from_output(scheduler_output, model_runner_output)` in the same file: for each request it appends the sampled ids (`request.append_output_token_ids`), calls `check_stop(request, self.max_model_len)` (stop token or length cap → `FINISHED_STOPPED` / `FINISHED_LENGTH_CAPPED`), frees finished requests via `self._free_request(request)` → `self.kv_cache_manager.free(request)`, and builds the `EngineCoreOutput` per request that the API server turns into SSE. That method is your `_append` + `_retire`.

**2. The step loop itself.** `vllm/v1/engine/core.py`, class `EngineCore`, method `step()`: three lines — `scheduler_output = self.scheduler.schedule()`, `model_output = self.model_executor.execute_model(scheduler_output)`, `engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)`. Then `EngineCoreProc.run_busy_loop()` below it: `_process_input_queue()` (drain new requests that arrived over ZMQ) then `_process_engine_step()`. Your `engine_loop` is that busy loop with an `asyncio.Event` instead of a ZMQ socket.

**3. The request object.** `vllm/v1/request.py`: `class Request` (`prompt_token_ids`, `_output_token_ids`, `num_computed_tokens`, `status`, `arrival_time`, `num_preemptions`, `block_hashes`) and `class RequestStatus` (`WAITING`, `RUNNING`, `PREEMPTED`, `FINISHED_STOPPED`, `FINISHED_LENGTH_CAPPED`, `FINISHED_ABORTED`, ...). Your `Request` dataclass is this with `finish_reason` + `cancelled` in place of the enum.

**4. How the real systems avoid padding, and how HF builds the mask you are about to build.** `vllm/v1/worker/gpu_model_runner.py`, method `_prepare_inputs`: the batch is *flattened* to `[total_num_scheduled_tokens]` with `query_start_loc` (where each request's tokens begin) and `seq_lens` — no `[B, T]` rectangle, no pad tokens, and the attention kernel (FlashAttention "varlen") takes those offsets. Padding is the price of using a plain `[B, T]` attention. Then see HF do the padded version, which is what you will reproduce:

```python
import torch
from transformers import AutoTokenizer
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok.padding_side = "left"                                   # HF warns if you generate with right padding — same reason as ours
enc = tok(["Hi", "The capital of France is"], padding=True, return_tensors="pt")
print(enc.input_ids)              # [2, T]: row 0 starts with pad ids (151643), real tokens end at column T-1 in BOTH rows
print(enc.attention_mask)         # [2, T]: 0 over pads, 1 over real tokens — a 2-D "is this column real" mask
# HF turns that 2-D mask into the 4-D additive one the model adds to scores: [B, 1, T, T], 0 = attend, finfo.min = masked.
m4 = AttentionMaskConverter(is_causal=True).to_4d(enc.attention_mask, query_length=enc.input_ids.shape[1],
                                                  dtype=torch.float32, key_value_length=enc.input_ids.shape[1])
print(m4.shape); print((m4[0, 0] == 0).int())   # row 0: a lower triangle that starts AFTER the pad columns
```

Notice the pad *rows* in `m4[0, 0]` (the first few query rows of the padded sequence): HF leaves them attending to themselves rather than fully masked — same reason your builder allows pad queries to see pad keys. Also note `position_ids`: HF's `generate` computes them as `attention_mask.cumsum(-1) - 1` (grep `cumsum` in `transformers/generation/utils.py`), which is "0..len_i-1 shifted past the pads" — your `build_prefill_batch` does the same arithmetic explicitly.

**5. Poisson arrivals in the real load generator.** `vllm/benchmarks/serve.py` (older checkouts: `benchmarks/benchmark_serving.py`), function `get_request`: it sleeps `np.random.gamma(shape=burstiness, scale=theta)` between requests, and with `burstiness=1` a gamma *is* an exponential with mean `1/request_rate` — identical to `random.expovariate(rate)` below. `vllm bench serve --request-rate 4` is your `--rate 4`.

**6. See the event loop being blocked.** Before you write `engine_loop`, run the M1 server with `PYTHONASYNCIODEBUG=1 uvicorn server:app` and add `asyncio.get_running_loop().slow_callback_duration = 0.005` at the top of `lifespan`. Every engine step now logs `Executing <Task ...> took 0.0xx seconds` — the exact amount of time during which no handler can run. That number is the "one process" divergence at the end of this document, measured.

### Code

Append to `engine.py`:

```python
def build_prefill_batch(token_lists, past_lens, L_past, pad_id, device, dtype):
    """Left-pad variable-length prompts into one [B, T] batch.
    past_lens[i] = tokens of row i already in the KV cache (0 unless prefix caching / M5),
    L_past       = padded length of the gathered past (0 if no past).
    Returns input_ids, position_ids [B,T], additive mask [B,1,T,L_past+T], pads [B].

    Key layout of the returned mask, per row i (columns = keys):
        [0, past_lens[i])            real cached past         -> 0 (attend)
        [past_lens[i], L_past)       other rows' past (garbage) -> neg (masked)
        [L_past, L_past+pad_i)       this row's pad tokens     -> neg for real queries, 0 for pad queries
        [L_past+pad_i, L_past+T)     this row's real new tokens -> causal (0 on and below the diagonal)
    """
    B, T = len(token_lists), max(len(t) for t in token_lists)          # T = longest prompt in the batch
    neg = torch.finfo(dtype).min                                        # "masked" value: most negative fp16 (-65504)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)            # [B, T] all pad, real tokens written on the right
    position_ids = torch.zeros((B, T), dtype=torch.long)                # [B, T] pads keep position 0 (masked anyway)
    mask = torch.full((B, 1, T, L_past + T), neg, dtype=torch.float32)  # [B, 1, T, L_past+T] start fully masked, then open up
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))             # [T, T] lower triangle incl. diagonal: query t sees keys <= t
    pads = []
    for i, toks in enumerate(token_lists):
        n, pad = len(toks), T - len(toks)                               # n real tokens, pad dummies on the left
        pads.append(pad)
        input_ids[i, pad:] = torch.tensor(toks)                         # real tokens occupy columns [pad, T)
        position_ids[i, pad:] = torch.arange(n) + past_lens[i]          # positions continue after the cached past
        mask[i, 0, :, :past_lens[i]] = 0.0                       # everyone sees the real past (columns [0, past_len_i))
        mask[i, 0, :, L_past:][causal] = 0.0                     # causal over the new tokens (whole [T, T] block incl. pads)…
        mask[i, 0, pad:, L_past:L_past + pad] = neg              # …but real tokens never see pads (pad queries still can:
                                                                 #    a fully masked row would softmax to NaN)
    return (input_ids.to(device), position_ids.to(device), mask.to(device), pads)


def build_decode_batch(last_tokens, cached_lens, L, device, dtype):
    """One new token per row. Row i attends to its first cached_lens[i] past slots
    plus its own new token at index L (garbage in [cached_lens[i], L) is masked).

    last_tokens [B] python ints (each row's most recent token), cached_lens [B] ints, L = gathered past length.
    Returns input_ids [B, 1], position_ids [B, 1], additive mask [B, 1, 1, L+1]."""
    B = len(last_tokens)
    neg = torch.finfo(dtype).min
    input_ids = torch.tensor(last_tokens, dtype=torch.long).unsqueeze(1)          # [B, 1] the token to feed
    position_ids = torch.tensor(cached_lens, dtype=torch.long).unsqueeze(1)       # [B, 1] its position = tokens before it
    ar = torch.arange(L + 1).unsqueeze(0)                                 # [1, L+1] key column indices 0..L
    # valid[i, c] = "column c belongs to row i's real past" OR "c is the new token's own column (L)".
    valid = (ar < torch.tensor(cached_lens).unsqueeze(1)) | (ar == L)     # [B, L+1] bool via broadcasting
    mask = torch.where(valid, 0.0, neg).view(B, 1, 1, L + 1)              # [B, 1, 1, L+1]: one query row per sequence
    return input_ids.to(device), position_ids.to(device), mask.to(device)
```

`past_lens`/`L_past` are zero everywhere until M5; writing the general version now means M5 changes nothing here.

`kv_manager.py` — the interface every KV backend implements, and the M2 backend:

```python
"""kv_manager.py — where the KV cache lives and how it is handed to attention.

All managers share one interface so the scheduler does not care which is plugged in:
    allocate(req, n_tokens) -> bool   make sure req can hold n_tokens of KV; False if no room
    free(req)                          release everything req holds
    gather(reqs) -> (past_kv, L)       contiguous per-layer (K, V) [B, kv_heads, L, head_dim]
    write_prefill(reqs, new_kv, pads, L_past)
    write_decode(reqs, new_kv, L)
    stats() -> dict

Storage layout for every backend: self.k[layer] and self.v[layer] are big preallocated tensors on the GPU.
gather() copies each running sequence's K/V out of them into the [B, kv_heads, L, head_dim] shape the model
expects; write_*() copies the model's freshly computed K/V for the new tokens back in.
"""
import math
from collections import OrderedDict
import torch
import torch.nn.functional as F


class SlotKV:
    """M2: contiguous preallocation. Every admitted sequence reserves max_len tokens.

    self.k / self.v: [num_layers, num_slots, kv_heads, max_len, head_dim]
    A sequence in slot s has token position p at self.k[layer][s, :, p, :]  (left-aligned: position 0 at index 0)."""

    def __init__(self, num_slots, max_len, num_layers, kv_heads, head_dim, device, dtype):
        shape = (num_layers, num_slots, kv_heads, max_len, head_dim)   # e.g. [24, 42, 2, 2048, 64] fp16 = 1 GiB
        self.k = torch.zeros(shape, device=device, dtype=dtype)        # allocated ONCE at startup; never grows
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len, self.num_layers = max_len, num_layers
        self.free_slots = list(range(num_slots))                       # slot ids nobody owns
        self.owner = {}                       # slot -> req  (who holds which slot; used by stats)

    def allocate(self, req, n_tokens):
        """Make sure req can hold n_tokens. A slot holds max_len, so beyond the first call this is a range check."""
        if n_tokens > self.max_len:
            return False                                                # will never fit — caller must not admit it
        if req.slot is None:                                            # first call for this request: grab a slot
            if not self.free_slots:
                return False                                            # no room right now; caller may retry next step
            req.slot = self.free_slots.pop()
            self.owner[req.slot] = req
        return True                                                     # already has a slot: max_len covers it

    def free(self, req):
        """Return the slot. Safe to call twice (second call is a no-op)."""
        if req.slot is not None:
            self.owner.pop(req.slot)
            self.free_slots.append(req.slot)
            req.slot = None

    def gather(self, reqs):
        """Pull the past of every request in `reqs` into one contiguous tensor per layer.
        Returns (past, L): past[layer] = (K, V) each [B, kv_heads, L, head_dim], L = longest past in the batch.
        Rows shorter than L carry stale data in [num_cached_i, L) — the decode mask hides it."""
        L = max(r.num_cached for r in reqs)
        slots = torch.tensor([r.slot for r in reqs], device=self.k.device)          # [B] slot ids
        # Advanced indexing with `slots` on dim 0 COPIES the selected rows: [B, kv_heads, L, head_dim] per layer.
        past = [(self.k[l][slots, :, :L], self.v[l][slots, :, :L]) for l in range(self.num_layers)]
        return past, L

    def write_prefill(self, reqs, new_kv, pads, L_past):
        """Copy the K/V the model just computed for each row's NEW tokens into its slot.
        new_kv[layer] = (K, V) each [B, kv_heads, L_past + T, head_dim] (past ++ padded new tokens)."""
        for i, r in enumerate(reqs):
            # s = where this row's new tokens start in its slot (= tokens already cached);
            # n = how many real new tokens it has = total width - past - its own left padding.
            s, n = r.num_cached, new_kv[0][0].shape[2] - L_past - pads[i]
            for l, (k, v) in enumerate(new_kv):
                # Skip the past columns and the pad columns; what remains is [kv_heads, n, head_dim].
                self.k[l][r.slot, :, s:s + n] = k[i, :, L_past + pads[i]:]
                self.v[l][r.slot, :, s:s + n] = v[i, :, L_past + pads[i]:]

    def write_decode(self, reqs, new_kv, L):
        """Store the single new token of every row. new_kv[layer] K/V are [B, kv_heads, L+1, head_dim];
        column L is the new token (columns < L are the gathered past we already have)."""
        slots = torch.tensor([r.slot for r in reqs], device=self.k.device)          # [B]
        pos = torch.tensor([r.num_cached for r in reqs], device=self.k.device)      # [B] where each row's new token goes
        for l, (k, v) in enumerate(new_kv):
            # Paired advanced indices on dims 0 and 2: row i lands at (slots[i], :, pos[i]) — one strided scatter per layer.
            self.k[l][slots, :, pos] = k[:, :, L]                       # k[:, :, L] is [B, kv_heads, head_dim]
            self.v[l][slots, :, pos] = v[:, :, L]

    def stats(self):
        """Utilization = tokens actually holding KV / tokens reserved. This is M3's headline comparison."""
        used = sum(r.num_cached for r in self.owner.values())
        reserved = len(self.owner) * self.max_len                       # every owner reserves the full max_len
        return {"tokens_used": used, "tokens_reserved": reserved,
                "utilization": used / reserved if reserved else 0.0}
```

`self.k[l][slots, :, pos] = k[:, :, L]` is one strided scatter per layer: advanced indices on dims 0 and 2 are paired, so row *i* lands at `(slots[i], :, pos[i])`. `gather` with `slots` is a copy (advanced indexing always is) — remember that when M3 asks what paging costs.

`scheduler.py`:

```python
"""scheduler.py — the request lifecycle and the iteration-level step loop.

Mirrors vllm/v1/core/sched/scheduler.py in miniature:
    Scheduler.step()  ≈ schedule() + execute_model() + update_from_output(), fused
    Request           ≈ vllm/v1/request.py::Request (num_cached ≈ num_computed_tokens)
"""
import asyncio
import itertools
import time
from collections import deque                  # O(1) popleft/appendleft: the waiting queue
from dataclasses import dataclass, field

import torch

from engine import build_decode_batch, build_prefill_batch, sample

_ids = itertools.count()                       # monotonically increasing request ids


@dataclass(eq=False)                       # identity semantics: requests live in sets and lists (no field-wise ==)
class Request:
    prompt_ids: list                           # chat-templated prompt token ids
    max_tokens: int                            # generation cap → finish_reason "length"
    temperature: float = 0.0                   # 0 = greedy
    out: asyncio.Queue = None                  # token ids to the HTTP handler; None = done. (None here = no consumer, e.g. tests)
    id: int = field(default_factory=lambda: next(_ids))
    arrival: float = field(default_factory=time.perf_counter)   # set at construction in the handler → TTFT includes queue wait
    output_ids: list = field(default_factory=list)              # generated so far
    num_cached: int = 0                        # tokens whose K/V are in the cache  (vLLM: num_computed_tokens)
    slot: int = None                           # SlotKV: which slot (M2)
    block_table: list = field(default_factory=list)   # BlockKV: physical block ids in logical order (M3+)
    first_token_time: float = None             # when the first output token was sampled (TTFT stop clock)
    last_token_time: float = None              # when the latest token was sampled (ITL start clock)
    finish_reason: str = None                  # "stop" | "length" | None while running
    cancelled: bool = False                    # client disconnected; treated as finished

    @property
    def tokens(self):                          # everything the model has seen or produced
        return self.prompt_ids + self.output_ids

    @property
    def finished(self):
        return self.finish_reason is not None or self.cancelled


class Scheduler:
    def __init__(self, runner, kv, max_num_seqs=16, max_prefill_tokens=2048, static=False):
        self.runner, self.kv = runner, kv                              # the model + the KV backend (any of the three)
        # max_num_seqs: cap on the running batch (vLLM --max-num-seqs).
        # max_prefill_tokens: cap on prompt tokens prefilled in ONE step — bounds the decode stall (vLLM's token budget).
        # static: baseline mode — do not admit until the running batch has fully drained.
        self.max_num_seqs, self.max_prefill_tokens, self.static = max_num_seqs, max_prefill_tokens, static
        self.waiting = deque()                                         # FCFS queue (vLLM: self.waiting)
        self.running = []                                              # the current batch (vLLM: self.running)
        self.wake = asyncio.Event()                                    # engine_loop sleeps on this when idle

    # ---- API-server side -------------------------------------------------
    def add(self, req):
        """Called from the HTTP handler. Enqueue and wake the engine loop if it is idle."""
        self.waiting.append(req)
        self.wake.set()

    def has_work(self):
        return bool(self.waiting or self.running)

    # ---- one engine iteration ------------------------------------------
    def step(self):
        """One iteration: admit (+ prefill), decode, retire. Synchronous; blocks the event loop while it runs."""
        if not (self.static and self.running):        # static batching: drain before admitting
            self._admit()
        if self.running:
            self._decode()
        self._retire()

    def _admit(self):
        """Move waiting requests into `running` while there is a seat, a prefill-token budget, and KV room;
        then prefill all of them in one batch."""
        batch, budget = [], self.max_prefill_tokens
        while self.waiting and len(self.running) + len(batch) < self.max_num_seqs:
            req = self.waiting[0]                                       # peek at the head (FCFS)
            if req.cancelled:                                   # client left while queued
                self.waiting.popleft()
                continue
            n_new = len(req.tokens) - req.num_cached                    # tokens that must be prefilled (all, unless M5 hit)
            # Fits if: it stays within the prefill budget (a single huge prompt is allowed when the batch is empty,
            # otherwise it could never run) AND the KV backend can reserve room for its full current length.
            fits = (n_new <= budget or not batch) and self.kv.allocate(req, len(req.tokens))
            if not fits:                                        # FCFS: head-of-line blocks
                self.kv.free(req)                                       # undo any partial reservation (M5 partial prefix match)
                req.num_cached = 0
                break
            self.waiting.popleft()
            batch.append(req)
            budget -= n_new
        if batch:
            self._prefill(batch)
            self.running.extend(batch)

    def _prefill(self, batch):
        """One forward pass over every admitted request's NOT-YET-CACHED tokens; samples each one's first token."""
        new_tokens = [r.tokens[r.num_cached:] for r in batch]          # per row: the tokens to compute K/V for
        # If any row already has cached tokens (M5 prefix hit, or never in M2), gather the past for the whole batch.
        past, L = self.kv.gather(batch) if any(r.num_cached for r in batch) else (None, 0)
        input_ids, position_ids, mask, pads = build_prefill_batch(     # [B, T], [B, T], [B, 1, T, L+T], [B]
            new_tokens, [r.num_cached for r in batch], L,
            self.runner.pad_id, self.runner.device, self.runner.dtype)
        logits, new_kv = self.runner.forward(input_ids, position_ids, past, mask)   # logits [B, vocab]; new_kv 24 × [B, 2, L+T, 64]
        self.kv.write_prefill(batch, new_kv, pads, L)                  # store the new tokens' K/V
        for r in batch:
            r.num_cached = len(r.tokens)                                # invariant restored: everything seen is cached
        self._append(batch, logits)                                    # sample token #1 for each row

    def _decode(self):
        """One forward pass over every running request with its single newest token; samples the next one."""
        batch = list(self.running)
        for r in batch:                                         # room for one more token each
            if not self.kv.allocate(r, r.num_cached + 1):
                raise RuntimeError("out of KV memory — M4 replaces this with preemption")
        past, L = self.kv.gather(batch)                                 # 24 × ([B, 2, L, 64], same); L = longest past
        input_ids, position_ids, mask = build_decode_batch(            # [B, 1], [B, 1], [B, 1, 1, L+1]
            [r.tokens[-1] for r in batch], [r.num_cached for r in batch], L,
            self.runner.device, self.runner.dtype)
        logits, new_kv = self.runner.forward(input_ids, position_ids, past, mask)   # new_kv 24 × [B, 2, L+1, 64]
        self.kv.write_decode(batch, new_kv, L)                          # store column L (the new token) for every row
        for r in batch:
            r.num_cached += 1
        self._append(batch, logits)

    def _append(self, batch, logits):
        """Sample one token per row, record timing, decide finish, and push the id to each request's handler."""
        temps = torch.tensor([r.temperature for r in batch], device=logits.device)   # [B]
        toks = sample(logits, temps).tolist()                   # the one GPU sync per step (.tolist() waits for the GPU)
        now = time.perf_counter()
        for r, t in zip(batch, toks):
            r.output_ids.append(t)
            if r.first_token_time is None:                              # first token → TTFT clock stops
                r.first_token_time = now
            r.last_token_time = now
            if t in self.runner.stop_ids:
                r.finish_reason = "stop"                                # model ended its turn
            elif len(r.output_ids) >= r.max_tokens:
                r.finish_reason = "length"                              # hit the cap
            if r.out is not None:
                r.out.put_nowait(t)                                     # non-blocking hand-off to the SSE generator

    def _retire(self):
        """Remove finished (or cancelled) requests from the batch, free their KV, close their streams."""
        for r in [r for r in self.running if r.finished]:              # copy: we mutate self.running inside
            self.running.remove(r)
            self.kv.free(r)
            if r.out is not None:
                r.out.put_nowait(None)                                  # the sentinel: handler sends the done frame
```

The invariant that makes this small: **`num_cached == len(tokens) - 1` for every running request.** Prefill computes K/V for every token not yet cached (`tokens[num_cached:]`) and samples one new token from the last logit; decode feeds `tokens[-1]` at position `num_cached`. The same prefill path therefore serves a fresh request (`num_cached = 0`), a resumed one after preemption (M4: `num_cached = 0`, `tokens` includes what it had generated), and a prefix-cache hit (M5: `num_cached = 16k`). Admission is FCFS with head-of-line blocking; `max_prefill_tokens` bounds how many prompt tokens one step may prefill, which bounds the decode stall. `static=True` is the baseline: no admission until the batch has drained.

`server.py` — replace everything between the imports and `app = FastAPI(...)`:

```python
from engine import Detokenizer, ModelRunner
from kv_manager import SlotKV
from scheduler import Request, Scheduler

HEARTBEAT_S = 5.0                                               # as in M1
DISCONNECT_POLL_S = 0.5
MAX_MODEL_LEN = int(os.environ.get("MV_MAX_MODEL_LEN", 2048))   # prompt + max_tokens cap; also SlotKV's max_len


class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0


def build_kv(runner, kv_mb):
    """Turn a memory budget in MiB into a SlotKV sized for it."""
    # Bytes per token = K and V (2) × layers × kv_heads × head_dim × bytes per element = 12,288 for this model (Step 0).
    bytes_per_token = 2 * runner.num_layers * runner.kv_heads * runner.head_dim * runner.dtype.itemsize
    tokens = int(kv_mb * 2**20 // bytes_per_token)                    # how many tokens the budget holds
    return SlotKV(num_slots=max(1, tokens // MAX_MODEL_LEN), max_len=MAX_MODEL_LEN,   # each slot = one full-length sequence
                  num_layers=runner.num_layers, kv_heads=runner.kv_heads,
                  head_dim=runner.head_dim, device=runner.device, dtype=runner.dtype)


async def engine_loop(sched):
    """The single background task that owns the GPU. ≈ EngineCoreProc.run_busy_loop in vLLM."""
    while True:
        if not sched.has_work():
            sched.wake.clear()                  # nothing to do: sleep until Scheduler.add() sets the event
            await sched.wake.wait()
        sched.step()                            # synchronous ~10–15 ms; the loop is blocked for its duration
        await asyncio.sleep(0)                  # give handlers a turn between steps


@asynccontextmanager
async def lifespan(app):
    runner = ModelRunner.load(os.environ.get("MV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
                              device=os.environ.get("MV_DEVICE", "cuda"))
    kv = build_kv(runner, float(os.environ.get("MV_KV_MB", 1024)))          # MV_KV_MB: KV budget in MiB (default 1 GiB)
    sched = Scheduler(runner, kv,
                      max_num_seqs=int(os.environ.get("MV_MAX_SEQS", 16)),  # MV_MAX_SEQS: batch cap
                      static=os.environ.get("MV_STATIC") == "1")            # MV_STATIC=1: static-batching baseline
    app.state.runner, app.state.kv, app.state.sched = runner, kv, sched
    task = asyncio.create_task(engine_loop(sched))
    yield
    task.cancel()
```

and the top of the handler becomes:

```python
@app.post("/generate")
async def generate(body: GenRequest, http: HttpRequest):
    runner, sched = app.state.runner, app.state.sched
    prompt_ids = runner.tokens_for_prompt(body.prompt)                     # chat template + tokenize
    # Reject anything that could never fit a slot: it would sit at the head of the FCFS queue forever.
    if len(prompt_ids) + body.max_tokens > MAX_MODEL_LEN or body.max_tokens < 1:
        raise HTTPException(400, f"prompt + max_tokens must be <= {MAX_MODEL_LEN}")
    req = Request(prompt_ids=prompt_ids, max_tokens=body.max_tokens,
                  temperature=body.temperature, out=asyncio.Queue())       # arrival timestamp is taken here
    sched.add(req)                                                         # into `waiting`; wakes the engine loop
```

Inside `stream()`, `job` becomes `req`, and the `done` frame gains `"finish_reason": req.finish_reason, "ttft_ms": round(1000 * (req.first_token_time - req.arrival), 1)`. Add:

```python
@app.get("/stats")
async def stats():
    """Point-in-time snapshot for humans and for bench/kv_waste.py. /metrics (M6) is the Prometheus version."""
    s = app.state.sched
    return {"waiting": len(s.waiting), "running": len(s.running), **app.state.kv.stats()}
```

The `MAX_MODEL_LEN` check matters: a request that can never fit would sit at the head of the FCFS queue forever and block everyone behind it. vLLM rejects these at the API layer for the same reason.

Note what the engine loop is: `sched.step()` is synchronous and blocks the event loop for the length of a GPU step (~10–15 ms), then yields once. Handlers run between steps. That is enough for 16 streams; it is also the honest reason vLLM moved the engine to another process (see the divergences section).

`bench/loadgen.py`:

```python
"""bench/loadgen.py — Poisson-arrival load generator against /generate.

python bench/loadgen.py --rate 2 --n 60 --label continuous
Writes bench/results/<label>_rate<rate>.json with throughput and latency percentiles.

One asyncio task per request; a main loop that starts them with exponential gaps (Poisson process at `rate`).
"""
import argparse, asyncio, json, os, random, statistics, time
import httpx                                                    # async HTTP client with streaming support

PROMPTS = [
    "Explain what a KV cache is in two sentences.",
    "Write a haiku about GPUs.",
    "List five uses of asyncio in Python and explain each briefly.",
    "Summarize the plot of a heist movie you invent, in one paragraph.",
    "Why is decode memory-bound? Give the arithmetic.",
    "Describe continuous batching to a backend engineer.",
    "What is paged attention and why does it need a custom kernel?",
    "Give me a recipe for a quick dinner using rice and eggs.",
]


def pct(xs, p):
    """p-th percentile by nearest rank: sort, then index at p% of the way through."""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


async def one(client, url, prompt, max_tokens, stats):
    """Send one request, consume its SSE stream, record TTFT / e2e / per-token gaps into `stats`."""
    t0 = time.perf_counter()                                    # arrival (from the client's point of view)
    first, prev, itls, tokens, done = None, None, [], 0, None
    async with client.stream("POST", url, json={"prompt": prompt, "max_tokens": max_tokens,
                                                "temperature": 0.7}) as r:
        async for line in r.aiter_lines():                      # SSE frames arrive as lines; blank lines separate them
            if not line.startswith("data: "):
                continue                                        # skips ': ping' heartbeats and blank separators
            now = time.perf_counter()
            ev = json.loads(line[6:])                           # strip the 'data: ' prefix
            if ev.get("done"):
                done = ev
                break
            if first is None:
                first = now                                     # first text frame → TTFT
            else:
                itls.append(now - prev)                         # gap since the previous frame → one ITL sample
            prev = now
    end = time.perf_counter()
    tokens = done["tokens"] if done else 0                      # server-side count (includes the stop token)
    stats.append({"ttft": (first or end) - t0, "e2e": end - t0, "itl": itls, "tokens": tokens,
                  "preemptions": done.get("preemptions", 0) if done else 0})   # field appears in M4


async def main(a):
    random.seed(a.seed)                                         # same prompt/length/arrival sequence every run
    stats, tasks = [], []
    async with httpx.AsyncClient(timeout=None) as client:       # timeout=None: a queued request may legitimately wait long
        t_start = time.perf_counter()
        for i in range(a.n):
            max_tokens = random.randint(a.max_tokens // 4, a.max_tokens)   # varied lengths matter (see note below)
            tasks.append(asyncio.create_task(one(client, f"{a.host}/generate", random.choice(PROMPTS),
                                                 max_tokens, stats)))       # fire and continue — do not await here
            await asyncio.sleep(random.expovariate(a.rate))    # Poisson arrivals: exponential gap with mean 1/rate
        await asyncio.gather(*tasks)                            # wait for every stream to finish
        wall = time.perf_counter() - t_start
    itl = [x for s in stats for x in s["itl"]]                  # flatten all per-token gaps into one list
    out = {
        "label": a.label, "rate": a.rate, "n": a.n, "max_tokens": a.max_tokens,
        "throughput_tok_s": sum(s["tokens"] for s in stats) / wall,       # achieved output tokens per second
        "req_per_s": a.n / wall,                                          # achieved request rate (≈ rate left of the knee)
        "ttft_p50": pct([s["ttft"] for s in stats], 50), "ttft_p99": pct([s["ttft"] for s in stats], 99),
        "e2e_p50": pct([s["e2e"] for s in stats], 50), "e2e_p99": pct([s["e2e"] for s in stats], 99),
        "itl_p50": pct(itl, 50), "itl_p99": pct(itl, 99),
        "preemptions": sum(s["preemptions"] for s in stats),
    }
    os.makedirs("bench/results", exist_ok=True)
    path = f"bench/results/{a.label}_rate{a.rate}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1:8000")               # server base URL
    p.add_argument("--rate", type=float, default=2.0, help="mean arrivals per second")   # λ, the offered load
    p.add_argument("--n", type=int, default=60)                              # requests per run (p99 needs ≥ ~100 to mean much)
    p.add_argument("--max-tokens", type=int, default=128)                    # upper bound of the random output length
    p.add_argument("--label", default="continuous")                          # tag in the result filename / plot legend
    p.add_argument("--seed", type=int, default=0)                            # reproducible arrivals and lengths
    asyncio.run(main(p.parse_args()))
```

`bench/sweep.sh`:

```bash
#!/usr/bin/env bash
# bench/sweep.sh <label> — run the load generator at several arrival rates against a running server
label=${1:-continuous}                 # first argument, default "continuous"; becomes the results-file prefix
for rate in 0.5 1 2 4 8 16; do         # offered load λ in req/s, doubling each step so the knee is bracketed
  python bench/loadgen.py --rate $rate --n 80 --max-tokens 128 --label $label   # 80 requests per point
done
```

Output lengths are randomized in `[max/4, max]` on purpose: if every request generated exactly 128 tokens, static batching would look almost as good as continuous. Variance in length is what continuous batching exploits.

### Test

First correctness, then the benchmark. `test_batch_parity.py` — batched, padded, slot-stored decode must produce the *same tokens* as the plain P1 contract at batch 1. Run it in fp32: fp16 matmuls over different padded shapes accumulate in a different order and can flip an argmax late in a sequence, which is noise, not a bug, but it makes "identical" impossible to assert.

```python
"""Batched + padded + paged decode must produce exactly the tokens the unbatched contract produces."""
import torch
from engine import ModelRunner
from kv_manager import SlotKV
from scheduler import Request, Scheduler

runner = ModelRunner.load(dtype=torch.float32)                  # fp32 so "identical" is a fair assertion
kvargs = dict(num_layers=runner.num_layers, kv_heads=runner.kv_heads, head_dim=runner.head_dim,
              device=runner.device, dtype=runner.dtype)          # shared constructor args for every KV backend


def reference(prompt, max_tokens):
    """Plain contract, batch 1, no padding, no manager. Greedy. This is the ground truth."""
    ids, out = list(prompt), []
    pos = torch.arange(len(ids), device=runner.device).unsqueeze(0)                      # [1, T]
    logits, kv = runner.forward(torch.tensor([ids], device=runner.device), pos, None, None)   # prefill
    for _ in range(max_tokens):
        t = logits.argmax(-1).item()                                                      # greedy next token
        out.append(t)
        if t in runner.stop_ids or len(out) >= max_tokens:
            break
        logits, kv = runner.forward(torch.tensor([[t]], device=runner.device),           # [1, 1] new token
                                    torch.tensor([[kv[0][0].shape[2]]], device=runner.device), kv, None)   # position = cache length
    return out


def run_sched(kv, prompts, max_tokens, **kw):
    """Drive a Scheduler synchronously (no event loop, no HTTP) until every request finishes."""
    s = Scheduler(runner, kv, **kw)
    reqs = [Request(prompt_ids=p, max_tokens=max_tokens) for p in prompts]   # out=None: no consumer queue needed
    for r in reqs:
        s.add(r)
    while s.has_work():
        s.step()
    return [r.output_ids for r in reqs], s


texts = ["Hi", "Explain paged attention in one paragraph.", "List three uses of asyncio.",
         "Write a haiku about GPUs and then explain each line in detail.", "Why is decode memory-bound?",
         "Tell me a story", "1 2 3 4 5 6 7 8", "What is the capital of France? Answer in one word."]
prompts = [runner.tokens_for_prompt(t) for t in texts]           # deliberately varied lengths → varied padding
ref = [reference(p, 40) for p in prompts]

out, _ = run_sched(SlotKV(8, 512, **kvargs), prompts, 40)        # 8 slots of 512 tokens: all 8 prompts admitted at once
assert out == ref, "SlotKV parity FAILED"
print("SlotKV       parity OK")
out, _ = run_sched(SlotKV(8, 512, **kvargs), prompts, 40, static=True)
assert out == ref, "static parity FAILED"
print("static mode  parity OK")
```

Then the benchmark, once per mode:

```bash
MV_STATIC=1 uvicorn server:app --port 8000 &   # baseline: static batching (MV_STATIC=1); & = run in the background
bash bench/sweep.sh static                     # six load points, results tagged "static"
kill %1                                        # stop background job #1 (the server)
uvicorn server:app --port 8000 &               # continuous (the default)
bash bench/sweep.sh continuous
```

### Expected

The parity test prints two `OK`s. If fp16 is all you can run, expect prompts to agree for the first 20–30 tokens and treat a late single-token divergence as fp16 noise; an early divergence (token 1–5) is a real mask/position bug.

For the benchmark, reason from the step-time model rather than a promised number. At λ = 0.5 req/s the batch is mostly size 1 and both modes look alike: TTFT p50 = one prefill (~15–40 ms for a chat-templated 40–80-token prompt) plus at most one step of queueing; ITL p50 ≈ your batch-1 step (8–12 ms). As λ rises, continuous keeps TTFT p50 in the tens of milliseconds until the batch is full (16), after which arrivals queue and TTFT climbs steeply — that is the knee, typically somewhere between λ = 4 and 12 on this hardware for 128-token outputs (capacity ≈ decode rate / mean tokens per request: ~1,000 tok/s ÷ ~80 tokens ≈ 12 req/s minus prefill overhead). Static's TTFT p99 at moderate load is the duration of a whole batch — 128 tokens × 10–15 ms ≈ 1.5–2 s plus prefill — because a new arrival waits for the drain, and its throughput plateaus lower because slots idle as short requests finish. Expect continuous to deliver 1.5–3× the throughput of static at equal p99, or the same throughput at a fraction of the p99; the gap widens with the variance in output length. ITL p99 in continuous mode will show 30–150 ms spikes: those are prefill steps stalling the running decodes, and they are the argument for chunked prefill.

### Reading a failure

| Symptom | Where the bug is |
|---|---|
| Parity fails at token 1 for every prompt | Mask: pad columns visible, or the model adds its own causal mask on top of yours |
| Parity fails only for the shortest prompts in a batch | `position_ids` not shifted by `pad`; RoPE sees positions starting at `pad_i` |
| Parity fails after the first decode step | `write_prefill` copied from column 0 instead of `pad_i`; decode mask `valid` off by one (`< len_i` vs `<=`) |
| NaN logits in a padded batch | A fully masked row: pad query rows must be allowed to see something; mask added in fp16 |
| `RuntimeError: out of KV memory` in `_decode` | A request's `num_cached` exceeds `max_len` — the `MAX_MODEL_LEN` check in the handler is missing |
| Throughput same in static and continuous | Output lengths identical (no variance to exploit) or λ far below the knee |
| Server hangs, no output | `wake` never set (`add` bypassed), or `engine_loop` raised — it dies silently inside `create_task`; log it |
| ITL p99 in seconds | One giant prompt admitted with tiny ones: raise the length check, or sort waiting by length |

### Close the milestone

```
git commit -m "M2: iteration-level scheduler with left-padded batching; static-vs-continuous Poisson benchmark"
```

README: the step-loop diagram; the padding/mask picture (one row, pads on the left, arrows to what it can see); the static-vs-continuous graph (throughput and TTFT p99 vs λ) with your measured ratio; one sentence on why output-length variance is what continuous batching monetizes.

---

## Milestone 3 — Block-based KV manager (paged KV)

### What it is

`SlotKV` reserves `max_len` tokens per sequence at admission because it does not know how long the sequence will be. Most of that is never used: a 60-token prompt with 40 generated tokens holds 2048 slots. The vLLM paper measured 60–80% of KV memory wasted this way in contemporary systems, which is why "how many sequences fit" was the binding constraint on throughput.

**PagedAttention's** fix is the operating-system one (virtual memory: a process sees contiguous addresses, the OS backs them with scattered physical pages via a page table): cut KV storage into fixed-size **blocks** (16 tokens each here), keep a **free list** (the ids of blocks nobody owns), and give each sequence a **block table** — a list of physical block ids, logical position `p` living at `table[p // 16]`, offset `p % 16`. Allocation happens when it is needed: at admission, enough blocks for the prompt; then one more block every 16 decode steps when the last block fills. Waste is bounded by one partial block per sequence (< 16 tokens; vLLM quotes < 4%). Freeing is `free_list.extend(table)`. Blocks need not be contiguous, so there is no *external fragmentation* (free memory that exists but not in one large enough piece) either — the reason the paper calls it paging.

**The simplification you are allowed, and its cost.** Attention wants K and V as `[B, kv_heads, L, head_dim]`. Real PagedAttention is a CUDA kernel that reads the block table and fetches each key/value from its block *inside* the attention computation, so the blocks are never materialized contiguously. Writing that kernel — with the tiling, the shared-memory staging, the head-group handling of GQA — is out of scope; FlashAttention/FlashInfer now ship it. What you do instead is **gather**: `self.k[l][table]` pulls each sequence's blocks into a contiguous `[B, L, ...]` tensor per layer, then attention runs as before. That is a copy of the entire active KV every step: 16 sequences × 600 tokens × 12 KiB ≈ 115 MB read + written per step, roughly 0.7 ms on 336 GB/s, plus the launch of 48 index kernels. At long contexts it becomes the dominant per-step cost after the weight read, which is precisely the cost the real kernel avoids. Say this in the README and you have answered the interview question before it is asked.

**Fragmentation accounting.** Both managers expose `tokens_used / tokens_reserved`. Sample it during a load run for each and you have the paper's headline number from your own system.

### Look first

**1. vLLM's block pool and free list.** `vllm/v1/core/block_pool.py`, class `BlockPool`. Read `__init__` (one `KVCacheBlock` object per physical block: `self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]`, all of them initially in `self.free_block_queue`), `get_new_blocks(num_blocks)` (pop from the free queue, bump `ref_cnt`), and `free_blocks(ordered_blocks)` (decrement `ref_cnt`; append to the free queue only when it hits 0). Then `vllm/v1/core/kv_cache_utils.py`: the `KVCacheBlock` dataclass (`block_id`, `ref_cnt`, `_block_hash`, `prev_free_block`, `next_free_block`) and `FreeKVCacheBlockQueue` — vLLM's free list is a *doubly linked list* so that a block can be removed from the middle in O(1) when a prefix-cache hit revives it (M5). Yours is a Python list; note what that would cost in M5.

**2. Allocation arithmetic.** `vllm/v1/core/kv_cache_manager.py`, method `KVCacheManager.allocate_slots(request, num_new_tokens, ...)`. Find the ceil-division that turns `num_computed_tokens + num_new_tokens` into a block count and subtracts the blocks the request already holds; the `return None` when that exceeds `self.block_pool.get_num_free_blocks()`; and `free(request)`, which frees a request's blocks in *reverse* order (a comment there explains why: the tail blocks, least likely to be reused as a prefix, should be evicted first). In newer checkouts the per-request `req_to_blocks` dict and the arithmetic live in `vllm/v1/core/single_type_kv_cache_manager.py` (`SingleTypeKVCacheManager.allocate_new_blocks`) behind a `KVCacheCoordinator`; the logic is the same.

**3. What the block table looks like on the GPU and where the kernel consumes it.** `vllm/v1/worker/block_table.py`, class `BlockTable`: a preallocated int32 tensor `[max_num_reqs, max_num_blocks_per_req]`, one row per running request, filled by `append_row` — the tensor form of your `req.block_table` lists. Then `vllm/v1/attention/backends/flash_attn.py`: the `block_table` field of `FlashAttentionMetadata` and the `flash_attn_varlen_func(..., block_table=...)` call — the attention kernel takes the table and reads K/V straight out of the blocks. No gather anywhere. The original hand-written kernel is `csrc/attention/paged_attention_v1.cu`; skim its inner loop to see it index `k_cache[physical_block_number * ...]`.

**4. Measure what advanced indexing costs before you depend on it.**

```python
import torch, time
k = torch.zeros(24, 5461, 2, 16, 64, device="cuda", dtype=torch.float16)   # [layers, blocks, kv_heads, block, hd] = 1 GiB
table = torch.randint(0, 5461, (16, 38), device="cuda")                    # 16 sequences × 38 blocks ≈ 600 tokens each
torch.cuda.synchronize(); t0 = time.perf_counter()
for l in range(24):
    g = k[l][table]                     # [16, 38, 2, 16, 64] — a COPY of 16×608 tokens of K for this layer
torch.cuda.synchronize(); print(f"gather K, 24 layers: {(time.perf_counter()-t0)*1e3:.2f} ms")   # V doubles it
```

Expect ~0.3–0.5 ms for K alone at this size; double for V. That is the number the "Why this is not real PagedAttention" README paragraph needs, and you now know it before writing `BlockKV`.

### Code

Append to `kv_manager.py`:

```python
class BlockKV:
    """M3: paged. KV lives in fixed-size blocks; a sequence owns a block table.

    self.k / self.v: [num_layers, num_blocks, kv_heads, block_size, head_dim]
    Logical token position p of a request lives at block req.block_table[p // bs], offset p % bs:
        self.k[layer][req.block_table[p // bs], :, p % bs, :]      -> [kv_heads, head_dim]
    ≈ vLLM's BlockPool (free list) + per-request block table, with the attention gather done in Python."""

    def __init__(self, num_blocks, block_size, num_layers, kv_heads, head_dim, device, dtype):
        shape = (num_layers, num_blocks, kv_heads, block_size, head_dim)   # e.g. [24, 5461, 2, 16, 64] fp16 = 1 GiB
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.bs, self.num_blocks, self.num_layers = block_size, num_blocks, num_layers
        self.kv_heads, self.head_dim = kv_heads, head_dim
        self.free_blocks = list(range(num_blocks))                     # the free list: physical ids nobody owns
        self.owners = set()                                            # requests currently holding blocks (for stats)

    def blocks_for(self, n_tokens):
        """How many blocks n_tokens need: ceil(n / bs). 17 tokens → 2 blocks."""
        return math.ceil(n_tokens / self.bs)

    def allocate(self, req, n_tokens):
        """Grow req's block table so it covers n_tokens. Called with len(tokens) at admission and
        num_cached + 1 before every decode step — a no-op 15 steps out of 16, one pop on the 16th."""
        need = self.blocks_for(n_tokens) - len(req.block_table)         # blocks still missing (0 most of the time)
        if need > len(self.free_blocks):
            return False                                                # not enough free blocks: caller decides (M4: preempt)
        for _ in range(need):
            req.block_table.append(self._take_block())                  # append keeps logical order
        self.owners.add(req)
        return True

    def _take_block(self):
        """Pop one free block. Split out so M5 can add ref counts and LRU eviction here."""
        return self.free_blocks.pop()

    def free(self, req):
        """Give every block back. Clearing the table inside free() makes a second call harmless."""
        for b in req.block_table:
            self._release_block(b)
        req.block_table.clear()
        self.owners.discard(req)

    def _release_block(self, b):
        """Put one block on the free list. Split out so M5 can decrement a ref count instead."""
        self.free_blocks.append(b)

    def gather(self, reqs):
        """Copy every request's blocks into contiguous [B, kv_heads, L, head_dim] tensors, one per layer.
        THIS is the simplification: the real PagedAttention kernel reads blocks in place and never does this."""
        nb = max(len(r.block_table) for r in reqs)                      # longest table in the batch
        # Pad shorter tables with block 0 so the index tensor is rectangular: [B, nb]. Whatever block 0 holds is
        # masked out by build_decode_batch because cached_lens[i] < L for those rows.
        table = torch.tensor([r.block_table + [0] * (nb - len(r.block_table)) for r in reqs],
                             device=self.k.device)
        B, L = len(reqs), nb * self.bs                                  # L is a multiple of bs (padded past length)
        past = []
        for l in range(self.num_layers):
            # self.k[l][table]: [B, nb, kv_heads, bs, head_dim]  (advanced indexing = copy)
            # permute -> [B, kv_heads, nb, bs, head_dim]; reshape merges (nb, bs) -> L: [B, kv_heads, L, head_dim]
            k = self.k[l][table].permute(0, 2, 1, 3, 4).reshape(B, self.kv_heads, L, self.head_dim)
            v = self.v[l][table].permute(0, 2, 1, 3, 4).reshape(B, self.kv_heads, L, self.head_dim)
            past.append((k, v))
        return past, L

    def _write_span(self, l, table, start, k, v):
        """Write k, v [kv_heads, n, head_dim] at token positions start..start+n-1.
        start is always a block boundary in this design (prefill starts at 0 or after
        whole cached blocks), so this is one strided copy per layer per sequence."""
        assert start % self.bs == 0, start
        n = k.shape[1]                                                  # number of tokens to write
        nb = self.blocks_for(n)                                         # blocks they span
        pad = nb * self.bs - n                                          # unused tail of the last block
        if pad:
            # F.pad pads from the LAST dim backwards: (0, 0) leaves head_dim alone, (0, pad) extends the token dim.
            k, v = F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))   # now [kv_heads, nb*bs, head_dim]
        # The physical blocks for this span, as an index tensor [nb].
        idx = torch.tensor(table[start // self.bs: start // self.bs + nb], device=self.k.device)
        # Reshape tokens into (nb, bs) and move nb first so it lines up with idx: [nb, kv_heads, bs, head_dim]
        # matches self.k[l][idx]'s shape [nb, kv_heads, bs, head_dim]. One scatter per layer.
        self.k[l][idx] = k.view(self.kv_heads, nb, self.bs, self.head_dim).transpose(0, 1)
        self.v[l][idx] = v.view(self.kv_heads, nb, self.bs, self.head_dim).transpose(0, 1)

    def write_prefill(self, reqs, new_kv, pads, L_past):
        """For each row, write its real new tokens (skip the gathered past and its own left pads)."""
        for i, r in enumerate(reqs):
            for l, (k, v) in enumerate(new_kv):                         # k, v: [B, kv_heads, L_past + T, head_dim]
                self._write_span(l, r.block_table, r.num_cached,
                                 k[i, :, L_past + pads[i]:], v[i, :, L_past + pads[i]:])   # [kv_heads, n_i, head_dim]

    def write_decode(self, reqs, new_kv, L):
        """Store each row's single new token at (its current last block, offset within that block)."""
        blk = torch.tensor([r.block_table[r.num_cached // self.bs] for r in reqs], device=self.k.device)   # [B] block ids
        off = torch.tensor([r.num_cached % self.bs for r in reqs], device=self.k.device)                   # [B] offsets
        for l, (k, v) in enumerate(new_kv):
            # Paired advanced indices (dims 0 and 2): row i → self.k[l][blk[i], :, off[i]]. k[:, :, L] is [B, kv_heads, head_dim].
            self.k[l][blk, :, off] = k[:, :, L]
            self.v[l][blk, :, off] = v[:, :, L]

    def stats(self):
        used = sum(r.num_cached for r in self.owners)                   # tokens actually holding K/V
        allocated = (self.num_blocks - len(self.free_blocks)) * self.bs # tokens reserved = owned blocks × bs
        return {"tokens_used": used, "tokens_reserved": allocated,
                "blocks_free": len(self.free_blocks), "blocks_total": self.num_blocks,
                "utilization": used / allocated if allocated else 0.0}
```

`gather` pads short block tables with block 0; whatever lives there is masked by `build_decode_batch` because `cached_lens[i] < L`. `_take_block`/`_release_block` are one-liners now so that M5 can override them with ref counts. The scheduler already calls `allocate(r, r.num_cached + 1)` before every decode step — with `BlockKV` that call is a no-op 15 steps out of 16 and pops one block on the 16th.

`server.py`: replace `build_kv` so the backend is selectable:

```python
from kv_manager import BlockKV, SlotKV

BLOCK_SIZE = 16                                                 # tokens per block (vLLM's default too)

def build_kv(runner, kind, kv_mb):
    """kind: "slot" (M2 baseline) or "block" (M3). Same MiB budget either way, so comparisons are fair."""
    bytes_per_token = 2 * runner.num_layers * runner.kv_heads * runner.head_dim * runner.dtype.itemsize   # 12,288
    tokens = int(kv_mb * 2**20 // bytes_per_token)                    # tokens the budget can hold
    common = dict(num_layers=runner.num_layers, kv_heads=runner.kv_heads,
                  head_dim=runner.head_dim, device=runner.device, dtype=runner.dtype)
    if kind == "slot":
        return SlotKV(num_slots=max(1, tokens // MAX_MODEL_LEN), max_len=MAX_MODEL_LEN, **common)
    return BlockKV(num_blocks=tokens // BLOCK_SIZE, block_size=BLOCK_SIZE, **common)   # 1 GiB → 5,461 blocks
```

and in `lifespan`: `kv = build_kv(runner, os.environ.get("MV_KV", "block"), float(os.environ.get("MV_KV_MB", 1024)))` — `MV_KV` selects the backend, default `block`.

`bench/kv_waste.py`:

```python
"""bench/kv_waste.py — sample /stats while a load run is in progress and report KV waste.
Run:  python bench/kv_waste.py & python bench/loadgen.py --rate 4 --n 80

Argument 1 (optional): how many seconds to sample for (default 30)."""
import json, sys, time, urllib.request

samples, t_end = [], time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 30)
while time.time() < t_end:
    s = json.load(urllib.request.urlopen("http://127.0.0.1:8000/stats"))    # one GET per sample; stdlib, no httpx needed
    if s["tokens_reserved"]:                                                 # skip idle moments (nothing reserved)
        samples.append((s["tokens_used"], s["tokens_reserved"]))
    time.sleep(0.1)                                                          # 10 Hz sampling
# Time-weighted mean utilization: sum of used over sum of reserved across all samples.
used = sum(u for u, _ in samples); reserved = sum(r for _, r in samples)
print(f"samples={len(samples)}  mean utilization={used / reserved:.1%}  waste={(1 - used / reserved):.1%}")
```

### Test

Append to `test_batch_parity.py`:

```python
from kv_manager import BlockKV
out, _ = run_sched(BlockKV(256, 16, **kvargs), prompts, 40)     # 256 blocks × 16 = 4,096 tokens: plenty for 8 prompts
assert out == ref, "BlockKV parity FAILED"
print("BlockKV      parity OK")
```

Then the waste measurement, one run per backend, same load:

```bash
MV_KV=slot  MV_KV_MB=256 uvicorn server:app --port 8000 &    # 256 MiB = 21,845 tokens = 10 slots of 2048
# Sampler in the background for 40 s, load generator in the foreground; `wait` blocks until both are done.
python bench/kv_waste.py 40 & python bench/loadgen.py --rate 4 --n 100 --max-tokens 128 --label slot; wait
kill %1
MV_KV=block MV_KV_MB=256 uvicorn server:app --port 8000 &    # same budget as 1,365 blocks of 16
python bench/kv_waste.py 40 & python bench/loadgen.py --rate 4 --n 100 --max-tokens 128 --label block; wait
```

Also time the gather: wrap `self.kv.gather(batch)` in `_decode` with `torch.cuda.synchronize()` + `perf_counter` for one run at batch 16 and record the fraction of step time. (`torch.cuda.synchronize()` makes the CPU wait for every queued GPU kernel to finish — without it, `perf_counter` measures only how long it took to *launch* the kernels.)

### Expected

Parity `OK` — paged storage is invisible to the model, which is the whole point. Slot utilization with `MAX_MODEL_LEN=2048` and ~100-token sequences: 3–8% (i.e. 92–97% waste — worse than the paper's 60–80% because your `max_len` is generous relative to your prompts). Block utilization: 85–97% (waste is the partial last block: with 16-token blocks and ~100-token sequences, roughly half a block per sequence ≈ 5–8%). With the 256 MiB budget the slot server also admits at most 10 concurrent sequences while the block server admits all 16 — visible as higher throughput and lower TTFT p99 for `block` at λ = 4. Gather cost: 3–10% of the step at short context, rising with `B × L`; at 16 × 1500 tokens it approaches the weight-read time.

### Reading a failure

| Symptom | Where the bug is |
|---|---|
| Parity fails right after a sequence crosses a 16-token boundary | `write_decode` index: `block_table[num_cached // 16]` must exist — `allocate(r, num_cached + 1)` not called before decode |
| Parity fails only for prompts longer than 16 | `_write_span` reshaping: the `view(kv_heads, nb, bs, hd).transpose(0, 1)` order, or `start` not a block boundary |
| Wrong tokens only when the batch has different block-table lengths | `gather` padding rows with block 0 but `L` computed from the wrong `nb` |
| `IndexError` in `free` / double free | `free(req)` called twice (retire + admission failure path): `block_table.clear()` must run inside `free` |
| Utilization > 1.0 | `num_cached` incremented before `write_decode`, or `stats` counting a freed request in `owners` |
| CUDA OOM at startup | `MV_KV_MB` too large for what is left after weights and prefill scratch; check `torch.cuda.mem_get_info()` |

### Close the milestone

```
git commit -m "M3: block-based KV manager (16-token blocks, free list, block tables); gather-based attention; waste measurement"
```

README: block table diagram (one sequence, three blocks, positions → physical ids); utilization for slot vs block under the same load; a paragraph titled "Why this is not real PagedAttention" with the gather cost measured.

---

## Milestone 4 — Preemption (recompute on resume)

### What it is

Paging lets you admit more sequences than the worst case would allow — which means the worst case can now actually happen: every running sequence wants one more block and the free list is empty. Something has to give. vLLM's answer is **preemption**: evict a running sequence, free its blocks, put it back at the *head* of the waiting queue (it keeps its priority), and continue with the rest. When it is re-admitted, its KV is rebuilt.

Two ways to rebuild. **Swap** copies the evicted blocks to *pinned* CPU memory (page-locked RAM the GPU can DMA to and from directly) and back over PCIe (the bus between CPU and GPU). **Recompute** throws them away and re-runs prefill over `prompt + tokens generated so far`. Recompute is what vLLM v1 does by default, and the arithmetic says why: prefill is compute-bound and fast (a few hundred tokens in tens of milliseconds), whereas swapping 100 tokens × 12 KiB = 1.2 MB each way over PCIe 3.0 x16 (~12 GB/s, and the 2060 is on x16) is fast too but needs pinned buffers, a copy stream, and bookkeeping for partially swapped state; for short-to-medium sequences the recompute is cheaper than the engineering, and the recomputed prefill produces bit-identical K/V. Swap wins only for very long sequences that were far along. Your invariant makes recompute trivial: set `num_cached = 0` and the existing prefill path recomputes everything, including sampling the *next* token as if the last decode step had run.

**Whom to evict.** The victim should be the one whose loss costs least and who has the weakest claim: the most recently *arrived* request (it has generated the least, and FCFS says it should be last anyway). Evict by arrival time, not by position in `running` — a resumed request re-enters `running` at the end but is old, and picking `running[-1]` would evict it again and again under a tight pool (*livelock*: everyone is busy, nobody makes progress).

Expose a **preemption counter**: it is the single best signal that the KV budget is too small for the offered load, and it is what an autoscaler should key on long before latency shows it.

### Look first

**1. vLLM's preemption branch.** Back in `vllm/v1/core/sched/scheduler.py`, `Scheduler.schedule()`, the loop over `self.running`: find the `while True:` around `allocate_slots`. When it returns `None`:

- the victim is `self.running.pop()` under the default FCFS policy (or `max(self.running, key=lambda r: (r.priority, r.arrival_time))` under priority scheduling — the arrival-time rule you are about to implement);
- `self.kv_cache_manager.free(preempted_req)`, `preempted_req.status = RequestStatus.PREEMPTED`, `preempted_req.num_computed_tokens = 0` (that line *is* recompute-on-resume: nothing else is needed because the scheduler only ever trusts `num_computed_tokens`), `preempted_req.num_preemptions += 1`;
- `self.waiting.prepend_request(preempted_req)` — head of the queue, same as your `appendleft`;
- `if preempted_req == request: can_schedule = False; break` — "I evicted myself; stop".

Then look just below, at the waiting-queue loop: it is guarded by `if not preempted_reqs:` — vLLM does **not** admit anyone in a step in which it preempted. Combined with popping the tail of `running`, that is how it avoids the livelock this milestone warns about; your design admits every step, so you evict by arrival time instead. Be able to explain both.

**2. The counter it exports.** `vllm/v1/metrics/loggers.py`: `counter_num_preempted_reqs` → `vllm:num_preemptions_total`. That is the metric M6 will mirror as `mv_preemptions_total`, and the one to alert on.

**3. Recompute vs swap in the real code base.** `grep -rn "preemption_mode\|swap_space" ~/src/vllm/vllm/config/` — `--preemption-mode {recompute,swap}` and `--swap-space` are v0-era options; the v1 scheduler you just read has only the recompute path. Then check your own PCIe link for the swap arithmetic: `nvidia-smi -q | grep -A3 "PCIe Generation\|Link Width"` (Gen 3 × 16 lanes ≈ 12 GB/s usable).

### Code

In `scheduler.py`, add `preemptions: int = 0` to `Request`, `self.num_preemptions = 0` to `Scheduler.__init__`, replace the allocation loop in `_decode`:

```python
    def _decode(self):
        batch = list(self.running)
        for r in list(batch):                                   # room for one more token each (iterate a copy: batch shrinks)
            # Keep evicting until r gets its block OR r itself was evicted (then it is no longer in batch).
            while r in batch and not self.kv.allocate(r, r.num_cached + 1):
                self._preempt(batch)                            # M4: evict someone (maybe r)
        if not batch:
            return                                              # everyone was evicted: nothing to decode this step
        past, L = self.kv.gather(batch)
        ...                                                     # unchanged from here
```

and add:

```python
    def _preempt(self, batch):
        """Out of KV: evict the most recently arrived running sequence, put it back at the
        head of the waiting queue, and recompute its KV when it is re-admitted."""
        victim = max(self.running, key=lambda r: r.arrival)    # newest arrival — NOT running[-1] (livelock, see text)
        self.running.remove(victim)
        if victim in batch:                                     # it may already have been dropped from this step's batch
            batch.remove(victim)
        self.kv.free(victim)                                    # blocks back to the pool (via ref counts in M5)
        victim.num_cached = 0                                   # recompute-on-resume: prefill will redo prompt + output so far
        victim.preemptions += 1                                 # per-request count → the done frame
        self.num_preemptions += 1                               # global count → /stats and mv_preemptions_total
        self.waiting.appendleft(victim)                         # head of the queue: keeps its FCFS priority
        return victim
```

The `while r in batch` loop reads: keep evicting until either `r` gets its block or `r` itself was the victim. Because eviction frees at least one block and the victim is always someone still running, the loop terminates. In `server.py`, add `"preemptions": s.num_preemptions` to `/stats` and `"preemptions": req.preemptions` to the `done` frame (the load generator already reads it).

### Test

Append to `test_batch_parity.py`:

```python
out, s = run_sched(BlockKV(40, 16, **kvargs), prompts, 40, max_num_seqs=8)   # 40 blocks: cannot hold 8 × ~90 tokens
assert out == ref, "preemption parity FAILED"
assert s.num_preemptions > 0                                                  # the pool must actually have been contended
print(f"preemption   parity OK  (preemptions={s.num_preemptions})")
out, s = run_sched(BlockKV(10, 16, **kvargs), prompts, 40, max_num_seqs=8)   # barely one sequence at a time
assert out == ref
print(f"tight pool   parity OK  (preemptions={s.num_preemptions})")
```

Then on the server, starve it and load it:

```bash
MV_KV_MB=8 uvicorn server:app --port 8000 &        # 8 MiB = 682 tokens = 42 blocks — deliberately far too small
python bench/loadgen.py --rate 8 --n 60 --max-tokens 128 --label starved
curl -s localhost:8000/stats                       # -s: silent (no progress bar); read the "preemptions" field
```

### Expected

Both parity cases `OK` with a nonzero count — recompute is exact, so preempted outputs are token-for-token identical to unpreempted ones (in fp32; in fp16 the recomputed prefill can differ from the original decode path by rounding and flip a late token, which is worth a sentence in the README). Under the starved server: `preemptions` in the tens to hundreds for 60 requests, every request still finishing with `finish_reason: length`, TTFT p99 and e2e p99 several times the un-starved run (each preemption costs a re-prefill plus a wait), and throughput well below the 1 GiB run. Reset `MV_KV_MB` to 1024 and the counter should read 0 across an entire sweep — that is the sizing claim from Step 0, demonstrated.

### Reading a failure

| Symptom | Where the bug is |
|---|---|
| Infinite loop in `_decode` | `_preempt` evicting by `running[-1]`; or the victim's blocks not actually freed (`free` skipped) |
| A preempted request never finishes | Re-admitted at the tail (`append` instead of `appendleft`), or `num_cached` not reset so prefill computes nothing |
| Preempted output differs from reference | Resume prefill fed `prompt_ids` instead of `tokens` (generated tokens dropped), or `output_ids` truncated |
| Counter stays 0 with `MV_KV_MB=8` | `MAX_MODEL_LEN` check rejects everything, or `max_num_seqs` so low that the pool is never contended |
| Same request preempted 10+ times | Pool smaller than one request's worst case: `blocks_total < blocks_for(prompt + max_tokens)`; guard at the API layer |

### Close the milestone

```
git commit -m "M4: preemption by recompute — evict newest arrival when blocks run out; preemption counter"
```

README: the eviction policy and why arrival-time; recompute vs swap with the PCIe arithmetic; the starved-vs-normal table (preemptions, TTFT p99, throughput).

---

## Milestone 5 (stretch) — Prefix caching

### What it is

Every chat request starts with the same system prompt; every request in a batch job shares the same instructions. Their first K/V blocks are identical — *if* the tokens are identical from position 0, because RoPE bakes absolute position into K. So make full blocks **content-addressed** (found by a fingerprint of what they contain, not by who owns them): the key of block *i* is `hash(key of block i-1, tokens in block i)`. The chain means "these 16 tokens at positions 16i..16i+15 after exactly this prefix", which is the only thing that makes two blocks' K/V interchangeable. On admission, walk the prompt's full blocks, look each hash up, and attach every hit to the block table; prefill starts from the first miss. This is vLLM's automatic prefix caching (SGLang's RadixAttention is the same idea on a trie — a prefix tree — with token-level granularity).

Three mechanisms make it safe:

- **Reference counts.** A block may sit in several block tables at once; it is only really free when the last owner lets go.
- **Cached-but-free pool with LRU.** When refs drop to zero, do not return the block to the free list — keep it, hash still registered, in an `OrderedDict`. The allocator takes from the plain free list first and evicts the least recently used cached block only when that is empty. Reuse costs nothing when memory is plentiful and degrades gracefully when it is not.
- **At least one token is always computed.** If the whole prompt is cached, prefill would have nothing to run and no logits to sample from; drop the last cached block from the match so one block is recomputed (vLLM does the same).

A **hit rate** metric (`hits / queries` over prompt blocks) tells you whether your workload has reuse worth the bookkeeping. Note that Qwen's chat template injects a default system prompt when you give none, so even unrelated requests share their first block.

### Look first

**1. The hash chain.** `vllm/v1/core/kv_cache_utils.py`: `hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None)` — it hashes the tuple `(parent_block_hash, tuple(curr_block_token_ids), extra_keys)`; the parent hash is the chain, and `extra_keys` is where LoRA ids and multimodal hashes go so two requests with the same tokens but different adapters do not alias. `hash_request_tokens(hash_function, block_size, request)` walks a request's full blocks calling it — your `block_hashes`. Then `NONE_HASH` / `init_none_hash`: the root of the chain is seeded from `PYTHONHASHSEED` if set, otherwise random per process — read the comment there about why, and note `CacheConfig.prefix_caching_hash_algo` (`--prefix-caching-hash-algo`, `sha256` variants) exists because Python's salted `hash()` cannot be shared across processes and a predictable hash invites collision attacks.

**2. Lookup, revival, registration, eviction — all in `BlockPool`.** `vllm/v1/core/block_pool.py`:

- `get_cached_block(block_hash)` — the dict lookup (`cached_block_hash_to_block`), your `block_of.get(h)`;
- `touch(blocks)` — `ref_cnt += 1`, and if it *was* 0, `self.free_block_queue.remove(block)`: a zero-ref cached block lives *in the free queue*, and a hit pulls it back out. This is why the free queue is a doubly linked list, and it is the one design difference from yours: vLLM has no separate `lru` dict. Free blocks — hashed or not — sit in one queue ordered by when they were freed; allocation pops from the head (oldest), and
- `_maybe_evict_cached_block(block)` (called from `get_new_blocks`) drops the popped block's hash mapping at that moment. Eviction is lazy and happens at allocation time. Your `_take_block` does "free list first, then LRU"; vLLM's is "oldest free block, whatever it is". Think about when the two differ (hint: when the free list still has never-used blocks);
- `cache_full_blocks(request, blocks, block_hashes, num_cached_blocks, num_full_blocks, block_size, ...)` — assigns hashes to newly full blocks: your `register`;
- `free_blocks(ordered_blocks)` — `ref_cnt -= 1`; at 0 the block goes to the free queue *keeping its hash*: your `_release_block`.

**3. Where the match happens and the one-token rule.** `vllm/v1/core/kv_cache_manager.py`, `KVCacheManager.get_computed_blocks(request)`: it fetches/caches `request.block_hashes`, and passes `max_cache_hit_length = request.num_tokens - 1` into the longest-prefix search (older checkouts instead drop the last block explicitly when the whole prompt is cached; grep the comment mentioning "recompute"). Either way: at least one token is always computed. Then back in `Scheduler.schedule()`, the waiting loop calls `get_computed_blocks` before `allocate_slots(request, num_new_tokens, num_new_local_computed_tokens, new_computed_blocks)`, and `allocate_slots` calls `touch` on the hits and `cache_blocks` on the blocks that will be full after this step — *before* the forward pass runs. Compare with your `register`, which runs *after* prefill. vLLM can register early because its attention backend writes every request's new K/V into the paged cache (`reshape_and_cache_flash` in `vllm/v1/attention/backends/flash_attn.py`) before it runs the attention kernel over the batch, so a same-step hit reads K/V that already exists by the time it is needed; your gather-before-forward design cannot, hence the ordering below.

**4. Watch the hit rate on a real server.** If you have vLLM running (`vllm serve ... --enable-prefix-caching` is the default), send the same long prompt twice and read the log line `Prefix cache hit rate: ...%` or `curl -s localhost:8000/metrics | grep prefix_cache` (`vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total`). `PrefixCacheStats` in `vllm/v1/core/kv_cache_utils.py` is the struct behind it; yours is two integers.

### Code

Append to `kv_manager.py`:

```python
class PrefixCachingKV(BlockKV):
    """M5: full blocks are content-addressed. A block's key is the chained hash of every
    token from position 0 up to the end of that block, so the same 16 tokens at a
    different offset (different RoPE positions) hash differently. Blocks have ref counts;
    a block with zero refs stays cached (LRU) until the free list runs dry.

    Block states:  free (in free_blocks, no hash)
                   owned (ref > 0; hash set once its K/V are registered)
                   cached-free (ref == 0, hash set, parked in self.lru; evictable)"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ref = [0] * self.num_blocks                 # block -> number of block tables holding it (vLLM: KVCacheBlock.ref_cnt)
        self.hash_of = [None] * self.num_blocks      # block -> hash (None = not cached)      (vLLM: KVCacheBlock.block_hash)
        self.block_of = {}                           # hash -> block                          (vLLM: cached_block_hash_to_block)
        self.lru = OrderedDict()                     # hash -> block, zero-ref cached blocks; oldest first
        self.hits = self.queries = 0                 # for the hit-rate metric

    @staticmethod
    def block_hashes(tokens, bs):
        """Chained hashes of every FULL block. A partial last block is never hashed (it can still change)."""
        h, out = 0, []                                                  # h = hash of the prefix so far (0 = empty prefix)
        for i in range(len(tokens) // bs):                              # only complete blocks
            h = hash((h, tuple(tokens[i * bs:(i + 1) * bs])))           # chain: previous hash + this block's tokens
            out.append(h)
        return out

    def match_prefix(self, req):
        """Attach cached blocks for the longest cached prefix of req.tokens.
        Leaves at least one token uncached so prefill always produces logits."""
        hashes = self.block_hashes(req.tokens, self.bs)
        if len(hashes) * self.bs == len(req.tokens):                    # prompt is an exact multiple of bs and fully hashed...
            hashes = hashes[:-1]                                        # ...drop the last block so one block is recomputed
        for h in hashes:                                                # walk from position 0; stop at the first miss
            b = self.block_of.get(h)
            if b is None:
                break
            if self.ref[b] == 0:
                self.lru.pop(h)                                         # revive: it was parked as evictable (vLLM: touch)
            self.ref[b] += 1
            req.block_table.append(b)                                   # share the physical block
        req.num_cached = len(req.block_table) * self.bs                 # prefill will start here
        req.block_hashes, req.prefix_hits = hashes, len(req.block_table) # remembered for register()
        self.owners.add(req)

    def _take_block(self):
        """Prefer a never-cached free block; only when none is left, evict the least recently used cached one."""
        if self.free_blocks:
            b = self.free_blocks.pop()
        else:
            h, b = self.lru.popitem(last=False)       # evict least recently used (front of the OrderedDict)
            self.block_of.pop(h)                                        # its content is about to be overwritten: forget the hash
            self.hash_of[b] = None
        self.ref[b] += 1
        return b

    def allocate(self, req, n_tokens):
        """Same as BlockKV.allocate, but the real capacity is free + evictable-cached."""
        need = self.blocks_for(n_tokens) - len(req.block_table)
        if need > len(self.free_blocks) + len(self.lru):
            return False
        for _ in range(need):
            req.block_table.append(self._take_block())
        self.owners.add(req)
        return True

    def register(self, req):
        """After prefill: make req's now-full prompt blocks findable by hash."""
        self.queries += len(req.block_hashes)                           # every full block was a lookup
        self.hits += req.prefix_hits                                    # the ones that were found
        for i, h in enumerate(req.block_hashes):                        # hashes[i] describes block_table[i]
            b = req.block_table[i]
            # Only register a block that is not yet hashed, and only if no other block already owns this hash
            # (two requests with the same new prompt in one batch both computed it; the first one wins).
            if self.hash_of[b] is None and h not in self.block_of:
                self.hash_of[b] = h
                self.block_of[h] = b

    def _release_block(self, b):
        """Drop one reference. At zero: free list if unhashed, LRU-parked (still findable) if hashed."""
        self.ref[b] -= 1
        if self.ref[b] > 0:
            return                                                      # someone else still uses it
        h = self.hash_of[b]
        if h is None:
            self.free_blocks.append(b)                                  # never registered: plain free
        else:
            self.lru[h] = b                            # cached, evictable
            self.lru.move_to_end(h)                                     # most recently released = last to be evicted

    def stats(self):
        used = sum(r.num_cached for r in self.owners)
        # Reserved = blocks with an owner. Cached-free blocks are not "reserved": they are reusable capacity.
        allocated = (self.num_blocks - len(self.free_blocks) - len(self.lru)) * self.bs
        return {"tokens_used": used, "tokens_reserved": allocated,
                "blocks_free": len(self.free_blocks) + len(self.lru), "blocks_total": self.num_blocks,
                "utilization": used / allocated if allocated else 0.0,
                "cached_blocks": len(self.lru),
                "hit_rate": self.hits / self.queries if self.queries else 0.0}
```

Two hooks in `scheduler.py` — in `_admit`, right after the `cancelled` check:

```python
            if hasattr(self.kv, "match_prefix"):                # only the M5 backend has it; SlotKV/BlockKV are untouched
                self.kv.match_prefix(req)                       # M5: reuse cached blocks; sets req.num_cached
```

and in `_prefill`, after `r.num_cached = len(r.tokens)`:

```python
        if hasattr(self.kv, "register"):
            for r in batch:
                self.kv.register(r)                             # M5: K/V now exist → make the blocks findable
```

Note that `_admit`'s failure path (`self.kv.free(req); req.num_cached = 0`) already releases a partial match correctly through the ref counts. `register` happens after prefill so a block is never findable before its K/V exists; two requests with the same new prompt in one batch both compute it and only the first registers. Python's `hash()` is salted per process, which is fine here (the cache is per process); a real system uses a stable hash so blocks can be shared across workers — which is exactly what your Project 3 router keys on.

`server.py`: `from kv_manager import BlockKV, PrefixCachingKV, SlotKV`, and in `build_kv`: `cls = PrefixCachingKV if kind == "prefix" else BlockKV; return cls(...)` — so `MV_KV=prefix` selects it.

`bench/prefix_test.py`:

```python
"""bench/prefix_test.py — 20 shared system prompts, cold pass then warm pass.
Compare TTFT and the server's prefix-cache hit rate between passes."""
import asyncio, json, random, statistics, time
import httpx

HOST = "http://127.0.0.1:8000"
random.seed(0)                                                             # same 20 "system prompts" every run
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi".split()
SYSTEMS = [" ".join(random.choices(WORDS, k=700)) for _ in range(20)]     # ~700 tokens each (one token per word here)
QUESTIONS = ["Summarize the above in one line.", "What is the third word?", "Count the words."]


async def ask(client, system, q):
    """Send system + question; return TTFT = time until the first SSE data frame."""
    t0 = time.perf_counter()
    async with client.stream("POST", f"{HOST}/generate",
                             json={"prompt": f"{system}\n\n{q}", "max_tokens": 8}) as r:   # 8 tokens: we only want TTFT
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                return time.perf_counter() - t0                            # TTFT


async def run_pass(name):
    """One request per system prompt, sequentially, then read the server's cumulative cache stats."""
    async with httpx.AsyncClient(timeout=None) as c:
        ttfts = [await ask(c, s, random.choice(QUESTIONS)) for s in SYSTEMS]   # sequential on purpose: no queueing in TTFT
        stats = (await c.get(f"{HOST}/stats")).json()
    print(f"{name:5s} TTFT p50 {statistics.median(ttfts)*1000:7.1f} ms   "
          f"hit_rate {stats.get('hit_rate', 0):.2f}   cached_blocks {stats.get('cached_blocks', 0)}")


async def main():
    await run_pass("cold")           # every system prompt is new: all misses
    await run_pass("warm")           # same prompts again: the ~44 full blocks of each are hits
    await run_pass("warm2")          # again: shows the cumulative rate converging

asyncio.run(main())
```

### Test

Append to `test_batch_parity.py` (a second wave must hit the cache and still match the reference; then a mixed batch where one row has a cached prefix and another none):

```python
from kv_manager import PrefixCachingKV
shared = runner.tokens_for_prompt("You are a terse assistant. " * 12)     # > 2 full blocks of shared prefix
wave1 = [shared + p[:5] for p in prompts[:2]]                             # shared prefix + a few distinct tokens
wave2 = [shared + p[:9] for p in prompts[2:4]]                            # same prefix, different tails → hits
kv = PrefixCachingKV(256, 16, **kvargs)                                   # ONE manager across both waves (the cache persists)
out, _ = run_sched(kv, wave1, 20)
assert out == [reference(p, 20) for p in wave1]
out, _ = run_sched(kv, wave2, 20)
assert out == [reference(p, 20) for p in wave2], "prefix cache parity FAILED"
assert kv.hits > 0 and all(x == 0 for x in kv.ref), "no hits, or ref-count leak"   # everything released after retire
print(f"prefix cache parity OK  (hit_rate={kv.stats()['hit_rate']:.2f})")
# Mixed batch: row 0 = 2 cached blocks + junk, row 1 = no cached prefix at all, row 2 = full shared prefix + tail.
mixed = [shared[:32] + [9, 9, 9], prompts[3], shared + prompts[0][:7]]
out, _ = run_sched(kv, mixed, 20)
assert out == [reference(p, 20) for p in mixed], "mixed-past prefill parity FAILED"
print("mixed past   parity OK")
```

Then:

```bash
MV_KV=prefix uvicorn server:app --port 8000 &      # MV_KV=prefix selects PrefixCachingKV
python bench/prefix_test.py
```

### Expected

Parity `OK` with a cumulative hit rate around 0.3–0.5 after two waves (the first wave is all misses by construction). `prefix_test.py`: cold TTFT p50 is one 700-token prefill — on the 2060 roughly 60–150 ms (700 tokens × ~1 GFLOP/token ≈ 0.7 TFLOP at a realistic 10–20 TFLOPS fp16, plus the KV write) — and warm TTFT p50 drops to one ~16–32-token prefill plus the block-table gather, roughly 15–30 ms; hit rate climbs to ~0.5 cumulative after `warm` and ~0.67 after `warm2` (the cold pass's misses stay in the denominator). `cached_blocks` after each pass ≈ 20 × 44 blocks of released, still-cached prefix. Shrink `MV_KV_MB` until 20 × 700 tokens no longer fit and watch LRU eviction turn the last few prompts of each pass back into misses.

### Reading a failure

| Symptom | Where the bug is |
|---|---|
| Warm pass no faster, hit rate 0 | `register` never called; or `hash` computed over `prompt_ids` in one place and `tokens` in another |
| Wrong tokens on a hit | The chain: hashing block content without the previous hash, so the same 16 tokens at a different offset alias |
| Wrong tokens only when a hit row shares a batch with a miss row | `build_prefill_batch`: `past_lens`/`L_past` per row — a row with 0 cached must have its past columns masked |
| `KeyError` in `lru.pop` | Ref count went negative or a block was released twice; assert `ref[b] >= 0` in `_release_block` |
| Free-list exhaustion despite idle server | Blocks parked in `lru` are not counted as free by `allocate`; `len(free) + len(lru)` is the real capacity |
| Hit on a fully cached prompt crashes prefill with empty input | The "leave one block uncached" trim missing |

### Close the milestone

```
git commit -m "M5: hash-chained prefix caching with ref counts and LRU eviction of unreferenced blocks; hit-rate metric"
```

README: the chained-hash diagram; ref-count state machine (allocated → cached-free → evicted); cold/warm TTFT table; a note that Project 3's router is this idea across machines.

---

## Milestone 6 — `/metrics`, load test, and the knee

### What it is

A serving engine that cannot tell you its TTFT distribution is not done. Prometheus text format is the lingua franca: `prometheus_client` keeps counters, gauges and histograms in process, and `generate_latest()` renders them for a *scrape* (Prometheus GETs `/metrics` every 15 s or so and stores what it reads; the server never pushes). A Prometheus **histogram** is a set of cumulative buckets: `mv_ttft_seconds_bucket{le="0.05"} 41` means 41 observations were ≤ 50 ms; plus `_sum` and `_count`. Percentiles are estimated from the buckets by `histogram_quantile`, so bucket edges must be chosen around the values you care about. The instruments that matter, and what each one is for:

- **TTFT histogram** — the user-perceived "did it start" latency: queue wait + prefill. Its p99 is the first thing to move when the system saturates.
- **ITL histogram** — the per-token cadence; p50 is your step time, p99 shows prefill stalls.
- **Queue depth gauge** — the autoscaling signal (never CPU%). Nonzero for more than a few steps means the engine is behind.
- **KV utilization gauge** and **preemption counter** — whether the memory budget matches the load.
- **Running sequences** and **generated tokens counter** — throughput is `rate(mv_generated_tokens_total[1m])`.

The **throughput-vs-p99 curve** is the graph every serving blog ends with: sweep λ, plot achieved token throughput on x and p99 latency on y. The curve is flat-then-vertical. The **knee** is where it turns: to the left, throughput grows with offered load and p99 is roughly constant (the system absorbs arrivals); to the right, throughput plateaus at engine capacity and p99 grows without bound because the queue grows without bound. Operationally you pick a point left of the knee with margin — that is your capacity per GPU, and it is what "how many replicas do we need for 100 req/s" is computed from. Identify it mechanically: the largest λ at which `req_per_s` still tracks λ (within ~10%) *and* TTFT p99 is within ~2× its unloaded value. Preemption count and queue depth at that λ tell you *which* resource bent the curve.

### Look first

**1. vLLM's instruments, by name.** `vllm/v1/metrics/loggers.py`, class `PrometheusStatLogger`, `__init__`. Read the bucket lists: `vllm:time_to_first_token_seconds` (buckets from 1 ms to thousands of seconds — TTFT includes queue wait, so the top must be huge) and `vllm:time_per_output_token_seconds` / `vllm:inter_token_latency_seconds` (buckets from 10 ms — decode steps are tight). Then the gauges `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc` (older: `gpu_cache_usage_perc`) and counters `vllm:num_preemptions_total`, `vllm:generation_tokens_total`, `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total`. Your eight instruments are these with an `mv_` prefix. Note which are labelled by `model_name` and why (one Prometheus can scrape many models).

**2. Where TTFT and ITL are actually measured.** `vllm/v1/metrics/stats.py`, `IterationStats.update_from_output`: the first output token stops the TTFT clock against `req_stats.arrival_time`; every later one appends `now - req_stats.last_token_ts` to the ITL list. Notice `arrival_time` is stamped in the *API-server* process when the request is created (grep `arrival_time` in `vllm/v1/engine/async_llm.py` / `processor.py`), so TTFT includes the queue — exactly what your `Request.arrival` does. `LoggingStatLogger.log()` in `loggers.py` is the periodic `Avg prompt throughput ... Running: N reqs, Waiting: M reqs, GPU KV cache usage: P%` log line; it reads the same stats.

**3. The library you are about to call.**

```python
import inspect, prometheus_client as pc
print(pc.Histogram.DEFAULT_BUCKETS)                    # (.005, .01, ..., 10) — sized for HTTP handlers, not TTFT; you override
print(inspect.getsource(pc.Histogram.observe))         # increments every bucket with le >= value, plus _sum and _count
h = pc.Histogram("demo_seconds", "demo", buckets=(.01, .1, 1)); h.observe(0.05)
print(pc.generate_latest().decode())                   # the exact text a scrape returns: _bucket{le=...}, _sum, _count
```

**4. What a scrape of a real engine looks like.** If a vLLM server is running: `curl -s localhost:8000/metrics | grep -E "^vllm:(time_to_first_token|num_requests|kv_cache|num_preemptions)"`. And `vllm bench serve --model ... --request-rate 4 --num-prompts 100` prints the client-side table (`Median TTFT`, `P99 TTFT`, `Median ITL`, `Output token throughput`) that your `loadgen.py` JSON mirrors field for field.

**5. The profiler, for README graph (6).** A *profiler* records what ran when (CPU functions, GPU kernels) so you can see where a step's time goes; a *flame graph* is one way to draw it. Wrap one `sched.step()` in `torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])` and print `prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)`; `prof.export_chrome_trace("step.json")` opens in `chrome://tracing` or Perfetto as a timeline. Look for the gap between CPU and CUDA totals — that is the Python/launch overhead fraction quoted below.

### Code

`metrics.py`:

```python
"""metrics.py — Prometheus instruments. Imported by scheduler.py and served by server.py.

Module-level singletons: prometheus_client registers each one in a global registry at import time,
and generate_latest() renders all of them. Names mirror vLLM's with an mv_ prefix."""
from prometheus_client import Counter, Gauge, Histogram

# Histogram buckets are upper edges in seconds. Choose them around the values you expect: TTFT 10 ms – 10 s
# (queue wait can be long), ITL 5 ms – 1 s (step time; the top buckets catch prefill stalls).
ttft = Histogram("mv_ttft_seconds", "Time to first token",
                 buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10))
itl = Histogram("mv_itl_seconds", "Inter-token latency",
                buckets=(.005, .01, .02, .03, .05, .075, .1, .2, .5, 1))
queue_depth = Gauge("mv_queue_depth", "Requests waiting for admission")            # the autoscaling signal
running = Gauge("mv_running_seqs", "Sequences in the current batch")
kv_utilization = Gauge("mv_kv_utilization", "Used KV tokens / allocated KV tokens")
preemptions = Counter("mv_preemptions_total", "Sequences evicted for lack of KV blocks")   # Counter: only goes up
tokens_out = Counter("mv_generated_tokens_total", "Generated tokens")             # throughput = rate() of this
prefix_hit_rate = Gauge("mv_prefix_cache_hit_rate", "Cumulative block hit rate")
```

`scheduler.py` — `import metrics`, then at the end of `step()`:

```python
        metrics.queue_depth.set(len(self.waiting))                     # Gauge.set: overwrite with the current value
        metrics.kv_utilization.set(self.kv.stats()["utilization"])
```

in `_append`, where the timestamps are set:

```python
            if r.first_token_time is None:
                r.first_token_time = now
                metrics.ttft.observe(now - r.arrival)                  # first token: TTFT from ARRIVAL (includes queue wait)
            else:
                metrics.itl.observe(now - r.last_token_time)           # later tokens: gap since the previous one
            r.last_token_time = now
```

and `metrics.preemptions.inc()` inside `_preempt`. TTFT is measured from `arrival` — set when the `Request` was constructed in the handler, so it includes queue wait, as it must.

`server.py` — `import metrics`, `from fastapi.responses import PlainTextResponse, StreamingResponse`, `from prometheus_client import CONTENT_TYPE_LATEST, generate_latest`; in `engine_loop` after `sched.step()`: `metrics.running.set(len(sched.running))`; in `stream()` after `n += 1`: `metrics.tokens_out.inc()`; and:

```python
@app.get("/metrics")
async def prom_metrics():
    """The scrape endpoint. Prometheus GETs this on its own schedule; we just render current values."""
    kvs = app.state.kv.stats()
    if "hit_rate" in kvs:                                              # only PrefixCachingKV reports it
        metrics.prefix_hit_rate.set(kvs["hit_rate"])
    # generate_latest() renders every registered instrument in the text exposition format;
    # CONTENT_TYPE_LATEST is the matching Content-Type header (text/plain; version=0.0.4).
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

`bench/plot.py`:

```python
"""bench/plot.py — throughput-vs-p99 curve from bench/results/*.json

One line per label (static, continuous, block, ...), one point per arrival rate."""
import glob, json
from collections import defaultdict
import matplotlib.pyplot as plt

runs = defaultdict(list)                                        # label -> list of result dicts
for f in glob.glob("bench/results/*.json"):
    d = json.load(open(f))
    runs[d["label"]].append(d)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))                   # left: throughput vs p99; right: TTFT p99 vs λ
for label, rs in runs.items():
    rs.sort(key=lambda d: d["rate"])                            # so the line is drawn in increasing-λ order
    # Left panel: x = achieved tokens/s, y = p99 end-to-end latency. Flat then vertical; the corner is the knee.
    ax[0].plot([d["throughput_tok_s"] for d in rs], [d["e2e_p99"] for d in rs], "o-", label=label)
    for d in rs:
        ax[0].annotate(f'λ={d["rate"]}', (d["throughput_tok_s"], d["e2e_p99"]), fontsize=7)   # label each point with its λ
    # Right panel: TTFT p99 against offered load — the first metric to climb when the queue starts growing.
    ax[1].plot([d["rate"] for d in rs], [d["ttft_p99"] for d in rs], "o-", label=label)
ax[0].set(xlabel="throughput (tok/s)", ylabel="p99 end-to-end latency (s)", title="throughput vs p99")
ax[1].set(xlabel="arrival rate λ (req/s)", ylabel="p99 TTFT (s)", title="TTFT vs offered load")
for a in ax:
    a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.savefig("bench/throughput_vs_p99.png", dpi=130)
print("wrote bench/throughput_vs_p99.png")
```

### Test

```bash
MV_KV=prefix uvicorn server:app --port 8000 &
curl -s localhost:8000/metrics | grep -E "^mv_"          # all instruments present, zeros  (-E: extended regex; ^ = line start)
bash bench/sweep.sh continuous
# Just the totals: histogram _count/_sum lines, the preemption counter, and the queue gauge.
curl -s localhost:8000/metrics | grep -E "^mv_(ttft|itl)_seconds_(count|sum)|^mv_preemptions_total|^mv_queue_depth"
python bench/plot.py
```

If you still have the `static` results from M2, `plot.py` draws both curves on the same axes. Also confirm `/metrics` costs nothing: scrape it in a loop at 1 Hz during the sweep and check p99 does not move.

### Expected

`mv_ttft_seconds_count` equals the number of requests served; `mv_itl_seconds_count` equals generated tokens minus requests. Histogram p50s read from the buckets agree with the load generator's client-side p50s to within a few milliseconds (client adds SSE serialization and loopback). The curve: throughput rising roughly linearly with λ from ~40 tok/s at λ = 0.5 (80 tokens × 0.5) to a plateau in the high hundreds to low thousands of tok/s, with p99 flat at a few hundred milliseconds until the knee, then p99 climbing to seconds as `req_per_s` falls below λ. The knee's λ is your capacity; expect it between 4 and 12 req/s for 128-token outputs, higher for shorter outputs. Where the knee lands depends on step time (Python overhead is a real fraction at this model size — profile one step with `torch.profiler` for the README and you will likely find 30–50% of wall time is not GPU kernels) and on how much prefill each admitted request costs.

### Reading a failure

| Symptom | Where the bug is |
|---|---|
| `mv_ttft_seconds_count` < requests served | `observe` in a branch that resumed requests skip; check the `first_token_time is None` guard |
| ITL count double counts | Observing ITL for the first token too (must be the `else` branch) |
| Client p99 ≫ server histogram p99 | The load generator is starved: run it on the same machine but not in the same process as anything heavy; or `n` too small for a p99 |
| Throughput never plateaus in the sweep | You did not push λ past capacity — extend the rates |
| p99 climbs before throughput flattens | Preemptions (check the counter) or prefill stalls; raise `MV_KV_MB` or lower `max_prefill_tokens` |
| Metrics reset to zero mid-run | Server restarted (OOM); check the uvicorn log |

### Close the milestone

```
git commit -m "M6: Prometheus /metrics (TTFT/ITL histograms, queue depth, KV utilization, preemptions); throughput-vs-p99 sweep and knee"
```

README graphs, in order: (1) throughput vs p99 with static and continuous curves and the knee marked; (2) TTFT p99 vs λ; (3) KV utilization slot vs block; (4) preemptions vs `MV_KV_MB`; (5) prefix cache cold vs warm TTFT; (6) a one-step profiler trace or flame graph with the GPU/Python split. Every graph regenerated by `bench/sweep.sh` + `bench/plot.py`.

---

## Design divergences from real vLLM you must be able to explain

Each of these is a deliberate cut. Know the cut, the cost, and what the real thing does. You have now read the real thing for each of them in the "Look first" steps; the file names are repeated here so the README can cite them.

**Gathered blocks instead of a paged attention kernel.** You copy every active sequence's KV into a contiguous tensor every step; vLLM's kernel (its own `csrc/attention/paged_attention_v1.cu`, or FlashAttention/FlashInfer with a block table via `vllm/v1/attention/backends/flash_attn.py`) reads blocks in place. Cost: a full read+write of the active KV per step (≈ B × L × 12 KiB), which at 16 × 1500 tokens rivals the weight read; plus 2 × 24 index kernels. Benefit: any attention implementation works, including yours. Say the number you measured.

**No CUDA graphs.** A *CUDA graph* is a recording of a fixed sequence of kernel launches that the driver can replay as one unit. A 0.5B decode step is ~400 small kernel launches; at ~5–10 µs each that is a large fraction of a 10 ms step, and it is why your batch-1 tok/s is ~35% of the bandwidth ceiling. vLLM captures the decode forward into CUDA graphs per batch size (`vllm/v1/worker/gpu_model_runner.py`, `capture_model` / `_dummy_run`) and replays it, cutting launch overhead to near zero; the price is static shapes, which is why vLLM pads decode batches to captured sizes. Your gather and mask building are dynamic-shape and would need to move out of the graph.

**No chunked prefill / no mixed batches.** You run prefill and decode as separate forward passes and a long prompt stalls every running decode for its full duration — your ITL p99. vLLM v1 schedules a token budget per step (`Scheduler.schedule`, `token_budget`), fills it with running decodes first, then slices waiting prompts into chunks to fill the remainder, all in one forward pass; the model runner sees a flattened `[total_tokens]` batch with per-request boundaries (`_prepare_inputs`, `query_start_loc`). That single change is what turns ITL p99 from "prefill duration" into "step duration".

**Python scheduler in the event loop, one process.** `step()` runs on the same thread as the HTTP handlers; handlers wait for the step, and the step waits for handlers (tokenization, detokenization, SSE encoding all share the loop) — you saw the number with `PYTHONASYNCIODEBUG`. vLLM v1 runs `EngineCore` in a separate process talking over ZMQ (`vllm/v1/engine/core.py`, `EngineCoreProc`), so the GPU loop never yields to HTTP, and it pipelines: the scheduler prepares step *n+1* while step *n* runs (async scheduling, `step_with_batch_queue`), and detokenization happens in the API-server process (`vllm/v1/engine/output_processor.py`). Your per-step `.tolist()` sync is also a divergence — vLLM avoids syncing on sampled tokens where it can.

**One sequence per request, greedy/temperature only.** No `n > 1`, no beam search, no copy-on-write of shared blocks when two sequences fork from one prompt (which is where ref counts earn their keep in vLLM), no logprobs, no guided decoding, no speculative decoding.

**Recompute-only preemption, FCFS scheduling, no priorities.** vLLM has priority scheduling (`--scheduling-policy priority`), and its v1 preemption is recompute only (v0 offered swap); you have neither swap nor priorities.

**Prefix caching is per process with a salted hash.** vLLM uses a stable hash (`--prefix-caching-hash-algo`, and includes extra keys such as LoRA id and multimodal hashes in `hash_block_tokens`) so the cache is deterministic across restarts and can be extended to external KV stores. Your Project 3 makes the cross-machine version of this the whole point.

**Tensor parallel, quantization, LoRA, multimodal:** absent, and you should say so before being asked.

## The five-minute narrative this project buys you

"I built a continuous-batching server over an inference engine I wrote from scratch. The API layer is FastAPI with SSE; the engine loop runs as a background task and does one iteration per step: admit waiting requests under a token budget, prefill them as a left-padded batch, decode every running sequence, retire the finished ones. I benchmarked it with Poisson arrivals against a static-batching baseline and got *[your ratio]* throughput at the same p99 — the win comes from filling slots that finish early. KV lives in 16-token blocks with a free list and per-sequence block tables; I measured utilization going from *[x]*% with contiguous preallocation to *[y]*% paged. I gather blocks into contiguous tensors for attention rather than writing a paged kernel, and I measured what that costs — *[z]* ms per step at batch 16. When blocks run out I preempt the newest arrival and recompute on resume; I can show the preemption counter climbing as I shrink the budget, and explain why recompute beats swap for short sequences. Prefix caching hashes full blocks chained from position 0, with ref counts and LRU over unreferenced blocks; warm TTFT on a 700-token system prompt dropped from *[a]* to *[b]* ms. The `/metrics` endpoint exposes TTFT and ITL histograms, queue depth, KV utilization and preemptions, and the throughput-vs-p99 sweep puts the knee at *[λ]* req/s on a 2060 — which is the number I'd use to size replicas. The things I didn't build, and why they matter: a real paged-attention kernel, CUDA graphs, chunked prefill, and moving the engine out of process — each one addresses a cost I can point to in my own measurements." Then stop talking and let them pick which one to dig into. You have measurements for every one of them.
