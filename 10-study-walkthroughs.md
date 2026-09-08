# Study-Guide Walkthroughs — every exercise in Modules 1–11, worked

Companion to `01-study-guide.md`. The guide tells you *what* to do; this document shows you *how*, one exercise at a time, with the arithmetic written out, the code complete enough to run, and the number you should see at the end. Where an exercise is already a project milestone, it says so and points at the milestone's walkthrough (`06-p1-walkthrough.md`, `07-p2-walkthrough.md`, `08-p3-walkthrough.md`, `09-p4-walkthrough.md`) instead of duplicating it.

How this document is built, and how to use it:

- **Every module opens with a "Terms this module uses" list.** Each term is defined in one plain sentence *before* the module uses it. If a word in an exercise is unfamiliar, it is defined either there or in the "Concepts you need before starting" section right below. You are not expected to know any of these words from the guide or the spec.
- **Every exercise has a "Look first" step before any code.** Your learning method is: print the real object, read its source, understand it, *then* write your own. So each exercise names the exact file, class, method, endpoint, or command to open in the real system — Hugging Face `transformers`, the vLLM v1 source, llama.cpp, Starlette/FastAPI, the Python standard library, `kubectl explain` — and what to notice there. Do the Look-first step even when it feels slow; it is the part that makes the exercise stick.
- **Every code block is commented line by line**, with tensor shapes wherever a tensor appears (`# [batch, num_heads, seq_len, head_dim]`) and a reason for every command-line flag. The comments are there so you can open the block cold in three months and read it without this document.

Conventions used throughout:

- **GB means 10⁹ bytes; GiB means 2³⁰ bytes.** Vendors quote bandwidth and VRAM in GB; `nvidia-smi` and `torch.cuda.mem_get_info` report in MiB/GiB. Half the "my numbers don't match" moments in this field are this unit mismatch. When a guide figure looks 7% off, check the unit first.
- **Hardware assumptions.** RTX 2060 (Turing, sm_75): 6 GB VRAM, 336 GB/s memory bandwidth (192-bit GDDR6 @ 14 Gbps). Peak compute is the number to be careful with. The commonly quoted "6.5 TFLOPS" is the **FP32 CUDA-core** figure (6.45 TFLOPS). Tensor-core FP16 is much higher on the datasheet: 51.6 TFLOPS with FP16 accumulate, and ~25.8 TFLOPS with FP32 accumulate, which is what PyTorch and vLLM use. In practice a 2060 sustains 15–25 TFLOPS on a large fp16 matmul. So: **I use 25.8 TFLOPS as the fp16 tensor-core peak for roofline arithmetic, and show the 6.5 TFLOPS figure alongside as the pessimistic bound.** The first thing you should do in Module 2 is measure it yourself (a script is given there); every roofline number below is easy to redo with your measured value.
- H100 SXM: 80 GB, 3.35 TB/s, ~990 TFLOPS dense fp16.
- Reference model configs are listed at the top of Module 1; every KV and parameter calculation reuses them.

---

## Concepts you need before starting

Read this once, top to bottom, before Module 1. Every term is defined in plain words with an analogy where it helps. Come back to it whenever a module uses a word you have forgotten; nothing later assumes more than what is here.

### The model and its pieces

- **Token.** The unit a language model reads and writes. Not a word: a tokenizer splits text into ~3–4-character chunks (the word "inference" might be two tokens). The model never sees letters, only integer token ids from a fixed list.
- **Vocabulary (vocab).** That fixed list of all tokens the model knows. Qwen2.5 has 151,936 of them; Llama-3 has 128,256. Every token has an id from 0 to vocab−1.
- **Embedding.** A lookup table with one row per vocab entry. Token id → a vector of numbers (896 numbers for Qwen2.5-0.5B). It turns an integer into something the arithmetic can work on.
- **Hidden size (d_model, `hidden_size`).** The length of the vector that represents each token as it flows through the model — 896 for Qwen2.5-0.5B, 4096 for Llama-3-8B. Every layer reads and writes vectors of this length.
- **Layer / block / decoder layer.** One repeated unit of the model: attention, then an MLP, each with a norm and a residual add. Qwen2.5-0.5B stacks 24 of them; Llama-3-70B stacks 80. "Layers" in a config file means the number of these blocks.
- **Attention.** The sub-block that lets each token look at earlier tokens and pull in information from them. It works by computing, for the current token, a score against every earlier token and taking a weighted average of their "values".
- **Q, K, V (query, key, value).** Three vectors computed from each token's hidden vector by three separate matrix multiplies. Analogy: the query is the question the current token asks; each earlier token's key is the label on its filing drawer; its value is what is inside the drawer. Score = how well the question matches the label; the answer is the values, weighted by scores.
- **Head / multi-head attention.** Attention is run several times in parallel with smaller vectors (14 "heads" of 64 numbers each for Qwen2.5-0.5B, instead of one head of 896), so different heads can attend to different things. `head_dim` is the size of one head's vector.
- **GQA (grouped-query attention).** A cost-saving trick: fewer K/V heads than Q heads, so several query heads share one key/value head. Qwen2.5-0.5B has 14 query heads but only 2 KV heads. **MHA** (multi-head) is the original: one KV head per query head. **MQA** (multi-query) is the extreme: one KV head total. **MLA** (multi-head latent attention, DeepSeek) compresses K and V into one small vector per token. The number of KV heads is what sets the KV-cache size (Module 3).
- **Causal mask.** The rule that a token may only attend to tokens *before* it, never after — because at generation time the future does not exist yet. Implemented by setting the scores for "future" positions to −∞ before the softmax, so they get weight 0.
- **Attention mask.** More generally, any per-position on/off switch on attention. The causal mask is one; a *padding* mask (below) is another.
- **Softmax.** Turns a list of raw scores into a list of positive numbers that sum to 1 — a set of weights, or a probability distribution. Bigger score → bigger share.
- **MLP / feed-forward / SwiGLU.** The other sub-block: two matrix multiplies that expand the token's vector (896 → 4864), a non-linearity, and one multiply that shrinks it back. SwiGLU is the modern variant with a third "gate" matrix. It works on each token independently; it holds most of the model's parameters.
- **RMSNorm.** A cheap normalization that rescales a token's vector so its root-mean-square is 1, then multiplies by a learned per-element weight. Applied before attention and before the MLP.
- **Residual connection.** Each sub-block *adds* its output to its input instead of replacing it, so the vector flowing through the model is a running sum that each layer nudges.
- **RoPE (rotary position embedding).** How the model knows the *order* of tokens: the Q and K vectors are rotated by an angle that depends on the token's position, so the score between two tokens depends on how far apart they are. It has no learned weights.
- **Logits.** The model's final output for one position: one raw score per vocabulary entry (151,936 numbers). Softmax over the logits gives the probability of each possible next token.
- **LM head.** The final matrix multiply that turns the last hidden vector (896 numbers) into logits (151,936 numbers). With **tied embeddings** it *is* the embedding table used in reverse, not a separate matrix.
- **Parameter / weight.** A number the model learned during training. "0.5B parameters" = 494 million such numbers. In this document "params" and "weights" are used interchangeably.
- **`config.json`.** The small file next to every Hugging Face checkpoint that lists the shapes: layers, hidden size, heads, KV heads, vocab. Every calculation in Modules 1–3 reads from it.
- **`state_dict`.** PyTorch's name for the dictionary `{parameter name → tensor}` that holds a model's weights. Loading weights into your own model is copying this dictionary in.

### Generation

- **Autoregressive generation.** The model produces one token at a time; each new token is appended to the input and the model runs again to produce the next. A 200-token answer is 200 forward passes.
- **Forward pass.** Running the input through every layer once to get logits. "Forward" because data flows forward through the layers (training also runs a backward pass; inference never does).
- **Prefill.** The first forward pass, over the whole prompt at once. All prompt tokens are processed in parallel in one go; its output is the first generated token plus the KV cache of the prompt. It is a big matrix multiply and is limited by compute.
- **Decode.** Every forward pass after prefill: one new token in, one token out, repeated. Each step is tiny in compute but must read every weight of the model once. It is limited by memory bandwidth.
- **KV cache.** During decode, each earlier token's K and V vectors are exactly what they were last step (they depend only on that token and the ones before it, which never change). Storing them instead of recomputing them is the KV cache. It is the biggest thing in GPU memory besides the weights, and it grows with every token of every live request.
- **Context / context length / sequence length.** The number of tokens the model is currently attending over: prompt plus everything generated so far. `max_model_len` is the cap.
- **Greedy decoding.** Always pick the highest-probability next token. Deterministic; the same prompt always gives the same answer.
- **Sampling.** Pick the next token *at random* according to the probability distribution — a token with probability 0.3 is chosen 30% of the time. **Temperature** scales the logits before softmax: below 1 sharpens the distribution toward greedy, above 1 flattens it. **Top-k / top-p** truncate the distribution to the k most likely tokens, or to the smallest set whose probabilities add up to p, before sampling.
- **Probability distribution over the vocabulary.** A list of 151,936 non-negative numbers summing to 1: the model's belief about what the next token is. Everything in sampling and speculative decoding is arithmetic on this list.
- **Rejection sampling.** A general trick for sampling from a distribution you want (`p`) using proposals from a distribution you have (`q`): draw from `q`, then keep the draw with a probability that corrects for the difference between `q` and `p`. Module 6 defines it fully before using it.
- **Speculative decoding.** A small "draft" model guesses several tokens cheaply; the big "target" model checks them all in one forward pass; a rejection-sampling rule keeps a prefix of the guesses. Output is identical in distribution to running only the big model.

### Numbers, memory, and speed

- **fp32 / fp16 / bf16 / int8 / int4 (dtypes).** How many bits store one number. fp32 = 4 bytes per weight, fp16 and bf16 = 2 bytes, int8 = 1 byte, int4 = half a byte. Halving the bytes halves the memory and halves the time to read the weights. bf16 has the same size as fp16 but a wider range; Turing GPUs (the 2060) do not support bf16, which is why every command here says `--dtype half` (fp16).
- **Quantization.** Storing weights (and sometimes activations) in fewer bits than they were trained in — e.g. rounding fp16 weights to int4 with a per-group scale factor. **Bits per weight** is the resulting average size (a "4-bit" scheme is usually ~4.5 bits per weight once the scale factors are counted). **Weight-only** quantization shrinks storage and read traffic but the multiplies still run in fp16 after de-quantizing; **weight-and-activation** (INT8/FP8) also makes the multiplies cheaper. **Calibration** is running a few hundred sample inputs through the model to choose the rounding scales well.
- **Perplexity.** The standard quality number for a language model: `exp(average negative log-probability of the true next token)` over a test text. Lower is better; a perplexity of 8 means the model is, on average, as unsure as if it were choosing among 8 equally likely tokens. Used to check how much quality quantization lost.
- **FLOP.** One floating-point operation: one multiply or one add. A multiply-add is 2 FLOPs. A matrix multiply of `[T, in]` by `[in, out]` costs `2 × T × in × out` FLOPs. **TFLOPS** = 10¹² FLOPs per second, the speed a GPU can do them.
- **Memory bandwidth.** How many bytes per second the GPU can read from its memory (VRAM) into its compute units. The 2060 does 336 GB/s; an H100 does 3,350 GB/s. Analogy: bandwidth is the width of the pipe between the warehouse (VRAM) and the factory (the compute units); FLOPS is the factory's speed. If the pipe is too narrow, the factory waits.
- **VRAM / HBM / GDDR.** The GPU's own memory, where weights and the KV cache live. HBM and GDDR are two memory technologies; the 2060 has 6 GB of GDDR6, the H100 has 80 GB of HBM3. Separate from the computer's RAM; moving data between them goes over PCIe and is slow.
- **"Bound" (memory-bound, compute-bound, launch-bound).** Which resource a job is waiting on. *Memory-bound*: the compute units are idle waiting for bytes; making the math faster changes nothing, making the bytes fewer does. *Compute-bound*: the bytes arrive faster than they can be used; the math is the limit. *Launch-bound*: the GPU is idle waiting for the CPU to tell it what to do next. Decode at batch 1 is memory-bound; prefill is compute-bound; a small model in a plain PyTorch loop is launch-bound.
- **Arithmetic intensity.** FLOPs performed per byte moved from memory. The single number that tells you which bound applies: compare it to the GPU's **ridge point** (peak FLOPS ÷ bandwidth). Below the ridge → memory-bound; above → compute-bound.
- **Roofline.** A plot with arithmetic intensity on the x-axis and achievable FLOPS on the y-axis. The "roof" is two lines: a slope (bandwidth × intensity) on the left and a flat ceiling (peak FLOPS) on the right; they meet at the ridge point. Module 2 draws it.
- **Kernel.** One function that runs on the GPU — a matrix multiply, a softmax, an add. A forward pass is a sequence of hundreds of kernels. **Kernel launch** is the CPU telling the GPU to run one; each launch costs ~5–15 µs of CPU time regardless of how much work the kernel does. **Kernel fusion** merges several small kernels into one to cut launches. **Eager mode** = PyTorch's default of launching kernels one by one as Python reaches them.
- **CUDA graph.** A recording of a whole sequence of kernels that can be replayed with a single launch, eliminating the per-kernel CPU overhead. vLLM uses them for decode; `--enforce-eager` turns them off.
- **Tensor cores.** Dedicated matrix-multiply hardware inside the GPU, much faster than the general "CUDA cores" but only for certain dtypes (fp16/bf16, and int8/fp8 on newer chips). The 2060's fp16 tensor-core peak is ~4× its fp32 CUDA-core peak.
- **MFU (model FLOPs utilization).** Achieved FLOPS ÷ peak FLOPS. 50% is good for prefill; decode at batch 1 is ~1%.
- **sm_75 / sm_80 / Turing / Ampere / Hopper.** NVIDIA GPU generations. RTX 2060 = Turing = compute capability 7.5. A100 = Ampere = 8.0. H100 = Hopper = 9.0. Many fast kernels (FlashAttention 2, Marlin) require sm_80 or newer, which is why several steps here have a "fallback on Turing" note.
- **WSL2.** Windows Subsystem for Linux, the Linux VM your GPU work runs in on the Windows machine. The NVIDIA driver is the Windows one; CUDA inside WSL2 talks to it.

### Serving

- **Serving / a serving engine.** A long-running process that loads the model once and answers many requests from many clients over HTTP. vLLM, SGLang, TensorRT-LLM, llama.cpp's server are serving engines; Project 2 builds a small one.
- **Request / sequence.** One client's prompt and its generated answer. Inside an engine each in-flight request is a "sequence" with its own KV cache.
- **Batching.** Running several requests through the model in one forward pass, so the weights are read once for all of them instead of once each. The main way an engine gets throughput.
- **Padding.** When sequences of different lengths are batched into one rectangular tensor, the short ones are filled with dummy tokens up to the longest length. A **padding mask** hides those dummies from attention. Modern engines avoid padding by concatenating all tokens into one flat list instead.
- **Static vs continuous batching.** Static: a batch is formed, run to completion, and only then is the next batch admitted — finished sequences leave their slots idle. Continuous (iteration-level): after every single decode step, finished sequences leave and waiting ones join. vLLM does the latter.
- **Scheduler.** The engine component that decides, before every step, which sequences run in that step.
- **Paged / block-based KV cache.** Instead of reserving one contiguous slab of memory per sequence, the KV cache is cut into fixed-size **blocks** (16 tokens each in vLLM) handed out on demand, like pages in an operating system's virtual memory. A sequence's **block table** is the list of which blocks hold its tokens. **PagedAttention** is the attention kernel that reads through the block table.
- **Preemption.** When KV memory runs out, the scheduler evicts a running sequence (frees its blocks) and re-runs its prefill later. It shows up as a sudden latency spike.
- **Chunked prefill.** Splitting a long prompt's prefill into pieces spread across several steps, so it does not stall the decode steps of other sequences.
- **Prefix caching.** Reusing KV blocks from an earlier request when a new request starts with the same tokens (e.g. the same system prompt). Saves the prefill of the shared prefix. A **hit** reuses; a **miss** recomputes.
- **Latency vs throughput.** Latency: how long *one* request waits (seconds). Throughput: how much work the system does per second across *all* requests (tokens/s or requests/s). They trade off: batching more raises throughput and raises every individual request's latency.
- **TTFT (time to first token).** From sending the request to receiving the first generated token. Dominated by queueing plus prefill.
- **ITL (inter-token latency).** The gap between two consecutive streamed tokens. What the user experiences as "typing speed".
- **TPOT (time per output token).** Total generation time ÷ number of output tokens for one request — an average ITL that includes the first token. **E2E / e2el** = end-to-end latency of the whole request.
- **p50 / p90 / p99 (percentiles).** Sort all measured latencies; p50 is the median (half were faster), p99 is the value that 99% were faster than — i.e. the worst 1%. p99 is what a spike looks like; p50 is what most users see. "p99 up, p50 flat" means a minority of requests got slow.
- **Histogram (Prometheus sense).** A metric that counts how many observations fell into each of a fixed set of latency buckets (≤ 10 ms, ≤ 25 ms, …). Percentiles are estimated from those bucket counts. A histogram metric is what you need to compute p99 over time.
- **SLO (service-level objective).** A target for a metric, stated with a percentile and a threshold: "TTFT p99 ≤ 1.5 s". **Goodput** is throughput counting only requests that met their SLO.
- **Closed-loop vs open-loop load.** Closed-loop: a fixed number of clients, each sending its next request only after its previous one finished (concurrency is capped, queues cannot grow). Open-loop: requests arrive on a schedule regardless of whether earlier ones finished (queues *can* grow — that is how you see p99 blow up).
- **Poisson arrivals.** The standard model for "independent users showing up at random": the gap between consecutive arrivals is drawn from an exponential distribution with a chosen mean rate. `random.expovariate(rate)` generates one gap.
- **Streaming / SSE (server-sent events).** An HTTP response that stays open and sends tokens as they are produced, one small chunk at a time, instead of the whole answer at the end. SSE is the specific text format: lines starting `data: ...` separated by blank lines, ending with `data: [DONE]`.
- **Event loop / asyncio.** Python's way of running many I/O-bound tasks in one thread: each task gives control back (`await`) whenever it waits on something, and the loop runs whichever task is ready. A **coroutine** is an `async def` function; a **Future** is a placeholder for a result that will arrive later; an **`asyncio.Queue`** is a queue whose `put`/`get` yield to the loop.
- **GIL.** Python's global interpreter lock: only one thread runs Python code at a time. The reason vLLM puts the GPU loop in a separate *process* from the HTTP server.
- **Gateway / router / worker.** In Project 3: the gateway is the HTTP front door; the router picks which worker (a vLLM server) handles each request; workers run the model.
- **Consistent hashing / hash ring.** A way to map keys (e.g. a prompt's prefix) to servers so that adding or removing one server moves only ~1/N of the keys instead of reshuffling everything. Servers and keys are both hashed to points on a circle; a key goes to the next server clockwise. **Virtual nodes** put each server at many points on the circle so load spreads evenly.
- **LRU (least recently used).** An eviction policy: when the cache is full, throw out the entry that has gone longest without being touched. Used for KV blocks in vLLM and as a classic interview exercise.
- **Rate limiter / token bucket.** A component that caps how many requests a client may send per second. A token bucket refills at a steady rate and each request spends one token; a full bucket allows a burst.
- **Tensor parallelism (TP).** Splitting every weight matrix across several GPUs so each does a slice of every matrix multiply; the slices are combined with an **allreduce** (defined in Module 7) after each layer. Needs a fast link (NVLink). **Pipeline parallelism (PP)**: giving whole layers to each GPU, passing activations along like an assembly line. **Data parallelism (DP)**: full copies of the model on each GPU, each serving different requests.
- **NVLink / PCIe / InfiniBand.** Links between GPUs (NVLink, ~900 GB/s, inside one machine), between a GPU and the rest of the computer (PCIe, ~25–64 GB/s), and between machines (InfiniBand, ~50 GB/s). Which link a collective runs over decides whether TP is cheap or ruinous.
- **Disaggregated serving.** Running prefill on one pool of GPUs and decode on another, shipping the prompt's KV cache between them.
- **Kubernetes terms.** *Pod*: one running container (a vLLM server). *HPA*: horizontal pod autoscaler, adds/removes pods based on a metric. *Readiness probe*: the health check a pod must pass before traffic is sent to it. *DaemonSet*: one pod on every node. `kind` is a local Kubernetes cluster in Docker.
- **Cold start.** The time from "start a new server" to "it answers its first request normally": pull the image, start the process, load weights, warm up.

### Tools

- **Profiler.** A tool that records what a program was doing over time — which functions ran and for how long, which GPU kernels ran and when. `torch.profiler` records both CPU and GPU sides of a PyTorch program; `py-spy` samples a running Python process from outside without changing it; **Nsight Systems** (`nsys`) is NVIDIA's whole-system timeline.
- **Trace.** The profiler's output: a timeline of events. Chrome-trace JSON files open in Perfetto (`ui.perfetto.dev`).
- **Flame graph.** A picture of where time went: each bar is a function, its width is the fraction of samples in which that function was on the stack, and bars stack upward by call depth. Wide bars near the top are where time is actually spent.
- **Prometheus / scrape / `/metrics`.** Prometheus is a metrics database that *pulls*: every few seconds it does an HTTP GET to each server's `/metrics` endpoint — that GET is a **scrape** — and stores the numbers it finds there with a timestamp. **PromQL** is its query language. **Grafana** draws PromQL results as dashboards.
- **Counter / gauge.** Two Prometheus metric types: a counter only goes up (total tokens generated), so you look at its `rate()`; a gauge goes up and down (KV cache usage now).
- **Alert rule.** A PromQL expression that, when true for a set duration (`for: 3m`), fires a notification.

---

## Module 1 — Transformer internals

**What you are actually learning here.** Every inference number — memory, FLOPs, KV size, why decode is slow — is derived from the shapes of the matmuls in one decoder block. If you can write those shapes from memory you can derive everything else at a whiteboard; if you can't, you are memorizing formulas. The interview question this module answers is the opener nearly every inference loop starts with: *"Walk me through what happens, matmul by matmul, when the model generates one token."* The follow-up is *"how many parameters is that, and where do they live?"* — which is exercise 3.

### Terms this module uses

- **Matmul (matrix multiply).** `x @ W`: an input of shape `[T, in]` times a weight of shape `[in, out]` gives `[T, out]`. Every "projection" below is a matmul, optionally plus a bias vector.
- **Projection (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).** A matmul that maps a vector from one size to another. The names are Hugging Face's and you must reuse them exactly so weights load.
- **`nn.Linear(in, out, bias)`.** PyTorch's matmul-plus-bias layer. It stores a weight of shape `[out, in]` (note the order) and, if `bias=True`, a bias of shape `[out]`.
- **Bias.** A vector added after a matmul. Qwen2.5 has biases on q/k/v; Llama has none. They are tiny in count but easy to forget when adding up parameters.
- **View / transpose / reshape.** Ways to reinterpret the same numbers as a different shape without copying them. `[1, 1, 896]` viewed as `[1, 1, 14, 64]` splits the 896 into 14 heads of 64.
- **`repeat_kv`.** The GQA helper that copies each of the 2 KV heads 7 times so the 14 query heads each have a matching K and V to attend against.
- **Tied embeddings (`tie_word_embeddings`).** The LM head reuses the embedding table's storage, so the model has one big `[vocab, hidden]` matrix, not two.
- **`torch.allclose(a, b, atol)`.** True if every element of `a` is within `atol` of the matching element of `b`. The parity test between your model and HF's.
- **`inspect.getsource(fn)`.** Python's way to print a function's source code from inside a notebook — the core of the "look first" method.

### Reference configs (used in every module)

| Model | layers | hidden | q heads | kv heads | head_dim | intermediate | vocab | params | tied embeddings |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 896 | 14 | 2 | 64 | 4864 | 151,936 | 494,032,768 | yes |
| Qwen2.5-1.5B | 28 | 1536 | 12 | 2 | 128 | 8960 | 151,936 | 1,543,714,304 | yes |
| Llama-3-8B | 32 | 4096 | 32 | 8 | 128 | 14,336 | 128,256 | ~8.03 B | no |
| Llama-3-70B | 80 | 8192 | 64 | 8 | 128 | 28,672 | 128,256 | ~70.6 B | no |

Qwen2.5 attention has biases on q/k/v projections (not on o_proj); Llama has no biases anywhere. Both use SwiGLU MLPs with no biases, RMSNorm, RoPE.

### Look first — the real model, printed and read (20 min, do this before any exercise)

Load the HF model once and keep it in the notebook for the whole module.

```python
import inspect                                     # lets you print a function's source from inside the notebook
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load in fp32 on CPU for reading: dtype and device do not matter for looking at structure.
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# 1. The whole tree. Notice: Qwen2ForCausalLM → .model (Qwen2Model) → .embed_tokens, .layers (24 × Qwen2DecoderLayer),
#    .norm, .rotary_emb — and a separate .lm_head Linear(896 → 151936, bias=False) which is TIED to embed_tokens (exercise 3).
print(hf_model)

# 2. One block, so the four named sub-modules and their shapes are in front of you:
#    self_attn (q_proj 896→896 w/ bias, k_proj 896→128 w/ bias, v_proj 896→128 w/ bias, o_proj 896→896 no bias),
#    mlp (gate_proj 896→4864, up_proj 896→4864, down_proj 4864→896, all no bias),
#    input_layernorm and post_attention_layernorm (Qwen2RMSNorm, each holding one 896-vector).
print(hf_model.model.layers[0])

# 3. The config: every number in the reference table comes from here.
print(hf_model.config)                              # num_hidden_layers=24, hidden_size=896, num_attention_heads=14,
                                                    # num_key_value_heads=2, intermediate_size=4864, vocab_size=151936,
                                                    # tie_word_embeddings=True, rms_norm_eps=1e-6

# 4. Read the three forward functions. type(obj) gives the class; .forward is the method; getsource prints it.
layer0 = hf_model.model.layers[0]
print(inspect.getsource(type(layer0).forward))            # Qwen2DecoderLayer.forward: residual → input_layernorm → self_attn → add;
                                                          # residual → post_attention_layernorm → mlp → add. Two residual adds per block.
print(inspect.getsource(type(layer0.mlp).forward))        # Qwen2MLP.forward: down_proj(act_fn(gate_proj(x)) * up_proj(x)) — one line, SwiGLU
print(inspect.getsource(type(layer0.input_layernorm).forward))   # Qwen2RMSNorm.forward: upcast to fp32, x * rsqrt(mean(x²) + eps), cast back, * weight
print(inspect.getsource(type(layer0.self_attn).forward))  # Qwen2Attention.forward: the q/k/v projections, view to heads, RoPE, cache update,
                                                          # then a call into an attention function (eager / sdpa / flash) — exercise 1's recitation
```

What to notice, in order: (a) the block's forward is *two* residual adds around *two* normed sub-blocks, nothing else; (b) the MLP forward is one line — three matmuls and an elementwise multiply; (c) RMSNorm upcasts to fp32 before the mean — the reason your own norm must do the same to match to 1e-5; (d) in the attention forward, `k_proj` and `v_proj` output 128 numbers, not 896, and there is a `repeat_kv` (or `num_key_value_groups`) somewhere between the cache update and the score computation — that is GQA. Once you have seen each of these in HF's own code, the exercises below are about reproducing them, not discovering them.

### Exercise 1 — multi-head self-attention from memory

**Look first.** Re-open `inspect.getsource(type(hf_model.model.layers[0].self_attn).forward)` from above and, with a pen, write next to each line the shape of the tensor it produces for batch 1, one token, using the 0.5B config. Then close it. The list below is the answer key.

**This is Project 1, Milestone 1.** Your `Attention` class in `qwen_from_scratch.py` (with GQA, RoPE, `repeat_kv`, causal mask) already passes the HF parity test; see `06-p1-walkthrough.md`. The remaining part of this exercise is the *from memory* part: close the file and rewrite the forward pass on a blank page in under 15 minutes, with shapes annotated at every line. For one token at decode time, with the Qwen2.5-0.5B shapes, the sequence you must be able to recite is:

```
x            [1, 1, 896]                        # [batch, seq_len, hidden]: one token's hidden vector
q = x Wq+bq  [1, 1, 896]   → view [1, 1, 14, 64] → transpose → [1, 14, 1, 64]     # [batch, q_heads, seq_len, head_dim]
k = x Wk+bk  [1, 1, 128]   → view [1, 1, 2, 64]  → transpose → [1, 2, 1, 64]      # [batch, kv_heads, seq_len, head_dim]
v = x Wv+bv  [1, 1, 128]   → same as k
RoPE on q, k (position = current sequence length)                                  # rotates q and k in place; shapes unchanged
append k, v to the cache → K, V are [1, 2, T, 64]                                  # T = tokens so far, including this one
repeat_kv: K, V → [1, 14, T, 64]   (each kv head is shared by 14/2 = 7 q heads)
scores = q Kᵀ / √64           [1, 14, 1, T]                                        # one score per (head, earlier token)
softmax over the last dim     (no mask needed at decode: the new token is last)    # weights over the T earlier tokens
out = scores V                [1, 14, 1, 64] → transpose, reshape → [1, 1, 896]    # weighted sum of values, heads re-joined
out = out Wo                  [1, 1, 896]                                          # o_proj back to hidden size
```

The self-check is a `torch.allclose` against `hf_model.model.layers[0].self_attn` at atol 1e-5, exactly as in `06-p1-walkthrough.md`. If you can produce that in 15 minutes from a blank file you're done with exercise 1.

### Exercise 2 — greedy generation loop with real weights

**Look first.** `print(inspect.getsource(hf_model.generate))` is too long to read; instead open `transformers/generation/utils.py` (find it with `import transformers.generation.utils as u; print(u.__file__)`) and read `_sample` — the loop that HF actually runs for `generate()`. Notice the shape of the loop: call the model, take `logits[:, -1, :]`, pick a token (`argmax` when `do_sample=False`), `torch.cat` it onto `input_ids`, repeat until a stop condition. Your `generate_naive` is that loop with everything but those five lines removed.

**This is Project 1, Milestone 2** (`generate_naive` in the last section of `06-p1-walkthrough.md`). Done when the loop prints ` Paris` after "The capital of France is" and you have the tok/s-vs-context plot. Nothing to add here.

### Exercise 3 — parameter count of one block, by hand

**Look first.** Before counting anything on paper, make PyTorch show you the pieces so the arithmetic below has something to match against:

```python
# Every (name, shape) pair in block 0. This is the list you are about to add up by hand.
for name, p in hf_model.model.layers[0].named_parameters():
    print(f"{name:45s} {tuple(p.shape)!s:20s} {p.numel():>12,}")     # numel() = number of elements = in × out (or out, for a bias)
# Expect 12 lines: 4 attention weights + 3 attention biases + 3 MLP weights + 2 norm weights.
# Note nn.Linear stores weight as [out, in]: q_proj.weight is [896, 896], k_proj.weight is [128, 896], down_proj.weight is [896, 4864].

# The top-level pieces outside the blocks:
print(hf_model.model.embed_tokens.weight.shape)     # torch.Size([151936, 896])  — the embedding table
print(hf_model.model.norm.weight.shape)             # torch.Size([896])          — final RMSNorm
print(hf_model.lm_head.weight.shape)                # torch.Size([151936, 896])  — same shape as the embedding; exercise asks: same STORAGE?
```

The rule: a `Linear(in, out, bias)` has `in × out` weights plus `out` biases. An `Embedding(vocab, hidden)` has `vocab × hidden`. An RMSNorm has `hidden`. Everything else (RoPE tables, masks, softmax) is parameter-free.

**Attention (Qwen2.5-0.5B, GQA, biased q/k/v):**

```
q_proj:  896 × (14 × 64) + 14 × 64  =  896 × 896 + 896  =  802,816 + 896  =  803,712
k_proj:  896 × (2 × 64)  + 2 × 64   =  896 × 128 + 128  =  114,688 + 128  =  114,816
v_proj:  same as k_proj                                                     =  114,816
o_proj:  (14 × 64) × 896, no bias   =  896 × 896                            =  802,816
                                                                    attention = 1,836,160
```

Notice what GQA did: with 14 kv heads (MHA) k_proj and v_proj would each be 803,712 — attention would be 3,213,952 instead of 1,836,160. GQA cuts attention parameters by 43% here, and (Module 3) cuts KV-cache size by 7×. It changes nothing about q_proj or o_proj.

**MLP (SwiGLU, three unbiased linears):**

```
gate_proj: 896 × 4864 = 4,358,144
up_proj:   896 × 4864 = 4,358,144
down_proj: 4864 × 896 = 4,358,144
                  MLP = 13,074,432
```

**Norms:** two RMSNorms per block, each with one `hidden`-sized weight: `2 × 896 = 1,792`.

**One block:** `1,836,160 + 13,074,432 + 1,792 = 14,912,384`. The MLP is 87.7% of the block. That is the number to remember: in a modern small LLM the block is mostly MLP, and attention's parameter share is small — but attention's *KV cache* is what dominates memory at long context, which is why the two are always discussed separately.

**The whole model:**

```
24 blocks:      24 × 14,912,384 = 357,897,216
embed_tokens:   151,936 × 896   = 136,134,656
final norm:                     =         896
lm_head:                        =           0   (tie_word_embeddings = true → reuses embed_tokens)
                          total = 494,032,768
```

That is exactly what HF reports for Qwen2.5-0.5B. The embedding table is 27.6% of the model — a fact worth remembering because in fp16 those 272 MB of weights are read *twice* per token (once as a lookup, which touches only one row, and once as the full LM-head matmul, which touches all of it). If the embeddings were untied the model would have 630 M parameters; the "0.5B" naming only works because they're tied.

**Verify in one line:**

```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# parameters() yields every learnable tensor exactly once (deduplicated by identity — see below); numel() counts its elements.
print(sum(p.numel() for p in m.parameters()))                       # 494032768  — the whole model
print(sum(p.numel() for p in m.model.layers[0].parameters()))       # 14912384   — one block

# data_ptr() is the address of the tensor's first element in memory. Equal addresses = the SAME storage, not a copy.
print(m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr())   # True: same tensor
```

The third line is the proof that tying means *the same storage*, not a copy: `parameters()` deduplicates by identity, so the head contributes nothing to the count.

**Expected:** 494,032,768 and 14,912,384. If you get something else:

| You got | Reason |
|---|---|
| 630,167,424 | You counted a separate lm_head (added 151,936 × 896 again) |
| 494,031,872 | You forgot the final `model.norm` (896) |
| 494,005,120 (27,648 short) | You forgot the q/k/v biases: 24 × (896 + 128 + 128) = 27,648. Short by 24 × 896 = 21,504 → only the q bias missing; by 6,144 → both k and v biases missing |
| ~527 M | You used 14 kv heads (MHA) for k/v — check `num_key_value_heads` |
| Block = 14,910,592 | You forgot the two norms (1,792) |

**Generalising to Llama-3-8B (no biases, untied):** attention = 4096×4096 (q) + 4096×1024 (k) + 4096×1024 (v) + 4096×4096 (o) = 41,943,040; MLP = 3 × 4096 × 14,336 = 176,160,768; norms 8,192; block = 218,112,000; 32 blocks = 6,979,584,000; embeddings 128,256 × 4096 = 525,336,576; lm_head another 525,336,576; final norm 4,096; total = 8,030,261,248. That is the 8.03 B in the table. Do this one on paper too — it is the model interviewers most often use. (Look first for this one: `AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")` — needs a Hub login — or just open the model's `config.json` in the browser and read `num_key_value_heads: 8`, `intermediate_size: 14336`, `tie_word_embeddings: false`.)

---

## Module 2 — Inference arithmetic and the GPU mental model

**What you are actually learning here.** Two numbers describe any (model, GPU) pair: how many bytes must move per token, and how many FLOPs must execute per token. Divide each by the hardware's bandwidth and compute peak and the larger of the two times is your floor. At batch 1, decode moves every weight byte for a handful of FLOPs — it is memory-bound — while prefill reuses each weight byte across hundreds of tokens and is compute-bound. The interview question is *"why is decode memory-bound and prefill compute-bound, and what tok/s would you expect for model X on GPU Y?"* — and the follow-up, *"what does batching change?"* — is the crossover calculation at the end of exercise 2.

### Terms this module uses

- **FLOP.** One floating-point operation — a single multiply or a single add of two numbers. A "multiply-accumulate" (`a × b + c`) is 2 FLOPs. A matmul of `[T, in]` by `[in, out]` does `T × in × out` multiply-accumulates, so `2 × T × in × out` FLOPs. **GFLOP** = 10⁹ FLOPs, **TFLOP** = 10¹²; **TFLOPS** (with an S) is TFLOPs *per second* — a speed, not an amount.
- **Peak compute.** The most FLOPs per second the GPU can do if every compute unit is busy every cycle. The 2060's fp16 tensor-core peak is ~25.8 TFLOPS. Real kernels get 60–90% of it at best.
- **Memory bandwidth.** How many bytes per second can flow from the GPU's memory (VRAM) into its compute units. The 2060: 336 GB/s. Analogy: the GPU is a factory (the compute units) fed by a conveyor belt (bandwidth) from a warehouse (VRAM). The belt carries 336 GB every second no matter how fast the factory is. If each item off the belt needs only a moment of work, the factory stands idle waiting for the belt — that is what "memory-bound" means.
- **"Bound".** The resource a job is waiting on. *Memory-bound*: the belt is the limit — time = bytes ÷ bandwidth; a faster factory changes nothing, fewer bytes helps. *Compute-bound*: the factory is the limit — time = FLOPs ÷ peak; a faster belt changes nothing. *Launch-bound*: neither — the GPU is idle because the CPU has not told it what to do next yet (each kernel launch costs ~10 µs of CPU time).
- **Arithmetic intensity.** FLOPs performed per byte moved from memory: `intensity = FLOPs / bytes`. It is a property of the *workload*, not the GPU. Decode at batch 1 does ~2 FLOPs per 2-byte weight → 1 FLOP/byte. Prefill over 512 tokens does 512× more FLOPs on the same bytes → ~500 FLOP/byte.
- **Ridge point.** A property of the *GPU*: `peak FLOPS ÷ bandwidth`, in FLOP/byte. It is the intensity at which the belt and the factory are exactly matched. A workload below the ridge is memory-bound; above it, compute-bound. RTX 2060: 25.8e12 / 336e9 ≈ 77 FLOP/byte. H100: ≈ 295.
- **Roofline plot.** A graph with intensity on the x-axis (log scale) and *achievable* FLOPS on the y-axis. Two lines form a "roof": on the left, a slope `bandwidth × intensity` (you cannot compute faster than the bytes arrive); on the right, a flat ceiling at peak FLOPS. They meet at the ridge. You plot each workload as a dot at its intensity; the roof above it is the best it can do.

```
 achievable
 FLOPS (log)
    ^
    |                     ridge point (77 FLOP/B on a 2060)
    |                          |
    |                          v
    |                    ______________________________  <- flat ceiling = peak compute (25.8 TFLOPS)
    |                   /
    |                  /   * prefill @512 tok (~535 FLOP/B): compute-bound, up by the ceiling
    |                 /
    |                /   <- slope = bandwidth x intensity (336 GB/s x FLOP/B)
    |               /
    |              /
    |   * decode  /
    |   batch 1 (1 FLOP/B): memory-bound, stuck low on the slope
    +-------------------------------------------------------> arithmetic intensity (FLOP/byte, log)
```

Batching decode moves its dot to the right (same bytes, B× the FLOPs); once it passes the ridge, decode becomes compute-bound. That is exercise 2, step 5.

- **Bandwidth floor / compute floor.** The two times you compute for any step: `bytes ÷ bandwidth` and `FLOPs ÷ peak`. The larger one is your floor; you cannot go faster than it.
- **Ceiling (tok/s).** 1 ÷ the floor, in tokens per second. "The ceiling is 340 tok/s" means no implementation can beat 340 on that hardware for that model at batch 1.
- **Kernel launch.** The CPU-side cost of starting one GPU function (~5–15 µs). A decode step runs ~400 of them; at 10 µs each that is 4 ms of CPU time the GPU spends partly idle.
- **CUDA graphs.** Record the whole step's kernels once, replay with one launch: the fix for launch-bound. `torch.compile(mode="reduce-overhead")` and vLLM's default (non-eager) mode both use them.

### Look first — ask the GPU what it is (5 min)

```bash
# What the driver thinks the card is. memory.total is in MiB (5,xxx on a "6 GB" card — the GB/GiB gap from the conventions above).
# clocks.max.memory is the memory clock in MHz: bandwidth = clock × 2 (DDR) × bus_width_bits / 8 — for the 2060: 7000 MHz × 2 × 192 / 8 = 336 GB/s.
nvidia-smi --query-gpu=name,memory.total,clocks.max.memory,compute_cap --format=csv
```

```python
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, p.total_memory / 2**30, "GiB", p.multi_processor_count, "SMs", f"sm_{p.major}{p.minor}")
# RTX 2060: 30 SMs, sm_75. The SM count × clock × FLOPs-per-SM-per-clock is where the datasheet's 25.8 TFLOPS comes from;
# you do not need to reproduce that — you are about to measure it instead.
```

### Measure your peak first (15 min)

Before trusting any datasheet, run this once on the 2060:

```python
import torch, time

# --- Compute peak: one big fp16 matmul, timed. 4096³ is large enough that launch overhead is negligible. ---
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)   # [4096, 4096] fp16 = 32 MiB
y = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)   # [4096, 4096]
for _ in range(10): x @ y                      # warm-up: first calls pay for cuBLAS kernel selection and lazy CUDA init
torch.cuda.synchronize(); t = time.perf_counter()   # synchronize = wait until the GPU has actually finished everything queued so far
for _ in range(50): x @ y                      # 50 timed matmuls; each is 2 × 4096³ ≈ 137 GFLOP
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 50   # seconds per matmul (sync again so the timer includes the GPU work)
print(f"fp16 matmul: {2 * 4096**3 / dt / 1e12:.1f} TFLOPS")     # FLOPs per matmul ÷ seconds ÷ 1e12 → TFLOPS

# --- Bandwidth: touch 1 GiB in place. add_(1) reads every byte and writes every byte → 2 GiB of traffic per call. ---
buf = torch.empty(2**30, device="cuda", dtype=torch.uint8)   # 1 GiB of bytes ([2^30] uint8)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): buf.add_(1)                                # read + write 1 GiB, in place
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 20  # seconds per pass
print(f"bandwidth: {2 * 2**30 / dt / 1e9:.0f} GB/s")           # bytes moved (2 × 1 GiB) ÷ seconds ÷ 1e9 → GB/s (decimal GB, to match the datasheet)
```

**Expected:** 15–25 TFLOPS and 280–320 GB/s (you never see the full 336; ~85–90% is normal). Write your two numbers down — they replace the datasheet values in everything below.

### Exercise 1 — decode ceiling for Qwen2.5-0.5B fp16 on the 2060

**Look first.** Two things to see with your own eyes before doing the division. (1) How many bytes the weights actually are: `sum(p.numel() * p.element_size() for p in m.parameters())` with `m` loaded in `torch_dtype=torch.float16` → 988,065,536. (2) That a decode step really reads all of them: in the Project-1 engine, put `torch.cuda.synchronize()` around one KV-cached decode step and time it; you will get 6–9 ms, not 2.94 — step 4 below explains the gap, and Module 11 item 1 lets you *see* it in a profiler trace.

**Step 1: bytes per decode step.** At batch 1, generating one token means running the whole forward pass once, and every weight must be read from HBM/GDDR into the SMs. There is no reuse: each weight is used for exactly one multiply-add per token.

```
weights   = 494,032,768 params × 2 bytes (fp16) = 988,065,536 bytes ≈ 0.988 GB (0.92 GiB)
KV read   = 12,288 bytes/token (Module 3) × context length
            at 1k context: 12.6 MB — 1.3% of the weight traffic, ignore at this scale
activations: a few MB, ignore
```

**Step 2: time per step at the bandwidth floor.**

```
t_step = 0.988 GB / 336 GB/s = 2.94 ms   →   ceiling = 1 / 2.94 ms = 340 tok/s
```

(The guide's "~336 tok/s" uses "~1 GB"; 340 is the same estimate with the exact byte count. With your *measured* ~300 GB/s it's ~305 tok/s. All three are "the ceiling"; don't argue over the third digit.)

**Step 3: fp32 halves it.** 494 M × 4 bytes = 1.976 GB → 5.88 ms → **170 tok/s**. Same FLOPs, twice the bytes, half the speed. This is the whole argument for weight quantization in one line, and it's why Module 5 exists: int8 would be 680 tok/s, int4 1,360 — *if* nothing else became the bottleneck, which at this model size it does (next).

**Step 4: measure your engine and explain the gap.** Your Project-1 KV-cached loop (M3) will land somewhere around 100–150 tok/s in fp16, i.e. 30–45% of ceiling. The interviewer wants the breakdown, not the excuse. At this model size the dominant term is not bandwidth at all:

```
kernel launches per step ≈ 24 layers × ~15 kernels (2 norms, 4 projections, RoPE,
                            repeat_kv, matmul, softmax, matmul, 3 MLP linears, silu, mul,
                            2 residual adds …) + head ≈ 360–400 launches
launch + Python overhead  ≈ 8–15 µs each  →  3–6 ms per step
```

That is *larger* than the 2.94 ms bandwidth floor. A plain PyTorch loop on a 0.5B model is **launch-bound, not memory-bound** — the GPU is idle waiting for the CPU to enqueue the next kernel. Proof: run the same loop with the model in fp32. If you were bandwidth-bound tok/s would halve; if it barely changes, you're launch-bound. The fixes, in the order engines apply them: CUDA graphs (capture the whole step once, replay with one launch — this is what vLLM's `--enforce-eager` turns *off*), kernel fusion (`torch.compile`, fused RMSNorm+residual, fused SwiGLU), and batching (amortise the launches over many sequences). Put the fp32-vs-fp16 experiment and the launch-count estimate in the Project-1 README; it is the honest "where the rest went" story.

### Exercise 2 — prefill FLOPs for 512 tokens, and why prefill is compute-bound

**Look first.** PyTorch can count the FLOPs of a forward pass for you, so you can check the hand arithmetic against the real model before trusting it:

```python
import torch
from torch.utils.flop_counter import FlopCounterMode      # counts FLOPs of every matmul/conv that runs inside the `with` block
from transformers import AutoModelForCausalLM

m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16).cuda().eval()
ids = torch.randint(0, m.config.vocab_size, (1, 512), device="cuda")   # [batch=1, seq_len=512] random token ids — FLOPs don't depend on which tokens
with torch.no_grad(), FlopCounterMode(display=True) as fc:              # display=True prints a per-module table when the block exits
    m(ids)                                                              # one prefill forward over 512 tokens → logits [1, 512, 151936]
print(f"{fc.get_total_flops() / 1e9:.1f} GFLOP")                        # expect ≈ 528 (linears + attention), matching step 2 below; the table
                                                                        # shows the LM head (2 × 512 × 896 × 151,936 ≈ 139 GFLOP) as the single biggest line
```

**Step 1: FLOPs from the linear layers.** A matmul of an `[T, in]` activation with an `[in, out]` weight costs `2 × T × in × out` FLOPs (one multiply and one add per weight per token). Summed over all weights that's `2 × params × T`, where params counts every matrix that gets multiplied — for Qwen2.5-0.5B that includes the tied embedding table used as the LM head, so the full 494 M is the right number (the embedding *lookup* is free, but the same matrix is multiplied at the head).

```
linear FLOPs = 2 × 494,032,768 × 512 = 5.059 × 10¹¹ = 505.9 GFLOP
```

**Step 2: the attention term.** Per layer, scores = QKᵀ costs `2 × T² × head_dim × n_q_heads` and PV costs the same. With `head_dim × n_q_heads = hidden`:

```
attention FLOPs = 4 × T² × hidden × layers = 4 × 512² × 896 × 24 = 22.5 GFLOP
```

That is 4.5% of the linear term at T = 512. The ratio grows linearly with T (it's `2T / params-per-layer-ish`); at T = 8k for this model attention is ~70% of the linear FLOPs. Kernels that respect causality skip half of it, so quote 11–22 GFLOP depending on the kernel; it doesn't change the conclusion.

```
total ≈ 528 GFLOP for a 512-token prefill
```

**Step 3: time at the compute floor vs the bandwidth floor.**

```
compute floor:  528 GFLOP / 25.8 TFLOPS = 20.5 ms   (or 81 ms at the 6.5 TFLOPS pessimistic figure)
bandwidth floor: 0.988 GB / 336 GB/s    =  2.9 ms   (weights are read ONCE for all 512 tokens)
```

Compute is the larger by 7× (or 28×): **prefill is compute-bound**. The weights are read once and reused 512 times; decode reads them once and reuses them once.

**Step 4: arithmetic intensity and the ridge point.** Arithmetic intensity = FLOPs / bytes moved. The GPU's ridge point = peak FLOPs / bandwidth; below it you are memory-bound, above it compute-bound.

```
prefill (512 tok):  528 × 10⁹ FLOP / 0.988 × 10⁹ B  ≈ 535 FLOP/byte
decode (batch 1):   2 × 494 M FLOP / 988 MB          ≈   1 FLOP/byte
RTX 2060 ridge:     25.8 × 10¹² / 336 × 10⁹          ≈  77 FLOP/byte   (19 at 6.5 TFLOPS)
H100 ridge:         990 × 10¹² / 3.35 × 10¹²         ≈ 295 FLOP/byte
```

Decode at batch 1 sits at 1 FLOP/byte — two orders of magnitude below either ridge. Prefill sits well above both. That's the whole module in four numbers. Go back to the roofline sketch in the terms list and place the two dots: decode is far down the slope on the left, prefill is under the flat ceiling on the right.

**Step 5: the crossover batch size.** Batching B sequences at decode multiplies the FLOPs by B but the weight bytes stay the same (each weight is read once and applied to B tokens). So decode's intensity is ≈ B FLOP/byte for fp16 weights (2 FLOPs per param ÷ 2 bytes per param — this tidy "intensity ≈ batch size" identity is worth remembering; for int4 weights it becomes ≈ 4B). Decode becomes compute-bound when B reaches the ridge:

```
RTX 2060: B* ≈ 77   (≈ 19 with the pessimistic peak)
H100:     B* ≈ 295
```

Two corrections to this in practice, both of which make B* *smaller*: (1) KV-cache reads grow with B (each sequence reads its own KV every step), so bytes aren't constant — at 4k context Qwen2.5-0.5B's KV is 50 MB per sequence, so at B = 20 the KV traffic equals the weight traffic and intensity flattens; (2) real kernels hit maybe 60–70% of peak. Interview answer: "batch 1 is 1 FLOP/byte; you need batch in the high tens on a consumer card and a few hundred on an H100 before decode is compute-bound, less at long context because KV reads add bytes per sequence." This is also why continuous batching (Module 4) matters: it keeps B high enough that the weight reads are amortised.

### Exercise 3 — the table

**Look first.** Every row starts from a `config.json`. Open each on the Hub in the browser (`huggingface.co/<model>/blob/main/config.json`) — `meta-llama/Meta-Llama-3-8B`, `meta-llama/Meta-Llama-3-70B`, `Qwen/Qwen2.5-0.5B` — and copy out `num_hidden_layers`, `num_key_value_heads`, `hidden_size`, `num_attention_heads` (head_dim = hidden ÷ heads unless `head_dim` is given), `torch_dtype`. Then `nvidia-smi --query-gpu=memory.total --format=csv` on any GPU you rent in Module 11 item 4, to see that "80 GB" reads as 81,559 MiB — the min-VRAM column is in the vendor's GB.

Weights = params × bytes/param. Batch-1 decode ceiling on an H100 = 3.35 TB/s ÷ weight bytes (single GPU; when the model doesn't fit on one, the ceiling row assumes TP with aggregate bandwidth, noted). "Min VRAM to serve" = weights + KV for one 8k-context request + ~1.5 GB for CUDA context, activations, and fragmentation, then rounded up to a card that exists. KV per token comes from Module 3: Llama-3-8B 131,072 B, Llama-3-70B 327,680 B, Qwen2.5-0.5B 12,288 B.

| Model | dtype | weights | KV @ 8k, 1 req | min VRAM to serve (practical card) | H100 batch-1 decode ceiling |
|---|---|---|---|---|---|
| Llama-3-8B (8.03 B) | fp16 | 16.06 GB | 1.07 GB | ~18.6 GB → 24 GB card (L4/A10G 24 GB, or H100) | 3350 / 16.06 = **209 tok/s** |
| | int8 | 8.03 GB | 1.07 GB | ~10.6 GB → 16 GB card | **417 tok/s** |
| | int4 | 4.02 GB | 1.07 GB | ~6.6 GB → 8 GB card (a 6 GB 2060 only at ≤2k ctx) | **834 tok/s** |
| Llama-3-70B (70.6 B) | fp16 | 141.2 GB | 2.68 GB | ~145 GB → 2×H100 80 GB (TP=2, tight — see Module 7) | 3350 / 141.2 = **24 tok/s** per GPU-equivalent; TP=2 → ~47, TP=8 → ~190 |
| | int8 | 70.6 GB | 2.68 GB | ~75 GB → 1×H100 80 GB, barely; 2× for headroom | **47 tok/s** |
| | int4 | 35.3 GB | 2.68 GB | ~40 GB → 1×H100, or 1×48 GB L40S | **95 tok/s** |
| Qwen2.5-0.5B (494 M) | fp16 | 0.99 GB | 0.10 GB | ~2.6 GB → anything | **3,390 tok/s** (unreachable: 0.3 ms/step is below launch overhead; real ≈ 500–1000) |
| | int8 | 0.49 GB | 0.10 GB | ~2 GB | 6,780 (nominal) |
| | int4 | 0.25 GB | 0.10 GB | ~2 GB | 13,560 (nominal — nobody int4s a 0.5B model; quality loss for no gain) |

Three things to say about the table out loud, because they're the follow-up questions:

1. **The 70B fp16 row is why TP exists.** 141 GB does not fit on any single GPU; the 24 tok/s "ceiling" is per-GPU bandwidth and the real ceiling scales with the number of GPUs whose bandwidth you aggregate (Module 7).
2. **Quantization moves the decode ceiling linearly** (fp16→int4 = 4×) but the *min VRAM* barely moves for the small model, because KV and overhead dominate. Quantization's memory win is only interesting when weights dominate the budget.
3. **The 0.5B row's ceiling is fiction.** At 0.3 ms per step you are limited by kernel launch latency and CPU scheduling, not by bytes. Small models are launch-bound; big models are bandwidth-bound; the ceiling formula only applies when the step time is comfortably above ~2–3 ms without CUDA graphs, or ~0.5 ms with them.

**Done-when check.** Given "Llama-3-8B fp16 on an A100 (2 TB/s)": 16 GB / 2 TB/s = 8 ms → 125 tok/s batch 1; KV per token 128 KiB; 80 GB − 16 GB − 2 GB overhead = 62 GB → ~470k tokens of KV → e.g. 58 sequences at 8k; ridge ≈ 312 TFLOPS / 2 TB/s = 156, so decode goes compute-bound around batch 150 before KV reads. If you can produce that in under two minutes you're done.

---

## Module 3 — KV cache and attention variants

**What you are actually learning here.** At decode time the only thing that changes from step to step is one new token; every previous token's K and V vectors are identical to what they were last step. Caching them turns each decode step from O(T) recomputation into O(1) new work plus an O(T) *read*, and turns the memory budget from "weights" into "weights + KV", where KV grows with batch × context. The size formula is the single most-asked calculation in inference interviews (*"how much KV cache does Llama-3-8B need at 8k context, batch 32?"*), and the reason GQA/MQA/MLA exist is that the formula has `kv_heads` in it.

### Terms this module uses

- **What K and V actually are.** For every token that has passed through a layer, the attention block computed two vectors from that token's hidden state: `k = x @ Wk + bk` (the **key**, 128 numbers for Qwen2.5-0.5B: 2 heads × 64) and `v = x @ Wv + bv` (the **value**, same size). The key is what *later* tokens compare their query against to decide how much to attend to this token; the value is what they pull in if they do. Each layer has its own K and V for each token, because each layer's `x` is different.
- **Why they can be cached.** Token 5's K and V at layer 3 depend only on token 5's hidden state entering layer 3, which depends only on tokens 1–5 (the causal mask means later tokens never influence earlier ones). Tokens 1–5 never change once they are in the sequence. So token 5's K and V at layer 3 are the same number at step 6, step 7, step 500. Recomputing them every step would be doing the same matmul over and over and getting the same answer; storing them is the KV cache. The *query* is not cached because only the newest token needs one, and it is used once.
- **What is *not* cached.** Q, the attention scores, the MLP activations, the residual stream — all of it is recomputed for the new token only and thrown away. The cache holds exactly two tensors per layer.
- **Cache shape.** Per layer, K is `[batch, kv_heads, seq_len, head_dim]` and V is the same. Appending a token means growing the `seq_len` axis by one.
- **Per-token KV bytes.** The size of one token's K and V across *all* layers: `2 × layers × kv_heads × head_dim × bytes_per_element`. Multiply by tokens live (seq_len × batch) for the total.
- **`DynamicCache`.** Hugging Face's KV cache object: a list of per-layer K and V tensors that grows with `torch.cat` on every step. `get_seq_length()` tells you how many tokens it holds; `crop(n)` drops everything after position `n` (used in Module 6).
- **MHA / GQA / MQA / MLA.** Defined in the concepts section; here the only thing that matters is the number of KV heads each implies: q_heads, a divisor of q_heads, 1, and (for MLA) a single compressed latent vector instead of heads.
- **O(T) vs O(1).** "Order of T": work that grows in proportion to the sequence length, versus work that is constant. Without the cache, each decode step recomputes K and V for all T tokens (O(T) matmuls); with it, only the new token's (O(1)) plus a read of the T stored ones.

### Look first — HF's cache, printed and read (15 min)

```python
import inspect, torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# 1. The update method is the entire KV-cache idea in ~15 lines: for layer `layer_idx`, either store (first call) or torch.cat
#    the new key/value states onto the stored ones along dim=-2 — the seq_len axis — and return the full K and V.
#    Notice the shapes in the docstring/comments: key_states is [batch_size, num_heads, seq_len, head_dim].
print(inspect.getsource(transformers.cache_utils.DynamicCache.update))

# 2. Watch it grow. Run one prefill and one decode step and print what the cache holds after each.
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16).cuda().eval()
ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()   # [1, 5] — five prompt tokens
cache = DynamicCache()                                                        # empty: get_seq_length() == 0
with torch.no_grad():
    out = m(input_ids=ids, past_key_values=cache, use_cache=True)             # prefill: all 5 tokens at once
print(cache.get_seq_length())                                                 # 5
k0, v0 = cache[0]                                                             # layer 0's (K, V) — DynamicCache is indexable by layer (older versions: cache.key_cache[0])
print(k0.shape, v0.shape)                                                     # torch.Size([1, 2, 5, 64]) each: [batch, kv_heads=2, seq_len=5, head_dim=64]
nxt = out.logits[0, -1].argmax().view(1, 1)                                   # greedy next token id as [1, 1]
with torch.no_grad():
    out = m(input_ids=nxt, past_key_values=cache, use_cache=True)             # decode: ONE token in, cache supplies the other 5
print(cache.get_seq_length(), cache[0][0].shape)                              # 6, torch.Size([1, 2, 6, 64]) — seq_len axis grew by one

# 3. Size it. Sum the bytes in every layer's K and V and compare with the formula below (2 × 24 × 2 × 64 × 2 = 12,288 per token).
n_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in cache)   # iterate layers as (K, V) pairs
print(n_bytes, n_bytes / cache.get_seq_length())                              # 73728, 12288.0
```

If `cache[0]` or iteration fails on your `transformers` version, use `cache.key_cache[0]` / `cache.value_cache[0]` (older) or `cache.layers[0].keys` (newer) — the point is the same: two tensors per layer, `[batch, kv_heads, seq_len, head_dim]`, growing along `seq_len`.

### Exercise 1 — add a KV cache, measure before/after at 1k context

**Look first.** You have now seen `DynamicCache.update` do `torch.cat` on every step. Notice that this makes every step allocate and copy the whole cache — an O(T) copy hidden inside the O(1) idea. It is fine for HF; it is the thing your Project-1 M3 should *not* do (pre-allocate `[1, kv_heads, max_len, head_dim]` and write into a slice), and it is what vLLM's block-based cache in Module 4 removes entirely.

**This is Project 1, Milestone 3.** See `06-p1-walkthrough.md` for the M2 baseline and M3. The acceptance number: decode tok/s at 1k context should go from "falling with context" (naive) to "flat" (cached), and the flat value is what you compare to the 340 tok/s ceiling from Module 2. Record both numbers; they are the first sentence of your Project-1 interview story.

### Exercise 2 — the size formula, re-derived

**Look first.** Open `meta-llama/Meta-Llama-3-8B/config.json` on the Hub and read the four fields: `num_hidden_layers: 32`, `num_key_value_heads: 8`, `hidden_size: 4096`, `num_attention_heads: 32` (so head_dim = 128), `torch_dtype: bfloat16` (2 bytes). Those four numbers and the batch/context you are asked about are the entire input to the formula.

**Derivation, not memorization.** For each layer, each kv head stores one K vector and one V vector per token, each of `head_dim` elements:

```
bytes = 2 (K and V) × layers × kv_heads × head_dim × seq_len × bytes_per_elem × batch
```

The `2` is K-and-V; `layers × kv_heads × head_dim` is "how many numbers per token per K"; `seq_len × batch` is "how many tokens are live". Group it as **(per-token bytes) × (tokens)** and you can do any model in your head.

**Llama-3-8B at 8k context, batch 32, fp16:**

```
per-token bytes = 2 × 32 layers × 8 kv heads × 128 head_dim × 2 bytes
                = 2 × 32 × 8 × 128 × 2
                = 131,072 bytes = 128 KiB per token

tokens          = 8,192 × 32 = 262,144

total           = 131,072 × 262,144 = 34,359,738,368 bytes
                = 34.36 GB = 32.0 GiB
```

Per sequence at 8k it's exactly 1 GiB — a handy anchor: "Llama-3-8B: 128 KiB per token, 1 GiB per 8k sequence." The guide's "~32 GB" is this figure quoted in GiB; both are right, and you should say "32 GiB, about 34 GB" so the interviewer knows you know the difference.

Consequence: on an 80 GB H100 with 16 GB of fp16 weights and ~2 GB overhead, 62 GB is left for KV → 62e9 / 131,072 ≈ 473k tokens. Batch 32 at 8k (262k tokens, 34.4 GB) fits; batch 64 at 8k (524k tokens, 68.7 GB) does not; and at 32k context only 14 sequences fit (473k / 32,768). The weights are a fifth of the card; the KV cache is what decides how many users it serves. "KV cache, not weights, limits batch at long context" is this arithmetic.

### Exercise 3 — the MHA counterfactual

**Look first.** Compare two configs side by side: `meta-llama/Llama-2-7b-hf/config.json` has no `num_key_value_heads` field at all (HF defaults it to `num_attention_heads` = 32 → MHA), while Llama-3-8B has `num_key_value_heads: 8`. Same layer count, same hidden size, same head_dim — the only KV-relevant difference is that one field, and it is a 4× difference in cache size.

If Llama-3-8B had 32 KV heads (MHA, one K/V per query head):

```
per-token bytes = 2 × 32 × 32 × 128 × 2 = 524,288 bytes = 512 KiB
total at 8k × 32 = 137.4 GB = 128 GiB
```

4× — the ratio of q heads to kv heads. On the H100 above, 62 GB of KV room would hold 118k tokens instead of 473k: 14 sequences at 8k instead of 57. GQA is why Llama-3-8B serves 4× the concurrency of an MHA model with the same weights. MQA (1 kv head) would be 32×; MLA (DeepSeek) compresses K and V into a single ~512-dim latent per token per layer, so per-token bytes ≈ 2 × layers × 512 × 2 — for a 60-layer model that's ~123 KB vs ~1 MB+ for GQA-8 at 128 head_dim, the "~90%+ reduction" the guide mentions.

### Qwen2.5-0.5B per-token KV, and how many tokens fit in 3 GB

```
per-token = 2 × 24 layers × 2 kv heads × 64 head_dim × 2 bytes = 12,288 bytes = 12 KiB

tokens in 3 GB  = 3,000,000,000 / 12,288 = 244,140
tokens in 3 GiB = 3,221,225,472 / 12,288 = 262,144  (exactly 2¹⁸ — GiB and power-of-two dims line up)
```

So on the 2060 after the 0.99 GB of weights, ~3 GB of KV (leaving ~2 GB for CUDA context, activations, Windows' display driver under WSL2) is 244k tokens: 59 concurrent sequences at 4k context, or 16 at 15k. That is why Project 2's acceptance criterion (≥16 concurrent streams on the 2060 with the 0.5B model) is comfortably achievable — KV is not the limit at this model size; scheduling overhead is.

### The reusable function

**Look first.** `from transformers import AutoConfig; c = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B"); print(c)` — and note that this config has *no* `head_dim` field (it must be derived as 896 ÷ 14), while e.g. `google/gemma-2-9b` has an explicit `head_dim: 256` that is *not* `hidden ÷ heads`. That is why the function below reads `head_dim` first and falls back to the division.

```python
from dataclasses import dataclass

# Bytes per stored element for each dtype name. fp8/int8 rows are for KV-cache quantization (Module 5): same formula, 1 byte per element.
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}

@dataclass
class KVConfig:
    """The three config fields that set KV size. Everything else in config.json is irrelevant to the cache."""
    layers: int      # num_hidden_layers — the cache has one (K, V) pair per layer
    kv_heads: int    # num_key_value_heads — NOT num_attention_heads; this is the GQA field
    head_dim: int    # elements per head; K for one token in one layer is kv_heads × head_dim numbers

    @classmethod
    def from_hf(cls, config):
        """Works for Qwen2/Llama-style HF configs."""
        # Explicit head_dim wins if present (Gemma-2 etc.); otherwise derive it from hidden_size / num_attention_heads.
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return cls(config.num_hidden_layers, config.num_key_value_heads, head_dim)

def kv_bytes(config: KVConfig, seq: int, batch: int = 1, dtype: str = "fp16") -> int:
    """Total KV-cache bytes for `batch` sequences of `seq` tokens each."""
    # 2 = one K and one V; × layers × kv_heads × head_dim = numbers per token; × bytes per number.
    per_token = 2 * config.layers * config.kv_heads * config.head_dim * DTYPE_BYTES[dtype]
    return per_token * seq * batch          # × how many tokens are live

# The four reference models from Module 1's table, as (layers, kv_heads, head_dim).
QWEN05 = KVConfig(24, 2, 64)
QWEN15 = KVConfig(28, 2, 128)
LLAMA8 = KVConfig(32, 8, 128)
LLAMA70 = KVConfig(80, 8, 128)

if __name__ == "__main__":
    # Each assert is one of the hand calculations above; if you change the formula and one fails, the formula is wrong, not the number.
    assert kv_bytes(QWEN05, 1) == 12_288                                   # Qwen2.5-0.5B per token
    assert kv_bytes(LLAMA8, 1) == 131_072                                  # Llama-3-8B per token (128 KiB)
    assert kv_bytes(LLAMA8, 8192, 32) == 34_359_738_368                    # exercise 2: 32 GiB
    assert kv_bytes(KVConfig(32, 32, 128), 8192, 32) == 4 * kv_bytes(LLAMA8, 8192, 32)   # exercise 3: MHA is exactly 4×
    print("Qwen2.5-1.5B per token:", kv_bytes(QWEN15, 1))       # 28,672
    print("Llama-3-70B per token:", kv_bytes(LLAMA70, 1))       # 327,680
    print("Llama-3-70B 2k prompt MB:", kv_bytes(LLAMA70, 2048) / 1e6)   # 671.1 — used in Module 7
```

Check it against HF: `KVConfig.from_hf(AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B"))` should give `(24, 2, 64)`. If `head_dim` in a config disagrees with `hidden // n_heads` (it does for some newer models), the config's explicit `head_dim` wins — that's why the function reads it first.

**Done-when check.** Open any `config.json` on the Hub, find the four fields, and say the per-token bytes within ten seconds. Practice on Mistral-7B (32 layers, 8 kv heads, 128 → 131,072, same as Llama-3-8B), Gemma-2-9B (42 layers, 8 kv heads, 256 → 344,064), Qwen2.5-7B (28 layers, 4 kv heads, 128 → 57,344 — note how much smaller than Llama-3-8B's).

---

## Module 4 — Serving engines: vLLM and SGLang internals

**What you are actually learning here.** A serving engine is a scheduler that decides, every few milliseconds, which sequences get to run one step, plus a memory manager that makes the KV cache behave like paged virtual memory so that decision can be made freely. Everything else — continuous batching, chunked prefill, preemption, prefix caching — is a policy on top of those two. The interview question is *"narrate the life of a request through vLLM"*, and the follow-ups probe whether you know *why* each mechanism exists (what breaks without it). Exercises 1 and 2 give you first-hand numbers to hang that narration on; exercise 3 (Project 2) makes you build the two core pieces yourself.

### Terms this module uses

- **vLLM v1 / V0.** vLLM's engine was rewritten in 2025; "v1" is the new engine (default now), "V0" the old one. File paths below are v1's (`vllm/v1/...`). On Turing you may need V0 (see the setup notes).
- **Block (KV block).** A fixed-size chunk of KV-cache memory holding 16 tokens' K and V for every layer. The engine hands out blocks one at a time as a sequence grows, so no sequence needs a contiguous reservation. "# GPU blocks: N" in the startup log is how many the card has room for.
- **Block table.** For one sequence, the list of block ids that hold its tokens in order. The attention kernel reads it to find where token *t*'s K and V are. It is the same idea as a page table in an OS.
- **Slot mapping.** For each token being processed this step, the exact physical position (block id × 16 + offset) to write its new K and V into.
- **Token budget (chunked prefill).** The maximum number of tokens the scheduler will process in one step across all sequences. A long prompt gets only part of its tokens each step until it is done, so the running decodes keep stepping.
- **Preemption (recompute vs swap).** When no free block is left for a running sequence, the scheduler frees a low-priority sequence's blocks and puts it back in the waiting queue. *Recompute*: re-run its prefill later. *Swap*: copy its blocks to CPU memory and back. vLLM v1 recomputes.
- **Prefix caching / block hash.** Each full block's contents are identified by a hash of its 16 tokens *chained* with the previous block's hash (so the hash encodes everything from the start of the prompt). A new request whose first blocks hash to blocks already in the pool reuses them and skips their prefill.
- **Free list / refcount.** The pool's list of unused blocks, and a per-block count of how many sequences currently use it. A cached block with refcount 0 stays in the pool (for a future prefix hit) until memory is needed, then is evicted LRU.
- **Engine core / API server split.** vLLM runs the GPU loop in one process and the HTTP server plus tokenization in another, connected by a ZMQ socket, so Python's GIL in the HTTP process never stalls the GPU loop.
- **Executor / worker / model runner.** Executor: fans a step out to one worker per GPU. Worker: owns one GPU. Model runner: builds the batch tensors and runs the model on that GPU.
- **Attention backend.** Which attention kernel vLLM uses (FlashAttention, Triton, xformers, FlexAttention). Chosen by GPU generation; Turing cannot run FlashAttention 2.
- **`--enforce-eager`.** Turn off CUDA-graph capture. Faster startup, less memory, slower decode.
- **`--gpu-memory-utilization`.** The fraction of VRAM vLLM is allowed to take, for weights + activations + KV blocks. It sizes the block pool from what is left after weights.
- **`--max-model-len` / `--max-num-seqs`.** Longest allowed sequence, and most sequences that may be running at once.
- **`vllm bench serve`.** vLLM's built-in load generator (formerly `benchmarks/benchmark_serving.py`). It sends prompts to a running server and reports TTFT/TPOT/ITL percentiles and throughput.
- **`--max-concurrency C` / `--request-rate R`.** Closed-loop (C clients each waiting for their previous request) vs open-loop (send at rate R regardless). Defined in the concepts section.
- **Metrics names (`vllm:...`).** The counters and gauges vLLM exposes at `/metrics`: `num_requests_waiting`, `kv_cache_usage_perc`, `num_preemptions_total`, `prefix_cache_hits`, and so on. Module 11 item 3 builds a dashboard from them.
- **Radix tree / RadixAttention (SGLang).** SGLang's prefix cache: a tree of tokens where every path from the root is a cached prefix. Equivalent purpose to vLLM's block hashes, finer granularity.

### Look first — open these before running the benchmark (one sitting per row, ~3 hours total)

Everything you measure in exercises 1 and 2 is produced by the code in this table. Read it *before* you run the sweep, so that when throughput bends over or TTFT halves you already know which function did it. Clone the repo (`git clone https://github.com/vllm-project/vllm && cd vllm`) or read on GitHub; paths are under `vllm/`; class names are stable across recent releases even where files move. Read with the file open and the question "what does this class own?" in mind.

| Step | File | What to find |
|---|---|---|
| 1. HTTP in | `entrypoints/openai/api_server.py` → `entrypoints/openai/serving_completion.py` / `serving_chat.py` | FastAPI route → `OpenAIServingChat.create_chat_completion`: applies chat template, builds `SamplingParams`, calls `engine.generate(...)` which returns an async iterator of `RequestOutput` |
| 2. Async front-end | `v1/engine/async_llm.py` (`AsyncLLM`) | Owns the `Processor` and `OutputProcessor`; `generate()` = `add_request()` + a per-request `asyncio.Queue` that the output loop feeds |
| 3. Preprocessing | `v1/engine/processor.py` (`Processor`) | Tokenization, prompt-length validation, multimodal handling → an `EngineCoreRequest` (ids, params, arrival time) |
| 4. Process boundary | `v1/engine/core_client.py` (`EngineCoreClient`, `AsyncMPClient`) | The engine core runs in a **separate process**; requests and outputs cross a ZMQ socket. This is the "API-server/engine split" — the GIL-bound HTTP/tokenization work never stalls the GPU loop |
| 5. The loop | `v1/engine/core.py` (`EngineCore.step`) | `scheduler.schedule()` → `executor.execute_model(scheduler_output)` → `scheduler.update_from_output(...)`. One iteration = one token for every running sequence. Read this function twice |
| 6. Scheduler | `v1/core/sched/scheduler.py` (`Scheduler.schedule`) | Two queues: `running` and `waiting`. Each step: give every running request its next token budget, then admit waiting requests while `token_budget` (chunked prefill) and KV blocks allow; on allocation failure, **preempt** the lowest-priority running request (free its blocks, move it back to waiting; recompute on resume). Chunked prefill is just "a prefill request may get fewer tokens than it has left" |
| 7. KV memory | `v1/core/kv_cache_manager.py` (`KVCacheManager`), `v1/core/block_pool.py` (`BlockPool`), `v1/core/kv_cache_utils.py` | `allocate_slots(request, num_new_tokens)` returns block ids; `BlockPool` is a free list + hash→block map with LRU eviction of *unreferenced* cached blocks; `hash_request_tokens` / `hash_block_tokens` is the chained block hash behind prefix caching. The per-request `block_ids` list is the **block table** PagedAttention reads |
| 8. Executor | `v1/executor/abstract.py`, `v1/executor/multiproc_executor.py` | Fan out `execute_model` to one worker per TP rank (Module 7) |
| 9. Model runner | `v1/worker/gpu_model_runner.py` (`GPUModelRunner.execute_model`) | Builds the flattened input batch (all sequences' new tokens concatenated — no padding), positions, slot mapping (token → physical KV slot), and the attention metadata; picks a captured **CUDA graph** for the batch size if not eager; runs the model; calls the sampler |
| 10. Attention | `v1/attention/backends/flash_attn.py` (or `triton_attn.py` on Turing) | Where the block table becomes a kernel argument: the kernel gathers K/V from non-contiguous blocks — this *is* PagedAttention |
| 11. Sampler | `v1/sample/sampler.py` | Temperature / top-k / top-p / penalties over the whole batch's logits in one go; returns `SamplerOutput` |
| 12. Back out | `Scheduler.update_from_output` → `v1/engine/output_processor.py` (`OutputProcessor`) | Appends tokens, checks stop conditions, frees blocks on finish; the output processor detokenizes incrementally and pushes to each request's queue → SSE chunk to the client |

For each exercise below, the specific rows to have fresh in mind are named in its own Look-first line. When you can walk 1→12 aloud in five minutes, naming what each of Scheduler / KVCacheManager / BlockPool / GPUModelRunner owns, you're done with the reading. The five "why" questions to be ready for: why a separate engine process (GIL + tokenization jitter); why blocks (fragmentation: contiguous per-sequence reservations waste 60–80%, blocks waste < 4%, and sharing prefixes becomes possible); why preempt-by-recompute rather than swap (recompute is bandwidth-free and simple; swap to CPU needs PCIe and bookkeeping); why chunked prefill (a 4k-token prefill would otherwise stall every running sequence's decode step by ~200 ms+ — chunking bounds ITL jitter); why CUDA graphs for decode only (decode shapes repeat; prefill shapes don't).

### Getting vLLM to run on a 6 GB Turing card under WSL2

Turing (sm_75) is at the bottom edge of vLLM's support. Expect one evening of friction and don't spend more than that.

```bash
# inside WSL2 Ubuntu; the NVIDIA driver is the WINDOWS driver — never install one inside WSL
nvidia-smi                                   # must show the 2060 and a CUDA 12.x driver; if this fails, fix Windows-side first
python3 -m venv ~/venv-vllm && source ~/venv-vllm/bin/activate    # a venv just for vLLM: it pins its own torch, which will fight any other install
pip install -U pip                           # old pip picks wrong wheels
pip install vllm                             # pulls a matching torch
python -c "import torch; print(torch.cuda.get_device_capability())"   # must print (7, 5) — confirms torch sees the card through WSL2
```

Then the serve command. Every flag has a reason:

- `--dtype half` — fp16. Turing has no bf16; vLLM will refuse or silently upcast otherwise.
- `--max-model-len 4096` — caps per-sequence KV; Qwen's default 32k would make vLLM reserve too much per sequence and refuse to start.
- `--gpu-memory-utilization 0.85` — fraction of VRAM vLLM may take. WSL2 + the Windows desktop already holds 0.3–0.8 GB; 0.9 may OOM at startup.
- `--max-num-seqs 16` — concurrency cap; keep small on 6 GB.
- `--enforce-eager` — skips CUDA-graph capture: saves ~0.3–0.5 GB and startup time. Remove later to measure the ITL win.
- `--port 8000` — where the OpenAI-compatible API listens.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --enforce-eager \
  --port 8000
```

**Memory budget (why these numbers):** 6 GB × 0.85 = 5.1 GB budget. Weights: 1.54 B × 2 = 3.09 GB. Activation workspace + CUDA context ≈ 0.4–0.6 GB. Leaves ~1.5 GB for KV → Qwen2.5-1.5B is 28,672 bytes/token → ~52k tokens of KV → 12 sequences at 4k. vLLM prints `# GPU blocks: N` at startup; with 16-token blocks, N × 16 should be ≈ 50k. If it's under ~20k, lower `--max-model-len` or raise utilization.

**If the attention backend refuses Turing** (V1 engine's default FlashAttention needs sm_80): try `VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve …` (or `FLEX_ATTENTION`). If that fails too, pin an older release that still has the V0 engine with the xformers backend: `pip install vllm==0.8.5` and run with `VLLM_USE_V1=0`. Note the version in your README; "V0 on Turing because V1's FA2 backend needs Ampere" is itself a correct and specific thing to say in an interview.

**If fp16 doesn't fit with enough KV headroom** (blocks < 20k, or OOM under load): use a 4-bit checkpoint. Qwen publishes `Qwen/Qwen2.5-1.5B-Instruct-AWQ` and `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4`. Weights drop to ~1.1 GB, freeing ~2 GB for KV. AWQ's fast Marlin kernel needs sm_80 — on Turing vLLM falls back to the older AWQ kernel, and GPTQ's exllama kernel runs on sm_75; try AWQ first and GPTQ if it complains. This is also exercise 1 of Module 5, so you'll want the quantized server anyway.

**Sanity check:**

```bash
# -s: silent (no progress bar).  -H: the JSON content-type header the API requires.  -d: the request body.
# temperature 0 = greedy, so the answer is deterministic.  jq pulls just the generated text out of the response JSON.
curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":"The capital of France is","max_tokens":8,"temperature":0}' | jq .choices[0].text
```

Expected: ` Paris.` and, in the server log, the decode throughput line. The batch-1 ceiling for this model on the 2060 is 336 / 3.09 = **109 tok/s**; vLLM with CUDA graphs should reach 70–90; with `--enforce-eager` maybe 45–60.

Also, right now while the server is idle, look at what it exposes:

```bash
curl -s localhost:8000/metrics | grep -v '^#' | grep '^vllm:' | head -40   # the counters/gauges the sweep will move; grep -v '^#' drops the HELP/TYPE comment lines
```

### Exercise 1 — benchmark sweep: throughput vs p50/p99

**Look first.** Reading-path rows 5, 6 and 9: `EngineCore.step` (one loop iteration = one token for every running sequence), `Scheduler.schedule` (how many sequences get in — the thing `--max-concurrency` is pushing on), and `GPUModelRunner.execute_model` (the flat, padding-free batch). Then open the benchmark itself: `vllm/benchmarks/serve.py` (or `benchmarks/benchmark_serving.py` in older checkouts) and find the function that computes the metrics — search for `ttft` and `itl` — so you know exactly how "TPOT" and "ITL" are defined by the tool (TPOT = (e2e − TTFT) ÷ (output tokens − 1); ITL = the list of gaps between streamed chunks). Also `vllm bench serve --help` — every flag below is in there.

vLLM ships its own load generator. In current releases it is `vllm bench serve`; in older ones it's `python benchmarks/benchmark_serving.py` in the repo with identical flags. The flags:

- `--backend vllm --host localhost --port 8000` — which server, and that it speaks vLLM's OpenAI-compatible API.
- `--model Qwen/Qwen2.5-1.5B-Instruct` — must match what the server loaded (it goes into every request body).
- `--dataset-name random --random-input-len 256 --random-output-len 128` — synthetic prompts of 256 random tokens, asking for 128 output tokens each. Random tokens defeat the prefix cache, which is what you want for a raw throughput number.
- `--num-prompts 200` — how many requests in total per run; enough that warm-up is amortised.
- `--max-concurrency $C --request-rate inf` — closed-loop: exactly C in flight at all times (rate inf = "send the next as soon as one finishes, up to C").
- `--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99` — which latency metrics and which percentiles to report.
- `--save-result --result-dir results --result-filename c$C.json` — write the numbers to a JSON per concurrency level, for the plot.

```bash
mkdir -p results                                 # one JSON per concurrency level goes here
for C in 1 2 4 8 16 32; do                       # the sweep: concurrency doubles each run
  vllm bench serve \
    --backend vllm --host localhost --port 8000 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 \
    --max-concurrency $C --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir results --result-filename c$C.json
done
```

`--max-concurrency C` with `--request-rate inf` is a closed-loop test: C clients, each sending the next request as soon as its previous one finishes. That's the right shape for "throughput vs latency at a given concurrency". (`--request-rate R` without a concurrency cap is an open-loop Poisson test — the shape you want for Project 2 M6 and the p99 story in Module 8, because open-loop is what lets queues build.)

Each JSON has `output_throughput` (tok/s), `mean_ttft_ms`, `p99_ttft_ms`, `mean_tpot_ms`, `p99_tpot_ms`, `p99_itl_ms`, `p99_e2el_ms`. Plot:

```python
import json, glob, matplotlib.pyplot as plt

# Load every results/c*.json and sort by the concurrency it was run at (the JSON records its own max_concurrency).
rows = sorted((json.load(open(f)) for f in glob.glob("results/c*.json")), key=lambda r: r["max_concurrency"])
c   = [r["max_concurrency"] for r in rows]      # x-axis: 1, 2, 4, 8, 16, 32
thr = [r["output_throughput"] for r in rows]    # generated tokens per second, whole run
p50 = [r["median_tpot_ms"] for r in rows]       # median time per output token, ms
p99 = [r["p99_tpot_ms"] for r in rows]          # worst-1% time per output token, ms

fig, ax1 = plt.subplots(); ax2 = ax1.twinx()    # two y-axes: throughput (left) and latency (right) share the concurrency x-axis
ax1.plot(c, thr, "o-", label="output tok/s"); ax1.set_xlabel("concurrency"); ax1.set_ylabel("tok/s"); ax1.set_xscale("log", base=2)   # log2 x: the sweep doubles
ax2.plot(c, p50, "s--", color="C1", label="TPOT p50 (ms)"); ax2.plot(c, p99, "^--", color="C3", label="TPOT p99 (ms)"); ax2.set_ylabel("ms/token")
fig.legend(loc="upper left"); plt.title("Qwen2.5-1.5B fp16, RTX 2060, vLLM"); plt.savefig("throughput_vs_latency.png", dpi=150)   # the README figure
```

**Expected shape:** throughput rises almost linearly from C=1 to C≈8 (decode is memory-bound; extra sequences are nearly free — Module 2 step 5), then bends over as the card runs out of KV blocks or hits its compute ridge; TPOT p50 stays flat until the bend and then climbs; p99 climbs earlier than p50 and much more steeply. The **knee** — where the throughput gain per added client stops being worth the p99 cost — is the number you cite. On this card it's probably C≈8–16 for this model. Also plot TTFT p99: it grows with C because new requests' prefills queue behind the running batch, and it's the metric that goes wrong first.

| Symptom | Reason |
|---|---|
| Throughput flat from C=1 | You're launch-bound (eager mode). Remove `--enforce-eager` |
| Throughput drops at high C, p99 explodes | KV exhausted → preemptions. Check `vllm:num_preemptions_total` in `/metrics`; lower `--max-num-seqs` or `--max-model-len` |
| TTFT p99 ≫ p50 at C=1 | Prefix-cache misses on random prompts plus first-request warm-up; discard the first 10 requests (`--num-prompts` ≥ 200 does this by averaging) |
| p50 ITL < p50 TPOT | Normal: TPOT includes the first token's share; ITL doesn't |

### Exercise 2 — prefix caching: TTFT hit vs miss with a 1k-token system prompt

**Look first.** Reading-path row 7: open `v1/core/kv_cache_utils.py` and read `hash_block_tokens` (the hash of one block = hash(parent block's hash, this block's 16 token ids)) and `hash_request_tokens` (walks the prompt in full 16-token blocks, chaining). Then in `v1/core/kv_cache_manager.py` find `get_computed_blocks` — the lookup that turns "these hashes already exist in the pool" into "skip prefill for these tokens". Two things to notice: only *full* blocks are hashed (a 1,050-token prompt hashes 65 blocks and recomputes the last 10 tokens), and the chain means a difference in block 0 changes every later hash. Both facts are used in the script and the failure table below. Also confirm the metric names on your version: `curl -s localhost:8000/metrics | grep prefix_cache`.

Prefix caching is on by default in V1 (`--enable-prefix-caching`; use `--no-enable-prefix-caching` for the control run). vLLM hashes each full 16-token block of the prompt, chained with the previous block's hash; a new request whose leading blocks hash to blocks already in the pool skips their prefill and only computes the tail.

The measurement: a fixed ~1k-token system prompt, followed by a short unique user turn. Send the same system prompt twice with different suffixes; the first is a miss, the second a hit. Measure TTFT client-side as time-to-first-streamed-chunk.

```python
import time, requests, json, statistics

URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# A fixed ~1k-token "system prompt": one sentence repeated, cut to 5,000 characters. Check the real token count with
# len(tokenizer(system).input_ids) — it should be ~1,000–1,100 Qwen tokens, i.e. ~65 full 16-token blocks.
system = ("You are a meticulous assistant. " * 120)[:5000]

def ttft(prompt):
    """Send one streaming request and return seconds until the first token chunk arrives."""
    t0 = time.perf_counter()
    # stream=True in the body makes the server send SSE chunks; stream=True in requests.post makes the client read them as they arrive.
    with requests.post(URL, json={"model": MODEL, "prompt": prompt, "max_tokens": 16, "stream": True}, stream=True) as r:
        for line in r.iter_lines():                                   # one SSE line at a time
            if line.startswith(b"data:") and b"[DONE]" not in line:   # the first real data line is the first token
                return time.perf_counter() - t0                       # TTFT as the client sees it (network + queue + prefill)
    raise RuntimeError("no tokens")

misses, hits = [], []
for i in range(10):
    # A unique user turn so the two requests are not byte-identical (identical requests would both hit after the first).
    unique = f"\nUser: question number {i * 7919}: summarise the rules.\nAssistant:"
    misses.append(ttft(f"[run {i}] {system}{unique}"))   # difference in the FIRST block → every later block's chained hash differs → full miss
    hits.append(ttft(f"{system}{unique}"))                # identical first 1k tokens → all full blocks hit; only the suffix is prefilled
print(f"miss TTFT p50 = {statistics.median(misses)*1e3:.0f} ms   hit TTFT p50 = {statistics.median(hits)*1e3:.0f} ms")
```

Why the miss tag goes at the *front*: block hashes are chained (block n's hash includes block n−1's), so a difference in block 0 invalidates everything after it. A tag placed *after* the system prompt would leave the first ~64 blocks hitting and only miss the tail — you'd measure a hit, not a miss. To get a hit, the first 1k tokens must be byte-identical.

**Expected.** Prefill of ~1,050 tokens on Qwen2.5-1.5B: FLOPs = 2 × 1.54 B × 1,050 ≈ 3.2 TFLOP → at ~15 TFLOPS sustained ≈ **210 ms** for the miss. On a hit only the last partial block plus the ~25-token suffix is computed → **20–40 ms**, mostly fixed overhead. A 5–10× TTFT reduction. In `/metrics`, `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` should show the hit ratio climbing to ~50% (half your requests are hits). Run the control with `--no-enable-prefix-caching` and confirm both columns become ~210 ms.

| Symptom | Reason |
|---|---|
| Hit ≈ miss | Caching disabled, or your "identical" prefixes differ (a timestamp, a request id, a trailing space). Diff the two strings' first 1,000 tokens |
| Hit is ~100 ms, not ~30 | Your unique suffix is long, or the system prompt is not a multiple of 16 tokens and the last partial block is being recomputed plus the whole suffix — expected; only full blocks are cached |
| Second run of the whole script is all hits | The pool still holds last run's blocks. Restart the server or change the system prompt to reset |

This experiment is the seed of Project 3: the ~180 ms you saved is exactly what prefix-affinity routing buys when the hit is on a *different machine* than the one a random router would have chosen.

### Exercise 3 — mini-vLLM

**Look first.** Reading-path rows 6 and 7 again, this time with a notebook: write down, in your own words, the fields `Scheduler` keeps (`running`, `waiting`, the token budget, the KV manager) and the three methods `BlockPool` exposes (get free blocks, cache full blocks by hash, evict/free). Those two lists are the interfaces of Project 2 M2 and M3.

**This is Project 2.** Milestone 2 (continuous batching) and 3 (block-based KV manager) are the two mechanisms this module is about; see `04-project-specs.md` and `07-p2-walkthrough.md`.

---

## Module 5 — Quantization

**What you are actually learning here.** Quantization is a lever with two separate targets: fewer *bytes* (weight-only: GPTQ/AWQ/K-quants — helps the memory-bound decode phase and VRAM) and fewer *FLOPs* (weight-and-activation INT8/FP8 — helps the compute-bound prefill phase, needs hardware that does low-precision math and calibration to handle activation outliers). The interview question is *"you have model X on GPU Y with latency target Z — what precision do you run and why?"*, and the trap is claiming int4 helps prefill.

### Terms this module uses

- **Quantization.** Storing a number in fewer bits. An fp16 weight is 16 bits; an int4 weight is 4 bits plus a share of a scale factor. Analogy: rounding every price in a shop to the nearest 25 cents — you lose a little precision, but the price list is a quarter the size.
- **Scale / zero-point / group size.** To turn fp16 weights into int4 you pick, for each *group* of weights (typically 128 in a row), a scale (the fp16 value one int step represents) and optionally a zero-point (which int maps to 0.0). `weight ≈ (int4 − zero) × scale`. Group size 128 means one scale per 128 weights → 16 bits ÷ 128 = 0.125 extra bits per weight.
- **Bits per weight (bpw).** Average storage per weight including scales and zero-points. "4-bit" AWQ with group size 128 is ≈ 4.25–4.5 bpw. llama.cpp's Q4_K_M is ≈ 4.8 bpw, Q5_K_M ≈ 5.7, Q8_0 ≈ 8.5.
- **Weight-only quantization.** Only the weights are stored small; before each matmul a kernel converts a tile back to fp16 (**dequantization**) and the multiply runs in fp16. Cuts bytes read, not FLOPs. GPTQ, AWQ, bitsandbytes NF4, llama.cpp K-quants.
- **Weight-and-activation quantization (W8A8, FP8).** Both the weights *and* the activations (the `x` in `x @ W`) are quantized, so the multiply itself runs on int8/fp8 tensor cores at 2× fp16 speed. Cuts FLOPs. Needs Ampere+ (int8) or Hopper+ (fp8).
- **Activations.** The intermediate tensors flowing between layers (`x`, the attention output, the MLP hidden). Unlike weights they change with every input, so they must be quantized on the fly.
- **Activation outliers.** A few channels of the activation vector are 10–100× larger than the rest in big LLMs (the LLM.int8 paper's observation). One scale for the whole vector would crush the small values to zero; this is why W8A8 needs per-channel/per-token scales or calibration.
- **Calibration.** Running a few hundred representative inputs through the model to measure activation ranges (for W8A8) or to decide which weights matter most (AWQ: protect the weights that multiply large activations; GPTQ: correct rounding error layer by layer using second-order information).
- **RTN (round-to-nearest).** Naive quantization with no calibration: just round. Visibly worse at 4 bits; fine at 8.
- **GPTQ / AWQ.** Two calibrated int4 weight-only methods for GPUs. They produce checkpoints with `qweight` (packed int4), `scales`, and `qzeros` tensors instead of `weight`. Interchangeable for your purposes; they differ in kernels available on Turing.
- **Marlin / exllama kernels.** Fast fused dequantize-and-matmul GPU kernels. Marlin needs sm_80; exllama runs on sm_75. Which kernel you get decides whether int4 is faster than fp16 on the 2060.
- **K-quants (llama.cpp: Q4_K_M, Q5_K_M, Q8_0).** llama.cpp's quantization formats; "K" = the k-quant scheme with per-block super-scales, M = medium (some layers kept at higher precision). Used in Project 4.
- **Perplexity.** `exp(mean negative log-likelihood)` on a test text; lower is better. The standard quality number for comparing quantized checkpoints (Project 4 measures it with llama.cpp's `llama-perplexity`).
- **Top-1 agreement.** A finer quality signal than 20 pass/fail prompts: for the same prompt at temperature 0, what fraction of the first N generated tokens are identical between fp16 and int4.
- **KV-cache quantization.** Storing the cached K and V in fp8/int8 instead of fp16. An independent lever: halves KV bytes (more sequences fit; less KV read per step at long context), changes nothing about weights.

### Exercise 1 — fp16 vs AWQ-int4 Qwen2.5-1.5B in vLLM

**Look first — what a quantized checkpoint is, and how vLLM decides what to do with it (20 min).**

```python
# 1. The checkpoint tells vLLM how it was quantized. Open the AWQ model's config.json on the Hub or via AutoConfig:
from transformers import AutoConfig
c = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct-AWQ")
print(c.quantization_config)    # {'bits': 4, 'group_size': 128, 'quant_method': 'awq', 'version': 'gemm', 'zero_point': True, ...}
                                # bits=4, group_size=128 → 4 + 16/128 (scale) + 4/128 (zero) ≈ 4.16 bpw for the quantized linears

# 2. The tensors inside. A quantized Linear does not have a `weight`; it has qweight / qzeros / scales.
from huggingface_hub import hf_hub_download
from safetensors import safe_open
path = hf_hub_download("Qwen/Qwen2.5-1.5B-Instruct-AWQ", "model.safetensors")   # the single-shard checkpoint (~1.1 GB)
with safe_open(path, "pt") as f:
    for k in f.keys():
        if "layers.0." in k:                       # block 0 only
            print(k, f.get_slice(k).get_shape(), f.get_slice(k).get_dtype())
# Notice: mlp.down_proj.qweight is [8960, 192] int32 — 192 int32 = 1536 int4 values packed 8 per int32, so it is the [8960, 1536] weight in 4 bits;
# mlp.down_proj.scales is [70, 1536] fp16 — one scale per (group of 128 input rows) × output column: 8960 / 128 = 70 groups;
# embed_tokens.weight and every norm weight are still fp16 — the reason AWQ is not a clean 4× smaller.
```

Then vLLM's side — the classes that read that `quantization_config` and pick a kernel. In the vLLM repo, `vllm/model_executor/layers/quantization/`:

- `awq.py` — `AWQConfig` (`from_config` reads `bits`, `group_size`, `zero_point` from the checkpoint; `get_min_capability()` returns 75 — Turing is allowed) and `AWQLinearMethod` (`create_weights` allocates `qweight`/`qzeros`/`scales` with exactly the shapes you just printed; `apply` calls the dequantize-then-matmul op — this is the "fallback kernel" on the 2060).
- `awq_marlin.py` — `AWQMarlinConfig`: `is_awq_marlin_compatible` checks the GPU's compute capability against 80. On the 2060 this returns False and vLLM logs that it is falling back to plain AWQ. Read that check so the fallback in the expected-results table is not a surprise.
- `gptq.py` — `GPTQConfig` / `GPTQLinearMethod` (the exllama-based kernel, sm_75-capable), and `gptq_marlin.py` for Ampere+. Same structure.
- `__init__.py` — `get_quantization_config(name)`: the registry that maps the string `"awq"` / `"gptq"` from `--quantization` (or the checkpoint) to one of those classes.

Notice that every one of these classes has the same three-method shape — `create_weights`, `process_weights_after_loading`, `apply` — replacing `nn.Linear` for the quantized layers and nothing else. Quantization in a serving engine is a per-Linear plug-in, not a different model.

Two servers, same flags apart from the checkpoint. Run them one at a time (the card can't hold both). Flags are the Module 4 set (`--dtype half`: fp16 compute — the dequantized weights and all activations; `--max-model-len 4096`, `--gpu-memory-utilization 0.85`, `--max-num-seqs 16`, `--port 8000` as before), plus for server B `--quantization awq`: tells vLLM which `QuantizationConfig` class to use (it can also infer it from the checkpoint; passing it explicitly makes a mismatch fail loudly instead of silently loading garbage).

```bash
# A: fp16 (no --enforce-eager this time: you want CUDA graphs on for both, so the comparison is bytes vs bytes, not launch overhead)
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --max-num-seqs 16 --port 8000

# B: AWQ int4 (fall back to Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 with --quantization gptq if the AWQ kernel refuses sm_75)
vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ --quantization awq --dtype half --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --max-num-seqs 16 --port 8000
```

Three measurements per server:

1. **VRAM.** Read the startup log: "Loading weights took … GB" (fp16 ≈ 3.1 GB; AWQ ≈ 1.1–1.2 GB — AWQ keeps embeddings, norms and often the LM head in fp16, so it's not a clean 4×) and "# GPU blocks: N". Multiply N × 16 × 28,672 bytes for KV capacity. Also `nvidia-smi --query-gpu=memory.used --format=csv` while idle.
2. **Speed.** The Module 4 sweep at C = 1, 4, 16, plus one prefill-heavy run: `--random-input-len 2048 --random-output-len 8 --max-concurrency 1` (2,048-token prompts, 8 output tokens, one at a time: TTFT ≈ prefill time). Record decode tok/s at C=1 and TTFT at 2k.
3. **Quality on 20 fixed prompts.** Write `prompts.jsonl` (20 short factual/instruction prompts with a reference answer you can check by exact or contains-match — e.g. "What is 17 × 23?" → 391, "Name the chemical symbol for gold." → Au, "Write a Python one-liner that reverses a string." → `[::-1]`), generate at temperature 0 from both, score with a 20-line script, and also log the fp16-vs-int4 top-1 token agreement over the first 32 generated tokens (a finer-grained signal than 20 pass/fail bits).

**Expected.**

| | fp16 | AWQ int4 | why |
|---|---|---|---|
| weights | ~3.1 GB | ~1.15 GB | 4-bit for the linears; fp16 for embeddings/norm |
| C=1 decode | 70–90 tok/s | 90–130 tok/s | bytes/step drop ~2.7×, but dequantization kernels and the fixed launch overhead cap the gain well below 2.7× on a small model. On the 2060 the AWQ fallback kernel (not Marlin) is not very fast; a gain of 1.2–1.6× is a realistic outcome |
| C=16 decode | higher total tok/s | similar or slightly lower | at batch 16 you're near this card's ridge; dequant adds FLOPs |
| TTFT @ 2k | ~400 ms | ~400–500 ms | prefill is compute-bound; int4 doesn't reduce FLOPs and adds dequant work |
| 20-prompt score | baseline | −0 to −2 prompts | AWQ on a 1.5B model is "good int4": small, visible loss on arithmetic/code more than on facts |

If AWQ decode is *slower* than fp16 at C=1, that's the Turing fallback kernel; note it as a finding — "int4 helps only when the dequant kernel is faster than the bytes it saves; on sm_75 without Marlin it wasn't" is a very good README sentence and an even better interview sentence.

### Exercise 2 — llama.cpp K-quant sweep

**Look first.** `llama-quantize --help` lists every format with its bits-per-weight and the perplexity delta measured on Llama-2-7B — read that table before choosing which formats to sweep. Then `llama-cli -m model.gguf -p "hi" -n 1 --verbose` (or `llama-server`'s startup log): it prints, per tensor type, how many tensors are in which format — you will see that even "Q4_K_M" keeps some tensors (attention v, output) at Q6_K. The quality/size trade-off is decided per tensor, not per model.

**This is Project 4** (milestones 1–4; `09-p4-walkthrough.md`). The table you produce there — Q4_K_M / Q5_K_M / Q8_0 / fp16 × (prefill tok/s, decode tok/s, VRAM, quality) on GPU and on CPU — is this exercise. Nothing to add.

### Exercise 3 — why does weight-only quantization speed up decode but barely help prefill?

**Look first.** The mechanism in the essay below is visible in one kernel. In the vLLM repo open `csrc/quantization/awq/gemm_kernels.cu` and find `dequantize_s4_to_fp16x2` (in `dequantize.cuh`) being called inside the GEMM kernel: each int4 tile is unpacked to fp16 *in registers* and then multiplied with fp16 activations by ordinary fp16 tensor-core instructions. Every FLOP of the fp16 matmul still happens; the only thing that got smaller is the load from memory. That single fact is the whole answer.

*(Write this out in your own words after reading it once; the interview version is three sentences.)*

Every decode step reads the entire weight set once and does about two FLOPs per weight per sequence. At batch 1 that's an arithmetic intensity of ~1 FLOP/byte for fp16 weights — two orders of magnitude below any GPU's ridge point — so the step's time is set by how many bytes cross the memory bus, not by how many FLOPs execute. Weight-only quantization attacks exactly that quantity: int4 weights are a quarter of the bytes, so the memory floor for a step drops 4×, and if nothing else intervenes decode gets up to 4× faster. What intervenes: the weights still have to be multiplied in fp16 (the tensor cores don't do int4 × fp16), so a kernel dequantizes each tile on the fly — extra instructions, which are free while you're memory-bound but not once the step gets short enough to be launch-bound or once the dequant work itself is comparable to the matmul. On a 0.5B–1.5B model on a small card, launch overhead is a large share of the step, so the realized gain is 1.2–2×, not 4×. On a 70B model on an H100, where the step is a long, clean stream of weight reads, the realized gain is close to the byte ratio.

Prefill is the opposite regime. A 512-token prompt reuses each weight 512 times; its intensity is in the hundreds of FLOP/byte, above the ridge, so the time is set by FLOPs, not bytes. Weight-only quantization does not remove a single FLOP: the multiplies still happen in fp16 after dequantization. It *adds* the dequant instructions, so prefill gets slightly slower or, at best, unchanged. The same goes for any compute-bound situation — high-batch decode, long-context attention — which is why int4 serving at high concurrency gains little.

To speed up the compute-bound phase you have to make the *multiplies* cheaper, which means quantizing activations too, so that the tensor cores can run in INT8 or FP8: half the bits per multiply, and on Ampere/Hopper roughly 2× the peak FLOPs of fp16. That requires either per-token dynamic activation scaling or calibration, because activations have outlier channels (LLM.int8's observation) that a single scale would crush. FP8 on Hopper is the practical version: near-lossless, 2× compute, half the weight bytes — it helps both phases. KV-cache quantization (fp8/int8 KV) is a third, independent lever: it reduces the bytes read per step *at long context* and doubles the number of sequences that fit, but does nothing for weights.

Decision rule to say out loud: memory-bound and latency-sensitive (chat at low batch, edge, one big model on one card) → weight-only int4/int8 (AWQ/GPTQ, K-quants). Compute-bound or throughput-oriented (batch serving, long prompts, high concurrency) → FP8/INT8 weight+activation on hardware that supports it; weight-only buys nothing there. Long context, batch-limited by KV → quantize the KV cache. Quality: int8 weight-only ≈ lossless; good int4 (AWQ/GPTQ with calibration, Q4_K_M) ≈ small loss, worst on math/code and on small models (< 3B); naive round-to-nearest int4 without calibration is visibly worse.

---

## Module 6 — Speculative decoding

**What you are actually learning here.** Decode at low batch is memory-bound (Module 2): a forward pass over 1 token and a forward pass over 5 tokens cost almost the same time, because both read every weight once. Speculative decoding exploits that slack: a cheap draft model guesses γ tokens autoregressively, the expensive target model scores all γ+1 positions in *one* pass, and a rejection-sampling rule accepts a prefix of the guesses such that the output distribution is *exactly* the target's. The interview question is *"explain speculative decoding, and why is the output distribution unchanged?"* — the three-sentence exactness argument at the end of exercise 1 — with the follow-up *"when does it stop helping?"* — exercise 2.

### Terms this module uses

- **A probability distribution over the vocabulary.** After a forward pass, softmax of the last position's logits gives a list of 151,936 numbers, each ≥ 0, adding up to 1. Entry `i` is the model's probability that the next token is token `i`. Write it `p(x)` for "the probability of token `x`". A greedy model's distribution is *one-hot*: a 1 at the argmax and 0 everywhere else.
- **Sampling from a distribution.** Choosing a token at random so that token `x` comes up with probability `p(x)`. `torch.multinomial(p, 1)` does it: imagine a wheel of fortune where each token's slice has area `p(x)`; spin once. Sampling 2,000 times and counting gives a histogram that approximates `p`.
- **Temperature.** Divide the logits by `T` before softmax. `T → 0` makes the distribution one-hot (greedy); `T = 1` is the model's raw distribution; `T > 1` flattens it.
- **Draft model / target model.** The small, fast model that guesses (`q`, its distribution) and the big, slow model whose output you actually want (`p`). Here: Qwen2.5-0.5B drafts for Qwen2.5-1.5B.
- **γ (gamma).** How many tokens the draft guesses per round before the target checks them.
- **α (alpha, acceptance rate).** The fraction of drafted tokens the target accepts. It measures how well `q` agrees with `p`.
- **Rejection sampling, in plain words.** You want to draw from `p` but you only have a cheap way to draw from `q`. Draw a candidate `x` from `q`. Now ask: *was `q` more eager to propose `x` than `p` would have been?* If `q(x) ≤ p(x)` — the draft was no more eager than the target — keep `x`. If `q(x) > p(x)` — the draft over-proposes `x` — keep it only a `p(x)/q(x)` fraction of the time, and otherwise throw it away. Tokens that `q` under-proposes come up too rarely this way, so when you throw a draft away you replace it by drawing from exactly the leftover: the part of `p` that `q` did not cover, `max(0, p − q)` renormalized. Kept tokens plus replacements together are distributed exactly as `p`. That is the whole trick; the rule below is this paragraph in symbols.
- **Residual distribution.** `max(0, p − q)` divided by its sum — "the mass `q` missed", used for the replacement token.
- **Bonus token.** If all γ drafts are accepted, the target's pass also produced a distribution for position γ+1 for free; sample one more token from it.
- **Verification pass.** The one target forward over the last committed token plus the γ drafts, which yields γ+1 distributions at once — the thing that costs "one decode step" regardless of γ while memory-bound.
- **`DynamicCache.crop(n)`.** Drop every cached position after `n`. Needed because after a rejection the caches contain K/V for draft tokens that were never committed.
- **Truncated geometric distribution.** With per-token acceptance α, "how many in a row are accepted before the first rejection, capped at γ" — the distribution behind the expected-tokens formula in exercise 2.
- **Total-variation distance.** `0.5 × Σ|f₁(x) − f₂(x)|` between two histograms: 0 if identical, 1 if disjoint. The check that spec-decoding's output histogram matches plain sampling's.

### Look first — the real implementations, read before writing the toy (30 min)

**Hugging Face** implements this as "assisted generation". Two files:

1. `transformers/generation/candidate_generator.py` — `AssistedCandidateGenerator`. Read `get_candidates(input_ids)`: it runs the *assistant* (draft) model's own `generate` for `num_assistant_tokens` steps (their γ; note it is adaptive — `update_candidate_strategy` raises it after a fully accepted round and lowers it after a rejection) and returns the candidate ids plus the draft's logits. Note also how it keeps the draft's `past_key_values` and crops it (`_crop_past_key_values`) after each round — the same bookkeeping your toy does with `crop`.
2. `transformers/generation/utils.py` — the function `_speculative_sampling(candidate_input_ids, candidate_logits, candidate_length, new_logits, is_done_candidate)`. This is the acceptance rule in ~30 lines. Find these lines and match them to the rule below: `q = candidate_logits.softmax(...)` (draft probs), `p = new_logits.softmax(...)` (target probs), `probability_ratio = p_i / q_i`, `r_i = torch.rand_like(probability_ratio)`, `is_accepted = r_i <= probability_ratio`, `n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()` (the accepted prefix length), and then the residual: `p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)`, `p_prime.div_(p_prime.sum())`, `t = torch.multinomial(p_prime, ...)`. Also see the greedy branch above it in `_assisted_decoding`: when `do_sample=False` it just compares argmaxes.

   ```python
   import inspect, transformers.generation.utils as u
   print(inspect.getsource(u._speculative_sampling))     # the acceptance rule, in HF's words
   ```

**vLLM** implements it inside the engine, as a batched Triton kernel: `vllm/v1/sample/rejection_sampler.py`. Read top to bottom:

- `RejectionSampler.forward` — takes the draft token ids, the draft probs, and the target logits for the whole batch, and dispatches to `rejection_sample`.
- `rejection_greedy_sample_kernel` — the temperature-0 case: accept while `draft_token_id == target_argmax_id`, stop at the first mismatch and emit the target's argmax there (your correctness check 1).
- `rejection_random_sample_kernel` — **the acceptance line.** Search the file for `uniform_prob`. The kernel loads `draft_prob = q[pos, draft_token]`, `target_prob = p[pos, draft_token]`, and a uniform random number, and accepts iff `target_prob / draft_prob >= uniform_prob` (guarded by `draft_prob > 0`). That comparison is `rand < min(1, p/q)` in kernel form.
- `sample_recovered_tokens_kernel` (and the `compute_probs` / `recovered` helpers around it) — the residual: `max(0, p − q)` renormalized, sampled once for the rejected position. vLLM calls the replacement the "recovered token" and the all-accepted extra the "bonus token".

Notice how vLLM's version does all sequences in the batch in one launch and never touches Python inside the loop — the reason the "honest caveat" below says a real speedup needs the engine, not a wrapper.

### Exercise 1 — toy speculative decoding: Qwen2.5-0.5B drafting for Qwen2.5-1.5B

Both models fit on the 2060 in fp16: 0.99 + 3.09 = 4.08 GB plus two small KV caches. They share a tokenizer, which is a hard requirement (the draft's token ids must be the target's).

**The acceptance rule.** For each proposed token `x` drawn from the draft distribution `q`, with target distribution `p` at the same position:

```
accept x with probability min(1, p(x) / q(x))
on reject: stop, and sample the replacement from  p'(y) = max(0, p(y) − q(y)) / Σ_y max(0, p(y) − q(y))
if all γ accepted: sample one bonus token from p at position γ+1 (the target already computed it)
```

Intuition for `p/q`: if the draft is *more* eager than the target for `x` (`q > p`), thin it out by accepting only a `p/q` fraction of the time; if the draft is *less* eager (`q ≤ p`), always keep it. The residual `max(0, p − q)` is exactly the mass the thinning removed from where `q` over-proposed, redistributed to where `q` under-proposed. Rejection ends the block because every later draft token was conditioned on the rejected one.

**Implementation.** Uses HF's `DynamicCache` and its `crop()` so that neither model recomputes accepted tokens; the invariant is *"each cache holds exactly the committed sequence minus its last token"* and `step_logits` feeds whatever the cache doesn't have yet.

```python
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

dev = "cuda"
tok    = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")          # one tokenizer for both models (they share it — verified in the symptom table)
draft  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16).to(dev).eval()   # q: the cheap guesser, ~1 GB
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16).to(dev).eval()   # p: the model whose output we want, ~3.1 GB

def dist(logits_row, temperature):
    """One position's logits [vocab] → a probability distribution [vocab] (fp32)."""
    if temperature == 0:                                   # greedy = one-hot distribution: 1.0 at the argmax, 0 elsewhere
        return torch.nn.functional.one_hot(logits_row.argmax(), logits_row.numel()).float()
    return torch.softmax(logits_row.float() / temperature, dim=-1)   # temperature scaling, then softmax → sums to 1

@torch.no_grad()
def step_logits(model, cache, tokens):
    """Feed tokens[cache_len:] (what the cache hasn't seen), return logits [n_fed, vocab] in fp32."""
    start = cache.get_seq_length()                         # how many positions this model's cache already holds
    new = torch.tensor([tokens[start:]], device=dev)       # [1, n_fed]: only the positions the cache is missing
    out = model(input_ids=new, past_key_values=cache, use_cache=True)   # cache grows by n_fed inside the model (DynamicCache.update)
    return out.logits[0].float()                           # [n_fed, vocab]: one logits row per fed position

@torch.no_grad()
def generate_spec(prompt_ids, max_new, gamma=4, temperature=1.0):
    seq = list(prompt_ids)                                 # the committed sequence (prompt + accepted tokens), as Python ints
    d_cache, t_cache = DynamicCache(), DynamicCache()      # separate KV caches for draft and target — different models, different K/V
    stats = {"proposed": 0, "accepted": 0, "target_passes": 0}
    while len(seq) - len(prompt_ids) < max_new:
        n = len(seq)                                       # committed length at the start of this round; caches hold n-1 (invariant)
        # --- draft phase: γ sequential cheap steps ---
        d_tokens, q = [], []                               # drafted token ids, and the draft distribution each was sampled from
        for _ in range(gamma):
            logits = step_logits(draft, d_cache, seq + d_tokens)   # feed committed + drafts-so-far that the draft cache lacks
            qi = dist(logits[-1], temperature)             # [vocab] draft distribution at the next position
            d_tokens.append(torch.multinomial(qi, 1).item()); q.append(qi)   # sample one draft token from q; remember q for the ratio
        # --- target phase: ONE pass over [last committed] + γ drafts → γ+1 distributions ---
        logits = step_logits(target, t_cache, seq + d_tokens)      # target cache holds n-1, so it is fed 1 + γ tokens → [γ+1, vocab] logits
        p = torch.stack([dist(row, temperature) for row in logits[-(gamma + 1):]])   # [γ+1, vocab]: p at each draft position, plus the bonus position
        stats["target_passes"] += 1; stats["proposed"] += gamma
        # --- verify: accept a prefix, then exactly one more token ---
        k = 0                                              # number of drafts accepted this round
        for i, x in enumerate(d_tokens):
            if torch.rand(()) < min(1.0, (p[i, x] / q[i][x]).item()):   # accept with probability min(1, p(x)/q(x)) — the acceptance rule
                seq.append(x); k += 1
            else:
                resid = torch.clamp(p[i] - q[i], min=0)    # [vocab] residual: the mass q under-proposed, max(0, p − q)
                resid = resid / resid.sum() if resid.sum() > 1e-9 else p[i]   # renormalize (fall back to p if p == q exactly, e.g. both one-hot)
                seq.append(torch.multinomial(resid, 1).item())   # replacement token from the residual; ends the round
                break
        else:                                              # all γ accepted → free bonus token from p_γ
            seq.append(torch.multinomial(p[gamma], 1).item())
        stats["accepted"] += k
        # --- restore the invariant: caches hold committed-minus-last = n + k tokens ---
        t_cache.crop(n + k)                                # target cache had n+γ; drop the rejected drafts and the last committed token
        d_cache.crop(min(d_cache.get_seq_length(), n + k)) # draft cache had n+γ-1; same target length (may already be shorter when k = γ)
    return seq[:len(prompt_ids) + max_new], stats          # trim the possible overshoot from a bonus token

@torch.no_grad()
def generate_plain(prompt_ids, max_new, temperature=1.0):
    """Baseline: target model only, one token per forward pass, same sampling."""
    seq = list(prompt_ids); cache = DynamicCache()
    while len(seq) - len(prompt_ids) < max_new:
        logits = step_logits(target, cache, seq)           # feeds the whole prompt on the first call, one token per call after
        seq.append(torch.multinomial(dist(logits[-1], temperature), 1).item())
    return seq

def timed(fn, *a, **kw):
    """Run fn and return (result, seconds), with GPU syncs so the timer includes all queued kernels."""
    torch.cuda.synchronize(); t = time.perf_counter(); r = fn(*a, **kw); torch.cuda.synchronize()
    return r, time.perf_counter() - t

prompt = tok("Write a short explanation of how a hash table works.\n", return_tensors="pt").input_ids[0].tolist()   # list of ints
for fn in (generate_plain, generate_spec): timed(fn, prompt, 16)     # warm-up both paths (first CUDA calls, allocator growth)

torch.manual_seed(0); (plain, t_plain) = timed(generate_plain, prompt, 200)          # same seed for both so RNG state is comparable
torch.manual_seed(0); ((spec, st), t_spec) = timed(generate_spec, prompt, 200, gamma=4)
print(f"plain : {200/t_plain:6.1f} tok/s")
print(f"spec  : {200/t_spec:6.1f} tok/s   acceptance α = {st['accepted']/st['proposed']:.2f}   "
      f"tokens/target-pass = {200/st['target_passes']:.2f}   speedup = {t_plain/t_spec:.2f}x")
print(tok.decode(spec[len(prompt):]))                      # the generated text, so you can see it is coherent
```

**Why `crop(n + k)` is right.** Before the loop iteration the caches hold `n − 1` tokens (everything committed except the last). The draft phase leaves the draft cache at `n + γ − 1`; the target phase leaves the target cache at `n + γ`. After accepting `k` drafts and appending one more token, the committed length is `n + k + 1`, so "committed minus last" is `n + k`: crop both there. When `k = γ` the draft cache is one token short (`n + γ − 1`), which is fine — `step_logits` feeds the missing token next round.

**Two correctness checks before you trust any speed number:**

1. *Greedy determinism.* With `temperature=0` both `dist`s are one-hot, so acceptance is "accept iff draft argmax == target argmax" and the residual is the target's argmax. Output must equal `generate_plain(..., temperature=0)` token for token. If it doesn't, the cache bookkeeping is wrong — print `t_cache.get_seq_length()` vs `len(seq) - 1` at the top of every iteration; they must match.
2. *Distribution match at T=1.* Take a 1-token prompt continuation, sample the first generated token 2,000 times with `generate_spec(prompt, 1)` and 2,000 times with `generate_plain(prompt, 1)`, and compare histograms: total-variation distance `0.5 × Σ|f_spec − f_plain|` should be ~0.03–0.06 (sampling noise at n=2,000 over a peaked distribution), and the top-10 tokens should agree in rank. If TV is > 0.15 you have a bug in the residual step.

**Expected numbers, and the honest caveat.** Draft and target are *both* small enough that on this card, in HF eager mode, per-step time is dominated by Python + kernel-launch overhead rather than by weight bytes: the 1.5B target's step is ~25–35 ms in HF eager (vs a 9 ms bandwidth floor) and the 0.5B draft's is ~15–25 ms (vs 3 ms). So the draft costs 50–80% of a target step, not the 10% assumed in exercise 2, and the speedup you measure will be **~1.0–1.4×**, sometimes < 1. That is the correct result, not a failure: measure `t_draft_step` and `t_target_step` separately first, then predict speedup as `E[tokens per pass] / (1 + γ × t_draft/t_target)` and check the prediction against the measurement. Expected α for this pair on English prose at T=1 is ~0.6–0.75 (higher for greedy, ~0.8). To see a real 2× you'd need the target step to be genuinely bandwidth-bound (a 7B+ target, or CUDA graphs so overhead disappears), which is why vLLM's speculative decoding is implemented inside the engine with graph-captured steps rather than as a wrapper like this one. Say all of that in the README.

| Symptom | Reason |
|---|---|
| α ≈ 0 | Tokenizers differ (using the 0.5B tokenizer for the 1.5B ids or vice versa) — they're the same for Qwen2.5 sizes, but verify `tok.vocab_size` equals both `config.vocab_size` |
| `crop` AttributeError | Old `transformers`; needs ≥ 4.41 or so. Upgrade |
| Output degenerates after ~50 tokens | Position ids drift: you fed tokens the cache already had. The `tokens[start:]` slice must use the cache length *of that model* |
| speedup 0.4× | You're recomputing the prefix every pass (cache not passed back) or `gamma` too large for α |

**The three-sentence exactness argument** (what you say when asked "why is the output distribution identical?"): For each position, a token `x` is emitted by acceptance with probability `q(x) · min(1, p(x)/q(x)) = min(q(x), p(x))`, and by the rejection path with probability `(1 − Σ_y min(p,q)(y)) · max(0, p(x)−q(x)) / Σ_y max(0, p−q)(y)`. Since `Σ_y max(0, p−q)(y) = 1 − Σ_y min(p,q)(y)`, the rejection path contributes exactly `max(0, p(x) − q(x))`, and `min(p,q)(x) + max(0, p(x) − q(x)) = p(x)`. So every emitted token is marginally distributed as `p`, regardless of how good or bad `q` is — `q` only affects *how many* tokens per pass you get, never *which* distribution they come from.

### Exercise 2 — expected speedup at α = 0.8, γ = 4, draft cost 10%; and the break-even α

**Look first.** vLLM keeps the running acceptance statistics you are about to model: `vllm/v1/spec_decode/metrics.py` (`SpecDecodingStats` / `SpecDecodingLogging` — `num_drafts`, `num_draft_tokens`, `num_accepted_tokens`, and the per-position acceptance counts). Start a server with `--speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}'` (n-gram drafting needs no second model, so it runs on the 2060), send a few requests, and read the "SpecDecoding metrics" log line: it prints exactly the α and "mean acceptance length" (= E[tokens] per pass) this exercise computes by hand.

**Expected tokens per target pass.** With acceptance probability α per draft token (the paper's simplifying assumption that α is the same at every position), the number of accepted tokens is a truncated geometric: P(k accepted) = α^k (1−α) for k < γ, and α^γ for k = γ. You always get one extra (the residual or the bonus), so tokens per pass = 1 + E[k]:

```
E[tokens] = Σ_{k=0}^{γ} α^k = (1 − α^{γ+1}) / (1 − α)

α = 0.8, γ = 4:  0.8⁵ = 0.32768
                 (1 − 0.32768) / (1 − 0.8) = 0.67232 / 0.2 = 3.3616 tokens per target pass
```

(Check by summing: 1 + 0.8 + 0.64 + 0.512 + 0.4096 = 3.3616.)

**Cost per pass in target-step units.** One target pass over γ+1 tokens costs the same as one plain decode step (memory-bound: the weights are read once either way — this assumption fails once you're compute-bound, see below). The draft runs γ sequential steps at 0.1 each:

```
cost = 1 + γ × 0.1 = 1 + 0.4 = 1.4 target-step-equivalents
```

**Speedup** = tokens per unit cost, relative to 1 token per 1 target step:

```
speedup = 3.3616 / 1.4 = 2.40×
```

**Break-even α.** Speculation is a net loss when E[tokens] < cost:

```
(1 − α⁵) / (1 − α) = 1.4
1 + α + α² + α³ + α⁴ = 1.4
α + α² + α³ + α⁴ = 0.4
```

No closed form; bisect. α = 0.30 → 0.3 + 0.09 + 0.027 + 0.0081 = 0.4251 (too high). α = 0.28 → 0.28 + 0.0784 + 0.02195 + 0.00615 = 0.3865 (too low). α = 0.287 → 0.287 + 0.08237 + 0.02364 + 0.00678 = 0.3998. **Break-even α ≈ 0.287.** Below that, four draft steps cost more than the extra tokens they earn. That's a low bar — which is why speculative decoding "usually helps" for a well-aligned draft at low batch.

**What the interviewer asks next**, with numbers from the same formula:

- *Is γ = 4 optimal at α = 0.8?* E[tokens]/cost for γ = 1…8: 1.64, 2.03, 2.27, **2.40**, 2.47 (γ=5), 2.47 (γ=6), 2.44, 2.40. Flat top around γ = 5–6; beyond that the extra draft steps outrun the geometric tail. Real systems pick γ ≈ 3–5 and some adapt it to the running α.
- *Draft cost 50% instead of 10%?* cost = 1 + 4 × 0.5 = 3.0 → speedup 3.36/3.0 = 1.12×. This is your Module 6 exercise 1 result explained: an overhead-dominated draft is an expensive draft.
- *Why does it lose at high batch?* The "target pass over γ+1 tokens is free" assumption holds only while decode is memory-bound. At batch B the target step already does B tokens of work; verifying γ+1 per sequence multiplies the FLOPs by γ+1, and once B(γ+1) exceeds the ridge point (Module 2: ~300 on an H100) the pass costs proportionally more and the speedup collapses toward `E[tokens]/(γ+1)` × (1/1.4) < 1. That's why vLLM disables/deprioritises speculation under load and why it's a low-batch, latency-tool.

**Done when** you can give the three sentences above without notes, and reproduce the 3.36 / 1.4 / 2.4 chain on a whiteboard in a minute.

---

## Module 7 — Distributed inference and disaggregation

**What you are actually learning here.** The moment `weights + KV > one GPU`, you're choosing how to split: tensor parallelism slices every matmul across GPUs and needs an allreduce per layer (two per block), so it wants NVLink and lives inside a node, but it also aggregates the GPUs' bandwidth and so *reduces* per-token latency; pipeline parallelism assigns whole layers to stages, costs only one activation hand-off per stage boundary, crosses nodes fine, but idles stages (bubbles) unless there are many requests in flight. Disaggregation is a third split — prefill on one pool, decode on another — and it moves KV instead of weights. The interview questions are *"how many GPUs for Llama-3-70B and how do you split it?"* and *"what exactly moves in disaggregated serving and how big is it?"*

### Terms this module uses

- **Node.** One physical machine, typically holding 8 GPUs connected to each other by NVLink. Two nodes are connected by a network (InfiniBand or Ethernet), which is 10–20× slower than NVLink.
- **Rank.** The index of one GPU (one worker process) inside a distributed job: rank 0 … rank N−1. "TP rank 3" = the fourth of the GPUs sharing a tensor-parallel model.
- **Collective (collective communication).** An operation that *every* rank in a group calls at the same time and that moves data among all of them — as opposed to point-to-point (one sends, one receives). Broadcast, all-gather, reduce-scatter and allreduce are the collectives. NCCL is NVIDIA's library that implements them over NVLink/PCIe/InfiniBand.
- **Allreduce.** The collective that takes a tensor of the same shape on every rank, adds them element-wise, and leaves the *sum* on every rank. Analogy: eight people each hold a column of numbers; after an allreduce every person holds the column of totals. In TP each GPU computes a *partial* result of a matmul (because it holds only a slice of the weight), and the allreduce turns the partials into the full result on all GPUs. Cost ≈ latency (~10 µs on NVLink) + `2 × bytes × (N−1)/N ÷ bandwidth`; at batch 1 the tensors are tiny (16 KB), so it is pure latency.
- **Tensor parallelism (TP).** Every weight matrix is cut into N slices, one per GPU. For a pair of matmuls (attention's `q/k/v` then `o`, or the MLP's `gate/up` then `down`) the first is cut by columns (each GPU gets a full-height, 1/N-width slice and produces 1/N of the output features) and the second by rows (each GPU multiplies its 1/N of the features by its slice and produces a partial sum). One allreduce after the second matmul yields the full output. So: two allreduces per block, and every GPU reads only 1/N of the weight bytes per token.
- **Pipeline parallelism (PP).** GPU 0 holds layers 0–9, GPU 1 holds 10–19, …; activations are sent point-to-point from stage to stage. One hand-off of a `[batch, hidden]` tensor per boundary, no allreduce.
- **Bubble.** In PP, the time a stage sits idle waiting for the previous stage. At batch 1 seven of eight stages are idle at any moment. Filled only by having many **micro-batches** in flight.
- **Data parallelism (DP).** N complete copies of the model, each serving its own requests. No communication at all; N× the memory.
- **Expert parallelism (EP).** For mixture-of-experts models: different experts on different GPUs. Awareness level only here.
- **NVLink / NVSwitch.** The GPU-to-GPU link inside a node: ~900 GB/s bidirectional on H100, sub-10 µs latency. `nvidia-smi topo -m` shows `NV#` between GPUs that have it.
- **PCIe.** The link between a GPU and the CPU/NIC (and between GPUs on a machine without NVLink): 25 GB/s (Gen4 x16) to 64 GB/s (Gen5), 30–50 µs per collective.
- **InfiniBand / RDMA / GPU-direct.** The inter-node network (400 Gb/s NDR ≈ 50 GB/s) and the technique that lets a NIC write directly into another machine's GPU memory without the CPU copying.
- **Disaggregated (prefill/decode) serving.** Prefill requests run on one pool of GPUs, decode on another; the prompt's KV cache is transferred between them. **KV connector / transfer engine** (NIXL, Mooncake, LMCache) is the software that ships the blocks.
- **Aggregate bandwidth.** With TP=N, each GPU reads 1/N of the weights per token, in parallel — as if the model were on one GPU with N× the memory bandwidth. This is why TP *reduces* latency, not just memory per GPU.
- **MFU.** Achieved ÷ peak FLOPS, used below to estimate prefill time on 8 GPUs.

### Look first — where the allreduce is, in the source and on the machine (30 min)

1. **The operation itself.** `python -c "import torch.distributed as d; help(d.all_reduce)"`: "Reduces the tensor data across all machines in a way that all get the final result." In-place; every rank must call it. That one sentence is the definition above.
2. **Where vLLM calls it.** `vllm/model_executor/layers/linear.py`: read `ColumnParallelLinear` (the `weight` it creates is `[output_size_per_partition, input_size]` — the column slice) and `RowParallelLinear.forward` — at the end, `if self.reduce_results and self.tp_size > 1: output = tensor_model_parallel_all_reduce(output_parallel)`. That line is the allreduce. Then in `vllm/model_executor/models/llama.py` (or `qwen2.py`), see which class each projection uses: `qkv_proj` and `gate_up_proj` are `QKVParallelLinear` / `MergedColumnParallelLinear` (column), `o_proj` and `down_proj` are `RowParallelLinear` (row, with the allreduce). Two `RowParallelLinear`s per block → two allreduces per block, exactly as the arithmetic below assumes.
3. **The wrapper.** `vllm/distributed/communication_op.py` → `tensor_model_parallel_all_reduce` → `get_tp_group().all_reduce(...)` in `vllm/distributed/parallel_state.py` (`GroupCoordinator.all_reduce`), which picks the fastest available path: a custom allreduce kernel over NVLink for small tensors, else NCCL.
4. **The KV-transfer side (exercise 2).** `vllm/distributed/kv_transfer/kv_connector/v1/` — `base.py` defines the `KVConnectorBase_V1` interface (`start_load_kv`, `wait_for_save`, `get_num_new_matched_tokens`, …); `nixl_connector.py` is a real implementation that writes blocks with RDMA. The `examples/online_serving/disaggregated_serving/` folder has `disagg_prefill.sh`: two `vllm serve` processes with `--kv-transfer-config` (one `kv_producer`, one `kv_consumer`) and a proxy that sends each request to prefill first, then decode. Read the script to see what actually crosses: the request goes to both; the KV goes producer → consumer.
5. **On a machine.** When you rent 2 GPUs in Module 11 item 4, run `nvidia-smi topo -m` first thing. `NV2`/`NV12` between the two GPUs means NVLink; `PIX`/`PHB`/`SYS` means PCIe. The whole Step-3 argument below hinges on that one letter.

### Exercise 1 — serve Llama-3-70B fp16: how many H100s, TP or PP?

**Look first.** `vllm serve --help | grep -A2 -E "tensor-parallel-size|pipeline-parallel-size|gpu-memory-utilization"` — the three flags this exercise is deciding. Then the weight-loading log line from any TP run ("Loading weights took … Model loading took X GiB") is per rank: with TP=2 on Llama-3-8B you will see ~8 GB per GPU, which is the "weights ÷ TP" term in step 2 made visible.

**Step 1: weights.** 70.6 B × 2 bytes = **141.2 GB**. One H100 has 80 GB. Weights alone need two.

**Step 2: what's left for KV at each TP degree.** Per GPU, subtract weights/TP and ~3 GB for CUDA context, activations, and CUDA-graph workspace. Llama-3-70B KV is 327,680 bytes/token (Module 3; 2 × 80 × 8 × 128 × 2).

```
TP=2:  per GPU 80 − 70.6 − 3 =  6.4 GB  → 2 GPUs: 12.8 GB → 12.8e9 / 327,680 =  39k tokens
                                            = 4 requests at 8k context, or 1 at 32k. Unusable for real serving.
TP=4:  per GPU 80 − 35.3 − 3 = 41.7 GB  → 4 GPUs: 167 GB → 509k tokens
                                            = 62 requests at 8k. Workable.
TP=8:  per GPU 80 − 17.65 − 3 = 59.4 GB → 8 GPUs: 475 GB → 1.45M tokens
                                            = 177 at 8k, 44 at 32k. Comfortable; this is the standard config.
```

So: **2 is the arithmetic minimum, TP=4 is the practical minimum (you need KV headroom to batch at all), TP=8 is what people actually run** — it's a whole node, it gives the best batch-1 latency, and it leaves room for long contexts. With TP=2 and `--max-model-len 4096` vLLM will start, and then preempt constantly under any real load.

**Step 3: latency, the second reason for TP.** Batch-1 decode reads all 141 GB per token. TP=8 splits that across 8 GPUs, each reading 17.65 GB at 3.35 TB/s = 5.3 ms, in parallel:

```
TP=1 (hypothetical): 141.2 / 3.35   = 42 ms/token →  24 tok/s
TP=2:                 70.6 / 3.35   = 21 ms       →  47 tok/s
TP=4:                                10.5 ms      →  95 tok/s
TP=8:                                 5.3 ms      → 190 tok/s  (minus allreduce cost: 2 per layer × 80 = 160 allreduces of 16 KB each at batch 1, ~10–15 µs each on NVLink ≈ 2 ms → ~140 tok/s realistic)
```

Aggregate bandwidth is the win; the allreduce is the tax, and it's only cheap on NVLink (900 GB/s, sub-10 µs latency). Over PCIe (64 GB/s, ~30–50 µs per collective) the tax on 160 allreduces is 5–8 ms per token — you'd give back most of what TP bought. That's the rule: **TP inside the NVLink domain only.**

**Step 4: why not PP within the node.** PP=8 would put 10 layers on each GPU, weights 17.65 GB per GPU — same memory as TP=8. But at batch 1 a token visits stages sequentially: GPU 0 computes layers 0–9 while GPUs 1–7 idle, and so on. Per-token time is the *sum* of the stage times = the full 42 ms, plus 7 hand-offs. PP doesn't touch latency at all; it only helps throughput, and only when there are ≥ 8 micro-batches in flight to fill the bubbles. Use PP across nodes (where the allreduce would be over 400 Gb/s InfiniBand — 160 collectives at ~20+ µs is fatal) and TP within them; a 405B model on 2 nodes is TP=8 × PP=2.

The under-three-minute version: *"141 GB fp16 → doesn't fit one H100; two is the floor but leaves ~13 GB KV total which serves four 8k requests; TP=4 gives ~170 GB KV, TP=8 ~475 GB and 8× the bandwidth for ~140–190 tok/s at batch 1. TP because it's intra-node on NVLink and cuts latency; PP would give the same memory split but the full single-GPU latency plus bubbles, so it's for crossing nodes."*

### Exercise 2 — what moves in disaggregated serving, how big, over what link?

**Look first.** Look-first item 4 above: read `disagg_prefill.sh` and the `KVConnectorBase_V1` interface, and write down the method names in the order one request calls them (producer side: run prefill, `wait_for_save`; consumer side: `get_num_new_matched_tokens` → `start_load_kv` → decode). The two blocks of data that cross — the KV blocks and the tiny request metadata — are the "what" paragraph below; the interface names are how you say it in vLLM's vocabulary.

**What.** The prefill worker runs the prompt through the model and produces two things: the first output token (or its logits) and the full KV cache of the prompt — K and V for every layer, every kv head, every prompt token. The decode worker needs exactly that KV to continue: it appends one token's K/V per step to what it received. Nothing else crosses: weights are replicated on both pools, and activations for the prompt are consumed inside prefill. So the transfer is **prompt_tokens × per-token KV bytes**, plus a few KB of metadata (block ids, sampling params, the first token).

**How big.** Llama-3-70B, fp16 KV:

```
per token = 2 × 80 layers × 8 kv heads × 128 × 2 bytes = 327,680 bytes = 320 KiB
2k prompt = 327,680 × 2,048 = 671,088,640 bytes ≈ 671 MB (640 MiB)
8k prompt =                                         2.68 GB
```

That's per request. For Llama-3-8B it's 4× less per token (131,072 B → 268 MB per 2k prompt); for a DeepSeek-style MLA model with a 512-dim latent it's ~10× less again, which is one of the reasons MLA models are the ones where disaggregation has been pushed hardest.

**Over what link, and how long.** Divide by the per-direction bandwidth you actually get:

| Link | practical GB/s (one direction) | 671 MB transfer | notes |
|---|---|---|---|
| NVLink (H100, intra-node) | ~450 | **1.5 ms** | only if prefill and decode GPUs are in the same node — which defeats some of the point (separate pools scale independently) |
| InfiniBand NDR 400 Gb/s (RDMA, GPU-direct) | ~50 | **13 ms** | the standard inter-node choice; NIXL / Mooncake-style transfer engines use RDMA writes straight into the decode GPU's KV blocks |
| PCIe Gen5 x16 (to a NIC or host) | ~50 | 13 ms | this is also the ceiling for any NIC attached over PCIe 5 |
| PCIe Gen4 x16 | ~25 | 27 ms | |
| 100 GbE, TCP | ~10–12 | 55–65 ms | plus CPU copies; too slow for 70B at scale |

**Is that acceptable?** Compare to the work it's attached to. Prefilling a 2k prompt for 70B is 2 × 70.6 B × 2,048 = 289 TFLOP; on 8 H100s at ~50% MFU (~4 PFLOPS) that's ~73 ms. A 13 ms transfer is ~18% of prefill — and it can be **overlapped**: the KV for layer ℓ is final as soon as layer ℓ's attention has run, so the transfer engine ships layer by layer while later layers compute, hiding nearly all of it. On the decode side, 13 ms is a couple of decode steps; the request is simply admitted one step later. Over 100 GbE the 60 ms is not hideable and TTFT visibly grows.

**Why bother.** Prefill is compute-bound and bursty; decode is memory-bound and steady. Co-locating them means a 4k-token prefill stalls every decode step on that GPU (ITL spikes — the thing chunked prefill only partially fixes), and the GPU count is sized for the sum of two workloads with different bottlenecks. Splitting lets you run prefill on fewer, compute-heavy GPUs at high batch and decode on many GPUs at high KV occupancy, scale each pool on its own signal (prefill on queue depth, decode on KV utilization), and hit TTFT and ITL SLOs independently. The costs are the transfer above, weight replication in both pools, and a scheduler that must pick a decode target *before* prefill finishes so the destination blocks exist.

**One-page structure to write:** (1) what moves, one paragraph; (2) the size derivation, with the table for 8B and 70B at 2k/8k; (3) link table; (4) the overlap argument with the prefill-time comparison; (5) when it's worth it — and note that on your own hardware you can't demo this, but the Project-3 prefix-affinity router (`08-p3-walkthrough.md`) is the same *idea* applied to *where a request goes* rather than *moving its KV*.

---

## Module 8 — Production serving ops

**What you are actually learning here.** Serving is judged on a handful of metrics — TTFT, TPOT/ITL, throughput, goodput (throughput that meets SLO), cost per million tokens — and each product weights them differently. Autoscaling and routing decisions are made on engine signals (queue depth, KV utilization, in-flight requests), never CPU%, because the GPU can be at 100% "utilization" while doing almost nothing useful. The interview question is the debugging drill: *"p99 TTFT spiked at 3pm, p50 was flat — walk me through it."* Exercise 1 gives you numbers to defend; exercise 2 gives you a story you actually lived.

### Terms this module uses

- **SLO / SLA / SLI.** *SLI*: the measured indicator (TTFT p99 over 5 min). *SLO*: the internal target for it (≤ 1.5 s). *SLA*: the contractual promise to a customer, with penalties. Interviews ask for SLOs.
- **Goodput.** Requests (or tokens) per second that *met* their SLOs. A system doing 100 req/s with half of them over the TTFT target has 50 req/s of goodput.
- **Cost per 1M tokens.** GPU-hour price ÷ tokens generated per GPU-hour, × 1e6. The number that decides whether a p99 improvement is worth the extra GPUs.
- **Autoscaling signal.** The metric an autoscaler watches to add or remove replicas. For LLM serving: queue depth (`num_requests_waiting`), KV utilization, or in-flight requests — because GPU "utilization" as reported by `nvidia-smi` is "a kernel was running", which is ~100% even at batch 1 doing nothing useful.
- **GPU utilization (the trap).** `nvidia-smi`'s Util column = fraction of time *any* kernel was executing. It cannot distinguish one sequence from sixty. Never autoscale on it.
- **Queue depth.** The number of requests admitted but not yet running. The earliest sign of overload.
- **Span / trace (tracing sense).** A span is one timed stage of a request with a start and end timestamp; a trace is all the spans of one request, linked by its request id. OpenTelemetry is the standard format. Distinct from a *profiler* trace (Module 11).
- **Structured log line.** One JSON object per event (`{"request_id": ..., "stage": "prefill", "ms": 390}`) instead of free text, so it can be queried.
- **Prometheus histogram, labels, `histogram_quantile`.** A histogram metric stores counts per bucket; **labels** (`worker="w2"`, `stage="queue"`) split it into series; `histogram_quantile(0.99, ...)` estimates the p99 from the bucket counts; `rate(...[1m])` turns cumulative counts into per-second rates over the last minute; `sum by (le, stage)` adds up across everything except the bucket boundary (`le`) and the stage.
- **Hot shard / hot worker.** One worker receiving far more than its share of load. The failure mode of affinity routing.
- **Bounded-load consistent hashing.** A rule on top of the hash ring: if the chosen node is over a load limit, spill to the next node. The fix in the spike write-up.
- **Preemption storm.** Many sequences being preempted per second because KV is full; TTFT and ITL both spike.
- **Chaos test / kill-a-worker.** Deliberately terminating a worker under load to measure detection time, rerouting, and the p99 impact. Project 3 M7.

### Exercise 1 — numeric SLOs for three products

**Look first — what a serving engine exposes, so your SLOs are stated in measurable metrics (10 min).** With the Module 4 server running: `curl -s localhost:8000/metrics | grep -E '^vllm:(time_to_first_token|time_per_output_token|e2e_request_latency|request_queue_time)' | head -60`. Each of those is a histogram with `_bucket{le="0.01"}`, `_bucket{le="0.025"}`, … lines: those are the boundaries a p99 will be estimated between. Then open `vllm/v1/metrics/loggers.py` and find `PrometheusStatLogger.__init__` — the `buckets=[...]` lists for TTFT and TPOT are hard-coded there; if your SLO threshold (say 120 ms for ITL) falls between two bucket edges, the p99 estimate is interpolated, which matters when you write the alert in Module 11.

State the metric, the threshold, the percentile, and *why that number* — the justification is what's graded.

**Consumer chat (ChatGPT-style).**

| Metric | SLO | Justification |
|---|---|---|
| TTFT | p50 ≤ 400 ms, p99 ≤ 1.5 s | Human perception: < 100 ms feels instant, < 1 s keeps attention, > 2 s reads as broken. p50 at 400 ms leaves room for a 1k-token prompt's prefill on a shared GPU; p99 at 1.5 s is the queueing budget under load |
| TPOT / ITL | p50 ≤ 40 ms (≥ 25 tok/s), p99 ≤ 120 ms | Reading speed is ~250 wpm ≈ 5–6 tok/s; 25 tok/s is comfortably faster than anyone reads, so faster buys nothing for a chat UI. p99 bounds the visible "stutter" — a 120 ms gap is noticeable, a 300 ms gap looks like a freeze |
| Availability / errors | 99.9% monthly; error rate < 0.1% | Consumer product; retries are cheap but visible |
| Cost | ≤ $X per 1M output tokens, set from unit economics | The reason not to over-provision for p99 |

**Coding copilot (inline completion in an editor).**

| Metric | SLO | Justification |
|---|---|---|
| TTFT | p50 ≤ 150 ms, p99 ≤ 400 ms | Completion must appear before the developer types the next character or moves on; typing cadence is 100–200 ms per keystroke. Past ~500 ms the suggestion arrives after the developer has already decided and gets dismissed — the acceptance rate falls off a cliff |
| Full-completion latency (~20–40 tokens) | p50 ≤ 500 ms, p99 ≤ 1 s | It's a short completion; the whole thing must land in the pause after the keystroke. This implies ITL ≤ ~12–15 ms — much tighter than chat, and the reason copilots use small models (1–7B) or speculation |
| Cancellation | ≥ 30% of requests cancelled mid-flight is normal; cancelled work ≤ 20% of GPU time | Every keystroke invalidates the previous request; the server must actually stop generating on disconnect (Project 2 M1) or the wasted decode steps eat the capacity |
| Prefix cache hit rate | ≥ 70% | The file context is nearly identical between keystrokes; a miss turns a 150 ms TTFT into a 600 ms one at 4k context |

Chat-mode copilot (the side panel) uses the consumer-chat SLOs.

**Offline summarization (batch, e.g. nightly over a corpus).**

| Metric | SLO | Justification |
|---|---|---|
| TTFT / ITL | none | Nobody is watching a token stream |
| Job completion | 100% of the batch within the window (e.g. 1M documents in 6 h → ≥ 46 docs/s sustained) | The only latency that matters is the deadline for the whole job |
| Throughput / cost | maximise tok/s/GPU; cost ≤ $Y per 1M tokens, 3–5× cheaper than the interactive tier | Run at high batch (`--max-num-seqs` 256+), long chunked-prefill budgets, int8/fp8 — push the GPU to the compute ridge (Module 2). Goodput = throughput here, since there's no per-request SLO to violate |
| Failure handling | ≤ 0.01% documents failed after retry; idempotent resubmission | Batch jobs are judged on completeness |

The point to make out loud: **the same engine is configured oppositely for chat vs batch** (small batch + CUDA graphs + speculation vs huge batch + no speculation), and a mixed workload on one pool always ends up with the batch traffic destroying the chat p99 — which is why priorities, separate pools, or disaggregation exist.

### Exercise 2 — TTFT breakdown and the p99 story of one real spike

**Look first — vLLM already keeps a per-request timeline; copy its shape (15 min).** Open `vllm/v1/metrics/stats.py` and read `RequestStateStats`: it holds `arrival_time`, `queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts` — timestamps at each hop inside the engine — and `FinishedRequestStats` derives `queued_time`, `prefill_time`, `inference_time`, `decode_time` from them. Then `RequestOutput.metrics` on a returned request (or the `--collect-detailed-traces` flag) exposes them per request. Your gateway spans below are the same idea extended *outward* by three hops (gateway receive, route, stream). Also read the docstring of `prometheus_client.Histogram` (`python -c "import prometheus_client as p; help(p.Histogram)"`) — `.labels(worker=..., stage=...).observe(seconds)` is the one call your instrumentation makes.

**This is Project 3, Milestones 6 and 7** (`08-p3-walkthrough.md`: benchmarks and the kill-a-worker test): the spike you write up should be one you produced during those runs. What follows is the instrumentation plan so the story has data.

**Spans.** Every request carries a request id and a monotonic timestamp at each hop; the gateway emits one structured log line (or an OpenTelemetry trace) with the four durations:

```
t0 gateway_recv        client request arrives at the gateway
t1 queued              request admitted to the gateway's per-client rate limiter / queue     → queue  = t1 − t0
t2 routed              coordinator picked a worker (hash ring lookup + health check)         → route  = t2 − t1
t3 worker_prefill_start  worker's engine dequeued it and began prefill                       → worker_queue = t3 − t2 (includes network + the worker's own queue)
t4 first_token_at_worker  engine emitted token 1                                             → prefill = t4 − t3
t5 first_token_at_client  gateway forwarded the first SSE chunk                              → stream  = t5 − t4
TTFT = t5 − t0 = queue + route + worker_queue + prefill + stream
```

Emit each as a Prometheus histogram labelled by worker and by `cache_hit` (Project 3 M5 exposes hit/miss). The one query that tells the story:

```promql
# p99 of each TTFT stage over time. Reading inside-out:
#   rate(ttft_stage_seconds_bucket[1m])   per-second increase of every histogram bucket over the last minute
#   sum by (le, stage) (...)               add up across workers/cache_hit, keeping the bucket edge (le) and the stage label
#   histogram_quantile(0.99, ...)          estimate the 99th percentile from the bucket counts → one line per stage
histogram_quantile(0.99, sum by (le, stage) (rate(ttft_stage_seconds_bucket[1m])))
```

During a spike, exactly one stage's p99 jumps; that stage is the story.

**Template for writing up the spike.** Half a page, past tense, numbers in every sentence:

1. **Symptom.** "At 14:32 TTFT p99 went from 380 ms to 4.1 s for 6 minutes; p50 moved from 210 to 240 ms. Throughput was flat at 31 req/s."
2. **What p50-flat-p99-spiking means.** Most requests were fine; a minority waited on something. That rules out a global slowdown (weights, GPU clocks, a bad deploy) and points at queueing, a hot shard, or a class of requests.
3. **Localise by stage.** "Per-stage p99: queue 12 ms, route 1 ms, worker_queue **3.6 s**, prefill 390 ms. So requests were waiting in a worker's queue."
4. **Localise by worker.** "Only worker-2. Its `num_requests_waiting` went 0 → 40; its KV utilization hit 98%; `preemptions_total` climbed by 130."
5. **Root cause.** "The hash ring sent 38% of traffic to worker-2 because two of the 20 system-prompt templates — the two most popular — hashed to adjacent positions on the ring; with only 50 virtual nodes per worker the ring was uneven. Prefix affinity was working *as designed* and concentrating load."
6. **Fix and verification.** "Raised virtual nodes to 200 (load spread std 4% → 1%) and added a bounded-load rule: if the affinity target's queue > 8, route to the next node on the ring. Re-ran the same trace: p99 590 ms, prefix hit rate down from 91% to 84% — accepted trade."
7. **What would have caught it earlier.** "An alert on per-worker queue depth > 10 for 60 s, or on KV utilization > 90%, instead of the cluster-average alert that never fired because the average was 40%."

Rehearse it as a 90-second spoken answer. The generic drill answer, for any p99-spike question, follows the same ladder: *classify (p50 flat → minority affected) → per-stage → per-worker/per-request-class → engine signals (queue, KV, preemptions, cache hit rate) → root cause → fix → prevention.* Other root causes you should be able to name, with their signatures: a long-prompt request class hogging prefill (prefill p99 up, `prompt_tokens` p99 up, chunked-prefill budget too big); preemption storms (KV util 95%+, preemptions counter, ITL and TTFT both spike); prefix-cache eviction after a deploy that changed the system prompt (hit rate drops to 0, TTFT p50 *and* p99 up ~equally); a GPU throttling (all stages slow on one worker, `nvidia-smi -q -d CLOCK,TEMPERATURE`); garbage collection or tokenizer stalls in the API process (worker_queue fine, `stream` stage spiking — the process-boundary reason in Module 4's reading path).

---

## Module 9 — Coding fitness

**What you are actually learning here.** Inference-engineer coding rounds are rarely LeetCode-hard; they are "build a small, correct, concurrent systems component in 45 minutes while explaining it." The nine items below are the ones that actually show up — several of them are literally pieces of vLLM or of your Project 2/3 (the batcher is the scheduler's admission loop, the hash ring is Project 3 M3, the SSE endpoint is Project 2 M1). For each: a Look-first pointer to the real implementation in the standard library, Starlette, or vLLM, then the interface you should write first, the data structure that makes it O(1)/clean, the trap that costs people the round, and the test that defines "done". Solutions are deliberately not given.

### Terms this module uses

- **O(1) / O(log N) / O(N).** How the cost of an operation grows with the size of the data: constant, logarithmic (doubling the data adds one step), linear. "Both O(1)" for an LRU means neither `get` nor `put` may scan the cache.
- **Doubly-linked list.** Nodes that each point to the *previous* and *next* node. Removing a node you already hold a reference to is O(1) (re-point its neighbours at each other); a Python list would be O(N).
- **Sentinel node.** A permanent dummy head and tail so the list is never empty and insert/remove never has to check "is this the first/last node?".
- **`dict` (hash map).** O(1) key → value lookup. In the LRU it maps key → list node, so `get` finds the node without walking the list.
- **`collections.OrderedDict`.** A dict that remembers insertion order and can move a key to the end in O(1) (`move_to_end`) and pop from either end (`popitem(last=False)`). It *is* an LRU cache in disguise.
- **`collections.deque`.** A double-ended queue: O(1) append and pop at *both* ends (a list's `pop(0)` is O(N)). Optional `maxlen` drops the oldest automatically.
- **Heap / `heapq`.** A list kept in "heap order" so the smallest element is always at index 0; push and pop are O(log N). Python's heap is a *min*-heap. Tuples compare element by element, so `(time, seq, task)` orders by time, then by seq.
- **`bisect`.** Binary search in a sorted list: `bisect_right(a, x)` finds the insertion index in O(log N) — the ring lookup.
- **Monotonic clock (`time.monotonic()`).** A clock that only goes forward and is unaffected by NTP or the user changing the date. Always use it for durations; `time.time()` can jump.
- **Coroutine / `await` / `asyncio.Queue` / `Future`.** Defined in the concepts section. In this module: `await q.put(x)` *suspends* the producer when the queue is full — that suspension is backpressure; a `Future` is the object one coroutine awaits and another later `set_result`s on.
- **Backpressure.** Making a fast producer wait for a slow consumer instead of letting a queue grow without limit.
- **Sentinel value (queue sense).** A special item (`None`) put on a queue to tell consumers "no more work".
- **Async generator.** An `async def` function containing `yield`; the shape of a streaming response body.
- **`StreamingResponse`.** Starlette/FastAPI's response type that sends an iterator's chunks as they are produced.
- **Lazy deletion / tombstone.** Instead of removing an item from a heap (O(N)), mark it cancelled and skip it when it reaches the top.
- **`itertools.count()`.** An infinite counter; used to give heap entries a unique, increasing tie-breaker.
- **`collections.Counter`.** A dict subclass that counts occurrences.
- **Zipf distribution.** A "few items are very common, most are rare" distribution; realistic for prompt prefixes.
- **Count-Min Sketch / Space-Saving.** Approximate algorithms for top-k over a stream with bounded memory. Know the names.
- **Rendezvous hashing.** The alternative to a ring: for each key, score every node with `hash(key, node)` and pick the max. O(N) per lookup, no virtual nodes needed.

**The 45-minute routine.** Start a timer. 0–5 min: write the class signature and three test cases *before* any implementation (interviewers score this). 5–30 min: implement, talking through the data structure choice aloud (record yourself once a week — you'll hear the gaps). 30–40 min: run the tests, fix. 40–45 min: state complexity, one limitation, one extension. Log `(item, minutes, passed?)` in a file; anything over 45 or failed gets repeated cold three days later. Two items per week, plus three LeetCode mediums; cycle the whole set every five weeks so each one is done cold at least twice before December. Do the Look-first *before* the timer starts, the first time through each item; on the cold repeats, skip it.

### 1. LRU cache

**Look first (20 min).** Three real LRUs, from simplest to the one in vLLM:

```python
import collections, functools, inspect

# (a) OrderedDict: the whole data structure in one class. Read the docstring, then note the two methods that make it an LRU:
help(collections.OrderedDict)                  # move_to_end(key, last=True): O(1) "mark as most recently used";
                                               # popitem(last=False): O(1) "evict the least recently used"

# (b) functools.lru_cache: the standard library's LRU, written with a hand-made doubly-linked list.
#     The C accelerator normally replaces it, but the pure-Python version is right there in functools.py — read it.
print(functools.__file__)                      # open this file and find `def _lru_cache_wrapper(`
print(inspect.getsource(functools._lru_cache_wrapper))
# What to notice inside it:
#   PREV, NEXT, KEY, RESULT = 0, 1, 2, 3      — each node is a 4-element list [prev, next, key, value]
#   root = []; root[:] = [root, root, None, None]   — a SENTINEL: a circular list whose root points to itself when empty
#   cache = {}                                 — the dict: key → node
#   on hit: unlink the node (link_prev[NEXT] = link_next; link_next[PREV] = link_prev) and re-insert it just before root — "move to front"
#   when full: reuse root as the new node and make the OLDEST node (root[NEXT]) the new root — eviction without allocating
#   every write happens under `with lock:` — the thread-safety extension, already there
```

(c) vLLM's version, for the extension question: `vllm/v1/core/kv_cache_utils.py`, class `FreeKVCacheBlockQueue` — a doubly-linked list of free KV blocks with `prev_free_block` / `next_free_block` pointers on each `KVCacheBlock`, `popleft()` to allocate the least-recently-freed block and `append()` to free one, plus a `ref_cnt` on each block: a block is only in the free queue when its refcount is 0. That is "LRU over prefix blocks with refcounts", in production.

- **Interface:** `LRU(capacity)`, `get(key) -> value | None`, `put(key, value) -> None`. Both O(1).
- **Data structure:** `dict: key → node` plus a doubly-linked list with *sentinel* head and tail nodes (so insert/remove never special-case an empty list). Most-recent next to head, eviction from before tail.
- **Trap:** `get` must move the node to the front (it's a "use"); `put` on an existing key must update *and* move, not insert a second node; eviction happens *before* insert when full, and must delete from the dict too. Using `OrderedDict.move_to_end` is fine as a follow-up but the interviewer wants to see the list.
- **Test:** capacity 2: `put(1,1) put(2,2) get(1)→1 put(3,3) get(2)→None get(3)→3 put(1,10) get(1)→10 put(4,4) get(3)→None`. Then 100k random ops against a naive list-based reference.
- **Extension they'll ask:** thread safety (one lock around both methods), TTL per entry, LRU over *prefix blocks* with refcounts (that's vLLM's `BlockPool`: an entry can only be evicted when its refcount is 0 — add a `pin/unpin`).

### 2. Token-bucket and sliding-window rate limiters

**Look first (10 min).** `python -c "import time; help(time.monotonic)"` — "cannot go backwards"; that sentence is the trap below. `help(collections.deque)` — note `popleft()` is O(1) and `maxlen`. For a production token bucket, read the docstring of `asyncio.Semaphore` (`help(asyncio.Semaphore)`) to see why it is *not* one (no refill), and then vLLM's simplest admission control: `--max-num-seqs` in `Scheduler.schedule` is a concurrency cap, not a rate limit — be able to say the difference.

- **Interface:** `TokenBucket(rate_per_s, capacity)`, `allow(n=1) -> bool`. `SlidingWindow(limit, window_s)`, `allow() -> bool`.
- **Data structure:** token bucket needs only two floats: `tokens` and `last_refill` (monotonic time). Refill *lazily* inside `allow`: `tokens = min(capacity, tokens + (now − last) × rate)`. Sliding window (exact): a `deque` of timestamps; pop-left everything older than `now − window`, allow iff `len < limit`. Sliding-window *counter* (approximate, O(1) memory): count for the current and previous fixed window, weighted by overlap.
- **Trap:** using `time.time()` (jumps with NTP) instead of `time.monotonic()`; refilling in a background thread (unnecessary and racy); integer tokens (a 0.5 tok/s rate breaks); forgetting that a burst of `capacity` is *allowed* by design — that's the point of the bucket. For the deque version, a long-quiet client leaves a stale deque; it's fine, but say so.
- **Test:** rate 10/s, cap 5: 5 immediate `allow()` → True, 6th → False; `sleep(0.1)` → one more True. Sliding window limit 3 / 1 s: 3 allowed, 4th denied, after 1.01 s allowed again. Inject a fake clock (`now=lambda: ...`) so tests don't sleep.
- **Extension:** per-client keys with an LRU of buckets (item 1); distributed version with Redis `INCR`/Lua — know the words.

### 3. Bounded producer/consumer with asyncio

**Look first (15 min).** The queue is pure Python — read it:

```python
import asyncio, inspect
print(inspect.getsource(asyncio.Queue.put))        # while self.full(): create a Future, append it to self._putters, await it — THAT is backpressure:
                                                   # the producer is parked on a Future until a consumer's get() wakes it
print(inspect.getsource(asyncio.Queue.get))        # symmetric: parks on self._getters while empty
print(inspect.getsource(asyncio.Queue.task_done))  # decrements _unfinished_tasks; at zero, sets _finished → join() returns
print(inspect.getsource(asyncio.Queue.join))       # awaits _finished — hangs forever if nobody calls task_done (the trap)
```

Then vLLM's real one: `vllm/v1/engine/async_llm.py` — `generate()` creates a per-request `asyncio.Queue` (look for `RequestOutputCollector` in `output_processor.py`) that the output loop `put`s into and the HTTP handler `get`s from: one producer (the engine output loop), N consumers (one per streaming response).

- **Interface:** `async def producer(q, items)`, `async def consumer(q, sink)`, `async def run(items, n_consumers, maxsize)`; must terminate cleanly and process every item exactly once.
- **Data structure:** `asyncio.Queue(maxsize)`; `await q.put()` blocks when full — that *is* the backpressure. Shutdown: put one sentinel per consumer, or use `q.join()` with `q.task_done()` and then cancel consumers.
- **Trap:** calling `q.join()` without ever calling `task_done()` (hangs forever); cancelling consumers while items remain; a consumer exception silently killing one worker so the queue drains slower and nobody notices (wrap the body, log, continue — or let it propagate via `gather(return_exceptions=False)` and decide). Mixing threads and asyncio (`queue.Queue` in a coroutine blocks the loop).
- **Test:** 1,000 items, maxsize 10, 4 consumers with random `await asyncio.sleep(0.001)`: `len(sink) == 1000`, `sorted(sink) == items`; assert the producer observed `q.full()` at least once (proves backpressure); total runtime < 1,000 × 0.001 (proves concurrency).
- **Extension:** this is Project 2's request queue; ask yourself where "admit a waiting request into the running batch if capacity allows" lives.

### 4. Request batcher (flush on max-size OR max-wait)

**Look first (15 min).** `help(asyncio.Future)` — `set_result`, `set_exception`, and that awaiting it suspends until one of those is called; that is how each caller gets *its own* result. Then `vllm/v1/core/sched/scheduler.py`, `Scheduler.schedule()`: it is a batcher whose "max size" is `max_num_seqs` and the token budget, and whose "max wait" is *zero* — it runs every engine step and takes whatever is waiting. Ask yourself why an engine does not need the timer (answer: the GPU step *is* the clock; a request waits at most one step). Your batcher is for the case where there is no natural clock.

- **Interface:** `Batcher(max_size, max_wait_s, process: Callable[[list], list])`, `async submit(item) -> result`. Each caller awaits *its own* result.
- **Data structure:** a pending list of `(item, asyncio.Future)`; one flush coroutine per batch, started when the *first* item of a new batch arrives, that `await asyncio.sleep(max_wait)` then flushes — unless a size-triggered flush already happened, in which case it must find an empty list (or be cancelled). Results are matched to futures by index.
- **Trap:** starting a timer per item (N timers, N flushes); not cancelling the timer when the size flush fires (double flush of an empty/half batch); an exception in `process` must be set on every pending future (`fut.set_exception`) or callers hang forever; calling `process` while still holding the pending list reference so items arriving during processing join the wrong batch (swap in a new list *before* awaiting `process`).
- **Test:** `max_size=4, max_wait=0.05`: submit 3 → all resolve at ~50 ms (timer path); submit 4 → resolve at ~0 ms (size path); submit 9 → two immediate batches plus one timed; `process` that raises → all 3 callers get the exception; `process` call count is exactly the number of batches.
- **Extension:** this is iteration-level scheduling in miniature — what changes when the batch is *persistent* (sequences stay for many steps) rather than one-shot? That's continuous batching, and a good five-minute conversation.

### 5. SSE streaming endpoint (FastAPI)

**Look first (20 min).** Two real implementations, the framework's and vLLM's:

```python
import inspect, starlette.responses as r
print(inspect.getsource(r.StreamingResponse))
# Notice: __call__ runs stream_response and listen_for_disconnect CONCURRENTLY (anyio task group / asyncio.wait) —
# when the client goes away, listen_for_disconnect returns and the stream task is cancelled: that cancellation is
# what raises CancelledError inside your generator, and your `finally` is where engine-side work gets stopped.
# stream_response: for each chunk from the iterator → encode → `await send({"type": "http.response.body", "body": chunk, "more_body": True})`.
```

Then `vllm/entrypoints/openai/serving_completion.py`, `completion_stream_generator`: every chunk is yielded as `f"data: {chunk.model_dump_json()}\n\n"` and the last line is `"data: [DONE]\n\n"` — the exact wire format your endpoint must produce. `curl -N localhost:8000/v1/completions -d '{"model":"...","prompt":"hi","max_tokens":3,"stream":true}' -H 'content-type: application/json'` shows it on the wire (`-N` disables curl's output buffering so you see chunks as they arrive).

- **Interface:** `POST /generate` with `{"prompt": ..., "max_tokens": ...}` → `StreamingResponse(media_type="text/event-stream")`; events `data: {"token": "..."}\n\n`, a final `data: [DONE]\n\n`, heartbeat comments `: ping\n\n` every N seconds while idle.
- **Data structure:** an async generator; the engine pushes tokens into a per-request `asyncio.Queue`, the generator yields them; `await request.is_disconnected()` (or catching `asyncio.CancelledError` in the generator's `finally`) cancels the engine-side work.
- **Trap:** forgetting the blank line (`\n\n`) — the client never sees an event; response buffering by a proxy (send `X-Accel-Buffering: no` and `Cache-Control: no-cache`); not cancelling generation on disconnect (Project 2 M1's requirement, and the copilot SLO's cancellation budget); heartbeats that aren't comment lines (a `data:` heartbeat becomes a bogus token for the client).
- **Test:** `curl -N localhost:8000/generate -d '{"prompt":"hi","max_tokens":5}'` shows 5 events then `[DONE]`; kill curl mid-stream and assert (via a counter or log) that the generator's `finally` ran within one token interval; an `httpx.AsyncClient` test that reads events and asserts the order and count.
- **Extension:** what changes for the OpenAI-compatible `chat.completion.chunk` format; how you'd multiplex 1,000 streams (uvicorn workers × asyncio — it's fine; the engine is the limit, not the HTTP layer).

### 6. Consistent-hash ring with virtual nodes

**Look first (10 min).** `help(bisect.bisect_right)` — returns the index where `x` would go to keep the list sorted; if that index equals `len(list)` the key is past the last point and wraps to index 0 (the trap). `python -c "import hashlib; print(hashlib.blake2b(b'w1#0', digest_size=8).hexdigest())"` twice, in two separate processes → identical; `python -c "print(hash('w1#0'))"` twice → *different* (hash randomization). Then your own Project 3 M3 ring in `08-p3-walkthrough.md`, which is this item with a health check bolted on.

- **Interface:** `Ring(vnodes=150, hash=...)`, `add(node)`, `remove(node)`, `get(key) -> node`, optionally `get_n(key, n)` for replicas.
- **Data structure:** sorted list of `(hash, node)` points (`bisect` for lookup: `bisect_right(hashes, h(key))`, wrapping to index 0 past the end) plus a `dict node → its vnode hashes` so `remove` is O(v log N). Hash with `hashlib.md5`/`blake2b` on `f"{node}#{i}"`; take the first 8 bytes as an int.
- **Trap:** Python's built-in `hash()` is salted per process — keys map differently after a restart, which destroys prefix affinity across worker restarts; forgetting the wrap-around at the end of the ring; too few vnodes (load imbalance — the exact bug in the Module 8 spike write-up); `remove` that leaves stale vnodes; hashing the *whole* prompt instead of a fixed prefix (Project 3: key on the first N tokens / the system-prompt template id, otherwise no two requests ever share a node).
- **Test:** 3 nodes, 150 vnodes, 100k keys: each node gets 33% ± 3%; remove one node, re-map: exactly the keys that were on the removed node move (≈ 1/3), and none of the others (assert with a dict of before/after); add a 4th node: ≈ 25% move, all *to* the new node. Deterministic across two processes (`python ring_test.py` twice → identical assignments).
- **Extension:** bounded-load consistent hashing (skip to the next node if the target is over 1.25× average load); rendezvous hashing as the alternative and why you'd pick one over the other (rendezvous: O(N) per lookup, perfectly even, no vnodes; ring: O(log N), needs vnodes). Be ready to defend the ring vs a central routing table (Project 3 acceptance).

### 7. Heap-based scheduler

**Look first (15 min).** The standard library has two:

```python
import inspect, sched, asyncio.base_events as be
print(inspect.getsource(sched.scheduler.run))      # the textbook version: heapq of Event(time, priority, sequence, action, ...) namedtuples —
                                                   # note `sequence` from itertools.count() as the tie-breaker; peek [0], sleep until due, pop, run
print(inspect.getsource(be.BaseEventLoop._run_once))   # the event loop itself: self._scheduled is a heap of TimerHandle; cancelled handles are
                                                       # skipped LAZILY when they reach the top (and compacted only when _timer_cancelled_count
                                                       # gets large) — exactly the tombstone approach below
```

- **Interface:** `Scheduler()`, `schedule(run_at, fn, *args) -> handle`, `cancel(handle)`, `run_pending(now) -> n_ran` (or `run_forever()` with sleeping until the next due time), plus a priority variant `schedule(priority, fn)`.
- **Data structure:** `heapq` of tuples `(run_at, seq, handle, fn, args)`; `seq` from `itertools.count()` guarantees FIFO among equal times and prevents Python from ever comparing `fn` objects. Cancellation is *lazy*: a `set` of cancelled handles (or a tombstone flag on the entry) checked when popped.
- **Trap:** `TypeError: '<' not supported` when two entries tie and Python compares the next tuple element (the missing `seq`); O(N) `heap.remove()` for cancel instead of lazy deletion; `run_pending` that pops one item and returns, or that runs a task scheduled *during* the loop for a time in the past forever (bound by the entries present at entry, or by `now`); using wall-clock time.
- **Test:** schedule A@30 ms, B@10 ms, C@20 ms → runs B, C, A; cancel C before it runs → B, A; two tasks at the same time run in submission order; a task that reschedules itself every 10 ms runs ~10 times in 100 ms and stops when cancelled; a raising task doesn't kill the scheduler.
- **Extension:** map it to an inference scheduler: priority = (SLO tier, arrival time); what's the cost of *fair* scheduling across tenants (a heap per tenant + round-robin over heaps, or weighted deficit)?

### 8. Top-k frequent items

**Look first (5 min).** `print(inspect.getsource(heapq.nlargest))` — after the small-k special cases, it builds a size-k heap of `(value, order, elem)`, then walks the rest with `heapreplace` whenever an element beats the heap's *minimum* (`top[0]`). That is the streaming min-heap below, in the standard library, including the tie-break `order` counter.

- **Interface:** `top_k(items: Iterable[T], k) -> list[T]` in descending frequency; ties broken deterministically (say, by value).
- **Data structure:** `collections.Counter` then either `heapq.nlargest(k, counter.items(), key=...)` — O(n log k) — or bucket sort by count (an array of lists indexed by frequency) — O(n). Streaming variant: a size-k *min*-heap of `(count, item)`, push and pop-if-larger-than-k.
- **Trap:** using a max-heap of size k (you need to evict the *smallest*, so it's a min-heap); sorting the whole counter (O(n log n) — fine, but say why it's not optimal); mutable/unhashable items; for the streaming version, counts changing after an item is in the heap (you can't do exact streaming top-k with bounded memory — know the name Count-Min Sketch / Space-Saving for the approximate answer).
- **Test:** `[1,1,1,2,2,3]`, k=2 → `[1,2]`; k=1 with a tie → deterministic; 1M random ints from a Zipf distribution → equals `sorted(counter, key=...)[:k]`; k ≥ number of distinct items.
- **Extension:** "top-k hottest prompt prefixes in the last 5 minutes" — sliding window (item 2's deque idea) plus a counter; that's the metric behind Project 3's cache-hit dashboard.

### 9. Interval merging

**Look first (5 min).** `help(bisect.insort)` for the insert-one extension. For the allocator connection: in `vllm/v1/core/block_pool.py`, `BlockPool.get_new_blocks` hands out blocks *individually* from the free queue — vLLM never needs contiguous runs, which is exactly why it never needs to coalesce. Be able to say why an allocator that *did* hand out contiguous runs (llama.cpp's KV slots before it went paged, or a `malloc`) needs this merge.

- **Interface:** `merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]`; state whether touching intervals (`[1,3],[3,5]`) merge (usually yes — say it).
- **Data structure:** sort by start, single pass with a running `current` interval; extend `current[1] = max(current[1], end)` when `start ≤ current[1]`, else emit and restart.
- **Trap:** not sorting; comparing to the *previous input* interval instead of the running merged one (`[1,10],[2,3],[4,5]` breaks); off-by-one on the touching rule; mutating the input list.
- **Test:** `[[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]]`; `[[1,4],[4,5]] → [[1,5]]`; `[[1,10],[2,3],[4,5]] → [[1,10]]`; empty; single; already-sorted vs reversed input give the same output.
- **Extension:** insert-one-interval into a sorted merged list in O(n) without re-sorting; "free KV block ranges" coalescing in an allocator (Project 2 M3's free list is exactly this if you allocate contiguous runs).

**Done when** any of the nine can be written from a blank file, with tests, in under 45 minutes, while you narrate — and you can name the trap for each before you start typing.

---

## Module 10 — Story bank and behavioral

**What you are actually learning here.** Behavioral rounds at infra teams are technical rounds in disguise: the interviewer wants to know whether you *own* the numbers on your resume and whether you can explain a decision's trade-offs. The STAR structure keeps you from rambling; the five-whys drill guarantees no bullet on your resume can be pushed past your knowledge. The questions: *"tell me about a hard production problem"*, *"why inference, why now?"*, and the silent one behind every resume line — *"is this real?"*

### Terms this module uses

- **STAR.** Situation, Task, Action, Result — a fixed order for telling a work story so it has context, your role, what you did, and a measured outcome. Reflection is a fifth part infra interviewers expect.
- **Five whys.** Ask "why?" of your own claim five times in a row, each answer one level more specific. If you run out of answers, the claim is not yours yet.
- **Interview hook.** A phrase in a story that invites a technical follow-up you *want* ("we hit the KV limit" → "tell me how you sized it").
- **SPOF.** Single point of failure — one component whose loss takes down the system.
- **qLoRA.** Fine-tuning a model by training small low-rank adapter matrices on top of a 4-bit-quantized frozen base. Appears in your Purdue×Microsoft story.
- **A/B eval.** Comparing two model versions on the same held-out prompts with a defined scoring rule, so "better" is a number.

### Look first (30 min, once)

Before writing any story, print the raw material: your current resume, and the README of each project as it stands today. For every quantitative claim, write next to it *the file or command that produced the number* (`results/c16.json`, `bench.py --profile`, the M6 plot). A number without a file is a bullet that fails the five-whys below. Then read one engineering-blog post from each company on your target list (vLLM blog, Anyscale, Baseten, Modal, Fireworks, Together, Character) and note one concrete thing in each you could reference in "why them".

### STAR template (half a page written, 90 seconds spoken)

```
Title:        one line, the way you'd name it in conversation ("the GymClan notification storm")

Situation:    2 sentences. System, scale (users, QPS, team size), what was at stake. Numbers.
Task:         1 sentence. What *you* specifically were responsible for — not the team.
Action:       4–6 sentences. The decision points, in order. For each: what you considered,
              what you chose, why. At least one thing you tried that didn't work.
Result:       2 sentences. Measured outcome (before → after), and what it cost.
Reflection:   1 sentence. What you'd do differently, or what you learned that you reused later.

Interview hooks (not spoken, for your notes): which technical questions this story naturally
leads to, and the 2–3 follow-ups you expect ("why not X?").
```

The 90-second spoken version is S (15 s) – T (5 s) – A (50 s) – R (15 s) – reflection (5 s). Time it. Write one story per resume project: Ordermatic (solo multi-tenant SaaS — the tenancy-isolation decision), GymClan (500k users on a 2-person team — the scaling incident), Hexion (GenAI eval mechanisms — how you decided what "better" meant), Purdue×Microsoft (qLoRA fine-tuning — the overfitting diagnosis and the A/B eval), and after each project in this plan, one for it (Project 3's spike write-up from Module 8 *is* a STAR story).

### The pivot story — "why inference?" (worked example, first person, ~150 words)

*(Built from the facts in the guide; replace the bracketed specifics with yours and keep it under 90 seconds.)*

> I've spent the last few years as a backend engineer — Python services on Kubernetes and GCP — where the problems I liked most were capacity problems: at GymClan we were two people serving 500k users, so everything came down to what a request actually costs and where the queue forms. At Purdue, working with Microsoft, I fine-tuned models with qLoRA, and the part that surprised me wasn't training — it was that the fine-tuned model was useless until I could serve it fast enough to run the A/B eval, and I had no idea what determined the speed. So I went and found out: I built a Qwen inference engine from scratch, measured a naive loop at [22] tok/s and a KV-cached one at [118] on my own 2060, and worked out where the rest of the bandwidth ceiling went. Inference is the place where my systems instincts — queues, caches, cost per request — meet the model directly, and it's the bottleneck for every product built on these models. That's where I want to work.

Why this shape: it names a concrete trigger (the eval that couldn't run), proves the pivot with something *done* (the engine, the numbers), and ends on why the field, not why the job. The follow-up will be "tell me about the engine" — Project 1's STAR story.

### The five-whys drill for a resume bullet

Take one bullet and answer "why?" five times in a row, each answer one sentence and one level deeper. If you can't answer the fifth, the bullet is rewritten to what you *can* defend.

```
Bullet: "Built a KV-cache-aware routing layer on Kubernetes that cut TTFT by 42% at 32 concurrent users."

Why did routing cut TTFT?      Because requests sharing a system-prompt prefix landed on the worker whose
                               engine already had that prefix's KV, so prefill only ran on the suffix.
Why does a cached prefix help?  Prefill is compute-bound; a 1k-token prefix is ~200 ms of matmuls on that
                               worker, and a prefix-cache hit skips them (Module 4 exercise 2).
Why 42% and not 90%?           Only ~60% of requests hit — 20 templates over 4 workers, LRU eviction under
                               memory pressure, and the suffix still costs prefill + queueing; at 32 users
                               the queue was ~40% of TTFT and routing doesn't shrink the queue.
Why consistent hashing?        So adding/removing a worker remaps ~1/N of prefixes instead of all of them,
                               which keeps the hit rate up through scaling events; a central table would
                               have been a SPOF and a rendezvous hash was O(N) per lookup — fine at N=4,
                               but I wanted the ring's O(log N) for the design to hold at 50.
Why did you measure at 32?     It was the knee of the throughput-vs-p99 curve for the workers I had; below
                               it the queue was empty and routing mattered less, above it p99 blew up
                               regardless of routing.
```

Run this on every bullet before resume v2 ships (Project 3's deadline). Any "why" whose answer is "that's what the tutorial did" or "I don't remember" is a bullet you will be caught on.

---

## Module 11 — Production toolchain

**What you are actually learning here.** "How would you find the bottleneck?" is answered with tool names *and* a story from a system you profiled; "how do you know it's working in production?" is answered with a dashboard you built and two alerts you can justify; "have you run multi-GPU?" with a scaling-efficiency number you measured. The six items below are each a half-day to a day. Each one ends with a graph or table for a README.

### Terms this module uses

- **Profiler.** A tool that records what a program was doing, when. Two kinds: an *instrumenting* profiler hooks into the program (`torch.profiler` sees every PyTorch op and every CUDA kernel, with start/end times) and a *sampling* profiler stops the process a few hundred times a second from outside and writes down the call stack each time (`py-spy`). Sampling costs almost nothing and needs no code changes; instrumenting is exact but heavier.
- **Trace (profiler sense).** The recorded timeline: one row per CPU thread and per GPU stream, with a box for each function or kernel. Saved as Chrome-trace JSON (`export_chrome_trace`) or Nsight's `.nsys-rep`. Open it in `ui.perfetto.dev` and *look at the gaps*.
- **Flame graph.** A picture built from stack samples. The x-axis is *not* time: each bar's width is the fraction of samples in which that function was on the stack; bars stack upward by call depth (the function at the bottom called the one above it). A wide bar at the top of a stack is where the program was actually spending its time; a wide bar with nothing above it is self time. Read it by looking for the widest top-most bars.
- **GPU busy fraction.** (Sum of kernel execution time) ÷ (wall-clock time). 40% means the GPU was idle 60% of the time, waiting for the CPU — the launch-bound signature.
- **`torch.cuda.synchronize()`.** Wait for every queued GPU kernel to finish. Without it a Python timer measures only how long it took to *enqueue* work, not to do it.
- **NVTX range.** A named marker (`torch.cuda.nvtx.range("decode_step")`) that shows up as a labelled box on the Nsight timeline so you can tell where a step starts and ends.
- **Kernel launch (`cudaLaunchKernel`).** The CUDA API call the CPU makes to start one kernel. Its count and total time in a profile are the launch-overhead number.
- **Prometheus scrape.** Prometheus does an HTTP GET to `http://<server>/metrics` every `scrape_interval` seconds (5 s below), parses the plain-text lines it receives (`vllm:num_requests_running 3`), and stores each as a timestamped sample. The server does nothing but answer the GET; that pull is the scrape. `scrape_configs` in `prometheus.yml` lists what to pull from.
- **Exporter / `/metrics` endpoint.** Any HTTP endpoint that serves metrics in Prometheus's text format. vLLM has one built in; your gateway exposes one with `prometheus_client`.
- **Time series / label.** Each distinct combination of metric name and labels (`vllm:num_requests_running{model="..."}`) is one time series. `sum by (le)` and friends aggregate across labels.
- **Counter / gauge / histogram.** Counter: only increases (`generation_tokens_total`) — plot `rate()`. Gauge: current value (`kv_cache_usage_perc`) — plot as is. Histogram: bucket counts — feed to `histogram_quantile`.
- **Grafana panel / dashboard / provisioning.** A panel is one graph of one or more PromQL queries; a dashboard is a grid of panels, stored as JSON; provisioning is Grafana loading data sources and dashboards from files at startup instead of clicks.
- **Alert rule / `for:`.** A PromQL expression evaluated every interval; it fires only after being true continuously for the `for:` duration, so one bad sample does not page.
- **Arrival trace.** A file listing when each request should be sent and what it contains, so the same load can be replayed identically against different engines.
- **`httpx.AsyncClient`.** An async HTTP client; `client.stream(...)` reads a streaming response chunk by chunk without blocking the event loop.
- **Scaling efficiency.** `throughput(N GPUs) ÷ (N × throughput(1 GPU))`. 1.0 is perfect; TP typically lands at 0.7–0.85 on a model that fits one GPU.
- **Warm pool / min replicas.** Keeping at least one server running so scale-up never starts from zero (a cold start) for the interactive tier.
- **Page cache (Linux).** RAM the kernel uses to keep recently read file data; a second weight load from the same disk is "warm" because it comes from page cache, not disk. `echo 3 > /proc/sys/vm/drop_caches` empties it for a cold measurement.
- **safetensors / mmap.** The weight file format that can be memory-mapped (read on demand, no unpickling) — the reason it loads faster than `.bin`.
- **Readiness probe / init job / DaemonSet.** Kubernetes: the check traffic waits on; a one-shot job run before a pod is marked ready; one copy of a pod on every node. `kubectl explain pod.spec.containers.readinessProbe` prints the field's documentation.

### 1. Profiling: where does a decode step go?

**Look first (20 min).** Three things before profiling your own engine:

1. `python -c "import torch.profiler as p; help(p.profile)"` — read the `schedule`, `on_trace_ready`, `with_stack`, `record_shapes` arguments; the script below uses each. Then `help(p.schedule)`: `wait` steps are skipped, `warmup` steps are traced but discarded, `active` steps are recorded.
2. vLLM has the same profiler wired in. In `vllm/v1/worker/gpu_worker.py` find `Worker.profile(is_start)` — it constructs `torch.profiler.profile(activities=[CPU, CUDA], with_stack=True, on_trace_ready=tensorboard_trace_handler(...))` — and in `vllm/entrypoints/openai/api_server.py` the `/start_profile` and `/stop_profile` routes that call it. Start a server with `VLLM_TORCH_PROFILER_DIR=./vllm_prof vllm serve ...` (the env var that enables those routes), `curl -X POST localhost:8000/start_profile`, send ten requests, `curl -X POST localhost:8000/stop_profile`, and open the resulting `.json.gz` in Perfetto. That is the trace of a *production* engine's decode step; yours, below, is what you compare it to.
3. `py-spy dump --pid <PID>` on the running vLLM server (and `--subprocesses`) prints every thread's current stack once — a one-sample flame graph. Do it three times; you will already see the engine-core process sitting in `execute_model` / a CUDA wait most of the time.

The question: in one decode step of your Project-1/2 engine, how much time is the GPU actually busy vs the CPU doing Python and launching kernels? (Module 2 predicted the answer for a 0.5B model on a 2060: the CPU side dominates.)

**torch.profiler — a short trace window inside your own engine.**

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],   # record both the Python/CPU side and the GPU kernels — the gap between them is the answer
    schedule=schedule(wait=5, warmup=2, active=5, repeat=1),    # skip 5 steps (JIT/allocator noise), warm 2 (traced, discarded), record 5, once
    on_trace_ready=lambda p: p.export_chrome_trace("decode_trace.json"),   # when the 5 active steps are done, write a Chrome-trace JSON for Perfetto
    record_shapes=True,                                         # keep tensor shapes per op, so the table can distinguish the LM-head GEMM from the MLP GEMMs
    with_stack=True,                                            # keep the Python call stack per op, so each kernel is attributable to a line of your code
)
prof.start()
for _ in range(12):                                             # 12 = wait 5 + warmup 2 + active 5
    step(model, cache, next_token)      # your one-token decode step: forward over [1, 1] input ids with the KV cache, returns the next token
    torch.cuda.synchronize()            # wait for the GPU so the step's kernels are inside this step's window, not the next one's
    prof.step()                         # tell the profiler "one step done" so the schedule advances
prof.stop()
# Aggregate table, one row per op, sorted by GPU time. Self CUDA = time the op's own kernels ran on the GPU.
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
```

Open `decode_trace.json` at `https://ui.perfetto.dev` (drag the file in). Read three things: (1) the CUDA row's *gaps* — idle GPU between kernels is launch/Python overhead; (2) `key_averages` `Self CUDA` total per step ÷ wall-clock per step = GPU busy fraction; (3) the top kernels by CUDA time (expect the LM-head GEMM — 896×151,936 — and the MLP GEMMs; if `aten::copy_` or `aten::cat` is near the top, your KV cache is doing a full copy per step — the M3 bug that `torch.cat` on the cache introduces).

**Expected** for Qwen2.5-0.5B fp16, eager, batch 1: step wall time 6–9 ms, CUDA busy 2.5–3.5 ms → GPU busy **35–50%**; ~350–400 kernel launches per step. With `torch.compile(mode="reduce-overhead")` (CUDA graphs) the wall time should drop toward 3–4 ms and GPU busy to 80%+. That before/after pair is the README figure.

**py-spy on the live server** (vLLM or your Project 2): no code changes, samples the Python stack.

```bash
pip install py-spy
# top: a live, refreshing view of which Python functions the process is in, like `top` for stack frames.
#   $(pgrep -f "vllm serve" | head -1): the PID of the first process whose command line contains "vllm serve" (the API server).
#   Under WSL2 attaching to another process usually needs sudo (ptrace permission).
py-spy top --pid $(pgrep -f "vllm serve" | head -1)
# record: sample for 30 s and write a flame graph SVG.
#   --duration 30      seconds to sample
#   --rate 250         samples per second (default 100; higher = finer, slightly more overhead)
#   --subprocesses     also sample child processes — REQUIRED for vLLM v1, whose engine core is a separate process (Module 4 step 4)
#   --native           include C/C++ frames (libtorch, libcuda) so you can see time inside kernels-launching code vs pure Python
#   -o flame.svg       output file; open it in a browser
sudo py-spy record --pid <PID> --duration 30 --rate 250 --subprocesses --native -o flame.svg
```

`--subprocesses` matters for vLLM v1: the engine core is a separate process (Module 4 step 4), and the interesting stack is there. `--native` includes C frames so you can see time inside `libtorch`/`libcuda` vs Python. The flame graph's wide bars are your answer: under load, expect `schedule()`, `execute_model` (mostly waiting on CUDA), sampling, and detokenization in the API process. Put `flame.svg` in the README.

**Nsight Systems — one trace, GPU and CPU timelines together.**

```bash
# install: apt-get install nsight-systems (or the .run from NVIDIA); on WSL2, CUDA tracing works, CPU sampling may need --sample=none
#   -t cuda,nvtx,osrt              trace CUDA API calls + kernels, NVTX ranges (your step markers), and OS runtime calls (sleeps, mutexes)
#   --sample=none                  disable CPU stack sampling (unsupported/flaky under WSL2; the CUDA timeline is what you want anyway)
#   --capture-range=cudaProfilerApi   record only between torch.cuda.profiler.start() and .stop() in the script — skips model loading
#   --capture-range-end=stop       stop recording (and let the script keep running) when .stop() is hit
#   -o decode_profile              output file decode_profile.nsys-rep
#   python bench.py --profile      your script; --profile is YOUR flag that enables the start/stop calls and NVTX ranges
nsys profile -t cuda,nvtx,osrt --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o decode_profile python bench.py --profile
# Text summaries from the trace:
#   cuda_gpu_kern_sum   per-kernel total GPU time, count, average — the "top kernels" list
#   cuda_api_sum        per-CUDA-API-call totals — the cudaLaunchKernel row is the launch count and CPU time spent launching
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum decode_profile.nsys-rep
```

In `bench.py`, bracket the steps you want with `torch.cuda.profiler.start()` / `torch.cuda.profiler.stop()` and wrap each step in `torch.cuda.nvtx.range("decode_step")` so the timeline shows step boundaries. Open the `.nsys-rep` in the Nsight Systems GUI (Windows side is fine — copy the file to `/mnt/c/...`): the CUDA HW row vs the CPU thread row makes the "GPU idle while Python runs" story literally visible. `cuda_gpu_kern_sum` gives per-kernel totals; `cuda_api_sum` shows `cudaLaunchKernel` count and time — the launch-overhead number.

**README paragraph you're producing:** "Per decode step, wall 7.8 ms, GPU busy 2.9 ms (37%), 372 launches at ~9 µs; LM head 31% of GPU time, MLP GEMMs 44%, attention 6%. CUDA graphs cut wall to 3.6 ms (81% busy)."

### 2. Three-engine comparison with a shared arrival trace

**Look first (15 min).** vLLM's benchmark already generates Poisson arrivals; read how before writing your own. In `vllm/benchmarks/serve.py` find `get_request(...)`: with `--request-rate R` it sleeps `np.random.gamma(shape=burstiness, scale=1/(R·burstiness))` between requests — at `--burstiness 1.0` (default) the gamma is an exponential, i.e. Poisson arrivals, and `--burstiness 0.5` makes them burstier. That is the `random.expovariate(4.0)` line below, in production. Also `vllm bench serve --help | grep -A3 -E "request-rate|burstiness|dataset-name"`. The reason for a *file* rather than the flag: three different engines' benchmark tools would each draw their own random arrivals; the file makes the three runs see byte-identical load.

**The trace** is what makes the comparison fair. Generate it once, replay it identically against each engine:

```python
# make_trace.py — Poisson arrivals, realistic prompt-length mix
import json, random
random.seed(0)                                                # fixed seed → the same trace every time you regenerate it
t = 0.0                                                       # arrival time of the current request, seconds from start
with open("trace.jsonl", "w") as f:                           # one JSON object per line
    for i in range(500):                                      # 500 requests ≈ 125 s at 4 req/s
        t += random.expovariate(4.0)                          # exponential gap with mean 1/4 s → Poisson arrivals at 4 req/s mean
        in_len  = random.choice([64, 256, 256, 1024, 2048])   # prompt length in words; 256 twice so it is the mode: mostly short, some long
        out_len = random.choice([32, 128, 128, 256])          # max output tokens requested
        # "lorem" × in_len ≈ in_len tokens; a real corpus would be better, but identical text across engines is what matters here
        f.write(json.dumps({"t": round(t, 4), "prompt": " ".join(["lorem"] * in_len), "max_tokens": out_len}) + "\n")
```

**The replayer** sends request i at `t_i` regardless of whether earlier ones finished (open loop), records TTFT, ITL list, e2e, output tokens, errors:

```python
# replay.py  — usage: python replay.py --url http://host:8000/v1/completions --model M --trace trace.jsonl --out vllm.jsonl
import asyncio, json, time, argparse, httpx

async def one(client, url, model, req, t0, out):
    """Send one request at its scheduled time and record its latencies into `out`."""
    await asyncio.sleep(max(0, t0 + req["t"] - time.perf_counter()))   # wait until this request's arrival time (t0 = when the replay started)
    start = time.perf_counter(); first = None; n = 0                    # first = time of first token; n = tokens received
    # stream=True in the body → SSE chunks; client.stream(...) reads them as they arrive; timeout 300 s so slow engines don't error out
    async with client.stream("POST", url, json={"model": model, "prompt": req["prompt"], "max_tokens": req["max_tokens"], "stream": True}, timeout=300) as r:
        async for line in r.aiter_lines():                              # one SSE line at a time
            if line.startswith("data:") and "[DONE]" not in line:       # a token chunk (skip the terminator and blank lines)
                now = time.perf_counter(); first = first or now; n += 1 # first stays at the first token's time; count every chunk
    out.append({"ttft": first - start, "e2e": time.perf_counter() - start, "tokens": n})   # per-request record

async def main(a):
    reqs = [json.loads(l) for l in open(a.trace)]; out = []             # load the trace; `out` collects results from all coroutines
    async with httpx.AsyncClient() as c:                                # one connection pool shared by all in-flight requests
        t0 = time.perf_counter()                                        # replay start; every request's `t` is relative to this
        await asyncio.gather(*(one(c, a.url, a.model, r, t0, out) for r in reqs))   # launch ALL requests as coroutines; each sleeps until its own time
    json.dump(out, open(a.out, "w"))                                    # write the results for the summary/plot step

p = argparse.ArgumentParser(); [p.add_argument(f"--{k}") for k in ("url", "model", "trace", "out")]   # four string flags
asyncio.run(main(p.parse_args()))
```

**The engines.** Same model (Llama-3-8B-Instruct fp16, or Qwen2.5-7B — pick one all three support), same `max_model_len` 4096, same GPU. TensorRT-LLM and SGLang both want Ampere or newer, so **do all three in one sitting on the rented GPU from item 4** (an A100 or L40S), not on the 2060.

- vLLM: `vllm serve MODEL --max-model-len 4096` (defaults are fine; note the version).
- SGLang: `python -m sglang.launch_server --model-path MODEL --context-length 4096 --port 30000` (`--context-length` is SGLang's name for max-model-len; OpenAI-compatible endpoint at `/v1/completions`).
- TensorRT-LLM: `docker run --gpus all -it nvcr.io/nvidia/tensorrt-llm/release:<tag>` (`--gpus all`: expose the GPUs to the container) then `trtllm-serve MODEL --max_seq_len 4096` (recent versions serve HF checkpoints directly with the PyTorch backend; older ones need `trtllm-build` first — budget an hour for the build path and note "friction" honestly).

**The table** (one row per engine): output tok/s over the trace, TTFT p50/p99, ITL p50/p99, e2e p99, peak VRAM (`nvidia-smi --query-gpu=memory.used --format=csv -l 1 > mem.log` during the run — `-l 1` = repeat every second), startup time, and a "friction" column in words (install time, what broke, what you had to read). Re-run the trace at 2× and 4× the arrival rate (edit `expovariate(8.0)`, `(16.0)`) to find where each engine's p99 breaks. Expect the three to be within ~15% on throughput for a dense 8B at these rates, with TRT-LLM ahead on ITL and behind on friction, and SGLang ahead on prefix-heavy traces (RadixAttention) — if your numbers say otherwise, that's more interesting, write it up.

### 3. Prometheus + Grafana for vLLM `/metrics`

**Look first — vLLM ships a working dashboard; read it before building yours (30 min).** In the vLLM repo, `examples/online_serving/prometheus_grafana/` contains four files: `README.md`, `docker-compose.yaml`, `prometheus.yaml`, and `grafana.json`. Open them in this order:

1. `prometheus.yaml` — a `scrape_configs` entry with `job_name: vllm`, `metrics_path: /metrics`, `targets: ['host.docker.internal:8000']`, and a short `scrape_interval`. Compare with the one below; they are the same thing.
2. `docker-compose.yaml` — the Prometheus and Grafana services, the volume mount for the config, and `extra_hosts` for reaching the host. Same as below.
3. `grafana.json` — the dashboard, exported as JSON. It is long; search it for `"expr":`. Each hit is one PromQL query on one panel: you will find `histogram_quantile(0.99, sum by(le) (rate(vllm:e2e_request_latency_seconds_bucket[$__rate_interval])))`, `rate(vllm:prompt_tokens_total[$__rate_interval])`, `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc` (or `gpu_cache_usage_perc` in older exports), and the TTFT/TPOT histograms. `$__rate_interval` is a Grafana variable meaning "a sensible window for the current zoom"; `[1m]` in the queries below is the fixed equivalent. Note also which panels *they* chose and in what order — then import the JSON into your Grafana (Dashboards → New → Import → paste) so you have a working reference next to your own.
4. `README.md` — the exact `vllm serve` and `vllm bench serve` commands they used to make the screenshot; run them so your dashboard is not empty.

Then confirm the metric names your version actually exports (they shifted between V0 and V1):

```bash
# List every vllm: metric name once. grep '^vllm:' keeps only the metric lines; cut -d'{' -f1 drops the {labels}; cut -d' ' -f1 drops the value; sort -u dedupes.
curl -s localhost:8000/metrics | grep '^vllm:' | cut -d'{' -f1 | cut -d' ' -f1 | sort -u
```

```yaml
# docker-compose.yml  (run from WSL2; vLLM runs on the host at :8000)
services:
  prometheus:
    image: prom/prometheus:v2.53.0                          # pinned version; the config format below is stable across 2.x
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]   # mount your scrape config over the default one inside the container
    ports: ["9090:9090"]                                    # Prometheus UI/API at localhost:9090 (use it to test PromQL before Grafana)
    extra_hosts: ["host.docker.internal:host-gateway"]      # makes "host.docker.internal" resolve to the WSL2 host, where vLLM listens on :8000
  grafana:
    image: grafana/grafana:11.1.0                           # pinned
    ports: ["3000:3000"]                                    # Grafana UI at localhost:3000
    environment: [GF_SECURITY_ADMIN_PASSWORD=admin]         # admin/admin login; fine on a laptop, never in production
    volumes: ["./grafana-provisioning:/etc/grafana/provisioning"]   # optional: datasource + dashboard YAML/JSON loaded at startup instead of by clicking
```

```yaml
# prometheus.yml
global: { scrape_interval: 5s }              # pull every 5 s from every target; 5 s is fine for a laptop (15 s is the production default)
scrape_configs:
  - job_name: vllm                           # the `job` label on every series from this target
    metrics_path: /metrics                   # the path to GET (this is the default; written out so it's explicit)
    static_configs: [{ targets: ["host.docker.internal:8000"] }]   # the vLLM server, reached through the extra_hosts mapping above
  - job_name: gateway            # your Project-3 gateway's own metrics
    static_configs: [{ targets: ["host.docker.internal:9100"] }]   # wherever your gateway's prometheus_client endpoint listens
rule_files: ["alerts.yml"]                   # alert rules are loaded from this file (mounted next to prometheus.yml)
```

`docker compose up -d` (`-d`: detached, run in the background), then Grafana at `localhost:3000` → add Prometheus data source `http://prometheus:9090` (the compose service name resolves inside the compose network). First check what your vLLM version actually exports — names shifted between V0 and V1 (the `grep` above).

**The seven panels** (PromQL; adjust names to what the grep printed):

| # | Panel | Query | How to read the query |
|---|---|---|---|
| 1 | TTFT p50 / p95 / p99 | `histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[1m])))` (×3 with 0.5, 0.95) | per-second bucket increments over the last minute, summed across labels except the bucket edge `le`, then the 99th percentile estimated from the buckets |
| 2 | ITL / TPOT p50 / p99 | `histogram_quantile(0.99, sum by (le) (rate(vllm:time_per_output_token_seconds_bucket[1m])))` | same shape, on the per-output-token histogram |
| 3 | Throughput | `sum(rate(vllm:generation_tokens_total[1m]))` and `sum(rate(vllm:prompt_tokens_total[1m]))` (two series) | counters → per-second rates; generated vs prompt tokens as two lines |
| 4 | KV cache utilization | `vllm:kv_cache_usage_perc` (V1) or `vllm:gpu_cache_usage_perc` (V0) — gauge, 0–1 | a gauge: plot the raw value; 0.9 = 90% of KV blocks in use |
| 5 | Queue depth / running | `vllm:num_requests_waiting`, `vllm:num_requests_running` | two gauges; waiting > 0 for long is the first overload sign |
| 6 | Preemptions | `rate(vllm:num_preemptions_total[5m])` | counter → preemptions per second, smoothed over 5 min; should be 0 |
| 7 | Prefix-cache hit rate | `rate(vllm:prefix_cache_hits[5m]) / rate(vllm:prefix_cache_queries[5m])` (V1; V0 exposes `vllm:gpu_prefix_cache_hit_rate`) | ratio of two counter rates → fraction of queried tokens that hit, 0–1 |

Panels 4–7 are the ones that explain 1–2; arrange them so the causes sit under the symptoms.

**Two alerts, with thresholds you can defend** (`alerts.yml`):

```yaml
groups:
  - name: vllm                                  # rule group name (evaluated together on the global interval)
    rules:
      - alert: TTFTp99BreachingSLO              # alert name as it appears in Alertmanager / Grafana
        # p99 TTFT over a 5-minute window (5m rather than 1m so a single slow request can't trip it) compared to the 1.5 s SLO
        expr: histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m]))) > 1.5
        for: 3m                                 # must stay true for 3 consecutive minutes before firing — filters a single burst
        labels: { severity: page }              # routing label: this one wakes someone up
        annotations:
          summary: "TTFT p99 > 1.5 s for 3 min (consumer-chat SLO from Module 8); users are seeing it"
      - alert: KVCacheSaturating
        # average KV usage over 2 minutes above 90%: preemptions start at ~95%, so this is the leading indicator
        expr: avg_over_time(vllm:kv_cache_usage_perc[2m]) > 0.90
        for: 2m
        labels: { severity: warn }              # a warning, not a page: there is still time to act
        annotations:
          summary: "KV cache > 90% for 2 min: preemptions start at ~95% and TTFT/ITL p99 follow within a minute — scale out or shed load now"
```

Why these two and not, say, GPU utilization: the first is the *user-facing* SLO itself (alert on symptoms); the second is the *leading indicator* that precedes the symptom by long enough to act (alert on the one cause you can act on). `for: 3m` and `2m` are there so a single burst doesn't page. A third you might add: `rate(vllm:num_preemptions_total[5m]) > 0` for 5 min — but it fires *after* the KV alert, so it's a diagnostic, not a page. Export the dashboard JSON into the repo and screenshot it under load (the Module 4 sweep makes a nice one).

### 4. Multi-GPU: a few hours on rented 2× GPUs

**Look first (before renting).** (1) `vllm serve --help | grep -B1 -A3 -E "tensor-parallel-size|distributed-executor-backend"` — the two flags that decide TP degree and how the workers are launched. (2) `vllm/v1/executor/multiproc_executor.py` — `MultiprocExecutor._init_executor` spawns one `WorkerProc` per TP rank and `execute_model` broadcasts the scheduler output to all of them over shared memory; `vllm/distributed/parallel_state.py` — `init_distributed_environment` / `initialize_model_parallel` build the NCCL process groups the allreduce runs on. (3) On the rented box, before anything else: `nvidia-smi topo -m` (the NVLink-vs-PCIe letter from Module 7) and `nvidia-smi --query-gpu=name,memory.total --format=csv`. (4) Module 7's Look-first item 2 — the `RowParallelLinear.forward` allreduce — is the line whose cost the A-vs-B comparison below measures.

**Budget and plan.** 2× A100 80 GB (NVLink if the provider offers it — check `nvidia-smi topo -m` for `NV12`/`NV#` vs `PIX`/`SYS`) at ~$3–4/h, or 2× L40S 48 GB at ~$2/h. Three hours is enough if the scripts are ready *before* you rent. Everything below is prepared and tested locally on the 2060 with a tiny model first.

**Script checklist (in the repo before renting):**

```
setup.sh        pip install vllm==<pinned>; huggingface-cli login (Llama needs it); pre-download weights to /workspace/hf
                (`huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct`); nvidia-smi topo -m > topo.txt
bench_one.sh    args: TP, port, model → starts `vllm serve $MODEL --tensor-parallel-size $TP --max-model-len 4096 --port $PORT`
                (--tensor-parallel-size N: shard every weight matrix over N GPUs, one worker process each),
                waits for /health, runs the Module-4 concurrency sweep (C = 1, 4, 16, 64) with `vllm bench serve`,
                saves results/<model>_tp<TP>_c<C>.json, kills the server
run_all.sh      A: Llama-3-8B fp16 TP=1 (GPU 0 only: CUDA_VISIBLE_DEVICES=0 — the env var that hides all other GPUs from the process)
                B: Llama-3-8B fp16 TP=2
                C: Llama-3-8B fp16 DP=2 — two TP=1 servers on ports 8000/8001 (CUDA_VISIBLE_DEVICES=0 and =1), a 10-line
                   round-robin proxy on 8002, same sweep
                D: Llama-3-70B-Instruct fp16 TP=2 --max-model-len 4096 --gpu-memory-utilization 0.95 (tight: 70.6 GB weights/GPU
                   + ~3 GB overhead vs 76 GB budget; if it OOMs, `--quantization fp8` (weight-only on Ampere, ~35 GB/GPU) and say so)
collect.sh      tar results/ topo.txt vllm-version → scp home. Then TERMINATE THE INSTANCE (set a phone timer at rental time)
```

**What you're measuring and what to expect.**

- *A vs B at C=1 (latency):* Llama-3-8B fp16 on one A100 (2.0 TB/s): 16.06 GB / 2 TB/s = 8 ms floor → ~100–110 tok/s real. TP=2 reads 8 GB per GPU → 4 ms floor, plus 64 allreduces of 8 KB (hidden 4096 × 2 B) at ~10 µs on NVLink ≈ 0.6 ms → ~5 ms → ~150–170 tok/s. **TP cuts batch-1 latency ~1.5×, not 2×.** Over PCIe the allreduces cost 30–50 µs each ≈ 2–3 ms and TP=2 lands at ~6.5–7 ms — barely better than TP=1. Report which topology you got.
- *B vs C at C=64 (throughput):* DP=2 should beat TP=2 — each replica batches independently with no per-layer sync, and at high batch you're compute-bound so the bandwidth aggregation TP brings stops mattering. Scaling efficiency = `throughput(2 GPUs) / (2 × throughput(1 GPU))`: expect DP ≈ 0.95–1.0, TP ≈ 0.7–0.85. **This is the "TP costs you on a model that fits on one card" figure the guide asks for**: TP buys latency at low batch and pays for it in throughput at high batch; use DP (replicas) when the model fits and TP only when it doesn't or when batch-1 latency is the product.
- *D (70B on 2 GPUs):* the "forced" case. Batch-1 decode: 70.6 GB per GPU / 2 TB/s = 35 ms floor → ~25 tok/s; and with `--max-model-len 4096` and ~2×5 GB of KV you'll get only ~30 sequences before preemption — reproduce Module 7's "TP=2 is the arithmetic minimum, not a serving config" claim with a real preemption counter.

README figure: two bars per config (tok/s at C=1 and at C=64) for A/B/C, plus the efficiency line, plus the `topo.txt` excerpt.

### 5. Managed serving: Ray Serve (or KServe) vs your own gateway

**Look first (30 min).** Ray Serve's LLM integration is thin on purpose — see how thin: `python -c "from ray.serve.llm import LLMConfig, build_openai_app; help(LLMConfig)"` (after `pip install "ray[serve,llm]"`) — `LLMConfig` takes `model_loading_config`, `deployment_config` (with `autoscaling_config: {min_replicas, max_replicas, target_ongoing_requests}`), and `engine_kwargs` that are passed straight to vLLM's `AsyncLLM`. The vLLM repo has `examples/online_serving/ray_serve_deepseek.py` — 30 lines that build exactly that. Then read Ray Serve's router: the docs page "Load balancing" names the policy (power of two choices by queue length, per replica); grep the `ray/serve` package for `PowerOfTwoChoicesReplicaScheduler` to see it. There is no prefix-affinity option — that absence is the axis your comparison is about.

The deliverable is a one-page comparison with a measurement, not an opinion. Plan:

1. **Deploy the same worker under Ray Serve.** vLLM ships a Ray Serve example (`python -m vllm.entrypoints.openai.api_server` wrapped in a `@serve.deployment` with `num_replicas=2`, or the `ray-serve` LLM API `serve.llm` / `LLMServer` in recent Ray versions). On the 2060 use Qwen2.5-0.5B with 2 replicas of Project-2 mini-vllm on CPU if VRAM is short; the comparison is about the layer above the engine.
2. **Run the Module-11-item-2 trace** against (a) your Project-3 gateway → workers and (b) Ray Serve's HTTP proxy → replicas, at 1×, 2×, 4× arrival rate. Record TTFT p50/p99, throughput, and the *proxy overhead*: TTFT at C=1 minus the engine's own TTFT (Ray's proxy → replica hop adds a serialization + actor call; measure it, expect 1–5 ms).
3. **Compare on the axes that matter for LLM serving**, in a table: routing (Ray: power-of-two-choices by ongoing requests, no prefix affinity unless you write a custom router; yours: consistent-hash prefix affinity — show the hit-rate difference on a prefix-heavy trace); autoscaling signal (Ray: `target_ongoing_requests` per replica; yours: queue depth / KV util via HPA custom metrics — argue which is the better proxy for GPU load and when); streaming (both fine; note any buffering); deployment/upgrade (Ray: `serve deploy` config, rolling; yours: K8s rollout); failure handling (kill a replica under load in both — Project 3 M7's chaos test — and time the recovery); observability (Ray dashboard vs your Prometheus); operational surface (a Ray cluster is another distributed system to run; KubeRay if on K8s).
4. **Verdict paragraph:** for a small team with one model and a prefix-heavy workload, when does the custom gateway's affinity routing pay for its maintenance, and at what point (multi-model, many replicas, non-LLM stages in the pipeline) would you move to Ray Serve/KServe. Cite the numbers from step 2.

### 6. Cold starts: measure, then attack each phase

**Look first (20 min).** The four phases each have a real place they are visible:

```bash
POD=$(kubectl get pods -l app=vllm -o name | head -1)          # the first pod carrying your vLLM label (adjust the label to your manifest)
kubectl describe $POD | sed -n '/Events:/,$p'                 # the Events table at the bottom: "Pulling image" → "Pulled" → "Created" → "Started" with timestamps — phases 1–2
kubectl logs $POD | grep -E "Initializing|Loading weights took|Model loading took|Graph capturing finished|Starting vLLM"   # phases 2–4 from vLLM's own log lines
kubectl explain pod.spec.containers.readinessProbe             # the field's documentation: what "ready" means to the Service, and the timing knobs (initialDelaySeconds, periodSeconds)
kubectl explain pod.spec.initContainers                        # containers that run to completion BEFORE the main one — where a warm-up job can go
kubectl explain daemonset.spec.template                        # the pod template a DaemonSet runs on every node — the image pre-puller
```

In the vLLM source, the log lines you just grepped come from `vllm/model_executor/model_loader/` (`Loading weights took`), `vllm/v1/worker/gpu_model_runner.py` (`capture_model` → `Graph capturing finished in X secs`), so you know which phase each timestamp belongs to. `vllm serve --help | grep -A2 -E "load-format|enforce-eager|compilation-config"` shows the three flags that shorten phases 3 and 4.

A cold start is four sequential phases; measure each with a timestamp, on your `kind` cluster (Project 3, `08-p3-walkthrough.md`) and, if you do the GKE run, there.

```
phase                     how to measure                                              typical (8B fp16, GPU node)
1 image pull              kubectl describe pod → "Pulling image" → "Pulled" timestamps;   vLLM image ≈ 10 GB → 60–180 s from a registry,
                          or `time docker pull <image>` on a fresh node                    5–10 s if pre-pulled
2 process + CUDA init     container "Started" → first vLLM log line ("Initializing")     5–15 s (imports, CUDA context, NCCL for TP)
3 weight load             vLLM logs "Loading weights took X s"; also measure cold vs     16 GB from local NVMe ≈ 8–15 s (page cache cold),
                          warm page cache: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` 2–4 s warm; from a network volume/S3 60–300 s
                          before a run  (sync flushes pending writes; "3" drops page cache + dentries + inodes)
4 graph capture + warm-up vLLM logs "Graph capturing finished in X s"; then the first     10–40 s capture; first request TTFT 1–3× normal
                          request's TTFT vs the tenth's                                    while caches warm
```

Wrap it in a script: `cold_start.sh` deletes the pod, records `date +%s.%N` (seconds.nanoseconds since epoch — sub-second timestamps), polls `/health` and then sends one request, and prints the four durations from the pod events and log timestamps. Run it three times cold and three times warm.

**Attacking each phase:**

1. *Image pull:* pre-pull with a DaemonSet that runs the image with `sleep infinity` on every GPU node (the image is already on disk when the real pod schedules); slim the image (don't ship CUDA toolkits you don't need; multi-stage build); use a registry in the same region/VPC. Target: phase 1 → < 10 s.
2. *Process init:* not much — but for TP, NCCL init scales with GPU count; `--distributed-executor-backend mp` (multiprocessing, the default for single-node) is faster to start than `ray`.
3. *Weight load:* the big one at scale. Keep weights on local NVMe (a PVC or a hostPath cache populated once per node), not on a network filesystem pulled per pod; use safetensors (mmap, no unpickling, parallel load); in vLLM `--load-format safetensors` and, on newer versions, the Run:ai model streamer / tensorizer loaders (`--load-format runai_streamer` / `tensorizer`) that overlap read and H2D copy. On GKE, a node-level image or a hostPath cache primed by a DaemonSet. Target: ~1 GB/s from NVMe → 16 s for 8B, ~2 s warm.
4. *Graph capture / warm-up:* `--enforce-eager` skips capture (faster start, slower decode — a trade you only take for dev); otherwise limit `cuda_graph_sizes` (in `--compilation-config`) to the batch sizes you actually serve; warm the server with a handful of synthetic requests *before* it passes readiness (`readinessProbe` on `/health` plus an init job that hits `/v1/completions` once), so users never see the warm-up TTFT.
5. *Structural:* keep a **warm pool** (min replicas ≥ 1 per model, scale from 1 → N, never from 0 for the interactive tier); scale-up latency = cold start, so set the HPA's target (queue depth) to trigger *before* saturation by at least one cold-start duration — that's the concrete link between this item and the Module 8 autoscaling-signal discussion.

The README table: phase × (cold, warm, after mitigation) with the mitigation named per row. "Cold start 4m10s → 38 s: pre-pulled image (−2m30s), NVMe weight cache (−50 s), readiness-gated warm-up (first-request TTFT 2.1 s → 0.3 s)" is the sentence you want to be able to say.

**Done when** you can answer "how would you find the bottleneck?" with: *torch.profiler for the step, py-spy for the process, nsys when you need both timelines; the dashboard tells me which of queue / KV / preemption / cache-hit moved first; and here's the time it was the hash ring* — and every clause is something you did.
