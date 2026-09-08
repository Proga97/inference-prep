# Project 4 · `quantization-tradeoffs` — Milestones 1–4

This project exists to replace one sentence on your resume. The Qwen2-VL-for-Snapdragon bullet is a team result you cannot currently defend number by number; four days from now it will be your own sweep — same base checkpoint, four precision levels, two pieces of hardware, one quality eval — with every number regenerable by a script. Along the way you build the single most-asked quantization intuition in inference interviews (decode speeds up with fewer bytes, prefill does not) from measurements instead of from a blog post.

Work top to bottom. Every milestone ends with a script that writes a CSV into `results/` and a number you should check against the "Expected" range. Don't move on until the number is in range or you understand why it isn't.

**How this document is organised.** It starts with a "Concepts you need before starting" section that defines every word the rest of the document uses, from "what is a weight" upward. Read it once now and come back to it whenever a term feels fuzzy. Each milestone then has the same shape: *What it is* (the idea) → *Look first* (open the real thing — a file in llama.cpp's source, a CLI's `--help`, a JSON output — and read it before writing anything) → *Code* (every line commented so you can read it cold in three months) → *Test* → *Expected* → *Reading a failure* → *Close*. The "Look first" steps are not optional warm-ups; they are where the understanding comes from. The code afterwards is just you writing down what you saw.

---

## Concepts you need before starting

Every term below is used later without re-explanation, so this is the reference. They are ordered so that each builds on the ones before it.

### Weights, floats, and what "4 bits" means

- **Model / parameters / weights.** A language model is a very large collection of numbers that were learned during training. Each number is a **weight** (also called a parameter). Qwen2.5-1.5B has about 1.54 billion of them. They are arranged into **matrices** (2-D grids of numbers) and vectors; almost all the storage is in the matrices. Running the model ("inference") means multiplying inputs by those matrices, layer after layer.
- **Tensor.** The general word for an n-dimensional array of numbers. A vector is a 1-D tensor, a matrix is a 2-D tensor. When you see a shape like `[151936, 1536]` that means a matrix with 151,936 rows and 1,536 columns. In this project every weight lives in a named tensor such as `blk.3.attn_q.weight`.
- **A weight is a float32 number.** In training, each weight is stored as a **float32** (fp32): 32 bits arranged as 1 sign bit, 8 exponent bits and 23 mantissa bits. The exponent says roughly "how big" (which power of two), the mantissa says "exactly where between two powers of two". Think of scientific notation: `1.2345 × 10^-3`, where the mantissa is the `1.2345` part and the exponent is the `-3` part. 32 bits give about 7 significant decimal digits.
- **fp16 vs bf16.** Both use 16 bits. **fp16** (half precision) is 1 sign + 5 exponent + 10 mantissa bits: about 3 significant digits, and it cannot represent anything above 65,504. **bf16** (bfloat16) is 1 sign + 8 exponent + 7 mantissa bits: the same *range* as fp32 (it is literally the top half of an fp32) but only about 2 significant digits. Qwen's checkpoint is stored in bf16. Your GPU (an RTX 2060) has fast hardware for fp16 but not for bf16, which is why Milestone 1 converts to fp16.
- **What storing a weight in 4 bits means.** 4 bits can hold only 16 distinct values (0 to 15). You obviously cannot store `0.0123456` in 16 values directly. Instead you agree on a *rule* that maps those 16 integer codes back to real numbers, store the code (4 bits) and the rule (a few extra numbers shared by many weights). The weight you get back is not the original; it is the nearest of 16 allowed values. That gap is **quantization error**. Analogy: measuring your height with a ruler that only has marks every 10 cm — you get "170" whether you are 168 or 173.
- **Quantization.** The general name for this: replacing high-precision numbers with low-precision codes plus a reconstruction rule, to save memory and/or compute. In this project all quantization is **post-training** (done once to a finished model, no retraining) and **weight-only** (only weights are stored compressed; see "activations" below).
- **Scale and zero-point (also called min or offset).** The simplest reconstruction rule is a straight line: `weight ≈ scale × code + offset`. The **scale** is the size of one step between neighbouring codes (the distance between ruler marks); the **offset** (llama.cpp calls it `min`, other libraries call it a **zero-point**) shifts the ruler so its lowest mark lands on the smallest weight in the group. A **symmetric** scheme skips the offset and uses `weight ≈ scale × code` with signed codes (e.g. −128…127); Q8_0 does this. An **asymmetric** scheme keeps the offset; the K-quants do this. The scale is chosen per group from the largest and smallest weight in that group.
- **Block-wise quantization.** If one scale had to cover a whole matrix of millions of weights, a single huge weight would force a coarse ruler for everyone else. So weights are cut into small **blocks** (32 or 256 consecutive weights) and each block gets its own scale (and offset). More blocks = better fit = more bytes spent on scales. This is the tradeoff every format in this project is making.
- **Super-blocks (the K-quants).** A two-level version: 256 weights form a **super-block** with one fp16 scale and one fp16 min; inside it, 8 **sub-blocks** of 32 each have a small 6-bit scale and 6-bit min that are *relative to* the super-block's fp16 values. You get per-32 granularity while paying only 6 bits per sub-block scale instead of 16. You will count the bytes of this by hand in Milestone 1.
- **Dequantization.** Turning codes back into floats (`scale × code + offset`) at the moment they are needed. llama.cpp does this *inside the multiply kernel*: weights go from disk to GPU memory in their compressed form and are expanded on the fly just before they are multiplied. The compressed form is what you store and what the GPU reads; the floats exist only transiently.
- **Activations.** The numbers flowing *through* the model during inference (the input to each layer and the output of each layer), as opposed to the weights, which are fixed. "Weight-only quantization" means activations stay in fp16/fp32. "Weight-and-activation quantization" (SmoothQuant, FP8) compresses both; it is *not* what this project does, and the distinction matters for why prefill does not speed up.
- **Why quality drops.** Every quantized weight is slightly wrong. The errors are individually tiny but there are 1.5 billion of them and they pass through 28 layers, and some tensors are more sensitive than others (a small error in the attention "V" matrix goes straight into the output with nothing to soften it). Smaller models suffer more because each weight carries more of the model's knowledge. Fewer bits = larger steps on the ruler = larger error. Q8_0 (8 bits) is practically indistinguishable from the original; Q4 (4 bits) is where you can measure the damage.
- **Bits per weight (bpw).** Total stored bits divided by number of weights, *including* the scales and offsets. A "4-bit" format is never exactly 4 bpw: Q4_K is 4.5 because of the scale overhead. There is a **nominal** bpw (what the format's math says) and a **measured** bpw (`file size in bits / parameter count`), which is a bit higher because some tensors are kept at higher precision. Use the measured one in graphs.
- **Outlier channels / activation-aware quantization.** Some inputs to a matrix are consistently much larger than others ("outlier channels"). Weights that multiply large inputs deserve more careful rounding. An **importance matrix** (llama.cpp's `llama-imatrix`) measures which inputs are large by running some sample text ("**calibration** data") through the model, and `llama-quantize` uses it to bias the rounding. AWQ is the same idea in another library.
- **Residual stream.** In a transformer, each layer *adds* its output to a running vector that passes straight through the whole network (`x = x + layer(x)`). That running vector is the residual stream. A tensor that "writes directly into the residual stream" (like `ffn_down`) has no later step that could absorb its error, which is why the `_M` recipe gives it more bits.

### Model files and llama.cpp

- **llama.cpp.** An inference engine written in C/C++ (with its own tensor library, **ggml**) that runs LLMs on CPUs and GPUs. It has its own quantization formats and its own file format. You build it from source in Milestone 1.
- **GGUF.** llama.cpp's single-file model format: a header of key/value metadata (architecture, vocabulary, chat template, layer count, …) followed by every tensor, each with a name, a shape and a **type**. Analogy: a zip file where every entry says what compression it used.
- **Tensor type.** The storage format of one tensor inside a GGUF: `F32`, `F16`, `Q8_0`, `Q4_K`, `Q6_K`, … Every tensor chooses independently. A "Q4_K_M model" is a GGUF where *most* tensors are `Q4_K` and some sensitive ones are `Q6_K` or `F32`. The suffix `_S`/`_M`/`_L` (small/medium/large) is the *recipe* deciding which tensors get more bits; it is not itself a type.
- **Tokens and the tokenizer.** Models do not read characters; they read **tokens**, integer IDs for common chunks of text ("the", "ing", " quant"). The **tokenizer** converts text to token IDs and back. The **vocabulary** is the list of all tokens; Qwen's has 151,936 entries. Roughly 1 token ≈ ¾ of an English word.
- **Embedding table.** The matrix `[vocab_size, hidden_size]` = `[151936, 1536]` that turns a token ID into the model's internal vector. It is 15 % of this model's weights. **Tied embeddings** means the same matrix is reused at the very end to turn the model's output vector back into scores over the vocabulary, so there is no separate `output.weight`.
- **Logits, softmax, next-token distribution.** At each position the model produces one raw score per vocabulary entry (151,936 numbers); those are the **logits**. **Softmax** turns them into probabilities that sum to 1 — the model's **next-token distribution**. Everything about "quality" in this project is about how much quantization changes that distribution.
- **Sampling, greedy decoding, temperature, seed.** To pick the next token you either take the single most probable one (**greedy**, also "temperature 0" or `top_k=1`) or draw randomly according to the probabilities (**sampling**; **temperature** scales how random). A **seed** fixes the random generator so a run can be repeated exactly. Every quality eval here is greedy so that the only variable is the weights.
- **Chat template and system prompt.** Instruct models expect a conversation wrapped in special tokens (`<|im_start|>system … <|im_end|>`). The **chat template** is the rule (stored in the GGUF as a Jinja string) for building that wrapping. The **system prompt** is the first message that sets instructions for the model. If the template is missing or wrong the model still produces text, but worse text — a classic silent failure.
- **Context length (`-c`).** The maximum number of tokens (prompt + generated) the model can hold at once. It determines the size of the KV cache below.

### How inference spends time

- **Prefill.** The first phase of a request: the whole prompt (say 512 tokens) goes through the model in one pass. Every weight is read from memory once and used for all 512 tokens. Lots of arithmetic per byte read.
- **Decode.** The second phase: generating the answer one token at a time. Each step reads *every weight once* to produce *one* token. Very little arithmetic per byte read. `pp512` and `tg128` in `llama-bench` mean "prefill 512 tokens" and "text-generate 128 tokens".
- **KV cache.** During attention, every token produces a **key** vector and a **value** vector per layer. Instead of recomputing them for all previous tokens at every step, they are stored — that store is the KV cache. It grows by a fixed number of bytes per token (28 KB for this model at fp16 — you will derive it) and lives in GPU memory next to the weights. Its size does not depend on how the weights are quantized.
- **GQA (grouped-query attention) / KV heads.** Attention is computed by several parallel "heads". In GQA, several query heads share one key/value head, so the KV cache is smaller. Qwen2.5-1.5B has 12 query heads and 2 KV heads; the KV math uses the 2.
- **Batching.** Processing several sequences (or several tokens) in the same pass so the weights, once read, are used more than once. Prefill is a batch of 512 tokens from one prompt; a server with many users batches across users. Batch size 1 decode is the worst case for hardware use and the case this project measures.
- **Latency vs throughput.** **Latency** is how long one thing takes (seconds per token, seconds to first token). **Throughput** is how much gets done per second across everything (tokens per second, requests per second). At batch 1 they are the same number turned upside down; at higher batch they diverge — throughput rises while each request's latency gets worse.
- **TTFT / ITL.** **Time to first token** ≈ prefill time. **Inter-token latency** ≈ one decode step. This project reports their inverses as tokens/sec.
- **Tokens per second (tok/s).** For prefill: prompt tokens ÷ prefill time. For decode: generated tokens ÷ generation time. Higher is better. Always say which one you mean.
- **FLOP.** One floating-point operation (a multiply or an add). A matrix multiply of a `[1, 1536]` vector by a `[1536, 8960]` matrix costs about 2 × 1536 × 8960 FLOPs (one multiply and one add per weight). A whole forward pass costs ≈ 2 × parameters FLOPs *per token*, so 3 GFLOP per token for this model. GPUs are quoted in TFLOPS (10¹² FLOP per second).
- **Memory bandwidth.** How many bytes per second the processor can pull from its memory. RTX 2060: 336 GB/s from its GDDR6. A desktop CPU with dual-channel DDR4: 35–50 GB/s from RAM. This is the ceiling on decode speed.
- **"Bound" (memory-bound vs compute-bound).** Whichever resource runs out first sets the speed. If a step needs to move 3 GB of weights and only do 3 GFLOP, the memory system is the bottleneck: it is **memory-bound** — faster arithmetic would not help, fewer bytes would. If a step needs 1.6 TFLOP but only 3 GB of reads, the arithmetic units are the bottleneck: **compute-bound** — fewer bytes would not help. Decode at batch 1 is memory-bound; prefill of 512 tokens is compute-bound. Weight quantization reduces bytes, not FLOPs, so it speeds up the first and not the second. That sentence is the project.
- **Arithmetic intensity and the roofline.** Arithmetic intensity = FLOPs performed per byte moved. The **roofline** is a chart with intensity on the x-axis and achieved speed on the y: a rising line (bandwidth × intensity) on the left, a flat line (peak FLOPS) on the right; the corner is where a workload flips from memory-bound to compute-bound. Decode sits far left, prefill sits right of the corner. You do not draw one here but you are measuring both ends of it.
- **Model bandwidth utilization (MBU).** `bytes of weights read per token × tokens per second ÷ hardware bandwidth`. It is the fraction of the memory system's ceiling you are actually achieving during decode. 100 % is impossible; 60–80 % is good; it falls at low bpw because fixed costs start to dominate.
- **Fixed per-step overhead / kernel launch.** A **kernel** is one function that runs on the GPU. Each decode step launches hundreds of them (every layer has several), and each launch costs a few microseconds regardless of how much data it touches. Plus the CPU side has to pick the next token and feed it back. These costs do not shrink when weights shrink, which is why Q4 does not decode 3× faster than F16 even though it reads 3× fewer bytes.
- **Tensor cores, compute capability, sm_75.** Tensor cores are special GPU units that do small matrix multiplies very fast in fp16/int8. NVIDIA numbers GPU generations by **compute capability** (CC); Turing (RTX 20xx) is CC 7.5, written `sm_75` in the compiler. Code compiled for one CC does not run on another, hence the build flag.
- **cuBLAS, MMQ, `dp4a`, flash attention.** Four kernel paths you will see named in logs. **cuBLAS** is NVIDIA's matrix-multiply library (used for fp16 prefill). **MMQ** ("mul-mat quantized") is llama.cpp's own kernel that multiplies quantized weights by int8-converted activations on tensor cores (quantized prefill). **`dp4a`** is a GPU instruction that does four int8 multiplies-and-adds at once (used by the decode kernel `mul_mat_vec_q`). **Flash attention** is an attention kernel that avoids writing the big attention-score matrix to memory; `-fa 1` turns it on.
- **VRAM.** The GPU's own memory (6 GB on the 2060). Weights, the KV cache and temporary activation buffers all have to fit. **Compute buffer** = llama.cpp's scratch space for activations. **CUDA context** = ~300 MB the driver takes just to run anything. **Sysmem fallback** = a Windows-driver feature that silently spills VRAM overflow into system RAM (making things 10× slower instead of failing loudly).
- **Threads, physical cores, SMT/hyperthreading.** A CPU with 6 physical cores may show 12 logical processors because each core can run two threads (**SMT**). For a memory-bound loop the second thread on a core adds nothing (the core is waiting on memory, not busy), so the best `-t` is usually the physical count.
- **mmap and the page cache.** **mmap** (memory-map) lets a program treat a file on disk as if it were already in memory; the OS loads pages on first touch. The **page cache** is RAM the OS uses to keep recently read file contents, so a second load of the same model is served from RAM, not disk — that is a **warm** load; a **cold** load is after a reboot or after dropping the caches. **RSS** (resident set size) is a process's own memory use; with mmap the model is in the page cache, not the RSS, so RSS understates "RAM needed".

### Measuring quality

- **Perplexity (PPL).** Feed the model a text it has never seen; at each token ask "how much probability did you assign to the token that actually came next?" Perplexity is `exp(average of −log p)` over all tokens: the average "surprise", expressed as an effective number of equally likely choices. Lower is better. PPL 10 means "on average the model was as unsure as if it were choosing among 10 options". The absolute value depends on the text; the *change* between F16 and a quantized model on the *same* text is what you report.
- **KL divergence (KLD).** A number that says how different one probability distribution is from another. Here: at each token position, compare the quantized model's next-token distribution with the F16 model's distribution *at the same position with the same input*. 0 means identical; 0.01 is a small change; 0.1 is noticeable. Averaged over ~10,000 positions it is a very precise measure of "how much did quantization change what the model thinks", independent of whether any particular answer changed. **Same top-1** (`Same top p` in llama.cpp's output) is the fraction of positions where both models would pick the same most-likely token.
- **Exact-match accuracy.** Ask a question with a known numeric answer, take the model's final number, count it correct only if it matches exactly. GSM8K is a set of grade-school math word problems with such answers. It measures what a user would notice; it is also coarse and noisy.
- **Agreement with F16.** For each question, did the quantized model give the *same* final answer as the F16 model (right or wrong)? This is a **paired** comparison: it cancels out the question-difficulty noise that makes absolute accuracy jumpy, so it is much more sensitive.
- **Mean vs median.** The **mean** is the sum divided by the count; one very slow outlier run drags it. The **median** is the middle value when sorted; an outlier cannot move it. For benchmark repetitions, report the median and show the spread.
- **Standard deviation (stddev).** How spread out repeated measurements are around their mean. `llama-bench` prints `mean ± stddev`. A stddev over ~5 % of the mean means something else was competing for the hardware.
- **Standard error (SE) and confidence.** When you estimate a percentage from *n* trials, the estimate itself is noisy. The **standard error** for an accuracy `p` on `n` questions is `√(p(1−p)/n)`; at p = 0.65 and n = 50 that is ≈ 6.7 percentage points. A 95 % confidence interval is roughly ±2 SE, so ±13 points. Two models whose scores differ by less than that on 50 questions cannot be told apart by that test. You will state this in the README.
- **Determinism.** Same inputs, same weights, same settings → byte-identical outputs. Floating-point addition is not associative, so if the *order* of additions changes (different batch composition, different kernel) the last digit changes, and greedy decoding can then flip a token. You prove determinism by running the F16 eval twice and diffing.
- **Calibration / contamination.** If any text you use to *tune* a quantization (the importance matrix) overlaps with text you *evaluate* on, the eval is contaminated and the number is optimistic. Keep them disjoint.

---

## What you are building

The finished repo:

```
quantization-tradeoffs/
├── README.md                  # the tradeoff table, two graphs, "what I'd ship" section
├── scripts/
│   ├── build_llamacpp.sh      # M1: clone + two builds (CUDA, CPU-only)
│   ├── get_model.sh           # M1: download HF checkpoint, convert to f16 GGUF
│   └── quantize_all.sh        # M1: f16 → Q8_0 / Q5_K_M / Q4_K_M
├── models/                    # gitignored; models/MANIFEST.md records sizes + sha256
├── bench/
│   ├── run_bench.py           # M2: llama-bench sweep → results/bench.csv
│   ├── memory_probe.py        # M2: load time + VRAM via llama-server → results/memory.csv
│   ├── membw.py               # M2: rough host-RAM bandwidth number for the CPU ceiling
│   └── plot.py                # M4: the two graphs
├── eval/
│   ├── build_prompts.py       # M3: 50 fixed GSM8K questions → eval/prompts.jsonl (committed)
│   ├── prompts.jsonl
│   ├── run_eval.py            # M3: greedy answers per level → results/eval_<level>.jsonl
│   ├── score.py               # M3: accuracy, agreement with f16 → results/quality.csv
│   └── perplexity.sh          # M3: llama-perplexity PPL + KL divergence → results/ppl.csv
└── results/                   # every CSV/JSON the README cites, plus results/plots/*.png
```

**Model decision.** Primary: **Qwen2.5-1.5B-Instruct** (text). Reasons: the quality eval is exact-match arithmetic, which is unambiguous to score and sensitive to quantization (math is where int4 hurts first, so you will actually see a signal); `llama-bench` benchmarks text models cleanly and has no image path; fp16 (3.1 GB) fits the 2060 with room for KV, so you get a real fp16 baseline on GPU; and 1.5B is big enough that the bandwidth math is not swamped by fixed overhead the way it is at 0.5B. If you want the resume bullet to literally say "Qwen2-VL", do the text sweep first and then the VL section at the end of M3 as an add-on day; you will reuse every script.

**Hardware facts you will use constantly.** RTX 2060: 6 GB GDDR6, **336 GB/s**, Turing **sm_75**. Turing has fp16 and int8 tensor cores but **no bf16** support, so the fp16 baseline must be F16, not BF16 (llama.cpp will run BF16 on Turing through a slow path — do not benchmark it). CUDA 12.x and 13.x both still support sm_75 (CUDA 13 dropped Maxwell/Pascal/Volta; Turing survived). Your CPU-only runs are limited by host RAM bandwidth: dual-channel DDR4-3200 is 51 GB/s theoretical (2 channels × 8 bytes × 3200 million transfers/s), ~35–40 achievable; DDR5-5600 roughly 90 theoretical, ~60–70 achievable. You will measure yours in M2.

**Time budget (~4 days).** Day 1: M1 (the CUDA build alone can take 20–40 min; start it first and do the download while it compiles). Day 2: M2. Day 3: M3. Day 4: M4 — graphs, README, the out-loud narrative. Add one day if you do the VL variant.

---

## Step 0 — Environment (60–90 min, mostly waiting)

### 0a. WSL2 (recommended)

**WSL2** (Windows Subsystem for Linux, version 2) runs a real Linux kernel in a lightweight VM on Windows, and the NVIDIA Windows driver already contains the WSL2 CUDA shim; you install **only the toolkit** inside WSL, never a driver. On the Windows side, update to a current Game Ready/Studio driver, then in PowerShell (admin) `wsl --install -d Ubuntu-24.04` (`-d` = which distribution to install). Inside Ubuntu, `nvidia-smi` (NVIDIA's command-line GPU status tool) should already print your 2060 (it lives at `/usr/lib/wsl/lib/nvidia-smi`). If it does not, the Windows driver is too old.

Install the toolkit using NVIDIA's WSL-Ubuntu repo (the `cuda-toolkit-12-x` package, **not** `cuda` or `cuda-drivers` — those try to install a Linux driver, which must not exist inside WSL):

```bash
# apt update: refresh the package index.  apt install -y: install without asking.
#   build-essential  → gcc/g++/make, needed to compile llama.cpp
#   cmake            → the build-system generator llama.cpp uses
#   git              → to clone llama.cpp
#   python3-venv / python3-pip → isolated Python env for the converter and the harness
#   curl unzip       → for downloading wikitext and the optional prebuilt binaries
sudo apt update && sudo apt install -y build-essential cmake git python3-venv python3-pip curl unzip
# CUDA toolkit for WSL — follow the "WSL-Ubuntu" tab on developer.nvidia.com/cuda-downloads,
# ending with something like:
sudo apt install -y cuda-toolkit-12-8          # compiler (nvcc) + libraries (cuBLAS etc.), NO driver
# Put nvcc on PATH permanently: append the export to ~/.bashrc and reload it for this shell.
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
# Both must succeed: nvcc is the CUDA compiler; nvidia-smi proves the driver shim sees the 2060.
nvcc --version && nvidia-smi
```

Two WSL-specific rules that affect your numbers:

1. **Keep the repo and the models on the ext4 filesystem** (`~/`), not under `/mnt/c/`. The 9P bridge to NTFS (the protocol WSL uses to reach Windows drives) is slow enough to make "load time" a measurement of the filesystem rather than of llama.cpp.
2. **Raise the WSL memory and CPU limits.** WSL2 defaults to half your RAM; CPU-only fp16 needs the 3.1 GB model plus page cache plus headroom. Create `C:\Users\<you>\.wslconfig`:
   ```
   [wsl2]
   memory=24GB        # RAM the WSL VM may use (default: 50 % of host)
   processors=12      # logical CPUs the VM sees (default: all; set explicitly so it is known)
   ```
   then `wsl --shutdown` from PowerShell and reopen. Set `processors` to your physical core count × 2 (all hardware threads); you will control the thread count llama.cpp uses with `-t`.

### 0b. Native Windows (if you must)

It works, with friction. CUDA on Windows requires MSVC (Microsoft's C++ compiler; no MinGW). Install **Visual Studio 2022 Build Tools** with the "Desktop development with C++" workload **before** the CUDA toolkit, so the toolkit installs its VS integration. Build from the "x64 Native Tools Command Prompt for VS 2022":

```bat
:: cmake -B build                        → configure into a directory named "build"
::   -G "Visual Studio 17 2022" -A x64   → generate an MSVC project for 64-bit
::   -DGGML_CUDA=ON                      → compile the CUDA backend
::   -DCMAKE_CUDA_ARCHITECTURES=75       → only build kernels for Turing (sm_75) = your 2060
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
:: --build build      → compile what was configured;  --config Release → optimised, no debug checks
:: -j                 → use all CPU cores for compilation
cmake --build build --config Release -j
```

Binaries land in `build\bin\Release\` (not `build\bin\`). Everything else in this document uses forward-slash paths and `$LLAMA_BIN`; on Windows set the same variable in PowerShell. Three caveats that bite measurements specifically: (a) Windows Defender scans multi-GB files on first open — exclude the `models/` folder or your load times will be garbage; (b) the NVIDIA driver's **"CUDA – Sysmem Fallback Policy"** defaults to spilling VRAM overflow into system RAM silently, so an oversized context does not OOM (out-of-memory error), it just makes decode 10× slower — set it to "Prefer No Sysmem Fallback" in NVIDIA Control Panel → Manage 3D settings so that an overflow is loud (this may also apply under WSL2 since the driver is the same; if tok/s ever collapses instead of erroring, this is why); (c) the desktop compositor holds 300–800 MB of VRAM at all times, so subtract a baseline reading in every VRAM number.

The zero-build fallback is the prebuilt `llama-<build>-bin-win-cuda-12.4-x64.zip` plus the matching `cudart-llama-bin-win-cuda-12.4-x64.zip` from llama.cpp's GitHub releases. Use it only if the build is genuinely blocked; "I built llama.cpp with CUDA for sm_75" is part of the story.

### 0c. Python environment

```bash
# Project root. -p: create parents, no error if it exists.
mkdir -p ~/quantization-tradeoffs && cd ~/quantization-tradeoffs
# New git repo + the directory layout from "What you are building".
git init && mkdir -p scripts models bench eval results/plots
# .gitignore: ignore every model file (multi-GB) EXCEPT the manifest; ignore the venv,
# Python bytecode caches, and the *.kld logits dump from M3 (it is ~3 GB).
printf 'models/*\n!models/MANIFEST.md\n.venv/\n__pycache__/\n*.kld\n' > .gitignore
# Isolated Python environment so the converter's pinned deps don't fight system packages.
python3 -m venv .venv && source .venv/bin/activate
#   huggingface_hub → `hf download`;  gguf → GGUFReader + gguf-dump;  requests → talk to llama-server
#   pandas/matplotlib → tables and plots;  datasets → GSM8K;  tabulate → pandas .to_markdown() needs it
pip install --upgrade pip huggingface_hub gguf requests pandas matplotlib datasets tabulate
```

---

## Milestone 1 — Build, convert, quantize (Day 1)

### What it is

**llama.cpp** is a C/C++ inference engine whose model format, **GGUF**, is a single file containing metadata (architecture, tokenizer, chat template, hyperparameters) plus every tensor, each tagged with its own storage type. That per-tensor type is the whole game: a "Q4_K_M model" is not a model where every weight is 4-bit; it is a file where most large matrices are Q4_K and a few sensitive ones are Q6_K, with the norms in f32. You will inspect this yourself in the test.

**Why you build it rather than pip-install it.** The CUDA backend is compiled for specific GPU architectures, and the fast paths differ by generation. For sm_75 the things that matter: quantized matmuls (matrix multiplications) for prefill use **MMQ** kernels that do int8 dot products on Turing's tensor cores (`INT8_MMA_AVAILABLE` is defined for CC ≥ 7.5, so your 2060 gets them); the decode path uses `mul_mat_vec_q` (matrix × single vector, quantized), which dequantizes on the fly with `dp4a` integer instructions; fp16 prefill goes through cuBLAS on fp16 tensor cores; flash attention uses fp16 `mma` (matrix-multiply-accumulate) and works from Volta up. Pinning `-DCMAKE_CUDA_ARCHITECTURES=75` also cuts compile time roughly in half by not building kernels for six other architectures.

**Why two builds.** A llama.cpp binary built with CUDA will, even at `-ngl 0` (zero layers offloaded to the GPU), still offload large-batch matrix multiplications to the GPU during prompt processing (the backend treats it as a helper for batches ≥ 32 tokens). Your "CPU-only" pp512 number would be partly a GPU number. The clean fix is a second, CUDA-free build directory. This is also a good interview aside: "I noticed my CPU prefill numbers were too good and traced it to the CUDA backend's batch offload." (You will find the exact line in "Look first".)

**What the quant types are.** All llama.cpp quantization is **weight-only, block-wise, post-training, no retraining**: weights are stored as small integers plus per-block scale factors, and are converted back to fp16/fp32 inside the kernel just before the multiply. Activations stay fp16/fp32 throughout (on CPU and in MMQ they are quantized to int8 on the fly for the dot product, but that is a kernel detail, not a stored format). Study Module 5's distinction: this is the family that helps memory-bound decode; it does not use int8 compute the way SmoothQuant/FP8 does.

- **Q8_0**: blocks of 32 weights, each an int8 plus one fp16 scale for the block. 32×8 + 16 = 272 bits per 32 weights = **8.5 bits/weight**. Effectively lossless.
- **K-quants** (Q4_K, Q5_K, Q6_K): **super-blocks of 256** weights, split into 8 sub-blocks of 32. Each sub-block has its own 6-bit scale and 6-bit minimum, and the super-block has one fp16 scale and one fp16 min that those 6-bit values are expressed relative to. The two-level scheme is the point: fine-grained scales (per 32) cost almost nothing because they are themselves quantized against the super-block. Q4_K: 256×4 + 16×6 + 2×16 = 1152 bits / 256 = **4.5 bpw**. Q5_K: **5.5 bpw**. Q6_K uses 8-bit scales: (256×6 + 16×8 + 16)/256 = **6.56 bpw**.
- **The `_S` / `_M` / `_L` suffix** is a *recipe*, not a type: which tensors get bumped to a higher-precision type. `Q4_K_M` uses Q4_K for most matrices but Q6_K for the attention **V** projection and for roughly half of the `ffn_down` matrices (the early and late layers plus every third in the middle), and Q6_K for the output head. `Q4_K_S` uses Q4_K nearly everywhere. The rationale, from the original K-quants work: V errors pass straight into the attention output as a weighted sum with no softmax to absorb them, and `ffn_down` writes directly into the residual stream. The exact rules are `llama_tensor_get_type` in `src/llama-quant.cpp` — about 200 lines, read it once (that is "Look first" step 2).

Published effective bit rates for the recipes (the file sizes below follow from these): Q4_K_M ≈ 4.83 bpw, Q5_K_M ≈ 5.67, Q8_0 = 8.50, F16 = 16.

**Expected file sizes for Qwen2.5-1.5B-Instruct.** It has 1.54 B parameters, of which the embedding table (151,936 × 1,536 ≈ 233 M) is 15 %; `tie_word_embeddings` is true, so there is no separate output matrix. Bytes ≈ params × bpw / 8:

| Level | bpw | Predicted | Expect on disk |
|---|---|---|---|
| F16 | 16 | 3.09 GB | ~3.1 GB |
| Q8_0 | 8.5 | 1.64 GB | 1.6–1.7 GB |
| Q5_K_M | 5.67 | 1.09 GB | 1.1–1.25 GB |
| Q4_K_M | 4.83 | 0.93 GB | 0.95–1.1 GB |

The on-disk number runs a little above the prediction because norms stay f32, the embedding may be kept at a higher type than the recipe's default, and metadata/tokenizer add a few MB. Compute your *measured* bpw as `file_bytes × 8 / n_params` and use that on the x-axis of every graph; it is the honest number.

### Look first

Do these in order. Steps 1–2 need only the cloned source (start them while the CUDA build compiles); steps 3–6 need the downloaded checkpoint and the f16 GGUF, so do them after `get_model.sh` and *before* `quantize_all.sh`. Each one ends with something you should write down in a scratch file, because the "Expected" section asks you to predict numbers before you measure them.

**1. The block layouts — count the bytes by hand.** Open `llama.cpp/ggml/src/ggml-common.h` and search for `block_q8_0`, `block_q4_K`, `block_q5_K`, `block_q6_K`. You will find C structs like:

```c
#define QK8_0 32
typedef struct {
    ggml_half d;          // delta: the block's fp16 scale        → 2 bytes
    int8_t   qs[QK8_0];   // quants: 32 signed 8-bit codes        → 32 bytes
} block_q8_0;             // 34 bytes per 32 weights = 8.5 bits/weight; NO min → symmetric

#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct {
    union { struct { ggml_half d; ggml_half dmin; }; ggml_half2 dm; };  // super-block scale + min → 4 bytes
    uint8_t scales[K_SCALE_SIZE];   // 8 sub-block scales + 8 sub-block mins, 6 bits each = 96 bits → 12 bytes
    uint8_t qs[QK_K/2];             // 256 4-bit codes packed two per byte → 128 bytes
} block_q4_K;                       // 144 bytes per 256 weights = 4.5 bits/weight; HAS a min → asymmetric
```

Notice: (a) `Q8_0` has a scale `d` but no minimum — a symmetric scheme, `w ≈ d × q`; the K-quants carry `dmin` too — asymmetric, `w ≈ d·scale_sub × q − dmin·min_sub`. (b) Right below each struct is a `static_assert(sizeof(block_q4_K) == …)`; that assert is the format's bpw written as C. (c) Do the same count for `block_q5_K` (extra `qh[32]` holds the 5th bit of each weight → 176 bytes → 5.5 bpw) and `block_q6_K` (`ql[128]` + `qh[64]` + `int8_t scales[16]` + `d` → 210 bytes → 6.5625 bpw). Write all four bpw numbers down; the manifest will reproduce them. This tiny script does the same arithmetic so you can check yourself:

```python
#!/usr/bin/env python3
"""Reproduce the bits-per-weight of each ggml block type from its struct layout
(ggml/src/ggml-common.h). Every entry is (bytes per block, weights per block)."""
# Q8_0:  2-byte fp16 scale + 32 int8 codes                         = 34 bytes / 32 weights
# Q4_0:  2-byte fp16 scale + 32 codes × 4 bits packed (16 bytes)   = 18 bytes / 32 weights  (older, 1-level format)
# Q4_K:  2 (d) + 2 (dmin) + 12 (16 six-bit sub-scales/mins) + 128  = 144 bytes / 256 weights
# Q5_K:  as Q4_K + 32 bytes of "high bits" (the 5th bit of each of 256 weights)
# Q6_K:  128 (low 4 bits) + 64 (high 2 bits) + 16 int8 sub-scales + 2 (d) = 210 bytes / 256 weights
BLOCKS = {
    "Q8_0": (2 + 32,                 32),
    "Q4_0": (2 + 32 // 2,            32),
    "Q4_K": (2 + 2 + 12 + 256 // 2,  256),
    "Q5_K": (2 + 2 + 12 + 256 // 2 + 256 // 8, 256),
    "Q6_K": (256 // 2 + 256 // 4 + 16 + 2,     256),
}
for name, (nbytes, nweights) in BLOCKS.items():
    # bits per weight = (bytes per block × 8) / weights per block
    print(f"{name}: {nbytes:4d} bytes / {nweights} weights = {nbytes * 8 / nweights:.4f} bpw")
```

Expected output: Q8_0 8.5000, Q4_0 4.5000, Q4_K 4.5000, Q5_K 5.5000, Q6_K 6.5625. Q4_0 and Q4_K cost the same bits — the K-quant spends them differently (a min per sub-block, sub-scales relative to a super-block) and that is why it is more accurate at the same size.

**2. The `_M` recipe in source.** `grep -n "llama_tensor_get_type" llama.cpp/src/llama-quant.cpp` and read the function top to bottom. Find and note four things:

- The lambda `use_more_bits(i_layer, n_layers)`: `return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8) % 3 == 2;`. For `n_layers = 28`: layers 0–2 (first eighth), 24–27 (last eighth), and every third layer in the middle (5, 8, 11, 14, 17, 20, 23) get "more bits". Write out the list — you will check it against `gguf-dump` after quantizing.
- The `attn_v.weight` branch: for `Q4_K_M`/`Q5_K_M` the tensor is bumped to `Q6_K` when `use_more_bits(...)` is true; note also any `n_gqa()` clauses in that branch (Qwen2.5-1.5B has 12 query heads and 2 KV heads, so `n_gqa() = 6`), since with GQA the V matrix is tiny (`[1536, 256]`) and extra bits cost almost nothing. Read which condition actually fires for this model and predict: which layers will show `attn_v.weight` as `Q6_K`?
- The `ffn_down` branch: for `Q4_K_M`, `Q6_K` when `use_more_bits(i_layer, n_layer)`, else `Q4_K`.
- The very first `if` in the function: tensors named `output.weight` — *or `token_embd.weight` when the model has no separate output tensor* (`!qs.has_output`) — are treated as the output head and get `Q6_K` under Q4_K_M. Qwen2.5-1.5B ties its embeddings, so predict `token_embd.weight → Q6_K` in the Q4_K_M file. That single rule explains most of the "measured bpw is above nominal" drift.

**3. The Hugging Face config.** After `get_model.sh` has downloaded the checkpoint: `cat models/Qwen2.5-1.5B-Instruct/config.json`. Note `hidden_size` (1536), `intermediate_size` (8960), `num_hidden_layers` (28), `num_attention_heads` (12), `num_key_value_heads` (2), `vocab_size` (151936), `tie_word_embeddings` (true), `torch_dtype` (bfloat16), `max_position_embeddings` (32768). Derive by hand: `head_dim = 1536/12 = 128`; the KV bytes per token at fp16 = 2 (K and V) × 28 layers × 2 KV heads × 128 × 2 bytes = 28,672 ≈ 28 KB; and the parameter count: embedding 151936×1536 = 233.4 M, per layer q/o 2×1536² + k/v 2×1536×256 + gate/up/down 3×1536×8960 ≈ 46.8 M, ×28 = 1.31 B, total ≈ 1.54 B. You will see this exact count from the GGUF reader in the manifest.

**4. The converter's Qwen2 class.** `grep -n "class Qwen2Model" llama.cpp/convert_hf_to_gguf.py` and read the class: the decorator `@ModelBase.register("Qwen2ForCausalLM", …)` is how the script picks a class from `config.json`'s `architectures` field; `model_arch = gguf.MODEL_ARCH.QWEN2` names the GGUF architecture; `set_gguf_parameters` copies the hyperparameters into GGUF metadata keys (`qwen2.block_count`, `qwen2.attention.head_count_kv`, …). The tensor *renaming* is not in that class — it is the table `TensorNameMap` in `llama.cpp/gguf-py/gguf/tensor_mapping.py`. Find these rows and write the mapping down: `model.embed_tokens` → `token_embd`; `model.layers.{bid}.self_attn.q_proj` → `blk.{bid}.attn_q` (likewise `k_proj`→`attn_k`, `v_proj`→`attn_v`, `o_proj`→`attn_output`); `mlp.gate_proj`→`ffn_gate`, `mlp.up_proj`→`ffn_up`, `mlp.down_proj`→`ffn_down`; `input_layernorm`→`attn_norm`, `post_attention_layernorm`→`ffn_norm`; `model.norm`→`output_norm`; `lm_head`→`output` (absent for this model — tied). Every name you will grep from `gguf-dump` is on that list.

**5. `gguf-dump` on the f16 file.** After conversion:

```bash
# gguf-dump comes from the `gguf` pip package (same code as llama.cpp/gguf-py/gguf/scripts/gguf_dump.py).
# It prints the metadata key/values first, then one line per tensor: index | element count | shape | type | name.
gguf-dump models/qwen2.5-1.5b-instruct-f16.gguf | less
```

Read the metadata block: `general.architecture = qwen2`, `qwen2.block_count = 28`, `qwen2.embedding_length = 1536`, `qwen2.attention.head_count = 12`, `qwen2.attention.head_count_kv = 2`, `tokenizer.ggml.tokens` (151,936 entries), `tokenizer.chat_template` (a Jinja string with `<|im_start|>`). Then the tensor list. Notice three things: (a) **GGUF prints shapes innermost-dimension first**, so the embedding shows as `1536, 151936` while PyTorch/HF says `[151936, 1536]` — same tensor, reversed order; (b) every matrix is `F16` and every `*_norm.weight` is `F32` (norms are tiny 1-D vectors; nobody bothers compressing them), and the `attn_q/k/v.bias` vectors are F32 too; (c) count the tensors: 28 layers × 12 tensors (q/k/v/o weights, q/k/v biases, attn_norm, ffn_norm, gate/up/down) = 336, plus `token_embd.weight` and `output_norm.weight` = 338 — and *no* `output.weight`. Save the shape of `blk.0.attn_v.weight` (`1536, 256`) and `blk.0.ffn_down.weight` (`8960, 1536`) for the next step.

**6. The batch-offload line behind "why two builds".** `grep -n "min_batch_size" llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`. The function `ggml_backend_cuda_device_offload_op` returns true for matrix multiplies whose batch dimension is `>= min_batch_size` (32). That is the rule that would quietly route your `-ngl 0` prefill through the GPU if you used the CUDA binary — the whole reason for `build-cpu`.

### Code

`scripts/build_llamacpp.sh`:

```bash
#!/usr/bin/env bash
# Build llama.cpp twice from one clone: a CUDA build pinned to sm_75 and a pure-CPU build.
# -e: exit on first error; -u: unset variables are errors; -o pipefail: a failing command in a pipe fails the pipe.
set -euo pipefail
# Always run from the repo root regardless of where the script was invoked from
# ($0 = this script's path; dirname = its folder; .. = the project root).
cd "$(dirname "$0")/.."
# Clone once; skip if the directory already exists so re-running the script is cheap.
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
# Record the exact commit you built (short hash %h and commit date %cs) — the README cites it.
# tee: print to the terminal AND write to the file.
git log -1 --format='llama.cpp commit %h (%cs)' | tee ../results/llamacpp_version.txt

# CUDA build, pinned to Turing (sm_75). 20–40 minutes the first time.
#   -B build-cuda                    → configure into this directory (separate from the CPU build)
#   -DGGML_CUDA=ON                   → compile the CUDA backend (MMQ, mul_mat_vec_q, flash-attn kernels)
#   -DCMAKE_CUDA_ARCHITECTURES=75    → only generate sm_75 kernels: halves compile time, matches the 2060
#   -DCMAKE_BUILD_TYPE=Release       → optimised (-O3), no debug asserts; never benchmark a Debug build
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
#   --build build-cuda → compile;  --config Release → same as above for multi-config generators
#   -j"$(nproc)"       → parallel jobs = number of CPUs (nproc prints that number)
cmake --build build-cuda --config Release -j"$(nproc)"

# CPU-only build for honest -ngl 0 numbers. GGML_NATIVE=ON (default) gives -march=native → AVX2/AVX-512,
# i.e. the compiler emits the fastest vector instructions YOUR CPU supports (the binary is not portable).
#   -DGGML_CUDA=OFF → no CUDA backend at all, so no batch offload can happen (see "Look first" step 6)
cmake -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release -j"$(nproc)"

# Sanity check: the six binaries this project uses must exist.
# grep -E = extended regex; ^…$ anchors so e.g. "llama-bench" matches but "llama-bench-matmult" does not.
ls build-cuda/bin | grep -E '^llama-(bench|quantize|perplexity|server|cli|imatrix)$'
```

Put these in your shell profile; every later script assumes them:

```bash
# Directory holding the CUDA binaries (llama-bench, llama-server, llama-quantize, …).
export LLAMA_BIN=$HOME/quantization-tradeoffs/llama.cpp/build-cuda/bin
# Directory holding the CPU-only binaries; used ONLY for the -ngl 0 rows of the benchmark.
export LLAMA_BIN_CPU=$HOME/quantization-tradeoffs/llama.cpp/build-cpu/bin
```

`scripts/get_model.sh` — download and convert. The converter needs PyTorch only to read safetensors (the HF weight file format); install the CPU wheel first so pip does not pull the 2.5 GB CUDA build:

```bash
#!/usr/bin/env bash
# Download the HF checkpoint and convert it to a single f16 GGUF file.
set -euo pipefail                      # strict mode (see build script)
cd "$(dirname "$0")/.."               # project root
source .venv/bin/activate              # use the project's Python env
# --index-url … /whl/cpu → PyTorch's CPU-only package index; the converter only needs torch to
# deserialise safetensors, and the default index would download the multi-GB CUDA build.
pip install torch --index-url https://download.pytorch.org/whl/cpu
# The converter's own pinned dependencies (transformers, sentencepiece, gguf, …).
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

MODEL=Qwen/Qwen2.5-1.5B-Instruct       # HF repo id: org/name
# hf download: fetch every file of the repo (config.json, model*.safetensors, tokenizer*.json, …)
#   --local-dir → put the files here as plain files (not in the ~/.cache symlink layout)
hf download "$MODEL" --local-dir models/Qwen2.5-1.5B-Instruct   # older CLI: huggingface-cli download

# Convert: read config.json + safetensors, rename tensors (see Look first step 4), write GGUF.
#   positional arg → the checkpoint directory
#   --outtype f16  → store every matrix as fp16 (bf16 → fp16 cast; norms stay f32)
#   --outfile      → output path (the naming convention <model>-<type>.gguf is what llama-bench reports)
python llama.cpp/convert_hf_to_gguf.py models/Qwen2.5-1.5B-Instruct \
    --outtype f16 \
    --outfile models/qwen2.5-1.5b-instruct-f16.gguf
```

`--outtype f16` is deliberate. The checkpoint is bf16; fp16 has the same 16 bits but trades 3 exponent bits for mantissa, so the conversion is exact for every weight of ordinary magnitude and only clips values above 65,504, which do not occur in this model. On sm_75, F16 is the fast path and BF16 is not.

`scripts/quantize_all.sh`:

```bash
#!/usr/bin/env bash
# Produce the three quantized GGUFs from the f16 one and write models/MANIFEST.md.
set -euo pipefail                      # strict mode
cd "$(dirname "$0")/.."               # project root
BASE=models/qwen2.5-1.5b-instruct-f16.gguf
# One llama-quantize run per recipe. Arguments are positional: <input gguf> <output gguf> <recipe name>.
# The recipe name (Q8_0 / Q5_K_M / Q4_K_M) selects the ftype; llama_tensor_get_type then picks a type per tensor.
for Q in Q8_0 Q5_K_M Q4_K_M; do
  OUT=models/qwen2.5-1.5b-instruct-${Q}.gguf
  # Skip if already present (idempotent).  2>&1 merges stderr into stdout;  tail -n 5 keeps just the summary lines
  # (total size, bpw, and the per-type histogram) — remove `| tail` once to watch the per-tensor decisions scroll by.
  [ -f "$OUT" ] || "$LLAMA_BIN/llama-quantize" "$BASE" "$OUT" "$Q" 2>&1 | tail -n 5
done

# Manifest: sizes, measured bits-per-weight, hashes. This is a README input.
# `python -` reads the program from stdin; <<'EOF' is a here-document, and the quotes around EOF stop bash
# from expanding $ inside the Python code.
python - <<'EOF'
import os, hashlib, glob
from gguf import GGUFReader                       # pure-Python GGUF parser from the `gguf` package
rows = []
for p in sorted(glob.glob("models/*.gguf")):      # every GGUF we have, alphabetically
    r = GGUFReader(p)                             # memory-maps the file; does NOT load tensor data
    # Parameter count = sum of element counts over all tensors (this is the honest denominator for bpw).
    n_params = sum(int(t.n_elements) for t in r.tensors)
    size = os.path.getsize(p)                     # bytes on disk, including metadata + tokenizer
    # Hash only the first 64 MB (1 << 26 bytes): enough to fingerprint the file, fast on a multi-GB model.
    h = hashlib.sha256(open(p, "rb").read(1 << 26)).hexdigest()[:12]   # first 64 MB, enough to identify
    # (file name, GB, measured bpw = bits / params, params in billions, hash prefix)
    rows.append((os.path.basename(p), size / 1e9, size * 8 / n_params, n_params / 1e9, h))
with open("models/MANIFEST.md", "w") as f:
    f.write("| file | GB | measured bpw | params (B) | sha256[:12] of first 64MB |\n|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]} |\n")
print(open("models/MANIFEST.md").read())
EOF
```

Quantizing a 1.5B model from f16 takes under a minute per level; it is a CPU job that reads each tensor, computes block scales, and rounds. `llama-quantize` also prints the per-tensor type decisions as it goes — that is the `_M` recipe being applied in front of you (and the list you predicted in "Look first" step 2).

**Optional: importance matrix.** `llama-imatrix` runs the f16 model over a calibration text and records, per input channel of every matrix, the mean squared activation. `llama-quantize --imatrix results/imatrix.dat …` then weights the rounding-error minimization by that importance, so channels that carry large activations get rounded more carefully. It is activation-*aware* quantization in the spirit of AWQ, without GPTQ's Hessian machinery. It typically cuts Q4_K_M's KL divergence by 20–40 % and does nothing measurable at Q8_0. If you do it, use ~100 KB of diverse general text that is **not** wikitext-test and **not** GSM8K (or you contaminate M3), and make it a fifth row in the table (`Q4_K_M+imatrix`) rather than replacing the plain Q4_K_M — the delta is the interesting part.

```bash
# llama-imatrix: run the model over calibration text, accumulate per-channel activation statistics.
#   -m  → the f16 model (always calibrate on the unquantized model)
#   -f  → calibration text file
#   -o  → where to write the importance matrix
#   -ngl 99 → offload all layers to the GPU (99 > 28 layers = "everything"); this is just for speed
#   -c 512     → context window per chunk of text
#   --chunks 100 → process 100 chunks of 512 tokens = ~51k tokens of calibration data
"$LLAMA_BIN/llama-imatrix" -m models/qwen2.5-1.5b-instruct-f16.gguf -f eval/calib.txt \
    -o results/imatrix.dat -ngl 99 -c 512 --chunks 100
# Same quantize call as before, plus --imatrix so the rounding minimises importance-weighted error.
"$LLAMA_BIN/llama-quantize" --imatrix results/imatrix.dat \
    models/qwen2.5-1.5b-instruct-f16.gguf models/qwen2.5-1.5b-instruct-Q4_K_M-imat.gguf Q4_K_M
```

### Test

Three checks. First, the build actually uses the GPU:

```bash
# llama-bench: the built-in benchmark. Here it is only a smoke test.
#   -m → model;  -p 128 → prefill test with a 128-token prompt;  -n 32 → generate 32 tokens
#   -ngl 99 → offload all layers to the GPU;  -r 2 → two timed repetitions (fast)
"$LLAMA_BIN/llama-bench" -m models/qwen2.5-1.5b-instruct-Q4_K_M.gguf -p 128 -n 32 -ngl 99 -r 2
```

The header must show `ggml_cuda_init: found 1 CUDA devices: Device 0: NVIDIA GeForce RTX 2060, compute capability 7.5`. Second, the per-tensor type map — this is the `_M` recipe made visible:

```bash
pip install gguf   # provides gguf-dump (already in the venv if you followed 0c)
# Sample layers 0, 1 (early), 13 (a middle layer use_more_bits does NOT select), 14 (a middle layer it does),
# and 27 (last), for the four tensors the recipe treats differently, plus the embedding/output rows.
# grep -E → extended regex; the \. escapes are literal dots in tensor names.
gguf-dump models/qwen2.5-1.5b-instruct-Q4_K_M.gguf | grep -E 'blk\.(0|1|13|14|27)\.(attn_v|attn_q|ffn_down|ffn_up)\.weight|token_embd|output\.weight'
```

Third, a smoke generation so you know the tokenizer and chat template survived conversion:

```bash
# llama-cli: interactive chat.  -m → model;  -ngl 99 → all layers on GPU.
"$LLAMA_BIN/llama-cli" -m models/qwen2.5-1.5b-instruct-Q4_K_M.gguf -ngl 99
# type: What is 17 * 23?   → expect 391, in a sensible sentence. Ctrl-C / "/exit" to leave.
```

### Expected

- `gguf-dump` shows `attn_q.weight` as `Q4_K`, `attn_v.weight` as `Q6_K` in the sampled layers, `ffn_down` as Q6_K in layers 0/1/14/27 and Q4_K in layer 13 (and the other middle layers `use_more_bits` skips), `ffn_up` as `Q4_K` everywhere, `token_embd.weight` as `Q6_K` (the tied-output rule), all `*_norm.weight` as `F32`. In the Q8_0 file every matrix is `Q8_0`. In the F16 file every matrix is `F16`. Compare against the list you wrote in "Look first" step 2 — if a layer differs from your prediction, re-read the branch, not the dump.
- `MANIFEST.md` measured bpw: F16 ≈ 16.0–16.3, Q8_0 ≈ 8.5–8.9, Q5_K_M ≈ 5.7–6.3, Q4_K_M ≈ 4.9–5.6. The upward drift from the nominal figure is the f32 norms and the embedding — larger on a 1.5B model than on a 7B because the embedding is a bigger fraction. `params (B)` should read 1.54 for every file — same weights, different storage.
- The smoke test answers arithmetic correctly at Q4_K_M. If it produces garbage or a wall of `!!!!`, see the table.

### Reading a failure

| Symptom | Cause |
|---|---|
| `cmake` says `No CUDA toolset found` (Windows) | CUDA toolkit installed before VS Build Tools; rerun the CUDA installer or copy the `CUDA 12.x.props/targets` files into the MSBuild `BuildCustomizations` folder |
| `nvcc` not found in WSL | `/usr/local/cuda/bin` not on `PATH`, or you installed `cuda` (driver package) instead of `cuda-toolkit-12-x` |
| Build succeeds, `llama-bench` prints no `ggml_cuda_init` line | You ran the `build-cpu` binary, or `-DGGML_CUDA=ON` was ignored because the build dir was configured earlier without it — delete `build-cuda` and reconfigure |
| `cudaErrorNoKernelImageForDevice` at runtime | `CMAKE_CUDA_ARCHITECTURES` did not include 75 (e.g. you copied a flag with `86`) |
| Converter fails with `Unknown architecture` or missing `tokenizer.json` | Incomplete download; `hf download` again — check the directory has `config.json`, `model*.safetensors`, `tokenizer.json`, `tokenizer_config.json` |
| Q4 model outputs `!!!!!` or repeated tokens; F16 is fine | Corrupt quantization (disk full mid-write is the usual cause) — check size against the manifest and requantize |
| F16 model outputs garbage too | Chat template or tokenizer mismatch; check `gguf-dump … \| grep chat_template` shows the template exists, and that you used `--outtype f16` on the Instruct checkpoint, not the base one |
| Q8_0 and F16 sizes fine, K-quants ~10 % larger than the table | Normal. Small model, big embedding, the f32 norms and Q6_K tensors are a larger share |

### Close M1

```
git add scripts/ models/MANIFEST.md results/llamacpp_version.txt .gitignore
git commit -m "M1: llama.cpp CUDA (sm_75) + CPU builds; Qwen2.5-1.5B-Instruct → f16 GGUF → Q8_0/Q5_K_M/Q4_K_M with per-tensor type manifest"
```

README section to write now — **"Setup and models"**: one paragraph on the build (two build dirs and why), the four file sizes with measured bpw from the manifest, and a three-sentence explanation of K-quant super-blocks and the `_M` recipe with the `gguf-dump` excerpt as evidence.

---

## Milestone 2 — Speed and memory (Day 2)

### What it is

You are measuring two different things and must keep them apart, because they have different bottlenecks:

**Prefill (`pp512`)**: 512 prompt tokens in one forward pass. Every weight is read once and used 512 times; arithmetic intensity (FLOPs per byte moved) is ~512 FLOPs per weight-byte at f16. The GPU is **compute-bound** here: time ≈ 2 × params × 512 / achieved FLOPS. Quantization does not reduce the FLOPs — the multiply still happens at fp16/int8 precision after dequantization — so **prefill throughput should be roughly flat across your four levels**, within kernel-path noise.

**Decode (`tg128`)**: 128 tokens generated one at a time. Each step reads every weight once for a single token: ~1 FLOP per byte. The GPU is **memory-bound**: time per token ≈ bytes / bandwidth + fixed overhead. Halving the bytes roughly halves the time. **Decode tok/s should scale close to 1 / bytes-per-weight**, bending flat at the low-byte end where fixed per-token overhead (kernel launches, sampling, the CPU side of the loop) stops shrinking.

This pair of facts is the whole interview payload of the project. Plot them side by side and you can narrate Module 2 from your own axes.

The concrete ceiling numbers for the 2060 (336 GB/s), using the manifest sizes:

| Level | weight bytes | decode ceiling = 336 / bytes | realistic (55–75 % MBU, lower at Q4) |
|---|---|---|---|
| F16 | 3.1 GB | ~108 tok/s | 70–90 |
| Q8_0 | 1.65 GB | ~204 | 120–160 |
| Q5_K_M | 1.15 GB | ~290 | 140–200 |
| Q4_K_M | 1.0 GB | ~336 | 160–230 |

Why the Q4 row does not reach the ratio the bytes predict: suppose the fixed cost per decode step is ~2 ms (hundreds of kernel launches across 28 layers, plus sampling). F16 costs 9.2 ms of weight reads + 2 ms = 11.2 ms → 89 tok/s; Q4_K_M costs 3.0 + 2 = 5.0 ms → 200 tok/s. The bytes ratio is 3.1×, the tok/s ratio is 2.2×. **Model bandwidth utilization (MBU) = measured tok/s × bytes / 336 GB/s** is the honest column to add; it will fall as bpw falls, and explaining why is a good answer.

**Memory** has three components you can read from the load log and one you sample from `nvidia-smi`: the **model buffer** (weights on the GPU), the **KV buffer** (sized by `-c`, the context length: for this model 2 × 28 layers × 2 KV heads × 128 head_dim × 2 bytes = **28 KB per token** at f16, so 4096 tokens = 115 MB and 32k tokens = 0.9 GB), and the **compute buffer** (activations for the largest batch, typically 100–400 MB). `nvidia-smi` gives you the sum plus CUDA context overhead (~300 MB) plus whatever Windows already had, so always subtract a baseline reading taken before the process starts.

**Load time** is wall-clock from process start to ready. With `mmap` (the default) the weights are memory-mapped and copied to VRAM; the second load after a reboot comes from page cache and is several times faster than the first. Report **warm** load times and say so, or drop caches between runs (`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` — `sync` flushes pending writes, `echo 3 > drop_caches` tells the kernel to discard the page cache; the `sh -c` wrapper is needed because the redirect must run as root too) and report cold. For CPU-only runs, mmap also means the process RSS understates memory — the model lives in page cache; `--no-mmap` forces a real read into process memory and is the number to quote for "RAM needed."

**Stability.** `llama-bench` does one warmup run, then `-r` timed repetitions and reports mean ± stddev; with `-o json` you also get every sample, so take the median yourself. Run each config with `-r 10`, close Chrome and anything else using the GPU, let the card reach steady clocks (the first config's warmup does that), and run the whole sweep twice on different occasions to check drift — if two sweeps disagree by more than ~5 %, something (thermal, a background process, Windows update) was running. The CPU numbers depend on thread count: sweep `-t` over {4, 6, 8, physical cores, all threads}; the best is nearly always the physical core count, and the all-threads number is often *worse* because the memory-bound loop gains nothing from SMT and pays for contention. Report the best and say what it was.

### Look first

**1. `llama-bench --help`.** Run `"$LLAMA_BIN/llama-bench" --help` and read every option once. The ones this milestone uses: `-m` (model file), `-p` (prompt-processing test size in tokens), `-n` (text-generation test size), `-ngl` (layers offloaded to GPU), `-t` (CPU threads), `-r` (timed repetitions after one warmup), `-fa` (flash attention 0/1), `-d` (depth: how many tokens are already in the KV cache when the test starts — decode at long context), `-o` (output format: `md`, `csv`, `json`, `jsonl`, `sql`). Also note `-b`/`-ub` (batch / micro-batch sizes, left at defaults 2048/512) and `-ctk`/`-ctv` (KV cache types, an M4 "next steps" lever). Notice that `-p` and `-n` are separate *tests*, not one combined run — a `-p 512 -n 128` invocation produces two result rows.

**2. One JSON run, read by eye.** Before writing the harness, run one configuration and look at what comes back:

```bash
# Same flags the harness will use, on the smallest model, with only 3 reps so it finishes in seconds.
# python3 -m json.tool pretty-prints the JSON array.
"$LLAMA_BIN/llama-bench" -m models/qwen2.5-1.5b-instruct-Q4_K_M.gguf -p 512 -n 128 -ngl 99 -t 6 -r 3 -fa 1 -o json 2>/dev/null | python3 -m json.tool
```

You get a JSON list with two objects (one per test). Find these keys, because `run_bench.py` reads exactly them: `n_prompt` and `n_gen` (512/0 for the prefill row, 0/128 for the decode row — that is how the harness tells the rows apart), `samples_ns` and `samples_ts` (the raw per-repetition times in nanoseconds and the tokens/sec derived from them — the median comes from here), `avg_ts` / `stddev_ts` (what the table view prints), `model_size` (bytes of tensor data the model occupies, i.e. the measured-bpw numerator), `model_n_params`, `n_gpu_layers`, `n_threads`, `flash_attn`, `n_depth`, `build_commit`, `gpu_info`, `cpu_info`. Also notice what is *not* there: no memory numbers — hence the separate `nvidia-smi` sampler and `memory_probe.py`.

**3. Where the numbers come from.** Open `llama.cpp/tools/llama-bench/llama-bench.cpp` and find `test_prompt` and `test_gen` (the two timed loops) and the `warmup` handling in `main` — one untimed pass runs first so the first sample is not paying for page-cache misses and clock ramp-up. Then find where `samples_ns` is turned into `avg_ts`/`stddev_ts` (`struct test`, methods `avg_ns`, `stdev_ns`, `avg_ts`, `stdev_ts`): confirm that `stddev_ts` is a sample standard deviation over the `-r` repetitions and that no median is computed — which is why the harness computes its own.

**4. `nvidia-smi` as a data source.** Run `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm --format=csv` with nothing else running and note the idle `memory.used` (this is your baseline: CUDA context of the compositor and whatever else). `nvidia-smi --help-query-gpu` lists every queryable field. The harness polls `memory.used` five times a second in a thread.

**5. One server load log.** Start `"$LLAMA_BIN/llama-server" -m models/qwen2.5-1.5b-instruct-Q8_0.gguf -ngl 99 -c 4096 -fa 1 -np 1 --port 8089` in one terminal, watch the log, and Ctrl-C it. Find the three lines `memory_probe.py` greps: `load_tensors: CUDA0 model buffer size = … MiB` (and the `CPU_Mapped model buffer size` line right next to it — that is the token embedding staying in host memory), `… CUDA0 KV buffer size = … MiB`, `… CUDA0 compute buffer size = … MiB`. Do the arithmetic: KV buffer ÷ 4096 tokens should be ≈ 28 KB. Also note the `n_ctx`, `n_batch`, `n_ubatch`, `flash_attn`, `type_k`/`type_v` lines: those are the settings that must not change between levels.

**6. Your CPU's shape.** `lscpu | grep -E 'Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket'` gives physical cores and threads-per-core; `PHYS_CORES` below must be the physical count. `sudo dmidecode -t memory | grep -E 'Speed|Configured' | head` (or Task Manager → Memory on Windows) gives the DIMM transfer rate for the theoretical-bandwidth formula `channels × 8 bytes × MT/s`.

### Code

`bench/run_bench.py` — one `llama-bench` process per configuration, `nvidia-smi` sampled in a thread for the GPU rows, medians computed from the raw samples.

```python
#!/usr/bin/env python3
"""Sweep quant levels × (GPU, CPU) with llama-bench; write results/bench.csv.

One llama-bench subprocess per (model, device, thread-count) configuration. For GPU rows a background
thread polls nvidia-smi so we also get peak VRAM. Every row of the CSV is one (config, test) pair, where
test is "pp512" (prefill) or "tg128" (decode)."""
import csv, json, os, subprocess, statistics, threading, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]         # repo root: bench/run_bench.py → parents[0]=bench, [1]=root
LLAMA_BIN = Path(os.environ["LLAMA_BIN"])          # CUDA build   (KeyError here = you forgot the export)
LLAMA_BIN_CPU = Path(os.environ["LLAMA_BIN_CPU"])  # CPU-only build, used for every -ngl 0 row
# The four precision levels, in the order the README table uses. Keys are the labels every later script imports.
MODELS = {
    "F16":    ROOT / "models/qwen2.5-1.5b-instruct-f16.gguf",
    "Q8_0":   ROOT / "models/qwen2.5-1.5b-instruct-Q8_0.gguf",
    "Q5_K_M": ROOT / "models/qwen2.5-1.5b-instruct-Q5_K_M.gguf",
    "Q4_K_M": ROOT / "models/qwen2.5-1.5b-instruct-Q4_K_M.gguf",
}
PHYS_CORES = int(os.environ.get("PHYS_CORES", "6"))   # physical core count — set it from lscpu (Look first 6)
# Thread sweep for the CPU rows: a set so duplicates collapse (e.g. PHYS_CORES == 6), sorted for stable output.
# os.cpu_count() is the logical count (all SMT threads) — expected to be the *worse* one.
CPU_THREADS = sorted({4, 6, 8, PHYS_CORES, os.cpu_count()})
REPS = int(os.environ.get("REPS", "10"))             # timed repetitions per config (llama-bench -r)
GPU_BW_GBS = 336.0                                   # RTX 2060 GDDR6 spec bandwidth, for MBU

def nvsmi_used_mb():
    """Current GPU memory in use, in MB, straight from nvidia-smi.
    --query-gpu=memory.used → ask for one field;  --format=csv,noheader,nounits → print just the number."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out.splitlines()[0])                # first line = GPU 0

class VramSampler(threading.Thread):
    """Background thread: poll nvidia-smi every 200 ms and remember the peak while a benchmark runs.
    daemon=True so a crash in the main thread does not leave this thread keeping the process alive."""
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0.0; self.stop = threading.Event()
    def run(self):
        while not self.stop.is_set():                # loop until main thread calls stop.set()
            self.peak = max(self.peak, nvsmi_used_mb()); time.sleep(0.2)

def run_bench(binary, model, ngl, threads, depth=0):
    """Run one llama-bench process and return its parsed JSON (a list with one dict per test).
      -m   model file
      -p 512 / -n 128   → the two tests: prefill 512 tokens, then generate 128 tokens
      -ngl  layers on GPU: 99 (= all 28) for GPU rows, 0 for CPU rows
      -t    CPU threads (matters for -ngl 0; for GPU rows it is the host-side thread count, keep it at PHYS_CORES)
      -r    timed repetitions (after one untimed warmup)
      -fa 1 flash attention on (newer builds also accept on/off/auto)
      -d    depth: tokens already in the KV cache before the test starts (0 = empty context)
      -o json → machine-readable output with per-repetition samples"""
    cmd = [str(binary / "llama-bench"), "-m", str(model), "-p", "512", "-n", "128",
           "-ngl", str(ngl), "-t", str(threads), "-r", str(REPS), "-fa", "1", "-d", str(depth), "-o", "json"]
    res = subprocess.run(cmd, capture_output=True, text=True)   # blocks until llama-bench exits
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:]); raise RuntimeError("llama-bench failed")   # show the tail of the log
    return json.loads(res.stdout)     # list of dicts, one per test (pp512, tg128)

def main():
    rows = []
    baseline_mb = nvsmi_used_mb()                    # VRAM in use before we start anything: subtracted from every peak
    # Configs: one GPU row (CUDA binary, all layers offloaded, host threads = physical cores)
    # plus one CPU row per thread count (CPU-only binary, nothing offloaded).
    configs = [("gpu", LLAMA_BIN, 99, PHYS_CORES)] + [("cpu", LLAMA_BIN_CPU, 0, t) for t in CPU_THREADS]
    for level, model in MODELS.items():
        for device, binary, ngl, threads in configs:
            sampler = VramSampler() if device == "gpu" else None   # only sample VRAM when the GPU is in use
            if sampler: sampler.start()
            t0 = time.time()
            results = run_bench(binary, model, ngl, threads)
            wall = time.time() - t0                  # whole-process wall time (load + warmup + REPS × 2 tests)
            if sampler: sampler.stop.set(); sampler.join()   # stop polling, wait for the thread to exit
            for r in results:                        # two entries: the pp512 row and the tg128 row
                # llama-bench marks the prefill test with n_gen == 0 and the decode test with n_prompt == 0
                test = f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}"
                samples = r["samples_ts"]            # tokens/sec for each of the REPS repetitions
                bytes_ = r["model_size"]; params = r["model_n_params"]   # tensor bytes and element count
                med = statistics.median(samples)     # our headline number: robust to one slow repetition
                rows.append({
                    "level": level, "device": device, "threads": threads, "ngl": ngl, "test": test,
                    "bpw": round(bytes_ * 8 / params, 2), "model_gb": round(bytes_ / 1e9, 2),   # measured bpw, x-axis of every plot
                    "tps_median": round(med, 1), "tps_mean": round(r["avg_ts"], 1),
                    "tps_std": round(r["stddev_ts"], 1), "reps": len(samples),
                    # MBU = achieved bytes/s ÷ spec bandwidth; only meaningful for GPU decode (memory-bound regime)
                    "mbu": round(med * bytes_ / (GPU_BW_GBS * 1e9), 3) if (device == "gpu" and test.startswith("tg")) else "",
                    "vram_peak_mb_over_baseline": round(sampler.peak - baseline_mb) if sampler else "",
                    "wall_s": round(wall, 1),
                })
                print(rows[-1], flush=True)          # live progress; flush so it shows even when piped to a file
    out = ROOT / "results/bench.csv"
    with open(out, "w", newline="") as f:            # newline="" is the csv module's requirement on Windows
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote", out)

if __name__ == "__main__":
    main()
```

Set `PHYS_CORES` to your CPU's physical core count before running. The `-d` (depth) flag is left at 0 here; a second pass with `-d 4096` gives decode at 4k context, which shows the KV-read cost growing on top of the weight-read cost — an optional extra row set.

`bench/memory_probe.py` — load time and the three buffer sizes, via `llama-server` so the process sits idle long enough to read steady-state VRAM. `/health` returns 503 until the model is loaded and `{"status":"ok"}` after.

```python
#!/usr/bin/env python3
"""Per level: warm load time (start → /health ok), model/KV/compute buffers from the log, steady VRAM.

Uses llama-server rather than llama-bench because a server sits idle after loading, which lets us read a
steady-state VRAM figure, and because its startup log prints the three buffer sizes we want."""
import csv, os, re, subprocess, time, requests
from pathlib import Path
from run_bench import MODELS, nvsmi_used_mb, LLAMA_BIN, ROOT   # reuse the level→file map and the nvidia-smi helper

CTX = int(os.environ.get("CTX", "4096"))     # context length (-c): sets the KV buffer size; keep identical across levels
PORT = 8089                                  # any free port; only the probe talks to it

def probe(level, model):
    """Start a server for one model, time until /health says ok, read VRAM, kill it, parse its log."""
    baseline = nvsmi_used_mb()               # VRAM already in use (compositor, other apps) before we launch
    log = open(ROOT / f"results/server_{level}.log", "w")
    t0 = time.time()
    # llama-server flags:
    #   -m model;  -ngl 99 all layers on GPU;  -c CTX context length;  -fa 1 flash attention on
    #   --port PORT listen here;  -np 1 one slot (one concurrent request) so the KV buffer is CTX tokens, not CTX×slots
    # stdout=log, stderr=STDOUT → everything the server prints goes to results/server_<level>.log
    p = subprocess.Popen([str(LLAMA_BIN / "llama-server"), "-m", str(model), "-ngl", "99", "-c", str(CTX),
                          "-fa", "1", "--port", str(PORT), "-np", "1"], stdout=log, stderr=subprocess.STDOUT)
    try:
        while True:                          # poll /health until the model is loaded
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=0.5).status_code == 200: break
            except requests.exceptions.RequestException: pass   # connection refused while the port is not open yet
            if p.poll() is not None: raise RuntimeError(f"server died, see {log.name}")   # poll() = exit code or None
            time.sleep(0.05)
        load_s = time.time() - t0            # "load time" = process start → first 200 from /health
        time.sleep(2)                        # let allocations settle before reading VRAM
        steady = nvsmi_used_mb() - baseline  # steady-state VRAM attributable to this server
    finally:
        p.terminate(); p.wait(); log.close() # always kill the server, even on error
    text = open(log.name).read()
    def grab(kind):
        # Pull "CUDA0 <kind> buffer size = 1234.56 MiB" out of the log; kind ∈ {model, KV, compute}
        m = re.search(rf"CUDA0 {kind} buffer size\s*=\s*([\d.]+) MiB", text)
        return round(float(m.group(1))) if m else ""   # "" rather than crash if a log line is renamed upstream
    return {"level": level, "ctx": CTX, "load_s_warm": round(load_s, 2),
            "model_buf_mb": grab("model"), "kv_buf_mb": grab("KV"), "compute_buf_mb": grab("compute"),
            "vram_steady_mb_over_baseline": round(steady)}

if __name__ == "__main__":
    rows = []
    for level, model in MODELS.items():
        probe(level, model)                       # first load warms the page cache; discard
        rows.append(probe(level, model)); print(rows[-1])   # second load is the "warm" number we keep
    with open(ROOT / "results/memory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
```

The regex names (`CUDA0 model buffer size`, `CUDA0 KV buffer size`, `CUDA0 compute buffer size`) match current logs; if a llama.cpp update renames them, open the saved `results/server_F16.log` and adjust — the values are always printed at load.

`bench/membw.py` — a rough host-RAM bandwidth number so the CPU rows get their own ceiling line:

```python
#!/usr/bin/env python3
"""Rough host-RAM bandwidth: time a large numpy memory copy. Used only for the dashed CPU ceiling line."""
import numpy as np, time
a = np.ones(2**28, dtype=np.float32)          # 1 GB: 2^28 elements × 4 bytes; shape [268435456]
b = np.empty_like(a)                           # destination buffer, same shape/dtype, uninitialised
np.copyto(b, a)                                # warm: first touch faults in the pages of b; not timed
ts = []
for _ in range(5):                             # five timed copies; keep the fastest (least interference)
    t = time.perf_counter(); np.copyto(b, a); ts.append(time.perf_counter() - t)
gb = 2 * a.nbytes / 1e9                        # read + write: a copy moves every byte twice
print(f"host copy bandwidth ≈ {gb / min(ts):.1f} GB/s  (single-threaded numpy copy; llama.cpp multi-threaded reads get ~1.3–2× this)")
```

This underestimates what a multi-threaded memory-bound loop achieves; treat it as a floor and also quote the theoretical figure from your DIMM spec (channels × 8 bytes × MT/s).

### Test

```bash
cd ~/quantization-tradeoffs && source .venv/bin/activate
export PHYS_CORES=6 REPS=10          # your physical core count (lscpu) and repetitions per config
python bench/membw.py | tee results/membw.txt     # tee: show it and save it
python bench/run_bench.py            # ~20–40 min: 4 levels × (1 GPU + 5 CPU configs) × 10 reps; CPU F16 is the slow one
python bench/memory_probe.py
# Quick look at the CSVs with pandas (heredoc Python, see quantize_all.sh for the `python - <<'EOF'` idiom).
python - <<'EOF'
import pandas as pd
b = pd.read_csv("results/bench.csv")
# GPU rows → a level × test grid of median tok/s (pivot = rows become levels, columns become tests)
g = b[b.device=="gpu"].pivot(index="level", columns="test", values="tps_median")
print(g.loc[["F16","Q8_0","Q5_K_M","Q4_K_M"]])
# CPU rows → keep only the best thread count per (level, test): sort fastest first, drop later duplicates
c = b[(b.device=="cpu")].sort_values("tps_median", ascending=False).drop_duplicates(["level","test"])
print(c[["level","test","threads","tps_median"]].sort_values(["test","level"]))
print(pd.read_csv("results/memory.csv"))
EOF
```

### Expected

- **GPU `tg128`**: monotonically increasing from F16 to Q4_K_M, in the bands from the table above; the F16→Q4_K_M ratio is ~2–2.6×, noticeably less than the 3.1× bytes ratio. MBU ~0.6–0.8 at F16 falling to ~0.4–0.6 at Q4_K_M. Q5_K_M and Q4_K_M are close (~10–20 % apart).
- **GPU `pp512`**: all four levels within roughly ±25 % of each other, somewhere in the low thousands of tok/s for this model on a 2060, and **not necessarily ordered by bpw** — F16 (cuBLAS) may beat or trail the quantized levels (MMQ int8 tensor cores). Non-monotonic prefill is not a bug; it is the compute-bound regime showing that bytes are not the lever. Write that sentence in the README.
- **CPU `tg128`** (best thread count): roughly host_bandwidth / bytes × 0.5–0.7. On ~40 GB/s effective: F16 ~6–9, Q8_0 ~12–18, Q5_K_M ~15–22, Q4_K_M ~18–28 tok/s. The CPU ratio F16→Q4 is closer to the pure bytes ratio than the GPU's, because fixed per-token overhead is a smaller fraction of a 100 ms step than of a 5 ms one.
- **CPU `pp512`**: tens to a few hundred tok/s, and here the quantized levels can be *slower* than expected relative to each other because K-quant dequantization on CPU is real work in the compute-bound regime — another data point for "prefill is not about bytes."
- **Memory**: model buffer ≈ manifest size minus ~200–250 MB (the token embedding stays in host memory as `CPU_Mapped`); KV buffer at `-c 4096` ≈ 112–120 MB for every level (KV size does not depend on weight quantization — say this out loud, it is a common confusion); compute buffer ~150–350 MB; steady VRAM over baseline ≈ model + KV + compute + ~250–350 MB CUDA context.
- **Warm load time**: proportional to file size, roughly 0.5–1.5 s per GB in WSL from page cache; F16 the slowest at ~2–5 s.

### Reading a failure

| Symptom | Cause |
|---|---|
| CPU `pp512` for a quantized level is suspiciously close to the GPU number | You used the CUDA build at `-ngl 0`; the batched matmul was offloaded. Check `LLAMA_BIN_CPU` points at `build-cpu` |
| GPU `tg128` at F16 far below 60 tok/s, or any level collapsing to single digits | VRAM overflow with sysmem fallback (Windows) or another process on the GPU. Check `nvidia-smi` baseline; close browsers; on Windows set "Prefer No Sysmem Fallback" |
| `tg128` ordering is F16 < Q8_0 but Q4_K_M ≈ Q5_K_M ≈ Q8_0 | Fixed overhead dominates. Check the flat region is real: `-d 4096` should re-separate them since KV reads add per-token bytes; if still flat, the CPU side is the bottleneck — check `-t` isn't set to all threads for GPU runs (leave it at physical cores) |
| stddev > 10 % of mean on GPU | Thermal or background load. Rerun after idle; look at `nvidia-smi -q -d CLOCK` for clocks bouncing |
| CPU numbers best at 4 threads, worse at 6+ | Memory-bound and contended; your effective bandwidth is lower than you assumed. Report it; it strengthens the "bandwidth is the ceiling" story |
| `memory_probe.py` hangs at `/health` | Wrong port or the server printed an error — read `results/server_<level>.log` |
| KV buffer differs across levels | You changed `-c` or `-fa` between runs; keep both fixed |
| CPU F16 run swaps or gets killed | WSL memory cap too low; raise `.wslconfig` memory to ≥ 16 GB |

### Close M2

```
git add bench/ results/bench.csv results/memory.csv results/membw.txt
git commit -m "M2: llama-bench sweep (pp512/tg128) over 4 quant levels on RTX 2060 and CPU-only; VRAM, load time, thread sweep, MBU"
```

README section — **"Speed and memory"**: the GPU table (level, bpw, GB, pp512, tg128, MBU, VRAM steady, warm load), the CPU table (best thread count per level), and one paragraph stating the decode-scales-with-1/bytes / prefill-is-flat observation with the numbers. The graphs come in M4; leave a placeholder.

---

## Milestone 3 — Quality (Day 3)

### What it is

Speed numbers without quality numbers are the mistake the old resume bullet made. You need two signals, because they fail in different ways:

**Task accuracy on a fixed prompt set.** 50 GSM8K test questions, greedy decoding, exact match on the final number. It measures what a user would notice. Its weakness is statistical: at ~65 % accuracy, the standard error on 50 questions is ≈ √(0.65 × 0.35 / 50) ≈ 6.7 points. That means **a 50-prompt eval cannot distinguish Q8_0 from F16, and probably not Q5_K_M either** — a difference of 2 questions is noise. You must say this in the README; an interviewer who hears you volunteer the confidence interval trusts every other number you present. Two mitigations: (1) also report **agreement with the F16 answers** (same final number, right or wrong), which is a paired measure with much less variance than absolute accuracy, and (2) if time allows, run 250 questions for the README table while keeping the 50 as the spec's quick check.

**Perplexity and KL divergence on a standard text.** `llama-perplexity` on a wikitext-2 slice (a standard Wikipedia-article test text every quantization writeup uses) gives PPL — the standard number every quantization writeup reports, so yours is comparable — and, with `--kl-divergence`, the **mean KL divergence** between each quantized model's next-token distribution and the F16 model's, plus the fraction of positions where the top-1 token is unchanged. KLD is the tightest signal you have: it is computed over ~10k tokens with a full distribution at each, so it separates Q8_0 from Q5_K_M from Q4_K_M cleanly even when the 50-question eval cannot. Its weakness is the opposite of accuracy's: it does not tell you whether anything a user cares about broke. Report both and explain that they answer different questions — that is question bank D3's caveat ("I'd verify on our evals, not perplexity") in practice.

**Determinism.** Every eval runs at temperature 0 with a fixed seed through `llama-server` with a single slot (`-np 1`; a **slot** is one concurrent-request lane in the server), so a prompt sees the same batch shape every time. Run the F16 eval twice and diff the outputs before trusting anything — identical files are the test. (Question bank J3 explains why this can fail with multiple slots: different batchmates change the reduction order.) Quality is evaluated on the GPU only: the quantization error lives in the weights, not the device, and the eval is not a speed measurement. If you want to prove that, run one level on CPU and compare agreement — expect ~98–100 %, not 100 %, because CPU and CUDA kernels round differently.

**Prompt format.** Qwen's own math evaluation uses the system prompt "Please reason step by step, and put your final answer within \boxed{}." Use exactly that; extract the last `\boxed{…}`, fall back to the last number in the text. `max_tokens` 512 is enough for GSM8K chains from a 1.5B model.

### Look first

**1. How PPL and KLD are actually computed.** Read `llama.cpp/tools/perplexity/README.md` top to bottom. Note: (a) the text is cut into chunks of `-c` tokens; (b) within each chunk only the *second half* of the tokens is scored (the first half is context, so every scored token has at least `c/2` tokens of history — this is why `--chunks 20 -c 512` gives ≈ 5k *scored* tokens even though 10k are processed); (c) PPL = `exp(mean negative log-likelihood)` over scored tokens, reported with `+/-` a standard error; (d) the `--kl-divergence-base` file stores the *full logits* at every scored position (`tokens × vocab × 2 bytes` — 151,936 × 2 = 304 KB per token, which is why `CHUNKS` is capped), and `--kl-divergence` then reloads them and computes, per position, `KL(P_f16 ‖ P_quant)` plus "same top p" (top-1 agreement) and the Δp statistics. Then open `tools/perplexity/perplexity.cpp`, find `kl_divergence(` and read the loop that accumulates `kld` — it is ~40 lines of softmax-and-sum, and seeing it makes the number un-mysterious.

**2. The GSM8K rows.** Before freezing the prompt set, look at two raw examples:

```python
#!/usr/bin/env python3
"""Print two raw GSM8K test rows so the answer format (the '#### 42' convention) is not a surprise."""
from datasets import load_dataset               # HF datasets library: downloads + caches the parquet files
ds = load_dataset("openai/gsm8k", "main", split="test")   # "main" config = the standard 1,319-question test split
print(len(ds))                                  # 1319
for i in (0, 1):
    print("Q:", ds[i]["question"])              # natural-language word problem
    print("A:", ds[i]["answer"])                # worked solution ending with "#### <final number>"
    print("-" * 60)
```

Notice that the gold answer is always after `####`, is an integer or plain decimal, and can contain thousands separators (`1,080`) — that is why `build_prompts.py` splits on `####` and strips commas.

**3. `llama-server --help` and one raw response.** Run `"$LLAMA_BIN/llama-server" --help | grep -E -- '-np|--parallel|-c |--ctx-size|--slots|--port|-fa'` to see the flags the eval relies on. Then start a server on the Q4_K_M file exactly as `run_eval.py` will (`-ngl 99 -c 4096 -np 1 -fa 1 --port 8090`) and send one request by hand:

```bash
# POST an OpenAI-compatible chat request; -s silent, -X POST method, -H header, -d body (JSON).
curl -s -X POST http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role":"system","content":"Please reason step by step, and put your final answer within \\boxed{}."},
               {"role":"user","content":"What is 17 * 23?"}],
  "temperature": 0, "top_k": 1, "seed": 42, "max_tokens": 256}' | python3 -m json.tool
```

Read the response: `choices[0].message.content` is the text; `usage` has prompt/completion token counts; and llama.cpp adds a non-standard `timings` object — `prompt_n`, `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, `predicted_per_second`. `run_eval.py` stores `predicted_per_second` and `predicted_n` from there. Also hit `GET /health` (200 + `{"status":"ok"}`) and `GET /slots` (shows the single slot and its state) to see what the harness polls. Finally look in the server log for the line that prints the chat template it loaded (`chat_format` / `Chat template:`), and compare with `gguf-dump … | grep chat_template` — the `<|im_start|>system\n…<|im_end|>` framing the model was trained on is what `/v1/chat/completions` applies for you and what `/completion` would not.

**4. `get-wikitext-2.sh`.** `cat llama.cpp/scripts/get-wikitext-2.sh` — it is a five-line curl + unzip; know the URL it uses so the 404 row in the failure table is a two-minute fix.

### Code

`eval/build_prompts.py` — run once, commit the output so the set is frozen:

```python
#!/usr/bin/env python3
"""Freeze N GSM8K test questions into eval/prompts.jsonl (one JSON object per line). Run once, commit the file."""
import json, random
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")       # 1,319 questions with worked answers
idx = list(range(len(ds))); random.Random(20260824).shuffle(idx)   # fixed seed → the same shuffle every time, on any machine
N = 50   # set 250 for the extended run; keep the first 50 identical either way (same seed, same prefix)
with open("eval/prompts.jsonl", "w") as f:
    for i in idx[:N]:
        # Gold answer is the text after the final "####"; strip whitespace and thousands separators ("1,080" → "1080").
        gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        f.write(json.dumps({"id": i, "question": ds[i]["question"], "gold": gold}) + "\n")   # id = row index in the dataset
print("wrote", N, "prompts")
```

`eval/run_eval.py` — starts a server per level, runs every prompt greedily, records answer text and the server's timing extension:

```python
#!/usr/bin/env python3
"""Usage: python eval/run_eval.py F16 Q8_0 Q5_K_M Q4_K_M   (or a subset). Writes results/eval_<level>.jsonl

For each level: start llama-server on that GGUF, send every prompt with greedy sampling, save the raw text,
the extracted final number, and whether it matched the gold answer."""
import json, os, re, subprocess, sys, time, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))   # make `from run_bench import …` work from eval/
from run_bench import MODELS, LLAMA_BIN, ROOT

PORT = 8090                                                       # different from memory_probe's 8089 so both can coexist
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."   # Qwen's own GSM8K prompt
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")                        # captures the inside of \boxed{...} (no nested braces)
NUM = re.compile(r"-?\d[\d,]*\.?\d*")                             # a number: optional minus, digits with commas, optional decimals

def extract(text):
    """Final numeric answer from the model's text. Prefer the last \\boxed{}, else the last number anywhere."""
    m = BOXED.findall(text)
    cand = m[-1] if m else (NUM.findall(text) or [""])[-1]        # last boxed span, or last number, or ""
    cand = cand.replace(",", "").replace("$", "").strip()         # "$1,080" → "1080"
    n = NUM.findall(cand)                                         # the boxed span may contain units: "18 dollars" → "18"
    if not n: return ""
    v = n[-1]
    return v[:-2] if v.endswith(".0") else v                      # "42.0" → "42" so it matches the integer gold

def start_server(model, tag):
    """Launch llama-server for one model and block until /health returns 200."""
    log = open(ROOT / f"results/eval_server_{tag}.log", "w")
    # -m model;  -ngl 99 all layers on GPU;  -c 4096 context;  -np 1 ONE slot (determinism: no batchmates)
    # -fa 1 flash attention on;  --port PORT.  All output → results/eval_server_<level>.log
    p = subprocess.Popen([str(LLAMA_BIN / "llama-server"), "-m", str(model), "-ngl", "99", "-c", "4096",
                          "-np", "1", "-fa", "1", "--port", str(PORT)], stdout=log, stderr=subprocess.STDOUT)
    while True:
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=0.5).status_code == 200: return p
        except requests.exceptions.RequestException: pass       # port not open yet
        if p.poll() is not None: raise RuntimeError(f"server died: {log.name}")
        time.sleep(0.1)

def ask(question):
    """One greedy chat completion. Returns (answer text, llama.cpp's timings dict)."""
    r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json={
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        # temperature 0 + top_k 1 → always the most probable token (greedy); seed fixed anyway for belt-and-braces
        "temperature": 0, "top_k": 1, "seed": 42, "max_tokens": 512}, timeout=300).json()
    return r["choices"][0]["message"]["content"], r.get("timings", {})   # timings is llama.cpp's extension to the OpenAI schema

def main(levels):
    prompts = [json.loads(l) for l in open(ROOT / "eval/prompts.jsonl")]   # the frozen set
    for level in levels:
        p = start_server(MODELS[level], level)
        try:
            with open(ROOT / f"results/eval_{level}.jsonl", "w") as f:
                for i, q in enumerate(prompts):
                    text, timings = ask(q["question"])
                    pred = extract(text)
                    f.write(json.dumps({"id": q["id"], "gold": q["gold"], "pred": pred,
                                        "correct": pred == q["gold"], "text": text,            # keep the full text for the divergence examples
                                        "gen_tps": timings.get("predicted_per_second"),        # decode tok/s for this answer (informational)
                                        "n_gen": timings.get("predicted_n")}) + "\n")           # tokens generated (truncation check: 512 = hit the cap)
                    print(f"{level} {i+1:3d}/{len(prompts)} pred={pred!r:>8} gold={q['gold']!r:>8} {'ok' if pred==q['gold'] else '--'}", flush=True)
        finally:
            p.terminate(); p.wait()                                    # always stop the server before the next level

if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))                                 # levels from the command line, default = all four
```

`eval/score.py`:

```python
#!/usr/bin/env python3
"""Turn results/eval_<level>.jsonl into results/quality.csv: accuracy ± SE, delta vs F16, agreement with F16."""
import json, math, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
levels = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M"]                       # F16 must be first: it is the reference row
# level → list of per-question records, for every level whose eval file exists
runs = {l: [json.loads(x) for x in open(ROOT / f"results/eval_{l}.jsonl")] for l in levels
        if (ROOT / f"results/eval_{l}.jsonl").exists()}
ref = {r["id"]: r["pred"] for r in runs["F16"]}                    # question id → F16's answer (right or wrong)
rows = []
for l, rs in runs.items():
    n = len(rs); acc = sum(r["correct"] for r in rs) / n           # fraction correct
    se = math.sqrt(acc * (1 - acc) / n)                            # binomial standard error of that fraction
    agree = sum(r["pred"] == ref[r["id"]] for r in rs) / n         # paired: same final answer as F16 on the same question
    rows.append({"level": l, "n": n, "accuracy": round(100 * acc, 1), "se_pp": round(100 * se, 1),   # pp = percentage points
                 # difference from the first row (F16); 0.0 for F16 itself
                 "delta_vs_f16_pp": round(100 * (acc - rows[0]["accuracy"] / 100), 1) if rows else 0.0,
                 "agree_with_f16": round(100 * agree, 1),
                 "mean_gen_tokens": round(sum(r["n_gen"] or 0 for r in rs) / n)})   # average answer length; rises if a level rambles
    print(rows[-1])
with open(ROOT / "results/quality.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
```

`eval/perplexity.sh` — PPL for each level and KLD against F16. The KLD base file stores the F16 model's full logits at every position: tokens × vocab × 2 bytes, and Qwen's vocabulary is 151,936 wide, so **limit the chunks** — 20 chunks × 512 = 10k tokens ≈ 3 GB, which is plenty of signal.

```bash
#!/usr/bin/env bash
# Perplexity for all four levels + KL divergence of each quantized level against F16 → results/ppl.csv
set -euo pipefail                      # strict mode
cd "$(dirname "$0")/.."               # project root
# Fetch wikitext-2 (raw, test split) once, using llama.cpp's own helper script.
[ -f wikitext-2-raw/wiki.test.raw ] || bash llama.cpp/scripts/get-wikitext-2.sh
CTX=512; CHUNKS=20; TXT=wikitext-2-raw/wiki.test.raw   # 20 chunks × 512 tokens; identical for every level or KLD is meaningless
F16=models/qwen2.5-1.5b-instruct-f16.gguf

# Base: F16 perplexity + save logits for KLD
#   -m model;  -f text file;  -c CTX tokens per chunk;  --chunks how many chunks;  -ngl 99 all layers on GPU
#   --kl-divergence-base FILE → additionally WRITE the full logits at every scored position to FILE (the ~3 GB *.kld)
"$LLAMA_BIN/llama-perplexity" -m "$F16" -f "$TXT" -c $CTX --chunks $CHUNKS -ngl 99 \
    --kl-divergence-base results/f16.kld 2>&1 | tee results/ppl_F16.log

echo "level,ppl,ppl_err,mean_kld,same_top1_pct" > results/ppl.csv   # header row
# grep -oP: -o print only the match, -P Perl regex so \K works ("forget everything matched before this point").
PPL=$(grep -oP 'Final estimate: PPL = \K[\d.]+' results/ppl_F16.log)    # the number after "PPL = "
ERR=$(grep -oP 'PPL = [\d.]+ \+/- \K[\d.]+' results/ppl_F16.log)        # the number after "+/- "
echo "F16,$PPL,$ERR,0,100" >> results/ppl.csv                              # F16 vs itself: KLD 0, top-1 agreement 100 %

for Q in Q8_0 Q5_K_M Q4_K_M; do
  # Same run for the quantized model, but READ the base logits and compute KLD against them:
  #   --kl-divergence-base FILE → the F16 logits to compare with;  --kl-divergence → compute and print the KLD statistics
  "$LLAMA_BIN/llama-perplexity" -m models/qwen2.5-1.5b-instruct-${Q}.gguf -f "$TXT" -c $CTX --chunks $CHUNKS -ngl 99 \
      --kl-divergence-base results/f16.kld --kl-divergence 2>&1 | tee results/ppl_${Q}.log
  PPL=$(grep -oP 'Final estimate: PPL = \K[\d.]+' results/ppl_${Q}.log)
  ERR=$(grep -oP 'PPL = [\d.]+ \+/- \K[\d.]+' results/ppl_${Q}.log)
  KLD=$(grep -oP 'Mean\s+KLD:\s*\K[\d.]+' results/ppl_${Q}.log | head -1)   # "Mean    KLD: 0.0123 ± …" → 0.0123; head -1 = first match only
  TOP=$(grep -oP 'Same top p:\s*\K[\d.]+' results/ppl_${Q}.log | head -1)   # "Same top p: 96.12 ± … %" → 96.12
  echo "$Q,$PPL,$ERR,$KLD,$TOP" >> results/ppl.csv
done
cat results/ppl.csv
```

The exact label strings in the KLD summary (`Mean    KLD:`, `Same top p:`) have been stable for a long time; if a grep comes back empty, look at the saved log and fix the pattern rather than the number.

### Test

```bash
python eval/build_prompts.py && git add eval/prompts.jsonl      # freeze and stage the prompt set
python eval/run_eval.py F16 && cp results/eval_F16.jsonl /tmp/f16_a.jsonl   # first F16 run, kept aside
# Second F16 run, then diff the first 200 characters of every line (cut -c1-200: id/gold/pred/correct + start of text).
# <( … ) is bash process substitution: feed a command's output to diff as if it were a file. Prints DETERMINISTIC only if identical.
python eval/run_eval.py F16 && diff <(cut -c1-200 /tmp/f16_a.jsonl) <(cut -c1-200 results/eval_F16.jsonl) && echo DETERMINISTIC
python eval/run_eval.py Q8_0 Q5_K_M Q4_K_M
python eval/score.py
bash eval/perplexity.sh
```

Budget: each level's 50 questions take 2–5 minutes on the GPU (~200–400 generated tokens per question); the perplexity runs are ~1–3 minutes each.

### Expected

- The two F16 runs are byte-identical → `DETERMINISTIC`. If not, see the table.
- **Accuracy (50 questions)**: F16 somewhere around 55–75 % for this model with this prompt; Q8_0 within ±1 question of F16; Q5_K_M within ±2; Q4_K_M anywhere from equal to ~4–5 questions lower. With `se_pp` ≈ 6–7, only a Q4_K_M drop of ≥ 8 points is outside noise. Say so.
- **Agreement with F16**: Q8_0 96–100 %, Q5_K_M 90–98 %, Q4_K_M 82–94 %. Agreement drops before accuracy does — the quantized model changes its reasoning path on a few questions and sometimes lands on the right answer anyway. That is the interesting finding; show two examples where Q4_K_M diverges from F16 in the README.
- **PPL** (wikitext-2, c=512, instruct model): an absolute value in the high single digits to mid-teens — the absolute number is uninteresting; the **relative** change is: Q8_0 within ~0.1–0.5 % of F16, Q5_K_M ~0.5–2 %, Q4_K_M ~2–6 %. Small models degrade more than the ~1 % figures quoted for 7B+ models, because each weight carries more information.
- **Mean KLD**: Q8_0 ~0.001–0.005, Q5_K_M ~0.01–0.03, Q4_K_M ~0.03–0.10; `Same top p` (top-1 unchanged) ~99 %, ~96–98 %, ~92–96 % respectively. This is the ordering you will not get from the 50-question accuracy, and the reason you ran it.

### Reading a failure

| Symptom | Cause |
|---|---|
| Two F16 runs differ | `-np` > 1 or `-c` changed between runs; or a llama.cpp update changed the default flash-attention/kv-cache type — check `results/eval_server_F16.log` header for cache types and slots |
| Accuracy near 0 for every level | Extraction bug: print five `text` fields; the model likely writes `\boxed{18}` with a `$` or `18.00`; fix `extract` |
| F16 accuracy under 40 % | Prompt not going through the chat template — check the server log shows the Qwen template loaded, and that you called `/v1/chat/completions` not `/completion` |
| Q4_K_M accuracy *higher* than F16 by 1–3 questions | Noise. Expected once in a while at n=50; the KLD row will still order correctly. Do not "fix" it |
| `--kl-divergence-base` fails with disk full | The base file is tokens × 151,936 × 2 bytes; reduce `CHUNKS` |
| KLD huge (> 1) for every quant | Base and quant runs used different `-c` or a different text; regenerate the base |
| `get-wikitext-2.sh` 404s | The script's URL moved; download `wikitext-2-raw-v1.zip` from the HF `ggml-org/ci` dataset manually and unzip to `wikitext-2-raw/` |
| Answers truncated mid-chain | `max_tokens` 512 too small for a handful of questions; they count as wrong at every level, which is fair; note it |

### Close M3

```
git add eval/ results/eval_*.jsonl results/quality.csv results/ppl.csv results/ppl_*.log
git commit -m "M3: fixed 50-question GSM8K eval (greedy, seeded, deterministic) + wikitext PPL and KL divergence vs f16 for each quant level"
```

README section — **"Quality"**: the table (level, accuracy ± SE, Δ vs F16, agreement with F16, PPL, ΔPPL %, mean KLD, same-top-1 %), one paragraph on why two signals, the confidence-interval sentence, and the two divergence examples.

### If you want the literal Qwen2-VL bullet (optional +1 day)

Everything above transfers; what changes is the model, the runner, and the eval. llama.cpp's multimodal support lives in `libmtmd` (multimodal): the language model and the vision tower (the image encoder plus the projector that turns image patches into token-like vectors) are two GGUF files, and the converter produces both from the same checkpoint:

```bash
# Download the VL checkpoint (language model + vision encoder in one HF repo).
hf download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen2-VL-2B-Instruct
# Convert the language model only (same flags as the text model).
python llama.cpp/convert_hf_to_gguf.py models/Qwen2-VL-2B-Instruct --outtype f16 --outfile models/qwen2-vl-2b-f16.gguf
# --mmproj → convert the vision encoder + projector into its own GGUF instead of the language model.
python llama.cpp/convert_hf_to_gguf.py models/Qwen2-VL-2B-Instruct --outtype f16 --mmproj --outfile models/qwen2-vl-2b-mmproj-f16.gguf
# Quantize ONLY the language model; the mmproj stays f16.
for Q in Q8_0 Q5_K_M Q4_K_M; do "$LLAMA_BIN/llama-quantize" models/qwen2-vl-2b-f16.gguf models/qwen2-vl-2b-${Q}.gguf $Q; done
# llama-mtmd-cli: one-shot multimodal prompt.  -m language model;  --mmproj vision GGUF;  --image input image
#   -p prompt text;  --temp 0 greedy;  -ngl 99 all layers on GPU
"$LLAMA_BIN/llama-mtmd-cli" -m models/qwen2-vl-2b-Q4_K_M.gguf --mmproj models/qwen2-vl-2b-mmproj-f16.gguf \
    --image eval/images/0001.jpg -p "Describe this image in detail." --temp 0 -ngl 99
# Server with image support: same flags as the text server plus --mmproj.
"$LLAMA_BIN/llama-server" -m models/qwen2-vl-2b-Q4_K_M.gguf --mmproj models/qwen2-vl-2b-mmproj-f16.gguf -ngl 99 -c 4096 -np 1
```

The rough edges, which are themselves resume-defensible detail: **only the language model is quantized** — the mmproj stays f16 (and you should say "I quantized the LLM; the vision encoder stayed fp16", which is also what most production deployments do). **`llama-bench` has no image path**, so VL prefill numbers come from `llama-server`'s `timings.prompt_per_second` on a fixed image size; resize every eval image to 448×448 so each contributes the same ~256 image tokens (Qwen2-VL merges 14-px patches 2×2, so one token per 28×28 pixels), otherwise "prompt length" varies per image and the prefill numbers are not comparable. **Image preprocessing runs on the CPU** and can take longer than prefill for a 2B model, so measure and report it separately from the model's prompt time (question bank J2). The older `llama-qwen2vl-cli` binary is gone; `llama-mtmd-cli` replaced it. Send images to the server as `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,…"}}` content parts in the chat message.

**VL eval**: 50 images (COCO val images or your own photos — commit the list of URLs/hashes, not the images, unless they are yours). For each, hand-write 4–6 keywords that a correct description must contain (`["dog","frisbee","grass","park"]`), stored in `eval/vl_prompts.jsonl`. Score = fraction of keywords present (lowercase substring match, with a small synonym list). Report per level, plus agreement with the F16 description's keyword set. The caveat you must state: keyword hits measure *drift* and gross omissions, not correctness of detail; hallucinated objects are not penalised unless you add a "must not contain" list, which is worth doing for 10 of the 50.

---

## Milestone 4 — The centerpiece (Day 4)

### What it is

The README is the deliverable; the repo is its evidence. It needs one table that carries every axis, two graphs, and a decision section written the way you would say it in an interview. The table:

| Level | bpw (measured) | GB on disk | VRAM steady (MB) | GPU pp512 tok/s | GPU tg128 tok/s | MBU | CPU tg128 tok/s (best -t) | Warm load s | GSM8K-50 acc (± SE) | Agree w/ F16 | ΔPPL % | Mean KLD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

fill from `results/bench.csv`, `results/memory.csv`, `results/quality.csv`, `results/ppl.csv`. Every number in that table must be traceable to a script; `plot.py` reads the same CSVs and renders the two graphs, so the README never contains a hand-typed measurement.

**Graph 1 — decode tok/s vs bits-per-weight**, GPU and CPU as two series, with the theoretical ceiling curves `bandwidth / (bpw/8 × params)` drawn as dashed lines. The measured points sit under the ceiling and the gap widens toward low bpw. Put pp512 on a second panel with the same x-axis: flat. The two panels next to each other *are* the project.

**Graph 2 — quality vs bits-per-weight**: agreement with F16 (or accuracy with error bars) on the left axis, mean KLD on a log right axis (a **log axis** spaces 0.001, 0.01, 0.1 evenly, so values that differ by 10× each are all readable on one chart). The knee is between Q5_K_M and Q4_K_M.

**The shipping decision** must be reasoned from four things: (1) decode is memory-bound, so bytes buy tok/s and nothing else does at batch 1; (2) VRAM not spent on weights is available for KV cache, and KV is what caps context length and concurrent requests; (3) each use case has a different quality floor; (4) prefill does not care. For this model on this hardware the KV math is 28 KB/token, so the VRAM freed by going F16→Q4_K_M (~2.1 GB) is ~75k tokens of KV: either one 64k-context session or ~18 extra concurrent 4k-context requests.

- **Single-user chat on the 2060**: **Q8_0**. A 1.5B model at F16 already decodes at 70–90 tok/s, far above reading speed, so Q4's extra speed is invisible to one user; Q8_0 is quality-free (KLD ~0.003, agreement ~98 %+) and halves VRAM, which turns into 32k context headroom. The "obvious" answer (Q4 because it's fastest) is wrong for this case, and saying why is the point.
- **Batch/throughput serving**: **Q4_K_M if the task eval is inside the floor, else Q5_K_M**. With many concurrent sequences the weight reads are amortised and decode drifts toward compute-bound, so the per-token speed advantage of int4 shrinks — but the VRAM saving still converts directly into batch size, which is throughput. The quality floor is task-specific: for GSM8K-like reasoning, Q4_K_M's agreement drop (~10 %) may be unacceptable while for summarisation it would not be. Cite your own agreement number.
- **Edge (Snapdragon-class)**: **Q4_K_M, non-negotiable** — and on ARM, Q4_0 with the runtime-repacked `dotprod`/`i8mm` kernels (ARM's int8 dot-product and matrix-multiply instructions; llama.cpp re-packs Q4_0 blocks at load time into the layout those instructions want) is usually faster than Q4_K_M, so benchmark both. The bandwidth math: a Snapdragon 8 Gen 3 has LPDDR5X at ~77 GB/s; X Elite-class laptop parts ~135 GB/s; call it 50–100 GB/s for a phone. Decode ceiling at Q4_K_M (1.0 GB) ≈ 50–100 tok/s, realistically 25–45 with mobile efficiency; at F16 (3.1 GB) ≈ 16–32 ceiling, realistically 8–15 — and 3.1 GB of weights in a phone's shared RAM is itself a problem before speed is. This is the honest form of the Snapdragon claim: *extrapolated by the same bandwidth arithmetic that predicted the 2060 and CPU rows to within X %*. You have two hardware points that validate the method; the third is a prediction, labelled as one.

### Look first

**1. The four CSVs, raw.** `head -n 3 results/bench.csv results/memory.csv results/quality.csv results/ppl.csv`. Confirm the column names `plot.py` uses exist exactly: `level, device, test, threads, bpw, model_gb, tps_median, mbu` in bench; `level, load_s_warm, vram_steady_mb_over_baseline` in memory; `level, accuracy, se_pp, agree_with_f16` in quality; `level, ppl, mean_kld` in ppl. Every KeyError in M4 is a column name that drifted.

**2. The pandas idiom the script leans on.** In a Python shell:

```python
#!/usr/bin/env python3
"""See what `best()` in plot.py does, one step at a time, on the real bench.csv."""
import pandas as pd
b = pd.read_csv("results/bench.csv")
d = b[(b.device == "cpu") & (b.test == "tg128")]           # only CPU decode rows: one per (level, thread count)
print(d[["level", "threads", "tps_median"]])                # 4 levels × 5 thread counts = 20 rows
d = d.sort_values("tps_median", ascending=False)            # fastest first
d = d.drop_duplicates("level")                              # keep the first (= fastest) row per level
print(d.set_index("level").loc[["F16", "Q8_0", "Q5_K_M", "Q4_K_M"]][["threads", "tps_median"]])   # reorder to the README order
```

That is exactly `best(b, "cpu", "tg128")`; the `threads` column of the result is the "best -t" the README reports.

**3. matplotlib's twin axis.** `python3 -c "import matplotlib.axes; help(matplotlib.axes.Axes.twinx)"` — `twinx()` returns a second Axes sharing the x-axis with its own y-axis on the right; that is how KLD (log scale, 0.001–0.1) and agreement (linear, 0–100 %) share one chart. Also `help(matplotlib.axes.Axes.errorbar)` for the `yerr`/`capsize` arguments used for the ± SE bars.

### Code

`bench/plot.py`:

```python
#!/usr/bin/env python3
"""Render the two README graphs and the master table from the four results CSVs.
Outputs: results/plots/speed_vs_bpw.png, results/plots/quality_vs_bpw.png, results/table.md"""
import pandas as pd, matplotlib.pyplot as plt, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ORDER = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M"]                                   # row/point order everywhere
# Bandwidth constants for the dashed ceiling lines. GPU = spec sheet; CPU = the number you measured/estimated in M2.
GPU_BW, CPU_BW = 336.0, float(open(ROOT / "results/cpu_bw_effective.txt").read())  # GB/s; write your measured/estimated CPU figure there
b = pd.read_csv(ROOT / "results/bench.csv")
params = 1.54e9                                                                # parameter count for the ceiling formula

def best(df, dev, test):
    """Best (highest median tok/s) row per level for one device and one test, in ORDER.
    For CPU rows this picks the best thread count; for GPU rows there is only one config per level."""
    d = df[(df.device == dev) & (df.test == test)].sort_values("tps_median", ascending=False)
    return d.drop_duplicates("level").set_index("level").loc[ORDER]

# ---- Graph 1: two panels, decode (left) and prefill (right), tok/s vs measured bpw ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))                              # 1 row × 2 panels, 11×4.2 inches
for dev, bw, c in [("gpu", GPU_BW, "C0"), ("cpu", CPU_BW, "C1")]:            # C0/C1 = matplotlib's default blue/orange
    tg = best(b, dev, "tg128")
    ax[0].plot(tg.bpw, tg.tps_median, "o-", color=c, label=f"{dev.upper()} measured")   # measured decode points, circles joined by lines
    x = np.linspace(4.5, 16.5, 100)                                          # 100 bpw values spanning the x-axis for a smooth ceiling curve
    # ceiling tok/s = bandwidth (bytes/s) ÷ bytes per token; bytes per token = bpw/8 × params
    ax[0].plot(x, bw * 1e9 / (x / 8 * params), "--", color=c, alpha=0.5, label=f"{dev.upper()} ceiling {bw:.0f} GB/s ÷ bytes")
    pp = best(b, dev, "pp512")
    ax[1].plot(pp.bpw, pp.tps_median, "s-", color=c, label=f"{dev.upper()} pp512")       # prefill points, squares
for a, t in zip(ax, ["Decode (tg128): memory-bound → ∝ 1/bytes", "Prefill (pp512): compute-bound → flat"]):
    a.set_xlabel("bits per weight (measured from file size)"); a.set_ylabel("tokens / s"); a.set_title(t)
    a.set_yscale("log"); a.grid(alpha=0.3); a.legend(fontsize=8)               # log y: GPU (hundreds) and CPU (tens) both readable
    for lvl, row in best(b, "gpu", "tg128").iterrows(): a.axvline(row.bpw, color="k", alpha=0.08)   # faint vertical line at each level's bpw
fig.tight_layout(); fig.savefig(ROOT / "results/plots/speed_vs_bpw.png", dpi=150)

# ---- Graph 2: quality vs bpw — accuracy ± SE and agreement (left axis, %), mean KLD (right axis, log) ----
q = pd.read_csv(ROOT / "results/quality.csv").set_index("level").loc[ORDER]
p = pd.read_csv(ROOT / "results/ppl.csv").set_index("level").loc[ORDER]
bpw = best(b, "gpu", "tg128").bpw                                              # x positions: the measured bpw per level
fig, a1 = plt.subplots(figsize=(6, 4.2)); a2 = a1.twinx()                      # a2 shares x with a1, has its own right-hand y-axis
a1.errorbar(bpw, q.accuracy, yerr=q.se_pp, fmt="o-", color="C2", capsize=3, label="GSM8K-50 accuracy ± SE")   # yerr = ± one standard error
a1.plot(bpw, q.agree_with_f16, "^-", color="C3", label="agreement with F16 answers")   # triangles
a2.plot(bpw, p.mean_kld.clip(lower=1e-4), "s--", color="C4", label="mean KLD vs F16")  # clip: F16's KLD is 0, which a log axis cannot draw
a2.set_yscale("log"); a1.set_ylim(0, 100)
a1.set_xlabel("bits per weight"); a1.set_ylabel("%"); a2.set_ylabel("mean KL divergence")
a1.set_title("Quality vs bits per weight"); a1.grid(alpha=0.3)
# Two axes = two legends; merge their handles/labels into one legend box
h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(ROOT / "results/plots/quality_vs_bpw.png", dpi=150)

# ---- Master table for the README: one row per level, every column traceable to a CSV ----
m = pd.read_csv(ROOT / "results/memory.csv").set_index("level").loc[ORDER]
g_tg, g_pp, c_tg = best(b, "gpu", "tg128"), best(b, "gpu", "pp512"), best(b, "cpu", "tg128")
tbl = pd.DataFrame({
    "bpw": g_tg.bpw, "GB": g_tg.model_gb, "VRAM MB": m.vram_steady_mb_over_baseline,
    "GPU pp512": g_pp.tps_median, "GPU tg128": g_tg.tps_median, "MBU": g_tg.mbu,
    "CPU tg128": c_tg.tps_median, "CPU -t": c_tg.threads, "load s": m.load_s_warm,   # CPU -t = the thread count that won
    "GSM8K-50 %": q.accuracy.astype(str) + " ±" + q.se_pp.astype(str), "agree F16 %": q.agree_with_f16,
    "ΔPPL %": ((p.ppl / p.ppl["F16"] - 1) * 100).round(2), "mean KLD": p.mean_kld,   # relative PPL change vs F16
})
(ROOT / "results/table.md").write_text(tbl.to_markdown())                     # to_markdown needs the `tabulate` package (installed in 0c)
print(tbl.to_markdown())
```

Write your effective CPU bandwidth (the larger of `membw.py`'s number × 1.5 and ~70 % of the DIMM theoretical) into `results/cpu_bw_effective.txt` first; it only affects the dashed CPU ceiling line.

### Test

```bash
# Render both PNGs and the table; ls -la lists the PNGs with sizes (a 0-byte file = a silent matplotlib failure).
python bench/plot.py && ls -la results/plots/ && cat results/table.md
```

Then the real test: read the README top to bottom and, for every number, name the script and CSV it came from. Delete any number you cannot.

### Expected

Graph 1: GPU decode points rising left-to-right (in bpw terms: falling as bpw increases), always under the dashed ceiling, with the gap between measured and ceiling widening at Q4_K_M; CPU decode tracking its ceiling more tightly; both pp512 series roughly horizontal. Graph 2: agreement stays > 95 % through Q5_K_M and drops at Q4_K_M; KLD rises by roughly an order of magnitude per step from Q8_0 to Q4_K_M. The master table has 4 rows and no blanks.

### Reading a failure

| Symptom | Cause |
|---|---|
| Ceiling line below a measured point | Wrong `params` or bandwidth constant; the CPU figure is a guess — check `cpu_bw_effective.txt` against the DIMM spec |
| GPU pp512 series has a big dip at one level | Rerun that one config; if it persists, note it as a kernel-path difference (MMQ vs cuBLAS) rather than hiding it |
| Table has blanks in `mbu` | Only GPU tg128 rows carry MBU by design; the plot script uses those rows |
| `loc[ORDER]` KeyError | A level is missing from one CSV; rerun the corresponding script for that level |
| `to_markdown` raises `ImportError: Missing optional dependency 'tabulate'` | `pip install tabulate` in the venv |

### Close M4

```
git add bench/plot.py results/plots/ results/table.md README.md
git commit -m "M4: tradeoff table, decode-vs-bpw and quality-vs-bpw plots, shipping recommendation for chat/batch/edge with bandwidth extrapolation to Snapdragon"
```

README structure (in this order): problem (one paragraph: the team bullet and what it lacked) → setup (M1 section) → method (M2/M3 sections: what was measured, how stability and determinism were handled, the SE caveat) → **results table + two graphs** → what surprised you (prefill flat and non-monotonic; Q4 gains less than bytes predict; agreement dropping before accuracy; KV size independent of weight quant) → what I'd ship and why → Snapdragon extrapolation → what's next (imatrix delta if you did it; AWQ-int4 in vLLM on the same prompts; KV cache quantization `-ctk q8_0 -ctv q8_0` — the llama.cpp flags that set the K and V cache storage types — as a separate lever; FP8 needs Ada/Hopper, so not on this card) → reproduce (five commands).

---

## Closing — turning this into interview answers

### The resume bullet

The old bullet claimed a team result about Qwen2-VL on Snapdragon. The replacement is first person, has your numbers, and names the method. Template — fill every bracket from `results/table.md`:

> Quantized Qwen2.5-1.5B-Instruct [/ Qwen2-VL-2B] to Q8_0 / Q5_K_M / Q4_K_M with llama.cpp and measured the full speed–memory–quality tradeoff on an RTX 2060 and CPU-only: Q4_K_M cut weight memory [3.1]× and raised decode throughput [2.2]× on GPU / [2.8]× on CPU with prefill unchanged (±[15] %), at [Δ] pts on a fixed GSM8K slice ([N] % agreement with fp16, KLD [x]); Q8_0 was lossless (KLD [y]). Wrote the ship/no-ship recommendation per use case and extrapolated to Snapdragon-class bandwidth by the same memory-bound model that predicted both measured platforms within [z] %.

If the original team work was real, keep one honest clause about it *before* this ("Contributed to a team effort quantizing Qwen2-VL for Snapdragon; rebuilt the measurement solo:"), and let the numbers be yours. Every "why?" an interviewer asks now lands on a CSV.

### Question bank D, answered from your own data

**D1 — weight-only vs weight+activation.** "Everything I ran is weight-only: K-quants store int4/int5 blocks with scales and dequantize inside the kernel; compute stayed fp16 (int8 dot-products in the MMQ prefill path, but that is a kernel detail, not a stored format). That is exactly why my prefill was flat across four levels while decode scaled with bytes — weight-only removes bytes, not FLOPs. To speed up prefill I'd need weight+activation quantization — SmoothQuant INT8 or FP8 — which needs calibration for activation scales and hardware with int8/fp8 tensor cores on the compute path. My 2060 has int8 tensor cores, which is why MMQ prefill kept up with cuBLAS fp16, but no FP8."

**Why int4 helps decode, not prefill** (the Module 5 exercise 3 one-pager, now with numbers). "At batch 1 decode reads 3.1 GB per token at F16 and 1.0 GB at Q4_K_M; on 336 GB/s that's 9 ms vs 3 ms — I measured [X] vs [Y] tok/s, a [2.2]× gain against a 3.1× bytes reduction, with the shortfall being ~2 ms of fixed per-step overhead that quantization can't touch. Prefill at 512 tokens does 2 × 1.5 B × 512 ≈ 1.6 TFLOP regardless of storage format; my pp512 numbers were [A]/[B]/[C]/[D] — within ±[15] % and not even monotonic."

**D2 — activation outliers.** You didn't hit them, and you should say why: "Outlier channels become a problem when you quantize *activations* per-tensor; weight-only schemes never quantize activations, so they sidestep it, and at 1.5B the outlier phenomenon is mild anyway — LLM.int8 reports it emerging around 6B+. The place I saw the same idea in weight-only form is the importance matrix: `llama-imatrix` records mean squared activation per input channel and `llama-quantize` uses it to protect the weights those channels multiply — that's AWQ's insight in llama.cpp clothing, and in my run it cut Q4_K_M's KLD by [P] %."

**D3 — what breaks at Q4, when Q8 is free.** "Q8_0 was free on every axis I measured: KLD [~0.003], top-1 unchanged [99] % of the time, [same] GSM8K score. Q5_K_M was within noise on the task and [~1] % PPL. Q4_K_M is where it shows: agreement with fp16 dropped to [88] % before accuracy moved outside its ±7-point confidence interval — the model takes different reasoning paths and sometimes recovers. Small models degrade more than the 7B+ figures people quote; each weight carries more. And the honest caveat: 50 questions can't separate Q8 from Q5, KLD can, and neither tells you about the task you actually ship — so I'd gate a production change on our own eval set."

**D4 — KV cache quantization.** "Separate lever, untouched by weight quantization — my KV buffer was the same [115] MB at 4k context for every level. Weight quantization freed [2.1] GB, which at 28 KB/token is ~75k tokens of KV — that's the batch/context headroom argument for quantizing even when speed is already enough. Quantizing the KV itself (`-ctk q8_0 -ctv q8_0` in llama.cpp, `kv_cache_dtype=fp8` in vLLM) would halve that 28 KB again and also cut the per-token bytes decode reads at long context."

**J8 — proving no regression.** Your `eval/` directory *is* the answer: fixed prompt set, seeds, greedy diff against the fp16 reference, KLD on a logits sample, one task eval with a stated confidence interval, one command.

### The five-minute narrative

**0:00 — Problem.** "A resume bullet about quantizing a VL model for Snapdragon was a team result I couldn't defend number by number. I rebuilt the measurement solo, on hardware I own, with an eval, so I could."

**0:45 — Method.** "Same base checkpoint, four storage formats: fp16 and three llama.cpp K-quants at 8.5, 5.7 and 4.8 bits per weight. Two platforms — a 2060 at 336 GB/s and a desktop CPU at ~[40] GB/s effective — because two points let you validate a model and predict a third. Prefill and decode benchmarked separately, ten repetitions, medians. Quality from two signals: a fixed 50-question GSM8K slice with greedy decoding, plus KL divergence against fp16 on wikitext, because accuracy on 50 items has a seven-point error bar and KLD doesn't."

**1:45 — The graph.** "Decode tokens/sec tracked one over bytes per weight on both platforms, sitting at 60–80 % of the bandwidth ceiling for fp16 and less at Q4 because ~2 ms of fixed per-step overhead doesn't shrink. Prefill was flat — all four levels within ±[15] % and not even monotonic, because it's compute-bound and weight-only quantization doesn't remove FLOPs. That pair of panels is the memory-bound/compute-bound distinction, measured."

**2:45 — The surprise.** "Q8 was completely free. Q5 was within noise on the task. Q4 showed up first not as accuracy loss but as *disagreement* with fp16 — it changed its reasoning path on [12] % of questions and got some of them right anyway. And the KV cache didn't change size at all across levels, which is the thing people get wrong: weight quantization buys you KV headroom, it doesn't shrink KV."

**3:45 — Decision.** "For single-user chat on that card I'd ship Q8: quality-free, and a 1.5B model already decodes faster than anyone reads, so Q4's speed buys nothing while its quality costs something. For throughput serving, Q4 or Q5 depending on the task's quality floor, because the freed VRAM is ~18 more concurrent 4k requests. For a phone at 50–100 GB/s, Q4 is the only option that clears 20 tok/s — by the same arithmetic that predicted both platforms I measured."

**4:30 — Next.** "The obvious extensions are the importance matrix delta, the same prompts through AWQ-int4 in vLLM to compare families, and KV quantization as the second lever. FP8 wasn't available to me — it needs Ada or Hopper — and that's the honest boundary of what this card can teach."

Practise it out loud with the graphs on screen. Time it. If it runs over five minutes, cut from the method section, never from the graph or the decision.
