# Project 1 — `inference-from-scratch`: the complete walkthrough

**Companion to** `04-project-specs.md` (the spec: milestones, acceptance, interview story) and `01-study-guide.md` Modules 1–3 (the theory). This document is the *how*: every milestone, in order, with the concepts explained from zero, the real Hugging Face code to read first, the code to write (fully commented), the test that proves it works, the numbers you should see, and what it means when you don't.

**Repo layout when you're done**

```
inference-from-scratch/
├── qwen_from_scratch.py     # the model: RMSNorm, RoPE, Attention, MLP, DecoderLayer, MyQwen, KVCache
├── generate.py              # generate_naive (M2), generate_cached (M3), Sampler (M4)
├── test_parity.py           # M1 acceptance test: logits match HF to 1e-4
├── test_generate.py         # naive == cached == HF generate; sampler tests
├── bench.py                 # M5: regenerates every graph in the README
├── results/                 # CSVs + PNGs written by bench.py (commit these)
├── dev_test_components.ipynb  # your scratch notebook (keep it; it shows the process)
└── README.md                # the engineering-blog-style write-up
```

**Dependencies:** `torch` (CUDA build for your 2060), `transformers`, `matplotlib`. On Windows, use WSL2 if you can — everything works natively too, but `torch.compile` and some profilers are WSL/Linux-only, and Project 2 will want Linux.

**How to work through this:** each milestone has a *Look first* step (print the real thing, read its source), then *Concepts*, then *Code*, then *Test*, then *Expected*, then *Reading a failure*. Do the Look-first step every time even when you think you know — that habit is what made your attention block correct on the first real try.

---

## Concepts you need before starting

Read this once now, and come back whenever a word in a later section is unfamiliar. Everything here is defined in plain language; the precise version appears where it is used.

**Token.** Models don't read characters or words; they read *tokens*, integer ids from a fixed vocabulary. Qwen2.5's vocabulary has 151,936 entries. "Hello, how are you?" becomes six ids. A **tokenizer** is the lookup that converts text to ids and back. Byte-level BPE (what Qwen uses) means any byte sequence can be tokenized, and a single token can be a fragment of a word or even part of a multi-byte character.

**Embedding.** A big table of shape `[vocab_size, hidden_size]` = `[151936, 896]`. Token id 42 maps to row 42, a vector of 896 numbers. This is the model's first layer: ids in, vectors out. The 896 is the **hidden size** (also `d_model`): the width of the vector every token carries through the whole network.

**Hidden state / residual stream.** The `[batch, seq_len, 896]` tensor flowing through the model. Each of the 24 layers reads it and adds a small correction. Think of it as the model's working memory for each token.

**Logits.** The model's final output: for each position, a vector of 151,936 raw scores, one per vocabulary entry. Higher score = the model thinks that token is more likely to come next. Logits are *not* probabilities yet.

**Softmax.** Turns a vector of raw scores into probabilities that sum to 1: `p_i = exp(x_i) / Σ exp(x_j)`. Applied to logits it gives the next-token distribution; applied to attention scores it gives attention weights.

**Greedy decoding.** Pick the token with the highest logit every step. Deterministic, and prone to repeating itself (Milestone 4 shows why).

**Sampling.** Instead of the argmax, draw a token at random with probability proportional to the softmax. Temperature, top-k and top-p are knobs on that draw (defined in M4).

**Transformer decoder block.** The repeating unit. Each block has two sub-blocks: **attention** (lets each token gather information from earlier tokens) and an **MLP** (transforms each token's vector on its own). Both are wrapped in a **residual connection** (`x = x + f(norm(x))`) so the block adds to the stream instead of replacing it. Qwen2.5-0.5B stacks 24 of these.

**Attention, Q/K/V.** For every token we compute three vectors from its hidden state: a **query** ("what am I looking for?"), a **key** ("what do I contain?"), and a **value** ("what do I hand over if someone attends to me?"). Token i's attention score for token j is `q_i · k_j` (a dot product). Softmax over all j gives weights; the output for token i is the weighted sum of the `v_j`. **Causal** attention means token i may only look at tokens j ≤ i — the future is masked out with `-inf` before the softmax.

**Heads.** Rather than one 896-dim attention, the vectors are split into 14 **heads** of 64 dims each and attention runs independently per head, then the results are concatenated. Different heads learn to look for different things. `head_dim = 896 / 14 = 64`.

**GQA (grouped-query attention).** Qwen2.5-0.5B has 14 query heads but only 2 key/value heads. Each K/V head is shared by 7 query heads. It exists to make the KV cache 7× smaller (see below) at almost no quality cost.

**RoPE (rotary position embedding).** Attention alone has no idea of *order* — `q_i · k_j` doesn't know whether j is next to i or 500 tokens away. RoPE fixes that by rotating each q and k vector by an angle proportional to its position, so the dot product ends up depending on the *distance* between positions. `theta` (1,000,000 for Qwen2.5) sets how fast the angles change.

**RMSNorm.** A normalization layer that rescales each token's vector to unit root-mean-square, then multiplies by a learned per-dimension weight. It keeps activations in a sane range so 24 layers can stack. It is LayerNorm without the mean subtraction.

**Weights / parameters / state_dict.** The learned numbers. Qwen2.5-0.5B has ~494 million of them. In PyTorch they live in `nn.Parameter` objects; `model.state_dict()` is a dict of `name → tensor`, and `load_state_dict` copies one into another *by name* — which is why your class and attribute names must match Hugging Face's exactly.

**dtype.** How each number is stored. `float32` = 4 bytes, `float16`/`bfloat16` = 2 bytes. Halving the bytes halves memory traffic, which (next item) roughly doubles decode speed. Your 2060 has no bf16 tensor cores, so use `float16` on it; do the parity tests in `float32` because 1e-4 agreement is not reachable in 16-bit.

**Prefill vs decode.** Generating text has two phases. **Prefill** processes the whole prompt in one forward pass — many tokens at once, lots of arithmetic per byte of weight read: *compute-bound*. **Decode** produces one token per step, and each step has to read every weight in the model to do a tiny amount of arithmetic: *memory-bandwidth-bound*. This distinction is the single most-asked inference interview question.

**Memory bandwidth and the decode ceiling.** Your 2060 can move 336 GB/s between VRAM and the compute units. If one decode step must read ~0.99 GB of fp16 weights, the fastest it can possibly run is 336 / 0.99 ≈ 340 steps per second — the **bandwidth ceiling**. Everything you measure in M3 is compared against this number.

**KV cache.** During decode, the new token's query needs the keys and values of *every earlier token*. Without a cache you recompute all of them every step (M2, slow). With a cache you store each token's K and V the first time you see it and just read them back (M3, fast). The cache for Qwen2.5-0.5B costs `2 (K and V) × 24 layers × 2 kv_heads × 64 head_dim × 2 bytes = 12,288 bytes per token` in fp16.

**tok/s, latency, `torch.cuda.synchronize()`.** Tokens per second = 1 / (seconds per decode step). GPU kernels run asynchronously — Python returns before the GPU finishes — so a timer around a forward pass measures nothing unless you call `torch.cuda.synchronize()` before stopping the clock.

**Seed / RNG.** A random-number generator started from the same seed produces the same sequence. Sampling with a fixed seed makes "random" output reproducible, which you need for tests and for the README examples.

---

## Step 0 — Housekeeping (15 min)

### Look first

```python
import inspect
from transformers.models.qwen2 import modeling_qwen2 as mq

print(hf_model.model.layers[0].input_layernorm)          # Qwen2RMSNorm((896,), eps=1e-06)
print(inspect.getsource(mq.Qwen2RMSNorm.forward))
```

Notice three things in HF's `forward`: it casts to `float32` first, it uses `torch.rsqrt(variance + eps)`, and it casts back to the *input* dtype before multiplying by `weight`. Your current version does none of the casting. In fp32 that's invisible; in fp16 (which you'll switch to for speed) it drifts.

### 0a. Make the notebook pick up edits to the `.py` file

Python imports a module once per kernel. Edit `qwen_from_scratch.py` afterwards and the kernel keeps running the old code — which is exactly what produced the `inf` you chased for an evening. First cell of the notebook, forever:

```python
%load_ext autoreload   # Jupyter extension that re-imports changed modules
%autoreload 2          # mode 2 = reload everything before every cell runs
```

Restart the kernel once after adding it.

### 0b. RMSNorm — match HF exactly

Replace your class with this. Default `eps` becomes `1e-6` (Qwen's value; `1e-8` was a bug waiting for you to forget the argument).

```python
class RMSNorm(nn.Module):
    """
    Root-Mean-Square normalisation.

    LayerNorm subtracts the mean and divides by the standard deviation.
    RMSNorm skips the mean subtraction and only divides by the root-mean-square:
        y = x / sqrt(mean(x^2) + eps) * weight
    It is cheaper (one reduction instead of two) and works just as well for
    transformers, so every modern LLM (Llama, Qwen, Mistral) uses it.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # One learnable scale per hidden dimension. Shape: [hidden_size].
        # Initialised to ones so the layer starts as "do nothing but normalise".
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # Small constant added inside the sqrt so we never divide by zero.
        # Qwen2.5 uses 1e-6 (config.rms_norm_eps).
        self.eps = eps

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        in_dtype = x.dtype
        # HF does the arithmetic in fp32 even when the model runs in bf16/fp16,
        # because squaring small numbers in 16-bit loses precision. We match that.
        x = x.float()
        # mean of squares over the hidden dimension -> [batch, seq_len, 1]
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt = 1/sqrt. Multiply instead of divide (same result, faster on GPU).
        x = x * torch.rsqrt(variance + self.eps)
        # Cast back to the input dtype, then apply the learned per-dimension scale.
        return self.weight * x.to(in_dtype)
```

### 0c. The attention module returns one tensor

Your `AttentionProjections.forward` returns ten tensors, eight of them debug clones. With 24 layers that is memory you don't have on a 6 GB card, and a call signature nobody can read. The M1 section below gives you the full rewritten `Attention` class (renamed — it stopped being "just projections" when you added RoPE and softmax). It is your verified code with the debug returns removed and two forward-looking changes: `cos`/`sin` are passed in rather than computed per layer (HF does the same; computing them 24 times per step is waste), and it accepts an optional `kv_cache` that M3 will use. Read every comment — this file is the one you'll rebuild from memory in the 30-minute drill.

---

## Milestone 1 — Your own model, logit parity with HF

**What "done" means:** `python test_parity.py` prints `M1 parity PASSED`, meaning `my_model(input_ids)` and `hf_model(input_ids).logits` agree to within 1e-4 on a real sentence, in fp32, without your code calling any HF model code.

You already have the hardest pieces verified — embeddings, RMSNorm, and the whole GQA attention block matched HF to ~1e-5. What's left is assembling them.

### Look first — the map of the model

```python
print(hf_model)                       # the whole tree: Qwen2ForCausalLM -> model (Qwen2Model) -> layers[0..23], norm; lm_head
print(hf_model.model.layers[0])       # one block: self_attn, mlp, input_layernorm, post_attention_layernorm
for name, p in hf_model.named_parameters():
    print(name, tuple(p.shape))       # every weight, by the exact name load_state_dict will look for
```

Things to notice: there are *four* named children in each layer and you must create exactly those four with exactly those names. `lm_head.weight` is listed, but check `hf_model.lm_head.weight.data_ptr() == hf_model.model.embed_tokens.weight.data_ptr()` — same memory. That's tied embeddings. And `model.rotary_emb` lives once at the model level, not inside each layer.

### 1a. The MLP

#### Look first

```python
print(hf_model.model.layers[0].mlp)
print(inspect.getsource(mq.Qwen2MLP.forward))
```

You'll see `down_proj(act_fn(gate_proj(x)) * up_proj(x))` and, in `__init__`, three `nn.Linear(..., bias=False)`. `act_fn` is SiLU (`x * sigmoid(x)`) — check `hf_model.config.hidden_act`.

#### Concepts

The **MLP** (also "feed-forward network") is the half of each block that works on one token at a time: no mixing between positions. It is **gated**: two projections go *up* from 896 to 4864 dims — `gate_proj` and `up_proj` — and one comes *down*. The gate goes through SiLU and is multiplied elementwise with `up`; the multiplication lets the network switch intermediate features on and off per token. This variant is called SwiGLU. Roughly two thirds of the model's parameters are in these three matrices (3 × 896 × 4864 ≈ 13 M per layer, × 24).

#### Code

```python
class MLP(nn.Module):
    """
    Gated feed-forward network (SwiGLU):
        out = down_proj( silu(gate_proj(x)) * up_proj(x) )
    Two projections go UP 896 -> 4864, one comes DOWN 4864 -> 896.
    The gate decides, per intermediate feature, how much of `up` gets through.
    All three have NO bias in Qwen2. ~2/3 of the model's parameters live here.
    """

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size          # 896
        inter = config.intermediate_size     # 4864
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        # x: [batch, seq, 896] -> [batch, seq, 4864] (twice) -> elementwise product -> [batch, seq, 896]
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

#### Test

```python
hf_mlp = hf_model.model.layers[0].mlp
my_mlp = MLP(hf_model.config).to(hf_model.device)
# load_state_dict replaces every hand-written copy_() — it cannot forget a weight
# the way the o_proj copy was forgotten. strict=True errors on any name mismatch.
my_mlp.load_state_dict(hf_mlp.state_dict(), strict=True)

x = torch.randn(4, 64, hf_model.config.hidden_size, device=hf_model.device)   # [batch, seq, hidden]
with torch.no_grad():
    print("MLP max diff:", (hf_mlp(x) - my_mlp(x)).abs().max().item())
```

**Expected:** `0.0` or ~1e-7.

### 1b. The attention block, final form

#### Look first

```python
print(inspect.getsource(mq.Qwen2Attention.forward))
print(inspect.getsource(mq.apply_rotary_pos_emb))
print(inspect.getsource(mq.repeat_kv))
print(inspect.getsource(mq.Qwen2Model.forward))     # find where rotary_emb is called ONCE and passed down
```

Notice in `Qwen2Attention.forward` the order: project → reshape to heads → RoPE → `past_key_values.update(...)` → attention. The cache update happens *after* RoPE and *before* attention. Your M3 will do the same.

#### Concepts

Everything in this class you have already built and verified; the new pieces are the `kv_cache` argument (inert until M3) and the mask that handles both prefill and decode. The **causal mask** rule in one sentence: a query at absolute position `p` may attend to keys at positions `≤ p`. When a cache holds `past_len` earlier tokens and this call adds `q_len` new ones, query `i` (0-based within the new tokens) sits at position `past_len + i`, so it may see keys `0 … past_len + i`. That is a `[q_len, kv_len]` mask whose forbidden region is the strict upper triangle offset by `past_len`. With no cache, `past_len = 0` and it's your familiar triangle; with a one-token decode step, `q_len = 1` and nothing is masked.

#### Code

The helpers first, then the class.

```python
class RotaryEmbedding(nn.Module):
    """
    Produces the cos/sin tables that `apply_rope` uses to rotate Q and K.

    RoPE encodes *position* by rotating each pair of dimensions of the Q and K
    vectors by an angle proportional to the token's position. Because a dot
    product between two rotated vectors depends only on the *difference* of
    their angles, attention scores end up depending on relative distance —
    which is exactly what we want.
    """

    def __init__(self, config):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads  # 896 // 14 = 64
        # Qwen2.5 uses theta = 1,000,000 (not the original 10,000).
        # HF stores it under config.rope_parameters["rope_theta"] in transformers 5;
        # older versions used config.rope_theta. Support both.
        theta = getattr(config, "rope_theta", None)
        if theta is None:
            theta = config.rope_parameters["rope_theta"]
        # inv_freq[i] = 1 / theta^(2i/head_dim)  for i in 0..head_dim/2-1
        # Low i -> fast-rotating dims (fine position detail),
        # high i -> slow-rotating dims (coarse position detail).
        # Shape: [head_dim/2] = [32]
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        # persistent=False: this buffer is derived from the config, not learned,
        # so it must NOT appear in state_dict (otherwise load_state_dict(strict=True)
        # complains about an unexpected key).
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):
        # position_ids: [batch, seq_len] integer positions of each token.
        # Compute angle = position * inv_freq for every (position, frequency) pair.
        # einsum "bs,d->bsd": [batch, seq] x [32] -> [batch, seq, 32]
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), self.inv_freq)
        # HF's "rotate-half" convention duplicates the angles so the first half
        # of head_dim and the second half use the same angles: [batch, seq, 64]
        emb = torch.cat((freqs, freqs), dim=-1)
        # Return cos and sin with a head axis inserted so they broadcast against
        # q/k of shape [batch, num_heads, seq, head_dim]:  [batch, 1, seq, 64]
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
```

```python
def rotate_half(x):
    # Split the last dim into two halves and swap them with a sign flip:
    # [x1, x2] -> [-x2, x1]. Combined with cos/sin below this is a 2-D rotation
    # applied to the pairs (x[i], x[i + head_dim/2]).
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)
```

```python
def apply_rope(x, cos, sin):
    # x:   [batch, heads, seq, head_dim]
    # cos: [batch, 1,     seq, head_dim]  (broadcasts over heads)
    # Standard RoPE formula in the rotate-half layout.
    return x * cos + rotate_half(x) * sin
```

```python
def repeat_kv(x, n_rep):
    """
    Grouped-Query Attention helper.

    Qwen2.5-0.5B has 14 query heads but only 2 key/value heads. Each K/V head
    is shared by 14/2 = 7 query heads. To do a plain batched matmul we expand
    K and V so there is one copy per query head.

    x: [batch, num_kv_heads, seq, head_dim] -> [batch, num_kv_heads * n_rep, seq, head_dim]

    Ordering matters: head j of the output must be KV head (j // n_rep), i.e.
    K0,K0,...,K0,K1,K1,...,K1 — NOT K0,K1,K0,K1,... . `expand` on an inserted
    axis followed by reshape gives exactly that (same as HF's repeat_kv).
    """
    if n_rep == 1:
        return x
    b, kv_heads, seq, hd = x.shape
    x = x[:, :, None, :, :].expand(b, kv_heads, n_rep, seq, hd)
    return x.reshape(b, kv_heads * n_rep, seq, hd)
```

```python
class Attention(nn.Module):
    """
    Grouped-Query causal self-attention, exactly as in HF's Qwen2Attention.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx                              # which KVCache slot is ours
        self.hidden_size = config.hidden_size                   # 896
        self.num_heads = config.num_attention_heads             # 14 query heads
        self.num_kv_heads = config.num_key_value_heads          # 2 key/value heads
        self.head_dim = self.hidden_size // self.num_heads      # 64
        self.n_rep = self.num_heads // self.num_kv_heads        # 7 query heads per KV head
        self.scaling = self.head_dim ** -0.5                    # 1/sqrt(64) = 0.125

        # Qwen2 quirk: q/k/v projections HAVE a bias, o_proj does NOT.
        # Names must match HF exactly so load_state_dict works.
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)      # 896 -> 896
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)   # 896 -> 128
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)   # 896 -> 128
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)     # 896 -> 896

    def forward(self, x, cos, sin, kv_cache=None):
        # x:   [batch, q_len, hidden]  — q_len is the number of NEW tokens this call
        #      (whole prompt during prefill, 1 during a cached decode step)
        # cos/sin: [batch, 1, q_len, head_dim] — RoPE tables for the NEW positions only
        batch, q_len, _ = x.shape

        # 1) Project to Q, K, V and split into heads.
        #    view: [batch, q_len, heads*head_dim] -> [batch, q_len, heads, head_dim]
        #    transpose(1,2): -> [batch, heads, q_len, head_dim]  (heads first so matmul batches over them)
        q = self.q_proj(x).view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2) Rotate Q and K by their positions. V carries no position information.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 3) KV cache: store the NEW (already-rotated) K/V and get back the FULL K/V.
        #    Rotation must happen BEFORE caching so cached keys already carry their position.
        past_len = 0
        if kv_cache is not None:
            past_len = kv_cache.get_seq_len(self.layer_idx)     # tokens cached for THIS layer before this call
            k, v = kv_cache.update(self.layer_idx, k, v)        # now [batch, kv_heads, past_len + q_len, head_dim]
        kv_len = k.shape[2]                                     # total keys each query can look at

        # 4) GQA: expand the 2 KV heads to 14 so shapes line up with Q.
        k = repeat_kv(k, self.n_rep)                            # [batch, 14, kv_len, head_dim]
        v = repeat_kv(v, self.n_rep)

        # 5) Scores = Q · K^T / sqrt(head_dim):  [batch, 14, q_len, kv_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # 6) Causal mask. Query i (absolute position past_len + i) may attend to
        #    key j only if j <= past_len + i. Build a [q_len, kv_len] boolean mask
        #    that is True where attention is NOT allowed, and fill those with -inf.
        #    During a single-token decode step q_len == 1 and the mask is all-False
        #    (the new token may see everything), so this is a no-op there.
        if q_len > 1:
            # triu with diagonal = past_len + 1 keeps everything strictly above the
            # "current position" diagonal, i.e. the future.
            future = torch.triu(
                torch.ones(q_len, kv_len, dtype=torch.bool, device=x.device),
                diagonal=past_len + 1,
            )
            scores = scores.masked_fill(future, float("-inf"))

        # 7) Softmax over keys, in fp32 for numerical stability, cast back.
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        # 8) Weighted sum of values: [batch, 14, q_len, head_dim]
        out = torch.matmul(probs, v)

        # 9) Merge heads back: -> [batch, q_len, 14*64 = 896], then output projection.
        out = out.transpose(1, 2).reshape(batch, q_len, self.num_heads * self.head_dim)
        return self.o_proj(out)
```

#### Test

Same as your existing attention test, but the rotary tables now come from your own `RotaryEmbedding` and the class takes a `layer_idx`:

```python
hf_attn = hf_model.model.layers[0].self_attn
my_attn = Attention(hf_model.config, layer_idx=0).to(hf_model.device)
my_attn.load_state_dict(hf_attn.state_dict(), strict=True)          # all 7 tensors, including o_proj
rope = RotaryEmbedding(hf_model.config).to(hf_model.device)

x = torch.randn(4, 64, 896, device=hf_model.device)
pos = torch.arange(64, device=x.device).unsqueeze(0).expand(4, 64)   # [batch, seq] positions 0..63
cos, sin = rope(pos)                                                  # [batch, 1, seq, 64]
# HF wants an additive float mask: 0 where allowed, -inf where not: [1, 1, seq, seq]
mask = torch.triu(torch.full((64, 64), float("-inf"), device=x.device), 1)[None, None]

with torch.no_grad():
    hf_out, _ = hf_attn(x, position_embeddings=hf_model.model.rotary_emb(x, pos[:1]), attention_mask=mask)
    my_out = my_attn(x, cos, sin)
print("Attention max diff:", (hf_out - my_out).abs().max().item())
```

**Expected:** ~1e-5.

### 1c. The decoder layer

#### Look first

```python
print(inspect.getsource(mq.Qwen2DecoderLayer.forward))
```

Read the two `residual = hidden_states` lines and where each norm is applied. That's the whole block.

#### Concepts

A **residual connection** means the block computes `x + f(x)` rather than `f(x)`: it *adds a correction* to the stream instead of replacing it. This is what lets gradients pass through 24 layers during training and, at inference, what makes each layer a refinement of the previous one. **Pre-norm** means the norm sits on the input to `f`, never on the residual path itself. `post_attention_layernorm` is HF's (confusing) name for the norm before the MLP; keep it.

#### Code

```python
class DecoderLayer(nn.Module):
    """
    One transformer block, pre-norm residual style:
        h   = x + Attention( input_layernorm(x) )
        out = h + MLP( post_attention_layernorm(h) )
    The residual stream `x` is never normalised itself — each sub-block reads a
    normalised copy and ADDS a correction back. That is what lets 24 of these
    stack without the signal blowing up or vanishing.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Misleading HF name: this norm is applied BEFORE the MLP, on the residual
        # stream after attention has been added. Keep the name; it must match HF.
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin, kv_cache=None):
        # --- attention sub-block ---
        residual = x                                  # keep the un-normalised stream
        x = self.input_layernorm(x)                   # normalise a copy
        x = self.self_attn(x, cos, sin, kv_cache)     # attention output, same shape as x
        x = residual + x                              # add back (residual connection)
        # --- MLP sub-block ---
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x
```

#### Test

```python
hf_layer = hf_model.model.layers[0]
my_layer = DecoderLayer(hf_model.config, layer_idx=0).to(hf_model.device)
my_layer.load_state_dict(hf_layer.state_dict(), strict=True)

x = torch.randn(4, 64, 896, device=hf_model.device)
pos = torch.arange(64, device=x.device).unsqueeze(0).expand(4, 64)
cos, sin = rope(pos)
mask = torch.triu(torch.full((64, 64), float("-inf"), device=x.device), 1)[None, None]
with torch.no_grad():
    hf_out = hf_layer(x, position_embeddings=hf_model.model.rotary_emb(x, pos[:1]), attention_mask=mask)
    if isinstance(hf_out, tuple):          # older transformers return (hidden, attn_weights)
        hf_out = hf_out[0]
    my_out = my_layer(x, cos, sin)
print("Decoder layer max diff:", (hf_out - my_out).abs().max().item())
```

**Expected:** ~1e-5. This is the last component test.

### 1d. The full model and the tied head

#### Look first

```python
print(inspect.getsource(mq.Qwen2Model.forward))          # embed -> position ids -> rotary once -> for layer in layers -> norm
print(inspect.getsource(mq.Qwen2ForCausalLM.forward))    # calls self.model, then self.lm_head on the hidden states
print(hf_model.config.tie_word_embeddings)               # True
```

#### Concepts

Mirroring HF's two-level structure (`Qwen2ForCausalLM.model` is a `Qwen2Model`) means the weight load is one line: `my.model.load_state_dict(hf.model.state_dict())`. The **LM head** projects each 896-dim hidden state to 151,936 logits. With `tie_word_embeddings=True` that projection *is* the embedding table transposed — there is no separate matrix to load, and creating one would double the largest tensor in the model. `position_ids` are computed from the cache length so M3 works without changing this code.

#### Code

```python
class KVCache:
    """
    Stores the Key and Value tensors of every layer for all tokens seen so far.

    Why this works: K and V for a token depend only on that token's hidden state
    and the layer's weights — never on tokens that come *after* it. So once a
    token's K and V are computed, they never change and can be reused for every
    later decode step instead of being recomputed.

    Layout: one (K, V) pair per layer, each of shape
        [batch, num_kv_heads, seq_len_so_far, head_dim]
    """

    def __init__(self, num_layers):
        # None until the first forward pass fills each layer.
        self.k = [None] * num_layers
        self.v = [None] * num_layers

    def get_seq_len(self, layer_idx=0):
        """
        Number of cached positions for one layer (0 if empty).

        Why per-layer: inside a forward pass, layers update the cache one after
        another. While layer 5 runs, layers 0-4 have ALREADY appended this call's
        tokens and layers 5-23 have not. Asking "how long is layer 0's cache?"
        from inside layer 5 would over-count by q_len and break the causal mask
        during prefill. Between forward passes all layers agree, so the model
        can safely ask layer 0 before it starts.
        """
        return 0 if self.k[layer_idx] is None else self.k[layer_idx].shape[2]

    def update(self, layer_idx, k_new, v_new):
        """
        Append this step's K/V for one layer and return the FULL K/V
        (cached + new) that attention should use.

        k_new/v_new: [batch, num_kv_heads, new_tokens, head_dim]
        returns:     [batch, num_kv_heads, past + new_tokens, head_dim]
        """
        if self.k[layer_idx] is None:
            # First call (prefill): the cache for this layer IS the new K/V.
            self.k[layer_idx] = k_new
            self.v[layer_idx] = v_new
        else:
            # Later calls (decode): concatenate along the sequence axis (dim=2).
            # torch.cat allocates a new tensor every step — O(n) copy per step.
            # Fine for this project; Project 2 replaces it with pre-allocated blocks.
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k_new], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v_new], dim=2)
        return self.k[layer_idx], self.v[layer_idx]
```

```python
class QwenModel(nn.Module):
    """
    embed_tokens -> 24 x DecoderLayer -> final norm.  Returns hidden states.
    Mirrors HF's `Qwen2Model` so `load_state_dict(hf_model.model.state_dict())` just works.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Lookup table: token id -> 896-dim vector. Shape [151936, 896].
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # nn.ModuleList (not a plain list!) so .to(device), .parameters() and
        # load_state_dict all see the layers.
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # One RoPE table generator shared by all layers (HF does the same).
        self.rotary_emb = RotaryEmbedding(config)

    def forward(self, input_ids, kv_cache=None, return_hidden=False):
        # input_ids: [batch, q_len] — the NEW tokens only when a cache is used.
        batch, q_len = input_ids.shape
        # Absolute positions of the new tokens. With an empty cache that is 0..q_len-1;
        # during decode it is past_len..past_len+q_len-1. Getting this wrong is the
        # #1 KV-cache bug: the new token would be rotated as if it were at position 0.
        past_len = 0 if kv_cache is None else kv_cache.get_seq_len()
        position_ids = torch.arange(past_len, past_len + q_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch, q_len)              # [batch, q_len]
        cos, sin = self.rotary_emb(position_ids)                      # [batch, 1, q_len, head_dim]

        h = self.embed_tokens(input_ids)                              # [batch, q_len, 896]
        hidden_states = [h]                                           # for the parity test only
        for layer in self.layers:
            h = layer(h, cos, sin, kv_cache)
            hidden_states.append(h)
        h = self.norm(h)
        return (h, hidden_states) if return_hidden else h
```

```python
class MyQwen(nn.Module):
    """
    Causal LM head on top of QwenModel. Mirrors HF's `Qwen2ForCausalLM`.

    Qwen2.5-0.5B has tie_word_embeddings=True: the output projection REUSES the
    embedding matrix transposed, so there is no separate lm_head weight to load.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = QwenModel(config)

    def forward(self, input_ids, kv_cache=None, return_hidden=False, last_only=False):
        out = self.model(input_ids, kv_cache=kv_cache, return_hidden=return_hidden)
        h, hidden_states = out if return_hidden else (out, None)
        if last_only:
            # During generation we only need the logits of the LAST position.
            # Skipping the other positions saves a [q_len x 896 x 151936] matmul
            # during prefill. h: [batch, q_len, 896] -> [batch, 1, 896]
            h = h[:, -1:, :]
        # [batch, q_len, 896] @ [896, 151936] -> [batch, q_len, 151936] raw scores per vocab entry
        logits = h @ self.model.embed_tokens.weight.T
        return (logits, hidden_states) if return_hidden else logits

    @classmethod
    def from_hf(cls, hf_model):
        """Build our model and copy every weight from a loaded HF Qwen2ForCausalLM."""
        m = cls(hf_model.config).to(hf_model.device, dtype=next(hf_model.parameters()).dtype)
        # strict=True: any missing/unexpected key is a naming bug and raises immediately.
        m.model.load_state_dict(hf_model.model.state_dict(), strict=True)
        return m.eval()
```

### 1e. The acceptance test — `test_parity.py`

Not a notebook cell: a script anyone can run. The spec says every README claim must be reproducible by a script.

```python
"""test_parity.py — Milestone 1 acceptance: our logits match Hugging Face's to 1e-4."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_from_scratch import MyQwen

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# fp32 on purpose: 1e-4 agreement is not reachable in fp16.
tok = AutoTokenizer.from_pretrained(MODEL)
hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device).eval()
my = MyQwen.from_hf(hf)                         # builds our model and load_state_dict(strict=True)

ids = tok("The capital of France is", return_tensors="pt").input_ids.to(device)   # [1, 6]
with torch.no_grad():
    hf_out = hf(ids, output_hidden_states=True)   # .hidden_states = tuple of 25 tensors: embeddings + after each layer
    my_logits, my_hidden = my(ids, return_hidden=True)

# Layer-by-layer comparison. The first index where the diff jumps is where the bug is.
# HF applies the final norm to its LAST hidden state, ours are pre-norm, so compare 0..23 only.
for i in range(len(my_hidden) - 1):
    print(f"hidden[{i:2d}] max diff = {(hf_out.hidden_states[i] - my_hidden[i]).abs().max().item():.2e}")

diff = (hf_out.logits - my_logits).abs().max().item()
print(f"logits max diff = {diff:.2e}")
print("HF next token:", repr(tok.decode(hf_out.logits[0, -1].argmax())))
print("My next token:", repr(tok.decode(my_logits[0, -1].argmax())))
assert diff < 1e-4, "M1 parity FAILED"
print("M1 parity PASSED")
```

**Expected:** hidden-state diffs start at `0.00e+00` and grow slowly with depth (floating-point error accumulates: maybe 1e-5 by layer 23); logits within 1e-4; both print ` Paris`.

### Reading a failure

| Symptom | Where the bug is |
|---|---|
| `load_state_dict` raises *Unexpected key* `...rotary_emb.inv_freq` | Your `register_buffer` lacks `persistent=False` |
| `load_state_dict` raises *Missing/Unexpected key* with another name | An attribute name differs from HF's — the message tells you which |
| `hidden[0]` already differs | Embedding weights not loaded (did `from_hf` run?) |
| `hidden[1]` differs, `hidden[0]` fine | Layer 0: re-run the 1b/1c tests |
| Fine until layer N, then a jump | Layer N loaded wrong weights — inspect `my.model.layers[N]` |
| All hidden states fine, logits wrong | The tied head: you must use `embed_tokens.weight.T` |
| Everything off by ~1e-2, growing with depth | dtype mismatch — one model is fp16; test in fp32 |
| Everything off by ~1e-3 | RMSNorm eps (1e-8 vs 1e-6) or missing fp32 upcast |
| Notebook shows old behaviour after an edit | Stale kernel: autoreload not on, or restart the kernel |

### Close M1

```
git add qwen_from_scratch.py test_parity.py
git commit -m "M1: Qwen2.5-0.5B forward pass in plain PyTorch, logit parity with HF to 1e-4"
```

README section: "Building the model" — one paragraph on the architecture (24 blocks, GQA 14/2, RoPE θ=1e6, SwiGLU, tied head), the parity test output pasted verbatim, and the one bug that cost you the most time (write it honestly; interviewers love that paragraph).

---

## Milestone 2 — The naive generation loop

**What "done" means:** `generate_naive` produces text, and you have a CSV + plot of decode tok/s against context length showing the number *falling* as context grows.

### Look first

```python
from transformers import generation
print(inspect.getsource(generation.utils.GenerationMixin._sample))   # the real loop; long — skim for the `while` and the forward call
```

Find the loop: it calls `self(**model_inputs)` each iteration, takes `outputs.logits[:, -1, :]`, picks a token, and `torch.cat`s it onto `input_ids`. Then time HF doing it *without* its cache — this is the naive loop, and it gives you a reference number:

```python
ids = tok("Once upon a time", return_tensors="pt").input_ids.to(device)
import time
for use_cache in (False, True):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    hf.generate(ids, max_new_tokens=64, do_sample=False, use_cache=use_cache)
    torch.cuda.synchronize(); print(f"use_cache={use_cache}: {64/(time.perf_counter()-t0):.1f} tok/s")
```

### Concepts

Text generation is **autoregressive**: the model predicts one token, you append it to the input, and predict again. The naive loop does exactly that, re-running the *whole* sequence every step. Step t therefore processes t tokens through 24 layers, and attention inside each layer compares t queries with t keys — `O(t²)` work per step, `O(n³)` for an n-token generation. This is what the M2 graph shows, and M3 exists to fix it.

`last_only=True` trims only the final `[t, 896] @ [896, 151936]` matmul to the last row. It does *not* make the loop fast — every layer still processes all t tokens. It's there so that the M2 → M3 comparison isolates the cache, not the head.

### Code — `generate.py`, part 1

```python
@torch.no_grad()
def generate_naive(model, input_ids, max_new_tokens, eos_id=None):
    """
    The deliberately slow way: every step re-runs the ENTIRE sequence through
    the model, so step t costs O(t) tokens of work (and attention costs O(t^2)).

    input_ids: [1, prompt_len] on the model's device
    returns:   (all_ids [1, prompt_len + n], per-step latencies in seconds)
    """
    step_times = []
    for _ in range(max_new_tokens):
        t0 = time.perf_counter()
        # Full forward over everything generated so far. last_only=True only
        # trims the final matmul; all 24 layers still process every token.
        logits = model(input_ids, last_only=True)             # [1, 1, vocab]
        next_id = logits[0, -1].argmax()                       # greedy: highest score
        if input_ids.is_cuda:
            torch.cuda.synchronize()                           # GPU work is async; wait so the timer is honest
        step_times.append(time.perf_counter() - t0)
        # Append the new token. torch.cat makes a new [1, len+1] tensor each step.
        input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return input_ids, step_times
```

### Test

```python
ids = tok("Once upon a time", return_tensors="pt").input_ids.to(device)
out, steps = generate_naive(my, ids, 32, eos_id=tok.eos_token_id)
print(tok.decode(out[0]))
hf_out = hf.generate(ids, max_new_tokens=32, do_sample=False)
print("matches HF greedy:", torch.equal(out[:, :hf_out.shape[1]], hf_out))
print(f"median step: {sorted(steps)[len(steps)//2]*1000:.1f} ms")
```

Then the measurement that becomes the first README graph. Load the model in **fp16** for this (`torch_dtype=torch.float16` in `from_pretrained`, then `from_hf`); fp32 halves your ceiling.

```python
import statistics
filler = tok("The quick brown fox jumps over the lazy dog. " * 300, return_tensors="pt").input_ids
for ctx in [64, 128, 256, 512, 1024]:
    ids = filler[:, :ctx].to(device)
    generate_naive(my, ids, 4)                       # warm-up: never time the first CUDA call
    _, steps = generate_naive(my, ids, 16)
    print(f"ctx={ctx:5d}  naive {1/statistics.median(steps):6.1f} tok/s")
```

### Expected

On the 2060 in fp16, something like 25–40 tok/s at 64 tokens of context, falling to single digits by 1024. The exact numbers don't matter; the *shape* does — a curve that drops steadily as context grows. Greedy text must match HF's `generate` exactly (both are deterministic argmax over the same logits).

### Reading a failure

| Symptom | Cause |
|---|---|
| Text differs from HF greedy after a few tokens | M1 parity is marginal; a tiny logit difference flipped an argmax tie. Check `test_parity.py` in fp32 first |
| tok/s doesn't fall with context | You're timing without `synchronize()` — you're measuring Python, not the GPU |
| First measurement wildly slow | Warm-up missing (kernel loading, cudnn autotuning) |
| CUDA out of memory at 1024 context | fp32 weights (2 GB) + activations for 1024 tokens. Use fp16 |
| Output is `!!!!!` or garbage in fp16 but fine in fp32 | RMSNorm without the fp32 upcast (Step 0b), or softmax not in fp32 |

### Close M2

```
git commit -am "M2: naive greedy loop + tok/s vs context sweep"
```

---

## Milestone 3 — The KV cache

**What "done" means:** `generate_cached` produces *identical* tokens to `generate_naive`, decode tok/s is flat-ish across context lengths and several times higher, and the README explains the gap to the 340 tok/s bandwidth ceiling. And you can rebuild this milestone from a blank file in 30 minutes.

### Look first

```python
from transformers.cache_utils import DynamicCache
print(inspect.getsource(DynamicCache.update))       # the real thing is a torch.cat along dim=-2 per layer
print(inspect.getsource(mq.Qwen2Attention.forward)) # see where .update() is called: after RoPE, before attention
```

Then watch the cache grow:

```python
cache = DynamicCache()
with torch.no_grad():
    hf(ids, past_key_values=cache, use_cache=True)
print(cache.get_seq_length())                                          # == prompt length
k0 = cache.layers[0].keys if hasattr(cache, "layers") else cache.key_cache[0]  # API differs by version
print(k0.shape)                                                        # [1, 2, prompt_len, 64]  <- 2 KV heads, not 14
```

Two kv heads, not fourteen: that's GQA saving you 7× cache memory, in front of your eyes.

### Concepts

**Why caching is legal.** A token's key and value vectors at layer L are computed from *that token's* hidden state at layer L. Hidden states depend only on the token itself and the tokens *before* it (causal attention never looks forward). So once token j has been processed, its K and V at every layer are final — nothing that comes later can change them. Recomputing them every step (M2) is pure waste.

**What decode looks like with a cache.** Prefill runs the prompt once and stores K/V for every layer. Each decode step then feeds *one* token: 24 layers each compute one new K and V (append to cache), and one query that attends over all cached keys. Per-step work is now `O(t)` (reading the cache) instead of `O(t²)`, and the dominant cost becomes reading the weights — 0.99 GB per step regardless of t. That's why the curve flattens.

**The bandwidth ceiling.** Batch-1 decode reads all weights once per token: `336 GB/s ÷ 0.99 GB ≈ 340 tok/s` is the physical limit on your card in fp16. You'll land well under it. The gap is the interesting part of the README: per step you also launch ~24 × 10 small kernels, each with a few microseconds of fixed overhead (that's Python and the CUDA launch queue, not bandwidth); the cache read adds `12,288 bytes × t` per step; `torch.cat` copies the whole cache every step (Project 2 fixes that with pre-allocated blocks). Measure, don't guess: `torch.profiler` in Module 11 shows you the split.

**Positions with a cache.** The new token is at absolute position `past_len`, not 0. If you rotate it as position 0, output degrades subtly rather than crashing — the nastiest bug in this milestone. `QwenModel.forward` derives positions from `kv_cache.get_seq_len()` so this can't go wrong as long as you never bypass it.

**Memory.** `12,288 bytes/token` → a 2048-token context costs 25 MB. Tiny for one stream; Project 2 is where the cache becomes the thing that limits batch size.

### Code

The model side is already in place from M1: `KVCache`, the `kv_cache` argument threaded through `Attention → DecoderLayer → QwenModel → MyQwen`, the per-layer `get_seq_len(layer_idx)` and the mask offset. Re-read those pieces now with M3 eyes — especially the comment on `get_seq_len` explaining why it takes a `layer_idx`. The generation side:

```python
@torch.no_grad()
def generate_cached(model, input_ids, max_new_tokens, eos_id=None, sampler=None):
    """
    Two phases:
      prefill — run the whole prompt ONCE, filling the cache for every layer;
      decode  — feed ONE new token per step; attention reads the cache.
    Step cost no longer grows with the sequence (except the cache read itself).

    sampler: optional callable logits[vocab] -> token id (Milestone 4).
             None means greedy (argmax).
    """
    cache = KVCache(model.config.num_hidden_layers)
    step_times = []

    # --- prefill: the only call that sees more than one token ---
    t0 = time.perf_counter()
    logits = model(input_ids, kv_cache=cache, last_only=True)  # fills cache; [1, 1, vocab]
    if input_ids.is_cuda:
        torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    generated = []
    next_id = _pick(logits[0, -1], sampler)

    # --- decode: one token in, one token out, cache grows by 1 each step ---
    for _ in range(max_new_tokens):
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        t0 = time.perf_counter()
        # Only the NEW token goes in. The model computes its position from the
        # cache length, so we do not pass position_ids explicitly.
        new_ids = torch.tensor([[next_id]], device=input_ids.device)  # [1, 1]
        logits = model(new_ids, kv_cache=cache, last_only=True)
        if input_ids.is_cuda:
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        next_id = _pick(logits[0, -1], sampler)

    out = torch.tensor([generated], device=input_ids.device)
    return torch.cat([input_ids, out], dim=1), prefill_time, step_times
```

```python
def _pick(logits, sampler):
    # logits: [vocab] for the last position. Returns a python int.
    if sampler is None:
        return logits.argmax().item()
    return sampler(logits)
```

### Test — `test_generate.py`

```python
"""test_generate.py — cached decode must equal naive decode must equal HF greedy generate."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_from_scratch import MyQwen, KVCache
from generate import generate_naive, generate_cached

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device).eval()
my = MyQwen.from_hf(hf)

ids = tok("The three laws of robotics are", return_tensors="pt").input_ids.to(device)
naive, _ = generate_naive(my, ids, 24)
cached, _, _ = generate_cached(my, ids, 24)
with torch.no_grad():
    ref = hf.generate(ids, max_new_tokens=24, do_sample=False, min_new_tokens=24)
print("naive == cached      :", torch.equal(naive, cached))
print("cached == HF generate:", torch.equal(cached[:, :ref.shape[1]], ref))

# Stronger check: at every decode step, logits from the cached path must equal
# the logits of a full forward over the same sequence. Catches wrong positions
# and wrong masks that happen to produce the same argmax for a while.
cache = KVCache(my.config.num_hidden_layers)
with torch.no_grad():
    my(ids, kv_cache=cache)
    seq, worst = ids, 0.0
    for _ in range(8):
        nxt = torch.randint(0, my.config.vocab_size, (1, 1), device=device)   # random next token: harder than greedy
        seq = torch.cat([seq, nxt], dim=1)
        step_logits = my(nxt, kv_cache=cache)[0, -1]     # one token through the cached path
        full_logits = my(seq)[0, -1]                     # whole sequence, no cache
        worst = max(worst, (step_logits - full_logits).abs().max().item())
print("cached-step vs full-forward logits, max diff:", worst)
assert worst < 1e-3
print("M3 correctness PASSED")
```

Then the measurement (fp16 model):

```python
for ctx in [64, 128, 256, 512, 1024, 2048]:
    ids = filler[:, :ctx].to(device)
    generate_cached(my, ids, 4)                                   # warm-up
    _, prefill, steps = generate_cached(my, ids, 32)
    print(f"ctx={ctx:5d}  cached {1/statistics.median(steps):6.1f} tok/s   prefill {prefill*1000:5.0f} ms")
```

### Expected

`naive == cached` and `cached == HF generate` both `True`; per-step logit diff ~1e-4 in fp32. Decode tok/s roughly flat across context — on a 2060 in fp16 with this plain-PyTorch implementation, expect somewhere in the range of 80–150 tok/s at short context, drifting down a little by 2048 as the `torch.cat` and cache read grow. That is 25–45% of the 340 tok/s ceiling. Prefill time grows roughly linearly with prompt length (it's compute-bound, so it's the one place the GPU is actually busy).

Whatever you measure, write *those* numbers. "118 tok/s, 35% of ceiling" is a better README line than a made-up 80%.

### Reading a failure

| Symptom | Cause |
|---|---|
| `naive == cached` False, text drifts after a few tokens | Positions: new token rotated at position 0. Check `past_len` in `QwenModel.forward` |
| Step-logit diff ~1e-4 for the first token, then huge | Mask offset wrong in prefill: `get_seq_len` read another layer's already-updated length (this exact bug was in the first draft of this code) |
| Cached output is fine but *identical* tok/s to naive | The loop is passing the whole sequence each step instead of only the new token |
| Shape error in `torch.cat` inside `KVCache.update` | K/V cached *before* `transpose(1, 2)`, or cached after `repeat_kv` (14 heads) then concatenated with 2-head tensors |
| Works at batch 1, wrong for batch > 1 | Expected for now — batching with padding is Project 2 |
| tok/s far below 80 | Profile it (Module 11). Usual suspects: fp32 weights, `synchronize` inside the layer loop, running on CPU by accident |

### Close M3

```
git commit -am "M3: KV cache — cached decode == naive == HF; tok/s vs context before/after"
```

README: this is the centrepiece. The before/after graph; the ceiling line; a paragraph on where the gap goes, backed by one profiler screenshot; and the sentence you'll say in interviews: "batch-1 decode reads every weight per token, so the ceiling is bandwidth/bytes; I measured X, and the rest is launch overhead and cache copies, which is exactly what CUDA graphs and paged KV attack in real engines."

### The 30-minute drill

Once a week until it's boring: blank file, write `KVCache` and a cached attention forward from memory, run `test_generate.py`. The first time takes an hour. By the third it's twenty minutes, and "implement a KV cache" in an interview becomes a formality.

---

## Milestone 4 — Sampling

**What "done" means:** `Sampler(temperature, top_k, top_p, seed)` turns logits into a token; temperature 0 reproduces greedy exactly; the same seed reproduces the same text; and you have a README example where greedy loops and sampling doesn't.

### Look first

```python
from transformers.generation import logits_process as lp
print(inspect.getsource(lp.TemperatureLogitsWarper.__call__))   # scores / temperature
print(inspect.getsource(lp.TopKLogitsWarper.__call__))          # everything below the k-th value -> filter_value (-inf)
print(inspect.getsource(lp.TopPLogitsWarper.__call__))          # sort, cumsum of softmax, keep the nucleus, scatter back
```

Note the *order* HF applies them in (`_get_logits_processor`: temperature, then top-k, then top-p) and that filtered tokens become `-inf` — which softmax turns into exactly zero probability.

### Concepts

Logits are scores; softmax makes them a **probability distribution** over the 151,936 tokens. **Greedy** takes the argmax and, because the most likely continuation of a sentence is often "more of the same," it falls into loops ("the the the", or a paragraph repeated verbatim). **Sampling** draws from the distribution instead, so the output varies and rarely loops. Three knobs shape the distribution before the draw:

*Temperature* divides the logits by T. T < 1 makes peaks sharper (more conservative), T > 1 flattens them (more random), T → 0 collapses to greedy. *Top-k* keeps only the k highest-scoring tokens. *Top-p* (nucleus) keeps the smallest set of tokens whose probabilities add up to at least p — adaptive: a confident distribution keeps few tokens, an uncertain one keeps many. A **seeded generator** makes the "random" draw repeatable: same prompt, same seed, same text, every time.

### Code — `generate.py`, part 2

```python
class Sampler:
    """
    Turns a logits vector into a token id using temperature, top-k, top-p,
    with a seeded RNG so a given (prompt, seed) always produces the same text.

    Order of operations (same as HF's default logits-processor chain):
        temperature -> top-k -> top-p -> softmax -> multinomial draw
    """

    def __init__(self, temperature=1.0, top_k=0, top_p=1.0, seed=0, device="cpu"):
        self.temperature = temperature      # <1 sharpens, >1 flattens, ->0 becomes greedy
        self.top_k = top_k                  # keep only the k highest logits (0 = off)
        self.top_p = top_p                  # keep the smallest set whose prob mass >= p (1.0 = off)
        # A private RNG so sampling is reproducible regardless of anything else
        # that uses torch's global RNG.
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def __call__(self, logits):
        # logits: [vocab], float. Work in fp32 for stable softmax.
        logits = logits.float().clone()

        # Temperature 0 (or tiny) == greedy. Avoid dividing by zero.
        if self.temperature <= 1e-5:
            return logits.argmax().item()
        logits = logits / self.temperature

        # top-k: everything outside the k largest logits gets -inf (prob 0).
        if self.top_k > 0:
            kth = torch.topk(logits, self.top_k).values[-1]          # k-th largest value
            logits[logits < kth] = float("-inf")

        # top-p (nucleus): sort descending, keep the shortest prefix whose
        # cumulative probability reaches p, drop the rest.
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            # Mask tokens where the cumulative mass BEFORE them already exceeded p.
            # Shifting by one guarantees the top token is always kept.
            remove = cum - probs > self.top_p
            sorted_logits[remove] = float("-inf")
            # Scatter back to vocabulary order.
            logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)                         # -inf -> exactly 0
        # Draw one token id proportional to probs, using OUR generator.
        return torch.multinomial(probs, num_samples=1, generator=self.gen).item()
```

### Test

Append to `test_generate.py`:

```python
from generate import Sampler
logits = torch.randn(151936, device=device) * 4               # a fake, spiky logits vector

# 1. temperature 0 must be greedy
assert Sampler(temperature=0.0, device=device)(logits) == logits.argmax().item()

# 2. same seed -> same draws; different seed -> (almost surely) different
a = Sampler(0.8, top_k=50, top_p=0.9, seed=42, device=device)
b = Sampler(0.8, top_k=50, top_p=0.9, seed=42, device=device)
assert [a(logits) for _ in range(20)] == [b(logits) for _ in range(20)]

# 3. top-k: every draw must come from the top 50
top50 = set(torch.topk(logits, 50).indices.tolist())
assert all(Sampler(1.0, top_k=50, seed=7, device=device)(logits) in top50 for _ in range(50))

# 4. top-p tiny -> only the argmax survives
assert Sampler(1.0, top_p=1e-9, seed=1, device=device)(logits) == logits.argmax().item()
print("M4 sampler PASSED")
```

Then the README example:

```python
ids = tok("Tell me something interesting. The", return_tensors="pt").input_ids.to(device)
for name, s in [("greedy", None),
                ("T=0.7 top-p=0.9 seed=0", Sampler(0.7, top_p=0.9, seed=0, device=device)),
                ("T=0.7 top-p=0.9 seed=1", Sampler(0.7, top_p=0.9, seed=1, device=device))]:
    out, _, _ = generate_cached(my, ids, 80, eos_id=tok.eos_token_id, sampler=s)
    print(f"[{name}]", tok.decode(out[0, ids.shape[1]:]))
```

### Expected

All four asserts pass. Greedy on a 0.5B model with an open-ended prompt usually repeats a phrase within 80 tokens; the sampled versions don't, and seeds 0 and 1 give different text while re-running seed 0 gives the same.

### Reading a failure

| Symptom | Cause |
|---|---|
| `multinomial` error: probabilities contain inf/nan | You divided by temperature 0 — the early-return for `temperature <= 1e-5` is missing |
| Same seed gives different text | Something else shares the RNG: you used `torch.multinomial` without `generator=` |
| Top-p drops the top token when p is small | The `cum - probs > p` shift is missing (HF does the same shift) |
| Draws outside top-k | You compared `logits < kth` on the *sorted* tensor and forgot to scatter back |
| `generator` device error | `torch.Generator(device=...)` must match the logits' device |

### Close M4

```
git commit -am "M4: temperature / top-k / top-p sampling with seeded RNG; greedy-loop vs sampled examples"
```

---

## Milestone 5 — `bench.py`

**What "done" means:** `python bench.py` on a fresh clone regenerates every CSV, every PNG and the sampling examples in `results/`, and the README embeds them. Anyone can reproduce your numbers.

### Look first

Open `vllm/benchmarks/` in the vLLM repo (or run `vllm bench --help` if you have it installed from Module 4). Notice what a serious benchmark records: warm-up excluded, medians and percentiles rather than means, model/dtype/hardware written into the output, and everything driven by CLI flags. Your harness is a small version of that.

### Concepts

A **benchmark harness** is a script that produces the numbers, not a notebook you ran once. Rules it follows: warm up before timing (first CUDA call loads kernels); `synchronize` before stopping the clock; report the **median** of many steps (a single hiccup would wreck a mean); write raw data (CSV) *and* the graph, so someone can re-plot; record the environment (GPU, dtype, model) next to the results. A graph without its CSV is a claim; with it, it's evidence.

### Code

```python
"""
bench.py — regenerates every number and graph in the README from scratch.

    python bench.py                 # full run on GPU, writes results/*.csv and results/*.png
    python bench.py --quick         # short run for smoke-testing the harness

Graphs produced:
    1. tok_s_vs_context.png   — naive vs cached decode tok/s as context grows (the money graph)
    2. step_latency.png       — per-step latency over the course of one long generation
    3. sampling_examples.md   — greedy vs sampled text for the README
"""
import argparse
import csv
import json
import os
import statistics
import time

import torch

from qwen_from_scratch import MyQwen
from generate import generate_naive, generate_cached, Sampler

# --- constants for the bandwidth-ceiling line on the graph ---
GPU_BANDWIDTH_GBS = 336.0        # RTX 2060: 336 GB/s memory bandwidth (spec sheet)


def load(model_name, dtype, device):
    """Load HF weights once, build our model from them, free the HF copy."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    hf = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model = MyQwen.from_hf(hf)
    del hf                              # we only needed it for the weights
    if device == "cuda":
        torch.cuda.empty_cache()
    return tok, model


def model_bytes(model):
    # Sum of every parameter's storage. For 0.5B in fp16 this is ~0.99 GB.
    return sum(p.numel() * p.element_size() for p in model.parameters())


def bench_context_sweep(model, tok, contexts, new_tokens, device, out_csv):
    """
    For each prompt length in `contexts`: run naive and cached generation,
    record median decode tok/s. Median, not mean — a single GC pause or clock
    ramp would otherwise skew the number.
    """
    rows = []
    # A long filler prompt we can truncate to any length.
    filler = tok("The quick brown fox jumps over the lazy dog. " * 400, return_tensors="pt").input_ids
    for ctx in contexts:
        ids = filler[:, :ctx].to(device)

        # --- warm-up: first CUDA call pays kernel-load costs; never time it ---
        generate_cached(model, ids, 4)

        # --- naive: skipped beyond a threshold because it gets painfully slow ---
        naive_tps = None
        if ctx <= 1024:
            _, steps = generate_naive(model, ids, new_tokens)
            naive_tps = 1.0 / statistics.median(steps)

        _, prefill, steps = generate_cached(model, ids, new_tokens)
        cached_tps = 1.0 / statistics.median(steps)

        rows.append(dict(context=ctx, naive_tok_s=naive_tps, cached_tok_s=cached_tps, prefill_s=prefill))
        print(f"ctx={ctx:5d}  naive={naive_tps and f'{naive_tps:6.1f}'}  cached={cached_tps:6.1f} tok/s  prefill={prefill*1000:.0f} ms")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return rows


def plot_context_sweep(rows, ceiling_tps, out_png):
    import matplotlib
    matplotlib.use("Agg")                 # headless backend: write files, no window
    import matplotlib.pyplot as plt

    ctx = [r["context"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ctx, [r["cached_tok_s"] for r in rows], marker="o", label="KV cache (M3)")
    naive = [(r["context"], r["naive_tok_s"]) for r in rows if r["naive_tok_s"]]
    if naive:
        ax.plot([c for c, _ in naive], [t for _, t in naive], marker="s", label="naive full re-forward (M2)")
    # Horizontal line: what pure memory bandwidth would allow at batch 1.
    ax.axhline(ceiling_tps, linestyle="--", color="gray", label=f"bandwidth ceiling ≈ {ceiling_tps:.0f} tok/s")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("decode tokens / s")
    ax.set_title("Qwen2.5-0.5B decode throughput, RTX 2060")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def bench_step_latency(model, tok, device, n_tokens, out_png):
    """Latency of every decode step across one long generation — shows the O(n) cache read creeping in."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = tok("Write a long story about a lighthouse keeper.", return_tensors="pt").input_ids.to(device)
    _, _, steps = generate_cached(model, ids, n_tokens)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot([s * 1000 for s in steps])
    ax.set_xlabel("decode step")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Per-step decode latency with KV cache")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def sampling_examples(model, tok, device, out_md):
    """Greedy vs sampled continuations of the same prompt, for the README."""
    prompt = "Tell me something interesting. The"
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    lines = [f"Prompt: `{prompt}`\n"]
    for name, sampler in [
        ("greedy", None),
        ("temperature 0.7, top-p 0.9, seed 0", Sampler(0.7, top_k=0, top_p=0.9, seed=0, device=device)),
        ("temperature 0.7, top-p 0.9, seed 1", Sampler(0.7, top_k=0, top_p=0.9, seed=1, device=device)),
        ("temperature 1.5, top-k 50, seed 0", Sampler(1.5, top_k=50, top_p=1.0, seed=0, device=device)),
    ]:
        out, _, _ = generate_cached(model, ids, 60, eos_id=tok.eos_token_id, sampler=sampler)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        lines.append(f"**{name}:** {text!r}\n")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    os.makedirs("results", exist_ok=True)

    tok, model = load(args.model, dtype, device)
    mb = model_bytes(model)
    # Batch-1 decode reads every weight once per token, so bandwidth / bytes is the ceiling.
    ceiling = GPU_BANDWIDTH_GBS * 1e9 / mb
    print(f"model bytes = {mb/1e9:.2f} GB -> bandwidth ceiling ≈ {ceiling:.0f} tok/s at batch 1")
    json.dump(dict(model_bytes=mb, ceiling_tok_s=ceiling, device=device, dtype=args.dtype),
              open("results/meta.json", "w"), indent=2)

    contexts = [64, 256] if args.quick else [64, 128, 256, 512, 1024, 2048]
    new_tokens = 8 if args.quick else 32
    rows = bench_context_sweep(model, tok, contexts, new_tokens, device, "results/context_sweep.csv")
    plot_context_sweep(rows, ceiling, "results/tok_s_vs_context.png")
    bench_step_latency(model, tok, device, 32 if args.quick else 512, "results/step_latency.png")
    sampling_examples(model, tok, device, "results/sampling_examples.md")
    print("done -> results/")


if __name__ == "__main__":
    main()
```

### Test

```
python bench.py --quick        # ~1 min: checks the whole pipeline
python bench.py                # the real run, ~10 min on the 2060
ls results/                    # context_sweep.csv  meta.json  sampling_examples.md  step_latency.png  tok_s_vs_context.png
```

### Expected

`tok_s_vs_context.png`: naive curve falling, cached curve roughly flat, dashed ceiling line above both. `step_latency.png`: a nearly flat line with a gentle upward drift and occasional spikes (those are Python/GC — worth a sentence in the README). `sampling_examples.md`: the greedy loop vs sampled text.

### Reading a failure

| Symptom | Cause |
|---|---|
| Plot is blank or crashes with a display error | matplotlib needs the `Agg` backend on a headless/WSL machine — `matplotlib.use("Agg")` before `pyplot` |
| First point of every sweep is an outlier | Warm-up not run for that configuration |
| Numbers differ run to run by >10% | GPU clocks ramping or another process on the GPU; run twice, keep the second; close the browser |
| Naive sweep takes forever | It's `O(n³)`; the harness caps naive at 1024 context for that reason |

### Close M5 and the project

```
git add bench.py results/
git commit -m "M5: bench.py regenerates all README graphs"
```

**README outline** (write it like a blog post, not a manual):

1. *What this is* — a 0.5B LLM inference engine in ~400 lines of PyTorch, no HF model code, with measurements at every step.
2. *Building the model* — architecture in a paragraph; the parity test output; the bug that cost you the most time.
3. *The naive loop and why it's O(n³)* — graph 1's falling curve.
4. *KV cache* — the before/after graph, the ceiling line, where the gap goes (with the profiler evidence).
5. *Sampling* — the greedy-loop example next to the sampled one.
6. *What surprised me* — three honest sentences.
7. *Reproduce* — `pip install -r requirements.txt && python test_parity.py && python bench.py`.

**The interview story this buys you.** Every question in question-bank sections A–B is now something you did. Practise the two-minute version out loud: what a forward pass does; why decode is memory-bound and what your ceiling was; what the KV cache stores, how big it is per token for this model (12 KB), and why GQA shrinks it 7×; why greedy loops. Say your measured numbers, not the spec's placeholders.

**Next:** Project 3 (`08-p3-walkthrough.md`) is the resume fix and comes before Project 2. Project 2 (`07-p2-walkthrough.md`) builds batching and paged KV on top of this engine. Its section "Interface contract with Project 1" asks for three small extensions to the code above, and gives you the code for them: `forward` must accept per-row `position_ids` (because left-padded batches put different rows at different positions) and an optional additive `attention_mask` (causal + padding in one tensor), and the cache is passed as a plain list of per-layer `(K, V)` tuples rather than the `KVCache` object. Make those changes when you start P2, not now — the `KVCache` class here is the clearest version for learning and for the 30-minute drill.
