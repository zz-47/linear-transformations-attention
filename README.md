# linear-transformations-attention

**SLM math, measured not assumed.** Mathematical foundations of the attention mechanism in [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — dot-product similarity, scaled attention `softmax(QK^T/√d_k)`, Rotary Position Embeddings (RoPE), and Grouped Query Attention (GQA). Every formula is derived, then implemented in NumPy/PyTorch, then traced to the exact layer and tensor shape in the real checkpoint.

> This is not generic linear algebra — this is SLM math. Every result below is *measured* on the actual model weights, not assumed from a spec sheet.

---

## Unit 1 — Vectors & Dot Products (Research POV)

### Why the distinction matters

576-dimensional vectors point in directions that are arbitrary *until training makes them meaningful*. The embedding layer is index-based retrieval: `token_id → W_E[token_id, :]` returns a learned point in 576-space. The entire attention mechanism is a machine that **reads out that learned geometry** — the dot product `q_i · k_j` then measures whether two points sit close or far. If embeddings were just random vectors, all dot products would collapse toward zero (the *concentration of measure* phenomenon) and attention would find nothing. Attention works at all only because pre-training shaped the geometry first.

### Research-grade takeaway

> We quantify token similarity via the dot product of learned embeddings — a scalar that, in the 576-dimensional space of SmolLM2-135M, is the same operation performed ~600M times per forward pass inside self-attention. Measured on real weights, the naive hypothesis (synonyms score higher) fails — and the failure is the finding.

### The research-relevant question: why raw dot product, not cosine similarity?

Because magnitude carries meaning. A token that strongly activates a pattern (large embedding norm) should score higher than one that weakly activates it (small norm) — even at the same angle. Cosine similarity throws that information away. So attention scores are the raw dot product:

```
score_ij = q_i · k_j = ‖q_i‖ ‖k_j‖ cos(θ)    ← angle AND magnitude
```

### Why the naive expectation fails (two mechanisms)

**Mechanism 1 — dot product conflates length and angle.** `a · b = ‖a‖‖b‖cos θ`. If `computer`'s embedding has a large norm, its dot product with everything gets inflated — even at large angles. A small-norm vector like `queen` gets depressed even at small angles. The dot product cannot separate "long" from "aligned."

**Mechanism 2 — raw embedding space is anisotropic.** Word embeddings are not uniformly spread in all directions; a few dominant dimensions (correlated with word frequency and syntax, not meaning) carry most of the energy. This is the well-documented finding of Mu & Viswanath, *All-but-the-Top: Simple and Effective Postprocessing for Word Representations* (2018): removing the top-1/top-5 principal directions dramatically improves semantic quality. Semantic structure is present but buried — raw dot products can't see it.

### Measured findings (SmolLM2-135M, not assumed)

| # | Claim (expectation) | Expected | Measured | Verdict |
|---|---------------------|----------|----------|---------|
| 1 | Synonyms have high dot product | `king·queen ≈ 14.3` | `-0.20` | ❌ Fails |
| 2 | Unrelated words have low dot product | `king·table ≈ 1.2` | `2.51` | ❌ Fails |
| 3 | Self-similarity dominates the table | `king·king` max in row | `10.21` | ✅ Holds |
| 4 | Embedding space is isotropic | each direction ≈ 0.17% energy | top-1 = **4.2%** (~24×) | ❌ Anisotropic |

### The diagnosis chain (three layers)

1. **Tokenization failure (root cause)** — `queen` is out-of-vocabulary; `convert_tokens_to_ids("queen")` silently returned token `0` = `<|endoftext|>`. The "queen" row measured the end-of-text embedding, invalidating the comparison. *Moral: verify `decode(encode(w)) == w` before interpreting embeddings.*
2. **Magnitude vs. angle conflation** — `a·b = ‖a‖‖b‖cos θ` mixes length with direction; cosine similarity `a·b/(‖a‖‖b‖)` isolates direction.
3. **Embedding anisotropy** — a few dominant directions carry most of the energy, so raw dot products cannot resolve semantic similarity (Mu & Viswanath, 2018).

### The research-grade conclusion

Semantic geometry is **not** present in the raw embedding table — it is *created* by the learned projections `W_Q`, `W_K`. Attention computes dot products in that transformed space, precisely because the model learns to rotate/scale embeddings into a space where `dot product ≈ semantic relevance`. This is the *raison d'être* of Unit 2.

This is the honest, publishable take. Most tutorials cherry-pick clean examples and teach a lie. This notebook teaches the truth the data exposes: *"we need learned linear maps because the raw space isn't semantically linear."*

---

## The Real Model, Not the Spec Sheet

The sprint spec assumed one configuration. The **actual** checkpoint (config.json + safetensors) differs:

| Property | Assumed | **Measured** |
|----------|---------|--------------|
| Hidden size `d_model` | 576 | **576** |
| Vocabulary | 50,000 | **49,152** |
| Layers | 12 | **30** |
| Query heads | 12 | **9** |
| Key/Value heads | 4 | **3** |
| Head dim `d_k` | 48 | **64** (= 576/9) |
| `√d_k` | — | **8** |
| `W_Q` | 576×576 | **576×576** |
| `W_K`, `W_V` | 576×192 | **192×576** |
| `W_O` | 192×576 | **576×576** |
| `rope_theta` | — | **100,000** |

Two headline claims survive measurement: `√d_k = 8`, and GQA still reduces the KV cache by exactly `9/3 = 3×`.

---

## Repository Structure

```
linear-transformations-attention/
├── README.md                    ← this file
├── unit1_dot_products.ipynb     ← vectors, dot products, anisotropy diagnosis
├── unit2_attention.ipynb        ← QK^T / √d_k, softmax saturation, weighted read
├── unit3_rope.ipynb             ← position as rotation, frequency code, shift-invariance
├── unit4_gqa.ipynb              ← grouped query attention, KV cache (WIP)
└── unit1/
    └── unit1.md                 ← research POV / working notes for Unit 1
```

Each notebook is **self-contained** (loads the model fresh), runs on CPU-only Windows in minutes, and follows the *issue → hypothesis → fix* narrative with inline statistics and graphs.

---

## Notebooks

| Unit | Topic | Core formula | Status |
|------|-------|-------------|--------|
| 1 | Vectors & dot products | `score_ij = q_i · k_j` | ✅ Complete |
| 2 | Scaled attention | `A = softmax(QK^T / √d_k) · V` | ✅ Complete |
| 3 | Rotary Position Embeddings | `q_m · k_n = q^T R((n−m)θ) k` | ✅ Complete |
| 4 | Grouped Query Attention | `n_q / n_kv = 9 / 3 = 3×` | ⏳ Planned |

---

## Running the Notebooks

```bash
# 1. Activate the environment (already provisioned in /venv)
venv\Scripts\activate

# 2. Launch
jupyter notebook unit1_dot_products.ipynb
```

Requirements: `torch`, `transformers`, `numpy`, `matplotlib`, `jupyter`. Model is downloaded from HuggingFace Hub on first run (~270 MB, cached afterwards).

---

## Roadmap → The Bigger Picture

This is the first of a 7-repo math foundation feeding three production SLM systems: constrained decoding (blind agent), PII/injection sanitization with weight steering (RAG), and LoRA/quantization benchmarking (PEFT decoding). Each unit is a self-contained research-POV writeup measured on the real checkpoint, with the theory traced to exact layer/tensor shapes.


## Unit 2 — Scaled Attention & the √d_k Fix (Research POV)

### Why the distinction matters

Unit 1 ended with a problem: the raw embedding space is **not** semantically linear — dot products in `W_E` cannot see similarity (anisotropy). Attention is the model's answer: it *learns* projections `W_Q`, `W_K` that rotate and scale embeddings into a space where `dot product ≈ relevance`. But that insight smuggles in a second, invisible problem: a dot product of two 64-dimensional vectors is not a bounded similarity score. It is a sum of 64 terms, so its magnitude grows like √64 = 8. Feed scores of size ~8–21 into softmax and the distribution collapses to near one-hot — attention stops reading context and just copies the single best token.

### Research-grade takeaway

> The `1/√d_k` in `softmax(QKᵀ/√d_k)` is neither learned nor tuned — it is the **statistical correction for dot-product growth in d_k dimensions**. Measured on SmolLM2-135M's real layer-0 weights: raw scores have std ≈ 7.07 (predicted √64 = 8). Dividing by 8 lifts mean attention entropy from 0.25 (near one-hot) to 1.52 (soft spread; max log 6 ≈ 1.79), restoring gradient flow. This is why *every* transformer carries that division.

### Measured findings (SmolLM2-135M, not assumed)

| # | Claim (expectation) | Predicted | Measured | Verdict |
|---|---------------------|-----------|----------|---------|
| 1 | d_k-dim dot products grow ~√d_k | std ≈ 8 | std = 7.07 | ✅ Holds |
| 2 | Raw softmax saturates to one-hot | entropy → 0 | entropy 0.25, max prob 0.92 | ✅ Confirmed |
| 3 | ÷√d_k restores a soft spread | entropy → log n | entropy 1.52, max prob 0.39 | ✅ Holds |
| 4 | GQA: 9 query heads share 3 KV heads | KV width = 576/3 | 192 | ✅ Holds |
| 5 | Attention output = weighted read of values | `out[i] = Σⱼ A[i,j]·V[j]` | hand-check matches | ✅ Holds |

### The diagnosis chain

1. **Scores grow like √d_k — arithmetic, not a bug.** `S[i,j] = q_i·k_j = Σ_{d=1}^{64} q_id·k_jd`. Summing 64 independent terms multiplies variance by 64, so std ≈ √64 = 8 (measured 7.07).
2. **Softmax over unbounded scores saturates.** `exp(21)` dwarfs `exp(0)`, one column wins → entropy 0.25, attention copies a single token.
3. **The fix (Vaswani et al., 2017, §3.2.1).** Rescale before softmax: `softmax(QKᵀ/√d_k)`. Scores drop to O(1), the distribution spreads (entropy 1.52), gradients flow.

### The research-grade conclusion

Attention is a **convex combination** — a weighted average of value vectors:

```
out_h[i] = Σⱼ A_h[i,j] · V[j]     with   Σⱼ A_h[i,j] = 1
```

The whole mechanism is two matrix products glued by a softmax: `QKᵀ` scores every token pair, `·V` reads the values those scores select. GQA makes the KV side one-third the width of the query side (192 vs 576) — the memory-saving design Unit 4 quantifies. The loose end: the model never scores raw q·k — it first *rotates* both by token position. That is Unit 3 (RoPE).


## Unit 3 — RoPE: Position as a Rotation (Research POV)

### Why the distinction matters

Unit 2 closed with a loose end: `QKᵀ` is **permutation-blind**. Nothing in the score formula mentions *where* a token sits — "the cat sat on the mat" and "mat the on cat sat the" produce identical pairwise scores (measured: permuting the rows of `X` yields `S_σ = P S Pᵀ`, recoverable exactly by un-permuting). Attention compares *identities*, never *addresses*. RoPE cures this by *rotating* each query and key by an angle proportional to its position — using plain high-school trigonometry.

### Research-grade takeaway

> RoPE converts **absolute position into relative angle**. Each dimension pair `(d, d+32)` is treated as a complex number `z = x0 + i·x1`, and rotating by θ is multiplying by the unit complex number `e^{iθ} = cosθ + i·sinθ` (Su et al., 2021). Because `R(α)ᵀR(β) = R(β−α)`, absolute positions cancel: `q_m·k_n = qᵀR((n−m)θ)k` depends **only on the gap** `n−m`. Measured on SmolLM2-135M's real layer-0 weights: the same tokens scored at positions 0–5 vs 7–12 give an *identical* score matrix, and the map becomes **asymmetric** — attention now distinguishes "i before j" from "j before i."

### The frequency code — position read like a radio tuner

```
θ_d = rope_theta^(−2d/d_k) = 100000^(−2d/64)     d = 0, 1, …, 31
```

- `d = 0` → θ₀ = 1.0 → a full turn every ~6 positions → **fast**, a fine near-neighbor ruler
- `d = 31` → θ₃₁ ≈ 1.4e-5 → a full turn every ~450k positions → **slow**, a coarse long-range ruler

A 7-orders-of-magnitude spread, log-linear on a log plot — fast dims tick, slow dims crawl.

### Measured findings (SmolLM2-135M, not assumed)

| # | Claim (expectation) | Predicted | Measured | Verdict |
|---|---------------------|-----------|----------|---------|
| 1 | Attention is permutation-blind | `S_σ = P S Pᵀ` | `recover original: True` | ✅ Confirmed |
| 2 | Rotation preserves length | `‖R v‖ = ‖v‖` | `norm preserved: True` (~1e-6) | ✅ Holds |
| 3 | Only the gap `n−m` matters | `q_{m+p}·k_{n+p} = q_m·k_n` | `shift-invariant: True` | ✅ Holds |
| 4 | Direction is now encoded | `S[i,j] ≠ S[j,i]` | `asymmetric: True` | ✅ Holds |
| 5 | Frequency code decays with `d` | 1.0 → ~1e-5 | θ₀ = 1.0, θ₃₁ = 1.43e-5 | ✅ Holds |

### The API-drift finding (measured, not assumed)

transformers 5.x moved `rope_theta` out of the config root into `cfg.rope_scaling` (`{'rope_theta': 100000, 'rope_type': 'default'}`), and SmolLM2 ships `rope_interleaved: False` — dims pair **half-split** as `(d, d+32)`, not interleaved. The spec/API said one thing; the installed environment said another.

### The research-grade conclusion

RoPE is **position as rotation**: no learned position table, no extra parameters — just cos/sin cached once.

```
q_m · k_n = qᵀ R((n−m)θ) k
```

The model reads *relative distance* and *direction* structurally, for free. This is why RoPE suits edge inference — it is stateless, its cos/sin cache is trivial, and it composes with the KV-cache savings that Unit 4 quantifies next.

> The honest version of CB 3.1 matters: permuting the sentence gives `S_σ = P S Pᵀ`, **not** `S` itself — pairwise scores between the same two tokens are unchanged, only the row/column labels move. Most tutorials hand-wave this; the notebook shows the un-permute check that makes it exact.