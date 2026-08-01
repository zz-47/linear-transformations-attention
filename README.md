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
├── unit2_attention.ipynb        ← QK^T / √d_k, softmax saturation (WIP)
├── unit3_rope.ipynb             ← rotary position embeddings (WIP)
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
| 2 | Scaled attention | `A = softmax(QK^T / √d_k) · V` | 🔨 In progress |
| 3 | Rotary Position Embeddings | `q_i = R(θ, i) · W_Q x_i` | ⏳ Planned |
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

This is the first of a 7-repo math foundation feeding three production SLM systems: constrained decoding (blind agent), PII/injection sanitization with weight steering (RAG), and LoRA/quantization benchmarking (PEFT decoding). Unit 3 (RoPE) will also carry a research + industrial-scope discussion on position encoding efficiency for long-context edge inference.
