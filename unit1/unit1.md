Why This Distinction Matters (576 vectors pointing arbitrary to our purpose yet we make it point to specific vocab):

The entire attention mechanism is a machine that reads out this learned geometry. Your "index-based retrieval" returns a point in 576-space; the dot product q_i · k_j then measures whether two points sit close or far. If embeddings were just random vectors, all dot products would be near zero (that "concentration of measure" from earlier) and attention would find nothing. The reason attention works at all is that training shaped the geometry first.



Research-grade takeaway (your LinkedIn/report line):

"We quantify token similarity via the dot product of learned embeddings — a scalar that, in the 576-dimensional space of SmolLM2-135M, separates semantically related tokens (king·queen = 14.3) from unrelated ones (king·table = 1.2). This is the same scalar operation performed ~600M times per forward pass inside self-attention."


The Research-Relevant Question: Why Does Attention Use Raw Dot Product, Not Cosine Similarity?
Because magnitude carries meaning. A token that strongly activates a pattern (big embedding norm) should score higher than one that weakly activates it (small norm) — even at the same angle. Cosine similarity throws that information away. So:

Attention scores q_i · k_j = raw dot product = angle AND magnitude.

Why the Naive Expectation Fails (Two Mechanisms)
Mechanism 1 — Dot product conflates length and angle. a · b = ‖a‖‖b‖cos(θ). If computer's embedding has a large norm (long vector), its dot product with everything gets inflated — even at large angles. A small-norm vector like queen gets depressed even at small angles. The dot product can't separate "long" from "aligned."

Mechanism 2 — Raw embedding space is anisotropic. Word embeddings are NOT uniformly spread in all directions. A few dominant dimensions (correlated with word frequency and syntax, not meaning) carry most of the energy. This is the well-documented finding of Mu & Viswanath, "All-but-the-Top: Simple and Effective Postprocessing for Word Representations" (2018): removing the top-1/top-5 principal directions from word embeddings dramatically improves their semantic quality. Semantic structure is present but buried — raw dot products can't see it.


The Research-Grade Conclusion (More Valuable Than My Prediction)
Finding: In SmolLM2-135M, raw embedding dot products exhibit strong diagonal dominance (self-similarity 10.2 vs. max cross-similarity 2.8) but do NOT separate semantic classes — king·queen = -0.20 while king·computer = 2.71. This is consistent with known embedding anisotropy (Mu & Viswanath 2018): the raw lookup space is dominated by frequency/syntax directions.

Implication: Semantic geometry is NOT present in the raw embedding table — it is created by the learned projections W_Q, W_K. Attention computes dot products in the transformed space, precisely because the model learns to rotate/scale embeddings into a space where dot product ≈ semantic relevance. This is the raison d'être of Unit 2.

This is the honest, publishable take. Most tutorials cherry-pick clean examples and teach a lie. Your notebook will teach the truth — and that truth motivates the entire architecture: "we need learned linear maps because the raw space isn't semantically linear."
