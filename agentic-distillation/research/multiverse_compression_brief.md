# How Multiverse Computing Makes Models Smaller With Fewer Parameters — and How It Compares to the Rest of the Field

*Technical brief. Every quantitative claim is attributed to a model and a benchmark. Where verification found a claim overstated, the caveat is stated inline rather than the headline repeated.*

---

## 1. The Core Mechanism

CompactifAI ([arXiv:2401.14109](https://arxiv.org/abs/2401.14109), v1 Jan 2024, v2 May 2024; peer-reviewed at [ESANN 2025](https://www.esann.org/sites/default/files/proceedings/2025/ES2025-8.pdf), pp. 531–537) replaces selected weight matrices of a pretrained transformer with Matrix Product Operators (MPOs) truncated at a bond dimension χ, then runs a short "healing" retrain. Precisely:

### 1.1 Reshape

Take a weight matrix $W \in \mathbb{R}^{M \times N}$ from self-attention (Q, K, V, O) or the MLP. Factor both dimensions:

$$M = \prod_{k=1}^{n} J_k, \qquad N = \prod_{k=1}^{n} I_k$$

and reinterpret $W$ as a $2n$-index tensor $W_{j_1 \ldots j_n,\, i_1 \ldots i_n}$, pairing one input index with one output index per "site" $k$.

For a 4096×4096 attention projection with $n=4$, a natural choice is $(J_k) = (I_k) = (8,8,8,8)$.

### 1.2 Decompose

Write the tensor as a chain of $n$ four-index cores contracted along virtual ("bond") indices:

$$W_{j_1 \ldots j_n,\, i_1 \ldots i_n} \;=\; \sum_{\alpha_1 \ldots \alpha_{n-1}} w^{(1)}_{j_1 i_1,\, \alpha_1}\, w^{(2)}_{\alpha_1,\, j_2 i_2,\, \alpha_2} \cdots w^{(n)}_{\alpha_{n-1},\, j_n i_n}$$

with $\alpha_k \in \{1,\ldots,\chi_k\}$. This is exactly Oseledets' Tensor-Train format ([SIAM J. Sci. Comput. 33(5):2295–2317, 2011](https://epubs.siam.org/doi/10.1137/090752286)) with a second physical index per core; in the physics literature the same object is an MPO and $\chi_k$ is the bond dimension.

### 1.3 The sequential SVDs (TT-SVD)

This is the constructive part a reader would implement. Let $T = W$ reshaped to $(J_1 I_1) \times (\prod_{k>1} J_k I_k)$ after permuting indices into site-major order $(j_1 i_1)(j_2 i_2)\ldots(j_n i_n)$. Then, for $k = 1 \ldots n-1$:

1. Reshape the current residual $T$ into a matrix $T_k$ of shape $(\chi_{k-1} J_k I_k) \times (\text{rest})$, with $\chi_0 = 1$.
2. Compute the SVD $T_k = U \Sigma V^\top$.
3. Truncate to the largest $\chi_k$ singular values: $U \to U_{:,1:\chi_k}$, etc.
4. Set core $w^{(k)} \leftarrow$ reshape$(U_{:,1:\chi_k})$ to $\chi_{k-1} \times J_k I_k \times \chi_k$.
5. Set $T \leftarrow \Sigma_{1:\chi_k} V^\top_{1:\chi_k,:}$ and continue.

The final residual becomes $w^{(n)}$. Oseledets proves this greedy sweep is quasi-optimal: the resulting error is at most $\sqrt{n-1}$ times the best achievable at those ranks. Cost is $O(MN \cdot \max_k \chi_k)$ — one pass, no gradients, no data.

Two implementation details CompactifAI specifies and most summaries omit:

- **Embedding and output-head matrices are left dense.** Only attention and MLP weights are tensorized.
- **The last MLP layer in each transformer block is excluded** as too sensitive (stated in [2401.14109v2](https://arxiv.org/html/2401.14109v2)).

### 1.4 Truncate non-uniformly by depth

The paper reports a layer-sensitivity profile and states it qualitatively: early layers "cannot be compressed below 50%," while "as we move towards the end of the network... we can compress the layers down to 10% of the original size without significant loss of accuracy." Widely-circulated block-index ranges (blocks 0–5 at 50%, blocks 15–31 at 10%) are **not** stated as numbers in the paper — cite the percentages, not the ranges.

**The paper never gives a numeric value for χ anywhere.** Verified across both v2 and the ESANN camera-ready. Any χ ≈ 100 figure in circulation is untraceable to a primary source.

### 1.5 Heal

A retrain of **less than a single epoch** on Ultrachat, Alpaca and OpenHermes, on **8× NVIDIA A10g** on one AWS EC2 instance. No token count, no dataset fraction, no wall-clock figure is disclosed. This matters: it makes healing-cost comparisons against pruning methods that *do* report token budgets (Sheared LLaMA, Minitron, LLM-Pruner) approximate at best.

### 1.6 Parameter-count arithmetic

**General MPO count.** With uniform bond χ and $\chi_0 = \chi_n = 1$:

$$P_{\text{MPO}} = I_1 J_1 \chi \;+\; \chi^2 \sum_{k=2}^{n-1} I_k J_k \;+\; I_n J_n \chi$$

Interior cores cost $\chi^2$; boundary cores cost $\chi$.

**Worked example — 4096×4096 attention projection, $n=4$, dims $(8,8,8,8)^2$, so $I_kJ_k = 64$:**

$$P_{\text{MPO}} = 64\chi + 64\chi^2 + 64\chi^2 + 64\chi = 128\chi + 128\chi^2$$

| χ | MPO params | Fraction of 16,777,216 |
|---|---|---|
| 16 | 34,816 | 0.21% |
| 32 | 135,168 | 0.81% |
| 64 | 532,480 | 3.2% |
| 100 | 1,292,800 | 7.7% |
| 128 | 2,113,536 | 12.6% |
| 200 | 5,145,600 | 30.7% |

*(My arithmetic from the formula above, not a CompactifAI-reported table.)*

**System-level arithmetic for the published result.** LlaMA-2 7B has ≈6.74B parameters, of which the tied embedding and head are ≈2 × 32000 × 4096 ≈ 262M, leaving ≈6.48B in transformer blocks. CompactifAI reports **7B → 2.1B**, i.e. ≈70% parameter reduction, which requires removing ≈4.64B from the ≈6.48B of block weights — about 72% of the tensorizable surface, consistent with a depth-graded χ schedule averaging roughly 28% retention.

### 1.7 The number everyone quotes is not a parameter count

This is the single most important correction in this brief, and it is confirmed from the paper body, not inferred.

| Model | Params | Precision | Size | Label |
|---|---|---|---|---|
| LlaMA-2 7B dense | 7B | float32 | 27.1 GB | — |
| CompactifAI | 2.1B | fp16 tensorized / **int4 non-tensorized** | 4.1 GB | "88% compressed" |
| CompactifAI | 2.1B | fp16 tensorized / **int4 non-tensorized, further 4-bit quantized** | 2.1 GB | "93% compressed" |

Both memory columns sit at **the same 2.1B parameters**. The 88% → 93% step removes zero parameters; it is purely additional 4-bit quantization of the not-tensorized layers. So:

- **The factorization-attributable number is 70% parameter reduction.** Not 88%. Not 93%.
- Both memory percentages are measured against an **fp32** baseline. Against fp16 (13.55 GB) the 2.1 GB model is ≈84.5% smaller. Against a plain int4 model of the same architecture (≈3.4 GB) it is roughly 38% smaller — that last figure is the honest answer to "what does the tensor network buy on top of quantization I would deploy anyway." Still real; roughly 1.6×; an order of magnitude away from the impression "93%" creates.
- Minor arithmetic gap worth knowing: 4.1 GB / 27.1 GB = 84.9%, not 88%. Do not treat the labels as exact byte ratios.

### 1.8 The accuracy claim, disaggregated

Headline: "small accuracy drop of 2%–3%." The per-benchmark table (LlaMA-2 7B, dense → 88% → 93%):

| Benchmark | Dense | 88% | 93% | Δ abs (93%) | Δ rel (93%) |
|---|---|---|---|---|---|
| MMLU | 46.41 | 45.32 | 44.16 | −2.25 | −4.8% |
| HellaSwag | 80.55 | 77.87 | 76.54 | −4.01 | −5.0% |
| BoolQ | 79.76 | 77.90 | 76.77 | −2.99 | −3.7% |
| TriviaQA | 19.03 | 18.33 | 18.10 | −0.93 | −4.9% |
| **GSM8K** | **23.05** | **22.58** | **17.74** | **−5.31** | **−23.0%** |

The "2–3%" is an average of **absolute points**, and it conceals a 23% relative collapse on GSM8K. Two things follow, and the second is more interesting than the first:

1. Hard multi-step reasoning degrades roughly 5× more, in relative terms, than the headline implies.
2. **GSM8K is nearly flat at 88% (−0.47) and collapses only when the extra 4-bit quantization is applied.** The reasoning damage tracks the quantization step, not the MPO truncation. This is a point *in favor* of the factorization mechanism and *against* the way the result is packaged.

---

## 2. The Physics Framing, Honestly

### 2.1 What the lens actually is

Reinterpret the reshaped weight tensor as an unnormalized bipartite quantum state. Cut the index chain between sites $k$ and $k+1$. The singular values of the corresponding unfolding are the Schmidt coefficients across that cut; the bond dimension $\chi_k$ is the Schmidt rank; and the entanglement entropy $S = -\sum_i \sigma_i^2 \log \sigma_i^2$ (with normalized $\sigma$) is a functional of the same singular spectrum the SVD already produced. Truncating at $\chi$ is truncating correlation across that cut.

This is a genuine and internally consistent lens. It also buys exactly two real things and one thing it does not buy.

**What it buys:**
1. **A vocabulary for where to cut and how to allocate bond dimension along the chain**, imported from DMRG and area-law intuition in condensed-matter physics. The reshaping choice — how you factor $M$ and $N$, and in what order you interleave indices — is a free hyperparameter with no guidance from linear algebra, and the physics literature has decades of heuristics for it.
2. **A mature algorithmic toolbox** — sweeping, variational core optimization, canonical forms, DMRG-style two-site updates — that the numerical-linear-algebra framing does not hand you as readily.

**What it does not buy:** anything computational. No quantum hardware. No superposition. No complex amplitudes. No speedup unavailable to classical numerical linear algebra. "Quantum-inspired" here describes *provenance* (White's DMRG, 1992; the MPS/MPO formalism) and not *mechanism*. The identical object was independently derived in applied mathematics as the Tensor Train ([Oseledets 2011](https://epubs.siam.org/doi/10.1137/090752286)) with no quantum content whatsoever.

Nor is the framing Multiverse's. [Gao, Cheng, He, Xie, Zhao, Lu and Xiang, *Phys. Rev. Research* 2, 023300 (2019)](https://arxiv.org/abs/1904.06194) used precisely this vocabulary — "matrix product operator," "bond dimension," area-law motivation — five years earlier, as condensed-matter physicists writing for a physics journal, not as a positioning device.

### 2.2 Does a 2-site MPO differ from a rank-χ SVD? Head-on.

**Short answer: yes as a hypothesis class, no as a mechanism, and the mechanism is from 1993.**

Take $W \in \mathbb{R}^{M \times N}$, $M = J_1 J_2$, $N = I_1 I_2$. A two-site MPO with bond χ is

$$W_{(j_1 j_2),(i_1 i_2)} = \sum_{\alpha=1}^{\chi} A_\alpha[j_1, i_1]\, B_\alpha[j_2, i_2]$$

which is, index-for-index, a sum of χ Kronecker products:

$$W \;\approx\; \sum_{\alpha=1}^{\chi} A_\alpha \otimes B_\alpha$$

Define the **rearrangement** $\mathcal{R}(W) \in \mathbb{R}^{(J_1 I_1) \times (J_2 I_2)}$ by $\mathcal{R}(W)[(j_1 i_1),(j_2 i_2)] = W[(j_1 j_2),(i_1 i_2)]$. Then the two-site MPO constraint is *literally* $\operatorname{rank}(\mathcal{R}(W)) \le \chi$. Because $\mathcal{R}$ is a permutation of entries it is Frobenius-isometric, so $\|W - \hat W\|_F = \|\mathcal{R}(W) - \mathcal{R}(\hat W)\|_F$, and **the optimal bond-χ truncation is computed by a single truncated SVD of $\mathcal{R}(W)$, with Eckart–Young applying verbatim.**

This is the Van Loan–Pitsianis nearest-Kronecker-product problem (1993). It was solved then, by an SVD of a rearranged matrix, and it is solved the same way now.

**Parameter counts.** Rank-$r$ SVD of $W$ costs $r(M+N)$. Two-site MPO at bond χ costs $\chi(J_1 I_1 + J_2 I_2)$. For the balanced square case $M = N = d$ with $J_1 I_1 = J_2 I_2 = d$, the MPO costs $2\chi d$ — **identical** to a rank-χ SVD at $r = \chi$. A concrete instance: $d = 4096$, $J_1=J_2=I_1=I_2=64$; SVD rank $r$ costs $8192r$, MPO bond χ costs $8192\chi$. Exactly equal.

**So what is different?** Only *which matricization the rank constraint is imposed on*. Two constraint sets of equal parameter budget, neither containing the other, both non-convex manifolds. Which fits a given weight better is a purely empirical question about that weight's index structure — there is no a priori argument either way.

**Is "beyond low-rank" defensible?** Yes, narrowly and literally. A χ=1 two-site MPO is a single Kronecker product $A \otimes B$, which is generically of rank $\min(J_1,I_1)\cdot\min(J_2,I_2)$ in the original matrix — full rank. So the MPO class genuinely contains operators no rank-χ approximation can reach. This is a real set-theoretic distinction and it should be granted. But it is the 1993 rearrangement, not a new approximation mechanism, and it is low-rank *in a permuted basis*.

**For $n \ge 3$ sites** the format is genuinely richer than either: a TT-rank *tuple* $(\chi_1,\ldots,\chi_{n-1})$ imposing rank constraints on a nested hierarchy of unfoldings simultaneously, with the interior $\chi^2$ scaling. That is a real generalization. It is also still strictly multilinear, still constructed by a sequence of SVDs of unfoldings, and still introduces no new function class, no new optimization landscape, and no non-classical operation at any point.

### 2.3 Does the extra expressivity pay off?

The best available independent evidence says no, at least in the regime it was tested. [Zagitov et al., "Rethinking the Role of Tensor Decompositions in Post-Training LLM Compression" (arXiv:2606.03465, June 2026)](https://arxiv.org/abs/2606.03465) evaluates Tucker, TT and CP against matrix-based methods on GPT-J 6B and LLaMA-2 7B (dense) plus Qwen3-30B-A3B and GPT-OSS-20B (MoE). Their finding: "Tensor formats fail to outperform their matrix counterparts at matched compression ratios," and "attention-only compression preserves perplexity but yields negligible size reduction; compressing FFN layers achieves higher compression at the cost of sharp quality degradation." Their theoretical diagnosis is a "fundamental mismatch between the shared subspaces assumed by tensor decompositions and the heterogeneous representations learned by modern LLMs," including that tensor methods discard "super-weight" coordinates carrying negligible Frobenius mass but dominating function.

Quantitatively, on GPT-J 6B at ≈17% bits saved: Tucker(MHA)+LASER(FFN) at rank 1024 gives ≈19 perplexity; TT-based combinations ≈28; and **plain round-to-nearest 4-bit quantization gives ≈12, essentially at baseline.**

**Two caveats that must travel with this, and they are load-bearing:**

1. **Zagitov et al. do not name, cite, or benchmark CompactifAI or MPO.** Their baselines are LASER, SVD-LLM, TensorLLM, LeSTD, TD-MoE (one verification pass also recorded HASSLE-free, SoLA, FLAT-LLM, Dobi-SVD, MoBE and RTN). It is evidence about the *method class*, not about the product, and citing it as a refutation of CompactifAI specifically would be wrong.
2. **Their operating points are 8–24% bits saved** — an order of magnitude below CompactifAI's claimed regime. It is not a matched-regime comparison. What it does establish, robustly, is that tensor decomposition loses to naive quantization at *low* compression on dense models.

---

## 3. Taxonomy of the Field

### 3.1 The identity

$$\text{Memory} \;=\; \underbrace{P}_{\text{parameter count}} \times \underbrace{b}_{\text{bits per parameter}}$$

$$\text{Latency} \;\approx\; f\big(\text{FLOPs},\; \text{memory bandwidth},\; \text{kernel support}\big)$$

Three families move different terms, and only the first two are post-hoc operations on an existing checkpoint.

### 3.2 Family A — reduce $P$ (parameter count)

The output is a smaller set of real numbers. Sub-families:

| Sub-family | Operation | Shape after | Speedup on stock GPU? |
|---|---|---|---|
| **Tensor networks (MPO/TT)** | $W \to$ chain of cores | small dense cores | Yes, but needs a contraction schedule |
| **Low-rank / SVD** | $W \to LR$ | two thin dense matrices | Yes, plain GEMMs |
| **Structured factorization** (Monarch, Kronecker, butterfly) | $W \to$ product of structured factors | block-diagonal + permutation | Only with custom kernels |
| **Structured pruning** | delete heads/channels/experts | smaller dense matrices | Yes, plain GEMMs |
| **Depth pruning** | delete or merge whole blocks | fewer layers | Yes, trivially |
| **Dimension slicing** (SliceGPT) | delete residual-stream dims | smaller $d_{\text{model}}$, plain transformer | Yes, best-in-class |
| **Unstructured / N:M sparsity** | zero entries | **unchanged** | No (unstructured); ~1.24–1.6× (2:4) |

Note the last row is the odd one out: SparseGPT and Wanda's headline 50–60% removal is *nominal*. Without a sparse storage format the memory is unchanged, and without hardware sparsity support the latency is unchanged. This is the central practical asymmetry favoring shape-changing methods.

### 3.3 Family B — reduce $b$ (bits per parameter)

Post-training quantization: [GPTQ](https://arxiv.org/abs/2210.17323) (Hessian-guided sequential rounding with error compensation), [AWQ](https://arxiv.org/abs/2306.00978) (per-input-channel scaling to protect ~1% salient channels), [SmoothQuant](https://arxiv.org/abs/2211.10438) (migrate activation outlier difficulty into weights, W8A8), [AQLM](https://arxiv.org/abs/2401.06118) (additive multi-codebook, Pareto-optimal below 3 bits), [QuIP#](https://arxiv.org/abs/2402.04396) and [QTIP](https://arxiv.org/abs/2406.11235) (randomized Hadamard incoherence processing + E8 lattice VQ / trellis coded quantization). Quantization-aware pretraining: [BitNet b1.58](https://arxiv.org/abs/2402.17764) at ternary weights, ≈1.58 bits. Hardware formats: FP8 (E4M3/E5M2), and the microscaling 4-bit floats NVFP4 and MXFP4.

**The scale of what this family already delivers is the calibration everyone skips.** AQLM reaches ≈2 bits/parameter with *zero* parameter reduction — 16× against fp32, 8× against fp16. BitNet reaches ≈1.58 bits, ≈20× against fp32, but requires pretraining from scratch. Against those, "93%" (≈12.9×) is not beyond what pure quantization reaches.

### 3.4 Family C — never build the big model (train-smaller)

Distillation ([Seq-KD, Kim & Rush 2016](https://arxiv.org/abs/1606.07947); [MiniLLM reverse-KL](https://arxiv.org/abs/2306.08543)), prune-then-distill ([Minitron](https://arxiv.org/abs/2407.14679)), curated-data small dense models ([Phi-3](https://arxiv.org/abs/2404.14219), [Gemma 2](https://arxiv.org/abs/2408.00118)), and architecture ([DeepSeekMoE](https://arxiv.org/abs/2401.06066), [GQA](https://arxiv.org/abs/2305.13245), Gated DeltaNet, [MatFormer](https://arxiv.org/abs/2310.07707)). Cost is orders of magnitude higher; quality per parameter is the best available. This family does not compete for the same buyer — it competes for the same *deployment slot*.

### 3.5 What multiplies

**A × B multiplies arithmetically.** 70% parameter reduction (3.3×) × fp32→int4 (4×) ≈ 13× ≈ 93%. That is exactly the CompactifAI headline decomposed, and it is why the headline is not a factorization result.

**Within A, methods sub-add rather than multiply where they mine the same redundancy.** Four independent measurement procedures converge on the same substrate fact — that middle-to-deep transformer blocks are the redundant ones:

| Method | Measurement | Conclusion |
|---|---|---|
| [ShortGPT](https://arxiv.org/abs/2403.03853) | Block Influence: cosine change of hidden state across a block | low-influence blocks are mid-to-deep |
| [Gromov et al.](https://arxiv.org/abs/2403.17887) | angular distance between block input and output | contiguous deep spans act near-identity |
| [LaCo](https://arxiv.org/abs/2402.11187) | mergeability of consecutive rear layers | rear layers compose |
| CompactifAI | accuracy loss under bond-dimension truncation, per layer | early layers ≥50%; late layers to 10% |

That convergence is strong corroboration of the underlying fact and a warning that depth pruning and depth-graded MPO truncation are mining the same seam. Similarly, Minitron's width pruning shrinks the MLP intermediate dimension (14336 → 9216 for Llama-3.1-8B → 4B), which is dimension reduction of *exactly the matrices* MPO factorizes — directly competing, not complementary.

### 3.6 Where A and B interfere — three mechanisms

This is the part the "orthogonal axes, just multiply them" framing hides.

**(1) Error amplification in the factor product.** If $W \approx LR$ and you quantize both factors, $(L+\delta L)(R+\delta R) = LR + \delta L\,R + L\,\delta R + \delta L\,\delta R$. Cross terms scale with the *other* factor's norm, so error is multiplicative rather than additive. Worse, SVD factors carry the singular-value spectrum, which for LLM weights is heavy-tailed, so $L$ and $R$ have far larger dynamic range than $W$ — and uniform scalar quantization error scales as range$/2^b$. Naive factorize-then-quantize is doubly penalized. This is the premise of [LPLR](https://arxiv.org/abs/2310.11028) and [CALDERA](https://arxiv.org/abs/2405.18886).

**(2) No error headroom left.** Rank truncation has already spent the layer's error budget against the calibration objective. Quantization error then lands on top with nothing to absorb it.

**(3) Loss of the structure the good quantizers require.** *(This third mechanism is structural analysis; I found no paper that measures it directly, and it should be labeled as a hypothesis.)* AWQ's salience and SmoothQuant's smoothing scale are both defined **per input channel of the original weight matrix**, and must fold into a preceding LayerNorm or linear to be free at runtime. A tensorized layer's interior bond indices are a latent basis with no activation-channel correspondence and no LayerNorm to absorb a scale. Meanwhile QuIP#/QTIP's incoherence processing wants weight energy *spread uniformly* across directions — the exact opposite of what a truncated factorization produces, since concentration of energy is what made truncation possible in the first place. A truncated factor is maximally coherent by construction.

**The right way to compose.** CALDERA (NeurIPS 2024) writes $W \approx Q + LR$ with $Q$, $L$ and $R$ *all* quantized, and refuses to do it sequentially: it solves one constrained problem by alternating minimization against calibration-weighted layer output error, fitting the low-rank term to the *quantization residual* rather than to $W$. It reports outperforming existing post-training compression below 2.5 bits per parameter on LLaMA-2 7B/13B/70B and LLaMA-3 8B, and reports a single honest **effective bits per original parameter**, which is the only unambiguous combined metric.

**The tell.** CompactifAI leaves its own tensorized layers at **float16** and applies int4 only to the *not*-tensorized layers. That design choice is the interference thesis appearing inside the method itself — the factors are the part they did not quantize aggressively.

---

## 4. Comparison Table

Every row is attributed to a specific model and benchmark. Where verification found a claim overstated or in need of a caveat, the caveat replaces the bare headline.

### 4.1 Parameter-count reduction (Family A)

| Method | Model | Operating point | Result (benchmark named) | Retrain | Caveat |
|---|---|---|---|---|---|
| **CompactifAI** ([2401.14109](https://arxiv.org/abs/2401.14109)) | LlaMA-2 7B | **70% params** (7B→2.1B) | MMLU 46.41→44.16; HellaSwag 80.55→76.54; BoolQ 79.76→76.77; GSM8K 23.05→**17.74** | <1 epoch, 8× A10g | The "93%" is memory vs **fp32** with int4 folded in; parameter reduction is 70% at both the 88% and 93% labels. GSM8K's −23% relative drop occurs only at the extra-quantization column. No numeric χ published. No comparison against any PTQ or pruning baseline at matched compression. |
| **SVD-LLM** ([2403.07378](https://arxiv.org/abs/2403.07378), ICLR 2025) | LLaMA-7B | 20 / 40 / 60 / 80% | WikiText-2 PPL 7.73 / 9.27 / 15.00 / 31.79 (plain SVD at 20%: **20061**) | Calibration (256 WikiText-2) + LoRA on 50K Alpaca | Headline config is not training-free. Reports no MMLU, no BoolQ; HellaSwag is on LLaMA-1. |
| **SVD-LLM V2** ([2503.12340](https://arxiv.org/abs/2503.12340), NAACL 2025) | LLaMA-3 8B / LLaMA-7B | 20% / 80% | — | Calibration | **The circulating "42% perplexity reduction and 2.71× speedup" merges two operating points.** The 42% is C4 perplexity 20.05→11.72 *relative to SVD-LLM*, on LLaMA-3 8B at **20%** compression. The 2.71× is throughput vs the **dense** model, on LLaMA-7B at **80%** compression. Never state as one configuration. |
| **ASVD** ([2312.05821](https://arxiv.org/abs/2312.05821)) | LLaMA-7B | self-declared 10–30% | WikiText-2 PPL 11.14 @20%; **1407 @40%** | Training-free | Collapses past ~30%. Own claimed range is 10–30%; the 40%+ figures are from SVD-LLM's reproduction. |
| **FWSVD** ([2207.00112](https://arxiv.org/abs/2207.00112), ICLR 2022) | LLaMA-7B (third-party) | 20% | WikiText-2 PPL **1727** | Fisher gradients | Works at BERT scale; **does not transfer** to decoder-only LLMs. Historically important for proving weight-space reconstruction error is the wrong objective. |
| **SliceGPT** ([2401.15024](https://arxiv.org/abs/2401.15024), ICLR 2024) | Llama-2 7B / 70B | 25% / 30% slicing | 7B avg (PIQA/WinoGrande/HellaSwag/ARC-e/ARC-c) 69.00 → 55.48 @25% → 51.50 @30%; 70B 76.57 → 69.75 @25% → 66.11 @30% | Calibration only; optional LoRA RFT | Retention is strongly scale-dependent. RFT is worth ≈8 avg points at 70B/30% (66.11→74.30), so unhealed SliceGPT numbers should never be compared against healed CompactifAI numbers. Compute drops to **64–66% of dense** at 25% — the best realized wall-clock gain in this table. |
| **LoRD** ([2309.14021](https://arxiv.org/abs/2309.14021)) | StarCoder 16B | 16B → 13.2B (17.5%) | **No drop** in HumanEval Pass@1; up to 22.35% inference speedup | **None** — one-shot, data-free | The honest floor of the family: what pure truncated SVD buys at zero risk and zero data. Code models only; no general-knowledge evidence. |
| **TT on linear layers** ([2501.19135](https://arxiv.org/abs/2501.19135)) | LLaMA2-7B / ChatGLM3-6B | whole-network | **1.60× / 1.94×** compression; FPGA first-token delay −1.57× / −1.45× | Not stated | **The most important discrepancy in this brief.** Independent, contemporaneous, Llama-scale TT compression reaches 1.6× where CompactifAI claims 3.3× on parameters. Either healing does nearly all the work, or the two measure different quantities. Perplexity numbers not retrieved. |
| **SparseGPT** ([2301.00774](https://arxiv.org/abs/2301.00774), ICML 2023) | LLaMA-7B / 65B / Llama-2-70B | 50% unstructured | WikiText PPL 5.68→7.22 / 3.56→4.57 / 3.12→3.98 | **None** | 50–60% unstructured gives **no memory or latency win** on commodity GPUs without a sparse format. 2:4 costs far more: LLaMA-7B 11.00. Accuracy holds mainly at the largest scales. |
| **Wanda** ([2306.11695](https://arxiv.org/abs/2306.11695), ICLR 2024) | LLaMA-7B / Llama-2-70B | 50% unstructured / 2:4 | 7.26 / 11.53; 3.98 / 5.16 | **None**, no weight update | Cheapest method here: 5.6s on LLaMA-65B vs SparseGPT's 1353.4s. Measured 2:4 speedup **1.24× end-to-end** on LLaMA-7B (251 vs 312 ms) — well short of the nominal 2×. Authors concede that at 80% sparsity a small dense model may simply be better. |
| **LLM-Pruner** ([2305.11627](https://arxiv.org/abs/2305.11627), NeurIPS 2023) | LLaMA / Vicuna / ChatGLM | ~20% (commonly cited; **not confirmed in the abstract**) | qualitative only in the abstract | 50K samples, LoRA, ~3h | Shallow operating point relative to everything else here. Real memory/latency win since shapes change. |
| **ShortGPT** ([2403.03853](https://arxiv.org/abs/2403.03853)) | Llama-2 13B | 10 of 40 layers (25%) | MMLU 55.0 → 52.2 (≈94.9% retention) | One-shot | Number sourced from paper HTML, flagged for direct table verification. MMLU is forgiving for depth removal; generation and reasoning degrade faster. |
| **Gromov et al.** ([2403.17887](https://arxiv.org/abs/2403.17887)) | Llama-2 7B/13B/70B, Qwen, Mistral, Phi-2 | up to ~half of layers | minimal QA degradation until an abrupt collapse (13B transitions to chance at 45–55%) | **164–328M tokens QLoRA, one 40GB A100** | The cleanest published answer to "how much healing does pruning need." Authors are careful that QA robustness ≠ preserved generation quality. |
| **Sheared LLaMA** ([2310.06694](https://arxiv.org/abs/2310.06694), ICLR 2024) | LLaMA2-7B → 1.3B / 2.7B | targeted architecture | beats Pythia, INCITE, OpenLLaMA, TinyLlama at matched size | **3% of from-scratch compute** (pretraining-scale) | Token budgets (≈0.4B prune + ≈50B continued) are widely cited but not confirmed from the abstract. Needs a pretraining corpus and cluster. |
| **Minitron** ([2407.14679](https://arxiv.org/abs/2407.14679)) | Nemotron-4 15B → 8B/4B; Llama-3.1-8B → 4B | 2–4× | **up to +16% MMLU vs training the same size from scratch**; <3% of original data; up to 40× fewer tokens | **≈94B distillation tokens** (Llama-3.1-Minitron-4B) | The strongest absolute quality-per-parameter result on this list, and the most expensive. NVIDIA finds **width pruning beats depth pruning** at this scale — a counterweight to the depth-redundancy consensus. |
| **MoBE** ([2508.05257](https://arxiv.org/abs/2508.05257), ICLR 2026) | Qwen3-235B-A22B-2507 / DeepSeek-V3-0324 / Kimi-K2-Instruct | 24% / **30%** / 24% | 15-benchmark avg 81.5→80.9 / 79.3→78.0 / 82.4→81.1 | **Data-free** | **Only DeepSeek-V3 reaches 30%**; the other two are 24%. Compresses *total* (VRAM) not *active* params. MoBE itself notes the form "requires multiple times calling of current optimized kernel fused-MoE... relatively inefficient" — no speedup today. Down-projections excluded as "less amenable." Its D2-MoE (−7 to −14%) and MoLAE (−4 to −8%) figures are **MoBE's own measurements of competitors**, not their self-reported results. |
| **RFID-MoE** ([2602.09316](https://arxiv.org/abs/2602.09316)) | Qwen3-30B | 60% | PTB PPL **16.92** (≈8.0 better than baselines); ≈+8% HellaSwag over baselines | Post-hoc SVD | 16.92 PTB is heavy degradation in absolute terms. Beating weak baselines at an aggressive ratio is not the same as a usable model. |

### 4.2 Bits-per-parameter (Family B)

| Method | Model | Operating point | Result | Caveat |
|---|---|---|---|---|
| **GPTQ** | OPT/BLOOM-175B class | 3–4 bit weight-only | "negligible degradation"; 175B quantized in ≈4 GPU-hours; 3.25× on A100, 4.5× on A6000 | Per-benchmark perplexity tables not retrieved from the abstract page. Layer-local objective, so error accumulates with depth. |
| **AWQ** | Llama-2 70B, instruction-tuned and multimodal LMs | W4A16 | >3× over HF fp16 via TinyChat; protecting 1% salient channels sharply cuts error | No backprop, no reconstruction — its claimed edge over GPTQ is calibration generalization. Head-to-head perplexity vs GPTQ not retrieved. |
| **SmoothQuant** | OPT, BLOOM, GLM, Llama-1/2, Falcon, Mistral, Mixtral | W8A8 | up to 1.56× speedup, 2× memory; enables 530B on one node | Only 8 bits. Value is INT8 tensor-core throughput, not footprint. |
| **AQLM** | LLaMA-2, Mixtral | **2–3 bits** | claimed **Pareto optimal below 3 bits/param** | Per-model perplexity tables not retrieved; only the qualitative claim is verified. Expensive to produce (block-wise codebook optimization). |
| **QuIP# / QTIP** | LLaMA-2 family | ≤4 bits, incl. 2-bit | claimed SOTA quality and speed | **Per-model perplexity and accuracy tables were not retrievable.** Widely repeated "2-bit Llama-2-70B is usable" claims should be checked against the papers' own tables before being repeated. QuIP# uses fine-tuning, so it is not one-shot. |
| **BitNet b1.58** | trained from scratch, 2B/4T variant | **1.58 bits** (ternary), W1.58A8 | claims parity with fp16 at equal size and tokens; 2B4T reports up to 6.17× on x86 CPU, 5.07× on ARM, −82.2% / −70.0% energy | **Requires pretraining from scratch**, so it cannot be applied to a released checkpoint and cannot be composed post-hoc with factorization. 2B4T figures are from the technical report / model card, flagged for verification. |
| **NVFP4 / MXFP4** | hardware formats | ≈4.5 / ≈4.25 effective bits | reported: NVFP4 more accurate than MXFP4 (16-element blocks, FP8 E4M3 scale vs 32-element, E8M0); Blackwell NVFP4 ≈2× FP8 throughput | **All of this rests on vendor and third-party blogs, not a hardware spec — high risk, verify before citing.** The point that survives regardless: a 2026 serving stack baselines at FP8 or NVFP4, not fp32, which is what makes fp32-referenced compression percentages generous. |
| **CALDERA** ([2405.18886](https://arxiv.org/abs/2405.18886), NeurIPS 2024) | LLaMA-2 7B/13B/70B, LLaMA-3 8B | **<2.5 effective bits/original param** | claims to outperform existing PTQ in that regime | The rigorous version of factorize-plus-quantize: joint, not sequential; quantizes the factors too; reports one unambiguous effective-bits number. Per-config perplexity tables not retrieved. |

### 4.3 Multiverse's own product and post-2025 claims

| Claim | Verdict | What to say instead |
|---|---|---|
| Llama 3.1 8B Slim / **ChickBrain is 3.2B parameters** | **needs caveat** | No primary source gives an absolute count. The product page states "60% reduction in parameter number"; 8.03B × 0.40 ≈ 3.2B is *derived*. Multiverse's own newsroom headline says "**compresses... by 80%**" — a memory figure. The vendor publishes a parameter number and a memory number 20 points apart, in the same campaign. |
| **ChickBrain exceeds Llama 3.1 8B on MMLU-Pro, MATH500, GSM8K, GPQA Diamond** | **unsupported** | Press-release text only ("exceeded the performance... across benchmarks"), **zero published scores on any of the four**, no model card, no eval log, no public weights. A ~60%-compressed model beating its own parent on GPQA Diamond and MATH500 would be extraordinary and requires an eval log to accept. Business fact only; do not enter in any quantitative table. |
| **SuperFly, 94M from SmolLM2-135M** | supported as a number | ≈30% parameter reduction — despite the "15,000× smaller than ChickBrain" framing, which refers to notional hardware, not parameters. Zero benchmarks published. |
| **Llama 3.3 70B Slim on Intel Xeon 6737P: 3.86 output tok/s, 7.81 total tok/s, +94.1% throughput** | **supported, vendor-reported** | Now traced to Multiverse's own newsroom page (July 2026), which also gives latency 5056→2598 ms, TTFT 6939→3706 ms, and at 256 concurrent users +107.0% throughput / −51.7% latency. Three cautions: (a) vendor-reported, no Intel-published benchmark document reached; (b) the baseline is uncompressed 70B on **CPU**, a memory-bandwidth-bound worst case where nearly any size reduction roughly doubles throughput — this does not transfer to GPU serving; (c) **3.86 output tok/s is below interactive reading speed**, so "nearly doubles performance" describes doubling a very low number. |
| Llama 3.3 70B Slim accuracy deltas: BoolQ −0.95%, GSM8K −1.14%, HellaSwag −1.94%, MMLU −2.48%, **WinoGrande +6.86%** | **needs caveat** | Absolute values on the vendor page: HellaSwag baseline **0.5905**. Published Llama 3.3 70B HellaSwag acc_norm is ≈0.85, so this is either raw `acc` or a nonstandard harness config — **not comparable to any other paper's HellaSwag column.** And WinoGrande *improving* 6.86% under compression is not a compression effect; it is a harness, prompt-format or noise artifact. Flag it; do not repeat it as a gain. No compression ratio is stated alongside these deltas. |
| Slim marketing: **"40% faster inference"** vs product page **"1.85× inference speed-up"** | **mutually inconsistent** | Two published speedup claims for the same product family that cannot both be right. Neither is reproduced by the one instrumented study (see §6). |
| **"300× fewer training tokens than Meta's Llama 3, 3× fewer than Llama-3.1-Minitron-4B"** | **needs caveat** | Compares a compression-plus-healing budget against a from-scratch pretraining budget. Not like-for-like. |
| **HyperNova 60B 2605** vs gpt-oss-120b | self-reported | MMLU-Pro 76.8 vs 79.6; AIME25 90.0 vs 93.7; IFBench 66.6 vs 67.0; **LiveCodeBench 68.7 vs 62.8 (compressed higher)**. 60B total / 4.8B active MoE; 65GB→32GB; Apache 2.0. LiveCodeBench rising 5.9 points after 50% compression suggests distillation-data skew toward code or a baseline evaluated under different settings; no harness version, sampling params or *n* is given. And **this model is a knowledge distillation, not an MPO compression** (see §5.4). |
| **Pulsar 16B** | **needs caveat** | Base: NVIDIA-Nemotron-3-Nano-30B-A3B-BF16, 31.6B → 16.15B ("50% compression"), **3.1B active — it is an MoE**, so this ratio is not comparable to dense-model compression. BF16 base→Pulsar: AIME 87.66→87.22; GPQA 74.04→71.41; MMLU-Pro 78.90→74.78; LiveCodeBench 71.11→68.04. NVFP4 costs *further* accuracy: AIME 82.00 (−5.2pp vs BF16 Pulsar), LiveCodeBench 65.60. **Quoting BF16 accuracy alongside NVFP4 memory would be a cross-configuration merge.** The compression workflow is credited to NVIDIA Model Optimizer and Megatron Bridge — there is no public statement that MPO is the mechanism, so **Pulsar is not evidence that MPO works at scale.** The "9B params" on the NVFP4 card contradicts its own prose ("16.15B"); my inference is a packed-4-bit storage artifact (16.15B/2 ≈ 8.1B plus FP8 block scales), *not verified*. |
| **Quasar 438B** | **unsupported** | Press only (GlobeNewswire, 2 Sept 2026): 438B params, 1M context, English/Spanish, Artificial Analysis Intelligence Index v4.1.1 score 43, 500 output tokens in 15.3s. **No source states whether it is a compression, a distillation, or from-scratch.** The "first large model" framing is in tension with calling it a CompactifAI output — a compression would normally name its base, and none is named. Do not assert provenance either way. |

**A pattern worth naming.** Of the vendor-originated claims checked across two verification passes, every single one exhibited at least one of: memory reported where parameters were implied (ChickBrain 80% vs 60%), quantization folded into a factorization figure (CompactifAI 93%), fp32-baseline framing in a 2026 fp8 world, single-configuration results stated generally (Pulsar NVFP4), or entirely unmeasured marketing (ChickBrain superiority, Quasar provenance). This is not unique to Multiverse — the academic SVD-LLM V2 case shows the same "up to X and up to Y" merge across two operating points — but the density is higher.

---

## 5. What Is Actually Novel

Being fair requires separating the mathematics, the workflow, the scale and the packaging. Only two of the four hold up as contributions, and neither is the one the marketing emphasizes.

### 5.1 The mathematics is not new, and the specific priors are identifiable

| Component | Prior art | Year |
|---|---|---|
| TT/MPO format, TT-SVD construction, √(d−1) quasi-optimality | [Oseledets, SIAM J. Sci. Comput.](https://epubs.siam.org/doi/10.1137/090752286) | 2011 |
| Two-site MPO = nearest sum-of-Kronecker-products, solved by SVD of a rearranged matrix | Van Loan & Pitsianis | 1993 |
| Reshape a dense FC weight matrix → TT/MPO, bond dimension as the knob | [Novikov et al., "Tensorizing Neural Networks"](https://arxiv.org/abs/1509.06569) | 2015 |
| MPO framing with bond dimension *D*, entanglement/area-law motivation, applied to NN weights | [Gao et al., *Phys. Rev. Research* 2, 023300](https://arxiv.org/abs/1904.06194) | 2019 |
| Tensor-decomposing **transformer self-attention**, at LM scale, with no quality loss | [Ma et al., NeurIPS 2019 (Block-Term Decomposition)](https://arxiv.org/abs/1906.09777) | 2019 |
| **MPO decomposition of a pretrained transformer + subsequent fine-tuning** | [Liu, Gao, Zhao, Lu, Wen, ACL 2021 (MPO-BERT/MPOP)](https://aclanthology.org/2021.acl-long.418/) | 2021 |
| **Training-free TT truncation of a pretrained LLM's weights** | [TensorGPT](https://arxiv.org/abs/2307.00526) | 2023 |
| MPO central tensor shared across MoE experts | [MPOE, COLING 2022](https://arxiv.org/abs/2203.01104) | 2022 |

Two of these deserve emphasis because they defeat the two claims most often made for CompactifAI.

**"First to apply MPO to a pretrained transformer and heal."** No. That is MPO-BERT (ACL 2021), from the same senior authorship lineage as Gao et al. 2019 (Ze-Feng Gao). Its workflow is exactly decompose-then-lightweight-finetune. One caveat in the other direction: MPO-BERT's headline "91% average reduction" is a reduction in **fine-tuning parameters**, a PEFT-style number closer in kind to LoRA's, not a deployed-size reduction — so it should not be placed in a compression table next to CompactifAI's 70%.

**"Novikov 2015 established the compression, so scale is the only gap."** Also not quite. Novikov's TT-layer parameters are "initialized with a Gaussian noise" and **trained from scratch** — TT is an architecture choice there, not a post-hoc operation on trained weights. So Novikov cannot be cited as precedent for decompose-then-heal. (It also cannot be cited for CompactifAI's benefit: its 194,622× figure applies to *one* matrix, the 25088×4096 first FC layer of VGG-16 on ImageNet at TT-rank 2, top-5 error 11.2%→11.5%; the whole-network figure is up to 7.4× at top-5 11.2%→12.3%. Two different operating points with different accuracy costs.)

### 5.2 What is legitimately CompactifAI's

Three things, all applied-systems contributions rather than mathematical ones:

1. **End-to-end demonstration at 7B decoder-only scale** with a full benchmark suite (MMLU, HellaSwag, BoolQ, TriviaQA, GSM8K). Prior MPO work stopped at BERT-340M (MPO-BERT), 0.16B (Ma et al.), or VGG/CIFAR (Gao et al.). This is a real gap that someone had to close, and closing it is not trivial.
2. **The depth-graded compression policy** — early layers ≥50%, late layers to 10%, last MLP per block excluded. This is a genuine empirical finding about where LLM redundancy lives, published in January 2024, *before* ShortGPT (March 2024) and Gromov et al. (March 2024) reported the same substrate fact by different means. Priority here is real and under-credited.
3. **The healing recipe** — showing that a sub-epoch pass on chat data recovers a model truncated far past what one-shot TT-SVD tolerates. Recall that independent Llama-scale TT work ([Huang et al. 2025](https://arxiv.org/abs/2501.19135)) reaches only **1.60×** whole-network compression on LLaMA2-7B. If CompactifAI's 3.3× is real, healing is doing most of the work.

### 5.3 The experiment that would settle the novelty question, and it is cheap

Because a 2-site MPO and a rank-χ SVD have **identical parameter counts** at balanced reshapes (§2.2), there is a clean, inexpensive ablation nobody has published:

> Take LlaMA-2 7B. Compress to 2.1B parameters two ways — (a) MPO truncation with the published depth schedule, (b) activation-whitened SVD (SVD-LLM V2 or ERC-SVD) at matched per-layer parameter budgets. Apply the *identical* healing recipe to both. Report MMLU, HellaSwag, BoolQ, GSM8K under one harness version.

If (a) ≈ (b), the tensor network is an initialization scheme and the contribution is the healing recipe plus the depth policy — still worth something, but not what is being sold. If (a) > (b), the Kronecker/TT hypothesis class genuinely fits transformer weights better than plain low-rank, which would be a real and publishable result. **No one has run this.** Its absence, five years and a company later, is itself information.

### 5.4 The strongest signal about novelty comes from Multiverse

As of September 2026, the public [MultiverseComputingCAI](https://huggingface.co/MultiverseComputingCAI) Hugging Face org contains roughly ten repositories. The flagship, [HyperNova 60B 2605](https://huggingface.co/MultiverseComputingCAI/Hypernova-60B-2605), is a **knowledge distillation** of gpt-oss-120b, trained with the company's own offline top-K-logit / fused-chunked-KL infrastructure ([arXiv:2608.03796](https://arxiv.org/abs/2608.03796), Aug 2026: ≈29% faster per iteration, up to 41% higher throughput on one H200, 4× context to 32,768 tokens), plus "Quantization-Aware Healing."

That is distillation plus quantization — precisely the "traditional" family the 2024 CompactifAI paper argued was suboptimal. Pulsar 16B's published workflow credits NVIDIA Model Optimizer and Megatron Bridge. **Nothing currently in the public zoo is documented as a pure MPO-truncated checkpoint**, and the models with the loudest accuracy claims (Llama 3.1/3.3 Slim, ChickBrain, SuperFly) have no downloadable weights at all.

This is circumstantially consistent with the Zagitov et al. structural thesis, though it is also consistent with ordinary commercial pragmatism at a scale where distillation simply wins. Either way: **the quantum-inspired tensor-network story is not what is shipping at the 60B scale.**

---

## 6. Practical and Deployment

### 6.1 Measured, and the best number in the corpus

[Fovet, Chamoli, Oury and Singhal (Sopra Steria), arXiv:2507.08836](https://arxiv.org/abs/2507.08836) instrumented a CompactifAI-compressed Llama 3.1 8B against the uncompressed model on a single NVIDIA Tesla V100S-PCIE-32GB (Xeon Gold 6226R, 15 cores, 43.05 GB RAM, PyTorch 2.5.1), on a 104-question RAG workload, with CodeCarbon for energy and France's grid intensity (0.05604 kgCO₂eq/kWh).

| Metric | 200-token cap | 1000-token cap |
|---|---|---|
| **GPU energy** | 0.0168 vs 0.0298 kWh (**−43.55%**) | 0.0397 vs 0.0801 kWh (**−50.5%**) |
| Total system energy | 0.0297 vs 0.0434 kWh (−30.04%) | 0.0691 vs 0.1141 kWh (−39.09%) |
| CO₂ | 2.75e-3 vs 3.93e-3 kg (−30.03%) | 6.40e-3 vs 1.05e-2 kg (−39.05%) |
| **Wall clock (104 questions)** | 10.29 vs 10.91 min (**−5.68%**) | 23.56 vs 28.79 min (**−18.17%**) |

**The GPU-energy figures (−43.5% to −50.5%) are the single most defensible efficiency numbers in the entire CompactifAI corpus.** They are also the numbers nobody quotes.

**The latency result contradicts the marketing.** −5.68% to −18.17% wall clock, against published claims of "40% faster inference" and "1.85× inference speed-up." Those marketing claims are also mutually inconsistent with each other.

**The accuracy half of this paper carries no weight and should not be cited as validation.** Ragas scores (ROUGE, BLEU, semantic similarity, factual correctness, answer correctness, response relevancy) have the compressed model "winning" 4 of 6 metrics at both budgets. With n=104, a single seed, no confidence intervals, and ChatGPT-4o generating the ground truth (so "factual correctness" measures agreement with another model), 4-of-6 is a **noise signature**, not evidence of improvement.

**Two structural problems.** First, **the paper never states the compression ratio of the model it measured** — no parameter count, no bond dimension, no size in GB — so the energy savings cannot be tied to any compression level, and cannot be matched to a shipped Slim variant. Second, Sopra Steria is a commercial systems integrator and Multiverse partner, and Multiverse hosts the paper on its own publications page while labeling it "independent." It is **second-party validation**. The authors themselves flag further limits: no cross-framework comparison (they note vLLM would likely be more efficient than their PyTorch harness), a single model, no comparison against quantization or any other compression baseline, and **no accounting for the one-off energy cost of performing the compression** — so the amortization break-even is unknown.

### 6.2 Edge and CPU

The CPU-deployment story is real but should be read carefully. Multiverse's Intel Xeon 6 results (§4.3) show a genuine ≈94% throughput improvement — on a memory-bandwidth-bound CPU baseline where nearly any size reduction roughly doubles throughput, arriving at 3.86 output tokens/second, which is below interactive reading speed. The claim is true and the regime is narrow.

The edge framing (ChickBrain on a MacBook with no network; SuperFly at 94M for appliance voice control) is the most commercially legible part of the offering and the least evidenced part of the record: no weights, no model cards, no benchmark scores.

### 6.3 Adoption

Total downloads across all ten public HF repos are in the single-digit thousands, dominated by one HyperNova revision (Hypernova-60B-2605 at ≈5.6k; LittleLamb 953; the rest in the hundreds). The one dataset, `llm-refusal-evaluation`, has 3.65k. This is thin for a company positioned on model compression, and it reflects the structural fact that the models with the loudest claims are the ones with no downloadable weights.

### 6.4 Independent scrutiny

There is essentially none.

- **No third party has replicated the 93%/2–3% LlaMA-2 7B result.**
- **No independent evaluation of any Slim, ChickBrain or SuperFly checkpoint exists** — they have no public weights.
- The only study on a real CompactifAI checkpoint is second-party and does not disclose the compression ratio.
- The nearest academic scrutiny (Zagitov et al.) does not name CompactifAI and operates two orders of magnitude away in compression regime.
- The closest independent Llama-scale TT number (Huang et al., 1.60× on LLaMA2-7B) is roughly half of CompactifAI's claimed 3.3×, and the gap has never been reconciled in public.
- A competing production tensor-network pipeline exists — [Minima](https://arxiv.org/abs/2602.01613) (Minima AI, Inc.), reporting Qwen3-32B peak VRAM 64 GiB → 40 GiB and ≈40 → ≈75 TPS at 8K context — but it bundles speculative decoding and runtime engineering, so its TN contribution is no better isolated than CompactifAI's.

Peer-review status, stated plainly: ESANN 2025 is a genuinely peer-reviewed European symposium; it is also a 7-page workshop-tier venue relative to NeurIPS/ICML, the paper covers one model family at one scale, and it contains no comparison against contemporary strong PTQ (GPTQ/AWQ) or structured pruning (SliceGPT/LLM-Pruner) at matched compression.

---

## 7. The Shrinking Addressable Surface

This is the strategic core of the brief, and unlike most of the argument it now rests on a published measurement rather than reasoning.

### 7.1 The geometry: SVD break-even rank

For a matrix $W \in \mathbb{R}^{m \times n}$, a rank-$r$ factorization costs $r(m+n)$ parameters versus $mn$. It saves nothing unless

$$r \;<\; r_t \;=\; \frac{mn}{m+n}$$

Expressed as a fraction of the available spectrum $\min(m,n)$:

$$\frac{r_t}{\min(m,n)} \;=\; \frac{\max(m,n)}{m+n}$$

Now apply this to the two matrix shapes in question.

| Matrix | $mn$ | $r_t$ | $\min(m,n)$ | Break-even as fraction of spectrum |
|---|---|---|---|---|
| Fine-grained expert, **2560 × 640** | 1,638,400 | **512** | 640 | 0.800 |
| Large dense MLP, **5120 × 17408** | 89,128,960 | **3,957** | 5120 | 0.773 |

**The geometry alone does not discriminate.** Both must discard roughly 20–23% of their spectrum just to reach zero savings. If aspect ratio were the whole story these two would behave alike. They do not, and the reason is empirical, not geometric.

### 7.2 The measurement that decides it

[MoBE (arXiv:2508.05257, ICLR 2026, Ant Group)](https://arxiv.org/html/2508.05257) measured the effective rank of individual expert weight matrices in Kimi-K2-Instruct. For expert matrices of shape 7168 × 2048 the break-even rank is $r_t \le 7168 \cdot 2048 / (7168 + 2048) = 1593$, and the paper reports (Fig. 11, Appendix B) that **the average effective rank exceeds 1593 in most layers.** Their conclusion, verbatim: *"the pure rank-decomposition-based method can't produce model compression without performance loss."*

Two softenings that must travel with this: it is an *average* effective rank, not a hard rank, and the paper says "most layers" without giving an exact count. The correct phrasing is "in most layers the average effective rank exceeds the break-even rank," not "experts are full-rank."

The 7168 × 2048 shape has break-even fraction 0.778 — geometrically almost identical to the 2560 × 640 case. The measurement therefore transfers by shape analogy, though it is a transfer and not a direct measurement of the smaller matrix.

### 7.3 Why fine-grained experts are hostile targets, in four ways

**(1) No spectral headroom.** As measured. For a 2560 × 640 expert, achieving even 30% savings requires $r = 358$ — retaining just **56% of the 640 available directions** — when the measured effective rank on comparable matrices already sits above the 512 break-even. You are truncating into signal from the first singular value you drop.

**(2) The architecture already removed the redundancy, deliberately.** [DeepSeekMoE](https://arxiv.org/abs/2401.06066)'s two mechanisms are fine-grained segmentation (split each FFN into $m$ smaller experts, activate $mK$) and **shared-expert isolation** — reserving always-on experts to absorb common knowledge *so that routed experts need not redundantly re-learn it*. The paper's own stated design goal is removing cross-expert parameter redundancy. That is exactly the quantity a post-hoc factorizer would have harvested. Fine-grained MoE **is structured compression applied at training time**, and it spends the redundancy budget before any compression vendor arrives.

**(3) Calibration is data-starved.** With hundreds of experts per layer and routing skew, each expert sees a sparse, uneven slice of tokens. Activation-aware calibration — the thing that separates a working low-rank method (SVD-LLM's whitening: LLaMA-7B WikiText PPL 7.73 at 20%) from a broken one (plain SVD: **20061** at the same 20%) — has thin per-expert statistics to work with. This is not speculation: it is the explicit premise of RFID-MoE, whose whole contribution is fusing expert *activation frequency* with effective rank to allocate ranks non-uniformly. And the method that performs best on fine-grained MoE, MoBE, is the one that needs **no calibration data at all**; the calibration-dependent D2-MoE is reported by MoBE to lose 7–14% relatively and to be infeasible on the largest models.

**(4) You compress the wrong number.** A sparse MoE already activates a small fraction of its parameters per token. Compressing total parameters is a **VRAM** win, not a **latency** win. MoBE concedes its compressed form "requires multiple times calling of current optimized kernel fused-MoE to mimic the factorization, which is relatively inefficient" — the parameter saving does not currently convert to speed at all.

### 7.4 The arithmetic that closes the argument

For the stated Qwen3.8-Flash-Next configuration — 512 experts of 2560 × 640, with **96.5% of parameters in these small near-full-rank matrices**:

- Per expert (gate/up/down): $3 \times 1{,}638{,}400 = 4.92$M parameters
- Per MoE layer: $512 \times 4.92\text{M} = 2.52$B parameters

Now the decisive step. **If 96.5% of a model's parameters sit in matrices with approximately zero usable low-rank headroom, then compressing everything else perfectly — to zero — caps total model reduction at 3.5%.** The addressable surface is not merely difficult; it is arithmetically bounded to irrelevance before any method quality enters the discussion.

The published ceiling confirms the shape of this. The best data-free method on 2025-era fine-grained MoE reaches **24–30%** total-parameter reduction at ≈1.3–1.4% average accuracy loss (MoBE, and remember only DeepSeek-V3 reaches 30%). Push to 60% and quality collapses: RFID-MoE on Qwen3-30B lands at 16.92 PTB perplexity. That is a long way from the 70–90% ratios quoted on older dense checkpoints.

**And the field has already moved off per-matrix factorization in response.** MoBE writes each expert's up/gate matrix as $W^i = A^i \cdot f\big(\sum_j \alpha^{i,j} B^j\big)$ with $\{B^j\}$ **shared across all experts in a layer**; MoE-SVD shares a single V-matrix across experts with top-k U selection; MPOE (COLING 2022) shared an MPO central tensor across experts. In every case the compressible quantity has shifted from *"this matrix is low-rank"* to *"this ensemble of experts is mutually redundant."* That is a different mathematical premise requiring a different algorithm, and it is precisely the premise shared-expert isolation was designed to weaken. All three methods also leave the down-projection alone; MoBE states down-projections are "less amenable to effective compression."

### 7.5 Hybrid linear attention removes a second pool

A Qwen3-Next-class layout — verified from the [Qwen3-Next-80B-A3B model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct): 48 layers arranged as 12 × (3 × Gated DeltaNet→MoE, 1 × Gated Attention→MoE), hidden 2048, gated attention 16 Q / 2 KV heads at head dim 256, Gated DeltaNet 32 V / 16 QK heads at head dim 128 — means **75% of layers have no full attention at all**. Gated DeltaNet layers carry compact fixed-size recurrent state instead of large per-layer attention projections plus a growing KV cache. Both "factorize the attention projections" and "compress the KV cache" shrink as targets.

### 7.6 Large dense MLPs remain good targets, and here is why

For the stated Qwen3.8-27B configuration, an MLP matrix of **5120 × 17408**:

- One matrix: 89.13M parameters
- Break-even rank: 3,957 of 5,120 available
- At $r = 1280$: $22{,}528 \times 1280 = 28.84$M → **67.6% reduction on that matrix**, retaining 25% of directions

The empirical evidence that this headroom is real and usable: activation-whitened SVD holds LLaMA-7B at WikiText PPL 9.27 at 40% compression and 15.00 at 60% (SVD-LLM), and CompactifAI reports 70% parameter reduction on LlaMA-2 7B at ≈4–5% relative accuracy loss on MMLU/HellaSwag/BoolQ. The mechanistic reading — and this is reasoning, not a cited finding — is that a 17408-wide FFN superposes a very large number of features into a wide space with heavy inter-feature correlation and no architectural pressure toward orthogonality, whereas a 640-wide fine-grained expert has already been narrowed by the architect and de-duplicated against its siblings by shared-expert isolation.

### 7.7 GQA is the worked historical precedent

The KV cache was once the obvious compression target. Then [GQA](https://arxiv.org/abs/2305.13245) absorbed the fix into the architecture, retrofittable onto existing checkpoints by "uptraining" at a claimed ≈5% of pretraining compute, and DeepSeek-V2's MLA went further by making the KV projection a **trained-in low-rank factorization**. Once the factorization ships inside the checkpoint, there is nothing left for a post-hoc factorizer to factorize. The same absorption is now visibly underway for expert matrices.

### 7.8 And the vendor may ship the small model for free

[MatFormer](https://arxiv.org/abs/2310.07707) trains nested, independently-extractable submodels — sub-block $i$ uses the first $m_i$ FFN neurons, all granularities jointly optimized, with Mix'n'Match yielding more valid submodels than were explicitly trained. Gemma 3n ships this: E4B contains a simultaneously-optimized E2B. If the model vendor hands the customer a correctly-trained smaller model at zero marginal cost, with no calibration data, no healing step and no accuracy claim to litigate, the post-hoc compressor is competing against free.

### 7.9 The strategic finding

[SlimQwen (arXiv:2605.08738, 2026)](https://arxiv.org/abs/2605.08738), applying prune-then-distill to a Qwen3-Next-80B-A3B parent to produce a 23B-A2B child, reports that **"different one-shot expert compression methods converge to similar final performance after large-scale continual pretraining,"** alongside the finding that pruning a pretrained MoE consistently beats training the target architecture from scratch.

If the sophistication of the one-shot compression operator washes out once you spend a continual-pretraining budget, then the entire remaining moat for any clever factorization method is the regime where the customer **cannot or will not retrain**: private or fine-tuned checkpoints, no pretraining corpus, no cluster, a hard VRAM target, and tolerance for ≈25% rather than ≈75% parameter reduction. That is a genuine market. It is a **memory-footprint market, not a latency market**, and it narrows every year.

---

## 8. Open Questions and Frontier

### 8.1 The four experiments that would settle the science

1. **The MPO-vs-SVD ablation at matched parameters and matched healing.** §5.3. Cheap, decisive, unpublished after five years. Its absence is the single loudest silence in the record.

2. **A single-harness cross-method table.** Llama-2-7B and Llama-3-8B, parameter-count-matched at 30/50/70% reduction, MMLU + HellaSwag + BoolQ, each method run both with and without an identical healing budget: CompactifAI/MPO vs SVD-LLM V2 vs a 2025–26 activation-aware method (ERC-SVD, PGSVD, BALF, AIR) vs SliceGPT+RFT vs SparseGPT/Wanda at matched *realized* memory. [LowRankArena (arXiv:2608.26389)](https://arxiv.org/html/2608.26389) is the likeliest existing vehicle for the low-rank half. Until this exists, cross-paper deltas are indicative only — and note that even the shared benchmark is contaminated: CompactifAI reports dense Llama-2-7B HellaSwag at **80.55** while SliceGPT reports **75.99**, almost certainly `acc_norm` vs `acc`, and the mismatch is unresolved.

3. **Reconciling 1.60× with 3.3×.** Huang et al. get 1.60× whole-network TT compression on LLaMA2-7B; CompactifAI claims 3.3× on parameters. Either healing recovers far more than one-shot TT-SVD alone (in which case the tensor network is an initialization scheme and should be evaluated as one), or the two measure different quantities. This is answerable with one experiment and has never been addressed publicly.

4. **Tying energy to a compression level.** The Sopra Steria GPU-energy result (−43.5% to −50.5%) is the strongest measured claim in the corpus and is currently uncitable for engineering purposes because the compression ratio of the measured model is unstated. Re-running it at three declared ratios, on vLLM rather than PyTorch, against an int4 baseline rather than fp16, and including the amortized one-off compression energy, would convert marketing into a procurement input.

### 8.2 Open technical questions

**Does joint factorize-and-quantize beat sequential at the 50–70% parameter regime?** CALDERA answers this below 2.5 effective bits with rank-64 additive terms. Nobody has run the CALDERA formulation with an MPO term instead of a rank-$k$ term, or at CompactifAI's operating point. And the interference mechanism (3) in §3.6 — that AWQ/SmoothQuant channel-scaling and QuIP#/QTIP incoherence processing are structurally incompatible with truncated factors — remains **pure analysis with zero published measurement.** It is the most testable unexamined hypothesis in this brief.

**Is deep-layer redundancy a property or an artifact?** Gromov et al. state the double-edged reading explicitly: either "pretraining is not properly leveraging the deeper-layer parameters," or shallow layers store the knowledge. If it is the former, better pretraining (or longer training, or better data) closes the gap and the redundancy that four independent methods depend on evaporates. NVIDIA's finding that **width pruning beats depth pruning** on Llama-3.1-8B is a mild data point in that direction.

**Do MoE-native shared-basis methods generalize back to dense models?** MoBE's structure ($W^i = A^i f(\sum_j \alpha^{i,j} B^j)$) shares a basis across a layer's experts. Dense transformers have layer-to-layer redundancy of a similar flavor — the whole premise of LaCo's layer merging. Nobody has tried a cross-layer shared basis on a dense model.

**What is the right unit of compression?** The field's trajectory answers this by revealed preference: from the matrix (SVD, MPO), to the residual stream (SliceGPT), to the block (ShortGPT, LaCo), to the ensemble of experts (MoBE, MoE-SVD, MPOE). Each move up in scope found redundancy the previous unit could not see. The next unit is probably cross-layer or cross-module, and nobody has claimed it.

### 8.3 Open questions specific to Multiverse

- **Will any pure-MPO checkpoint ever ship with public weights?** As of Sept 2026 none has. Pulsar's workflow credits NVIDIA tooling; HyperNova is distillation. This is the question that determines whether CompactifAI is a method or a brand.
- **What is Quasar 438B?** Compression, distillation, or from-scratch. No source says. "First large model" and "CompactifAI output" are in tension.
- **What compression ratio was measured in arXiv:2507.08836?** Without it, the best number in the corpus cannot be used.
- **Why has no eval log, harness version, or model card accompanied any Slim or ChickBrain benchmark claim?** The HellaSwag 0.5905 baseline for Llama 3.3 70B and the WinoGrande +6.86% "gain" both indicate a nonstandard evaluation setup that has never been described.

### 8.4 The honest summary

**What CompactifAI does:** reshapes attention and MLP weight matrices of a pretrained transformer into higher-order tensors, truncates them as Matrix Product Operators via sequential SVDs at a depth-graded bond dimension, and heals with a sub-epoch retrain. It reports **70% parameter reduction** on LlaMA-2 7B at roughly 4–5% relative loss on MMLU, HellaSwag and BoolQ, with GSM8K holding until an additional 4-bit quantization step drops it 23% relative.

**What the mathematics is:** structured low-rank approximation in a permuted basis — for two sites, exactly the 1993 nearest-Kronecker-product problem, solved by one SVD; for more sites, Oseledets' 2011 Tensor Train. Strictly multilinear. Genuinely a different hypothesis class from rank-χ SVD at identical parameter count, and genuinely not a different mechanism. "Quantum-inspired" is accurate about provenance and empty about computation.

**What is novel:** the depth-graded sensitivity policy (published before ShortGPT and Gromov et al. found the same fact), the demonstration at 7B decoder-only scale, and the healing recipe. Not the format, not the physics framing, not decompose-then-heal on pretrained transformers, and not tensor decomposition of self-attention — those are Oseledets 2011, Gao 2019, MPO-BERT 2021 and Ma 2019 respectively.

**What the headline number is:** a memory footprint against an fp32 baseline with int4 quantization folded in. Against fp16 it is 84.5%; against an int4 model of the same architecture, roughly 38%. The factorization-attributable figure is 70%, and the method's own design — leaving tensorized layers at fp16 and quantizing only the rest — is evidence that the two axes do not compose as cleanly as the headline implies.

**What the field says:** an independent systematic study finds tensor decompositions structurally mismatched to LLM weights and losing to plain 4-bit round-to-nearest at matched bit savings (though at 8–24% compression, not 70%, and without naming CompactifAI). Independent Llama-scale TT work reaches 1.60× where CompactifAI claims 3.3×, unreconciled. No third party has replicated the headline result, and the models with the loudest claims have no public weights.

**Where the method still fits:** compressing a specific existing dense checkpoint, cheaply, when you have no pretraining corpus and no cluster, and you need to hit a hard VRAM target on stock kernels. Large dense MLPs — 5120 × 17408 with 3,957 rank break-even and 77% spectral headroom — are exactly the right substrate. Fine-grained MoE experts at 2560 × 640, measured near or above their 512-rank break-even, holding 96.5% of the model, are exactly the wrong one. That surface is shrinking, and the vendors are shrinking it on purpose.