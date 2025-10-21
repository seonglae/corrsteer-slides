---
theme: default
title: CorrSteer
info: |
  Correlation-based Steering with Sparse Autoencoders
class: text-center
drawings:
  persist: true
mdc: true
layout: cover
fonts:
  sans: 'Times New Roman'
  serif: 'Times New Roman'
  mono: 'Courier New'
---

# CorrSteer

Correlation-based Steering with Sparse Autoencoders

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next page <carbon:arrow-right class="inline"/>
  </span>
</div>

<div class="abs-bottom m-6 flex gap-2 items-center justify-center text-sm">
  <div>Seonglae Cho · Zekun Wu · Adriano Koshiyama</div>
</div>

<div class="abs-br m-6 flex gap-2">
  <a href="https://github.com/seonglae/CorrSteer" target="_blank" alt="GitHub" title="Open in GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:logo-github />
  </a>
  <a href="https://huggingface.co/spaces/seonglae/CorrSteer" target="_blank" alt="HuggingFace" title="Open in HuggingFace"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24"><path fill="currentColor" d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12s12-5.4 12-12S18.6 0 12 0m.7 4.7c.4 0 .7.3.7.7v1.8c0 .4-.3.7-.7.7h-1.4c-.4 0-.7-.3-.7-.7V5.4c0-.4.3-.7.7-.7zm-5.9 2c.2-.3.6-.4.9-.2l1.5.9c.3.2.4.6.2.9l-.9 1.5c-.2.3-.6.4-.9.2l-1.5-.9c-.3-.2-.4-.6-.2-.9zm10.4 0l.9 1.5c.2.3.1.7-.2.9l-1.5.9c-.3.2-.7.1-.9-.2l-.9-1.5c-.2-.3-.1-.7.2-.9l1.5-.9c.3-.2.7-.1.9.2M12 8.3c2 0 3.7 1.7 3.7 3.7s-1.7 3.7-3.7 3.7s-3.7-1.7-3.7-3.7s1.7-3.7 3.7-3.7m-7.4 2.9c.4 0 .7.3.7.7v1.8c0 .4-.3.7-.7.7H2.8c-.4 0-.7-.3-.7-.7v-1.8c0-.4.3-.7.7-.7zm14.8 0c.4 0 .7.3.7.7v1.8c0 .4-.3.7-.7.7h-1.8c-.4 0-.7-.3-.7-.7v-1.8c0-.4.3-.7.7-.7zM7.7 16l.9 1.5c.2.3.1.7-.2.9l-1.5.9c-.3.2-.7.1-.9-.2l-.9-1.5c-.2-.3-.1-.7.2-.9l1.5-.9c.3-.2.7-.1.9.2m8.6 0c.2-.3.6-.4.9-.2l1.5.9c.3.2.4.6.2.9l-.9 1.5c-.2.3-.6.4-.9.2l-1.5-.9c-.3-.2-.4-.6-.2-.9zm-4.9 1.7h1.4c.4 0 .7.3.7.7v1.8c0 .4-.3.7-.7.7h-1.4c-.4 0-.7-.3-.7-.7v-1.8c0-.4.3-.7.7-.7"/></svg>
  </a>
</div>

---

# Motivation

**Problem**: Existing SAE-based steering methods have limitations

- Require contrastive datasets or large activation storage
- Rely on hidden states of context tokens
- Limited to specific applications (bias, unlearning, jailbreaking)
- Do not directly reflect generation capabilities

**Solution**: CorrSteer

- Uses generation-time activations
- Correlates with task outcomes
- Fully automated pipeline
- No hyperparameter tuning required

---

<div class="flex justify-center items-center h-full">
  <img src="./corrsteer-emnlp/image/system_diagram.png" class="max-h-100" />
</div>

<div class="text-xs text-center mt-4">
Generation-time steering: selecting features by correlating activations with outcomes
</div>

---

# Method Overview

<div class="grid grid-cols-2 gap-8">

<div>

**Correlation-based Selection**:

$$
r_i = \frac{\text{Cov}(z_i, y)}{\sqrt{\text{Var}(z_i) \cdot \text{Var}(y)}}
$$

**Coefficient from Positive Samples**:

$$
c_i = \frac{1}{|\{j : y_j > 0\}|} \sum_{j : y_j > 0} z_{i,j}
$$

**Steering at Inference**:

$$
\mathbf{x}' = \mathbf{x} + c_i \cdot \mathbf{W}_{\text{dec}}[:, i]
$$

</div>

<div>

**Three Variants**:

- **CorrSteer-S**: Single feature (global)
- **CorrSteer-A**: Top per layer (all)
- **CorrSteer-P**: Validated pruning

<img src="./corrsteer-emnlp/image/corrsteer_methods.png" class="mt-4" />

**Key Properties**:
- $O(1)$ memory streaming
- Max-pooling aggregation
- Generation-time focus

</div>

</div>

---

# Results: Accuracy (Gemma-2 2B)

<div class="text-sm">

| Method | MMLU | MMLU-Pro | BBQ Ambig | BBQ Disambig | HarmBench | XSTest |
|--------|------|----------|-----------|--------------|-----------|--------|
| Non-steered | 52.21 | 30.40 | 59.46 | 75.38 | 46.61 | 86.35 |
| **CorrSteer-S** | 52.99 | 30.38 | 62.39 | 75.70 | 46.61 | 86.77 |
| **CorrSteer-P** | 54.70 | 30.63 | **66.00** | 76.48 | **66.08** | 86.46 |
| **CorrSteer-A** | **55.48** | **30.93** | 62.06 | **76.53** | **73.75** | **86.98** |
| Fine-tuning | **55.75** | **35.32** | -- | -- | -- | -- |

</div>

**Key Findings**:
- **+3.3%** on MMLU (4000 samples), **+27.2%** on HarmBench (108 samples)
- Outperforms MI/Fisher methods, competitive with fine-tuning
- Multi-layer (A, P) > Single-layer (S) → circuit effects

---

# Results: Side Effect Ratio (SER)

$$
\text{SER} = \frac{\# \text{ negatively changed answers}}{\# \text{ all changed answers}}
$$

<div class="text-sm mt-4">

| Task | CorrSteer-S | CorrSteer-P | CorrSteer-A | Fine-tuning |
|------|-------------|-------------|-------------|-------------|
| MMLU | 0.36 | **0.17** | 0.20 | 0.41 |
| MMLU-Pro | **0.42** | **0.42** | **0.42** | 0.46 |
| BBQ Ambig | **0.00** | **0.00** | 0.08 | -- |
| HarmBench | 0.25 | 0.14 | **0.04** | -- |

</div>

**CorrSteer halves side effects** compared to fine-tuning while maintaining competitive accuracy

---

# Discovered Features: BBQ Bias Mitigation

<div class="text-sm">

**Disambiguous Context**:
- [L17/5137](https://neuronpedia.org/gemma-2-2b/17-gemmascope-res-16k/5137): Mathematical symbols and functions related to field theories (corr: **0.559**)
- [L20/12748](https://neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/12748): Structured data representations and their attributes (corr: 0.394)
- [L19/15745](https://neuronpedia.org/gemma-2-2b/19-gemmascope-res-16k/15745): Phrases related to decision-making and choice in social contexts (corr: 0.226)

**Ambiguous Context**:
- [L17/11021](https://neuronpedia.org/gemma-2-2b/17-gemmascope-res-16k/11021): Phrases related to scientific research and findings (corr: **0.554**)
- [L18/14447](https://neuronpedia.org/gemma-2-2b/18-gemmascope-res-16k/14447): References to medical data and statistics (corr: 0.533)
- [L10/4557](https://neuronpedia.org/gemma-2-2b/10-gemmascope-res-16k/4557): Interactions involving guessing or determining the correctness of information (corr: 0.202)

</div>

<div class="text-xs mt-2">
Features align with structured reasoning and format adherence rather than demographic content
</div>

---

# Discovered Features: HarmBench Safety

<div class="text-sm">

**Top Correlated Features** (all >0.7 correlation):

- [L7/11722](https://neuronpedia.org/gemma-2-2b/7-gemmascope-res-16k/11722): Phrases related to legal terms and the rejection of arguments in court cases (corr: **0.779**)
- [L9/9298](https://neuronpedia.org/gemma-2-2b/9-gemmascope-res-16k/9298): Strongly negative or dismissive opinions about claims and arguments (corr: 0.775)
- [L14/2949](https://neuronpedia.org/gemma-2-2b/14-gemmascope-res-16k/2949): Statements and phrases related to political criticism and condemnation (corr: 0.739)
- [L15/1570](https://neuronpedia.org/gemma-2-2b/15-gemmascope-res-16k/1570): Judgments regarding moral and ethical standards related to exploitation and human rights issues (corr: 0.742)
- [L25/3912](https://neuronpedia.org/gemma-2-2b/25-gemmascope-res-16k/3912): Negative sentiments or refusals (corr: 0.711)

</div>

<div class="grid grid-cols-2 gap-4 mt-4">

<div>
<img src="./corrsteer-emnlp/image/gemma-harmbench.png" />
</div>

<div>
<img src="./corrsteer-emnlp/image/gemma2b_harmbench_global_frequency.png" />
<div class="text-xs text-center">Near 100% activation frequency</div>
</div>

</div>

---

# Feature Frequency Analysis

<div class="grid grid-cols-2 gap-4">

<div>
<img src="./corrsteer-emnlp/image/gemma2b_simpleqa_global_frequency.png" />
<div class="text-xs text-center">SimpleQA: ~0% frequency</div>
</div>

<div>
<img src="./corrsteer-emnlp/image/gemma2b_harmbench_global_frequency.png" />
<div class="text-xs text-center">HarmBench: ~100% frequency</div>
</div>

</div>

**Key Insight**: Task-relevant features show high activation frequency (~100%), contrasting with typical SAE sparsity (<5%)

This indicates deeply embedded task-specific capabilities in the model

---

# Sample Efficiency

<div class="flex justify-center">
<img src="./corrsteer-emnlp/image/gemma2b_mmlu_progress.png" class="h-100" />
</div>

- Effective with **~100 samples**, stable at **~4000 samples**
- Feature selection stabilizes quickly
- No improvement beyond 4000 samples

---

# Cross-Task Transferability

<div class="text-sm">

| Source → Target | MMLU | MMLU-Pro | BBQ Disambig | BBQ Ambig |
|-----------------|------|----------|--------------|-----------|
| **MMLU** | **56.32** (52.23) | **19.67** (14.00) | 74.62 (75.42) | **64.01** (59.10) |
| **MMLU-Pro** | 55.73 (52.23) | 17.56 (14.00) | 76.10 (75.42) | 60.97 (59.10) |
| **BBQ Disambig** | 54.74 (52.23) | 16.11 (14.00) | **76.53** (75.42) | 60.85 (59.10) |
| **BBQ Ambig** | 53.85 (52.23) | 11.01 (14.00) | 76.10 (75.42) | 62.08 (59.10) |

</div>

- MMLU ↔ MMLU-Pro: Strong transfer (shared format)
- BBQ → MMLU: Bias features capture general reasoning
- Task-specific > Transferred features

---

# Ablation: Pooling Strategy

<div class="text-sm">

| Task | Non-steered | Max-pooling | Mean-pooling | All-token |
|------|-------------|-------------|--------------|-----------|
| MMLU | 52.23 | **56.32** | 56.32 | 52.91 |
| BBQ Disambig | 75.42 | **76.53** | 76.53 | 75.00 |
| HarmBench | 44.64 | **67.50** | 0.00 | 47.14 |
| XSTest | 86.35 | **87.30** | 53.65 | 86.35 |

</div>

**Max-pooling wins**: Captures peak activations, filters noise

Mean-pooling fails on multi-token generation (HarmBench: 0%, XSTest: 53.65%)

---

<style>
.compact-list li {
  margin-bottom: 0.25rem;
}
.compact-list ul {
  margin-top: 0.25rem;
}
</style>

# Key Advantages

<div class="compact-list text-sm">

1. **Low Side Effects**: SER = 0.17 (vs 0.41 fine-tuning) on MMLU
   - Halves side effects compared to fine-tuning
   - CorrSteer-P achieves best balance

2. **Efficiency**: $O(1)$ memory, ~100 samples minimum, stable at ~4000
   - Streaming algorithm, no SAE at inference
   - Works with minimal data

3. **Automation**: No hyperparameter tuning, fully automated pipeline
   - Three variants: S (single), A (all), P (pruned)
   - No manual feature engineering

4. **Interpretability Alignment**: Features align with task requirements
   - BBQ: Structured reasoning, decision-making
   - HarmBench: Legal rejection, moral judgments

</div>

---
layout: center
class: text-center
---

# Thank You!

CorrSteer: Correlation-based Steering with Sparse Autoencoders

<div class="mt-8">

**Code:** [github.com/seonglae/CorrSteer](https://github.com/seonglae/CorrSteer)

**Demo:** [huggingface.co/spaces/seonglae/CorrSteer](https://huggingface.co/spaces/seonglae/CorrSteer)

**Paper:** ICLR 2026 (Submitted)

**Contact:** seonglae.cho@holisticai.com

</div>

<div class="abs-br m-6 text-sm opacity-50">
  Seonglae Cho · Zekun Wu · Adriano Koshiyama
</div>
