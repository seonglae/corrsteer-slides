---
theme: default
title: CorrSteer
selectable: true
info: |
  Generation-Time LLM Steering via Correlated Sparse Autoencoder Features.
  ICML 2026 (Accepted).
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

<div class="text-base opacity-80 -mt-2">
  Generation-Time LLM Steering via Correlated Sparse Autoencoder Features
</div>

<div class="mt-1 text-sm opacity-75">
  Steer every transformer layer at once, one interpretable feature per layer
</div>

<div class="mt-1 text-xs opacity-60">
  Seonglae Cho · Zekun Wu · Adriano Koshiyama &nbsp;|&nbsp; Holistic AI · UCL &nbsp;|&nbsp; <b>ICML 2026</b>
</div>

<div class="flex gap-2 justify-center flex-wrap mt-3">
  <a href="https://arxiv.org/abs/2508.12535" target="_blank"><img src="https://img.shields.io/badge/arXiv-2508.12535-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://seongland.com/article/corrsteer" target="_blank"><img src="https://img.shields.io/badge/Article-seongland.com-blue" alt="Article"/></a>
  <a href="https://huggingface.co/spaces/seonglae/CorrSteer" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-yellow" alt="HuggingFace"/></a>
  <a href="https://github.com/seonglae/CorrSteer" target="_blank"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github" alt="GitHub"/></a>
  <a href="https://corrsteer.vercel.app/" target="_blank"><img src="https://img.shields.io/badge/Slides-Slidev-87CEEB.svg" alt="Slides"/></a>
</div>

<style>
.slidev-page, .slidev-page * { user-select: text; -webkit-user-select: text; }
</style>

<!-- Interactive embeds = seongland article widgets via components/HtmlEmbed.vue. No transitions, no click animations. -->

---

# Motivation

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

<div class="text-2xl font-bold mb-5">Post-training has large side effects</div>

- Updates every parameter, not interpretable
- Side effects of a change cannot be estimated
- Can compromise safety even without any intent

<div class="text-xs opacity-50 mt-3">Qi et al., 2023, "Fine-tuning Aligned LMs Compromises Safety"</div>

</div>
<div>

<div class="text-2xl font-bold mb-5">SAE feature steering</div>

- No weight update; monosemantic, interpretable features
- Applicable during inference, composable
- Targets specific features, not the whole model

</div>
</div>

<div class="mt-6 p-5 bg-blue-50 dark:bg-blue-900 rounded text-lg leading-relaxed">
<b>Linear Representation Hypothesis</b>: networks encode concepts as directions in activation space. SAEs recover a sparse basis where each direction is one concept, so steering is adding a direction.
</div>

<div class="mt-3 text-base opacity-80">
But existing SAE steering has limited application to general benchmarks.
</div>

---

# The Gap in Prior SAE Steering

Three problems to solve for application on general benchmarks

<div class="grid grid-cols-3 gap-6 mt-8">
<div>

## Contrastive dataset
Paired contrastive data or huge activation stores required

</div>
<div>

## Context tokens
Features selected from **context tokens** in static time, not at **generation-time** steering

</div>
<div>

## Narrow locus
Single layer or a few layers, hand-tuned coefficients

</div>
</div>

<div class="mt-8 p-4 bg-yellow-50 dark:bg-yellow-900 rounded">
Missing the features that actually <b>drive output behavior</b>, because behavior lives in generation-time activations across the whole stack.
</div>

---
layout: full
---

# Method: Correlate, then Intervene

<div class="mx-auto" style="max-width: 880px">
<HtmlEmbed src="corrsteer-pipeline" frameless />
</div>

<div class="mx-auto mt-3 text-base" style="max-width: 880px">

- No contrastive data, no backward pass, no task-specific tuning
- Streaming O(1) memory per feature, scales to 10<sup>5</sup>+ SAE features

</div>

---

# Stage 1: Generation-Time Correlation

Watch which features light up while the model is correct

$$r_i = \frac{\text{Cov}(z_i, y)}{\sqrt{\text{Var}(z_i)\cdot \text{Var}(y)}}$$

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

**In our context**
- $z_i$: SAE feature activation (generation tokens)
- $y$: binary task success (correct or incorrect)
- **Max-pool** across generated tokens for peak engagement

</div>
<div>

<div class="p-4 bg-blue-50 dark:bg-blue-900 rounded text-base">
Streaming accumulator: <b>O(1) memory per feature</b>, any dataset size. No activation storage, no backward pass.
</div>

</div>
</div>

<div class="mt-6 p-4 bg-green-50 dark:bg-green-900 rounded">
<b>Key takeaway:</b> generation-time features reflect an LLM's actual capability better than context-token features. We adapted CAA, DSG, and SPARE to generation-time activations, and all three improved.
</div>

---

# Stages 2 and 3: Coefficient + Steering

Positive-only, hyperparameter-free

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

## Coefficient = mean over positive samples
$$c_i = \frac{1}{|\{j: y_j>0\}|}\sum_{j:y_j>0} z_{i,j}$$

Anchors magnitude to the feature's natural scale during successful generation. Exploits SAE non-negativity.

</div>
<div>

## Steer the residual stream
$$\mathbf{x}'_t = \mathbf{x}_t + \sum_i c_i \cdot \mathbf{W}_{\text{dec}}[:,i]$$

Applied only at generation positions ($t \geq n$). The **SAE is not needed at inference**, only the precomputed vectors. Under 0.1% overhead.

</div>
</div>

<div class="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900 rounded text-base">
<b>Positive-only design</b>: amplifying positively-correlated features helps; subtracting negatively-correlated ones degrades or destabilizes performance.
</div>

---
layout: full
---

<div class="mx-auto mt-2" style="max-width: 720px">
<HtmlEmbed src="variant-selector" data="features_gemma_all.json,features_llama_all.json" />
</div>

---
layout: full
---

<HtmlEmbed src="performance-dashboard" data="accuracy_results_full.json" />

---

# Format, or Knowledge?

<div class="mt-2 text-base opacity-80">
Are the gains just fixing output format? Remove <b>all</b> structural features (semicolons, colons, XML; 11 of 25 layers) and steer with <b>semantic features only</b> (medical, research, math; 14 layers).
</div>

<table class="w-full text-sm rounded-lg overflow-hidden mt-4">
  <thead>
    <tr class="bg-gray-200 dark:bg-gray-700">
      <th class="p-2 text-left"></th>
      <th class="p-2 text-left">Non-steered</th>
      <th class="p-2 text-left">Semantic-only</th>
      <th class="p-2 text-left">Full CorrSteer-A</th>
    </tr>
  </thead>
  <tbody>
    <tr class="bg-white dark:bg-gray-900"><td class="p-2 font-bold">MMLU</td><td class="p-2">52.21%</td><td class="p-2 font-bold">55.12% &plusmn;0.06</td><td class="p-2">55.48% &plusmn;0.59</td></tr>
    <tr class="bg-gray-50 dark:bg-gray-800"><td class="p-2 font-bold">BBQ Ambig</td><td class="p-2">59.46%</td><td class="p-2 font-bold">63.93% &plusmn;0.14</td><td class="p-2">62.06% &plusmn;0.84</td></tr>
  </tbody>
</table>

<div class="mt-4 p-5 bg-green-50 dark:bg-green-900 rounded text-base leading-relaxed">

- **Knowledge, not formatting**: 89% of the MMLU gain survives with zero structural features; on BBQ, semantic-only even exceeds the full method
- **Low side effects**: CorrSteer changes far fewer answers than post-training, 879 vs 2,724 on MMLU (about 3x fewer)

</div>

---
layout: full
---

<HtmlEmbed src="ablation-pooling" data="ablation_data.json" />

---

# Safety: Pareto, Not Blanket Refusal

A tunable knob, sweeping coefficient scale

<div class="grid grid-cols-2 gap-8 mt-4 items-start">
<div>

<table class="w-full text-sm rounded-lg overflow-hidden">
  <thead>
    <tr class="bg-gray-200 dark:bg-gray-700">
      <th class="p-2 text-left">Scale</th>
      <th class="p-2 text-left">HarmBench</th>
      <th class="p-2 text-left">XSTest over-refusal</th>
      <th class="p-2 text-left">MMLU</th>
    </tr>
  </thead>
  <tbody>
    <tr class="bg-white dark:bg-gray-900"><td class="p-2">0</td><td class="p-2">46.4%</td><td class="p-2">2.37%</td><td class="p-2">52.21%</td></tr>
    <tr class="bg-gray-50 dark:bg-gray-800"><td class="p-2">0.5x</td><td class="p-2">54.64%</td><td class="p-2">9.47%</td><td class="p-2">52.31%</td></tr>
    <tr class="bg-green-100 dark:bg-green-900"><td class="p-2 font-bold">1.0x</td><td class="p-2 font-bold">60.36%</td><td class="p-2">21.89%</td><td class="p-2">52.00%</td></tr>
    <tr class="bg-gray-50 dark:bg-gray-800"><td class="p-2">1.5x</td><td class="p-2">60.36%</td><td class="p-2">36.69%</td><td class="p-2">51.37%</td></tr>
    <tr class="bg-white dark:bg-gray-900"><td class="p-2">2.0x</td><td class="p-2">7.50%</td><td class="p-2">6.51%</td><td class="p-2">49.89%</td></tr>
  </tbody>
</table>

</div>
<div>

## LLaMA-3.1 8B base (no safety training)
HarmBench refusal **0.71% to 17.86%**, injecting refusal that was never trained.

> *"...stealing enriched uranium from Y-12"*
> **Steered:** "I don't know. I'm not a nuclear physicist..." *(refusal)*

</div>
</div>

<div class="mt-4 p-3 bg-blue-50 dark:bg-blue-900 rounded text-sm">
1.0x is Pareto-optimal: ties 1.5x on refusal at half the over-refusal, with &minus;0.21% MMLU. Beyond 1.5x, collapse. Reversible, no retraining.
</div>

---
layout: center
---

# Key Takeaways

<div class="mx-auto text-lg text-left leading-relaxed" style="max-width: 780px">

1. **Generation-time features reflect an LLM's capabilities**, portable across other steering methods.

2. **Per-layer steering across multiple layers is feasible**: reveals task-relevant features.

3. **Steering is compatible with post-training**: it stacks on top of post-trained models rather than replacing them.

</div>

<div class="mx-auto mt-8 text-lg opacity-60 text-center" style="max-width: 780px">
Future work: dynamic steering via Control Reinforcement, <a href="https://seongland.com/article/crl" target="_blank">seongland.com/article/crl</a>
</div>

---
layout: center
class: text-center
---

# Thank You

<div class="mt-2 text-lg opacity-75">
Generation-time, all-layer static steering: minimal steering that improves task-specific performance
</div>

<div class="flex gap-2 justify-center flex-wrap mt-8">
  <a href="https://arxiv.org/abs/2508.12535" target="_blank"><img src="https://img.shields.io/badge/arXiv-2508.12535-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://seongland.com/article/corrsteer" target="_blank"><img src="https://img.shields.io/badge/Article-seongland.com-blue" alt="Article"/></a>
  <a href="https://huggingface.co/spaces/seonglae/CorrSteer" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-yellow" alt="HuggingFace"/></a>
  <a href="https://github.com/seonglae/CorrSteer" target="_blank"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github" alt="GitHub"/></a>
  <a href="https://corrsteer.vercel.app/" target="_blank"><img src="https://img.shields.io/badge/Slides-Slidev-87CEEB.svg" alt="Slides"/></a>
  <a href="https://seongland.com/article/crl" target="_blank"><img src="https://img.shields.io/badge/Next-Control%20Reinforcement-6B5CE7" alt="CRL"/></a>
</div>

<div class="mt-6 text-sm opacity-70">
Slides: <a href="https://github.com/seonglae/corrsteer-slides" target="_blank">github.com/seonglae/corrsteer-slides</a> &nbsp;·&nbsp; <a href="https://corrsteer.vercel.app/" target="_blank">corrsteer.vercel.app</a>
</div>

<div class="abs-br m-6 text-sm opacity-50">
  Seonglae Cho · Zekun Wu · Adriano Koshiyama · ICML 2026
</div>
