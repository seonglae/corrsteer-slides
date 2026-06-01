---
theme: default
title: CorrSteer
info: |
  Generation-Time LLM Steering via Correlated Sparse Autoencoder Features.
  ICML 2026 (Accepted). Pushing static steering to its limit: every layer, at once.
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

Generation-Time LLM Steering via Correlated Sparse Autoencoder Features

<div class="pt-6 text-lg opacity-75">
  Steer every transformer layer at once, and still improve performance
</div>

<div class="mt-6 text-sm opacity-60">
  Seonglae Cho · Zekun Wu · Adriano Koshiyama &nbsp;|&nbsp; Holistic AI · UCL &nbsp;|&nbsp; <b>ICML 2026 (Accept)</b>
</div>

<div class="flex gap-2 justify-center flex-wrap mt-8">
  <a href="https://arxiv.org/abs/2508.12535" target="_blank"><img src="https://img.shields.io/badge/arXiv-2508.12535-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://seongland.com/article/corrsteer" target="_blank"><img src="https://img.shields.io/badge/Article-seongland.com-blue" alt="Article"/></a>
  <a href="https://huggingface.co/spaces/seonglae/CorrSteer" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-yellow" alt="HuggingFace"/></a>
  <a href="https://github.com/seonglae/CorrSteer" target="_blank"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github" alt="GitHub"/></a>
  <a href="https://corrsteer.vercel.app/" target="_blank"><img src="https://img.shields.io/badge/Slides-Slidev-87CEEB.svg" alt="Slides"/></a>
</div>

<div class="abs-br m-6 flex gap-3 items-center">
  <a href="https://arxiv.org/abs/2508.12535" target="_blank" title="arXiv" class="text-xl slidev-icon-btn opacity-60 !border-none">
    <carbon:document />
  </a>
  <a href="https://github.com/seonglae/CorrSteer" target="_blank" title="GitHub" class="text-xl slidev-icon-btn opacity-60 !border-none">
    <carbon:logo-github />
  </a>
  <a href="https://huggingface.co/spaces/seonglae/CorrSteer" target="_blank" title="HuggingFace" class="text-xl slidev-icon-btn opacity-60 !border-none">
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24"><path fill="currentColor" d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12s12-5.4 12-12S18.6 0 12 0m.7 4.7c.4 0 .7.3.7.7v1.8c0 .4-.3.7-.7.7h-1.4c-.4 0-.7-.3-.7-.7V5.4c0-.4.3-.7.7-.7zm-5.9 2c.2-.3.6-.4.9-.2l1.5.9c.3.2.4.6.2.9l-.9 1.5c-.2.3-.6.4-.9.2l-1.5-.9c-.3-.2-.4-.6-.2-.9zm10.4 0l.9 1.5c.2.3.1.7-.2.9l-1.5.9c-.3.2-.7.1-.9-.2l-.9-1.5c-.2-.3-.1-.7.2-.9l1.5-.9c.3-.2.7-.1.9.2M12 8.3c2 0 3.7 1.7 3.7 3.7s-1.7 3.7-3.7 3.7s-3.7-1.7-3.7-3.7s1.7-3.7 3.7-3.7m-7.4 2.9c.4 0 .7.3.7.7v1.8c0 .4-.3.7-.7.7H2.8c-.4 0-.7-.3-.7-.7v-1.8c0-.4.3-.7.7-.7zm14.8 0c.4 0 .7.3.7.7v1.8c0 .4-.3.7-.7.7h-1.8c-.4 0-.7-.3-.7-.7v-1.8c0-.4.3-.7.7-.7zM7.7 16l.9 1.5c.2.3.1.7-.2.9l-1.5.9c-.3.2-.7.1-.9-.2l-.9-1.5c-.2-.3-.1-.7.2-.9l1.5-.9c.3-.2.7-.1.9.2m8.6 0c.2-.3.6-.4.9-.2l1.5.9c.3.2.4.6.2.9l-.9 1.5c-.2.3-.6.4-.9.2l-1.5-.9c-.3-.2-.4-.6-.2-.9zm-4.9 1.7h1.4c.4 0 .7.3.7.7v1.8c0 .4-.3.7-.7.7h-1.4c-.4 0-.7-.3-.7-.7v-1.8c0-.4.3-.7.7-.7"/></svg>
  </a>
  <a href="https://seongland.com/article/corrsteer" target="_blank" title="Article" class="text-xl slidev-icon-btn opacity-60 !border-none">
    <carbon:link />
  </a>
</div>

<!--
Interactive embeds = the seongland article widgets, ported via components/HtmlEmbed.vue.
HTML fragments live in public/embeds/, data in public/data/.
No slide transitions, no click animations: everything renders on slide load.
-->

---

# The One-Line Claim

What CorrSteer actually does that is new

<div class="mt-6 p-4 bg-green-50 dark:bg-green-900 rounded">
<b>To our knowledge, the first automated method to steer ALL transformer layers simultaneously</b>, one correlation-selected SAE feature per layer, at generation time, <b>with a net task-performance gain</b> and <b>lower side effects than fine-tuning</b>.
</div>

<div class="grid grid-cols-2 gap-8 mt-8">
<div>

## What is genuinely novel
- **Per-layer** SAE feature selection
- **All layers** steered at once
- **Generation-time** activations, not context
- **Net accuracy up**, not a control-for-capability trade

</div>
<div>

## Stay honest (reviewers watch this)
- Prior representation engineering steers many layers too, but single-direction, context-token, contrastive, or with no net gain
- The **combination** is the contribution
- Say "first automated all-layer SAE generation-time steering with improvement", not "first ever"

</div>
</div>

---

# Why Steering Matters

Fine-tuning is a shotgun; we want switches

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

## Fine-tuning, a genome edit with a shotgun
- Hits the target, damages unrelated abilities
- A few adversarial examples break safety alignment
- Even benign data can re-surface blocked capabilities

</div>
<div>

## SAE features, interpretable switches
- Decompose activations into human-readable directions
- "refusal to harmful requests", "mathematical reasoning"
- Flip a handful instead of rewriting weights

</div>
</div>

<div class="mt-6 p-4 bg-blue-50 dark:bg-blue-900 rounded text-sm">
<b>Linear Representation Hypothesis</b>: networks encode concepts as directions. SAEs find a sparse basis where each direction is one concept, so steering is just adding a direction.
</div>

---
layout: full
---

# Steering in Action

<div class="text-sm opacity-60 mb-2">Same prompt, steered vs non-steered. Pick a task.</div>

<HtmlEmbed src="response-demo" data="response_examples.json" />

---

# The Gap in Prior SAE Steering

Three bottlenecks we remove

<div class="grid grid-cols-3 gap-6 mt-8">
<div>

## Data cost
Contrastive datasets or huge activation stores required

</div>
<div>

## Wrong tokens
Features picked from **context** (the prompt), not the **generated output**

</div>
<div>

## Wrong locus
Single layer or few layers, hand-tuned coefficients

</div>
</div>

<div class="mt-8 p-4 bg-yellow-50 dark:bg-yellow-900 rounded">
Missing the features that actually <b>drive output behavior</b>, because behavior lives in generation-time activations across the whole stack.
</div>

---
layout: full
---

# Method: Correlate, then Intervene

<div class="text-sm opacity-60 mb-1">Two stages: correlation selects, intervention validates. Hover the pipeline.</div>

<HtmlEmbed src="corrsteer-pipeline" frameless />

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

<div class="p-3 bg-blue-50 dark:bg-blue-900 rounded text-sm">
Streaming accumulator (Welford): <b>O(1) memory per feature</b>, any dataset size. No activation storage, no backward pass.
</div>

</div>
</div>

<div class="mt-6 p-4 bg-green-50 dark:bg-green-900 rounded">
<b>Key takeaway 1:</b> generation-time features reflect an LLM's actual capability better than context-token features. We adapted CAA, DSG, and SPARE to generation-time activations, and all three improved. The insight is portable, not method-specific.
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

<div class="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900 rounded text-sm">
<b>Positive-only is not a detail.</b> Subtracting negatively-correlated features <i>degrades</i> the model (JumpReLU and TopK activations are non-negative): amplify success, do not suppress failure.
</div>

---
layout: full
---

# Three Variants: S / A / P

<div class="text-sm opacity-60 mb-1">Toggle S (single global), A (one per layer), P (validation-pruned). Red = selected.</div>

<HtmlEmbed src="variant-selector" data="features_gemma_all.json,features_llama_all.json" />

<div class="mt-2 grid grid-cols-3 gap-4 text-sm">
<div class="p-2 bg-blue-50 dark:bg-blue-900 rounded"><b>S</b>: single most-correlated feature, globally</div>
<div class="p-2 bg-green-50 dark:bg-green-900 rounded"><b>A</b>: top feature from every layer</div>
<div class="p-2 bg-yellow-50 dark:bg-yellow-900 rounded"><b>P</b>: A plus pruning to a minimal subcircuit</div>
</div>

---
layout: section
---

# Pushing Static Steering to Its Limit

Every single layer, at the same time

---
layout: full
---

# One Feature Per Layer, All At Once

<div class="text-sm opacity-60 mb-1">Each point is an SAE feature. X = layer, Y = correlation, Z = coefficient. Rotate it.</div>

<HtmlEmbed src="feature-space-3d" data="features_gemma_all.json,features_llama_all.json" frameless />

<div class="mt-2 p-3 bg-yellow-50 dark:bg-yellow-900 rounded text-xs">
<b>Static steering taken to its maximum extent:</b> one fixed, correlation-selected direction in <i>every</i> layer. Task signal is distributed (later layers carry more, corr 0.140 to 0.336); A beats S on 5 of 8 tasks, so multi-layer combinations gain beyond single features. <i>(Speculative analogy: filling sparse per-layer activations raises an "always-on" task baseline, intuition only, not a neuroscience claim.)</i>
</div>

---
layout: full
---

# Results

<div class="text-sm opacity-60 mb-1">Accuracy across methods on Gemma-2 2B. Hover for std.</div>

<HtmlEmbed src="performance-dashboard" data="accuracy_results_full.json" />

<div class="mt-2 p-3 bg-green-50 dark:bg-green-900 rounded text-sm">
<b>+3.3% MMLU, +27.1% HarmBench.</b> Matches fine-tuning on MMLU (55.48 vs 55.75) while <b>halving the Side Effect Ratio (0.21 vs 0.41)</b>.
</div>

---
layout: full
---

# Side Effect Ratio: the Alignment Tax

<div class="text-sm opacity-60 mb-1">SER = fraction of changed answers that become wrong. Lower is more precise. First standardized metric for steering.</div>

<HtmlEmbed src="ser-comparison" data="ser_results.json" />

<div class="mt-2 p-3 bg-blue-50 dark:bg-blue-900 rounded text-sm">
On MMLU, CorrSteer-A changes <b>879</b> answers vs fine-tuning's <b>2,724</b>: same accuracy, far fewer side effects.
</div>

---

# Format, or Knowledge?

The experiment that won the rebuttal

<div class="mt-2">

Reviewers asked: are gains just fixing output format? We removed **all** structural features (semicolons, colons, XML; 11 of 25 layers) and steered with **semantic only** (medical, research, math; 14 layers).

| | Non-steered | Semantic-only | Full CorrSteer-A |
|---|---|---|---|
| MMLU | 52.21% | **55.12% &plusmn;0.06** | 55.48% &plusmn;0.59 |
| BBQ Ambig | 59.46% | **63.93% &plusmn;0.14** | 62.06% &plusmn;0.84 |

</div>

<div class="mt-4 p-4 bg-green-50 dark:bg-green-900 rounded text-sm">
<b>89% of the MMLU gain survives</b> with zero structural features (10x lower variance). On BBQ, semantic-only <b>exceeds</b> the full method, on bias reasoning where formatting cannot explain +4.47%. Knowledge, not formatting tricks.
</div>

---
layout: full
---

# Pooling and Positive-Only Ablations

<div class="text-sm opacity-60 mb-1">Left: max vs mean vs all-token pooling. Right: positive vs negative features. Toggle tabs.</div>

<HtmlEmbed src="ablation-pooling" data="ablation_data.json" />

---
layout: full
---

# Safety Is a Continuous Dial

<div class="text-sm opacity-60 mb-1">XSTest discrimination: safe prompts (low over-refusal) vs unsafe contrast (appropriate refusal).</div>

<HtmlEmbed src="safety-dashboard" data="safety_breakdown.json" />

---

# Safety: Pareto, Not Blanket Refusal

A tunable knob, sweeping coefficient scale

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

| Scale | HarmBench | XSTest over-refusal | MMLU |
|---|---|---|---|
| 0 | 46.4% | 2.37% | 52.21% |
| 0.5x | 54.64% | 9.47% | 52.31% |
| **1.0x** | **60.36%** | 21.89% | 52.00% |
| 1.5x | 60.36% | 36.69% | 51.37% |
| 2.0x | 7.50% | 6.51% | 49.89% |

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

# Interpretability: Discovered Features

Every steering decision traces to a human-readable SAE feature

<div class="text-sm mt-4">

**BBQ Disambiguous** (corr 0.559):
[L17/5137](https://neuronpedia.org/gemma-2-2b/17-gemmascope-res-16k/5137) Mathematical symbols · [L20/12748](https://neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/12748) Structured data · [L19/15745](https://neuronpedia.org/gemma-2-2b/19-gemmascope-res-16k/15745) Decision-making

**BBQ Ambiguous** (corr 0.554):
[L17/11021](https://neuronpedia.org/gemma-2-2b/17-gemmascope-res-16k/11021) Scientific findings · [L18/14447](https://neuronpedia.org/gemma-2-2b/18-gemmascope-res-16k/14447) Medical statistics · [L10/4557](https://neuronpedia.org/gemma-2-2b/10-gemmascope-res-16k/4557) Correctness checking

**HarmBench** (corr 0.779):
[L7/11722](https://neuronpedia.org/gemma-2-2b/7-gemmascope-res-16k/11722) Legal rejection · [L9/9298](https://neuronpedia.org/gemma-2-2b/9-gemmascope-res-16k/9298) Dismissive opinions · [L25/3912](https://neuronpedia.org/gemma-2-2b/25-gemmascope-res-16k/3912) Negative refusals

</div>

<div class="mt-4 p-3 bg-blue-50 dark:bg-blue-900 rounded text-sm">
Interpretability that fine-tuning cannot offer: feature descriptions from Neuronpedia, traceable per task.
</div>

---
layout: section
---

# The Twist Everyone Expects to Be Easy

Does dynamic steering just win?

---

# Dynamic Steering Did Not Resolve It

An unpublished negative result

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

## The common assumption
"Static steering is the weak version. Add input-dependent **gating**, fire each feature only at the right token, and performance obviously improves."

GSM8K seems to demand exactly this: reasoning features fire sparsely at pivotal steps; static amplification over-fires at narrative tokens.

</div>
<div>

## What we actually found
Adding a gating mechanism on top of CorrSteer **worsened** performance.

<div class="mt-4 p-4 bg-yellow-50 dark:bg-yellow-900 rounded text-sm">
This implies <b>linear steering itself carries a clear instability</b>. Gating does not rescue it, it amplifies the instability. <b>Dynamic linear steering may not be feasible</b> the way people assume.
</div>

</div>
</div>

<div class="mt-6 p-4 bg-blue-50 dark:bg-blue-900 rounded text-sm">
CorrSteer pushes <b>static</b> steering to its ceiling using the full depth of the transformer. The open question for the next paper: can dynamic steering beat it, or is the linear-direction paradigm itself the bottleneck?
</div>

---
layout: center
class: text-center
---

# What's Next: Control Reinforcement

Dynamic steering, learned rather than fixed

<div class="mt-6 text-lg opacity-80">
If a fixed per-token gate cannot beat static steering, learn the control policy with reinforcement.
</div>

<div class="mt-8">
  <a href="https://seongland.com/article/crl" target="_blank">
    <img src="https://img.shields.io/badge/Read-Control%20Reinforcement-6B5CE7?style=for-the-badge" alt="CRL"/>
  </a>
</div>

<div class="mt-4 text-sm opacity-70">
  <carbon:link class="inline"/> <a href="https://seongland.com/article/crl" target="_blank">seongland.com/article/crl</a>
</div>

---

# Key Takeaways

<div class="mt-6">

1. **Generation-time features better reflect an LLM's capabilities**, portable across CAA, DSG, and SPARE.

2. **Enhance LLMs by filling sparse activations with one feature per layer**: static steering, taken to the full depth of the transformer, with a net gain and half the side effects of fine-tuning.

3. **Dynamic steering is not the free win people expect**: gating on top of CorrSteer worsens performance, hinting that linear steering's instability is the real bottleneck.

</div>

<div class="mt-8 p-4 bg-blue-50 dark:bg-blue-900 rounded">
Our next direction is <b>Control Reinforcement</b> for dynamic steering. Follow along on X and LinkedIn.
</div>

---
layout: center
class: text-center
---

# Thank You

<div class="mt-2 text-lg opacity-75">
CorrSteer: steer every layer, stay interpretable, keep the side effects low
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
These slides: <carbon:logo-github class="inline"/> <a href="https://github.com/seonglae/corrsteer-slides" target="_blank">github.com/seonglae/corrsteer-slides</a> &nbsp;·&nbsp; live at <a href="https://corrsteer.vercel.app/" target="_blank">corrsteer.vercel.app</a>
</div>

<div class="abs-br m-6 text-sm opacity-50">
  Seonglae Cho · Zekun Wu · Adriano Koshiyama · ICML 2026
</div>
