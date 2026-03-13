Yes — here is a faithful text version of the workflow, rewritten as an **RL-stage algorithm**, not as standard diffusion/DiT supervised training.

One small caveat: the **top-right formula coefficients are a bit hard to read from the image**, so I’ll preserve the structure exactly, but write the reward-weighted old-policy regularized loss in a slightly abstracted form.

---

## 1. What the figure is doing, in plain words

The model does **online candidate generation + reward scoring + reward-weighted policy update**.

Instead of training directly against a ground-truth edited image with a plain denoising loss, it does this:

1. Build a condition from:

   * input image
   * glyph image
   * instruction

2. Use the **old policy** (\pi^{old}) to generate **multiple candidate edited images**
   [
   x_0^{1:K}
   ]

3. For each generated image, run a forward noising step to obtain noisy states
   [
   x_t^{1:K}
   ]

4. Evaluate each candidate with a **reward model** made of four dimensions:

   * instruction adherence
   * text clarity
   * background preservation
   * relative quality

5. Convert the reward model’s discrete-score logits into a continuous score, then normalize it to
   [
   r^{1:K}\in[0,1]
   ]

6. Update the current denoiser/vector-field model by:

   * **pulling toward** high-reward samples
   * **downweighting or pushing away from** low-reward samples
   * while staying regularized against the **old model** (v^{old})

So this is much closer to **policy optimization over generated edits** than normal paired-image diffusion training.

---

## 2. Symbols

Let the condition be

[
c = (\text{input image},\ \text{glyph image},\ \text{instruction})
]

Let the old rollout policy generate (K) candidates:

[
x_0^{1:K} \sim \pi^{old}(\cdot \mid c)
]

For a sampled noise level (t) and Gaussian noise (v),

[
x_t^k \sim q(x_t \mid x_0^k)
]

or, equivalently in diffusion notation,

[
x_t^k = \alpha_t x_0^k + \sigma_t v,\qquad v\sim\mathcal N(0,I)
]

The current model predicts a denoising target / velocity / noise term:

[
v_\theta(x_t^k \mid c)
]

The frozen reference model predicts:

[
v^{old}(x_t^k \mid c)
]

---

## 3. Reward model written as equations

Each generated candidate (x_0^k) is scored by four reward heads.

### 3.1 Dimension-wise rewards

[
R^{\text{adh}}_k,\quad
R^{\text{clar}}_k,\quad
R^{\text{pres}}_k,\quad
R^{\text{qual}}_k
]

corresponding to:

* instruction adherence
* text clarity
* background preservation
* relative quality

The figure suggests the total task reward is an additive merge:

[
R^{\text{task}}_k
=================

R^{\text{adh}}_k
+
R^{\text{clar}}_k
+
R^{\text{pres}}_k
+
R^{\text{qual}}_k
]

You could also write this more generally as

[
R^{\text{task}}_k
=================

\sum_{m\in{\text{adh,clar,pres,qual}}}
\lambda_m R_k^{m}
]

if unequal weights are used.

---

### 3.2 VLM-based discrete-to-continuous scoring

Each reward head uses a VLM that outputs logits over discrete scores (0,\dots,9).

If the logits for one dimension are (z_{k,m}\in\mathbb R^{10}), then:

[
p_{k,m}(s)=\mathrm{softmax}(z_{k,m})_s,\qquad s\in{0,\dots,9}
]

Convert this into an expected continuous score:

[
E_{k,m}=\sum_{s=0}^{9} s\cdot p_{k,m}(s)
]

Then normalize:

[
R_{k,m}=\mathrm{Norm}(E_{k,m})
]

and after aggregation:

[
r_k=\mathrm{Norm}(R_k^{\text{task}})\in[0,1]
]

This matches the bottom-right part of the figure:
**discrete score logits (\rightarrow) continuous score (E) (\rightarrow) normalization (\rightarrow) (r^{1:K}\in[0,1]).**

---

## 4. Denoising-side optimization signal

Define the current model denoising error on sample (k):

[
\ell_\theta^k
=============

\left|v_\theta(x_t^k\mid c)-v\right|_2^2
]

Define the old/reference model error:

[
\ell_{old}^k
============

\left|v^{old}(x_t^k\mid c)-v\right|_2^2
]

The top-right part of the figure is showing a **reward-weighted combination** of:

* a **positive term** for good samples
* a **negative term / counter-term** for bad samples
* both measured relative to the denoising target
* with a (\beta)-controlled balance against the old model

A clean abstract transcription is:

[
\mathcal L_{\text{RL}}
======================

\sum_{k=1}^K
\Big[
r_k\cdot \phi_{+}(\ell_\theta^k,\ell_{old}^k;\beta)
+
(1-r_k)\cdot \phi_{-}(\ell_\theta^k,\ell_{old}^k;\beta)
\Big]
]

where:

* (\phi_+) should make the current model better than the old one on **high-reward** samples
* (\phi_-) suppresses or penalizes **low-reward** samples
* (\beta) controls the strength of the old-policy regularization / trust-region-like effect

A very common equivalent way to express the same structure is through an improvement margin:

[
\Delta_k
========

\ell_\theta^k-\ell_{old}^k
]

and then optimize so that:

* for large (r_k), make (\Delta_k < 0)
* for small (r_k), avoid reinforcing that sample

So the RL signal is not “match the provided GT image,” but rather:

> “Among the model’s own sampled edits, increase likelihood of those that the reward model judges good.”

---

## 5. Pseudocode

```python
# Inputs:
#   x_in   : input image
#   g      : glyph image
#   instr  : edit instruction
#   pi_old : frozen rollout policy / old model
#   v_theta: current trainable denoiser / vector field
#   v_old  : frozen reference denoiser / vector field
#   RM     : reward model with 4 sub-reward dimensions
#   K      : number of sampled candidates

def rl_stage_step(x_in, g, instr):
    c = (x_in, g, instr)

    # ----------------------------------------------------
    # 1) Rollout: generate multiple candidates with old policy
    # ----------------------------------------------------
    x0_list = []
    for k in range(K):
        x0_k = sample_image_from_policy(pi_old, condition=c)   # x_0^k
        x0_list.append(x0_k)

    # ----------------------------------------------------
    # 2) Score each candidate with the reward model
    # ----------------------------------------------------
    raw_rewards = []
    for x0_k in x0_list:
        # reward model consumes:
        #   input image, generated image, instruction,
        #   task-specific eval instructions,
        #   and GT image only as quality reference
        reward_dict = RM(
            input_image=x_in,
            generated_image=x0_k,
            instruction=instr,
            task="rearrange",
            eval_instructions="task-specific evaluation instructions"
        )

        # example keys:
        #   adherence, clarity, preservation, quality
        R_task = (
            reward_dict["instruction_adherence"]
            + reward_dict["text_clarity"]
            + reward_dict["background_preservation"]
            + reward_dict["relative_quality"]
        )

        raw_rewards.append(R_task)

    r = normalize_to_unit_interval(raw_rewards)   # r^1:K in [0, 1]

    # ----------------------------------------------------
    # 3) Diffusion forward process for each sampled image
    # ----------------------------------------------------
    losses = []
    for k, x0_k in enumerate(x0_list):
        t = sample_noise_timestep()
        v = sample_gaussian_noise()

        x_t = forward_diffusion(x0_k, t, v)

        # current model error
        pred_cur = v_theta(x_t, condition=c)
        ell_cur = l2_squared(pred_cur, v)

        # old/reference model error
        pred_old = v_old(x_t, condition=c)
        ell_old = l2_squared(pred_old, v)

        # abstract reward-weighted old-policy-regularized objective
        loss_k = (
            r[k]     * positive_term(ell_cur, ell_old, beta)
            + (1-r[k]) * negative_term(ell_cur, ell_old, beta)
        )
        losses.append(loss_k)

    loss = mean(losses)

    # ----------------------------------------------------
    # 4) Optimize current model
    # ----------------------------------------------------
    optimize_theta(loss)

    return loss
```

---

## 6. A more explicit “math-only” rewrite

If I compress the entire figure into a compact training objective, it becomes:

### Sampling

[
x_0^{1:K}\sim \pi^{old}(\cdot\mid c)
]

### Forward process

[
x_t^k = \alpha_t x_0^k + \sigma_t v,\qquad v\sim\mathcal N(0,I)
]

### Reward

[
R_k^{\text{task}}
=================

\sum_{m} R_k^m,
\qquad
r_k=\mathrm{Norm}(R_k^{\text{task}})
]

### Current vs old denoising errors

[
\ell_\theta^k=|v_\theta(x_t^k\mid c)-v|*2^2,\qquad
\ell*{old}^k=|v^{old}(x_t^k\mid c)-v|_2^2
]

### RL-style update

[
\min_\theta;
\sum_{k=1}^K
\Big[
r_k\cdot \phi_+(\ell_\theta^k,\ell_{old}^k;\beta)
+
(1-r_k)\cdot \phi_-(\ell_\theta^k,\ell_{old}^k;\beta)
\Big]
]

That is the cleanest textual rendering of the figure.

---

## 7. Why this is **not** standard DiT / standard diffusion training

Standard supervised diffusion training would look like:

[
\min_\theta
\mathbb E_{(x_{src},x_{gt},c),t,v}
\left[
|v_\theta(x_t\mid c)-v|_2^2
\right]
]

where (x_{gt}) is the fixed target edited image.

This figure is different in several important ways:

### A. The model trains on its **own sampled outputs**

It generates (K) candidates from (\pi^{old}), then learns from those.

### B. Learning signal comes from a **reward model**

The key target is not “reproduce this GT edit exactly,” but “increase preference for highly rewarded edits.”

### C. The reward is **multi-dimensional**

Not just one scalar aesthetic score, but multiple axes:

* adherence
* clarity
* preservation
* quality

### D. There is an **old-policy / reference-model anchor**

The update is regularized against (v^{old}), which is much more RL-like than pure supervised diffusion regression.

### E. The GT image is not the training target in the usual sense

From the figure, the GT image appears only in the **relative quality** reward branch as a **reference quality anchor**, not as the direct denoising target for every update.

That last point is especially important.

---

## 8. A one-paragraph interpretation

A good way to summarize the whole figure is:

> Freeze an old editor, sample multiple edited images from it, evaluate them with a VLM-based multi-objective reward model, normalize those rewards into per-sample weights, then train the current diffusion policy so that its denoising behavior becomes more consistent with high-reward samples and less reinforced by low-reward ones, while staying close to the old policy through a (\beta)-controlled regularization term.

---

If you want, I can also rewrite this into one of these forms:

1. a **paper-style Method subsection**,
2. a **PyTorch-like training loop**, or
3. a **diagram-to-equation mapping table**.
