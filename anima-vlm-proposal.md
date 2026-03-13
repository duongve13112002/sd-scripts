# Anima VLM Proposal

## Summary

This proposal describes a practical way to use a VLM judge for Anima postfix training.
The main recommendation is:

- do not start with full-model RL
- do not start with a pure aesthetic scalar
- start with postfix-only optimization using a VLM error critic

The most reliable first task for the judge is:

`tags + generated image -> what is wrong, missing, or unwanted in the image?`

This is better than asking a small VLM for a universal "beauty" score.

## Why Postfix

This repository already has a good small control surface for conditioning-side optimization:

- `networks/postfix_anima.py` supports `hidden`, `embedding`, `cfg`, and `dual` postfix modes
- `anima_train_network.py` already contains special handling for CFG postfix
- `search_discrete_postfix.py` already provides a strong non-RL baseline for postfix search

That makes postfix the safest target for VLM-guided optimization:

- few trainable parameters
- lower risk of damaging the base model
- easier to attribute reward changes to the tuned component
- easier to revert or compare against baselines

## Main Position

The VLM should behave like an error critic, not a beauty critic.

Good questions:

- Are required visual tags present?
- Are forbidden or incorrect traits present?
- Are there visible defects such as bad hands, broken eyes, extra limbs, artifacts, or unreadable structure?
- Between two images from the same prompt family, which one has fewer errors?

Weak questions:

- How aesthetic is this image from 0 to 10?
- Is this image masterpiece quality?
- Is this image beautiful?

Small judges can often do acceptable narrow visual error detection. They are much less reliable as general-purpose aesthetic scorers.

## Reward Design

### 1. Use tag preprocessing

Do not feed the raw training caption directly into the judge and ask it to score everything.

Split tags into classes:

- core subject tags: `1girl`, `2girls`, `boy`, `solo`
- visual attribute tags: hair color, eye color, accessories, clothing items
- scene tags: indoors, classroom, beach, night
- pose or framing tags: full body, upper body, looking at viewer
- forbidden tags: traits that must not appear
- ignore tags: artist names, `masterpiece`, `best quality`, vague style tags

Only judge tags that are visually verifiable.

### 2. Judge output schema

Recommended structured output:

```json
{
  "present_correct": ["1girl", "blue eyes", "smile"],
  "missing": ["long hair"],
  "wrong": ["hat", "2girls"],
  "defects": ["bad hands", "asymmetric eyes"],
  "confidence": 0.81,
  "notes": "Main subject is correct but anatomy is unstable."
}
```

This gives the training loop something more stable than a raw prose answer.

### 3. Convert critic output into reward

A simple reward can be:

```text
reward =
  + 1.0 * num_present_correct_core
  + 0.5 * num_present_correct_secondary
  - 2.0 * num_missing_core
  - 1.0 * num_missing_secondary
  - 2.0 * num_wrong_core
  - 1.0 * num_wrong_secondary
  - 1.5 * num_defects
```

Then clamp or normalize the result per prompt group.

The important part is relative consistency, not the absolute value.

### 4. Prefer pairwise ranking over scalar scoring

For the same prompt or same prompt-plus-seed family:

- generate image A and image B
- ask the VLM which image has fewer errors
- optionally ask why

This is usually more reliable than asking for absolute image quality.

Recommended pairwise question:

> Given these requested tags and two generated anime images, which image better matches the tags and contains fewer visible defects? Answer with A or B and a short structured reason.

## What This Can Train

This approach is best suited for:

- postfix vectors in `hidden` or `dual` mode
- postfix `cfg` mode if the judge is used to shape positive vs negative postfix behavior
- discrete postfix token search reranking

This approach is not the best first step for:

- full DiT RL
- large LoRA modules spread across the whole model
- reward functions dominated by vague taste or style language

## Recommended Training Stages

### Stage 0: Baseline without VLM

Use the existing postfix paths first:

- continuous postfix tuning in `networks/postfix_anima.py`
- discrete postfix search in `search_discrete_postfix.py`

This establishes whether postfix itself is useful on the target dataset.

### Stage 1: Offline VLM reranking

This is the recommended first VLM stage.

Loop:

1. Freeze the base model.
2. Generate multiple candidates per prompt using several postfix variants.
3. Preprocess tags into judgeable targets.
4. Use the VLM to rank or criticize outputs.
5. Keep the best postfix candidates or best generated outputs.

This stage does not require changing the training loop yet.

Good first use:

- rerank candidate postfix vectors from `search_discrete_postfix.py`
- compare continuous postfix checkpoints at fixed prompt grids

If the VLM cannot reliably choose better candidates here, it is not good enough for RL.

### Stage 2: Reward-weighted postfix training

Once offline reranking is stable, move to reward-weighted training.

Practical version:

- generate `K` candidates per prompt with the current or reference postfix
- score them with the VLM critic
- convert critic output to normalized per-sample weights
- apply those weights to the standard denoising loss

This is much easier to stabilize than a full policy-optimization objective.

In this repository, the cleanest insertion points are:

- sample-level loss weighting in `train_network.py`
- `post_process_loss` override in `anima_train_network.py`

This means an initial implementation can remain "denoising training plus reward weights" instead of fully replacing the trainer with a separate RL loop.

### Stage 3: Full online VLM-RL

This corresponds more closely to the workflow described in `VLM-RL.md`:

- old policy generates multiple samples
- VLM reward scores each sample
- update is regularized against the old model

This is the most expensive and least stable stage.
It should only be attempted after Stage 1 and Stage 2 show clear gains.

## Why Not Pure Aesthetic Reward

The main concern is reward hacking.

If the judge is weak, the model may learn shortcuts such as:

- oversaturation
- too much contrast or sharpness
- exaggerated face rendering
- visually loud but semantically wrong images
- repeated artifact patterns that correlate with the judge's bias

An error critic is harder to game than a raw beauty score because it is tied to concrete failures.

## Where a Small Judge Can Still Work

A small VLM may still be useful if its task is narrow:

- count subjects
- detect missing visual attributes
- flag common anatomy defects
- compare two images from the same prompt

It is much less trustworthy for:

- nuanced composition judgment
- subtle taste or style ranking
- fine-grained anime aesthetic preference across different prompts

So if a small judge is used, it should mostly answer:

`what is wrong here?`

and only secondarily:

`which of these two is better?`

## Proposed MVP

### Goal

Improve postfix quality using a VLM critic without rewriting the entire trainer.

### MVP plan

1. Build a tag preprocessor:
   - split captions into judgeable tags, forbidden tags, and ignore tags
2. Build a VLM critic wrapper:
   - input: judgeable tags plus generated image
   - output: structured JSON with missing, wrong, and defect fields
3. Build an offline evaluator:
   - fixed prompt grid
   - fixed seeds
   - compare baseline postfix, continuous postfix, and discrete postfix results
4. Add pairwise ranking mode:
   - choose better image within the same prompt family
5. Only if that is stable, add reward-weighted postfix training

### Success criteria

- lower defect count on the evaluation grid
- better tag faithfulness on core tags
- better pairwise win rate against baseline postfix
- no obvious collapse toward loud but semantically wrong outputs

## Integration Ideas

### Option A: VLM reranking for discrete postfix search

This is the safest first experiment.

Current search already:

- creates placeholder postfix vectors
- scores candidates using denoising loss
- performs greedy token search

A VLM can be added as a reranking signal for:

- final candidate selection
- tie-breaking among low-loss candidates
- pairwise comparison of selected postfix token sets

This keeps the existing search pipeline mostly intact.

### Option B: Reward-weighted postfix checkpoint selection

Train postfix normally.
At evaluation time:

- generate a fixed prompt grid
- score outputs with the VLM critic
- keep the best checkpoint by critic score plus human spot checks

This is cheap and likely useful even if RL never lands.

### Option C: Reward-weighted training

Add a per-sample reward term that scales denoising loss.

Conservative form:

```text
loss = base_denoising_loss * reward_weight
```

where `reward_weight` is normalized within a prompt group and clipped to a narrow range.

Do not begin with strong negative pushing on low-score samples.
That tends to destabilize training when the judge is noisy.

## Practical Risks

### 1. Tag ambiguity

Many anime tags are weakly visible or depend on style conventions.
Raw captions contain many tokens that the judge cannot verify.

Mitigation:

- only judge visually grounded tags
- maintain an ignore list

### 2. Reward noise

The judge will make mistakes.
If training pressure is too strong, the postfix will learn the judge's bugs.

Mitigation:

- pairwise ranking first
- reward clipping
- confidence thresholding
- mixed evaluation with human spot checks

### 3. Compute cost

Generating `K` candidates and calling a VLM for each one is expensive.

Mitigation:

- start offline
- cache prompt grids and generated samples
- use VLM only on selected batches or checkpoints

### 4. Misalignment with postfix capacity

Postfix can bias conditioning, but it cannot fix everything.
If the reward is dominated by things outside postfix control, learning will be noisy.

Mitigation:

- focus reward on semantic faithfulness and common visible defects
- avoid overly broad aesthetic objectives at the beginning

## Recommendation

Recommended order:

1. postfix baseline
2. offline VLM error critic
3. pairwise reranking
4. reward-weighted postfix training
5. only then consider full VLM-RL

The core idea is sound, but the first version should be narrow:

`tags + image -> what is wrong?`

not:

`image -> how beautiful is this?`

## Open Questions

- Which tag classes should be considered judgeable for the target dataset?
- Should the first judge operate on single images, or only pairwise comparisons?
- Is the immediate target continuous postfix, discrete postfix search, or CFG postfix?
- How much human evaluation bandwidth is available to calibrate the critic?

## Final Recommendation

For this repository, the strongest first experiment is:

- postfix-only
- offline
- pairwise or structured error criticism
- VLM used as reranker before it is used as trainer

If that works, then a reward-weighted postfix trainer is justified.
If that does not work, jumping directly to full VLM-RL will mostly amplify noise.
