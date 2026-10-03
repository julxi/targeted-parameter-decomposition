# SUMMARY: Overview of goal (tPD on Qwen for truthfulness)

**Last updated:** 26-10-03 (topic **closed**; batch/steps in the shared settings updated to the tuned values; training continues in `convos/julian/26-10-02_training_run_SUMMARY.md`)

**Status: closed** [decided: Julian, chat transcribed in the LOG]. Moved on:
- **To the training-run topic:** batch size and GPU use; untuned hyperparameters; running the untrained baseline (not on an H100); committing the code, which Julian defers until the training run has been discussed.
- **To a future analysis topic:** the probe script; which CI value to probe; the no-truth-signal control dataset.
- **Still open here:** the `eval_data_files` crash-mechanism verdict stays `[concluded]`.

## Goal and motivation

**Goal** [decided]: find out *to what extent* tPD (targeted parameter decomposition) on Qwen recovers mechanisms related to **truthfulness** (statement truth), and find evidence for them. Julian first wrote "honesty", then clarified that he meant truthfulness. Not one concrete decomposition. Keep the first experiment simple (bare statements, no chat template) and expand step by step [decided].
- **Long-term:** can tPD serve truthfulness detection, and how does it compare to other methods?

**Motivation:** someone else's unreplicated result, which Julian believes but whose setup may have had errors [assumed].
- Setup: tPD on Qwen's middle layers; target statements from domains of the "Universal Truthfulness Hyperplane" dataset; Pile as non-target; padding included in training.
- Reported finding: linear probes (logistic regression, mass-mean) on the **CI values**, probably mean-pooled over positions, detect statement truth.
- **No baseline was used** [Julian].

## First experiment

[decided: Julian's comments; training side implemented 26-10-02]

**Three training arms on Qwen2.5-7B-Instruct.** Configs: `spd/experiments/lm/honesty_targeted_decomposition/config_truth_{all_tokens,last_token,padded}.yaml`. They differ only in the target `loss_positions`:
- (A) `tokens`: all real tokens.
- (B) `last_k k=1`: the last token only.
- (C) `all`: padding included. ≈58% of trained positions are padding [verified: mean train length 10.03 Qwen tokens, `max_seq_len` 24]. Julian required ≥50%.

**Shared settings:**
- Target: all 20 tiu datasets (true + false, affirmative + negated): 6,560 train / 1,794 test statements, about 12 passes over the data in 5k steps × batch 16 [verified: counts]. tiu has no more data. Batch and steps were changed from 10k × batch 4 on 26-10-03, after tuning on arm A (see `convos/julian/26-10-02_training_run_SUMMARY.md`).
- Model, losses and layers (15–19 `down_proj`, C=96) are carried over from the legacy config and untuned.

**Code changes for it** [verified: unit test, full suite, CPU smoke test of arms B and C on a tiny Qwen2]:
- Positions outside `loss_positions` now run on the **original weights** in all masked forward passes (`route_only_selected_positions`, `spd/models/components.py`) [decided: Julian]. So the decomposition only has to reproduce the computation at the selected positions.
- **Untrained baseline:** `config_truth_untrained.yaml` (= the all-tokens arm with `steps: 0`). Purpose: do untrained CI functions already leak truth information? [Julian] It saves the initial decomposition (`model_0.pth`) and stops. That checkpoint is bit-identical to a trained run's starting weights and shared by all arms [verified: tiny-model smoke tests]. We initially saved step 0 in every run; Julian rejected that.
- **Non-target eval** reads the Pile's `val.jsonl.zst` (`eval_data_split: validation`, `eval_data_files`), disjoint from the training files [verified]. `eval_data_files` is honoured only by `lm_decomposition.py`, `targeted_ci_heatmap.py` and `spd/scripts/validation/common.py`. Other scripts that ignore it fail loudly on our configs (`Bad split: validation`) rather than silently reading the training files [verified]. Before, it re-read the first training documents (same split, same seed); that only affected a monitoring metric.

**Launch:** `python spd/experiments/lm/lm_decomposition.py <arm config>`. Checkpoints, including a 0-step run's `model_0.pth`, are uploaded to WandB (`sync_checkpoints_to_wandb: true`). On vast.ai: `spd-vast --mode provision --config h100`, then run on the instance. `spd-vast <name>` only takes registry experiment names, and the arms aren't registered. Not yet run at 7B. Julian expects iterations on GPU utilisation first.

**Analysis** (a separate script, parked until Julian confirms the runs are going [decided]):
- CI readouts: at the last token, and mean over real tokens. For arm C, also the mean including pads. Arm B: last token only, since its other CIs get no training signal.
- Probes: logistic regression and mass-mean, on the held-out test split.
- Baselines:
  1. CI from the step-0 (untrained) checkpoint;
  2. a probe on the raw activations;
  3. statement log-probability.
- Generalisation: across domains, and affirmative → negated.

**Points to keep in mind:**
- **Null hypothesis** [concluded]: the CI net is a learned nonlinear readout (5 × 18,944 MLP-hidden dims → 512 → 480 CIs) of mid-layer activations, from which truth is reportedly linearly decodable. So CIs may carry truth for mundane reasons. Hence the baselines.
- "The end-of-statement token carries the truth signal" is a working hypothesis of the setup [assumed].
- The last token is `.` for most statements, but `'.` for the 702 Spanish-translation statements [verified]. Token identity differs across domains.

**Follow-up after results** [Julian's TODO]: if arm A shows a truth signal, test with similar sentences that carry no truth signal (data design open; a random-label "control task" is one option).

**Known issues outside this topic** (tracked in `FUTURE_WORK.md`):
- 6 basedpyright errors in `spd/experiments/rotgrid/analysis/rank_rotgrid_decompositions.py`.
- `tests/test_rotgrid_decomposition.py::test_the_checked_in_sweep_expands_to_distinct_runs` fails (expects 4 sweep runs, gets 8).

## tPD in brief (reference; details in the LOG entry *tPD background and legacy inventory*, part A)

- **PD** splits each weight matrix into rank-1 subcomponents `U_c V_c^T`. A causal-importance (CI) network predicts which subcomponents matter on each input/position. Losses: reconstruction under stochastic and adversarial (PGD) masking of low-CI subcomponents (KL for LMs), plus importance-minimality (few subcomponents active per input).
- **tPD** (Vigouroux & Sharkey 2026, arXiv:2607.13047; paper in `papers/Targeted_Recovery_of_Weight_Space_Mechanisms/`) decomposes only what a chosen target dataset uses. A full-rank catch-all `Δ = W − Σ U_c V_c^T` per matrix absorbs everything else. On target batches Δ is ablated (random or adversarial), so the subcomponents alone must reproduce target outputs. On non-target batches (general data) Δ is fully on. Importance-minimality runs on both streams, which pushes non-target-only mechanisms into Δ and keeps subcomponents low-interference off-target. Without the non-target stream ("naive" PD), subcomponents have heavy off-target side effects.
- **Reported results:** consistency across seeds and nested targets on a 4-block Pile transformer; a CSS-only submodel at about 7% of full-PD FLOPs; erasing and rewiring `import numpy/pandas as` completions in a 12-block model. Largest model ≈ GPT-2-small size.
- **Caveats that matter for a truth target** [concluded from the paper's own appendix arguments]:
  - Subcomponents only cover the activation slice the target spans.
  - Mechanisms that co-activate in lockstep on the target get merged.
  - Redundant mechanisms can vanish into Δ.
  - "Truthfulness" enters only through the choice of target data.

  Consequence: to see components that differ by truth value, the target needs both true and false data, varied enough to split them.
- **In this code** [verified: code read 26-10-02, no run]:
  - `nontarget_task_config` switches targeted mode on; also relevant are `use_delta_component`, `nontarget_batch_size`, `nontarget_impmin_coeff_ratio`, and `LMTaskConfig.loss_positions` (`all`/`tokens`/`last_k`; default `all` includes padding).
  - Training loop: `spd/run_spd.py`. Non-target batches skip `UnmaskedReconLoss` and the persistent-PGD losses and use Δ=1.
  - Paper reproduction configs: `spd/experiments/lm/targeted_decomposition/`. Paper analysis tools: `spd/scripts/validation/`, `spd/scripts/completeness_tests/`.
- **How the paper actually trained numpy/pandas** [verified: published configs, code, GPT-NeoX tokenizer, 26-10-02]:
  - The target is just the two prompts `import numpy as` / `import pandas as`, each exactly 3 tokens, `max_seq_len: 3`, so no padding. The batch is the 2 prompts repeated; KL at all 3 positions.
  - The Pile is the *non-target* stream: 64-token packed chunks with Δ=1, stochastic-subset recon + importance-minimality ×2, no PGD and no unmasked loss.
  - Masks are drawn independently per (sequence, position, component), and all masks act in one forward pass. Perturbations therefore propagate to later layers and, via attention, to later positions. That is intended: PD requires robustness to any combination of ablations.
  - No position masks existed; none were needed, since the data had no padding. Our `loss_positions: tokens` restores that for variable-length statements. Since 26-10-02, unselected positions also run on the original weights; the paper has no counterpart to this.
  - The appendix places the CSS run's unmasked loss on non-target data; the code runs it on target batches with Δ off. [concluded: the appendix is likely wrong]
- **Code authorship:** targeted training (`cfcbd8a`, Antovigo) vs position masks / `loss_positions` and prepared datasets (`c774a4f`, Julian).
- **Lineage** [verified: git log, remotes, paper]: Antovigo/targeted-parameter-decomposition (paper code) → bbzmdn (Madina Babazhanova's fork; added all honesty experiments in commit `5b49341`, 26-09-01) → julxi (this fork).

## Legacy honesty folder (`spd/experiments/lm/honesty_targeted_decomposition/`)

**Deleted** on Julian's approval [decided]: everything except `build_tiu.py` and the tiu config, plus the README honesty section. The tiu config was since replaced by the three arm configs. The folder name stays [decided]. Per Julian, the deleted `prompts_liu` came from the repo of "On the Universal Truthfulness Hyperplane Inside LLMs". What was found before deletion:
- **(1) Data provenance:** `prompts_liu/` has no generator in the repo and no citation ("Liu et al." only). `contrast_pairs.jsonl` (source of the capitals prompts) has no recorded generation procedure.
- **(2) Padding:** the legacy configs train on all positions including right-padding (`prompts_file` with default `loss_positions: all`; one short sentence padded to 64/128 tokens). [concluded]
- **(3) Analysis scripts:** they rank components by CI difference at the answer token between true and false statements. That is confounded with the answer token's identity and surprisal. Together with the padding issue, `analysis_capitals/SUMMARY.md` has no evidential value. [concluded]

Salvaged from the legacy config comments [assumed: not reproduced]:
- Decomposing all layers of Qwen2.5-7B OOMs because of the dense Δ tensors, hence 5 `down_proj` matrices (layers 15–19, C=96).
- The ~152k-vocab logits are the per-step memory bottleneck; the legacy configs used batch 4 for that reason. Measured since: batch 32 peaks at ≤ 60 GB on an H100 80GB [verified: throughput trial, `convos/julian/26-10-02_training_run_SUMMARY.md`]; the arms now use batch 16.
- The non-target corpus must be raw text tokenised with Qwen's tokenizer (`monology/pile-uncopyrighted`), not the GPT-NeoX pre-tokenised Pile.
- Layer choice 15–19 is an unvalidated guess.

The arm configs inherit these settings.

## Other open design points (LOG part C)

- The Pile is only a proxy for Qwen-Instruct's training distribution; the chat template is open [Julian: later].
- Step and batch budget: 5k steps × batch 16 (tuned on arm A, 26-10-03), vs the paper's 30k × 128–256. Convergence may be uncheckable at 7B [Julian]. Options: watch eval-curve plateaus; iterate on Qwen2.5-0.5B/1.5B.
- True-only vs mixed target: now mixed. "A true-only target yields no truth-separating components" stays a hypothesis to test, not a fact [Julian: experiments decide].

- See also: [convos/julian/26-10-02_prepared_datasets_SUMMARY.md] — the tiu (Truth-is-Universal) datasets and `build_tiu.py` that the arm configs target.
- See also: [convos/julian/26-10-02_training_run_SUMMARY.md] — the 7B training runs of the arms designed here.
- See also: [convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md] — probe test accuracies of the finished decompositions; runs on the laptop CPU.
