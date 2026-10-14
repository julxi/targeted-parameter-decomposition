# SUMMARY: Bolder experiments — decompositions whose target loss sees the truth computation more directly

**Last updated:** 26-10-13 (run status checked after the sync: Exp 1 finished, Exp 0 crashed when the H100 went away, Exp 3 not started)

Julian wants to be less methodical and more exploratory: run fairly principled experiments, analyse them, and follow whatever is surprising. There are no success thresholds; he wants to see the numbers [decided: C: comment]. His premise: the model may compute a truth representation even where the output doesn't use it, while a KL loss on next-token outputs only sees output changes (point raised in `convos/julian/26-10-06_truth_writing_components_LOG.md`, *Q4 rephrased*). His two routes: (1) the target loss on inner activations; (2) an appended stimulus whose next-token output is the loss, with the decomposition applied only where truth is suspected to be computed.

## Design

[decided: C: comments in the LOG, unless tagged]
- **Data: tiu** (Truth-is-Universal, all 20 v1 datasets and their split, as in arm A, the all-tokens tiu decomposition `s-bd23f0d1`). Own generated data (option C) and UTH were considered; Julian chose tiu. Caveat kept in mind: in larger_than and animal_class the object word alone predicts the label at 0.70–0.75 [verified 26-10-10].
- **Common structure:** decompose only at the statement end (the period, `last_k 1`), in Exp 0–2, although R2 shows that only 26% of the verdict separation passes through MLPs 15–19 there; all other positions run on the original weights. Non-target stream unchanged (Pile, Δ on, KL). Model and decomposition as in arm A (Qwen2.5-7B-Instruct, `down_proj` of layers 15–19, C = 96, 5k steps, LR 5e-4) [assumed: carried over], except the **target batch: 64** instead of 16 (Pile batch stays 16), from the H100 trial below [decided: "plan approved", transcribed from chat, LOG entry *full runs launched*].
- **Runs, differing only in the target lens** (what the target loss compares):
  - **Exp 0, next-token lens** (the former "arm B", `config_truth_last_token.yaml`): the natural next-token KL at the period. Name proposed by the agent, accepted by Julian.
  - **Exp 1, residual lens** (`config_truth_residual_lens.yaml`): squared error of the residual stream after layer 23 at the period, each channel divided by its std over the training statements' periods, averaged over channels.
  - **Exp 2, stimulus lens** (`config_truth_stimulus_lens.yaml`): statement + one stimulus, full next-token KL at the last stimulus token. Mixing stimuli and a binary-verdict loss (variant 2b) were rejected. Stimulus: " Is this true or false? Answer:" (Julian's proposal without "?." and without the trailing space, chosen after the R1 screen).
  - **Exp 3, stimulus decomposed too** (`config_truth_stimulus_span.yaml`, label `tiu_stimulus_span`): as Exp 2 (same stimulus, same loss), but the decomposed positions are the period and all 8 stimulus tokens (9 per statement), so the verdict computation at the stimulus tokens also goes through the components. Requested by Julian in chat and queued with his approval [decided: transcribed from chat]; the exact position set is the agent's reading of the request [assumed].
- **No rescaling of loss coefficients:** Exp 1–3 keep Exp 0's coefficients.
- **Measurements before training are reference, not gates**; all runs happen regardless.
- **Analyses** for all runs: `truth_writers.py` with the untrained baseline `s-7fad0c14`; analysis V (Julian's request; verdict accuracy under component masks); component overlap between runs (not yet written; `spd/scripts/validation/compare_components.py` not yet checked on these checkpoints).
- Framing corrected by Julian: the runs share positions and inputs, but each loss can select entirely different mechanisms; whether they overlap is an open result, not a premise. The goal is not to "move the components-vs-Δ share" [concluded: agreed in the LOG].
- Deferred: decomposing at the object token instead of the period (Julian: "maybe later"; in FUTURE_WORK.md).

## Measured on CPU (frozen Qwen2.5-7B-Instruct, 546 tiu test statements, up to 30 per dataset) [verified: single runs 26-10-13; `~/spd_out/26-10-10_bolder_experiments/`]

**R1 stimulus screen** (`r1_stimulus_screen.{py,out}`):

| stimulus | truth-token mass | verdict accuracy | AUC |
|---|---|---|---|
| " This statement is" | 0.387 | 0.951 | 0.992 |
| " Is this true or false?" | 0.184 | 0.929 | 0.979 |
| " Is this true or false? Answer:" | 0.736 | 0.903 | 0.998 |
| same with a trailing space | 0.083 | 0.912 | 0.994 |
| Julian's string " Is this true or false?. Answer: " | 0.055 | 0.910 | 0.991 |

- A trailing space makes the digit "1" the most likely next token: Qwen writes " True" with its space as one token.
- The model's natural response after " Is this true or false?" is step-by-step reasoning in 22/40 samples and a direct True/False in 9/40. With " Answer:" it answers directly in 33/40; the other 7 are number comparisons.
- Accuracy is limited by the model: most errors are true negated statements called false (e.g. neg_element_symb_true 0.07–0.29 under every stimulus).

**R2/R3: zeroing MLPs 15–19** (`r2r3_zero_ablation.{py,out}`; = tPD's all-off state):
- Zeroed at the period: 86% of the layer-23 truth separation at the period disappears, but only 26% of the verdict separation after " … Answer:". Zeroed at all statement tokens: 97% and 60%. The verdict mostly reaches the stimulus by other routes, so the stimulus lens sees only a minority of it at the period [concluded].
- Loss scales in the all-off state at the period: next-token KL 1.68 nats, residual lens 2.26, stimulus lens 0.28 nats (factor 1.35 between the first two), hence no rescaling [decided].

**Earlier (26-10-10):** truth is 10–15% of the across-statement variance of the last-token residual (about 40% within a family, an upper bound), so a residual-level loss puts real weight on truth. A massive-activation channel (2570) holds 22–29% of the squared norm but 1–3% of the variance, hence the per-channel normalisation [verified: `dilution.{py,out}`].

## Implemented [verified: `tests/test_output_lens.py` incl. 2 mutation checks; CPU smoke training of the new configs on the tiny Qwen2; V checked on the real model (arm A, n = 100); `make test`/`make check` show only the known rotgrid failures; 26-10-13]

- **Config `target_lens`** (`spd/configs.py`): `{type: next_token}` (default, all earlier configs), `{type: residual, layer: L}`, `{type: stimulus, stimulus: "<text>", decompose_stimulus: bool}` (the flag, default false, is Exp 3). `config.target_output_loss_type` (mse for residual, else kl) applies to target losses, PPGD and target eval metrics; `output_loss_type` stays the nontarget loss.
- **Mechanism** (`spd/models/output_lens.py`): inside `ComponentModel.output_lens(lens)` a forward returns the lens view instead of logits: the output of `model.layers.L` / channel std, or the logits at each row's stimulus end repeated at every position. The tensor keeps its (batch, pos, …) shape, so existing losses select the decomposed positions unchanged and no second position mask was needed (the first design planned one); training and eval enter the context for target batches only. Channel std: `spd/models/residual_lens_stats.py`, computed at startup over one pass of the training data, saved as `residual_lens_channel_std.pt`.
- **Data** (`prompts_dataset.py`): with a stimulus, prompt batches look like `{"input_ids": (batch, pos), "position_mask": (batch, pos) bool, "stimulus_end": (batch,) int}`; `position_mask` covers the statement tokens only, or with `decompose_stimulus` also every stimulus token. The stimulus is tokenized separately; trimming keeps it.
- **Analysis V** (`verdict_accuracy.py`): verdict accuracy/AUC/separation/KL at the stimulus end with replacement at the period only, for the states original, components only (rounded CI, Δ off), Δ only, all off, full (components + Δ) and full minus each component active on ≥ 1% of statements. Single ablations are compared with "full" because bf16 rounding alone moves verdict scores by up to 0.25 on CPU. `--decompose_stimulus` replaces at the period and all stimulus tokens (for Exp 3; output `verdict_<run>_span.json`). The n = 100 CPU check's all-off state keeps 76% of the separation, R2 found 74% with independent code.
- The tiny test model's outputs are nearly uniform (logit std 0.01), so the Exp 2 smoke run shows only that the code runs; the unit test scales its final norm to get meaningful KLs (pitfall recorded in FEEDBACK.md).

## H100 tuning trials [verified: `~/spd_out/26-10-10_bolder_experiments/tuning/`, one seed each, 26-10-13]

- Smoke runs (300 steps) of Exp 1 and Exp 2 on the real model ran without errors. Exp 1's startup channel std (min/median/max 0.825 / 1.581 / 17.51) agrees with the CPU computation from stored residuals (0.824 / 1.581 / 17.46). Exp 2's eval PGD loss was very high (9.7 nats at step 150); the full run brought it to 0.60.
- Exp 0, 2,000 steps, target batch 16 vs 64: batch 64 better on every metric (active components 13.7 → 8.9, target rounded KL 0.173 → 0.135, eval PGD 2.04 → 1.00, Pile 0.054 → 0.041) at the same wall time (19.5 vs 19.8 min) [concluded: same direction at every eval; Pile difference at the edge of run-to-run noise]. Batch 128 untested.

## Exp 2 result (stimulus lens) [verified: run `s-05db01ad` (47 min on the H100) and V, one seed, 26-10-13]

- Training, step 5,000: 0.47 active components per statement; target rounded KL 0.029 nats at the stimulus end, Δ only 0.014, PGD 0.60, Pile 0.040.
- V on the full test split (n = 1,794; `~/spd_out/26-10-10_bolder_experiments/full/v/`):

| decomposition | active per statement | components-only KL | separation | Δ-only KL | singles ≥ 1% | largest single \|Δsep\| |
|---|---|---|---|---|---|---|
| Exp 2 `s-05db01ad` | 0.8 | 0.033 | 10.88 | 0.030 | 13 | 0.38 |
| arm A `s-bd23f0d1` | 9.3 | 0.092 | 9.84 | 0.044 | 55 | 0.33 |
| untrained `s-7fad0c14` | 220 | 0.115 | 10.31 | 0.008 | 475 | 0.07 |
| all off (same for every run) | 0 | 0.116 | 10.21 | – | – | – |

  Original: separation 12.04, accuracy 0.953. On the full split, zeroing MLPs 15–19 at the period removes 15% of the verdict separation (R2's capped sample: 26%).
- Reading [concluded, one seed]: Exp 2's components (under one per statement) recover ~70% of the verdict KL lost by switching the five layers off, arm A's ~20%, the untrained ones nothing. The recovery concentrates in negated statements: on negated Spanish translations (64 test statements) accuracy is original 0.86, all off 0.06, Exp 2 components 0.95, arm A 0.50, untrained 0.17; negated element symbols 0.55 / 0.36 / 0.59 / 0.50 / 0.23; affirmative families 0.96–1.00 in every state.
- Report: `reports/26-10-13_exp2_stimulus_lens.typ` (4 pages, self-contained; PDF gitignored, build with `typst compile`).

## Open and next

1. Runs [verified: WandB and vast.ai, 26-10-13 ~20:30 UTC, LOG entry *run status check: H100 gone, Exp 0 crashed*]: Exp 1 `s-cd129592` finished (5,000 steps, `model_5000.pth` on WandB; metrics not copied locally). Exp 0 `s-773bfec6` crashed at step 1,600 when the H100 rental (instance 55752929) ended, no checkpoint, cause not recorded [assumed: instance loss]. Exp 3 never started. Next: rerun Exp 0 and Exp 3 on a new rental.
2. Analyses: V on Exp 0, Exp 1 and Exp 3 (Exp 3 with `--decompose_stimulus`); `truth_writers.py` on Exp 0–3; component overlap between runs (to write).
3. Open: is Exp 2's recovery a negation mechanism rather than a truth mechanism? Exp 0, at the same position, is the like-for-like reference.

## See also

- See also: [convos/julian/26-10-06_truth_writing_components_SUMMARY.md] — source of the KL-blindness argument, the Q4 loss options this topic implements, gate G (on tiu the next token already depends on truth), the analysis script `truth_writers.py` and its stored features (R3 uses their layer-23 residuals), and the single-component ablation numbers R2 is compared with.
- See also: [convos/julian/26-10-13_direction_free_interventions_SUMMARY.md] — proposes extending analysis V (ablation at all statement tokens, random-component controls, a non-truth control question, interchange patching) as a direction-free test of the A7 lead; the same code would compare Exp 0–3.
- See also: [convos/julian/26-10-06_state_report_SUMMARY.md] — the Typst state report of all earlier truth experiments; its `reports/template.typ` is the layout of this topic's Exp 2 report.
- See also: [convos/julian/26-10-05_no_truth_baseline_SUMMARY.md] — origin of arm A `s-bd23f0d1` and the untrained decomposition `s-7fad0c14`, the two reference decompositions analysis V is run on here.
- See also: [convos/julian/26-10-02_training_run_SUMMARY.md] — arm B (= Exp 0 here) was never run there and was flagged as needing its own tuning; the batch-16 setting came from there, and this topic's H100 trial moved the period-only runs to batch 64.
- See also: [convos/julian/26-10-05_uth_experiments_SUMMARY.md] — the UTH series; Julian's argument against UTH data for mechanism finding (diverse tasks help find a transferring direction, not the mechanism) is recorded in this topic's LOG.
