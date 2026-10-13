# SUMMARY: Bolder experiments — decompositions whose target loss sees the truth computation more directly

**Last updated:** 26-10-13 (data = tiu; R1–R3 measured on CPU; lenses, configs and analysis V implemented and tested; stimulus, naming, coefficients and position decided; being committed; nothing trained)

Julian wants to be less methodical and more exploratory: run fairly principled experiments, analyse them, and follow whatever is surprising. There are no success thresholds; he wants to see the numbers [decided: C: comment]. His premise: the model may compute a truth representation even where the output doesn't use it, while a KL loss on next-token outputs only sees output changes (point raised in `convos/julian/26-10-06_truth_writing_components_LOG.md`, *Q4 rephrased*). His two routes: (1) the target loss on inner activations; (2) an appended stimulus whose next-token output is the loss, with the decomposition applied only where truth is suspected to be computed.

## Design

[decided: C: comments in the LOG, unless tagged]
- **Data: tiu** (Truth-is-Universal, all 20 v1 datasets and their split, as in arm A). Caveat kept in mind: in larger_than and animal_class the object word alone predicts the label at 0.70–0.75 [verified 26-10-10].
- **Common structure:** decompose only at the statement end (the period, `last_k 1`), in all three runs, although R2 shows that only 26% of the verdict separation passes through MLPs 15–19 there; all other positions run on the original weights. Non-target stream unchanged (Pile, Δ on, KL). Model and decomposition as in arm A (Qwen2.5-7B-Instruct, `down_proj` of layers 15–19, C = 96, batch 16, 5k steps, LR 5e-4) [assumed: carried over].
- **Three runs, differing only in the target lens** (what the target loss compares at the period):
  - **Exp 0, next-token lens** (the former "arm B", `config_truth_last_token.yaml`): the natural next-token KL. Name proposed by the agent, accepted by Julian.
  - **Exp 1, residual lens** (`config_truth_residual_lens.yaml`): squared error of the residual stream after layer 23, each channel divided by its std over the training statements' periods, averaged over channels.
  - **Exp 2, stimulus lens** (`config_truth_stimulus_lens.yaml`): statement + one stimulus, full next-token KL at the last stimulus token. Mixing stimuli and a binary-verdict loss were rejected. Stimulus: " Is this true or false? Answer:" (Julian's proposal without "?." and without the trailing space, chosen after the R1 screen).
- **Measurements before training are reference, not gates**; all runs happen regardless.
- **Analyses** for all three runs: `truth_writers.py` with the untrained baseline `s-7fad0c14`; analysis V (Julian's request; verdict accuracy under component masks); component overlap between runs (not yet written).
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
- Loss scales in the all-off state at the period: next-token KL 1.68 nats, residual lens 2.26, stimulus lens 0.28 nats. Exp 1 and Exp 2 keep Exp 0's coefficients, no rescaling [decided].

**Earlier (26-10-10):** truth is 10–15% of the across-statement variance of the last-token residual (about 40% within a family, an upper bound), so a residual-level loss puts real weight on truth. A massive-activation channel (2570) holds 22–29% of the squared norm but 1–3% of the variance, hence the per-channel normalisation [verified: `dilution.{py,out}`].

## Implemented [verified: `tests/test_output_lens.py` incl. 2 mutation checks; CPU smoke training of both new configs on the tiny Qwen2; V checked on the real model (arm A, n = 100); `make test`/`make check` show only the known rotgrid failures; 26-10-13]

- **Config `target_lens`** (`spd/configs.py`): `{type: next_token}` (default, all earlier configs), `{type: residual, layer: L}`, `{type: stimulus, stimulus: "<text>"}`. `config.target_output_loss_type` (mse for residual, else kl) applies to target losses, PPGD and target eval metrics; `output_loss_type` stays the nontarget loss.
- **Mechanism** (`spd/models/output_lens.py`): inside `ComponentModel.output_lens(lens)` a forward returns the lens view instead of logits: the output of `model.layers.L` / channel std, or the logits at each row's stimulus end repeated at every position. The tensor keeps its (batch, pos, …) shape, so existing losses select the decomposed positions unchanged; training and eval enter the context for target batches only. Channel std: `spd/models/residual_lens_stats.py`, computed at startup over one pass of the training data, saved as `residual_lens_channel_std.pt`.
- **Data** (`prompts_dataset.py`): with a stimulus, prompt batches look like `{"input_ids": (batch, pos), "position_mask": (batch, pos) bool, statement tokens only, "stimulus_end": (batch,) int}`. The stimulus is tokenized separately; trimming keeps it.
- **Analysis V** (`verdict_accuracy.py`): verdict accuracy/AUC/separation/KL at the stimulus end with replacement at the period only, for the states original, components only (rounded CI, Δ off), Δ only, all off, full (components + Δ) and full minus each frequently active component. Single ablations are compared with "full" because bf16 rounding alone moves verdict scores by up to 0.25 on CPU. First real-model check (arm A, n = 100): components only keep the same verdict separation as all off (8.57 of 11.23) [assumed: small sample, preliminary].
- The tiny test model's outputs are nearly uniform (logit std 0.01), so the Exp 2 smoke run shows only that the code runs; the unit test scales its final norm to get meaningful KLs.

## Next

1. Commit (two commits: code, then records).
2. H100: Exp 0, Exp 1, Exp 2 (~40 min each [assumed]).
3. Analyses: V on Exp 0–2, the untrained baseline and arm A (full test split); `truth_writers.py` on Exp 0–2; component overlap (to write).

- See also: [convos/julian/26-10-06_truth_writing_components_SUMMARY.md] — source of the KL-blindness argument, the Q4 loss options this topic implements, gate G (on tiu the next token already depends on truth), the analysis script `truth_writers.py` and its stored features (R3 uses their layer-23 residuals), and the single-component ablation numbers R2 is compared with.
- See also: [convos/julian/26-10-02_training_run_SUMMARY.md] — arm B (= Exp 0 here) was never run and was flagged as needing its own tuning.
- See also: [convos/julian/26-10-05_uth_experiments_SUMMARY.md] — the UTH series; Julian's argument against UTH data for mechanism finding is recorded in this topic's LOG.
