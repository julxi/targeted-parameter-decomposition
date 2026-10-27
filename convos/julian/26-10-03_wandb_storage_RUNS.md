# WandB run register (project `bitt-j-personal/spd`)

**Last updated:** 26-10-27 (paper phase C/D: 6 final checkpoints on WandB, tuning checkpoints copied locally; the 4 tier-2 checkpoints deleted, their rows moved to the metrics-only table)

Living list of the decomposition runs on WandB, so Julian can delete the ones nobody needs from time to time. Agents add a row when they launch a run, and update its row when its checkpoint or status changes; rows of deleted runs are removed (the deletion itself is recorded in `convos/julian/26-10-03_wandb_storage_LOG.md`). Agents never delete anything themselves (`docs/CODING.md`); Julian does it or approves it.

How to delete: on the website, open the project, search the run by its id or name, then *Delete run* (whole run) or the *Files* tab (only the `.pth`). Runs launched from 26-10-21 on carry their label as the WandB name (e.g. `e4_r2_lr_seed_alltok_lr5e-3`); older runs are named by their id.

What deleting frees: only checkpoints (`.pth`, 238.2 MB for runs on MLPs 15–19, 333.5 MB for Exp 4a–4c on MLPs 11–17) matter for the 5 GB cap. A run without a checkpoint holds 1–3 MB (metrics, logged code, config); deleting it loses its training curves only. Numbers marked "recorded" exist in the named LOG and, for analyses, as local JSONs under `~/spd_out/`.

Columns: *size* = the whole run on WandB, checkpoint included [verified: WandB API, 26-10-21]; deleting only the `.pth` frees 238.2 or 333.5 MB of it, the rest is metrics and logged code; *local* = a copy under `~/spd_out/`; *advice* = the agent's suggestion, not a decision.

## Runs with a checkpoint on WandB (the storage that counts)

| run id | date | label | size | local | used by | advice |
|---|---|---|---|---|---|---|
| s-7fad0c14 | 26-10-05 | tiu_untrained (bf16) | 238 MB | yes | baseline of all tiu probe analyses | keep |
| s-bd23f0d1 | 26-10-05 | tiu_all_tokens (arm A, bf16) | 241 MB | yes | arm A, truth-writing analyses | keep |
| s-d2ded461 | 26-10-05 | uth_all_tokens | 260 MB | **no** | UTH series (closed) | keep (no copy) |
| s-cd129592 | 26-10-13 | tiu_residual_lens (Exp 1) | 240 MB | yes | bolder experiments | keep |
| s-4cd6abe5 | 26-10-16 | tiu_sf_verdict_L11_17 (Exp 4a) | 336 MB | **no** | Exp 4, reference for the 4e analysis check | keep |
| s-717b50c9 | 26-10-16 | tiu_sf_verdict_recon_L11_17 (Exp 4b) | 336 MB | **no** | Exp 4 | keep |
| s-5ecb5791 | 26-10-16 | tiu_sf_verdict_recon_alltok_L11_17 (Exp 4c) | 337 MB | **no** | Exp 4c, component steering | keep |
| s-7152548c | 26-10-27 | paper_final_plain_plain_s1 (paper final, plain tPD, seed 1) | 334 MB (`.pth` 333.5 MB) | yes | paper topic, canonical run candidate | keep |
| s-46efc525 | 26-10-27 | paper_final_plain_plain_s2 (paper final, plain tPD, seed 2) | 334 MB (`.pth` 333.5 MB) | yes | paper topic, canonical run candidate | keep |
| s-eeac663f | 26-10-27 | paper_final_rest_recon_s1 (paper final, + reconstruction, coef. 10, seed 1) | 334 MB (`.pth` 333.5 MB) | yes | paper topic, canonical run candidate | keep |
| s-edc17041 | 26-10-27 | paper_final_rest_recon_s2 (paper final, + reconstruction, coef. 10, seed 2) | 334 MB (`.pth` 333.5 MB) | yes | paper topic, canonical run candidate | keep |
| s-0c2fa0d7 | 26-10-27 | paper_final_rest_ortho_s1 (paper final, + orthogonality, LR 2e-3, coef. 10, seed 1) | 334 MB (`.pth` 333.5 MB) | yes | paper topic, canonical run candidate | keep |
| s-4186f973 | 26-10-27 | paper_final_rest_ortho_s2 (paper final, + orthogonality, LR 2e-3, coef. 10, seed 2) | 334 MB (`.pth` 333.5 MB) | yes | paper topic, canonical run candidate | keep |

## Runs without a checkpoint (metrics only, 1–3 MB each)

All deletable whenever Julian wants a tidy project; their results are recorded where named.

| run id | date | label | status | recorded in |
|---|---|---|---|---|
| s-c2562082 | 26-10-02 | tiu_all_tokens (run 1, tuning) | finished | `26-10-02_training_run_LOG.md` |
| s-b3401a03 | 26-10-03 | e2c_b4_s10000 | finished | same |
| s-ba5f9dd5 | 26-10-03 | e2b_b16_s5000 | finished | same |
| s-a77ae04e | 26-10-03 | e3a_b16_s2500_lr1e-3 | finished | same |
| s-83033a96 | 26-10-03 | e3b_b16_s5000_lr1e-3 | finished | same |
| s-286aa6a9 | 26-10-03 | tiu_untrained (fp32, replaced) | finished, 0 MB | `26-10-03_test_accuracy_analysis_LOG.md` |
| s-1c8cd417 | 26-10-13 | trial_smoke_exp1 | finished | `26-10-10_bolder_experiments_LOG.md` |
| s-6e70a646 | 26-10-13 | trial_smoke_exp2 | finished | same |
| s-50d0c294 | 26-10-13 | trial_exp0_b16 | finished | same |
| s-54d0f743 | 26-10-13 | trial_exp0_b64 | finished | same |
| s-773bfec6 | 26-10-13 | tiu_last_token | crashed at step 1,600 | same |
| s-ad22ae02 | 26-10-16 | trial300_sf_verdict_recon | finished | same |
| s-4c0fff71 | 26-10-16 | trial300_sf_verdict | finished | same |
| s-5c284ebb | 26-10-20 | tiu_sf_verdict_ortho1_L11_17 (Exp 4d; checkpoint deleted 26-10-21) | finished | `26-10-16_faithful_targeting_LOG.md` |
| s-16271b38 | 26-10-20 | e4_trial300_end | finished | same |
| s-294c4acf | 26-10-20 | e4_trial300_alltok | finished | same |
| s-114f2c0e | 26-10-20 | e4_r1_lr_alltok_lr1e-4 | finished | same |
| s-c706fefb | 26-10-20 | e4_r1_lr_end_lr1e-4 | finished | same |
| s-e6688f03 | 26-10-20 | e4_r1_lr_end_lr5e-4 | finished | same |
| s-348832ad | 26-10-20 | e4_r1_lr_alltok_lr5e-4 (checkpoint local) | finished | same |
| s-9dc19f9f | 26-10-20 | e4_r1_lr_end_lr2e-3 | finished | same |
| s-eb1c20d5 | 26-10-20 | e4_r1_lr_alltok_lr2e-3 (best 4e run; checkpoint local) | finished | same; keep until 4e is done |
| s-2db4e79d | 26-10-20 | e4_r2_lr_ext_alltok_lr1e-3 | killed in training | same |
| s-52b2abc7 | 26-10-21 | e4_r2_lr_seed_alltok_lr2e-3_seed1 (seed 1 of s-eb1c20d5; checkpoint local) | finished | same; keep until 4e is done |
| s-9d172777 | 26-10-21 | e4_r2_lr_seed_alltok_lr5e-3 | finished | same |
| s-7944818e | 26-10-21 | e4_r2_lr_seed_alltok_lr1e-2 | killed at start | same |
| s-e9552de2 | 26-10-21 | e4_r3_impmin_alltok_imp1e-2 (sparse, 8 components; checkpoint local) | finished | same; keep until 4e is done |
| s-8ec00fb3 | 26-10-21 | e4_r3_impmin_alltok_imp3e-3 (false-selective components; checkpoint local) | finished | same; keep until 4e is done |
| s-a55b67a4 | 26-10-21 | e4_r3_impmin_alltok_imp3e-4 (checkpoint local) | finished | same |
| s-72534289 | 26-10-26 | paper_t300_plain (300-step trial; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-72534289/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-6fccc66b | 26-10-26 | paper_t300_recon (300-step trial; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-6fccc66b/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-4acee2cb | 26-10-26 | paper_t300_ortho (300-step trial; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-4acee2cb/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-f03881a2 | 26-10-26 | paper_c1_lr_lr5e-4 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-f03881a2/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-63e23c1b | 26-10-26 | paper_c1_lr_lr2e-3 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-63e23c1b/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-13746f5b | 26-10-26 | paper_c1_lr_lr5e-3 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-13746f5b/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-42fd706b | 26-10-27 | paper_c2_imp_imp0.0003 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-42fd706b/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-e4d259ea | 26-10-27 | paper_c2_imp_imp0.003 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-e4d259ea/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-614cd983 | 26-10-27 | paper_c3_coef_recon_c0.1 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-614cd983/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-2821a387 | 26-10-27 | paper_c3_coef_recon_c1 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-2821a387/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-9fa49cc7 | 26-10-27 | paper_c3_coef_recon_c10 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-9fa49cc7/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-baae9857 | 26-10-27 | paper_c3_coef_ortho_c0.1 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-baae9857/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-a53966f6 | 26-10-27 | paper_c3_coef_ortho_c1 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-a53966f6/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-d383b59a | 26-10-27 | paper_c3_coef_ortho_c10 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-d383b59a/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-fecf62f5 | 26-10-27 | paper_c4_lr_ortho_lr2e-3_c0.1 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-fecf62f5/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-53f816bf | 26-10-27 | paper_c4_lr_ortho_lr2e-3_c1 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-53f816bf/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-ae91d311 | 26-10-27 | paper_c4_lr_ortho_lr2e-3_c10 (tuning, 3,000 steps; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-ae91d311/`) | finished | `convos/julian/26-10-22_tpd_truth_paper_LOG.md` |
| s-b6cce5de | 26-10-05 | code_control_all_tokens (`.pth` deleted from WandB 26-10-27, tier 2, Julian approved; local copy `~/spd_out/runs/spd-s-b6cce5de/model_5000.pth`) | finished | code control |
| s-05db01ad | 26-10-13 | tiu_stimulus_lens (Exp 2) (`.pth` deleted from WandB 26-10-27, tier 2, Julian approved; local copy `~/spd_out/runs/spd-s-05db01ad/model_5000.pth`) | finished | bolder experiments |
| s-161f8a7b | 26-10-14 | tiu_stimulus_span (Exp 3) (`.pth` deleted from WandB 26-10-27, tier 2, Julian approved; local copy `~/spd_out/runs/spd-s-161f8a7b/model_5000.pth`) | finished | bolder experiments |
| s-aeaa70d8 | 26-10-14 | tiu_last_token (Exp 0) (`.pth` deleted from WandB 26-10-27, tier 2, Julian approved; local copy `~/spd_out/runs/spd-s-aeaa70d8/model_5000.pth`) | finished | bolder experiments |

## Older runs (before this project's LM work)

55 rotgrid toy runs from 26-09-21 to 26-09-25 (labels `rotgrid_3L_C*`, 3–8 MB each with their small checkpoints, together ≈ 0.3 GB) and the other WandB projects of the account. They predate the Epistemic Memory records, so no LOG describes them [verified: WandB listing 26-10-21; the 26-10-03 inventory counted 56].
