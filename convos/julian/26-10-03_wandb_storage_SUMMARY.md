# SUMMARY: WandB storage (5 GB cap)

**Last updated:** 26-10-21 (Exp 4d checkpoint and incomplete runs deleted; listed files 3.31 GB; counter re-check pending)

Julian's WandB account (`bitt-j-personal`) has 5 GB of storage, and the website showed ~9.9 GB. Two goals: free space, and keep future runs under the cap.

The topic also started a convention [decided: Julian, this LOG's opening and a transcribed chat message]. A topic may begin as a file the user writes. If it is not already named `{date}_{topic}_LOG.md` (here it was `new_topic.md`), the agent renames it to that pattern. Recorded in CLAUDE.md (*Topic starts from a file*) and `docs/HOW_TO_USE.md`.

## Findings [verified: WandB API, 26-10-03]

- WandB counts 9.87 GB, but the files that exist total 5.32 GB, almost all in project `spd`.
  - Of that, 5.00 GB is 21 checkpoints of 238 MB each, from five runs: run 1, e2b, e2c, e3a, e3b of `convos/julian/26-10-02_training_run_SUMMARY.md`.
  - Old toy runs and the other 18 projects are negligible.
- The ~4.5 GB gap belongs to nothing WandB still lists: no deleted runs, no artifacts.
  - It includes the four run-1 checkpoints Julian deleted (0.95 GB, still counted).
  - WandB documents that the meter lags deletions by hours. The rest (~3.6 GB) is unexplained [assumed: pending earlier deletions or a stuck counter].
  - If it persists a day after cleanup: the *Manage storage* page, then WandB support.
- Even counted correctly, 5.32 GB is over the cap.

## Changes made

- `config_truth_{all_tokens,last_token,padded}.yaml`: `save_freq: null`, so only the final checkpoint is saved (one 238 MB file per run instead of five) [decided: Julian's opening request in the LOG, "the experiments for the decompositions should only save the last decomposition"; verified 26-10-04: all three arm configs have `save_freq: null`]. `config_truth_untrained.yaml` keeps `save_freq: 1000`, which never applies at `steps: 0`.
- `docs/CODING.md` (*Running jobs on a rented vast.ai GPU*):
  - analysed decompositions keep the final checkpoint only;
  - tuning/trial runs set `sync_checkpoints_to_wandb: false`, because their result is the metrics [concluded, not ratified: Julian offered "don't save" or "delete later"; the agent chose "don't save" because deletion gets forgotten];
  - delete unneeded checkpoints promptly;
  - how to read the usage counter.
- `docs/CODING.md` also says: agents never delete checkpoints on WandB or locally without asking Julian first, listing run ids and sizes [decided: Julian, chat, transcribed in the LOG].
- Not changed: `spd-vast --experiment` still forces `sync_checkpoints_to_wandb: true` (suits arm runs; hand-launched trial configs keep their own setting). Also noted in `docs/CODING.md`.
- Budget: a full set (3 arms + untrained baseline) is ~0.95 GB.
- Not done: no pruning script (one-off API deletion suffices); storing checkpoints in bf16 would halve them but is untested for the analysis, so not proposed.

## Checkpoint size (tiu arm settings)

[verified: state dict of e3a `model_2500.pth`, 26-10-03]

- Settings: `down_proj` of layers 15–19, `C: 96` each, global shared CI MLP with `hidden_dims: [512]`, fp32.
- Size: 238.2 MB per checkpoint.
  - 194 MB (82%): the CI function's first layer, 94,720 × 512. Its input is the 5 × 18,944 MLP inputs, concatenated.
  - 43 MB: the components (`V` 36.4 MB, `U` 6.9 MB).
  - The delta component is not saved.
- Other settings give other sizes [concluded]: the size scales mainly with the summed input width of the decomposed modules times the CI hidden width.

## Cleanup done (option B, Julian's choice)

- Julian deleted 4 of run 1's 5 checkpoints himself.
- I deleted the 16 intermediate checkpoints of e2c, e2b, e3a and e3b (3.81 GB).
- Each of the five runs keeps only its final checkpoint. When cleaned up, e2b's `model_5000.pth` matched the all-tokens arm config (except `label` and WandB project). Since then the arm configs load the frozen model in bf16 (`pretrained_model_dtype`, `convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md`), while e2b loaded it in fp32, so it is no longer an exact match.
- Option B was chosen by Julian in a multiple-choice chat dialog; no free-text message exists to transcribe.
- Real usage now: 1.50 GB in `spd` [verified: API listing].
- Not chosen: option A, deleting all checkpoints except e2b's final (~0.56 GB left).

## Second cleanup (26-10-20): tier 1 deleted

[verified: WandB API, 26-10-20 13:27 UTC] Usage 4.73 GB, equal to the listed files, so the 26-10-03 counter excess was lag and is gone. 17 checkpoints hold ~4.4 GB: 14 × 238.2 MB (tiu-style) and 3 × 333.5 MB (Exp 4a/4b/4c, MLPs 11–17). 14 of them have byte-identical local copies under `~/spd_out/`; UTH `s-d2ded461` and Exp 4a/4b/4c do not.

Urgent because the two Exp 4d runs (`s-5c284ebb` running, second to follow) upload ~334 MB each [assumed: same settings as 4a–4c], which would reach ≈ 5.40 GB.

Ranking (the tiers are the agent's proposal; deleting tier 1 was Julian's decision):
- **Tier 1, deleted 26-10-20 (1.43 GB)** [decided: Julian, chat transcribed in the LOG; verified: no `.pth` left in these runs, local copies kept]: tuning runs run 1 `s-c2562082`, e2c `s-b3401a03`, e3a `s-a77ae04e`, e3b `s-83033a96`; e2b `s-ba5f9dd5` (fp32 twin of arm A); fp32 untrained `s-286aa6a9` (replaced by bf16 `s-7fad0c14`). All closed, local copies exist, nothing uses them. Afterwards with both 4d uploads ≈ 3.97 GB.
- **Tier 2 (238 MB each, local copies exist, still referenced):** Exp 0 `s-aeaa70d8`, Exp 2 `s-05db01ad`, Exp 3 `s-161f8a7b`, code control `s-b6cce5de`. Cost of deleting: instance-side re-analysis needs an upload from the laptop.
- **Keep:** arm A `s-bd23f0d1`, untrained `s-7fad0c14`, Exp 1 `s-cd129592`, Exp 4a/4b/4c, UTH `s-d2ded461` (no local copy).

## Third deletion (26-10-21): Exp 4d

- Deleted [decided: Julian, C: comment in `convos/julian/26-10-16_faithful_targeting_LOG.md`; verified: re-listing]: the 4d checkpoint `s-5c284ebb/model_5000.pth` (333.5 MB, no local copy; the run and its metrics stay) and the three incomplete 4d runs `s-b8c8c610`, `s-37397e7d`, `s-15cc0844` (whole runs, 3.3 MB). The second 4d run never uploaded a checkpoint.
- Usage [verified: API listing over all projects, 26-10-21]: listed files **3.31 GB** (11 checkpoints, 3.04 GB: the 8 × 238.2 MB and 3 × 333.5 MB of the *Keep* and *Tier 2* lists above). No Exp 4e checkpoint is on WandB. The counter showed 3.65 GB right after; the excess equals the file just deleted, so it is lag [concluded].
- Open: re-check `entity.storageBytes` (expected ≈ 3.31 GB, 1.69 GB under the cap). Tier 2 stays until space is short again.

Details and per-run reasons: this LOG, entries *second cleanup: deletion ranking 26-10-20*, *tier 1 deleted 26-10-20* and *4d checkpoint and incomplete runs deleted 26-10-21*.

- See also: [convos/julian/26-10-02_training_run_SUMMARY.md] — the runs whose checkpoints fill the storage, and the tuning verdict that makes them disposable.
- See also: [convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md] — probe test accuracies of the finished decompositions; runs on the laptop CPU.
- See also: [convos/julian/26-10-02_epistemic_memory_setup_SUMMARY.md] — set up the memory conventions; this topic added one (*Topic starts from a file* in CLAUDE.md).
- See also: [convos/julian/26-10-02_overview_of_goal_SUMMARY.md] — designed the tiu arm configs whose `save_freq` this topic set to final-checkpoint-only, and set `sync_checkpoints_to_wandb: true` in them.
- See also: [convos/julian/26-10-05_no_truth_baseline_SUMMARY.md] — uploaded three new final checkpoints (bf16 arm A, code control, untrained; ~238 MB each), budgeted against this topic's 5 GB cap and post-cleanup usage.
- See also: [convos/julian/26-10-16_faithful_targeting_SUMMARY.md] — the Exp 4d runs whose checkpoint uploads made the 26-10-20 cleanup urgent.
