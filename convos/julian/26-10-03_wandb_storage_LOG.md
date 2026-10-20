This topic is about wandb, but before that another point. I want to be able to start conversations by creating a new log file that doesn't have to fit the pattern. You can then rename the file to fit the `date_topic_LOG` pattern.

So now to the main point. I only have 5GB of space on wandb ond currently I'm using nearly 10GB. So there are some things that have to be done:

1. Make some space. This is a bit awkward. When I checkout on the website the `spd/runs` folder it says it needs 9.9GB of storage. However in the runs folder I can see 4 runs that have 1.2GB and the rest of the runs are like 7.6MB (and not nearly enought to sum up to some GB)
2. Make sure that I stay under the 5GB restriction. This means for me that we should only save checkpoints when they are necessary. For example the experiments for the decompositions should only save the last decomposition, as  I don't think I need any in between checkpoints (I don't see for what). For the experiments or the hyperparameter tuning, either don't save any checkpoints, or delete the checkpoints when we are reasonably sure that we don't need them anymore.
---

26-10-03, Claude Opus 5.5 — *WandB storage inventory and checkpoint policy*

I have read INDEX.md, the training-run SUMMARY, `docs/CODING.md`, and the checkpoint code (`spd/run_spd.py`, `spd/configs.py`, `spd/scripts/run_vast.py`). I queried the WandB API live.

C: (transcribed from chat) I deleted the checkpoints from the run

(Julian sent this mid-session. It refers to run 1, `s-c2562082`. That run had five checkpoints on 26-10-02 (training-run LOG, *first look at the finished all-tokens run s-c2562082*), and now has only `model_10000.pth`.)

## 1. Free-form topic start (done)

The file was `convos/julian/new_topic.md`. I renamed it to this LOG; Julian's text above is unchanged. The convention is now in two places:
- CLAUDE.md, a paragraph *Free-form topic starts* below *Conversation index*: any file name, the agent renames to `{date}_{topic}_LOG.md`, keeps the user's text verbatim at the top, then creates SUMMARY and INDEX row.
- `docs/HOW_TO_USE.md`, step 1.

## 2. Where the 9.9 GB come from

[verified: WandB public + GraphQL API, 26-10-03]

| quantity | value | source |
|---|---|---|
| account storage counted by WandB | 9.87 GB | `entity.storageBytes` |
| project `spd`, counted | 9.85 GB | `storageTree` |
| project `spd`, sum over its 60 listed runs | 6.03 GB | `storageTree` run nodes |
| project `spd`, files that actually exist | 5.32 GB | `run.files()`, matches `project.storageBytes` |
| all other 18 projects together | 0.05 GB | `run.files()` + logged artifacts |

The 5.32 GB of real files are almost all checkpoints: 21 files of 238.2 MB = 5.00 GB. The 56 old toy runs (Sept. 21–25) are 6–8 MB each, 0.3 GB total.

| run | label | checkpoints on WandB |
|---|---|---|
| `s-c2562082` | run 1, all-tokens, batch 4, 10k (old default) | `model_10000` (4 deleted by Julian) |
| `s-b3401a03` | e2c: batch 4, 10k | 5 (2k … 10k) |
| `s-ba5f9dd5` | e2b: batch 16, 5k, LR 5e-4 (the adopted setting) | 5 (1k … 5k) |
| `s-a77ae04e` | e3a: batch 16, 2.5k, LR 1e-3 | 5 (500 … 2.5k; trial config had `save_freq: 500`) |
| `s-83033a96` | e3b: batch 16, 5k, LR 1e-3 | 5 (1k … 5k) |

**Why the website shows more than the visible runs:** the remaining ~4.5 GB (9.87 − 5.32) belongs to nothing that WandB still lists.
- No deleted runs: `project.deletedRuns` is empty.
- No artifacts of size.
- Part of it is visible: run 1 is still counted at 1.19 GB, i.e. with the four checkpoints Julian deleted (0.95 GB).
- WandB documents that the storage meter lags deletions by a few hours, because deletion runs in the background ([WandB docs](https://docs.wandb.ai/support/models/articles/why-does-the-storage-meter-not-update-af)).

[concluded] So the 0.95 GB from run 1 should disappear from the counter on its own. The other ~3.6 GB I cannot explain from the API. Candidates [assumed]:
- deletions made earlier that are still pending;
- a stuck counter, which the WandB community forum reports for other users.

If the counter is still above the real size a day after the next deletion, the step is the *Manage storage* page (`https://wandb.ai/account-settings/bitt-j-personal/usage/manage/`), then WandB support.

**Even with a correct counter, 5.32 GB is over the 5 GB cap**, so checkpoints have to go.

## 3. Cleanup proposal (waiting for Julian; deletion is irreversible)

The tuning verdict is ratified (training-run SUMMARY), so all five runs are tuning runs whose checkpoints have no planned use, with one exception:
- e2b's `model_5000.pth`. e2b's config equals the current `config_truth_all_tokens.yaml` except for `label` and the WandB project [verified 26-10-03 in the training-run topic].
- So it is in effect the all-tokens arm's decomposition and could be probed directly, saving a ~28-minute H100 run.

Options:
- **A (recommended): keep only e2b's `model_5000.pth`, delete the other 20 checkpoints** (4.76 GB). Afterwards ~0.56 GB.
- **B: delete only the intermediate checkpoints** (16 files, 3.81 GB). Keeps the five final checkpoints; afterwards ~1.51 GB.

Metrics, configs and figures stay on WandB in both cases.

## 4. Staying under 5 GB (done, apart from the open choice above)

- **Arm configs** `config_truth_{all_tokens,last_token,padded}.yaml`: `save_freq: 1000` → `null`, with a comment why.
  - `run_spd.py` always saves at `step == config.steps`, so each arm run now uploads one 238 MB checkpoint instead of five.
  - [verified: all four configs load, `save_freq` None for the three arms.]
  - `config_truth_untrained.yaml` unchanged: it has `steps: 0` and returns after saving `model_0.pth`, so `save_freq` never applies.
- **Rule in `docs/CODING.md`**, section *Running jobs on a rented vast.ai GPU*:
  - analysed decompositions save the final checkpoint only;
  - tuning and trial runs don't sync checkpoints (`sync_checkpoints_to_wandb: false`), because their result is the metrics;
  - delete checkpoints on WandB once unneeded;
  - how to read the usage counter.
  - [concluded] I chose "don't sync" over "sync, delete later" for tuning runs: deletion is a step that gets forgotten, and the e2/e3 checkpoints were never used.
- **Not changed:** `spd-vast --experiment` still forces `sync_checkpoints_to_wandb: true` (`_build_config` in `spd/scripts/run_vast.py`). That suits arm runs. The e2/e3 trials were launched by hand over ssh with their own configs, which the forcing doesn't touch.
- **Budget** [concluded]: three arms + untrained baseline = 4 × 238 MB ≈ 0.95 GB per full set, so about 4 more sets fit next to option A's 0.56 GB.

Not done: no pruning script. A one-off API deletion suffices for now (YAGNI). Halving the checkpoint size (bf16 storage) is possible but untested for the analysis, so not proposed.

---

26-10-03, Claude Opus 5.5 — *intermediate checkpoints deleted (option B)*

Julian chose option B in the chat question dialog ("Delete intermediates only"). That was a multiple-choice selection, so there is no free-text message to transcribe.

Deleted via the WandB API: 16 checkpoints, 3.81 GB. Four per run, from e2c `s-b3401a03`, e2b `s-ba5f9dd5`, e3a `s-a77ae04e` and e3b `s-83033a96`. A guard asserted five checkpoints per run, including `model_{steps}.pth`, and skipped that final one.

[verified: file listing after deletion] Each of the five runs now holds exactly one checkpoint, its final one:
- run 1: `model_10000`
- e2c: `model_10000`
- e2b: `model_5000`
- e3a: `model_2500`
- e3b: `model_5000`

Real size of project `spd`: 1.50 GB.

The WandB counter will lag by hours. Next check: `entity.storageBytes` should fall toward ~1.55 GB (1.50 GB in `spd` plus 0.05 GB in the other projects). If a ~3.6 GB excess remains a day later, it is the unexplained part from *WandB storage inventory and checkpoint policy*: go to *Manage storage*, then WandB support.

---

C: (transcribed from chat) Can you change the start a topic, that I can either define `{date}_{topic}_LOG.md` or if it doesn't fit that formalt it gets renamed?
In the CODING.md can you add that agents shouldn't delet checkpoints themselves but ask first. Also the reference to a Qwen-7B checkpoint is too general. I think it depends on the exact settings. I think the sice of a checkpoint should live somewhere else.

---

26-10-03, Claude Opus 5.5 — *topic-start wording, no-delete rule, checkpoint size moved*

- **Topic start:**
  - CLAUDE.md paragraph renamed to *Topic starts from a file*. A file already named `{date}_{topic}_LOG.md` keeps its name; any other name gets renamed to that pattern.
  - `docs/HOW_TO_USE.md` step 1 says the same.
- **`docs/CODING.md`, WandB bullet:**
  - New rule: agents never delete checkpoints on WandB or locally; they list the files (run ids, sizes) and ask Julian first.
  - Destroying an instance is excluded, because the trial-run policy relies on it, after the existing "before the instance is destroyed" copy step.
  - The "one Qwen-7B checkpoint is 238 MB" line is replaced by: the size depends on the decomposition settings; check it for the config at hand; measured sizes live in the runs' topic SUMMARY (this one for the tiu arms).
- **Checkpoint size, measured** [verified: e3a `model_2500.pth` downloaded and its state dict inspected, 26-10-03]. 238.2 MB, all fp32, for the tiu arm settings: `down_proj` of layers 15–19, `C: 96` each (480 components), `ci_config` global shared MLP with `hidden_dims: [512]`. The breakdown:

  | part | shape | size |
  |---|---|---|
  | CI function, first layer `W` | 94,720 × 512 (inputs of all 5 layers, 5 × 18,944, concatenated) | 194 MB |
  | CI function, rest | 512 × 480 + biases | 1 MB |
  | components `V` | 5 × (18,944 × 96) | 36.4 MB |
  | components `U` | 5 × (96 × 3,584) | 6.9 MB |

  So 82% of a checkpoint is the CI function's input layer; the components are 43 MB. The delta component is not stored. [concluded] The size grows with the summed input width of the decomposed modules times the CI hidden width, and only weakly with `C`.

Recorded in this topic's SUMMARY. Earlier entries still state "238 MB per Qwen-7B checkpoint" without the settings; they are accurate for the tiu arms only.

---

My WandB storage is getting full. I want to know which decompositions I should delete first

---

26-10-20, Claude Opus 5.5 — *second cleanup: deletion ranking 26-10-20*

I have read INDEX.md, this topic's SUMMARY and LOG tail, the WandB rules in `docs/CODING.md`, and the SUMMARYs that name each run. Nothing deleted (CODING.md rule: list, then Julian decides).

**Inventory** [verified: WandB API over all projects of `bitt-j-personal`, 26-10-20 13:27 UTC; script and JSON in the session scratchpad, not kept]:
- `entity.storageBytes` = 4.73 GB; the listed files also total 4.73 GB, so the ~3.6 GB unexplained excess from 26-10-03 is gone (the counter had lagged).
- 17 checkpoints hold ~4.4 GB of that: 14 at 238.2 MB (tiu-style, layers 15–19) and 3 at 333.5 MB (Exp 4a/4b/4c, layers 11–17). Everything else (toy rotgrid runs, metrics) is < 8 MB per run.
- Local copies under `~/spd_out/runs/spd-<id>/` or `~/spd_out/spd/<id>/` exist for 14 of the 17, each byte-identical in size to the WandB file [verified: byte counts compared]. No local copy: UTH stage 1 `s-d2ded461`, Exp 4a/4b/4c.

**Why now:** the Exp 4d full run `s-5c284ebb` (running since 13:03 UTC) has `sync_checkpoints_to_wandb: true`, and per `convos/julian/26-10-16_faithful_targeting_SUMMARY.md` a second 4d run (coefficient 10) follows. With 7 layers × `C: 96` like 4a–4c, each final checkpoint should be ~334 MB [assumed: same settings as 4a–4c]. 4.73 + 0.67 ≈ 5.40 GB, over the 5 GB cap.

**Tier 1 — delete first** (6 checkpoints, 1.43 GB). Each experiment is closed, its numbers are in the records, a local copy exists, and no code or open plan uses it:

| run | what | why disposable |
|---|---|---|
| `s-c2562082` | run 1, all-tokens arm, batch 4, 10k steps | tuning run; verdict (batch 16, 5k, LR 5e-4) applied to the configs |
| `s-b3401a03` | e2c, batch 4, 10k | tuning run (same verdict) |
| `s-a77ae04e` | e3a, LR 1e-3, 2.5k | tuning run |
| `s-83033a96` | e3b, LR 1e-3, 5k | tuning run |
| `s-ba5f9dd5` | e2b, fp32 frozen model, final arm-A settings | replaced by the bf16 arm A `s-bd23f0d1`, which matches it within run-to-run noise |
| `s-286aa6a9` | untrained baseline, fp32 | replaced by the bf16 untrained `s-7fad0c14`; untrained checkpoints regenerate in ~2 min on the laptop CPU anyway |

After tier 1 and both 4d uploads: ≈ 3.97 GB [concluded: 4.73 − 1.43 + 0.67].

**Tier 2 — closed experiments, local copy exists, but a script or record still points at them** (each 238 MB). Deleting costs convenience, not data: analyses on a vast.ai instance fetch checkpoints from WandB (CODING.md), so a later re-analysis would need an upload from the laptop.
- `s-aeaa70d8` Exp 0 (next-token lens, recovered 8% of the verdict KL; the weakest lens).
- `s-05db01ad` Exp 2 and `s-161f8a7b` Exp 3 (both re-implement the lens per the 26-10-15 cancellation test; Julian read Exp 3 as no truth-mechanism evidence).
- `s-b6cce5de` code control (topic closed; named in `truth_writers.py`).

**Keep:**
- `s-bd23f0d1` arm A and `s-7fad0c14` untrained bf16: reference runs in `truth_writers.py`, `verdict_accuracy.py`, and the A1 test filed in FUTURE_WORK.md.
- `s-cd129592` Exp 1: the only run whose components partly decompose the MLP outputs (cancellation test 0.83).
- `s-4cd6abe5` / `s-717b50c9` / `s-5ecb5791` Exp 4a/4b/4c: current work (steering, 4d comparison), no local copy.
- `s-d2ded461` UTH stage 1: series closed, but no local copy. If you want its space, download it first.

Recommendation [concluded]: delete tier 1 now (before the first 4d checkpoint uploads); leave tier 2 until space is short again.

---

C: (transcribed from chat) ok, so can you delte the 6 checkpoints (1.43 GB) from above?

---

26-10-20, Claude Opus 5.5 — *tier 1 deleted 26-10-20*

Deleted on WandB (project `spd`), the checkpoint file only; runs and their metrics stay [decided: Julian, chat above]:
`s-c2562082` `model_10000.pth`, `s-b3401a03` `model_10000.pth`, `s-a77ae04e` `model_2500.pth`, `s-83033a96` `model_5000.pth`, `s-ba5f9dd5` `model_5000.pth`, `s-286aa6a9` `model_0.pth` (238.2 MB each, 1.43 GB total).

[verified: file listing after deletion] None of the six runs holds a `.pth` file anymore. The local copies under `~/spd_out/runs/spd-<id>/` and `~/spd_out/spd/s-286aa6a9/` are untouched.

`entity.storageBytes` right after deletion: still 4.73 GB (the counter lags deletions by hours, as on 26-10-03). Expected after the lag: ≈ 3.30 GB, then ≈ 3.97 GB once both Exp 4d checkpoints are in. Not yet re-checked.
