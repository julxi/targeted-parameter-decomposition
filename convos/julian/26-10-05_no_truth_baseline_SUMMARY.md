# SUMMARY: Code-lines control decomposition (low-truth baseline for the tiu truth experiments)

**Last updated:** 26-10-05 (topic closed: Julian ended the arm A experiment; summary in `scratch/arm_A_experiment_summary.md`; dataset committed in `157e7e7`)

**Status: closed** [decided: Julian, chat transcribed in the LOG, *topic closed; arm A experiment summary written*]. Julian ended the arm A experiment after the code-control result and declined the form-matched control. A self-contained summary of the whole arm A experiment is in `scratch/arm_A_experiment_summary.md` (gitignored, local). Bottom line [concluded]: no evidence that tPD recovers truth-specific mechanisms here.

Julian wants another baseline for the simple truth experiments: tPD (targeted parameter decomposition) on Qwen2.5-7B-Instruct with Truth-is-Universal (tiu) statements, whose CIs (causal-importance values) are probed for statement truth. "Simple" because tiu statements are unambiguous. This picks up the open TODO in `convos/julian/26-10-02_overview_of_goal_SUMMARY.md` (*Follow-up after results*).

## Design

- **Control decomposition** [decided: C: comment in the LOG]: a tPD run with arm A's settings (the all-tokens arm), but the target is data with little truth content. Its CIs are probed on the tiu statements next to the tiu arms. Rejected reading: replacing the Pile non-target stream.
- **Target: code** [decided: Julian asked for "something completely unrelated. Maybe coding"; the agent chose code]. Caveat [concluded]: an unrelated control can show that the *target* matters, not that *truth* matters rather than "short statements about cities/elements/animals/numbers". A form-matched control (e.g. "The city of Krasnodar.") would separate those; suggested only if code shows a difference.
- **Straight to the control run** [decided: C: comment]; the sparsity-matched untrained baseline (each statement keeps its untrained network's top ~10 CIs) is deferred.
- **What it can show** [concluded; Julian wants to check]: plain probe accuracy probably can't separate the runs: untrained CIs already reach 0.998, the residual-stream ceiling (trained arm A: 0.970–0.984; `convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md`). Informative: the on/off pattern of the active components, the top-k components, generalisation. The control's sparsity on tiu (inputs it was never trained on) must be reported; if it is far from arm A's ~10 active components per statement, compare at matched sparsity.
- **"Little truth feature" criterion** [decided: C: comment, "let's try this out"]: project the control data onto the tiu truth directions of the layer-19 residual stream. Caveat: code is far from tiu in activation space, so the spread is compared against random directions too; a rough sanity check, not proof.

## Implemented [verified: linters clean; dry run inspected by eye; committed by Julian in `884052a`; no training yet]

- **Generator** `spd/experiments/lm/honesty_targeted_decomposition/build_pile_code.py --version v1` → `data/pile_code/github_lines/v1/`. Takes ~2.5 min on the laptop and prints nothing until it finishes (it streams the whole Pile test file); don't stop it early. Single stripped lines of code from the GitHub subset of the Pile's `test.jsonl.zst` (`monology/pile-uncopyrighted`, revision pinned). The tiu arms' Pile stream never reads this file. Comment and prose lines are filtered heuristically; lines are deduplicated; the split is by source document. The per-split length histogram in Qwen tokens equals tiu v1's (6,560 train / 1,794 test). Record:

  ```json
  {"text": "GraphicPrimitive graphic = (GraphicPrimitive) element;", "n_tokens": 10, "pile_doc_index": 122187, "doc_line": 86}
  ```
  (a record from the dry run's train split; `pile_doc_index` counts all documents of the file, `doc_line` is 0-based.)
- **Dry run** (`~/spd_out/26-10-05_no_truth_baseline/`): all 18,195 GitHub documents of the file read; the length pools hold ≥ 2.4× the needed count at every length. The kept lines are code from 4,894 documents, a few with short English fragments.
- **Config** `config_code_control.yaml` = `config_truth_all_tokens.yaml` with only the label and the target changed (bf16 frozen model, batch 16, 5k steps, LR 5e-4).
- **`probe_ci.py`**: runs may now have different targets; the probe data is the first run's tiu data (a non-tiu first run fails loudly); `results.json` records each run's target.

## Projection check [verified: `~/spd_out/26-10-05_no_truth_baseline/projection_report.json`, dry-run code test lines, single run]

Layer-19 residual stream, truth directions fit on tiu train, code test lines (1,794) projected. d = tiu test true/false separation along the direction.
- Last token: code spread = 0.099 d (mass-mean direction) and 0.096 d (logistic regression), *narrower* than tiu's within-class spread (0.47× and 0.64×). Along random directions code spreads 1.14–2.62× (5%–95%).
- Mean over tokens (logistic regression direction): code spread 0.21 d, the same ratio as along random directions (1.04× vs median 1.05×).
- Code's mean offset from the midpoint flips sign between the two last-token directions (+0.22 d vs −0.12 d), so it reads as an input-distribution offset, not as "code looks true" [concluded; not compared with random-direction offsets].

Conclusion [concluded]: code carries little of tiu's truth feature, so it passes the agreed criterion. Limit: it says nothing about any code-specific "correctness" feature along other directions.

## Runs [decided: bf16 route, Julian's chat transcribed in the LOG, *bf16 decision; untrained baseline regenerated; both runs launched on the H100*]

All existing tiu checkpoints loaded the frozen model in fp32, and `probe_ci.py` needs one shared dtype. So everything compared here is new and bf16:
- **Untrained baseline** `s-7fad0c14` (laptop CPU, `config_truth_untrained.yaml`), on WandB [verified]. It replaces the fp32 `s-286aa6a9` for this comparison.
- **Arm A** (`config_truth_all_tokens.yaml`, run `s-bd23f0d1`): finished 11:08 UTC, checkpoint on WandB [verified]. Final metrics match e2b (same settings, fp32) within run-to-run noise; target L0 7.01 vs 6.19 is the largest difference (table in the LOG). Then the **control** (`config_code_control.yaml`, run `s-b6cce5de`), finished 11:46 UTC (see below). Running on a vast.ai H100 (575 W, not throttled), launched 10:28:58 UTC by `/root/run_two.sh` (copy in `~/spd_out/26-10-05_no_truth_baseline/`). Logs in `/root/runs/`. The `exit=` field in `/root/runs/status.log` is wrong (always 0, a `$?` bug), so judge success from the run logs and WandB.

## Control result (training side) and throughput

- **Control `s-b6cce5de`** finished, checkpoint on WandB [verified]. It is a much smaller decomposition than arm A: on its own target, switching all components off costs 0.41 nats/token (arm A on tiu: 3.54), and it has 2.24 active components per token (arm A 7.01). Its target reconstruction is worse (rounded KL 0.315 vs 0.136), and it is more active on the Pile (0.47 vs 0.11). So it is not "equally sparse"; if its CIs are nearly all off on tiu, compare at matched sparsity [concluded; table in the LOG].
- **fp32 vs bf16 throughput** [verified: 150-step trials on the 26-10-05 instance, batch 16]: bf16 is 8–10% faster (296 vs 322 ms/step without PPGD, 591 vs 652 with) and uses ~27 GB instead of 49–54 GB. The arm-A slowdown versus e2a came from the host, which was ~37% slower than the 26-10-02 one in fp32. `nvidia-smi` utilisation is misleading here: bf16 shows lower utilisation while being faster.
- The vast.ai instance holds nothing unsaved; it can be destroyed.

## Probe results [verified: `~/spd_out/26-10-05_no_truth_baseline/probe/results.json`, `followup.out`; one seed per run; tiu test, 1,794 statements, chance 0.50]

Last-token readout, logistic regression: residual stream 0.998; untrained 0.998; arm A 0.991; **code control 0.654**.
- **The low score mostly reflects inactivity.** On tiu the control has 1.4 active components per statement (arm A 9.3, untrained 220), and 29% of statements have none. A probe has almost nothing to read; the top-k-matched readout can't fix this. So the control does **not** cleanly show "the truth target matters" [concluded]. The agent's prediction (≥ 0.97) was wrong.
- It does show that a trained sparse CI network carries truth only through components active on the probed inputs; the untrained 0.998 is a property of a dense readout [concluded].
- Odd detail: the control still reaches 0.95 on affirmative cities and animal classes, and chance on negated families.
- **Sparsity-matched untrained baseline** (by-product; this is the control "per statement keep the untrained network's largest CIs, not selected by truth", so it needs no further run): arm A beats untrained by 1–2 points at matched k (k=2: 0.946 vs 0.926). This is within the seed-to-seed spread of trained runs (fp32 arm-A-like runs 0.970–0.984, this bf16 run 0.991), so not evidence yet [concluded].

**Next control (not pursued: Julian declined it): form-matched (proposed 26-10-05, construction in the LOG entry *proposal: form-matched control dataset*).** One control sentence per tiu statement, built from its subject and object in a frame that mentions both without relating them ("Anna wrote a postcard about Krasnodar and Russia."). Negated statements get "not" frames; several hand-written frames per domain; the tiu split carries over. Built-in leak check: project the control sentences built from true vs false statements onto the tiu truth direction; if they separate, fall back to subject-only frames. Python imports (Julian's idea) were judged less suitable, since they would likely stay off on tiu like the code lines. **No extra arm-A run and no seeds** [decided: Julian, chat transcribed in the LOG: exploration, looking for visible effects]; only differences well beyond ~2 points get read.

## Open

- Nothing. The dataset `data/pile_code/github_lines/v1` was committed unchanged by Julian (`157e7e7`) [verified: `git diff HEAD` empty for it].
- Not pursued (Julian): form-matched control, Python-imports control, second seeds, GPU concurrency test, per-statement k matching for the sparsity-matched baseline.

- See also: [convos/julian/26-10-02_overview_of_goal_SUMMARY.md] — experiment design; holds the open TODO ("similar sentences that carry no truth signal") this topic picks up.
- See also: [convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md] — the probe results (untrained CIs at ceiling) that decide which metrics this baseline can inform; `probe_ci.py` was written there.
- See also: [convos/julian/26-10-02_prepared_datasets_SUMMARY.md] — the tiu datasets and the generator/data convention `pile_code/github_lines` follows.
- See also: [convos/julian/26-10-02_training_run_SUMMARY.md] — e2b (arm A's final settings, fp32 frozen model), the reference run for this control; also lists "run the arms at final settings" as open.
