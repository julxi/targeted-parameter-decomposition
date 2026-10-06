# SUMMARY: tPD truth experiments on the Universal Truthfulness Hyperplane (UTH) data

**Last updated:** 26-10-05 (series ended by Julian after the negative stage 1 result; self-contained summary in `scratch/uth_experiment_summary.md`)

Julian wants a new series of tPD (targeted parameter decomposition) truth experiments on Qwen2.5-7B-Instruct. The target data comes from "On the Universal Truthfulness Hyperplane Inside LLMs" (Liu et al., EMNLP 2024, arXiv:2407.08582; data repo `hkust-nlp/Universal_Truthfulness_Hyperplane`), chosen for its diversity despite noisier data. The series should "recreate their experiment" at the compute scale of the arm A runs (about one H100 hour per run). This follows the arm A experiment on Truth-is-Universal (tiu) statements, which ended without evidence for truth-specific mechanisms (`convos/julian/26-10-05_no_truth_baseline_SUMMARY.md`).

**Status: closed** [decided: Julian, transcribed chat in the LOG, "end this series"]. Bottom line [concluded]: on this setup tPD did not recover truth-related mechanisms that generalise across tasks. The trained CI values carry truth only within the training tasks (0.705); on unseen task categories they are at chance (0.488), worse than an untrained CI network at every matched sparsity. Caveat: one seed, and the decomposition fits this data poorly. Self-contained summary of the whole series: `scratch/uth_experiment_summary.md` (gitignored, local). Not pursued: a better-fitting decomposition (more components per layer); why the trained probe inverts on copa, story_cloze and sciq.

## Key facts

- **Paper** [verified: arXiv HTML, 26-10-05]:
  - 49 datasets in 17 task categories; up to 800 training samples per dataset.
  - Last-token probes (mainly on attention heads) on LLaMA2-7b-chat / Mistral-7b.
  - *Cross-task* evaluation: probes trained on 14 categories are tested on 3 held-out ones (sentence completion, short-answer QA, summarization). They reach only about 70% (Probe-MM 70.47 on LLaMA2-7b).
  - Diversity matters more than quantity.
- **Why this protocol helps us** [concluded]: arm A could not discriminate because every readout sat at the ceiling (0.998). Cross-task accuracy around 0.70 leaves room for trained CIs (causal-importance values) to beat untrained ones. The held-out-task protocol, not the dataset itself, is what creates this room.
- **Local data** [verified 26-10-05]: all 49 datasets are in `scratch/uth_data/` (loader `scratch/uth_data.py`, gitignored).
  - 37,400 train samples; the `vali` files (≤ 5,000 each) are the paper's test sets. Labels are balanced.
  - Lengths range from a median of 9 to 873 Qwen tokens. A 64-token cap keeps 21–23 datasets in 10 of the 17 categories; 128 adds about 10 more.
  - Summarization (cnn_dailymail_re, xsum_re: median 440–849 tokens) fits no affordable training cap. It only matters for testing, which has no cap (see *Plan*).
- **Label artefacts** [verified]: 34% of arithmetic's false answers are non-numeric ("house"), against 0% of the true ones, so a probe can solve them by format. Single odd examples were also seen in capitals, counterfact and hotpot_qa_re; their frequency is unmeasured [assumed: noise].

## Implemented: generator `build_uth.py` and data `data/uth/*/v1` [verified: generator committed by Julian (`93ff44b`); v1 built from it, all manifests carry that commit and the script hash; data not committed yet]

`python spd/experiments/lm/honesty_targeted_decomposition/build_uth.py --version vN` (defaults: cap 128 tokens, coverage 0.8, 200 held-out per training dataset, 1,000 per test dataset, seed 0). It downloads the files from the pinned upstream commit `f509513`. Output `data/uth/<dataset>/<version>/`; manifests carry `role` (`train_task` / `test_task`) and `category`. Test tasks have no `train` split, so the loader refuses them for training [verified]. Record (first line of the dry run's `strategy_qa/v1/train.jsonl`):

```json
{"text": "Question: Can the Palace of Westminster tell time in the dark?\nAnswer: Yes.", "label": true, "dataset": "strategy_qa", "category": "multi_step_reasoning_qa", "n_tokens": 17, "upstream_file": "data/strategy_qa/strategy_qa_probe_train.json", "upstream_index": 0}
```

- **QA format = the paper's** `"Question: {q}\nAnswer: {a}"` [verified: upstream `src/utils.py`]; the scratch loader's space-join was wrong.
- **Upstream label bug** [verified]: story_cloze and definite_pronoun_resolution are built as (right answer = 1, wrong answer = 0) pairs. In about half of the pairs both records carry the same answer (story_cloze `vali` 960 of 1,871; definite_pronoun_resolution `vali` 282 of 564), so one label is wrong. Only intact pairs are kept. story_cloze (a test task) is then clean; definite_pronoun_resolution keeps 134 train records.
- **Other cleaning**: texts with both labels are dropped, duplicates removed, held-out samples that also occur in train dropped. No test-task text occurs in any training file [verified].
- **Dry-run result at cap 128**:
  - training data: 29 datasets in 10 categories, 20,292 train sequences (mean 46.6 tokens), 5,694 held-out;
  - test data: 8 datasets, 7,200 samples;
  - about 20 MB on disk.
  - Skipped for length: anli, imdb, yelp_polarity, multirc, squad, boolq, race, dream, piqa, web_nlg_re, record, cosmos_qa.
- **Data stays in git** despite the missing upstream license [decided: Julian, transcribed chat in the LOG].

## Implemented: stage 0 probe script `probe_uth.py` [verified: CPU smoke test on the real model, 4 samples per dataset; lint/type clean]

`python spd/experiments/lm/honesty_targeted_decomposition/probe_uth.py <out_dir> <data_version> <run_id>...`
- Splits from the manifest roles: fit = train-task train (20,292), tune = train-task held-out (5,694), test = test tasks (7,200).
- Features: residual stream after layers 7/11/15/19/23/27, the 480 CI values per run, mean log-prob; last-token and mean readouts.
- Probes: mass-mean, and logistic regression with C chosen on tune; scores pooled, mean over datasets (the paper's average), per dataset and per category.
- Length-sorted batches of ≤ 4,096 padded tokens. Job: 3.06 M tokens, minutes on an H100 [assumed].
- **Padding check** [verified: CPU, 87 samples batched vs alone]: samples of ≥ 16 tokens are bit-identical however much padding they get; only samples of ≤ 15 tokens differ, by up to 3–5%. That points to a kernel path for very short single sequences, not padding leakage [concluded].
- Outputs and smoke-test scripts in `~/spd_out/26-10-05_uth_experiments/`.

## Stage 0 results [verified: `~/spd_out/26-10-05_uth_experiments/stage0/results.json`; RTX 4090 VM, 26-10-05; one fit each]

Logistic regression, last-token readout, C chosen on the train-task held-out split. Test = mean over the 8 held-out test datasets (standard error roughly ±0.006):
- **residual stream after layer 19: 0.807** (best layer on both tune and test; layers 15 / 23: 0.739 / 0.769);
- **untrained CIs (`s-7fad0c14`): 0.715**;
- mean log-probability: 0.559.

Reading [concluded]:
- **Gate passed**: 0.807 is far below 0.95, so the held-out-task protocol leaves room. On tiu everything was at 0.998.
- **Untrained CIs lose 9 points**, mostly on long-context tasks (summarization 0.583 vs 0.772; hellaswag 0.574 vs 0.789), and match on short ones (copa, sciq, story_cloze). A trained decomposition therefore has room to beat untrained.
- Stronger regularisation (C = 1e-3) would give 0.834 for residual L19. It is not selected, because C is chosen on in-distribution data and choosing on test would be cheating.
- Not comparable to the paper's ~0.70: different model, readout and test cleaning.
- Per-dataset table and the full reading: LOG entry *stage 0 results*.

Practicalities:
- `probe_uth.py` now has two subcommands: `run <out_dir> <data_version> <n_workers> <run_id>...` and `probe <out_dir> <n_workers>`. The latter refits from `features.npz` in parallel; sequential fitting took ~4.5 min per 3,584-dim set. Probes fit on the stored float16 features.
- C grid: {1e-4, 1e-3, 1e-2, 0.1, 1}.
- **scikit-learn is only in the `dev` dependency group.** VMs synced by `spd-vast` lack it; install with `uv sync --group dev`. Proposed: move it to the main dependencies (Julian's call).
- Feature extraction: 7.5 min on the 4090, peak ~20.5 of 24 GB, after cutting `BATCH_TOKENS` to 2,048 and computing log-probs in chunks.

## Implemented: stage 1 code [verified: unit test incl. a mutation check, CPU smoke training on the tiny Qwen2, `make test` (only the known rotgrid failure); not committed]

- **Per-batch padding**: `prompts_dataset.trim_to_loss_positions` cuts every prompt batch after its last loss position (only with a position mask, i.e. `tokens` / `last_k`; with `all` padding is trained on). Batches stay random; there is no sorting. On the uth data: mean width 101.6 of 128 (0.79× target tokens) [verified: 401 batches].
- **Persistent PGD**: sources are sized (batch, `max_seq_len`) for LM tasks (`run_spd.py`) and sliced to each batch's width (`persistent_pgd._slice_to_batch`).
- **`tests/test_prompts_padding.py`**: trimmed and full-width batches give identical ImportanceMinimality, PersistentPGDRecon and UnmaskedRecon losses on a tiny Qwen2 decomposition. The stochastic losses can't be compared exactly (random masks depend on the batch shape).
- **WeightMagnitude fix**: the eval figure crashed on batches of different widths, and it ignored the position mask, so its CI mean and max included padding. That also affected the existing tiu runs' weight-magnitude figures (diluted by ~58% padding positions).
- Not adapted: the multibatch PGD eval (fails loudly with varying widths; unused by our configs).
- **`config_uth_all_tokens.yaml`** = the tiu all-tokens config with label, `max_seq_len: 128` and the 29 train-task datasets changed.
- **`spd-vast` installs the `dev` group** (scikit-learn for the probe scripts) [decided: Julian, transcribed chat in the LOG].
- **Timing trial scripts** in `~/spd_out/26-10-05_uth_experiments/trials/`: arm A vs uth, with and without PPGD, 150 steps each, on one H100. Keep cap 128 if uth costs ≤ ~2.5× arm A.
- The RTX 4090 cannot train (a tPD step needed ~27 GB at tiu sizes); training needs an H100.

## H100 timing trials [verified: `~/spd_out/26-10-05_uth_experiments/trials/h100/`, single trial each, H100 80GB at 700 W]

- Per step, weighted 80% without PPGD and 20% with (as in a full run): arm A 203 ms, uth at cap 128 with trimmed batches 319 ms, i.e. **1.57×**. That is below the agreed ~2.5× limit, so **cap 128 stays** [concluded]. A 5,000-step uth run takes ~27 min of steps plus evals, roughly 35–40 min [assumed].
- Peak GPU memory: uth 34.6 GB without PPGD, 45.3 GB with PPGD (arm A ~26 GB).
- **Two concurrent runs gain nothing** (Julian asked): one uth run already uses 98% of the GPU at ~680 W. Two at once are 6–8% slower than running them one after the other, and with PPGD they peak at 78.7 of 80 GB. Further runs (e.g. a second seed) should run sequentially.

## Stage 1 run `s-d2ded461` [verified: `~/spd_out/26-10-05_uth_experiments/stage1_train/metrics.jsonl`; checkpoint on WandB; single run, seed 0, commit `8e8cc1c`]

`config_uth_all_tokens.yaml` on the H100, 35 min. Final eval vs arm A tiu (`s-bd23f0d1`):

| | uth | arm A |
|---|---|---|
| active components per token | 7.8 | 7.0 |
| target rounded KL | 0.433 | 0.136 |
| all components off | 2.80 | 3.54 |
| all 480 on, no delta (unmasked) | 0.479 | 0.135 |
| Pile L0 / rounded KL | 0.62 / 0.052 | 0.11 / 0.038 |

Reading [concluded]:
- Sparse, but 15% of the components' effect stays unreconstructed (arm A: 4%), and the target KL plateaued after ~1,500 steps.
- Even all 480 components reproduce the layers poorly, so the limit is capacity or training, not sparsity. Possible causes: C = 96 is too few for 29 diverse datasets, ~4 passes over the data, settings tuned on tiu [assumed].
- Consequence: a null probe result would be weak evidence against tPD. A larger C or more steps would be the follow-up.

**Matched-sparsity analysis**: `probe_uth.py sparsity <out_dir> <n_workers> <run_id>...` (method from the tiu follow-up `~/spd_out/26-10-05_no_truth_baseline/followup.py`, now in the repo). It reports active-component counts and probes on each sample's top-k CIs (k = 1–50) and on the binary on/off pattern, cross-task.

## Stage 1 probe results [verified: `~/spd_out/26-10-05_uth_experiments/stage1/results.json`, `sparsity.json`; one run each]

Cross-task test (mean over the 8 test datasets), last-token logistic regression; in-distribution held-out (tune) in brackets:
- residual L19: 0.806 (0.849);
- untrained CIs: 0.717 (0.793);
- **trained CIs `s-d2ded461`: 0.488 (0.705)**;
- log-probability: 0.561.

**Matched sparsity**: the trained CIs are worse than the untrained ones at every k ≥ 2 (k = 5: 0.504 vs 0.585; k = 10: 0.499 vs 0.593), at chance throughout. Trained: median 12 active components per sample (untrained ~222); 8% of test samples have none (0.4% on fit).

Below chance on copa (0.285), story_cloze (0.401) and sciq (0.438), i.e. systematically inverted; cause unknown [assumed: component features whose relation to the label flips between tasks].

Reading [concluded]:
- **The success criterion is not met, and the effect goes the opposite way.** The trained CI network's truth signal is task-specific: 0.705 in-distribution, chance on new task categories. The cross-task protocol exposed what tiu's in-distribution test (0.991 vs 0.998) could not.
- Caveat: the decomposition fits poorly (see the stage 1 run), so this concerns *this* decomposition. A better-fitting one is untested. That trained CIs lose information even in-distribution argues against expecting much from it.
- Re-extraction reproduced stage 0 to within 0.2 points [verified].

## Plan

Full consolidated design: LOG entry *design consolidated after the second round of comments*.

**Decided** [decided: Julian's C: comments in the LOG]:
- **Settings as arm A**: Qwen2.5-7B-Instruct, `down_proj` of layers 15–19, C = 96, batch 16, 5k steps, LR 5e-4, bf16 frozen model, Pile non-target stream, loss on all real tokens, no chat template, one seed (a second if the result is borderline).
- **Test set = the paper's**: 8 datasets in 3 held-out categories, up to 1,000 samples each from the `vali` files (copa has only 200), uncapped in length; never used in tPD training:
  - sentence completion: copa, hellaswag, story_cloze;
  - short-answer QA: nq_re, triva_qa_re, sciq;
  - summarization: cnn_dailymail_re, xsum_re.
  - This works because the length cap applies only to tPD's training data; test tasks only go through forward passes for the probes. Caveat: test sequences (up to ~3,000 tokens) are much longer than anything tPD trained on; all readouts face the same shift.
  - We first proposed our own short held-out categories; the switch keeps the paper's protocol and all training categories for diversity.
- **Training data** = the paper's 14 training categories:
  - a dataset enters if ≥ 80% of its samples fit the cap, and its longer samples are dropped;
  - all samples are kept, with no per-dataset balancing;
  - arithmetic stays, despite its format shortcut; probe accuracy is reported per training dataset so any reliance on the shortcut shows.
  - Counts [verified: token counts, 26-10-05]: cap 64 gives 7 categories, 19 datasets, 12,999 sequences (~6 passes in the 80,000 sequences of a run); cap 128 gives 10 categories (adds NLI, topic classification, structure-to-text), 29 datasets, 21,041 sequences (~4 passes). Arm A: ~12 passes. Statement fact checking is 4,600 sequences (35% / 22%).
- **Eval split**: each training-category dataset gets a `test` split from its `vali` file, within the cap. It serves the built-in training eval and the probe tuning.
- **Cap 64 vs 128**: decided by a timing trial on the H100 (fixed padding at both caps, 150 steps each).
- **Padding**: each random batch is padded only to its longest sample. Julian rejected sorting or pooling by length, because it changes the training order.
  - Saving: 15% of target tokens at cap 64, 22% at cap 128 [verified: simulation on the real lengths]. Sorting would have given 0.40–0.45×.
  - Expected to leave training unchanged apart from speed [concluded: right padding plus causal attention, and every loss in the config applies the position mask; not tested]. Checked by an exact unit test (the same batch padded to two lengths gives equal losses).
  - Code: unpadded storage and a collate in the loader; persistent PGD masks (fixed shape (batch, seq_len, C), `spd/persistent_pgd.py`) sliced to the batch length.
- The cost estimate is still open: the first one (1.8× arm A at cap 64) ignored the Pile batch and padding and was withdrawn. The trial measures it.

**Probe protocol** [concluded; Julian agreed after asking what "probe" referred to]:
- logistic regression and mass-mean on three feature sets: the residual stream (layer 19 plus a coarse layer sweep), untrained CIs (`s-7fad0c14`), trained CIs;
- last-token readout primary, mean over tokens secondary;
- any tuning (regularisation, layer) only on the training categories' held-out split, never on the test datasets.

**Order of work:**
1. Generator for `data/uth/`, committed before building.
2. Stage 0 code: a probing length limit in `probe_ci.py` separate from the run's `max_seq_len`; cross-task split.
3. Per-batch padding plus its unit test.
4. H100 rental: stage 0 features and probes (gate: continue only if the residual stream is well below 0.95 on the test datasets), timing trial, stage 1 tPD run at the chosen cap, stage 1 probes.

**Success criterion for stage 1, fixed in advance** [concluded]: trained CIs beat untrained CIs cross-task at matched sparsity by clearly more than the run-to-run spread (tiu: 0.970–0.991). Otherwise the series ends.

**Caveat** [concluded]: tPD's importance minimality may favour task-specific components over shared ones; diversity helping residual probes does not imply it helps tPD. Julian: "maybe. We will see."
- **Stage 0 runs on a rented VM** (feature extraction and probes), not the laptop [decided: Julian, transcribed chat in the LOG]; the agent says when the code is ready, Julian rents.

- See also: [convos/julian/26-10-05_no_truth_baseline_SUMMARY.md] — the arm A experiment and its code control; its untrained bf16 checkpoint `s-7fad0c14` and the matched-sparsity method carry over.
- See also: [convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md] — `probe_ci.py`, the probe setup to be reused with a cross-task split.
- See also: [convos/julian/26-10-02_overview_of_goal_SUMMARY.md] — the original motivation (an unreplicated result that used UTH domains) and the deleted legacy `prompts_liu` data, which came from the same UTH repo.
- See also: [convos/julian/26-10-02_prepared_datasets_SUMMARY.md] — the `data/` + generator convention a `data/uth/` family would follow.
