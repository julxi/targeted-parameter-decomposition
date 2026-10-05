I would like to add another baseline for the simple truth experiments (I'm saying simple because it's from the very unambigous truth is universal paper). I would like to use some non-target data that has little truth feature.

---

26-10-05, Claude Opus 5.5 — *reading of the request, design options, open questions*

I have read INDEX.md and the SUMMARYs of the overview-of-goal, test-accuracy, prepared-datasets and training-run topics, the last-token arm config, and one statement from each of the 20 tiu datasets. I have not read any LOG in full, and no code beyond the config.

File renamed from `convos/julian/new.md` (untracked, so plain `mv`).

**Where this fits.** The overview topic left an open TODO from Julian: "test with similar sentences that carry no truth signal (data design open; a random-label control task is one option)" (`convos/julian/26-10-02_overview_of_goal_SUMMARY.md`, *Follow-up after results*). I read this request as that TODO [assumed].

**Two readings of "non-target data that has little truth feature".**

1. **Control decomposition** (my default reading): run tPD (targeted parameter decomposition) with a *target* dataset that has little truth content, keeping everything else as in arm A (the all-tokens arm). Then probe its CIs (causal-importance values) for truth on the tiu (Truth-is-Universal) test split, exactly like the real arms. "Non-target" here would mean "not the truth data", not the Pile stream.
   The question it answers: do the probe results depend on decomposing *truth* data, or does any equally trained and equally sparse tPD give them?
   C: yes, that what I meant.
   Answer (26-10-05, Claude Opus 5.5): Noted: reading 1, a control decomposition [decided: this C: comment]. Reading 2 is dropped.
2. **Swap the non-target stream:** keep tiu as the target, replace the Pile (the general-text stream on which the catch-all Δ (delta) component is on) with low-truth text. I don't see how this works as a baseline. It changes the method itself, and I can't say which way it should move the probe results [concluded: under tPD, a mechanism used on both streams must be active on both, so making the non-target stream truth-free changes which mechanisms pay the importance-minimality cost on non-target data; the direction of the effect on truth probes is not predictable from the method]. If this is what you meant, tell me what result would count as evidence for what.

**Critique: what such a baseline can and cannot show.** Plain probe accuracy is already at ceiling without training: untrained CIs reach 0.998 test accuracy, the same as the residual stream after layer 19, while trained arm A runs reach 0.970–0.984 (`convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md`, *Results*). A control decomposition will almost certainly also land around 0.97–1.0 [concluded: its CI network is the same kind of nonlinear readout of the same activations, and even the untrained one hits the ceiling]. So comparing *plain* accuracy against it would show nothing. It becomes informative only with metrics that depend on *which* components are active:
- the on/off pattern of the active components (arm A: 0.916–0.964);
- the top-k most truth-related components (arm A top-5: 0.877–0.953);
- generalisation (affirmative → negated, leave-one-domain-out), not yet run on any checkpoint.
C: maybe you are right, but I have to check this.
Answer (26-10-05, Claude Opus 5.5): Fair. The control run checks it anyway: `probe_ci.py` reports plain accuracy for every run, so the control's plain accuracy will appear next to arm A's and the untrained baseline's. My prediction (≥ 0.97) stays a prediction until then.
OUTDATED (26-10-05): the prediction above was wrong. The code control reached 0.654, mostly because its CI network was nearly inactive on tiu (1.4 active components per statement); see the entry *probe results: the control's CIs are mostly off on tiu*. The `[concluded]` tag on it should have been `[assumed]`.

**Where this beats the untrained baseline.** The control decomposition is trained and sparse, just like the real arm. Its only difference is the target data. So it controls for "sparsity plus training" better than the proposed sparsity-matched untrained baseline does (untrained network, each statement keeps its own ~10 largest CIs) [concluded]. OUTDATED (26-10-05): it did not; the control turned out far sparser on tiu than arm A (1.4 vs 9.3 active components per statement), see *probe results: the control's CIs are mostly off on tiu*. I would still do the sparsity-matched baseline: it is cheap (laptop, existing `features.npz`), it answers a different question (tPD's choice of components vs choice by size), and its result tells us whether the GPU run is worth it.

One complication: the control's CI functions see tiu statements as *off-target* inputs. Their sparsity on tiu may differ a lot from arm A's 8.5–11.4 active components per statement. We need to report the L0 (number of active components) on tiu for both, and compare at matched sparsity where they differ [concluded].

**"Little truth feature" must be measured, not assumed.** Qwen evaluates the plausibility of anything that reads like a factual claim, including claims about made-up entities [assumed: commonly reported in the truth-probe literature, not checked here]. Proposed check: take the residual-stream truth probe (layer 19, trained on tiu) and project the control data's activations onto its direction. A control set with little truth feature should have a spread along that direction much smaller than the true/false separation on tiu. This is a cheap laptop job.

**Candidate control datasets.** All keep tiu's subjects, sentence length and final token where possible, so that token identity and domain are matched:

| option | example | truth feature? | form match |
|---|---|---|---|
| (a) claim removed, entities kept | "The city of Krasnodar." / "Fifty-one and fifty-two." | low [assumed] | same entities and `.`, shorter, not a full sentence |
| (b) non-assertive framing | "Anna would like to visit the city of Krasnodar." | low-medium: carries an implicit claim (Krasnodar is a city) | full sentence, same `.` |
| (c) fictional subjects | "The city of Velmora is in Russia." | medium: plausibility gets evaluated | exact template |
| (d) questions | "Is the city of Krasnodar in Russia?" | probably high: the model likely computes the answer | last token `?` instead of `.` |
| (e) length-matched Pile sentences | — | mixed, uncontrolled | no domain match |

My recommendation: **(a) or (b), generated per domain from the tiu subjects, checked with the projection test above before training** [assumed: preference, not tested]. (d) is likely to carry truth. (e) breaks the domain and token match that makes the comparison clean.
C: I was thinking about something completely unrelated. Maybe coding maybe soemthing else?
Answer (26-10-05, Claude Opus 5.5): OK, I'll go with **code**. It is clearly unrelated, it contains almost no factual claims about the world [assumed], and the Pile has a GitHub subset in a file that neither the training stream nor the eval stream reads (details in the next entry). Two things to keep in mind:
- An unrelated control answers a coarser question than a matched one. If arm A's components beat the code control on the component-specific metrics, that shows the *target* matters. It does not show that *truth* matters rather than "short statements about cities, elements, animals, numbers". A form-matched control (option (a) or (b)) would separate those. I suggest: code first; a matched control only if code shows a difference [concluded].
- On tiu statements, the code control's CI functions read inputs from outside their training distribution. Their sparsity there is unpredictable (nearly all off, or dense). If it ends up far from arm A's ~10 active components per statement, the on/off-pattern comparison is unfair, and we'd need the matched-sparsity readout (keep each statement's top ~10 CIs) for both runs.

**Practicalities if we go ahead.**
- Generator under `spd/experiments/lm/honesty_targeted_decomposition/`, data under `data/<family>/v1/`, following the prepared-datasets convention. Commit the generator before building.
- One config = arm A's config (e2b settings: batch 16, 5,000 steps, LR 5e-4, bf16 frozen model) with the target swapped. One H100 run (~28 min projected for e2b's settings), then probing on the laptop.

**Questions for Julian:**
1. Reading 1 (control decomposition) or reading 2 (swap the Pile stream)?
2. Which control data: (a), (b), something else?
3. Is the projection test an acceptable definition of "little truth feature"?
C: ok, let's try this out.
Answer (26-10-05, Claude Opus 5.5): Will do, before the GPU run. Caveat for code: its activations are far from tiu's, so a large spread along the truth direction could reflect that distance rather than truth content. I'll report the spread in units of tiu's true/false separation and read it as a rough sanity check, not as proof.
4. Sparsity-matched untrained baseline first (cheap, laptop), or straight to the control run?
C: straight to the control run
Answer (26-10-05, Claude Opus 5.5): Noted [decided: this C: comment]; the sparsity-matched untrained baseline is deferred.

---

26-10-05, Claude Opus 5.5 — *code-lines control: generator, config, probe-script change*

Following Julian's answers above (control decomposition; unrelated target such as code; projection test as the criterion; straight to the control run). I have read `docs/CODING.md`, `data/README.md`, `build_tiu.py`, `prepared_datasets.py` and `probe_ci.py` in full.

**Dataset: `pile_code/github_lines`** (generator written, a dry run done, **not built into `data/` yet**).
- Generator: `spd/experiments/lm/honesty_targeted_decomposition/build_pile_code.py --version v1`. Output: `data/pile_code/github_lines/v1/` (train, test, manifest).
- Source: `test.jsonl.zst` of `monology/pile-uncopyrighted`, pinned to revision `3be90335…`. The tiu arms' Pile stream reads `train/*` (training) and `val.jsonl.zst` (eval), never the test file, so target and non-target documents are disjoint [verified: repo file listing; configs' `eval_data_files`]. Only documents with `pile_set_name == "Github"`. In the first 20,000 documents of the file, 1,994 were GitHub [verified: stream count]; the whole file has 18,195 GitHub documents [verified: dry run read them all; `max_docs` 30,000 was never reached].
- One record = one stripped line of code. Filter (heuristic): ASCII only; no comment-marker prefix (`//`, `/*`, `*`, `# `, `--`, …); at least 2 code-symbol characters (`(){}[];=<>&|_$@\`); no run of 4+ plain words, to drop English prose (comments, license lines, which make factual claims). Lines are deduplicated across splits, at most 10 candidate lines per document, and the split is by document (20% test, hash of document index).
- **Length-matched to tiu v1, per split:** the same number of lines of each Qwen-token length (6–20) as tiu has statements. So the split sizes are equal (6,560 train / 1,794 test), and so is the number of trained positions per step [verified: dry-run pools hold ≥ 2.4× the needed count at every length].
- **Dry run** [verified, 26-10-05, `~/spd_out/26-10-05_no_truth_baseline/dry_*`]: train lines come from 4,894 documents, at most 5 per document. Inspected 60 random kept lines by eye: all code (C/C++, Java, Go, JS, Python, XML/HTML, CSS, LaTeX, …). A few carry short English fragments (`'<h3>The TOML Mode</h3>'`); accepted. Dropped examples per rule also checked: comments and license prose land in `comment`/`prose_run`; `prose_run` also drops some real code lines with 4 plain words (e.g. `static constexpr auto apply(...)`), harmless. Line outcomes (all lines of the 18,195 documents): passed the filters 605,095; few symbols 888,525; comment 387,241; wrong length 362,708; duplicate 158,283; prose run 64,595; non-ASCII 49,161.

**Config:** `spd/experiments/lm/honesty_targeted_decomposition/config_code_control.yaml` = `config_truth_all_tokens.yaml` with label `code_control_all_tokens` and target `data/pile_code/github_lines/v1` [verified: `diff`; it parses as a `Config`]. So it is arm A's final settings: batch 16, 5,000 steps, LR 5e-4, `loss_positions: tokens`, final checkpoint only, frozen model in bf16.

**`probe_ci.py` change:** runs no longer have to share the target datasets. The probe data is the tiu data of the *first* run id given; a non-tiu run given first fails loudly (the existing `_true`/`_false` directory-name assertion). `results.json` now records each run's target. Linters clean (ruff, basedpyright).

**A dtype problem for the comparison** [verified: e2b's config has no `pretrained_model_dtype`, so it loaded fp32]. `probe_ci.py` serves all runs from one forward pass, so it requires one shared frozen-model dtype. All existing tiu checkpoints (run 1, e2b, e2c, e3a, e3b, the untrained `s-286aa6a9`) loaded fp32. The control config, like all current configs, loads bf16. No bf16 tiu checkpoint exists. Options:
- (i) Train the control with `pretrained_model_dtype: float32`. It then matches e2b exactly (e2b = arm A at its final settings) and needs one GPU run. Cost: fp32 is 10–20% slower per pass [verified in the training-run topic].
- (ii) Keep bf16 and also run arm A in bf16 on the same rental. Two runs (~1 h); these would be the first runs at the final configs, which the training-run topic lists as open anyway.
- (iii) Probe in two invocations and compare across dtypes. The CI inputs differ slightly between dtypes, which would muddy a control comparison.

My recommendation: **(ii) if the rental also runs the other arms; otherwise (i)** [assumed: preference].

**Steps left before the GPU run:**
1. Julian commits the generator (the lesson from the prepared-datasets topic: commit before building, so the manifest's `repo_commit` contains it).
2. Build v1 into `data/` and commit the data.
3. Projection check (running now on the dry-run test lines; results in the next entry).
4. Rent an H100 and launch.

---

26-10-05, Claude Opus 5.5 — *projection check: code carries little of the tiu truth feature*

Ran `~/spd_out/26-10-05_no_truth_baseline/projection_check.py dry_test.jsonl` on the laptop CPU (exit 0; output `projection_report.json`, `projection_features.npz`, `projection_run.log`; about 30 min, from file timestamps 10:15 to 10:44). Setup: Qwen2.5-7B-Instruct, bf16, layers 0–19 only. The loader reported only layers 20–27 as unused and no weight as newly initialised [verified: log]. Readouts: the residual stream after layer 19, at the last real token and averaged over real tokens. Truth directions were fit on tiu train (6,560): the mass-mean direction (mean true − mean false), and a standardised logistic regression mapped back to activation space. Projected: tiu test (1,794) and the dry-run code test lines (1,794).

Units: d = tiu test separation along the direction (mean true − mean false). "Within" = tiu within-class std along the direction. The reference row gives the code std / within ratio along 100 random directions, i.e. how much code spreads along a direction that has nothing to do with truth.

| readout / direction | tiu test acc | code std / d | code mean − midpoint, / d | code std / within | same ratio, random directions (5%–95%) |
|---|---|---|---|---|---|
| last / mass-mean | 0.981 | 0.099 | +0.224 | 0.47 | 1.14–2.62 (median 1.83) |
| last / logreg | 0.998 | 0.096 | −0.122 | 0.64 | 1.14–2.62 (median 1.83) |
| mean / logreg | 0.997 | 0.213 | −0.015 | 1.04 | 0.95–1.41 (median 1.05) |
| mean / mass-mean | 0.524 | — | — | — | (unusable: this direction doesn't separate tiu) |

[verified: `projection_report.json`; single run, no resampling]

Reading [concluded]:
- **Code spreads very little along the truth directions.** At the last token, its spread is about a tenth of the true/false separation. That is *narrower* than tiu's own within-class spread, and well below what code shows along random directions (the 5% quantile is 1.14). With the mean readout, the spread is about a fifth of the separation, the same as along a random direction. So by the agreed criterion (spread small compared with the true/false separation), code passes.
- **The offsets don't amount to a consistent "code reads as true/false":** at the last token, code sits +0.22 d from the midpoint on the mass-mean direction (98% of lines on the "true" side), but −0.12 d on the logistic-regression direction (90% on the "false" side). The sign depends on the direction, which fits an offset from the shift in input distribution rather than truth content. I did not compute mean offsets along random directions, so this reading is not tested.
- **Limit:** these are tiu's truth directions. The check says code does not carry *tiu's* truth feature, which is what the probes on tiu read. It says nothing about whether code has its own "correctness" feature elsewhere.
- The mass-mean direction on the mean readout separates tiu only at chance (0.524). Averaging over tokens mixes in token-identity variance that dominates the class-mean difference. The logistic regression direction is fine (0.997).

The dry run uses the same code (apart from formatting and a renamed stats key), seeds and pinned Pile revision as the real build will, so v1 should reproduce these lines exactly [assumed: deterministic streaming order]. After building I'll check that v1's `test.jsonl` texts equal `dry_test.jsonl`.

Still open: the dtype question to Julian (previous entry); commit the generator → build → GPU run.

---

26-10-05, Claude Opus 5.5 — *v1 built from the committed generator*

Julian committed the generator, config and `probe_ci.py` change (`884052a`, "preparaitons for no-truth control"). His own run of the build looked stuck and he stopped it after a minute. That was expected behaviour: the script streams the whole Pile test file (18,195 GitHub documents among all the others) and prints nothing until it is done. The interrupted run wrote nothing (`data/pile_code` did not exist) [verified].

Built `data/pile_code/github_lines/v1` with `python spd/experiments/lm/honesty_targeted_decomposition/build_pile_code.py --version v1`: 2 min 28 s, exit 0, log in `~/spd_out/26-10-05_no_truth_baseline/build_v1.log`. Checks [verified, 26-10-05]:
- Manifest `repo_commit` = `884052a` = HEAD, and `script_sha256` = the hash of the committed script.
- Train and test texts are identical, line by line, to the dry run's. So the projection check of the previous entry applies to v1's test split.
- `load_prepared_datasets` (the training loader, with its sha256 check) loads 6,560 train / 1,794 test lines, and the Qwen-token length histogram of each split equals tiu v1's.

Possible improvement, not made (it would need another commit, and the script has done its job): a progress line every few thousand documents, so a build doesn't look hung.

Next: Julian commits the data; the dtype question (fp32 to match e2b, or bf16 plus an arm-A bf16 run) decides the GPU run.

---

26-10-05, Claude Opus 5.5 — *bf16 decision; untrained baseline regenerated; both runs launched on the H100*

**Precision of the frozen model.** I asked whether to run the control in fp32 (to match e2b) or bf16. Julian replied:

C: (transcribed from chat) So only e2b is in fp32? Then I would rather replace e2b

That rested on a misunderstanding: *every* existing tiu checkpoint (run 1, e2b, e2c, e3a, e3b, untrained `s-286aa6a9`) loaded fp32; only the configs had been switched to bf16. I corrected this in chat and laid out the bf16 route: arm A and the control in bf16 on one rental, plus regenerating the untrained baseline in bf16 (`probe_ci.py` needs one shared dtype). Julian then wrote:

C: (transcribed from chat) Ok, the vm is up. Can you do the two experiments on it?

I take that as choosing the bf16 route [decided: the two transcribed messages above]. The old fp32 runs stay as they are; the new arm-A run becomes the reference for this comparison.

Julian also asked whether the uncommitted dataset is a problem. Not for the runs: the instance has a copy with identical sha256 hashes (checked), and the loader checks every split file against its manifest before training. It matters for provenance, since the runs record `data/pile_code/github_lines/v1` and repo commit `884052a`, which lacks the data. So commit the files unchanged, and don't rebuild v1.

**Untrained baseline in bf16:** `s-7fad0c14`, generated on the laptop CPU with `config_truth_untrained.yaml` (steps 0, `pretrained_model_dtype: bfloat16` [verified: WandB run config]); `model_0.pth` is on WandB [verified: WandB API, state finished]. Log in `~/spd_out/26-10-05_no_truth_baseline/untrained_bf16.log`.

**GPU runs** (vast.ai H100 80GB HBM3, power limit 575 W [verified: `nvidia-smi`]; SM clock 1815 MHz under load, so not throttled):
- The repo on the instance is at `884052a`, plus a copy of the local working tree, including the uncommitted `data/pile_code/` with identical hashes [verified].
- `~/spd_out/26-10-05_no_truth_baseline/run_two.sh` (copied to `/root/run_two.sh`), launched detached from a login shell at 10:28:58 UTC. It runs `config_truth_all_tokens.yaml` (arm A) and then `config_code_control.yaml`, logs in `/root/runs/`.
- Arm A: run `s-bd23f0d1`, ~3.2 it/s in the first 120 steps.
- **Bug in the runner:** the `exit=` field in `/root/runs/status.log` always shows 0, because `$?` is read after a `$(date)` command substitution resets it [verified: local bash test]. I did not edit the script while it was running. Success is judged from each run's log and from WandB instead.
- WandB storage: 3 new checkpoints at ~238 MB each (the size measured for the tiu arms in the WandB-storage topic); usage was 1.50 GB real after the 26-10-03 cleanup, so well under the 5 GB cap [assumed: no large uploads since].

---

26-10-05, Claude Opus 5.5 — *arm-A run slower than e2a: utilisation question*

Julian asked why GPU utilisation is low (I had reported 43%), given that earlier sessions had made sure it was high, and whether the dtype change is the cause.

**What was established earlier** (training-run topic; `~/spd_out/26-10-02_training_run/trials/*.smi`): we never optimised `nvidia-smi` utilisation itself. We optimised throughput (samples/s) by raising the batch from 4 to 16. Under load on the 26-10-02 instance (700 W limit, fp32 frozen model), the batch-16 throughput trials averaged 69% (no PPGD, 31 samples) and 73% (PPGD, 44 samples) utilisation at ~450 W [verified: computed from the `.smi` files, samples with > 20 GB GPU memory in use, which includes model loading]. e2a, batch 16, ran at 4.24 steps/s before PPGD there.

**Now** (10:36 UTC, arm A `s-bd23f0d1`, batch 16, bf16 frozen model, step ~600): 3.2 steps/s; `nvidia-smi` 42–74% over 6 one-second samples, ~390–410 W of 575 W, SM clock 1935–1980 MHz [verified]. So about **25% slower per step** than e2a, with less GPU activity and power. Training process: 136% CPU (main thread 67%, autograd thread 42%). The host has 192 cores (Xeon Platinum 8568Y+), 85% idle, load average 16–20, so other tenants are on it [verified: `top`, `lscpu`].

**Cause: unknown.** Two candidates:
- **The machine.** Different host, different CPU, shared with other tenants. Per `docs/CODING.md`, wall times are only comparable on the same machine.
- **The bf16 frozen model.** The 26-10-02 micro-benchmark measured bf16 loading as 10–20% *faster* per forward+backward pass. So bf16 should not slow things down, but that benchmark was a single pass, not a full training step (which also contains the fp32 components and CI network and the KL losses) [assumed: no reason found in code for a slowdown; not tested].

**Test proposed:** after both runs finish, on this same instance, two 150-step trials of the all-tokens config with `pretrained_model_dtype` float32 vs bfloat16, timed from tqdm as in the 26-10-02 throughput trial (`sync_checkpoints_to_wandb: false`). About 10 min of GPU time. This settles whether bf16 helps or hurts training throughput, which matters for all later runs. The current runs are not affected in their results, only in wall time (~35 min each instead of ~28).

---

26-10-05, Claude Opus 5.5 — *dtype throughput trial queued*

Julian approved the fp32-vs-bf16 throughput trial ("OK, let's do it then", chat).

**Setup:** `~/spd_out/26-10-05_no_truth_baseline/make_dtype_trials.py` and `run_dtype_trials.sh`, copied to `/root/` on the instance. Same design as the 26-10-02 throughput trial (`~/spd_out/26-10-02_training_run/scripts/make_trial_configs.py`):
- the all-tokens config, batch 16, 150 steps;
- no WandB, no evals or checkpoints during the timed steps;
- PPGD either active from step 0 or removed;
- frozen model in float32 or bfloat16.

Four trials, alternating dtypes to limit drift (bf16 no-PPGD, fp32 no-PPGD, bf16 PPGD, fp32 PPGD). `nvidia-smi` is sampled every 2 s during each trial. Changes from 2026-10-02: no `HF_HUB_OFFLINE=1` (it crashes runs, per `docs/CODING.md`), and the `$?` bug of `run_two.sh` is fixed (`rc=$?` captured first). The runner waits until `/root/runs/status.log` says `ALL DONE`, so it does not compete with the two real runs for the GPU. Outputs go to `/root/trials_dtype/`.

Launched 10:43 UTC (process waiting, verified). My local ssh call for the launch exited with "Terminated" (143). Nothing on the instance was affected: arm A was at step 2,065 at 3.36 it/s, and the runner was alive [verified: `ps`, tqdm].

Measure: steps/s from tqdm between steps 30 and 145, as on 2026-10-02.

---

26-10-05, Claude Opus 5.5 — *arm A (bf16) finished; matches e2b within noise*

Arm A `s-bd23f0d1` ended at 11:07:56 UTC (started 10:28:58, so 39 min including startup and evals), with no traceback in the log; `model_5000.pth` (238 MB) is on WandB, run state finished [verified: WandB API]. Copied `final_config.yaml`, `metrics.jsonl`, `run_metadata.json` and the training log to `~/spd_out/26-10-05_no_truth_baseline/s-bd23f0d1/`. The control `s-b6cce5de` started at 11:07:56 and runs at ~3.4 it/s.

Final eval (step 5000), compared with e2b (the same settings, fp32 frozen model, 26-10-02/03 instance). KL is in nats per token; L0 = active components per token, of 480:

| metric | e2b fp32 | arm A bf16 |
|---|---|---|
| target L0 | 6.19 | 7.01 |
| target rounded KL | 0.144 | 0.136 |
| target stochastic KL | 0.110 | 0.106 |
| target unmasked KL | 0.139 | 0.135 |
| PGD recon KL | 0.357 | 0.388 |
| Pile L0 | 0.097 | 0.109 |
| Pile rounded KL | 0.038 | 0.038 |

[verified: both `metrics.jsonl`, single seed each]

Reading [concluded]: equivalent within the run-to-run noise established in the training-run topic (up to 10% on target metrics, 15–25% on PGD/Pile KL). The largest difference is target L0 (+13%, just above the 10% band), with a 5% better target KL. This is consistent with an ordinary seed/hardware difference; one run each cannot tell whether bf16 shifts sparsity. bf16 does not visibly hurt the decomposition.

---

26-10-05, Claude Opus 5.5 — *control finished; it is a much smaller decomposition than arm A*

Control `s-b6cce5de` ended 11:46:07 UTC (38 min), with no traceback; `model_5000.pth` (238 MB) is on WandB; the run config shows `pretrained_model_dtype: bfloat16` and target `data/pile_code/github_lines/v1` [verified: WandB API]. Files copied to `~/spd_out/26-10-05_no_truth_baseline/s-b6cce5de/`.

Final eval (step 5000). "Target" means each run's own target, so tiu for arm A and code lines for the control; the rows are not like-for-like:

| metric | arm A `s-bd23f0d1` | control `s-b6cce5de` |
|---|---|---|
| target L0 (active components per token, of 480) | 7.01 | 2.24 |
| target rounded KL | 0.136 | 0.315 |
| target stochastic KL | 0.106 | 0.291 |
| target unmasked KL | 0.135 | 0.359 |
| target delta-only KL (all components off) | 3.54 | 0.41 |
| PGD recon KL | 0.388 | 0.570 |
| Pile L0 | 0.109 | 0.474 |
| Pile rounded KL | 0.038 | 0.053 |

[verified: both `metrics.jsonl`]

Reading [concluded unless marked]:
- **The five decomposed `down_proj` layers (15–19) matter far less for the next-token predictions on code lines than on tiu statements:** switching all components off costs 0.41 nats/token on code against 3.54 on tiu. So the control had much less to decompose, and it ended up with ~2 active components per token instead of ~7. Its reconstruction is *worse* (rounded KL 0.31 vs 0.14), even though there is less to reconstruct. [assumed: plausible causes are that single code lines without context are hard to predict, so the KL is spread over many low-probability tokens, or that the untuned importance-minimality coefficient is too strong for a target with weak signal; not tested]
- **Its components are more active on the Pile** (0.47 vs 0.11 per token). This is expected, since the Pile contains code and the control's components serve code-like text [assumed].
- **Consequence for the probe comparison:** the control is *not* "equally sparse and equally trained" in the sense the design hoped for. It is a smaller decomposition with weaker reconstruction. Its sparsity on tiu inputs is still unknown; `probe_ci.py` will show it. If its CIs on tiu are nearly all off, a low probe accuracy would reflect "no active components" rather than "no truth structure", and the matched-sparsity readout becomes necessary.

Probe run started on the laptop: `probe_ci.py ~/spd_out/26-10-05_no_truth_baseline/probe s-bd23f0d1 s-b6cce5de s-7fad0c14` (all bf16). The dtype trials started on the instance once the two runs had finished.

---

26-10-05, Claude Opus 5.5 — *dtype trial result: bf16 is faster; the slowdown is the machine*

The four trials ran 11:46–11:57 UTC, all exit 0, no traceback. Files copied to `~/spd_out/26-10-05_no_truth_baseline/trials_dtype/`. Steps/s comes from tqdm between steps 30 and 145 (tqdm prints whole seconds, so ±3% over 34–75 s windows). `nvidia-smi` sampled every 2 s; "under load" = samples with > 20 GB in use, which includes model loading.

| trial (batch 16, this instance) | ms/step | util mean | power mean | peak GPU memory |
|---|---|---|---|---|
| bf16, no PPGD | 296 | 40% | 317 W | 27.2 GB |
| fp32, no PPGD | 322 | 53% | 364 W | 48.8 GB |
| bf16, PPGD | 591 | 38% | 306 W | 27.6 GB |
| fp32, PPGD | 652 | 53% | 345 W | 54.4 GB |

[verified: trial logs and `.smi` files; one trial per cell]

Reading [concluded]:
- **bf16 is 8–10% faster per step than fp32 and halves GPU memory.** So the dtype change is not the cause of arm A's slowdown, and keeping bf16 is right.
- **The instance is the cause.** fp32 without PPGD takes 322 ms/step here against 235 ms on the 26-10-02 instance (26-10-02 throughput table); with PPGD, 652 ms against 409 ms. Same config apart from eval settings, which are outside the timed window. This is ~37% / ~59% slower on a different host. The mechanism is not established [assumed: CPU-side kernel-launch overhead on a shared host (load average 16–20 from other tenants), since GPU clocks were at full speed and utilisation was low].
- **`nvidia-smi` utilisation is a misleading target here:** bf16 shows *lower* utilisation than fp32 while being faster. Each step gives the GPU less work per kernel, while the CPU-side overhead stays the same. Steps/s is the measure to watch.
- Practical: at batch 16, a full 5k-step run takes ~39 min on this instance. Faster hosts exist (26-10-02). `spd-vast` can't select on CPU or host load; if wall time matters, a short tqdm check after renting would show a slow host early.

The instance holds nothing that isn't saved elsewhere: both checkpoints are on WandB [verified], and metrics, logs and trial outputs are copied to `~/spd_out/26-10-05_no_truth_baseline/`. It can be destroyed.

---

C: (transcribed from chat) Maybe the code lines in control are just a bit too random. Maybe we can use other data that gives more active components. I was thinking about python imports for example

26-10-05, Claude Opus 5.5 — *Python imports as the control target: assessment*

(Before this, in chat: Julian asked whether the under-used GPU could do more. I proposed running two trainings concurrently (bf16 peaks at ~27 GB), testing that in 5 min, and using the slots for second seeds of arm A and the control. No answer yet; it is unclear whether the VM is still up.)

**On "too random"** [concluded, partly assumed]: the control's numbers support it. On code lines, switching off all components of layers 15–19 costs only 0.41 nats/token (tiu: 3.54). So these layers carry little of what the model does on isolated code lines, and the decomposition ends up small (2.2 active components per token). Single lines without file context are hard to predict anyway, so the KL is spread thinly over many uncertain tokens [assumed].

**Python imports: for.**
- Highly predictable continuations (`import numpy as` → `np`, `from django.db import` → `models`). The paper decomposed exactly such prompts: `import numpy as` / `import pandas as` on a 4-layer and a 12-block model. So imports are known to produce clean, active components under tPD.
- Still clearly not truth statements, and still unrelated in content.

**Python imports: the catch.** Predicting `np` after `import numpy as` is *recall of a memorised association*, which is very likely the same kind of MLP mechanism the model uses to judge "The city of Krasnodar is in Russia" (it has to recall Krasnodar → Russia) [assumed: standard picture of factual recall in mid-layer MLPs, not checked for Qwen]. This changes what the control tests:
- Code lines asked: "does any tPD give these probe results?"
- Imports ask: "does truth data give more truth structure than a target that shares the recall mechanism but has no true/false contrast?"

That is the sharper and more interesting control. But if imports reach arm A's probe results, the conclusion is "recall components carry the truth signal", not "tPD is target-agnostic". Both are worth knowing; the interpretation just has to be fixed in advance.

**Practical doubts:**
- **Enough distinct lines?** We need 8,354 lines of 6–20 Qwen tokens, deduplicated (`import os` is 2 tokens and excluded). The Pile test file has 18,195 GitHub documents; how many are Python, and how many distinct import lines of that length they contain, is unknown. It may fall short; then a second source would be needed, e.g. a later Pile train shard (tiny overlap risk with the non-target stream) or a dedicated code dataset. A count takes ~3 min (same streaming as the generator).
- **Length matching** is harder: import lines cluster at a few lengths.
- **Truth content:** re-run the projection check (~30 min on the laptop).

**A cheap pre-check before any GPU run** [proposed]: measure, per candidate dataset, how much the model's predictions depend on layers 15–19 `down_proj`: the KL when their outputs are mean-ablated, compared with tiu. This roughly predicts the "all components off" KL that sized the code control. Candidates: imports; the form-matched sentences from the first design entry ((a) "The city of Krasnodar." / (b) "Anna would like to visit the city of Krasnodar."); code lines as the reference. A data set close to tiu's dependence should give a decomposition of comparable size. [assumed: the mean-ablation KL predicts tPD's delta-only KL only roughly; untested]

**Recommendation:**
1. **Wait for the running probe results first** (~30 min). They show the code control's sparsity *on tiu* and its probe accuracies. If its CIs on tiu are reasonably active, the code control may already answer the question with the matched-sparsity readout, and a new target is a nice-to-have.
2. If a new control is wanted: count import lines + ablation pre-check (laptop, under 1 h total, after the probes finish to avoid CPU contention), then decide between imports and a form-matched target.

---

26-10-05, Claude Opus 5.5 — *probe results: the control's CIs are mostly off on tiu*

`probe_ci.py ~/spd_out/26-10-05_no_truth_baseline/probe s-bd23f0d1 s-b6cce5de s-7fad0c14` ran on the laptop CPU, exit 0. Output files `examples.txt` 13:48 to `features.npz` 14:25 local time (≈ 37 min plus model loading). Follow-up analysis: `~/spd_out/26-10-05_no_truth_baseline/followup.py` (same measures as the 26-10-03 `followup.py`, plus a sparsity-matched top-k readout), output `followup.out`. tiu test split, 1,794 statements, chance 0.50; one seed per run.

**Plain probes** (test accuracy):

| features | last / logreg | last / mass-mean | mean / logreg | mean / mass-mean |
|---|---|---|---|---|
| residual stream L19 | 0.998 | 0.981 | 0.997 | 0.530 |
| untrained `s-7fad0c14` | 0.998 | 0.981 | 0.993 | 0.824 |
| arm A `s-bd23f0d1` | 0.991 | 0.918 | 0.994 | 0.865 |
| **code control `s-b6cce5de`** | **0.654** | 0.640 | 0.599 | 0.542 |
| mean log-probability | 0.575 | | | |

[verified: `probe/results.json`]

**Component level** (last-token readout; "active" = CI > 0.01; top-k matched = each statement keeps only its own k largest CIs, then logreg on all 480):

| run | active per statement: mean [p10, p50, p90] | statements with 0 active | binarised on/off | best single component | top-5 by label | all 480 | top-k matched k=2 / 7 / 10 |
|---|---|---|---|---|---|---|---|
| arm A | 9.3 [5, 7, 15] | 0% | 0.941 | 0.741 | 0.936 | 0.991 | 0.946 / 0.991 / 0.994 |
| control | 1.4 [0, 2, 2] | 29% | 0.582 | 0.640 | 0.644 | 0.654 | 0.654 / 0.653 / 0.654 |
| untrained | 219.9 [209, 219, 232] | 0% | 0.996 | 0.831 | 0.939 | 0.998 | 0.926 / 0.979 / 0.985 |

Control accuracy per tiu family (all 480, last token): animal_class 0.95, cities 0.95, element_symb 0.82, sp_en_trans 0.78, larger_than 0.60, smaller_than 0.56, all negated families 0.50–0.59. Arm A: 0.95–1.00 everywhere.

[verified: `followup.out`]

**Reading:**
1. **My prediction was wrong.** I predicted the control would reach ≥ 0.97 like every CI network so far; it reaches 0.654. Julian's "I have to check this" was right. [verified]
2. **But the low score mostly reflects inactivity, not "no truth structure".** On tiu the control's CI network is nearly off: 1.4 active components per statement, none at all for 29%. Training for sparsity on code taught it to stay off on inputs that aren't code-like. With 0–2 active components, a probe has almost nothing to read. This is the failure mode flagged before the run. The top-k-matched readout cannot fix it, because the control has too few active components to fill even k=2 [concluded].
3. **What the control does show** [concluded]: the untrained network's ceiling (0.998) is a property of a *dense* random readout. A trained, sparse CI network carries truth information only through components that are active on the probed inputs. Training for sparsity does not automatically preserve truth information; arm A keeps it because its components are active on tiu. That is real but weak evidence about tPD on truth data: it says "active components carry truth", not "the components are truth-specific".
4. **Odd detail:** with only 1–2 active components, the control still reaches 0.95 on affirmative cities and animal classes, and chance on all negated families. So its few components fire differently for true and false affirmative statements in those domains [verified: per-family accuracies]. I have not looked into why [assumed: code-trained components that respond to how familiar or plausible the sentence is].
5. **Sparsity-matched untrained baseline** (deferred earlier, now done as a by-product): at matched k, arm A is 1–2 points above untrained (k=2: 0.946 vs 0.926; k=7: 0.991 vs 0.979; k=10: 0.994 vs 0.985). The test-set standard error of such a difference is ~0.005, but the seed-to-seed spread of trained runs is larger: the fp32 arm-A-like runs scored 0.970–0.984 on all 480, this bf16 run 0.991. So the 1–2-point gap is **not** evidence yet; it needs seeds [concluded].

**Consequence for the next control** (Julian's idea: more active components, e.g. Python imports): what matters is how active the control's CI network is **on tiu**, not on its own target. An imports control might give a bigger decomposition and still be off on tiu sentences, for the same reason as the code lines. A target whose inputs resemble tiu (the form-matched options from the first design entry: same entities and sentence shape, no true/false claim, e.g. "The city of Krasnodar." or "Anna would like to visit the city of Krasnodar.") is far more likely to be active on tiu [concluded: the CI network is trained on its target and stays off elsewhere; form-matched inputs are closest to tiu]. Proposal: prefer a form-matched control over imports. Activity on tiu can only be measured on a trained control. So the check is the run itself (~40 min of GPU): look at its L0 on tiu before reading its probe accuracies.

---

C: (transcribed from chat) Can you propose how you would construct the form-matched control? I'm not sure about needing another arm A run. I'm looking for visible effects here and not for exact statements about how much this and that effects the outcome. This is more exploration than engineering.

26-10-05, Claude Opus 5.5 — *proposal: form-matched control dataset*

**On the extra arm-A run:** agreed, none needed. The comparison uses the existing bf16 arm A `s-bd23f0d1`. (What I had proposed was second *seeds*, to read 1–2-point differences. In exploration mode we only read effects that are large compared with the seed spread seen so far, i.e. clearly more than ~2 points, and skip the seeds.)

**Goal of the dataset.** Inputs that look like tiu statements to the model (same entities, same kind of sentence, same final `.`, similar length, negation present in the same share) but make **no claim whose truth the model could check**. Then the control's CI network is trained on tiu-like inputs, so it should be active on tiu, and any truth structure in its CIs cannot come from the training target.

**Construction:** one control sentence per tiu statement, built from that statement's **subject and object**, put into a frame that mentions both without relating them:
- *Mentioning* both keeps the tokens of the true and false variants: "Krasnodar … Russia" and "Krasnodar … South Africa" both appear.
- *Not relating* them removes the claim.
- Each tiu statement's split carries over, which keeps the split by subject.

Several frames per domain, picked at random per statement, so that no single fixed phrase dominates. Negated tiu statements get a frame containing "not"/"didn't", which keeps the negation tokens without negating a fact. Draft frames (subject **S**, object **O**):

| domain | tiu example | control frames (draft) |
|---|---|---|
| cities | The city of Krasnodar is in Russia. | Anna wrote a postcard about Krasnodar and Russia. / The quiz card lists the city of Krasnodar and Russia. / Tom read a story about Krasnodar and Russia. |
| animal_class | The lobster is a crustacean. | The children drew a lobster and a crustacean. / The poster shows a lobster next to a crustacean. |
| element_symb | Barium has the symbol Sb. | The worksheet lists barium and the letters Sb. / Mia wrote barium and Sb on the board. |
| sp_en_trans | The Spanish word 'tener' means 'to have'. | The worksheet lists the Spanish word 'tener' and 'to have'. / Leo wrote 'tener' on one card and 'to have' on another. |
| larger/smaller_than | Fifty-two is larger than fifty-one. | She wrote down fifty-two and then fifty-one. / The two cards show fifty-two and fifty-one. |
| negated (all) | The city of Krasnodar is not in Russia. | Anna did not write about Krasnodar and Russia. / The poster does not show a lobster or a crustacean. / … one "not" frame per domain |

Objects of `animal_class`, `element_symb` and `sp_en_trans` are parsed from the statement text by regex, because the records have no object field. The parse is logged with examples, per the Observe & Verify rule.

**Built-in truth check (stronger than the code check).** Each control sentence has a *source label*: was it built from a true or a false tiu statement? Project the control sentences onto the tiu truth directions (the same script as the code check) and compare the two groups. If "Krasnodar and Russia" lands on the true side and "Krasnodar and South Africa" on the false side, the model still reads truth from the co-mention alone (plausibility of the association) [assumed: possible; this is what the check is for]. Read it as:
- separation small compared with tiu's: good control;
- separation large: the frames leak truth. Then fall back to subject-only frames ("Anna would like to visit Krasnodar."), which lose the object tokens.

**What to expect, and what would count as a visible effect:**
- First look at the control's L0 (active components) on tiu. If it is again ~1–2, this control failed the same way as the code one.
- If it is active on tiu (several components per statement): compare the probe accuracies.
  - Control ≈ arm A (both ~0.99): truth structure in the CIs comes from tiu-like *inputs*, not from training on true/false contrasts.
  - Control clearly below arm A (e.g. ≤ 0.9 while comparably active): training on true/false statements made the components more truth-related.

  Per-family accuracies (affirmative vs negated) tell more than the overall number.

**Work and cost:**
- generator `build_tiu_formmatched.py` (~150 lines: templates + regex parse), dry run with examples to read;
- projection check with the source-label comparison (~30 min, laptop);
- Julian commits the generator, then build and commit the data;
- config = code-control config with the target swapped;
- GPU: one run, ~40 min (concurrency not needed for a single run);
- probe: ~40 min on the laptop, or minutes on the GPU if the VM is up.

**Open choices for Julian:**
1. Frames: are subject+object co-mention frames OK (with the leak check), or subject-only from the start?
2. Should the frames be written by hand, as drafted (a few per domain), or should more varied ones be generated? Hand-written is enough for exploration [assumed].

---

C: (transcribed from chat) Ok, no. I don't want to do this. I think this is the end for the arm A experiment. Can you write a summary in the scratch folder?

26-10-05, Claude Opus 5.5 — *topic closed; arm A experiment summary written*

Julian declined the form-matched control and ended the arm A experiment [decided: the transcribed message above]. Not pursued: the form-matched control, Python imports as a target, second seeds, and the concurrency test.

Wrote a self-contained summary of the whole arm A experiment (26-10-02 to 26-10-05: question, setup, results 1–4, conclusion, what was not done, side results, run ids) to `scratch/arm_A_experiment_summary.md` (the project's gitignored scratch folder). Its numbers were re-read from `convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md` (*Results*), `~/spd_out/26-10-05_no_truth_baseline/followup.out` and `projection_report.json`, and this LOG's tables.

Bottom line, as written there [concluded]: no evidence that tPD recovers truth-specific mechanisms on this setup. CI probes read truth because the activations encode it (untrained 0.998). The trained sparse components keep the signal, but are not measurably more truth-related than the largest components of a random network at matched sparsity. The code control stayed silent on tiu (1.4 active components, 0.654), so it could not separate the two.

Left for Julian: commit `data/pile_code/github_lines/v1` unchanged (runs `s-b6cce5de` and the projection check refer to it) and these records. The vast.ai instance can be destroyed (nothing unsaved on it).

---

26-10-05, Claude Opus 5.5 — *sparsity-matched baseline already done; conversation ended*

Julian asked to run the earlier-proposed control "with an untrained decomposition, taking for each example the components with the highest CI, not selecting by truth", on a rented GPU (asking whether an RTX 4090 is enough). I answered in chat: this is the sparsity-matched untrained baseline, and it is **already done**, as a by-product of the probe follow-up in the entry *probe results: the control's CIs are mostly off on tiu* (top-k matched: k=2 0.946 vs 0.926, k=7 0.991 vs 0.979, k=10 0.994 vs 0.985, arm A vs untrained; also section 3 of `scratch/arm_A_experiment_summary.md`). It runs in seconds on the laptop from the saved `features.npz`, so no GPU is needed. For the record: an RTX 4090 (24 GB) would suffice for probing (7B in bf16 ≈ 15 GB for inference) but not for training (≈ 27 GB peak at batch 16) [assumed: from the memory measured on the H100].

Offered and not run: matching k per statement to arm A's own active count for that statement. Julian declined and ended the conversation.

Closing note: Julian committed `data/pile_code/github_lines/v1` unchanged in `157e7e7` ("add besline code data") [verified: `git diff HEAD` empty for it]. Added an `OUTDATED` marker next to my wrong pre-run prediction in the first entry, and a FEEDBACK.md entry: predictions about experiments not yet run should be `[assumed]`, not `[concluded]`.
