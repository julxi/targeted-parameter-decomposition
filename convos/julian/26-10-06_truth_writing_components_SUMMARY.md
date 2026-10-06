# SUMMARY: Do tPD components write into the truth direction? (follow-up on the UTH and tiu decompositions)

**Last updated:** 26-10-06 (sync: numbers aligned to the LOG, missing UTH A5/G results, design and Open sections added, lint/type status verified; pkill rule adopted into CODING.md)

Julian started this topic from his own notes at the top of the LOG. The first agent response was written against an unsaved shorter version. Julian then supplied the full one, which replaced the opening text; the entry *revision after Julian's full notes* records what changed. The notes ask for clarifications, state a hypothesis, list three ideas, and ask for help designing an experiment on "more specific truthfulness". Julian then commented inline. The answers sit in place beneath each comment, and the entry *answers to Julian's comments; plan for the VM session* collects the result. The entry *analysis script truth_writers.py; first results on tiu arm A* holds the code description and the tiu results; *UTH results (cross-task)* the UTH results and the overall reading.

Decompositions in play:
- UTH `s-d2ded461` (UTH = Universal Truthfulness Hyperplane datasets; has the cross-task split);
- tiu arm A `s-bd23f0d1` (tiu = Truth-is-Universal statements; the better fit);
- untrained `s-7fad0c14`;
- code control `s-b6cce5de`.

## Decisions

- Analyses on both UTH and tiu arm A [decided: C: comment].
- Everything runs on a rented RTX 4090, nothing on the laptop [decided: C: comments].
- Truth direction: the mass-mean direction is the primary guess, but multiple directions, including task- or domain-specific "overfitted" ones, are part of the analysis [decided: C: comments].
- "More specific truthfulness" is exploratory: how different kinds of truth work together, and what tPD shows that probes can't. It is not a single hypothesis test [decided: C: comment].
- Follow the probes and analyses first, new training later; the H100 memory trial waits until A2 has pointed at layers [decided: C: comments].

## Clarifications

- **"tPD decomposes only what the target uses" is a hope, not a fact** (Julian).
  - tPD (targeted parameter decomposition) splits weight matrices into rank-1 components plus a remainder Δ. Training balances target reconstruction (Δ randomly or adversarially scaled down on target batches) against importance minimality (few active components). A mechanism that would need many components can stay in Δ, at the price of target reconstruction loss [concluded: from the code, `spd/utils/component_utils.py`, `spd/persistent_pgd.py`].
  - Cost in our runs [verified: earlier SUMMARYs]: UTH target KL 0.433 with rounded CIs (causal-importance values), 0.479 with all 480 components and no Δ; tiu arm A 0.136 / 0.135.
  - Wording fixed in `convos/julian/26-10-02_overview_of_goal_SUMMARY.md`, `scratch/uth_experiment_summary.md` and `scratch/arm_A_experiment_summary.md`.
- **State of evidence** (Julian, agreed): no evidence that tPD recovers truth mechanisms, none that it can't.
- **The untrained baseline is not luck** [concluded]: a random nonlinear readout of activations that carry truth linearly keeps that information for almost any draw.
- **The code control does say something** (Julian's argument, confirmed on re-check) [verified: recomputed 26-10-06 from `~/spd_out/26-10-05_no_truth_baseline/probe/features.npz`, and `followup.out`].
  - The probes standardise each CI feature, so a weak signal doesn't matter.
  - Where the control's components are on (1,462 of 1,794 tiu test statements), its probe reads truth at 0.656. The untrained network's top 2 CIs per statement read it at 0.926, arm A's top 2 at 0.946. The control fires few generic components: 66 are ever non-zero, against 385 for arm A.
  - Reading [concluded]: training on an unrelated target removes truth from the CIs. Training on truth keeps it, but adds only ~2 points over a random readout at matched sparsity (one seed each). The target shapes what the CIs read; "tPD learned truth" remains unshown.
- **KL-trained tPD can't see an unused computation** [concluded, from the loss definition; but see gate G below: on tiu the output does use truth]. Julian's hypothesis: the model always computes truth at the statement end for possible later use, but the output need not use it. Our runs train only on the next-token KL. A computation that changes no next-token output in the training text costs nothing to lose; in Δ it is degraded for free. To give tPD a reason to keep it: text after the statement that depends on its truth, or a hidden-activation loss (`StochasticHiddenActsReconLoss` exists and is selectable as a training loss, but probably ignores `loss_positions` [assumed: read, not tested]).
- **Layer budget:** 5 decomposed matrices is a choice, not a limit. The config comment says decomposing all 28 × 7 matrices OOM'd. The bf16 runs peak at ~27 GB on the 80 GB H100, so more layers likely fit [assumed: 1–3 GB per extra `down_proj`, unmeasured; a 150-step trial would measure it].

## Hypothesis and existing evidence

Julian's weak hypothesis: components write into the truth direction on some training tasks but don't fire on held-out tasks. Alternatively, they *use* truth rather than write it. The UTH paper's universal direction leaves open which kind of truth triggers it.
- The UTH inversions argue against "doesn't fire" for three test datasets [concluded]. Trained-CI probes score copa 0.285 (n 200), story_cloze 0.401 and sciq 0.438 (n 1,000 each). That is 4–6 standard errors below chance, so the information is present with flipped sign. hellaswag (0.500) fits "silent". Julian finds the loud failure worth investigating, which became A6. The `truth_writers.py` results below bear this out: within-task probes find truth in the trained CIs on all four held-out datasets, and the inversions come from an overfit probe [verified: A4, A6].
- In-distribution, the trained UTH CIs read truth worse than the untrained ones (mean over datasets 0.706 vs 0.794; residual layer 19: 0.850; the UTH SUMMARY's 0.705 is the pooled accuracy of the same probe) [verified: `~/spd_out/26-10-05_uth_experiments/stage1/results.json`].
- Layers: cross-task residual probes after layers 7/11/15/19/23/27 give 0.585/0.651/0.768/0.806/0.771/0.741 [verified: same file]. The biggest gain lies between layers 11 and 15, before the decomposed layers 15–19. Caveats: where a probe reads truth is not where it is written; only MLP `down_proj` matrices are decomposed, while the UTH paper probes attention heads. The tiu A2 attribution below found that direct writes depend on the readout layer, so it did not single out layers 12–15 as writers.

## Design: "more specific truthfulness" (exploratory) [concluded: agent proposal; Julian made it exploratory]

What tPD can show that a probe can't [concluded: a probe yields a direction; a component is a piece of the weights with input side, output side and causal role]: shared computation vs shared readout (kinds of truth readable along one direction but written by different or the same components, relevant to the UTH paper's "other probes overfitted" argument); whether negation reuses the affirmative writers with flipped sign; causal reach of a component across kinds; what drives a writer (its input direction V_c). Probe accuracy alone can't show a decomposition is "more specific"; the success criterion is a causal double dissociation.
- Step 0: per-domain writers on tiu arm A (kinds × components matrix; noise floor: split halves; reference: untrained). Done as A5 below.
- Step 1: double dissociation by ablation (domain X's writers hurt X more than other domains, beyond matched random ablations). This is B, not yet run.
- Step 2 (only if steps 0–1 show specificity): purpose-built minimal true/false pairs (arithmetic, temporal order, contradiction with an earlier sentence, optionally made-up entities) and a new decomposition.
- Other readings the agent offered (model belief vs statement truth; lying in the model's own answers) were not taken up.

## Implemented: `truth_writers.py` [verified: CPU and GPU smoke tests; full tiu and UTH runs 26-10-06; ruff clean and basedpyright 0 errors (2 untyped-decorator warnings), re-run at sync 26-10-06]

`spd/experiments/lm/honesty_targeted_decomposition/truth_writers.py` (uncommitted).
- `extract <out_dir> <tiu|uth> <run_id>...` (GPU) stores per sample at the last token:
  - residual stream after layers 15–19 and 23;
  - per run CI values and inner activations a_c;
  - next-token log-probs on the 1,000 most likely tokens;
  - per (dataset, label) group sums of all attention/MLP outputs of layers 0–23 and of each decomposed layer's Δ part (exact class means for A2).

  The A7 pass (mask gradients, ≤ 200 samples per dataset) follows, with a real-ablation check of the top 5 components per run. Outputs: `features.npz`, `a7.npz`, `samples.jsonl`, `meta.json`.
- `analyze <out_dir> <n_workers>` (CPU) → `results.json`, `report.txt`, `a5_matrices.npz`. Analyses:
  - A1: alignment of U_c with truth directions, against a permuted-label null;
  - A2: exact per-layer attribution, components vs Δ;
  - A3: probes on CI and activation features;
  - A4: within-dataset CV on the test datasets;
  - A5: writers per dataset;
  - A6: inversion drill-down;
  - A7: total vs direct effects;
  - G: paired KL of the last-token output.
- tiu gets a `tune` split (20% of the train subjects) for choosing C. Built-in exactness checks pass on the real runs (residual = sum of sublayers within 1.4%; components + Δ = original model within 3%).
- An A7 linearity check re-runs the top 5 components per run and split as real ablations; it was added after a smoke test showed suspiciously large first-order effects.
- On the 48 GB RTX 4090 the tiu extraction takes 2.4 min; UTH extraction 23 min plus 6 min analysis on the VM's CPUs. Results are copied to `~/spd_out/26-10-06_truth_writers/<data>/` (UTH checksums match).

## tiu results (arm A vs code control vs untrained; one seed each) [verified: `~/spd_out/26-10-06_truth_writers/tiu/report.txt`]

1. **Most direct truth writing in layers 15–19 stays in Δ.** Along the layer-19 truth direction, MLPs 15–19 write 81% of the true/false separation. Arm A's components carry 14.5% of it, Δ 67%. Control and untrained components carry ≈ 0%. With per-dataset directions, arm A's components carry 4–13%.
2. **Direct attribution depends on the readout layer** [concluded]. The layer-19 direction is written mostly by MLPs 18–19 (25%, 41%); the layer-23 direction 84% by layers 20–23. Truth looks rewritten layer by layer, so direct attribution doesn't locate "the" writer layers.
3. **Two arm A components have large indirect effects** [verified: real ablation, A7 linearity check]. Ablating L17:33 or L19:34 changes the layer-23 truth separation by −19.7 / −18.2 out of 62.5 (fit subsample, n 1,594) and −19.0 / −18.8 out of 70.1 (test), i.e. ~27–30%. Next: L16:69 −10.5, L18:15 −8.5 (fit). L17:33 writes almost nothing along the direction itself (direct write +0.01). Control and untrained: at most 1.3–1.7, 10× smaller. The first-order estimate underestimates arm A's effects by 15–50%. Unspecific disruption is not yet excluded [assumed risk, untested]; that needs B's controls.
4. **Writers differ by kind of statement** [verified: A5]. L18:62 is active only on numeric comparisons (larger_than 51%, smaller_than 90%, the other 8 families 0%). L15:84, L16:69 and L18:15 are active on factual recall and translation and ~0–1% on numeric. L19:34's write is concentrated on negated statements (29–30% of S for neg_element_symb and neg_sp_en_trans, 7–9% for neg_animal_class and neg_cities, ≈ 0 for affirmative cities and animal_class). Between-dataset correlation of write vectors: arm A 0.50, untrained 0.26, control 0.19 (split-half floor 0.98–1.00). Caveat: a domain-specific component may encode the topic (numbers vs entities) rather than truth; B's per-domain dissociation decides.
5. **Gate G: the output after the statement does depend on truth** [verified: G] (against the agent's prediction, and against Julian's "not used for the output" premise for affirmative statements). After a false city statement the model predicts " Is" with p 0.58 (true: 0.005); after a true one " It" 0.33 (false: 0.013); after a false larger_than statement " (" 0.95 (true: 0.39). Paired KL(true ∥ false): cities 3.89 vs baseline 1.39 nats; larger_than 2.39 vs 0.40; animal_class 1.12 vs 0.62; negated weaker (neg_cities 0.71 vs 1.09). A probe on the output log-probs reads truth at 0.996. Caveat: the paired KL also contains the content difference of the last word. So a plain last-k decomposition (Q4 option a) would get truth pressure on tiu [concluded].
6. **Uninformative, as expected:**
   - A1 alignment is weak: arm A max |cos| along the layer-19 direction 0.082 vs null median 0.057, p = 0.03 (one of 7 global directions tested); p < 0.05 for 5 of 10 per-dataset directions, untrained 1, control 0.
   - Probes on ungated inner activations read truth at 0.997–0.998 for every decomposition, including untrained and control. CI probes: arm A 0.991, untrained 0.997.
   - No inversions on tiu.

## UTH results (UTH run vs untrained; one seed) [verified: `~/spd_out/26-10-06_truth_writers/uth/report.txt`]

1. **Components carry even less direct truth writing than on tiu** [verified: A2]. Along the layer-19 direction: components 6.6% (fit) / 8.4% (test), Δ 61% / 59%; untrained components ≈ 0. With each training dataset's own direction: −1% to 8%.
2. **The cross-task failure is non-transfer, not absence** [verified: A3, A4].
   - Cross-task (mean over test datasets): trained CIs 0.490 (chance; stage 1 gave 0.488), untrained 0.717, residual L19 0.807, trained inner activations 0.773.
   - Within each held-out dataset (5-fold CV), trained CIs read truth well: copa 0.920, story_cloze 0.920, sciq 0.784, hellaswag 0.666 (untrained 0.945/0.998/0.890/0.896).
3. **The inversions are an overfit probe** [verified: A6; reading concluded].
   - Strong regularisation removes them: with the selected C = 1.0, copa 0.300, story_cloze 0.395, sciq 0.447; with C = 1e-4, 0.77, 0.765, 0.66. Binarised CIs give 0.52–0.66, never inverted.
   - Correlation of per-component standardised true − false differences, training tasks vs test dataset: copa +0.24, story_cloze +0.13, sciq −0.02: what separates true from false differs by task.
   - Drivers: a few heavily weighted components whose true/false difference flips sign (L15:10, L17:47, L17:26 for copa/story_cloze; L15:21 for sciq). C selection chose the weakest regularisation because it fits the training tasks best; nothing deeper than overfitting is needed to explain the loud failure.
4. **Large causal components on UTH too** [verified: real ablation, A7 linearity check]. Actual change of the layer-23 separation when ablated: L16:5 −4.5 (fit, S = 22.6, n 5,734) / −5.9 (test, S = 13.0, n 1,600); L15:10 −5.8 / −5.8; L18:36 −1.9 / −1.6. So L16:5 or L15:10 each removes ~45% of the cross-task separation; untrained at most 0.28. For L15:10 the first-order estimate on fit is far off (−1.6 vs −5.8). L15:10 is also the top inversion driver: it matters for truth on held-out tasks, but its CI relates to truth differently there. Same caveat as tiu: unspecific disruption not excluded until B.
5. **Writers per dataset** [verified: A5]. Between-dataset correlation of write vectors: trained 0.26, untrained 0.20, split-half floor 0.90; weaker sharing than tiu arm A (0.50). Several top writers are active only on some tasks (e.g. L17:8: median activity 0, maximum 0.14).
6. **Gate G** [verified: G]: in the UTH format ("Answer: X."), truth changes the next-token output less than content does. Paired KL vs baseline (two different true items): copa 0.38 vs 0.30, story_cloze 0.60 vs 0.44, sciq 1.97 vs 3.92, nq_re_long 1.56 vs 4.32 (crude: the baseline is large where items differ a lot). Probe on the output log-probs: cross-task 0.725 (residual 0.790 pooled). The format matters for a future last-k decomposition.

## Overall reading and next [concluded; one seed per run; mostly correlational]

- **Both trained decompositions contain a few components whose ablation removes a large share of the truth separation at layer 23** (~30% tiu, ~45% UTH cross-task), mostly indirectly. Nothing comparable exists in the untrained or code-control decompositions. This is the first project result where tPD components look specifically involved in truth. It is not claimable until B's controls have run.
- Most direct truth writing in layers 15–19 stays in Δ (tiu: components 14.5%, Δ 67%; UTH fit: 6.6%, 61%). Writers differ by kind of statement on tiu. On UTH the cross-task probe failure comes from task-specific readouts of components that do matter (L15:10).
- **Proposed next: B** (not yet ratified), as a third subcommand of `truth_writers.py`, minutes on a GPU.
  - Ablate the candidates: tiu L17:33, L19:34, L16:69, L18:62 and the per-domain writers; UTH L16:5, L15:10, L18:36.
  - Measures: separation along the layer-23 and layer-19 mass-mean directions, per dataset.
  - Controls: (i) 20 random components with matched mean activity per candidate; (ii) the effect on a non-truth separation (family/domain; on UTH dataset identity); (iii) on tiu the numeric vs factual double dissociation.
- **New training later.** Loss options for a decomposition focused on the statement end (LOG, *Q4 rephrased*): (a) plain last-k KL; (b) an appended truth question with KL on the answer position; (c) a hidden-activation loss at the last k positions (needs code work). The agent recommended deciding after G and A2. On tiu, (a) now looks viable (G); in the UTH format less so. A2 suggests that writer layers are not one fixed place.

## Bookkeeping

- An agent killed its own shell during this topic with `pkill -f` followed by a relaunch in the same call. `docs/CODING.md` (*Kill by a bracket pattern or PID*) now covers this case: the bracket pattern protects only a command line that names the target nowhere else, so kill in a call of its own or by PID [decided: Julian, `C: (transcribed from chat)` in the LOG, entry *pkill rule added to CODING.md*].

## Open

- B (with controls) — proposed, not yet run or ratified.
- Loss for a new decomposition: Julian found the point that the KL loss sees only next-token outputs, not hidden computation, important, will think about it and "this might need a different approach" (C: comment under *Q4 rephrased*). Not decided.
- Before using option (c): check whether `StochasticHiddenActsReconLoss` respects `loss_positions` [assumed: probably not].
- H100 memory trial for more decomposed layers: waits until the analyses point at layers [decided].
- All C: comments in the LOG are answered in place or acted on in the next entry. The comment "yes, but I'm not strongly sold" (under *The universal direction and "more specific truthfulness"*) has no direct answer of its own; the answer to the comment right after it covers it.

- See also: [convos/julian/26-10-05_uth_experiments_SUMMARY.md] — the UTH run `s-d2ded461`, cross-task protocol, `probe_uth.py`, stage 0 extraction cost on the 4090, and the inversions analysed here.
- See also: [convos/julian/26-10-05_no_truth_baseline_SUMMARY.md] — tiu arm A `s-bd23f0d1`, code control `s-b6cce5de`, untrained `s-7fad0c14`, matched-sparsity method, bf16 memory trials; this topic re-reads its code-control result.
- See also: [convos/julian/26-10-02_overview_of_goal_SUMMARY.md] — tPD method reference (wording corrected by this topic); the never-run last-token arm.
- See also: [convos/julian/26-10-02_training_run_SUMMARY.md] — the last-token arm was flagged there as needing its own tuning.
- See also: [convos/julian/26-10-05_vast_rentals_SUMMARY.md] — renting the 48 GB RTX 4090 this topic runs on: host blacklist, A100 alternative (`--config a100`), and the open question why ≥ 40 GB is needed when this topic's plan named a 24 GB 4090.
