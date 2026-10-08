# SUMMARY: State report on the tPD truthfulness experiments (Typst)

**Last updated:** 26-10-08 (PDF gitignored; report written and self-contained check fixes in on 26-10-06)

Julian asked for a detailed Typst report of all tPD (targeted parameter decomposition) truthfulness experiments so far, so he can decide the next steps: what was tested, what each probe takes as input (position, layers, top-k CIs, probe type), and the results. The report has a shelf life, so its name starts with the date; helper files for future reports were welcome.

**Files:**
- `reports/26-10-06_truth_experiments_state.typ`: the report. The compiled PDF is not in git (`reports/*.pdf` is gitignored, [decided: Julian, 26-10-08 in the LOG]); build it with `typst compile reports/26-10-06_truth_experiments_state.typ`.
- `reports/template.typ`: shared layout for future reports (title block, "bottom line" box, status tags matching the memory system's markers, callouts, compact tables). Typst pitfalls are listed in the LOG.
- Julian said "the new `report` folder"; the existing empty folder is `reports/`, which is what was used.

**What the report covers:** tiu CI probes with untrained baseline (Experiment 1), the code-lines control (2), UTH cross-task probes (3), the component-level analyses of `truth_writers.py` (4: A1–A7, gate G), a claim/status table, caveats and a table of next-step options with cost and expected information. Probe and analysis numbers were re-read from the raw result files under `~/spd_out/` [verified: 26-10-06].

**Bottom line it states** [concluded, from the earlier topics' results]: CI probes read truth for a trivial reason (an untrained CI network does as well); on UTH the trained CIs don't transfer across tasks; most direct truth writing in layers 15–19 stays in Δ; the only positive lead is that ablating single trained components removes 27–30% (tiu) / ~45% (UTH unseen tasks) of the layer-23 truth separation, uncontrolled so far.

**New points raised in the report** [assumed, details in the LOG]: the untrained-ablation reference may be weak (random components at their initial scale are smaller perturbations); B should match random trained components on activity *and* overall effect, and report S relative to the class spread; ablation shows involvement, not whether a component computes truth or reacts to it.

**Recommendation** [assumed: agent view, not ratified]: run B (controlled ablations) before any new training; then a second arm-A seed and characterisation of the candidates if B holds.

- See also: [convos/julian/26-10-06_truth_writing_components_SUMMARY.md] — source of Experiment 4 and of the proposed ablation experiment B the report recommends.
- See also: [convos/julian/26-10-05_uth_experiments_SUMMARY.md] — source of Experiment 3 (UTH cross-task).
- See also: [convos/julian/26-10-05_no_truth_baseline_SUMMARY.md] — source of Experiment 2 (code control) and the matched-sparsity numbers.
- See also: [convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md] — source of Experiment 1 (fp32 runs).
