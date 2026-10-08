C: (transcribed from chat) Another session has done quite a few experiments on the current decompositions and probes in general. To enable me better to decide how we sohuld proceed from here I want you to write a report into the new `report` folder. It should be in typst and it should explain in detail what was being tested and what are the results.
So for example it should describe what the probe takes (sequence pos, layers, top-k CI, etc.), what kind of probe. I need to get a full picture of the current state to understand what are the best next steps.
C: (transcribed from chat) Please write the report as a typst file. You can also create helper typst files for future reports.
The report is not a paper, but has a shelf life. So start the name with the date.

---

26-10-06, Claude Opus 5.5 — *state report written: reports/26-10-06_truth_experiments_state.typ*; I have read INDEX.md, the SUMMARYs of all tPD-truth topics (overview of goal, training run, prepared datasets, test accuracy, no-truth baseline, UTH experiments, truth-writing components), the full LOG of the truth-writing topic, the two local experiment summaries in `scratch/`, the probe scripts (`probe_ci.py`, `probe_uth.py`, `truth_writers.py`) and the arm A config.

Julian asked for the new "report" folder; the folder that exists (empty, created 26-10-06) is `reports/`, so I used that.

## Files

- `reports/26-10-06_truth_experiments_state.typ` (+ compiled `.pdf`, 12 pages): the report. Compile with `typst compile reports/26-10-06_truth_experiments_state.typ` (typst 0.14.2).
- `reports/template.typ`: shared layout for future reports: `report.with(title, subtitle, date, author, summary)`, status tags `#verified[…]`, `#concluded`, `#assumed`, `#decided` (mirroring the memory system's markers), callouts `#finding`, `#caveat`, `#note`, `#runid`, and a compact table `#tbl` (short tables stay on one page; `breakable: true` for long-celled ones). Typst pitfalls met on the way, worth knowing for future reports: bare underscores (`mm_L19`, `story_cloze`) start emphasis, so identifiers go in backticks; a table cell starting with `+` or `-` becomes a list item, so negatives use the Unicode minus and positives `\+`; `~` is a non-breaking space.

## Content (sections)

Bottom line; scope; tPD background and null hypothesis; shared setup (model, layers 15–19 `down_proj`, C = 96, CI network = per-position MLP on the five concatenated `down_proj` inputs → 512 → 480, clamp to [0, 1]); datasets; runs with fit metrics; **what the probes take as input** (one forward pass of the unmodified model; last real token; feature sets CI / top-k / binary / top-5 / residual / inner activations a_c / CI-gated / output log-probs / mean log-prob; logistic regression with standardisation vs mass-mean; tiu in-distribution, UTH cross-task, within-task CV); Experiments 1–4 with tables; claim/status table; caveats; options for next steps with cost and expected information; glossary; file locations.

## Numbers

All probe and analysis numbers were re-read from the raw result files [verified: 26-10-06], not from the SUMMARYs: `~/spd_out/26-10-05_no_truth_baseline/probe/results.json` and `followup.out`, `~/spd_out/26-10-03_test_accuracy_analysis/results.json`, `~/spd_out/26-10-05_uth_experiments/stage1/results.json` and `sparsity.log`, `~/spd_out/26-10-06_truth_writers/{tiu,uth}/report.txt`. Training-fit metrics (target KL etc.) come from the SUMMARYs / the no-truth-baseline LOG table, which mark them verified against `metrics.jsonl`; the G token examples (" Is" 0.58 vs 0.005) come from the truth-writing LOG.

Small discrepancies with the SUMMARYs, none affecting conclusions:
- The truth-writing SUMMARY quotes the UTH gate-G line "nq_re_long 1.56 vs 4.32"; that is the *fit* split (a training dataset). The report uses the test split (copa, story_cloze, sciq, nq_re, ...).
- The SUMMARY's "untrained/control at most 1.7 of 62.5" mixes splits: the largest real-ablation effects are control ±1.1–1.3 (fit) / ±1.2–1.6 (test), untrained ±0.6–1.5 (fit) / ±0.6–1.7 (test).
- Mean-over-tokens readout: I first wrote in the report that it was "never better" than the last token. Wrong: on tiu it is slightly better for trained CIs (arm A 0.994 vs 0.991; fp32 runs 0.979–0.996 vs 0.970–0.984). Corrected in the report before delivery.

## New points the report raises (not in earlier records)

- **The untrained-ablation reference may be weak** [assumed]. Untrained components are random rank-1 pieces at their initial scale (V entries std 1/√18,944, U entries std 1/√96, `init_param_` in `spd/models/components.py`), so removing one may simply perturb the model much less than removing a trained component. Nobody measured the perturbation size. The fair reference for B is random *trained* arm-A components with matched activity *and* matched overall effect (e.g. the same output-KL change when ablated).
- **S is in raw units**: an ablation that shrinks the residual stream generally would shrink S without truth specificity; B should report S in units of the class spread as well.
- **"Using vs computing truth"**: G shows the model reacts to false tiu statements (" Is"); a component that is part of that reaction would also reduce the layer-23 separation when removed. Ablation shows involvement, not that the component computes truth.
- UTH mass-mean on the trained CIs reaches 0.574 cross-task, against 0.488 for the (overfit) logistic regression; consistent with the overfit-probe reading of A6. Not mentioned in the report (minor).

## Recommendation in the report [assumed: agent view, not agreed]

Run B (controlled ablations, with the controls above) before any new training; if it holds, a second arm-A seed to see whether the high-effect components recur, then characterise them (input direction, path through layers 20–23). If it fails, the remaining question is whether a different objective (truth question on the answer position, or a hidden-activation loss) gives tPD a reason to decompose truth. Not recommended: more CI-probe accuracy comparisons, the padded arm.

---

26-10-06, Claude Opus 5.5 — *self-contained check and fixes*

Ran `docs/procedures/self_contained_check.md`: a fresh sub-agent (Sonnet) read only the `.typ` file and listed unresolved references, unexplained results, undefined terms, history statements and inconsistencies. Real problems it found, all fixed in the report:
- **Inconsistencies:** code-control table mixed two thresholds (29% of statements with no CI > 0.01 vs 18.5% with all CIs exactly 0; the 1,462-statement probe uses the latter), now both rows shown; A2 shares rounded to whole percent didn't add up (components 15% + Δ 67% > MLPs 81%), now one decimal (14.6 + 66.9 = 81.5); ablation share was "27–30%", actually 27–31% (L17:33 fit 19.65 / 62.54); first-order underestimate "15–50%", actually 13–54%; copa 0.300 (Experiment 4) vs 0.285 (Experiment 3) now explained as a re-extraction difference (cause [assumed]).
- **Overstated readings:** the "run-to-run spread" mixed fp32 and bf16 runs, now the 1.4-point spread of the five fp32 runs, which differ in batch/steps/LR (not "near-identical"); gate G reading now says negated statements show no truth effect beyond the content baseline; A1's p = 0.03 does not survive correction for 7 directions; within-task UTH gap to untrained is up to 0.23, not "somewhat less".
- **Missing context added:** what the motivating result's setup was; definitions of the last-token and padded arms, the run names (run 1, e2b, …), fp32 vs bf16, the labels A1–A7 / G / B, the form-matched control, double dissociation, the hidden-activation loss; a reading key at the top of the summary box; a new table "Overview: what each experiment probes" (position, features, probe, fit → scored on).
While checking, I also corrected two of my own A2 numbers along the layer-23 direction (code-control components 0.3%, untrained −0.2%; I had written 0.1% / −0.1%).

Not changed (judged fine or out of scope): the claim/status table uses tags rather than prose; per-dataset sample sizes for UTH test sets beyond copa are not listed.

Typst pitfall added to the list above: a table cell starting with "1." becomes a numbered list, and `auto` column widths collapse when cells hold long text, so give long-text tables fixed widths and `breakable: true`.

---

26-10-08, Claude Opus 5.5 — *PDF kept out of git*; I have read this LOG's tail, the SUMMARY and the `.gitignore` diff

C: (transcribed from chat) I have added the pdf to gitignore. Can you add this to the relevant conversation

Julian added `*.pdf` to `.gitignore` (a repo-wide pattern, not only `reports/`), so the compiled report `reports/26-10-06_truth_experiments_state.pdf` is not committed [decided: Julian, above]. Only the Typst sources (`reports/26-10-06_truth_experiments_state.typ`, `reports/template.typ`) are tracked; anyone who wants the PDF rebuilds it with `typst compile reports/26-10-06_truth_experiments_state.typ`. Updated the SUMMARY and the INDEX.md mentions of "(+ PDF)" accordingly.

Side effect to be aware of: the pattern ignores every PDF in the repository, e.g. a paper PDF someone drops into `docs/` later would silently not be tracked. No PDF was tracked before this change, so nothing is lost now.

---

26-10-08, Claude Opus 5.5 — *gitignore narrowed to reports*

C: (transcribed from chat) You are right. Can you change it to pdf's in reports

Changed the `.gitignore` pattern from `*.pdf` to `reports/*.pdf` [decided: Julian, above], so only compiled reports are ignored and PDFs elsewhere in the repository are tracked normally. The side effect noted in *PDF kept out of git* no longer applies. [verified: 26-10-08, `git check-ignore` still ignores `reports/26-10-06_truth_experiments_state.pdf`]
