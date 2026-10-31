# SUMMARY: Writing the tPD truth paper (analysis, explanations, lesson, text)

**Last updated:** 26-10-30 (stage 1a done and read by Julian; plain tPD's failure mode located; candidate lesson L4; side-effect test approved as a staged GPU list, probes first; ortho s3 added to the list, conditions open; nothing run)

Julian opened this topic to turn the finished experiments into the paper: analyse the data, discuss explanations, ideally find a lesson, write it. The experiments, the frozen analysis plan and all run details live in the experiments topic (`convos/julian/26-10-22_tpd_truth_paper_SUMMARY.md`); this topic starts where its phase D ended: all analysis outputs computed and checked (457 invariant checks), no summary tables yet.

## Plan [assumed: proposal, LOG entry *topic opened: state and four-stage plan*]

1. **Analysis** (offline, CPU): one summary script reads the phase D outputs (`~/spd_out/26-10-22_tpd_truth_paper/phase_d/h200_paper_d/`) and writes the frozen plan's tables, with both seeds and controls in every row; then the exploratory val browsing (hypotheses tested on test/OOD before they count); then a results digest.
2. **Explanations**: per finding, the candidate explanations and which number separates them.
3. **Lesson**: kept open until stage 1 is done; candidate lessons are written down now with what would refute each (L1 data-aligned control, L2 the target loss decides what gets decomposed, L3 protocol lesson; L4 added 26-10-30, see *Where plain tPD fails*).
4. **Writing**: draft in `reports/`, results first, intro and abstract last; earlier negative results in an appendix [decided in the experiments topic].

## Stage 1a: tables (done 26-10-29)

Julian: one step at a time; tables and plots first, then he comes back [decided: C: comment under 1a]. `spd/experiments/lm/honesty_targeted_decomposition/paper_tables.py` (CPU, ~20 s) writes `scratch/26-10-29_paper_writing/tables/{tables.md, details.md, wide_val_reverse_<run>.csv, figs/}` from the phase D outputs and the finals' val readout. Every table puts all 9 bases side by side (6 finals, 2 untrained decompositions, data-aligned basis). Stored summaries are recomputed from the per-statement arrays on every run and asserted equal [verified]. KL recovered is shown pooled and macro (aggregation open). Gaps: Pile side effects on val only; activity profiles of the wide tables not tabulated. No interpretation yet, by design.

## Separation shift (added 26-10-30)

Julian asked whether reporting a separation shift alongside the separation share would be easier than reading the shift off the 1→0 / 0→1 flips. Shift = the signed move of the midpoint between the true and false class mean verdict scores, in units of the original gap (> 0 = toward '1'). Together with the share it fixes both class means; the flips mix a closing gap with a shift. On sample conditions the shift was mostly near 0 while the flips were large in both directions (the gap closes, the scores don't move as a whole). recon s1 was the exception (+0.10 to +0.15, flips almost only 0→1). Julian: "Yes, implement it" [decided: transcribed C: line, LOG entry *separation shift implemented*]: shift and |shift| (macro of per-family |shift|, because the signed macro can cancel) added next to the flips, which stay (they are the decision outcome and in the frozen plan). The shift is labelled descriptive, added after the data were read. `paper_tables.py` computes it from the npz arrays for every summary with per-statement scores and asserts the stored separation share first (6,902 summaries equal) [verified]. Not available for T1's val ranking and the 5a direction removal. New figure `figs/m_share_shift.png` leads section 4a.

## Cancellation in plain tPD (26-10-30)

Julian asked whether `tables.md` has data on plain tPD's cancellation. It doesn't; `details.md` does (write norms per layer). Plain s1/s2: components at mask 1 and Δ each write 1.2–2.3 |y| against each other, ⟨C,D⟩/|y|² −3.6 to −14 (mean of per-position ratios); recon, ortho and untrained ≈ 0 [verified: `details.md`]. By the norm-based estimate used for 4a, the paper's plain runs give about −2 to −4 vs 4a's −11 to −183 [concluded: arithmetic], so this is moderate cancellation, not 4a's giant one (set-ups differ, not a controlled comparison). Julian: not needed, `details.md` is fine [C: comment].

## Active-set curve shapes (26-10-30)

Julian asked why plain and recon have dips and sudden jumps in the active-set curve (KL recovered of the top-m components by mean CI) while ortho rises monotonically. Answer [verified: stored `mean_ci_val`, `active_set.curve`, `t1_val`]: the curve orders by mean CI, not importance, and is measured only at m = 1, 2, 4, 8, 16, 32 plus bisection points, so the jumps are straight segments over doublings. ortho: one dominant component that is also the most important (T1 drop 0.47), so the curve is monotone. plain: flat mean CIs (0.02–0.10), and the T1-important components are spread over mean-CI ranks 1–29. Subsets below about 16 components recover less than nothing (−0.55 to −0.91), and recovery reaches 0.89–0.92 at 24–32. Candidate explanation [assumed]: opposing groups, so a one-sided subset biases every verdict. recon: high, broad mean CIs; the T1-important components sit at mean-CI ranks 34–251, so the curve measures how much of the generic MLP output has been rebuilt (dip −0.26/−0.38 at 32, recovery at 128–256). For the paper [concluded]: the curve is not an importance curve. Listed GPU checks, not run: the top-m states with per-statement scores (shift test of the one-sided-subset idea); recovery along the T1 ranking.

## T0 with KL recovered (26-10-30)

Julian: T0 (all components removed, Δ only) should also report KL recovered, to share recovery's measure [C: comment]. Done from stored values (no new run): section 2a now leads with Δ-only KL recovered, pooled and macro with bootstrap intervals [verified: regenerated]. Test/training prompt pooled, Δ only: plain 0.081 / −0.050, recon −0.238 / −0.440, ortho 0.170 / 0.011, untrained 0.888 / 0.984 (components only: 0.89–0.97 for the finals). KL recovered in T2: Julian, "that can wait" [C: comment].

## K1 OOD probes: threshold, not information (26-10-30)

Julian asked why the K1 label/OOD probe accuracy doesn't rise with k. Refit offline (`scratch/26-10-29_paper_writing/k1_ood_probe_check.py`; all 36 stored OOD accuracies reproduced exactly) [verified]: the probe direction transfers and improves with k (within-family AUC, e.g. plain s1 0.80 → 0.96, recon s1 0.57 → 0.96), but the threshold fit on tiu train is off on OOD (plain s1 calls 94–99% of OOD statements true; plain s2 at k = 10 calls 4% true), so accuracy stays flat. ortho s1 is a real failure (AUC ≈ 0.50 at every k). Cause of the offset [assumed]: OOD forms shift activation levels. For the paper [concluded]: report within-family AUC beside accuracy in K0/K1. Proposal (within-family AUC columns and figure row): Julian, "Nah" for now [C: comment].

## Where plain tPD fails (26-10-30)

Julian expected plain tPD's presumed re-implementation to fail some performance metric and saw only hints. Answer [verified: numbers read from `tables.md`/`details.md`; reading [concluded]]: the verdict metrics are measured inside the decomposition (components + Δ = W) and are blind to how the work is split, and removing cancelled components injects large writes, so termination even looks strong. The failure shows in four places, each in both seeds:
1. T2: plain's top-k removal is no worse than random k of its active set (k = 10: 0.248 vs 0.242, 0.393 vs 0.310), while recon/ortho top-k clearly beat random.
2. A8: projecting plain's 96-dim write span out of the original MLP output leaves the separation at 0.97–1.01, the random-subspace level; recon/ortho drop to about the top-PCA level.
3. Write vectors align with the truth direction only at the random level (max |cos| 0.044–0.084 vs 0.042–0.046 expected).
4. Cancellation norms.

Counter-hint: plain recovers best on OOD (0.32–0.37). Candidate lesson L4 [assumed]: within-decomposition metrics can't tell decomposing from re-implementing; original-model-anchored tests and random controls can. GPU test listed (T1 drop vs write norm per component), not run.

## Why Δ alone isn't worse than all-off (26-10-30)

Julian expected plain's large Δ (|D| ≈ 2 |y|, largely in "new directions") to damage the model more than the other methods' Δ. Answer [verified: code and section 2a]: during training on target data Δ's mask is uniform in [0, 1] (components in [CI, 1]), so the verdict must not depend on Δ. To first order, Δ's write is verdict-neutral, so Δ alone ≈ writing nothing = all-off, whatever its norm [concluded: first-order argument]. Data: Δ-only KL recovered plain 0.081 / −0.050; the only clearly negative method is recon (−0.24 / −0.44), which has the smallest Δ. On Pile, Δ is forced to 1 and the components' masks are random in [CI ≈ 0, 1], so Δ alone is nearly a trained state there. The large cancelling writes exist only at the statement tokens [assumed: C on Pile not measured]. Pile Δ-only KL: plain 0.10, ortho 0.02, recon 0.14. Proposed GPU test, not run [assumed]: readouts nothing was trained on (next-token KL within the statement, a recall prompt). If Julian's intuition holds, plain's Δ alone damages these more than all-off; that would be another original-model-anchored test for L4.

**Side-effect test (26-10-30).** Julian: we need a test showing that plain's strange writes have side effects the other methods' writes don't [C: comment]. Design after his comments (LOG entry *side-effect test: revised design after Julian's comments*), approved for the GPU list, with the condition "probes first and then when they are promising further evaluations" [decided: C: comment under that entry]:
- Framing: how close is each method to an ideal truth decomposition, under which the components matter only where truth is computed.
- State: Δ alone, against the original and all-off.
- Readouts: next-token KL within the statement, split into prefix (truth not yet determined), recall position, and object + end; digit questions about other properties (humor, emotion, surprise, word count; Julian's idea), also under the reverse-top10 edit; a Yes/No truth question as a transfer test, whose prompt is iterated first for Yes/No mass (Julian).
- Dropped: matched-verdict-effect comparison, write norms, copy, content and knowledge questions.
- Predictions: ortho near the ideal, recon at all-off by design, plain worse than all-off if its cancelling writes have side effects (the only outcome that separates plain from recon).

(The second prompt is a different truth question with the same 1/0 format, so it tests transfer; an earlier reason given for this was wrong, marked OUTDATED in the LOG.)

## GPU list (26-10-30)

Staged; each stage runs only if the previous one shows signal [decided: Julian, C: comment]. Signal criteria are the agent's proposal, written before any run, awaiting Julian's confirmation [assumed]; details in the LOG (answer under Julian's comment after *side-effect test: revised design…*). Nothing run; code not written.
1. **G0, original model only**: Yes/No prompt iteration (Yes/No mass, accuracy, AUC; val subset); digit-mass check for the humor/emotion/surprise/word-count questions.
2. **G1, probes** (s1 of each method + untrained s1, val subset): next-token KL within the statement (original, all-off, Δ alone; prefix / recall / object + end); digit questions under Δ alone and reverse-top10; Yes/No components-only recovery vs 1/0. Go criteria: plain's Δ-alone KL > all-off (interval excludes 1) while recon's is not; plain's digit-question effect clearly above recon's and ortho's; plain's recovery drop under Yes/No clearly larger.
3. **G2, full evaluation** (only readouts with signal): all 9 bases, test, bootstrap intervals; Yes/No also T2.

4. **Ortho s3** [decided: Julian, C: comments and "Yes ortho only."]: ortho only, same config as the finals, seed 3, full phase D analysis (all numbers), on the same GPU type as phase D (H200 NVL). Seed choice: s3 may replace s1 or s2 in the main figures if it agrees well with the other; Julian picks after seeing all numbers, nothing fixed in advance; the dropped seed is still reported. Agent's suggestion [assumed]: one sentence in the paper that the shown seeds were chosen after seeing the results, all three in the appendix. Background: s1 and s2 match on the training's own numbers. s2 fails second-prompt transfer (test pooled 0.735 vs 0.925); s1's train-selective single components fail on OOD (K1 AUC ≈ 0.5), though s1's whole-decomposition OOD recovery is not lower than s2's (0.198 / 0.291 vs 0.168 / 0.101).
5. **WandB space before s3** [decided: Julian, C: comment]: paper checkpoints are 333.5 MB; the register's 13 checkpoints sum to ≈ 4.0 GB, so one more is tight under the 5 GB cap. Steps: read actual usage via the API before the rental; if the margin is unclear, Julian picks a deletion (candidates with local copies: `s-7fad0c14`, `s-bd23f0d1`, `s-cd129592`); or train with checkpoint sync off and upload in phase E. Add the run-register row at launch.

**Shelf** (Julian: no need yet, not on the GPU list): top-m states of the active-set curve rerun with per-statement scores (tests the one-sided-subset explanation of plain's negative dip); T1 drop vs mask-1 write norm per component (whether plain's removal effects come from write size).

## Open questions for Julian

Venue/audience/format and co-authors; aggregation (macro over families is in the frozen plan but is dominated by families with near-zero all-off KL; proposal: per-family tables, pooled headline, macro alongside); where phase E (registry entries) lives. Deferred by Julian until the tables are read.

## See also

- See also: [convos/julian/26-10-22_tpd_truth_paper_SUMMARY.md] — the experiments, methods, tuning, frozen analysis plan and phase D runs this paper reports.
