# SUMMARY: Ablation and steering experiments that need no truth direction

**Last updated:** 26-10-13 (design options proposed; nothing decided or run. Sync: no drift; See-also to the state report added)

Julian asked for an ablation or steering experiment that works independently of any truth direction. Motivation: the project's causal lead, A7 (ablating tiu arm A's L17:33 or L19:34 removes 27–31% of the true/false separation along the layer-23 mass-mean direction `mm_L23`), is measured as a projection onto a fitted direction. It can't tell lost truth from a generic shrinkage of the residual stream, and it misses truth that only rotates [concluded].

## Direction-free outcomes

- **Verdict** after " Is this true or false? Answer:" (log P(true-like) − log P(false-like)), i.e. analysis V, `verdict_accuracy.py`, already implemented and run on Exp 2 [verified: `convos/julian/26-10-10_bolder_experiments_LOG.md`, entry *V on Exp 2*]. Use AUC as the main metric, because it ignores an ablation-induced bias towards "True" or "False".
- **Natural continuation** after the bare statement: paired KL(true ∥ false) over same-subject pairs, with the content baseline from gate G. Weak for negated statements.
- **Re-fit probe** on the ablated activations: handles rotation, but sits at the ceiling on tiu. Low priority.

## Designs proposed [assumed: agent proposal, not discussed]

- **D1, necessity:** ablate single candidates (L17:33, L19:34, L16:69, L18:15, L19:54, L18:62, the per-family writers) at all statement and stimulus tokens. V currently replaces at the period only, which understates L17:33: half of its first-order effect comes from earlier positions. Readout: verdict AUC, accuracy, KL. Controls: about 20 random trained components per candidate, matched on activity and on the Pile KL their ablation causes; a non-truth control question; the numbers-vs-facts double dissociation. This is experiment B (`convos/julian/26-10-06_truth_writing_components_SUMMARY.md`) with the verdict as outcome.
- **D2, sufficiency:** interchange patching of one component on true/false pairs of the same subject: add (a_c^true − a_c^false)·U_c in the false run, then measure the share of the verdict gap closed. Controls: true/true pairs (content change without a truth change), matched random components, the same layer's Δ. Handles generic disruption by design.
- **D3, steering:** amplify a component's mask (×2, ×4) or force it on. Natural target: the model's main error, true negated statements called false, and L19:34, whose write is concentrated on negated statements. Weaker evidence than D2.
- **D4:** verdict AUC vs the number of ablated top-active components, against random sets of the same size.

Recommendation: D1 first, then D2. The same code applies to Exp 0–3 once their runs finish.

## Open

- Julian's choice of primary outcome: verdict after the stimulus vs natural continuation.
- The A7 table in `reports/26-10-06_truth_experiments_state.typ` omits L19:54 from the tiu rows (really ablated on fit: estimate −3.35, actual −4.08) and lists L16:75 instead; no wrong numbers. Fix offered, not made.

- See also: [convos/julian/26-10-06_truth_writing_components_SUMMARY.md] — the A7 lead these designs test, the A5 per-family writers used as candidates, gate G (natural-continuation outcome), and experiment B, which D1 extends.
- See also: [convos/julian/26-10-10_bolder_experiments_SUMMARY.md] — analysis V (`verdict_accuracy.py`), the R1 stimulus screen and R2/R3 zero-ablations, and Exp 0–3, whose decompositions the same code would compare.
- See also: [convos/julian/26-10-06_state_report_SUMMARY.md] — the Typst state report whose Experiment 4 section (A1–A7, gate G) motivated the question; its A7 table has the L19:54 omission listed under Open.
