C: (transcribed from chat) I'd like to do an ablation or steering experiment that works independently of any truth directions. Any ideas for such a setup?

(Context from the same chat: Julian had asked what "Estimate (fit)" / "Actual (fit)" mean in the A7 table of `reports/26-10-06_truth_experiments_state.typ`, and whether all of that report's section 8 (Experiment 4) compares the decomposition to the truth direction. Answer given in chat: A1, A2, the write half of A5 and A7 are measured against mass-mean truth directions; A3/A4/A6 are label-trained probes; G uses no decomposition. The agent also noticed that the A7 table's tiu rows omit L19:54, one of the 5 components really ablated on the fit split (estimate −3.35, actual −4.08, `~/spd_out/26-10-06_truth_writers/tiu/report.txt`), and list L16:75 (7th by fit estimate, checked only on test) instead; no number in the table is wrong. The fix was offered, not made.)

---

26-10-13, Claude Opus 5.5 — *direction-free ablation and steering designs*; I have read INDEX.md, the SUMMARYs of `convos/julian/26-10-10_bolder_experiments_SUMMARY.md` and `convos/julian/26-10-06_truth_writing_components_SUMMARY.md`, the last three entries of the bolder-experiments LOG (V on Exp 2, Exp 3), the state report, the docstring of `verdict_accuracy.py`, and the A7 code and tiu report of `truth_writers.py`. Nothing was run.

## Why the current causal evidence depends on a direction

A7's lead (ablating tiu arm A's L17:33 or L19:34 removes 27–31% of the separation along `mm_L23`; L17:33 = component 33 of layer 17's `down_proj`) is measured as a change of S, the mean true minus mean false projection onto a mass-mean direction fit on the unablated model. That readout has two weaknesses [concluded]:
- it counts any shrinkage of the class means along w, including a generic shrinkage of the residual stream, as lost truth (the "generic disruption" caveat);
- it sees only the part of truth that lies along one fixed direction; a component could change how truth is represented (rotate it) without changing S, or vice versa.

A direction-free experiment needs an outcome that the model itself defines. Three such outcomes exist in this project already, and two are already measured.

## Outcome measures that need no truth direction

1. **The model's verdict** after an appended question, " Is this true or false? Answer:" (R1 screen; verdict score = log P(true-like) − log P(false-like) at the last stimulus token). This is analysis V, `verdict_accuracy.py`, already implemented and run on Exp 2 [verified: LOG entry *V on Exp 2*]. It is direction-free: the readout is the model's own output, not a fitted vector.
   - Use **AUC** (and accuracy after re-fitting only the threshold) as the main metric, not the separation of scores: an ablation that shifts every verdict towards "False" lowers raw accuracy without touching discrimination. AUC is invariant to such a bias shift.
2. **The model's natural continuation** after the bare statement (gate G): after a false city statement the model predicts " Is" at p 0.58 vs 0.005 after the true one [verified: G]. Outcome: paired KL(true ∥ false) of the next-token distribution over true/false pairs of the same subject (tiu pairs = same family and subject), with the content baseline (two different true items) as reference. No stimulus needed, so no question of whether the stimulus changes the computation. Weak for negated statements (neg_cities 0.71 vs baseline 1.09).
3. **Probe-free information content**: does the truth label remain *decodable* from the residual stream after the ablation, with a probe re-fit on the ablated activations? Re-fitting removes the dependence on a fixed direction: if truth only rotates, the re-fit probe still finds it. Weakness: at layer 23 everything is near the ceiling (residual probes 0.998 on tiu), so only large losses would show. Lower priority.

## Experiment designs

### D1. Necessity: single-component ablation with a verdict readout (extends V; cheapest)

Ablate each candidate (tiu arm A: L17:33, L19:34, L16:69, L18:15, L19:54, L18:62, per-family writers from A5) at **all statement positions** and at all stimulus positions, mask 0, Δ on, and measure verdict AUC, accuracy and KL to `full`, per family.
- V's single ablations currently replace at the period only. A7 found that half of L17:33's first-order effect (8.8 of 17.1) comes from earlier positions, so a period-only ablation would understate it. R2/R3 found the same at the MLP level: zeroing MLPs 15–19 at the period removes 26% of the verdict separation (R2 sample), at all statement tokens 60% [verified: R2/R3]. So V needs a mode "replace at every statement token" (small change: the position mask).
- Controls, all direction-free:
  - **random trained components**: for each candidate about 20 random components of the same decomposition, matched on activity (share of tokens with CI > 0.01) and, better, on the KL their ablation causes on Pile text (matched overall disruption);
  - **a non-truth verdict**: the same ablation, measured on a control question whose answer does not depend on truth, e.g. " Is this about a number? Answer:" (yes/no by family) or " Does this sentence contain a negation? Answer:". A truth-specific component should hurt the truth verdict and leave the control verdict alone;
  - **double dissociation by family** (from the truth-writing topic's step 1): L18:62 (active only on number comparisons) should hurt verdicts on numbers and not on cities; L15:84 / L16:69 / L18:15 the reverse.
- What it tells us: whether A7's lead survives when "truth" is the model's own judgement rather than a projection. If L17:33's ablation leaves the verdict AUC unchanged while a matched random component also leaves it unchanged, the A7 effect was a representation change that the model's behaviour does not depend on (or one that other routes compensate for).
- Caveat [assumed]: the verdict mostly bypasses MLPs 15–19 at the period, so effects may be small in absolute terms; the matched random-component distribution gives the scale.

### D2. Sufficiency: interchange patching of single components on true/false pairs (strongest)

For a true/false pair of the same subject ("The city of Krasnodar is in Russia." / "… in Ukraine."), run both. In the false run, replace component c's contribution at chosen positions with its value from the true run, i.e. add (a_c^true − a_c^false)·U_c to that `down_proj` output (a_c = V_c·h, the component's inner activation). Measure how far the false statement's verdict (or its natural continuation, outcome 2) moves towards the true statement's.
- Direction-free: the intervention is defined by the component and by the paired input, not by a fitted truth vector. Since each component is rank-1, the patch is a one-dimensional, well-defined edit.
- Outcome: the fraction of the true−false verdict gap closed, per pair, averaged; or the share of pairs whose verdict flips.
- Positions: only tokens after the point where the pair differs (the object word, the period, the stimulus). Earlier positions are identical in both runs, so there is nothing to patch there.
- Controls: matched random components; a pair of two *true* statements with different objects (content change without a truth change) as the baseline for how much any content patch moves the verdict; patching Δ's contribution in the same layer as a reference for "how much this layer carries in total".
- Confound [concluded]: the pair differs in content (Russia vs Ukraine) as well as truth. Patching moves content too; the true/true baseline estimates how much of the effect is content alone. With components rather than whole layers, a component that only encodes the object word should show the same effect on true/true pairs.
- Cost: two forward passes per pair plus one per (component, pair); for ~1,000 test pairs and ~30 candidates plus controls, minutes on a 4090 [assumed: from V's and A7's run times, not measured].

### D3. Steering: amplify or force a component and watch the verdict

Scale component c's mask above 1 (e.g. 2, 4) or force it to 1 where its CI is 0, and measure whether the verdict moves systematically towards "True" or "False", or whether discrimination (AUC) improves.
- Natural target: the model's main error. Most verdict errors are true negated statements called false (neg_element_symb_true 0.07–0.29 under every stimulus, R1) [verified: R1]. L19:34's direct write is concentrated on negated statements (29–30% of S for neg_element_symb and neg_sp_en_trans) [verified: A5]. Question: does amplifying L19:34 on negated statements fix or worsen the errors?
- Direction-free in the sense that the steering vector is the component's own U_c, which tPD learned without labels. It is not label-free in the selection of which component to steer (A5/A7 used labels to pick candidates); that is acceptable because the evaluation does not use a direction.
- Weaker than D2 as evidence: amplification out of the training range can produce arbitrary effects. Use the same matched random-component controls.

### D4. Run-level ablation: all components vs matched random sets

The coarsest version: "components only" vs "Δ only" vs "all off" is already in V. Add: ablate the top-k components ranked by an *outcome-free* criterion (activity on tiu, or CI mass) vs k random components, and plot verdict AUC against k. Shows whether the decomposition concentrates verdict-relevant computation in its most active components. Cheap once D1's code exists.

## Recommendation [assumed: not discussed with Julian yet]

D1 first, then D2.
- D1 is a small extension of V (position mask "all statement tokens", candidate list, random-component controls, one control question). It directly tests the A7 lead with a direction-free outcome and doubles as the controlled ablation experiment B proposed on 26-10-06 (`convos/julian/26-10-06_truth_writing_components_SUMMARY.md`), with the model's verdict instead of S as the outcome.
- D2 answers the stronger question (does a single component *carry* truth from one statement to its counterpart) and handles the generic-disruption objection by design: a patch from the true twin is not a disruption.
- The same code works for Exp 0–3 once their runs are in, so it compares decompositions trained with different target lenses on a common behavioural outcome.

Open question for Julian: which outcome he wants as primary, the verdict after the stimulus (well calibrated by R1, but the stimulus adds tokens the decomposition did not see in arm A) or the natural continuation (no stimulus, but weak for negated statements).
