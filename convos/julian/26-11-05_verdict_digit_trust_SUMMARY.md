# SUMMARY: Can the verdict digit be trusted?

**Last updated:** 26-11-05 (200-statement check done: the verdict digit is trustworthy in-distribution, the only effect is "0.5" hedges at near-ties; OOD and interventions not covered)

The paper's verdict is the first answer token of "<statement>\nThe amount of truth in this statement is ", read as s = log P('1') − log P('0'). Julian asked whether that digit is the model's answer: Qwen writes numbers digit by digit, so a '1' may start "10%" (mostly false) and a '0' may start "0.99" (mostly true). Question, sharpened by Julian: P(the full answer means the same as the digit | the first token is the verdict digit).

## What was known before

**Not checked systematically.** Earlier checks covered only digit mass (P('1') + P('0'), 0.984 pooled in-distribution) and accuracy/AUC of sign(s). Only 6 unsteered greedy continuations existed ("1.0 (completely true).", "0%.", "100%.", "0 out of 1.", "100%.", "0."), all consistent with their digit; under interventions "0.5" hedges appear [verified: `~/spd_out/26-10-10_bolder_experiments/steering_sf/full/examples.out`, anecdotal].

## Results so far

- **Confidence between the digits** (a first answer to a misreading of the question, still useful): q = P('1')/(P('1')+P('0')) is beyond 0.1/0.9 for 80% of in-distribution and 42% of OOD statements; errors are mostly near-ties [verified: B2 scores, `~/spd_out/26-10-22_tpd_truth_paper/b2_full/b2_scores.npz`]. Accuracy-type claims on OOD cut through a soft distribution; KL-based metrics are unaffected.
- **Smoke test, next-token probabilities after the digit** (10 statements, all false by a sampling bug since fixed) [verified: `~/spd_out/26-11-05_evaluation_categories/smoke.log`]: hedges like "0.5" carry some mass on false negated statements (≈ 0.13 for one neg_element_symb statement), and the small '1' mass on false statements is partly "10%" (≈ 81% of it for one cities statement). The sign of s was unaffected in these 10.
- **Greedy can mislead** [concluded: from the smoke test]: the same meaning is split over merged tokens ('%.' vs '%.\n\n'), so on one statement greedy writes "0.5" (≈ 0.18 of the '0' branch) while plain-false readings hold ≥ 0.63. Hence Julian's proposal (greedy + manual check) was replaced by sampling.

## Result: sampled continuations on 200 statements (26-11-05)

Method [assumed: agent design, run on Julian's go-ahead in chat ("yes, run this"), not ratified as a method]: `~/spd_out/26-11-05_evaluation_categories/verdict_trust.py`. 200 in-distribution test statements (10 per `_true`/`_false` dataset); first token forced to the verdict digit; 10 continuations of 10 tokens sampled at temperature 1 (hand-written loop, no top-k/top-p) plus greedy; regex-parsed into a truth value; all contradicting, hedging, unclear and partial samples read by eye. 36 min on the CPU.

Result [verified: `trust_full_analysis.out`, one run, 2,000 samples]: **the digit can be trusted in-distribution.**
- After '1' (96 statements): 99.6% of continuations mean "true". The rest: one "1/3", one "1 (on a scale of 1-10", three degenerate ("1.2", "12.", "110%"). No "10%".
- After '0' (104 statements): 96.1% "0"/"0%", 1.2% on the false side ("0.1"–"0.4"), **2.4% "0.5" hedges**, 0.4% contradict ("0.6", "0.63"). No "0.7"–"0.9".
- Hedges sit at the near-ties: hedge share 0.145 for the 11 statements whose larger digit probability (renormalised over '0'/'1') is < 0.7, against 0.003 for the 152 at ≥ 0.9. So s already shows them; at the near-ties P('0') contains a small "undecided" share that s scores as false.
- Greedy: 196 agree, 4 hedges, no contradiction.

Not covered: OOD (much softer verdicts), decompositions and interventions (where "0.5" answers were seen), and the minority digit's meaning (affects the size of s, not its sign).

## See also

- See also: [convos/julian/26-11-05_evaluation_categories_SUMMARY.md] — where this question came up: the definition of the model's truth value τ (as the verdict) for the swap properties P4–P5.
- See also: [convos/julian/26-11-04_paper_draft_SUMMARY.md] — the paper defines the verdict score s and reports verdict accuracy (Table 1); the result here says how far that readout is the model's answer.
