# SUMMARY: Steering by reversing component writes (Exp 4b / 4c)

**Last updated:** 26-10-21 (sync: numbers re-checked against the output files, small corrections; current state: flip experiment and gradient mask search done, one seed, 4c/4b only; levers for "true → 0", sentence-form verdicts and per-family inversions found, none of them a belief change; knowledge survives the inversion; follow-ups proposed, none decided)

Julian asked what happens to the model's truth verdict if the write direction of a single component, some components, or all components of the Exp 4 decompositions is reversed. Started as a new topic because the bolder-experiments LOG is long [decided: transcribed from chat]. Run on the vast.ai RTX 4090 [decided: transcribed from chat].

**Decompositions used** (background: `convos/julian/26-10-10_bolder_experiments_SUMMARY.md`, section *Causal layer scan and Exp 4*): Exp 4c `s-5ecb5791` (tPD on the `down_proj` of Qwen2.5-7B-Instruct layers 11–17 at every statement token, verdict lens + per-layer reconstruction term) and Exp 4b `s-717b50c9` (same at the statement end `'.\n'` only). Prompt "<statement>\nThe amount of truth in this statement is "; verdict score log P('1') − log P('0'). When this topic started these were the only runs whose components are a part of the MLP output (no cancellation), so reversing one is a meaningful edit; the later Exp 4e (`convos/julian/26-10-16_faithful_targeting_SUMMARY.md`) also shows no cancellation.

## Method [assumed: agent's design, not discussed]

- **Intervention** [verified: masks are linear in `LinearComponents.forward`]: original weights, except the chosen components get mask −1 at the run's decomposed positions, so their write u_c (v_c · x) is negated (the weight there becomes W − 2 u_c v_cᵀ). The question tokens keep the original weights. Sets were also run at mask 0 (removal) and −2. A gated variant reverses only where the component is active (rounded CI > 0.01).
- **Conditions:** every component active on ≥ 1% of statements, singly (4c: 200, 4b: 45); sets: L17:5, the "false negated statement" group (L15:51, L16:28, L14:34, L15:84, L13:60, L16:20, L11:91), both, top 10 by removal effect, all label-selective (4c: 68, 4b: 17), all alive, all 672. Random sets of matching size from non-selective components as controls (4b's pool has only 14, so its controls are weak).
- **Outcomes** on all 1,794 test statements: accuracy, separation (mean score true − false; original 8.68), T→0 / F→1 = share of correctly answered true / false statements whose score changes sign, digit mass P('0') + P('1') (format), KL, per-family AUC.

## Results [verified: one run per decomposition, one seed each; LOG entry *flip experiment: results*]

- **Sanity checks:** the unchanged state reproduces the original (KL 0.0003), and L17:5 removal reproduces the earlier component analysis (accuracy 0.880).
- **4c, L17:5 reversed:** 42.6% of correctly answered false statements go to "1" (53.8% at ×−2; removal 19.1%), true → "0" only 3.6%, format intact (digit mass 0.988). Random single components: ≤ 0.7% in either direction. With the "false negated" group added: 49.8%.
- **Family-specific:** L17:5 ×−1 flips false → "1" in neg. animal 0.83, neg. cities 0.65, neg. element symbols 1.00, neg. translations 1.00, smaller_than 0.94. In the four affirmative fact families it flips ≤ 0.03.
- **Inversions:** per-family AUC under L17:5 ×−1 drops from 1.00 to 0.12 in negated translations (×−2: 0.04) and from 0.97 to 0.38 in negated element symbols; affirmative families stay at 1.00. Removing the 68 label-selective components already inverts these two families (AUC 0.00 / 0.02). 4b, which edits only the statement end, inverts the same two families when all 45 of its alive components are reversed (neg. element symbols 0.02, neg. translations 0.05; all others ≥ 0.99).
- **Reading** [concluded, alternative open]: these components look like part of the computation that combines negation with the relation's truth, not a family-independent truth writer. An inversion confined to negated families fits "the 'not' is ignored". This revises the earlier description of L17:5 as a "cross-family 'false' writer", which rested on its activity, not on its effect.
- **One-sided:** the steering found pushes false statements to "1" (removing a "false" signal). No single component pushes true statements to "0" selectively. The largest T→0 is L11:70 (0.153), which mostly weakens the verdict (separation 3.56, answers like "0.5").
- **Reversing everything destroys rather than inverts:**
  - 4c, all 672 at ×−1: almost everything goes to "0" (T→0 0.862, separation 0.06, digit mass 0.810).
  - Top-10 and all-alive at ×−1: separation 0.27 / 0.11, flips in both directions.
  - 4b, all 672: mild (separation 5.34), since only one position is edited.
- **Gated ≈ ungated** for all singles and sets except all 672, so the effects sit where the CI network marks the component as active.
- **4b singles and sets** are weak and barely beat its random sets (top-10 F→1 0.313 vs random median 0.138, max 0.215).
- **Example continuations** (12 statements, anecdotal): affirmative statements never flip. When the digit flips, the text after it follows in about 6 of 15 cases and contradicts it in about 7 (e.g. "100%. The Spanish word "uno" does indeed mean "one.""). Factual knowledge survives every intervention; only the verdict on negated or compared statements changes.

## Gradient mask search (26-10-20) [verified: one run, seed 0, 4c only; LOG entries *gradient mask search: design / results* and *recall test*]

Julian rented a 48 GB RTX 4090 [decided: transcribed from chat] and picked the gradient mask search from four offered options (the others: amplify + inject first, train 4d / 4b with reconstruction coefficient 0.1, both gradient search and amplify); the pick is recorded only as an agent's note of his multiple-choice answer, not verbatim.

**Method** [assumed: agent's design]: one learned mask value 1 + d_c per component (all 672), at 4c's decomposed positions and shared by all statements. A found mask is therefore a fixed weight edit. Fit on the tiu train split (Adam, 400 steps), scored on the 1,794 test statements. Sparsified masks (|d| < 0.1 reset to 0) behave like the full ones. Controls: the same values put on random components. Every objective adds an L1 penalty, swept over λ ∈ {0.001, 0.003, 0.01, 0.03}, and a digit-mass term that keeps the answer a digit. Objectives:
- `true0`: true statements → "0", false statements keep their output distribution;
- `twin`: a true affirmative → "0" together with its negated twin → "1" (851 train pairs). This is the belief-shaped objective;
- `invert`: every statement → the opposite digit.

**Results** (test split; random controls: median flip rate ≤ 0.03 in every cell). "Gap" below = mean score of true minus false statements within a family, original 6.3–12.4:
- **true0** (λ = 0.01, 13 components): 94% of correctly answered true affirmatives and 99% of true number comparisons go to "0", false ones stay. All scores drop to the false level; within affirmative families true still scores above false (gap 7.34 → 1.84 in cities). So this is a "say 0" bias that compresses the distinction, not an inversion.
- **twin** (λ = 0.01, 23 components): the trained cells reach their targets (true affirmative → "0" 0.96, false negated → "1" 1.00). The held-out cells don't move (false affirmative → "1" 0.00, true negated → "0" 0.06), so the verdict now follows the sentence form (negated → "1", affirmative → "0"), not the fact. This rules out "relation evaluation inverted".
- **invert** (λ = 0.001, 401 components, |d| up to 6.9): a real inversion in cities (gap 7.34 → −5.23, AUC 0.01), the number comparisons and three of the four negated families (AUC ≤ 0.02; neg. animal class 0.24). It collapses rather than inverts in animal_class, element_symb and sp_en_trans (gap ≤ 1.1). The inverted families dominate the train split [assumed: likely cause, untested]. With 19 components (λ = 0.01) only the comparisons and the same three negated families still invert (AUC 0.00–0.11; cities collapses, gap −0.16).
- **L17:5** is reversed and amplified (d −2.2 to −6.1) in 11 of 12 fits. The fits also amplify other components (d > 0), which the flip experiment never tried.
- **Recall test** (166 test cities, mask as a full weight edit, direct question "In which country is the city of X?"): recall stays at or above the original 0.771 under every invert mask (0.771 / 0.922 / 0.825). After an inverted digit the model mostly states the true fact ("0 out of 1. The statement … is actually true"); it confabulated to match the digit in 1 of 8 examples (Zhabei "in Kandahar"). twin's lower question recall (0.590) is a format change: it answers in sentences cut off by the 12-token window before the country [assumed: from the examples read].
- **Reading** [concluded, one seed, one decomposition]: no belief lever in 4c's components. Everything found changes the verdict readout (bias, sentence form, per-family inversion) and leaves the model's knowledge intact. A single mask shared by all facts can't encode a particular false fact.
- **Caveat from a later topic** [concluded]: the same mask search run on Exp 4e (26-10-21, `convos/julian/26-10-16_faithful_targeting_SUMMARY.md`, *Reprogramming*) found that an untrained basis aligned with the MLP inputs swaps the verdict about as efficiently as the trained components. The control used here (the same mask values on other 4c components) rules out "any component of 4c works" but not "any data-aligned direction works", so the swaps found here need not be specific to 4c's trained components; that basis control was not run for 4c.

Files (`~/spd_out/26-10-16_component_steering/`): `mask_search.py`, `analyse_masks.py`, `recall_test.py`; outputs `mask_search_s-5ecb5791.{log,npz}`, `analysis_masks_s-5ecb5791.out`, `cell_means_s-5ecb5791.out`, `recall_s-5ecb5791.{log,out}`.

The vast.ai RTX 4090 used here (instance 56873443) is gone [verified: `vastai show instances` lists no instances, 26-10-21 sync].

## Next proposed [assumed: not discussed]

Toward a belief lever (after the mask search):
- (a) A fact-specific mask: one mask per fact, fit on a few phrasings of that fact and tested on the direct question (ROME-like).
- (b) A decomposition trained on recall prompts (subject's last token), not on verdict prompts.
- (c) Look in Δ, which carries 73% of the direct truth write.

Earlier proposals, still open:

1. Paired negation test: does the score on a negated statement under L17:5 ×−1 track the original score on its affirmative twin (tiu pairs share subject and object)?
2. Background to the "true → 0" lever (LOG entry *ideas for a true → "0" lever*; the gradient search above came out of it):
   - **Diagnosis:** 4c has almost no "true" components. Only 3 of 200 are noticeably more active on true statements, and none writes along the truth direction. A reversal can't affect statements where the component is off, so reversing false detectors only pushes false → "1".
   - **True/false activity overlap** (26-10-17, answer to Julian) [verified: recomputed from `components_s-5ecb5791.npz`]: 157 of 200 components have true and false activity rates within 0.05 (correlation 0.966), because "active" counts any statement token, most of them before the truth is known. Label-dependent components sit in the negated families, mostly on the false side (false-side on/off components: 9–29 per negated family vs 0–6 in affirmative ones). The "false negated" group fires on "relation holds + not" (e.g. L15:51: 0.92–1.00 on false negated, 0.00 on true affirmatives), which fits Julian's "evaluate the relation, then apply the negation" picture [concluded]. Affirmative truth (AUC 1.00) has almost no on/off components; it is graded, in attention, or in Δ.
   - **Reading vs belief change:** Julian wants the model to *believe* the fact is false, not to *read* a different statement [C: comment]. Everything found so far is a reading change (knowledge survives, no affirmative fact family inverts). Test for a belief change: both twins flip and the answer to a direct question changes. Belief components would be active on true and false alike (subject lookup), so label selection misses them; the gradient search can target them with a twin objective.
   - **Hardware:** the gradient mask search ran on the 48 GB 4090 at 62–95 s per 400-step fit [verified: 26-10-20].
   - **Ideas not run:** amplify the false detectors (mask > 1; the mask search's fits already amplify some components); inject L17:5's write direction into all statements; the true-side within-family set.
3. Second seed of 4c: do an L17:5-like component and the same inversions reappear?

**Files:** `~/spd_out/26-10-16_component_steering/`
- scripts: `flip_components.py` (main), `make_sets.py` (sets → `sets_<run>.json`), `analyse_flips.py`, `family_auc.py`, `flip_examples.py`, `run_flips.sh`;
- outputs: `flips_<run>.{log,npz}`, `analysis_<run>.out`, `family_auc_<run>.out`, `examples_s-5ecb5791.out`.

## See also

- See also: [convos/julian/26-10-10_bolder_experiments_SUMMARY.md] — origin of Exp 4b/4c and of the component analysis (L17:5, the "false negated" group) whose sets are reversed here. The earlier mask −1 "flips" on Exp 3 broke the answer format there, because Exp 3's components re-implement the lens. This topic revises that topic's description of L17:5: its effect is confined to negated statements and number comparisons.
- See also: [convos/julian/26-10-13_direction_free_interventions_SUMMARY.md] — design D3 (steering) proposed there; this topic is a component-based version of it.
- See also: [convos/julian/26-10-16_faithful_targeting_SUMMARY.md] — Exp 4d/4e, the later decompositions without cancellation; its *Reprogramming* section reuses this topic's `mask_search.py` on 4e and adds the untrained-basis control that qualifies the mask-search results here.
