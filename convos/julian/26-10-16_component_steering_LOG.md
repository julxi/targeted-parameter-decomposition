C: (transcribed from chat) I'd like to try to use the decompositions of experiment 4 for some "steering". Can you check what happens if we reverse the write direction of a single component some components or all?
Can you start this in a new logfile since the current "bolder experiments" is already quite long

C: (transcribed from chat) oh, and you can use the vastai RTX

---

26-10-16, Claude Opus 5.5 — *flip experiment: design and launch*

I have read INDEX.md, the bolder-experiments SUMMARY (`convos/julian/26-10-10_bolder_experiments_SUMMARY.md`; it holds the background for everything below), the LOG's component-analysis entry (the "false negated" group), the component analysis script and its outputs for 4b and 4c (`~/spd_out/26-10-10_bolder_experiments/exp4/full/components/`), and `docs/CODING.md`.

**Background in two lines.** Exp 4b (`s-717b50c9`) and Exp 4c (`s-5ecb5791`) are tPD runs on the `down_proj` of Qwen2.5-7B-Instruct's layers 11–17 with a verdict lens (KL at the answer digit of "<statement>\nThe amount of truth in this statement is ") plus a per-layer reconstruction term; 4b decomposes only the statement-end token `'.\n'`, 4c every statement token. They are the only decompositions in this project whose components are a part of the MLP output rather than a re-implementation of the lens (cancellation test: no cancellation), which is what makes "reverse a component's write" a meaningful edit here; in Exp 0–3 a single component's write was mostly undone by the others.
OUTDATED (26-10-21): "the only decompositions … whose components are a part of the MLP output" no longer holds; the later Exp 4e decompositions (global orthogonality term instead of reconstruction) also show no cancellation, see `convos/julian/26-10-16_faithful_targeting_SUMMARY.md`.

**What "reverse the write direction" means in the code** [verified: `LinearComponents.forward` in `spd/models/components.py` computes `((x @ V) * mask) @ U`, linear in the mask]. Start from the original model (every component mask 1, Δ on; this equals the original weights), then set the chosen components' mask to −1 at the run's decomposed positions. Component c's write `u_c (v_c · x)` becomes its negative, i.e. that layer's weight becomes W − 2·u_c v_cᵀ there. All other positions (the question tokens) run on the original weights, as in training.

**Conditions** (my design [assumed: not discussed with Julian; he asked to "check what happens"]):
- *Single*: every component active on ≥ 1% of test statements (4c: 200, 4b: 45), reversed (mask −1).
- *Sets*, each at mask 0 (removal, the reference from the component analysis), −1 (reverse) and −2 (stronger push):
  - 4c: L17:5 (the cross-family "false" writer, removal −15% of the verdict separation), the "false negated statement" group (L15:51, L16:28, L14:34, L15:84, L13:60, L16:20, L11:91), both together;
  - both runs: the 10 components whose removal lowers the verdict separation most; all components that are on/off by label within some family (4c: 68, 4b: 17; from `family_selectivity_*.out`); all alive components; **all** components of all 7 layers (4c and 4b: 7 × 96 = 672; reversing all = weight Δ − Σ components instead of Δ + Σ components).
- *Gated* variant of every single and set at −1: the reversal applies only where the component is active (rounded CI > 0.01), the mask stays 1 elsewhere. The ungated version is a pure weight edit; the gated one asks whether the component's effect sits where the CI network says it is used.
- *Random controls*: 10 random draws of the same sizes as the named sets, from the components that are neither on/off nor graded by label (4c: 128 of them; 4b: only 14, so its random controls are weak).

**Measured per condition** (all 1,794 test statements): verdict accuracy; separation = mean score of true minus false statements, score = log P('1') − log P('0') at the final space; mean score of true and of false statements; flip rates among statements the original answered right (T→0: true now scored < 0; F→1: false now scored > 0); digit mass P('0') + P('1') (does the answer format survive); KL from the original answer distribution. Per-statement scores are saved, so per-family breakdowns can be made afterwards.

Sanity checks built in: the unchanged state (all masks 1) must reproduce the original verdicts (assertion); mask 0 on L17:5 should reproduce the component analysis's removal numbers (separation −1.30, accuracy 0.880).

Script: `~/spd_out/26-10-16_component_steering/flip_components.py`; sets built by `make_sets.py` there (`sets_<run>.json`); runner `run_flips.sh`. Running on the vast.ai RTX 4090 (48 GB) that was already up, 4c first, then 4b; outputs `flips_<run>.{log,npz}` in the same folder on the instance (to be copied back).

What I expect [assumed: predictions]: reversing L17:5 should push false statements towards "1" (it is more active on false statements and its removal lowers the separation), but a single rank-1 component writing 22% of the object-token truth gap probably moves the verdict only on part of the statements; reversing all 672 components moves each layer's output by −2× the components' part (|A| is 0.8–1× |y| at the end token in 4b, 0.7× at the first token in 4c), which may break the answer format rather than steer.

---

26-10-16, Claude Opus 5.5 — *flip experiment: results*

Both runs finished on the 4090 (exit 0; 4c 479 conditions, 4b 125; about 5.5 s per condition). Outputs copied to `~/spd_out/26-10-16_component_steering/`: `flips_<run>.{log,npz}`, `analysis_<run>.out` (sets vs random, per-family flip rates, single rankings; `analyse_flips.py`), `family_auc_<run>.out` (per-family verdict AUC; `family_auc.py`), `examples_s-5ecb5791.out` (greedy continuations; `flip_examples.py`). All numbers below are re-read from these files [verified: one run per decomposition, one seed each, all 1,794 test statements].

**Sanity checks passed.** Decomposed positions printed for one prompt: exactly the statement tokens (4c), answer read after the final space. The unchanged state reproduces the original (mean KL 0.0003, accuracy 0.978 vs 0.976: bf16). Removing L17:5 reproduces the component analysis (accuracy 0.880, separation 7.37 vs 8.65 − 1.30 = 7.35).

Reference: original accuracy 0.976, separation 8.68 (mean score of true statements +4.86, false −3.82), digit mass 0.985. "T→0" / "F→1" = share of the correctly answered true / false statements whose score changes sign.

**4c (`s-5ecb5791`), sets** (ungated; random = 10 draws of the same size from the 128 non-selective components, median [max]):

| set (size) | ×0 (removal) F→1 | ×−1 sep | ×−1 T→0 | ×−1 F→1 | ×−2 F→1 | digit mass ×−1 | random ×−1 T→0 / F→1 |
|---|---|---|---|---|---|---|---|
| L17:5 (1) | 0.191 | 5.59 | 0.036 | **0.426** | 0.538 | 0.988 | 0.000 [0.007] / 0.001 [0.002] |
| "false negated" group (7) | 0.143 | 7.35 | 0.003 | 0.212 | 0.233 | 0.990 | 0.001 [0.009] / 0.002 [0.005] |
| L17:5 + group (8) | 0.353 | 4.39 | 0.018 | **0.498** | 0.581 | 0.992 | 0.001 [0.031] / 0.002 [0.008] |
| top 10 by removal (10) | 0.468 | 0.27 | 0.439 | 0.426 | 0.643 | 0.870 | 0.004 [0.019] / 0.002 [0.007] |
| label-selective (68) | 0.585 | 1.07 | 0.077 | **0.706** | 0.710 | 0.983 | 0.061 [0.221] / 0.003 [0.026] |
| all alive (200) | 0.381 | 0.11 | 0.582 | 0.279 | 0.518 | 0.755 | – |
| all 672 | 0.386 | 0.06 | **0.862** | 0.073 | 0.056 | 0.810 | – |

- The targeted sets push false statements to "1" far beyond random sets of the same size, with the answer format intact (digit mass ≥ 0.98). Reversal adds to removal: L17:5 0.191 → 0.426 → 0.538 at ×0 / ×−1 / ×−2.
- Large sets don't steer, they destroy: top-10 and all-alive at ×−1 bring the separation to 0.27 / 0.11 with flips in both directions and lower digit mass. **Reversing all 672 components** collapses almost every statement to "0" (mean score true −0.74, false −0.80; T→0 0.862), separation 0.06, digit mass 0.810.
- **Gated ≈ ungated** for every single component and every set except all 672 (gated: separation −0.04, T→0 0.458, F→1 0.483, the same as gated all-alive). So the effects sit where the CI network says the component is active; reversing the 472 components that are never active matters only through their small writes summed over all statement tokens.

**Per family: steering is confined to negated statements and number comparisons** (`analysis_s-5ecb5791.out`, `family_auc_s-5ecb5791.out`). Under L17:5 ×−1, F→1 is 0.83 (neg. animal class), 0.65 (neg. cities), 1.00 (neg. element symbols), 1.00 (neg. translations), 0.94 (smaller_than), 0.08 (larger_than), and ≤ 0.03 in the four affirmative fact families. The "false negated" group ×−1 gives 0.89 / 0.90 / 0.07 / 0.90 in the negated families and ≤ 0.02 elsewhere, matching its activity profile.

**Per-family AUC shows inversions, not only shifts.** Original AUC is 1.00 in every family except neg. element symbols (0.97).

| condition (4c) | neg. animal | neg. cities | neg. element symb. | neg. translations | smaller_than | affirmative families |
|---|---|---|---|---|---|---|
| L17:5 ×0 | 1.00 | 1.00 | 0.65 | 0.64 | 1.00 | 1.00 |
| L17:5 ×−1 | 0.83 | 0.97 | **0.38** | **0.12** | 0.88 | 1.00 |
| L17:5 ×−2 | 0.54 | 0.85 | 0.26 | **0.04** | 0.59 | 0.99–1.00 |
| L17:5 + group ×−1 | 0.70 | 0.37 | 0.28 | 0.11 | 0.86 | 1.00 |
| label-selective ×0 | 0.73 | 0.24 | **0.02** | **0.00** | 0.37 | 0.99–1.00 |

Reversing the single rank-1 component L17:5 inverts the verdict ordering in negated translations (AUC 0.12) and partly in negated element symbols, while the affirmative versions of the same facts stay at 1.00. Removing all 68 label-selective components already inverts those two families completely (0.00 / 0.02), with no reversal needed.

Reading [concluded, with the alternative open]: an inversion confined to negated families is what "the model evaluates the statement as if the 'not' weren't there" would produce. So L17:5 and the label-selective set look like part of the computation that combines negation with the relation's truth, not a family-independent "truth" writer. The earlier name for L17:5, cross-family 'false' writer, rested on its activity (on for false in 8 of 10 families); its causal effect is confined to negated statements and smaller_than. Whether the intervened model really ignores the "not" can be tested with the tiu pairs: negated and affirmative statements share subject and object, e.g. "The Spanish word 'uno' does not mean 'one'." / "… means 'one'.". Not done yet.

**Single components (4c).** 13 of the 200 single reversals flip ≥ 5% of statements in either direction; random singles flip at most 0.7%. Strongest F→1: L17:5 0.426, L11:46 0.334 (separation drops only 8.68 → 8.30), L15:52 0.211, L14:56 0.162, L11:29 0.145. Strongest T→0: L11:70 0.153 (separation 3.56, digit mass 0.944, i.e. it mostly weakens the verdict), then L14:56 0.066, L13:5 0.051. No single reversal pushes true statements to "0" selectively. The steering found here is one-sided: towards "1" for false statements, as if a "false" signal were removed. True → "0" happens only through the inversions in negated families.

**4b (`s-717b50c9`, statement end only).** Much weaker and barely specific:
- Top-10 at ×−1 gives F→1 0.313, but random sets of 10 give 0.138 median [max 0.215]. The pool of non-selective components has only 14 members, and the end-token components are all needed there, so this control is weak.
- Strongest single reversal: L15:36, F→1 0.150.
- All 45 alive components at ×−1: separation 4.96, F→1 0.242, and **AUC 0.02 in neg. element symbols and 0.05 in neg. translations**, every other family ≥ 0.99. These are the same two families 4c inverts, from interventions at a different position set.
- All 672 at ×−1: separation 5.34, digit mass 0.970. Mild compared with 4c, since only one position is edited.

**Example continuations** (`examples_s-5ecb5791.out`; 12 statements = the first correctly answered true and false statement of 6 families, greedy 24 tokens, intervention at the statement tokens only) [verified: read all; anecdotal]:
- Affirmative statements never flip under L17:5, the group or both.
- Where the digit flips under L17:5, the group or both (15 cases; L11:70 flips one more, into a "0.5" hedge), the text after it follows the flip in about 6, contradicts it in about 7, and is unclear in 2 (my count).
  - Follows: "Fifty-four is smaller than fifty-two." → "100%, as fifty-four is indeed smaller than fifty-two"; "The Spanish word 'o' does not mean 'octopus'." under L17:5 ×−2 → "0 out of 1. The Spanish word "o" does indeed mean "octopus.""
  - Contradicts: "The Spanish word 'uno' does not mean 'one'." → "100%. The Spanish word "uno" does indeed mean "one."" (all four conditions); "The city of Lu'an is not in China." → "100%. The statement … is entirely false."
- The model's knowledge (what 'uno' means, where Lu'an is) survives every intervention; what changes is the verdict on the negated or compared statement.
- L11:70 ×−1 often turns the answer into a hedge ("0.5").

Prediction check: I expected reversing all 672 components to break the format rather than steer. It collapsed the verdict to "0" with the format mostly intact (digit mass 0.810), so partly right. I expected L17:5 to steer only part of the statements: right overall (0.426), but the per-family picture (0.65–1.00 in negated families, ~0 in affirmative ones) was not predicted.

**Summary of what reversal does** [concluded, one seed]:
1. Reversing small targeted sets of 4c components steers in a controlled way: one component (L17:5) moves 43% of the correctly answered false statements to "1" (54% at ×−2), keeps the format, and needs nothing at the question tokens. Random components of the same count do nothing.
2. The steering is family-specific (negated statements, smaller_than) and one-sided (false → "1"). In two negated families it inverts the verdict, which fits "negation ignored" better than "truth flipped".
3. Reversing everything destroys the verdict (4c: collapse to "0") rather than inverting it. The decomposition's components are not a single "truth axis" that flips as a whole.

Proposed next [assumed, not discussed]:
- (a) The paired negation test above: does the score under L17:5 ×−1 on a negated statement track the original score on its affirmative twin? (CPU-sized, can reuse the stored npz plus one original pass on the affirmative twins.)
- (b) Look for a "true → 0" lever: reverse components that are active on true statements of the affirmative families (the true-side selective ones are weak, label AUC ≤ 0.56).
- (c) A second seed of 4c, to see whether an L17:5-like component and the same inversions reappear.

The vast.ai 4090 is still running (idle since the examples finished). I have not destroyed it.

---

C: (transcribed from chat) Any ideas how it could be possible to find a true -> "0" group?

26-10-16, Claude Opus 5.5 — *ideas for a true → "0" lever*

**Diagnosis first: 4c may contain no such group.** Pooled over all test statements (`components_s-5ecb5791.json`), only 3 of the 200 alive components are active on true statements at least 0.10 more often than on false ones: L16:35 0.81 vs 0.65, L13:47 0.77 vs 0.66, L11:32 0.44 vs 0.34. Their direct writes along the truth direction after L17 are ≤ 0.003 of the gap. The components that write truth-relevant signal are "false" detectors (L17:5, the "false negated" group). Within families, the true-side on/off components (e.g. L16:35 in neg. animal class T 1.00 / F 0.00, L15:79, L12:10, L11:77 in sp_en_trans) did nothing as single reversals (T→0 ≤ 0.024) [verified: re-read 26-10-16].  
OUTDATED (26-10-21): "4c may contain no such group" — a set that sends true statements to "0" while false ones keep their verdict does exist: the gradient mask search's `true0` mask (13 components at λ = 0.01: 8 amplified, 5 weakened or reversed, L17:5 among them) does it on the test split, though as a "say 0" bias that compresses the true/false distinction rather than a true-side detector; see entry *gradient mask search: results*. The activity numbers in this paragraph still stand.
C: how do I read "3 of the 200 alive components are active on true statements at least 0.10 more often than on false ones". Is that a statement about that basically every component that is active on truth is also active on false? If so then it could just be that the false mechanism somehow builds ontop of the true mechanism. But actually I'm not quite sure myself what I mean by that. In the case of negations it makes sense that the model first determines the truth value of the unnegated sentence and then inverst this truth (surely this is overly simplistic). Then I don't know how the truth value of "The Capital of France is Berlin" should be construct from a true value...

Answer (26-10-17, Claude Opus 5.5): Yes, that is the right reading, with one caveat about what "active" means. Numbers [verified: recomputed 26-10-17 from `~/spd_out/26-10-10_bolder_experiments/exp4/full/components/components_s-5ecb5791.npz`, Exp 4c, all 1,794 test statements]:
- Of the 200 alive components, 157 have activity rates on true and on false statements within 0.05 of each other. Across components, the two rates correlate at 0.966. 55 components are active on ≥ 20% of true statements; 53 of those are active on false statements at least 0.8× as often.
- The few differences mostly go one way: 28 components are active on false statements at least 0.10 more often, 3 on true ones.
- Caveat: "active" means rounded CI > 0.01 at *any* statement token (4c decomposes all of them). Most of those tokens come before the model can know the truth (subject, relation words). A component that fires on every statement's subject counts as "active on true and on false". So the pooled rates mostly say that which components fire depends on the statement's content (family, words), not on its truth.

Within families the picture is sharper. Here "on/off" means the activity rate differs by ≥ 0.5 between the family's true and false statements:

| family | true-side on/off | false-side on/off |
|---|---|---|
| animal_class / neg. | 5 / 2 | 0 / 16 |
| cities / neg. | 1 / 1 | 1 / 22 |
| element_symb / neg. | 0 / 12 | 1 / 9 |
| sp_en_trans / neg. | 8 / 3 | 6 / 29 |
| larger_than, smaller_than | 2, 0 | 0, 4 |

So the label-dependent components sit in the negated families and are mostly on the false side. That fits your picture for negations. A false negated statement is a true relation plus "not" ("The city of Lu'an is not in China."). Take L15:51 from the "false negated" group: it is active on 1.00 / 0.92 / 1.00 / 1.00 of the false negated statements in the four negated families, and on 0.00 of the true affirmative statements in the four matching affirmative families. True affirmatives are the same "relation holds" case without the "not". So L15:51 is not a "the relation holds" detector; it fires on the conjunction "relation holds and negated". Reversing it makes the model act as if the "not" were missing. In that sense the "false negated" signal is built on top of "relation true": the extra components only appear when both are present [concluded: from activity plus the reversal results. The paired negation test (a) would check it directly].

For affirmative statements ("The capital of France is Berlin") I would not expect false to be built from true. The usual picture of factual recall is a comparison [assumed: from the interpretability literature, not tested here]. The model retrieves what it knows about the subject (France → Paris) and compares it with the stated object. Match gives true, mismatch gives false, and neither is built from the other. In 4c's affirmative families there are almost no on/off components (0–8 per family), although the model separates true from false perfectly there (AUC 1.00). So for affirmatives the comparison is not split into on/off components in MLPs 11–17. It is graded within CI values, done by attention (not decomposed), or held in Δ (Δ holds 73% of the direct truth write at the object token).

A mechanism explains the asymmetry. Reversal multiplies the component's write u_c (v_c · x) by −1, and a component that is off on true statements has v_c · x ≈ 0 there, so reversing it changes nothing on them. A false detector can therefore only be removed from false statements, which pushes them to "1"; it can't be put onto true statements. The model does have a true → "0" lever outside the components: in the bolder-experiments steering run on this prompt, adding the native truth direction negatively at the statement end flipped true → "0" in 70–100% of cases (`convos/julian/26-10-10_bolder_experiments_SUMMARY.md`), and 73% of the direct truth write is in Δ, not in components.

So the ideas fall into "use the false detectors differently" and "look where 'true' lives".

1. **Amplify instead of reverse (mask > 1)** [cheapest; one line]: run the sets at ×+2, ×+4. Where L17:5 is on for true statements too (negated families, number comparisons: active on true 1.00 there), amplification should push true → "0", the mirror of reversal. Prediction [assumed]: works there, does nothing on affirmative cities / translations, where L17:5 is off on true statements (active on true 0.04 / 0.44).  
C: That might work, yes. But it might not quite what I am hoping for. I was hoping to get the model believe "Paris is not the Capital of France" instead of making the model believe it read "Paris is not the Captial of France". But still better than nothing.

   Answer (26-10-17, Claude Opus 5.5): Agreed. The distinction applies to every idea in this list, so it helps to name both targets:
   - *reading change*: the model evaluates a different statement, e.g. drops or adds a "not". Reversing L17:5 is this. Amplification acts on the same components, so expect the same kind of effect.
   - *belief change*: the model thinks Paris is not the capital of France.

   A test that separates them: a belief change should flip both twins at once ("Paris is the capital of France" → "0" **and** "Paris is not the capital of France" → "1"), and it should change the answer to a direct question ("What is the capital of France?"). A reading change flips one twin and leaves the direct answer alone. Everything found so far is a reading change. The continuations kept the knowledge under every intervention ("uno does indeed mean one"). No existing set inverts an affirmative family either. The lowest AUCs in the four affirmative fact families and larger_than are 0.47 (cities, all alive ×−2) and 0.46 (larger_than, top 10 ×−2), i.e. destroyed rather than inverted. smaller_than does invert (0.16, label-selective ×−2), but it already behaved like the negated families above [verified: `family_auc_s-5ecb5791.out`, re-read 26-10-17].

   Where a belief lever could sit: in the literature, factual recall localizes in mid-layer MLPs at the subject's last token, and rank-1 edits there change the stored fact (ROME, Meng et al. 2022) [assumed: shown for GPT-type models, not checked for Qwen]. 4c decomposes MLPs 11–17 at every statement token, the subject included, so such components could be in it. They would look like the components in the first answer above: active on true and false alike, because the lookup depends on the subject, not on the truth. That is why selecting by label never picks them. Idea 3 can target belief directly with a twin objective: on the train split, lower the score of true affirmatives and raise the score of their negated twins together. The direct-question answer then serves as the test.
2. **Inject a false detector's write into true statements** [cheap]: add α · u_c at the statement tokens (at the object token, where L17:5 writes most), with α = L17:5's mean activation on false statements, for every statement. This is activation steering along the component's write direction, so no CI gating, and it reaches affirmative families where L17:5 is off. It also tests whether u_c itself means "false" or whether its effect needs the context it normally fires in.
3. **Gradient search for a true → "0" mask** [principled, more work]: on the *train* split, compute d(score)/d(mask_c) per component on true statements (one backward pass per batch, attribution-patching style), or optimise a continuous mask on true statements to lower the score with an L1 penalty and a term that keeps false statements and digit mass unchanged. Then test the chosen set on the test split against random sets. Choosing on train and testing on test avoids selecting on the outcome. If the best set found this way is weak, that confirms the components hold no "true" lever.  
C: That sounds more promising. Does this need H100 or RTX 4090 works too?

   Answer (26-10-17, Claude Opus 5.5): The 4090 works [assumed: an estimate, not measured]. The model, the CI networks and the components stay frozen. All that is computed is one gradient per component (672), or a continuous mask of 672 values is optimised. Memory: Qwen2.5-7B in bf16 is about 15 GB (7.6 B parameters × 2 bytes), plus the stored activations of one batch of ~30-token prompts for the backward pass. For comparison, *training* 4c took 35 GB on the H100, and that also held the CI networks, the component gradients, optimizer state and PPGD. So the 48 GB 4090 fits with a comfortable batch, and a 24 GB card would need small batches. Time: a forward pass over all 1,794 test statements took about 5.5 s per condition on the 4090, and a backward pass costs about two forward passes. Per-component gradients over the train split would then take minutes, and a mask optimisation with a few hundred steps under an hour. The 48 GB 4090 was idle but still rented at the end of 26-10-16; I have not checked whether it still is.
4. **True-side within-family set** [cheap]: reverse together all components that are on/off by label on the true side in some family (L16:35, L13:47, L15:79, L12:10, L11:77, L12:70, L15:46, …). The singles did nothing, but redundancy may hide a group effect (as with the "false negated" group, whose singles were also small).
5. **Look in Δ / a new decomposition** [larger]: if 1–4 find nothing on affirmative statements, "true" is the absence of a falsity signal or is carried in Δ. Two ways to test: project Δ's write at the end/object token on the truth direction per layer to see where the true-side write sits; or train a 4c variant whose loss pulls more of y into components (higher reconstruction coefficient, or more components) and repeat the search.

Recommendation [assumed]: 1 and 2 first (minutes on the 4090; they decide whether the false detectors work both ways), then 3 if a true → "0" lever is still missing in affirmative families. Not run; waiting for Julian.

---

C: (transcribed from chat) OK, I've rendet an RTX 4090 for the experiments

(Asked which experiment to run first, Julian picked "Gradient mask search" from four options: gradient mask search, amplify + inject first, train 4d / 4b@0.1, both gradient and amplify.)

---

26-10-20, Claude Opus 5.5 — *gradient mask search: design*; I have read INDEX.md, this topic's SUMMARY and the LOG entries above, the faithful-targeting and vast-rentals SUMMARYs, `docs/CODING.md` and `flip_components.py`

Instance: vast.ai 56873443, RTX 4090 48 GB (49,140 MiB), power limit 450 W, $0.69/h, ssh alias `vastai` [verified: `nvidia-smi` and `vastai show instances`, 26-10-20]. Fresh disk, so the model and the 4c checkpoint download first.

Script: `~/spd_out/26-10-16_component_steering/mask_search.py` (on the instance in the same path). Decomposition: Exp 4c `s-5ecb5791` only; 4b was weak in the flip experiment and edits a single position.

**Intervention.** As in the flip experiment, ungated: every one of the 672 components gets mask 1 + d_c at 4c's decomposed positions (all statement tokens), the question tokens keep the original weights. One learned d_c per component, shared by all positions and statements, so a found mask is a fixed weight edit W + Σ d_c u_c v_cᵀ at those positions. d = 0 is the original model (checked at run time).

**Fit on train, score on test.** The mask is optimised on the tiu train split (Adam, lr 0.05, 400 steps, batch 32) and evaluated on all 1,794 test statements, so a set is never scored on the statements it was chosen on.

**Objectives** (binary cross-entropy of the verdict toward a target digit: toward "0" = softplus(s), toward "1" = softplus(−s), with s = log P('1') − log P('0')):
- `true0`: true statements → "0", and false statements keep their original output distribution (KL term). This is Julian's literal question. Without the KL term the known trivial lever would win: reversing all components collapses every verdict to "0".
- `twin`: in the four fact families that have negated twins (animal_class, cities, element_symb, sp_en_trans), true affirmatives → "0" **and** their negated twins → "1" (e.g. "The lobster is a crustacean." → "0", "The lobster is not a crustacean." → "1"). This is the belief-shaped test from the *ideas* entry: a model that stops believing "X is Y" flips both twins. Twins pair exactly by removing the negation (851 train pairs; every pair in the same split) [verified: pairing script, 26-10-20]. Held out of this objective: the other twin pair (false affirmative / true negated) and the number comparisons. How they move separates three readings:
  - relation evaluation inverted: false affirmatives → "1", true negated → "0" too;
  - negation ignored plus something else: the held-out negated statements move with their affirmative twins' original verdicts;
  - verdict destroyed: everything drifts to one digit or to a hedge.
- `invert`: every statement → the opposite digit ("truth flipped", if such a lever exists).
- All three add −log(P('0') + P('1')), so the answer stays a digit, and an L1 penalty λ Σ|d_c|, swept over λ ∈ {0.001, 0.003, 0.01, 0.03}, which trades the size of the set against its effect.

**Evaluation per fit:** the full learned mask; the sparsified mask (|d_c| < 0.1 reset to 0); and 10 random controls that put the sparsified mask's nonzero values on random other components (same count, same values). Outcomes as before (accuracy, separation, digit mass, KL), with the flip rates split by group: affirmative fact families, negated families, number comparisons.

Limits, stated before the results:
- A single mask shared by all statements can't make the model believe a *particular* false fact. At best it inverts how the relation is evaluated for all facts, or it suppresses recall. A fact-specific belief edit (ROME-like) would need one mask per fact; that is a possible next step, not this run.
- The direct-question test ("In which country is Krasnodar?") is not in this script yet. I will add it if a twin mask works.
- Predictions [assumed]: `true0` finds little beyond the destruction mode (4c has almost no "true" components). `twin` gets the negated half (L17:5-like) more easily than the affirmative half. `invert` with small λ finds a dense mask that works on train but transfers worse to test.

Smoke test launched (twin and true0, λ = 0.01, 20 steps), log `smoke.log`.

---

26-10-20, Claude Opus 5.5 — *gradient mask search: results*

Run [verified: `~/spd_out/26-10-16_component_steering/mask_search_s-5ecb5791.{log,npz}`, tables `analysis_masks_s-5ecb5791.out` and `cell_means_s-5ecb5791.out`; one run, seed 0]. 12 fits (3 objectives × λ ∈ {0.001, 0.003, 0.01, 0.03}), 400 steps each, 62–95 s per fit on the 4090. Sanity checks: d = 0 reproduces the original (test accuracy 0.978 vs 0.976, KL 0.000); the routing printed for one prompt decomposes the statement tokens through `'.\n'` and leaves the question tokens on the original weights. Sparsified masks (|d| < 0.1 reset to 0) behave like the full ones everywhere (differences ≤ 0.03 in every flip rate), so the numbers below are for the sparsified masks. Every random control (same values on random components) stays near the original: median flip rate ≤ 0.03 in every cell and fit, worst single draw 0.44 (numF, twin λ = 0.03).

Cells, used below: the four fact families with negated twins split into affT / affF (true / false affirmative) and negT / negF (true / false negated), and numT / numF (larger_than, smaller_than). Flip rate = share of the statements the original answered correctly whose verdict changes sign.

**Mean verdict score per cell** (s = log P('1') − log P('0'), all test statements):

| | affT | affF | negT | negF | numT | numF |
|---|---|---|---|---|---|---|
| original | 2.31 | −4.89 | 5.21 | −1.87 | 6.18 | −4.35 |
| true0 λ=0.01 (13 comp.) | −1.88 | −3.89 | −3.09 | −1.60 | −4.54 | −3.94 |
| twin λ=0.01 (23 comp.) | −3.78 | −7.39 | 8.07 | 8.28 | 0.87 | −0.69 |
| invert λ=0.001 (401 comp.) | −2.74 | 0.62 | −4.81 | 5.50 | −4.03 | 4.72 |
| invert λ=0.01 (19 comp.) | −0.61 | −0.88 | −2.69 | 1.90 | −1.78 | 2.28 |

**Within-family gap**, mean(true) − mean(false), the size of the true/false distinction:

| | cities | sp_en_trans | animal_class | element_symb | larger_than | smaller_than | neg_cities |
|---|---|---|---|---|---|---|---|
| original | 7.34 | 7.10 | 6.30 | 7.02 | 12.42 | 8.65 | 7.40 |
| true0 λ=0.01 | 1.84 | 1.64 | 3.66 | 2.32 | 1.01 | −2.20 | −0.77 |
| twin λ=0.01 | 2.85 | 4.98 | 6.13 | 4.86 | 6.86 | −3.74 | 0.68 |
| invert λ=0.001 | −5.23 | 0.07 | 0.83 | 1.05 | −8.78 | −8.72 | −11.23 |
| invert λ=0.01 | −0.16 | 0.35 | 1.25 | 2.18 | −3.37 | −4.75 | −4.37 |

**true0** (true → "0", false statements keep their output): the masks reach the objective on test (affT flip 0.94–1.00 at λ ≤ 0.01, numT 0.99–1.00, affF / numF ≤ 0.02, random ≤ 0.00), but by moving *every* score down to the false level, not by flipping a belief. In the affirmative families true statements still score above false ones (gap 1.5–3.7, per-family AUC 0.82–1.00): the distinction is compressed, not reversed. Reading [concluded]: a "say 0" lever that partly erases the truth distinction. In the negated families and smaller_than the gap reverses (the known negation × relation components). Not a belief change.

**twin** (true affirmative → "0" together with its negated twin → "1"): the trained cells reach it on test (affT 0.96–1.00, negF 1.00 at λ ≤ 0.01). The held-out cells do not move (affF → "1" 0.00, negT → "0" 0.00–0.06 at λ ≤ 0.01): the masks push all affirmative statements to "0" and all negated ones to "1", whatever their truth. The affirmative gap stays positive (2.9–6.1 at λ = 0.01). This rules out "relation evaluation inverted", which would have flipped affF and negT too. Reading [concluded]: the cheapest global solution of the twin objective is a verdict that follows the sentence form (negation present or not), not a belief change. This was the expected limit of a single shared mask (design entry), now observed.

**invert** (every statement → the opposite digit): a real inversion, but only in some families. Dense mask (λ = 0.001, 401 components, |d| up to 6.9): cities gap 7.34 → −5.23 (AUC 0.01), larger_than / smaller_than −8.8 / −8.7, neg_cities −11.2, every negated family AUC ≤ 0.02. In animal_class, element_symb and sp_en_trans the distinction collapses instead (gap 0.1–1.1, AUC 0.57–0.92). The inverted families are the ones that dominate the train split (cities 582 + 582, comparisons 786 per cell, negated cities 582 + 582, out of 6,560), so the mask probably follows the train mix [assumed: not tested by reweighting]. With 19 components (λ = 0.01) the inversion is left only in the comparisons and the negated families; cities collapses (gap −0.16). Digit mass stays 0.95–0.98 for λ ≤ 0.01.
OUTDATED (26-10-21): "every negated family AUC ≤ 0.02" is wrong for negated animal class: its AUC is 0.24 under the dense mask (λ = 0.001) and 0.43 under the 19-component mask (λ = 0.01); the other three negated families are at 0.00–0.08 under both [verified: per-family AUC table in `analysis_masks_s-5ecb5791.out`, re-read 26-10-21 by the sync]. So both masks invert three of the four negated families, not all four.

**Components** [verified: `analysis_masks_s-5ecb5791.out`]: L17:5 gets d between −2.2 and −6.1 in 11 of 12 fits (mask −1 to −5, i.e. reversed and amplified); the exception is invert λ = 0.03, a 2-component mask (L11:70 −4.4) that only weakens the verdict. Other recurring members: L12:54, L13:13, L14:84, L12:7, L15:71, L14:34. Most fits also amplify components (d > 0), which the flip experiment never tried.

**Summary so far** [concluded, one seed, one decomposition]:
1. 4c's components contain levers that move true statements to "0" (true0) and that invert the verdict in cities and the comparisons (invert); random component sets of the same size and values do nothing.
2. None of them looks like a belief change. true0 compresses the distinction, twin swaps it for sentence form, and invert works per family and needs hundreds of components to reach cities.
3. Whether invert's cities inversion touches the model's knowledge of the cities is the open test: recall test launched (`recall_test.py`: completion "The city of X is in" and a direct question, on the 166 test cities, masks applied as a full weight edit; plus continuations of the verdict prompt).

---

26-10-20, Claude Opus 5.5 — *recall test: knowledge survives the inversion*

Run [verified: `~/spd_out/26-10-16_component_steering/recall_s-5ecb5791.{log,out}`, script `recall_test.py`]. 166 test cities with a true statement, greedy 12 tokens. Recall = share whose generation contains the correct country. The mask was applied at every position, a full weight edit and so stronger than in the verdict condition (statement tokens only).

| mask (sparsified) | completion "The city of X is in" | question "Question: In which country is the city of X?\nAnswer:" |
|---|---|---|
| original | 0.614 | 0.771 |
| invert λ=0.001 (401 comp.) | 0.386 | 0.771 |
| invert λ=0.003 (175) | 0.410 | 0.922 |
| invert λ=0.01 (19) | 0.578 | 0.825 |
| twin λ=0.01 (23) | 0.572 | 0.590 |
| true0 λ=0.01 (13) | 0.578 | 0.789 |

- **The question prompt is the clean measure.** The original answers with the country name and stops ("China<|endoftext|>"). The completion prompt is a weak measure: the original names the right country for only 61% of cities and often continues with something else ("in the central part of Anhui Province", "in the shape of a rectangle"). Under the dense invert masks the completions drift to unrelated text ("in the air pollution control area", "in the province-wide examination. The mathematics test score $X$"), not to wrong countries. In the examples read, no mask makes the model name a wrong country.
- **Direct-question recall does not drop under any invert mask** (0.771 / 0.922 / 0.825 vs 0.771 original), although the dense invert mask inverts the cities verdict (AUC 0.01). The rise at λ = 0.003 looks like a format change, not more knowledge [assumed: not checked]: the answers carry explanations more often ("morocco\nStep 1: Identify the city in question"). twin's 0.590 is a format change too: it answers in sentences ("Lu'an is a city located in the Anhui province") that the 12-token window cuts off before the country.
- **Verdict-prompt continuations** (4 true, 4 false cities statements, mask at the statement tokens as in the search; anecdotal). Under invert λ = 0.001 the digit flips in 7 of 8 (Marrakesh/Malaysia: "10%"), and the text after it mostly contradicts the digit:
  - "0 out of 1. The statement "The city of Jiaxing is in China" is actually true, so …" (same for Marrakesh/Morocco)
  - "1.0. The city of Jiaxing is indeed located in China, not in Ukraine. Therefore, …"
  - "1.0. The statement "The city of Lu'an is in Turkey" is entirely true. Lu'an is indeed a city located in China, …"
  - One confabulation that follows the digit: "1.0 (True). The city of Zhabei is actually a district in the city of Kandahar, Afghanistan."
  - Under invert λ = 0.01 (19 components) the cities verdicts mostly stay as in the original. That fits its cities gap of −0.16 (collapsed, not inverted).

**Reading** [concluded, one seed, one decomposition, 8 anecdotal continuations]: the cities inversion is a verdict-only change. The model still knows where the cities are, and says so right after the inverted digit. Together with the mask-search results: the gradient search over 4c's components finds no belief lever. What it finds:
- a "say 0" bias (true0);
- a verdict that follows sentence form (twin);
- a per-family verdict inversion that leaves knowledge intact (invert).

This agrees with the picture so far: 4c's components (MLPs 11–17 on the statement tokens) carry how the verdict is read out and combined with negation, not the stored facts.

What a belief lever would still need [assumed: proposals, not discussed]:
- (a) A **fact-specific** mask: one mask per fact, fit on a few phrasings of that fact and tested on the direct question. This is the ROME-like setting; a single mask shared by all facts can't encode a particular false fact.
- (b) A decomposition whose positions include the **subject's last token in a recall prompt**, where factual recall is localised in the literature (ROME, Meng et al. 2022; not checked for Qwen). 4c does decompose those tokens, but only inside verdict prompts, and was trained only to preserve the verdict.
- (c) Look in Δ, which carries 73% of the direct truth write.

The vast.ai instance (56873443) is idle and still billing ($0.69/h); not destroyed.
