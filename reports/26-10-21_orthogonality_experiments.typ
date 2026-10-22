#import "template.typ": *

#show: report.with(
  title: "Exp 4d/4e: an orthogonality term instead of reconstruction",
  subtitle: "Targeted parameter decompositions of Qwen2.5-7B-Instruct's MLPs 11–17 that carry the truth verdict without cancellation, compared with Exp 4a/4b/4c",
  date: "2026-10-21",
  author: "Written by Claude Opus 5.5 for Julian Bitterlich",
  summary: [
    - *Problem.* Of the earlier decompositions of MLPs 11–17, *4a* (loss on the answer only) re-implements the verdict with a huge write that the rest of the weights cancel, and *4b/4c* (plus reconstruction of the MLP outputs) are faithful but reconstruct most of each output, so they are not targeted at the verdict. The orthogonality term is meant to sit in between: the components may write any part of the MLP output, but the rest may not undo it.
    - *4d* (the term on the active components only) removed the re-implementation but made the components trivial: the remainder Δ alone gave the verdict (accuracy 0.960), the components recovered 3% (CI-weighted) of what switching the MLPs off loses.
    - *4e* (the term on all components together, ⟨C, Δx⟩ = 0), tuned: the components alone, weighted by their importance values, recover *91%* of the verdict in two seeds (accuracy 0.97, as the original), while writing at most 0.16 of the MLP output's norm. No run shows cancellation. 4c, the closest earlier run, recovered 74% (rounded masks, other split and learning rate, so not a clean comparison).
    - *What the components carry.* Without their statement-specific activity the model calls most false statements "true": MLPs 11–17 at the statement tokens supply a "this is false" signal against a "true" default, and 4e's components carry it. At layer 11 this split is a property of the model itself: removing the components' write subspace (35–50% of the output's energy) from the *original* output flips 22–28% of false statements to "true"; random subspaces of the same dimension do nothing.
    - *Limits.* The individually important components are mostly sentence-form detectors (negation, comparison), as in 4c, and which component does what differs between seeds. A verdict swap by a fixed weight edit works, but an untrained data-aligned basis does it about as well, and no edit changes a belief. Screening runs only (3,000 steps, no adversarial masks, validation split, one prompt, one seed per setting except one).
  ],
)

= Background

== Setting shared by all Exp 4 runs

*Model and prompt.* Qwen2.5-7B-Instruct reads `<statement>\nThe amount of truth in this statement is ` and answers with the digit 1 (true) or 0 (false). The statements are the 20 Truth-is-Universal datasets (*tiu*; Bürger et al. 2024): 10 families of simple facts, number comparisons and their negations, each in a true and a false version (e.g. "The city of Krasnodar is in Russia.", "Fifty-one is larger than fifty-eight.", "The Spanish word 'uno' does not mean 'one'."). The model answers 97–98% of them correctly.

*Why MLPs 11–17.* A causal layer scan (report of 2026-10-16) found that replacing the outputs of MLPs 11–17 at the statement's tokens takes the verdict to chance while the model keeps answering with a digit. No single layer is necessary on its own.

*Targeted parameter decomposition (tPD).* Each decomposed weight matrix (here the `down_proj`, the MLP's output projection) is written as $W = sum_c u_c v_c^top + Delta$: 96 rank-1 *components* per layer plus a full-rank remainder Δ. A *causal-importance (CI)* network gives each component a value in $[0, 1]$ per token; a component is *active* where its CI exceeds 0.01. Training asks that the model's answer stays the same when the components are masked by their CI values (with random noise), with Δ switched off on the target statements, and that few components be active (the *importance-minimality* term). On general text (the Pile) Δ stays on.

*Notation used throughout.* At a decomposed position, the MLP output $y$ splits exactly as $y = A + I + D$: $A$ = write of the active components, $I$ = of the inactive ones, $D = Delta x$ = of the remainder. $C = A + I$ is the write of all components.

*Evaluation states* (weights replaced only at the decomposed positions): *components only* = components on, Δ off, either *rounded* (every active component at full strength) or *CI-weighted* (each component scaled by its CI); *Δ only* = components off, Δ on; *all off* = the decomposed MLP outputs set to zero. *KL recovered* = 1 − KL(state)/KL(all off), with KL taken against the original model's answer distribution: 1 = original verdict, 0 = no better than switching the MLPs off.

== The earlier decompositions and the problem

#tbl(
  columns: (auto, 1fr),
  align: (left, left),
  [run], [what it is and what it showed],
  [*4a*], [Loss on the answer only, at the statement's last token `'.\n'`. Reproduces the verdict best (94% recovered), but through a layer-11 write 12.9 times the norm of the real MLP output, cancelled by the inactive components and Δ (each 3–14 times the output): a *re-implementation*, not a decomposition.],
  [*4b*], [4a plus a reconstruction term (each MLP output must be reproduced under random masks). Faithful: the components point along the real output (cosine 0.88, $|A| \/ |y|$ 0.81–0.96), 88% recovered. But they reproduce most of each output, so the decomposition is no longer *targeted* at the verdict.],
  [*4c*], [4b at every statement token. Components are a smaller part of the output ($|A| \/ |y|$ 0.29–0.66), 74% recovered. Most components detect the task family; one (L17:5) acts on negated statements. Most of the direct writing along the truth direction (73%) stays in Δ.],
)

Julian's question: is there a loss between 4a (principled but too free) and 4b/4c (faithful but untargeted)? His idea: replace the reconstruction term by an *orthogonality* condition. If the components find a genuine part $t$ of the output $y = t + r$, the rest $r$ need not be reconstructed, but it must not undo $t$. The condition $chevron.l t, r chevron.r = 0$ expresses that.

= Exp 4d: the condition on the active components

*Form* #decided (Julian, chat): one condition for all components together in the CI-weighted state, $chevron.l A_"ci", y - A_"ci" chevron.r = 0$, divided per position by $|y|^2$, so the components may interact. Geometrically, it puts $A_"ci"$ on the sphere with diameter $y$: a write along a direction $u$ can be at most $y$'s own coordinate along $u$, which caps 4a's layer-11 write at 0.03 $|y|$. Everything else as in 4a (statement end only, coefficient 1).

*Result* (#runid("s-5c284ebb"), all 1,794 test statements, one seed) #verified[`exp4d_analysis.py`, one run]:

#tbl(
  columns: 5,
  [state], [accuracy], [KL (nats)], [KL recovered], [note],
  [original], [0.978], [0], [1], [],
  [all off], [0.855], [0.193], [0], [],
  [components only, rounded], [0.884], [0.147], [24%], [2.3 active components per statement],
  [components only, CI-weighted], [—], [0.187], [3%], [],
  [*Δ only*], [*0.960*], [*0.027*], [86%], [],
)

#finding(title: "4d moved the cancellation instead of removing it")[
  The large writes are gone from the active components (they write 0.04–0.17 $|y|$, almost orthogonal to $y$), but the inactive components and Δ still write 3–11 $|y|$ each and cancel each other. Δ alone carries the verdict. The condition only constrains $A$, and which components count as active is itself learned, so training moved the cancellation into $I$ versus $Δ$. The training's own verdict loss was the same as 4a's: with Δ randomly masked during training, nothing in tPD's loss forces the verdict into the components. #concluded
]

= Exp 4e: the condition on all components

*Form* #decided (Julian, LOG comment): $chevron.l C, Delta x chevron.r = 0$, with $C$ = the unweighted write of *all* components (no dependence on the CIs) and $Delta x = y - C$. It bounds $|C| <= |y|$ and $|Delta x| <= |y|$, so neither part can cancel the other on a large scale, whatever the CIs say. It does not force the verdict into the components ($C = 0$, $Delta = W$ satisfies it). Estimated from per-layer norms, 4a and 4d violate it heavily ($chevron.l C, Delta x chevron.r \/ |y|^2$ between −183 and −9), 4b satisfies it (≈ 0).

*Other changes* #decided (Julian): the goal became "any result", so hyperparameters were tuned freely and comparability with 4a–4c became secondary. Objective: KL recovered by the *CI-weighted* components-only state. Tuning on a new validation split (tiu v2: the old training split divided by subject into 5,574 train / 986 validation statements; the 1,794 test statements are untouched). Screening runs: 3,000 steps without the adversarial-mask (PGD) term; the planned final setting is 5,000 steps with it. Configuration otherwise = 4c's, with the reconstruction term replaced by the orthogonality term (coefficient 1, never tuned).

== Tuning

All runs on the 986 validation statements, one seed each unless stated #verified[`verdict_lens_analysis.py`, one run per row]. Original accuracy there 0.966; all off: 0.838 (end token), 0.511 (all statement tokens).

*Round 1, learning rate × positions* (importance-minimality coefficient 1e-3):

#tbl(
  columns: 6,
  [positions], [learning rate], [KL recovered: CI-weighted / rounded / Δ only], [accuracy, CI-weighted], [active per statement], [run],
  [end token], [1e-4], [0.25 / 0.35 / 0.88], [0.863], [1.9], [#runid("s-c706fefb")],
  [end token], [5e-4], [0.08 / 0.08 / 0.79], [0.844], [0.7], [#runid("s-e6688f03")],
  [end token], [2e-3], [0.86 / 0.72 / 0.84], [0.947], [2.1], [#runid("s-9dc19f9f")],
  [all tokens], [1e-4], [0.29 / 0.48 / 0.42], [0.674], [60.5], [#runid("s-114f2c0e")],
  [all tokens], [5e-4], [0.37 / 0.38 / 0.12], [0.756], [19.7], [#runid("s-348832ad")],
  [*all tokens*], [*2e-3*], [*0.91* / 0.87 / −0.34], [*0.974*], [27.5], [#runid("s-eb1c20d5")],
  [all tokens, seed 1], [2e-3], [*0.91* / 0.92 / 0.13], [0.971], [24.0], [#runid("s-52b2abc7")],
  [all tokens], [5e-3], [0.86 / 0.83 / −0.53], [0.948], [90.1], [#runid("s-9d172777")],
)

At the end token the verdict is carried twice (components 0.86, Δ alone 0.84), so that setting cannot separate them. At all statement tokens with learning rate 2e-3 the components carry it and Δ alone does not (its true/false separation is at the all-off level in both seeds). The learning rate mattered most: 4c's 5e-4 gave only 0.37. The negative Δ-only values mean Δ alone is worse than all off; that differs between seeds and is a calibration effect (Δ alone answers "1" for 70–85% of statements), not a truth effect.

*Round 3, importance-minimality coefficient* (learning rate 2e-3, all tokens; this coefficient sets how many components are active):

#tbl(
  columns: 5,
  [coefficient], [KL recovered / accuracy (CI-weighted)], [active per statement], [components active in ≥ 2% of statements], [run],
  [3e-4], [0.903 / 0.946], [105], [483], [#runid("s-a55b67a4")],
  [*1e-3*], [*0.910 / 0.974*, 0.912 / 0.971], [27.5, 24.0], [79, 211], [seeds 0, 1],
  [3e-3], [0.701 / 0.863], [7.4], [41], [#runid("s-8ec00fb3")],
  [1e-2], [0.599 / 0.740], [2.1], [8], [#runid("s-e9552de2")],
)

By Julian's objective 1e-3 is best; recovery plateaus at 0.91. Julian #decided (LOG comment): fewer components are not better per se, and there is no "right number" of components; the sparser runs are kept because they do different things (section 6).

== Comparison with 4a–4d

#tbl(
  columns: 7,
  [run], [positions], [components only: KL recovered (rounded) / accuracy], [Δ only: accuracy], [largest write of the components], [largest write of Δ], [cancellation?],
  [4a], [end], [94% / 0.980], [—], [12.9 $|y|$ (L11)], [3.5–13.7 $|y|$], [*yes*],
  [4b], [end], [88% / 0.968], [—], [0.96 $|y|$], [0.44 $|y|$], [no],
  [4c], [all], [74% / 0.919], [—], [0.66 $|y|$], [0.75 $|y|$], [no],
  [4d], [end], [24% / 0.884], [*0.960*], [inactive: 3–11 $|y|$], [3–11 $|y|$], [*yes* (inactive vs Δ)],
  [4e, end token], [end], [72% / —], [0.961], [$|C|$ ≤ 0.66 $|y|$], [≤ 0.88 $|y|$], [no],
  [*4e, best*], [all], [*87% / 0.966*; CI-weighted 91% / 0.974], [0.556], [$|A|$ ≤ 0.16, $|C|$ ≤ 0.45 $|y|$], [≤ 0.97 $|y|$], [no],
)
#verified[4a–4d: `exp4_analysis.py` / `exp4d_analysis.py` on the 1,794 test statements; 4e: `verdict_lens_analysis.py` (the same analysis, moved into the repository and checked to reproduce 4a's numbers exactly) on the 986 validation statements]

#caveat(title: "The 4c-versus-4e comparison is not controlled")[
  4a–4d were evaluated on the test split, 4e on the validation split. 4e's best run also differs from 4c in learning rate (2e-3 vs 5e-4), steps (3,000 vs 5,000), the adversarial-mask term (off vs on) and its training split (5,574 vs 6,560 statements), not only in the loss term. 4e at 4c's learning rate recovered just 37%, and 4c was never run at 2e-3. So "4e beats 4c" is a statement about tuned 4e versus untuned 4c. 4a–4c's "components only" is the rounded state; 4e's tuning objective is the CI-weighted one, and both are given for 4e.
]

#finding(title: "What 4e adds over 4a–4d")[
  4e is the first Exp 4 decomposition in which (1) the components alone carry the verdict, (2) without cancellation, and (3) Δ alone does not carry it. 4a has (1) only through cancellation, 4d has neither (1) nor (2), and 4b/4c have (1) and (2) but reconstruct a large part of each output. 4e's components are a *small* part of the output: the active ones write at most 0.16 $|y|$ (4b: up to 0.96 $|y|$), i.e. 4e is targeted in the sense Julian asked for. Whether that small part is faithful enough is Julian's judgement, still pending. #concluded
]

= What 4e's components carry

All on the best setting (learning rate 2e-3, coefficient 1e-3), both seeds, validation split #verified[`ablate_4e.py`, one run per seed].

*It is statement-specific activity, not a generic enabler.* Replacing each component's write by its average (per position type) restores only 5–7% of the verdict (KL recovered 0.070 seed 0, 0.046 seed 1), against 91% with the real activity.

*The components supply "false" against a "true" default.* Every state without the components' real activity calls most false statements "true":

#tbl(
  columns: 4,
  [state (seed 0)], [accuracy], [true → "0"], [false → "1"],
  [CI-weighted components only], [0.974], [0.019], [*0.002*],
  [all off], [0.506], [0.190], [0.742],
  [Δ only], [0.557], [0.067], [0.789],
  [components frozen at their averages], [0.576], [0.231], [0.562],
)

(Flip rates = share of the originally correct true / false statements whose verdict flips.)

*Layer by layer* (one layer in a state, all others original): the components of any single layer suffice for it (KL recovered 0.94–0.98). At layers 11–14, Δ alone is *worse* than switching the layer off (layer 11: 0.38 vs 0.81 in seed 0, 0.52 vs 0.81 in seed 1); at 15–17 both parts help. So at 11–14 the remainder of the output pushes towards "true" and the components outweigh it. Same pattern in both seeds.

*At layer 11 the split exists in the original model* (test A8). The original MLP output, with no decomposition involved, loses its projection onto the span of the layer's 96 component write vectors:

#tbl(
  columns: 4,
  [removed from the original layer-11 output], [share of output energy (range over layers 11–17)], [KL recovered, seed 0 / seed 1], [false → "1", seed 0 / seed 1],
  [the whole output (layer off)], [100%], [0.81 / 0.81], [0.17],
  [span of the 96 component writes], [35–50%], [*0.77 / 0.49*], [*0.22 / 0.28*],
  [top 96 principal directions], [86–99%], [0.84 / 0.84], [],
  [random 96-dimensional subspaces (3 draws)], [2.3–2.9%], [≥ 0.996], [],
)

#finding(title: "A finding about the model, located by the decomposition")[
  At layer 11 the decomposition found a real division of the MLP output: a "false" signal in a specific subspace and a remainder that alone pushes towards "true". Removing that subspace hurts more than removing the whole output or its 96 main directions; random subspaces of the same dimension do nothing. At layers 12 and 14, Δ alone hurts more than removing the subspace, so there part of the effect is decomposition-specific (a small opposition between components and Δ, which the global condition allows because it constrains only the sum $C$); layers 13 and 17 differ by seed. #concluded
]

*Which components matter.* In the components-only state the verdict is spread redundantly (largest single removal: separation 0.69 → 0.58). In the full model a handful matter individually, each mostly by moving false statements to "true": in seed 0, L11:34 (comparison and animal families), L13:62 (active on every statement), L11:36, L12:59, L17:5 (every negated statement) and L11:87 (only false larger_than statements). Seed 1 has the same kinds (comparison and negation detectors plus a few false-selective ones, same families most affected) but different components: no always-on component, 211 instead of 79 candidates, and L17:5 never active.

#note(title: "Compared with 4c")[
  By component identity 4e looks like 4c: sentence-form detectors plus a few truth-selective components, and most of the output's truth difference (measured along each layer's true-minus-false mean direction) stays in Δ (components 0.1–6.6% at the statement end, up to 33% at layer 11's last content token). 4e's L17:5 resembles 4c's L17:5 (the negation component), but the shared index comes from the shared random seed: the seed-1 run has no such component. The robust result is functional (a "false" signal on top of a "true" default), not a stable set of components.
]

= The sparsity front: which decomposition does what

Raising the importance-minimality coefficient keeps components only where switching the MLPs off hurts most. Verdict accuracy per family in the CI-weighted components-only state (validation):

#tbl(
  columns: 7,
  [family], [original], [all off], [1e-3 seed 0], [1e-3 seed 1], [3e-3], [1e-2],
  [animal_class], [1.00], [0.80], [1.00], [0.87], [0.87], [0.80],
  [cities], [0.97], [0.71], [0.99], [0.97], [0.97], [0.71],
  [element_symb], [0.79], [0.64], [0.61], [0.68], [0.64], [0.64],
  [larger_than], [1.00], [0.50], [1.00], [1.00], [0.99], [1.00],
  [neg_animal_class], [1.00], [0.53], [1.00], [1.00], [1.00], [0.53],
  [neg_cities], [0.96], [0.53], [0.98], [0.98], [0.97], [0.53],
  [neg_element_symb], [0.86], [0.46], [0.79], [0.93], [0.79], [0.46],
  [neg_sp_en_trans], [0.90], [0.35], [1.00], [0.90], [0.97], [0.42],
  [smaller_than], [0.98], [0.31], [0.99], [0.98], [0.60], [0.71],
  [sp_en_trans], [1.00], [0.82], [0.95], [1.00], [0.93], [0.82],
)
#verified[`compare_decomps.py`, one run per decomposition]

- *3e-4 and 1e-3*: cover every family; sentence-form detectors plus label-selective components (3e-4 spreads the same function over 483 components, none individually important).
- *3e-3: the most truth-specific decomposition of Exp 4.* All 40 of its label-selective components fire *only on false statements*, in clusters (false smaller_than, false larger_than, false negated statements). Removing all 41 active components from the full model calls 75% of the false statements "true" and leaves the true ones true (1.7% flip). But it has no false-selective components for the affirmative fact families, where MLPs 11–17 matter least.
- *1e-2*: 8 components. L11:22 fires on every larger_than statement, true or false; removing it alone *inverts* the larger_than verdicts (family accuracy −0.88), so it encodes something the comparison needs, not truth. Five components fire only on true statements (true smaller_than, true negated) but barely matter when removed: a CI that tracks truth is not a component that carries it.

= Reprogramming: can a fixed edit swap the verdict?

Method (taken from the component-steering topic, where it was run on 4c): learn one number $d_c$ per component, the same for every statement, so that the weights become $W + sum_c d_c u_c v_c^top$ at the statement tokens; fit on the training split to make the answer the opposite digit, with an L1 penalty for sparsity and a term that keeps the answer a digit; test on the validation split. Controls: the same search over a *random basis* (random directions of the same norms) and over a *PCA basis* (untrained but data-aligned: $v_c$ = the 96 main directions of the MLP's input, $u_c = W v_c$).

#tbl(
  columns: 5,
  [basis (number of terms → validation accuracy; original 0.967, 0 = all swapped)], [L1 0.001], [L1 0.003], [L1 0.01], [L1 0.03],
  [4e, 1e-3 seed 0], [454 → 0.050], [257 → 0.059], [63 → 0.116], [19 → 0.280],
  [4e, 1e-3 seed 1], [411 → 0.037], [269 → 0.041], [103 → 0.134], [9 → 0.438],
  [4e, 3e-3], [424 → 0.033], [220 → 0.047], [95 → 0.090], [20 → 0.340],
  [4e, 1e-2], [407 → 0.026], [207 → 0.054], [84 → 0.115], [13 → 0.461],
  [PCA basis (untrained)], [234 → 0.045], [129 → 0.057], [47 → 0.158], [11 → 0.575],
  [random basis], [322 → 0.094], [150 → 0.124], [32 → 0.542], [3 → 0.693],
)
#verified[`mask_search_4e.py`, one fit per cell]

#finding(title: "Swapping does not tell decompositions apart, and is not a belief change")[
  Every 4e decomposition supports a held-out verdict swap with the answer still a digit. On 4c the same search (test split, 401 components) inverted cities, the number comparisons and three of four negated families, but only collapsed animal_class, element_symb and sp_en_trans to no verdict. 4e's edits also swap most of animal_class and sp_en_trans (e.g. 78% of sp_en_trans with 257 terms); element_symb resists every edit (at most 39% answered against the truth; the original already errs on 21%). But the untrained PCA basis swaps about as efficiently (129 terms → 0.057); the decompositions lead only with ≤ 20 terms and in the affirmative fact families. A "belief-shaped" objective (a true statement → "0" while its negated twin → "1") yields verdicts by sentence form for every basis, random included, as it did for 4c. So no belief lever, and the PCA control, never run for 4c, also qualifies 4c's swaps. #concluded
]

= Caveats and open questions

#caveat(title: "How far the numbers go")[
  One seed per setting except the best one (two seeds). All 4e numbers are on the validation split of one prompt and the tiu data; the test split is untouched. Screening runs only: 3,000 steps without the adversarial-mask term; the planned 5,000-step final runs with it were not done. The orthogonality coefficient, the number of components per layer, the loss routing and the CI-network width were never tuned. The comparison with 4a–4c mixes splits and training settings (section 4.2).
]

Open, as of the pause on 2026-10-21:
- *Julian's decisions pending*: whether 4e's decomposition is faithful enough (deferred until the numbers were in), and the seed policy for final runs.
- *Proposed, not decided* #assumed:
  - final 5,000-step runs with the adversarial-mask term, evaluated on the test split;
  - whether the verdict swap inverts the readout or the model's reading of the statement: apply the same edit under another verdict prompt and in a recall prompt;
  - mechanism study of the 3e-3 false-selective clusters and of L11:22 (e.g. swap the relation word, or steer smaller_than statements with L11:22);
  - test A8 with the span of only the active components;
  - a controlled 4c-versus-4e comparison (4c at learning rate 2e-3 on the same split), if the comparison matters.

*Where things are.* Records: `convos/julian/26-10-16_faithful_targeting_LOG.md` and its SUMMARY (4d/4e), `convos/julian/26-10-10_bolder_experiments_SUMMARY.md` (4a–4c), `convos/julian/26-10-16_component_steering_SUMMARY.md` (4c steering and mask search). Checkpoints of the six analysed 4e runs and the one-off scripts: `~/spd_out/26-10-16_faithful_targeting/e4/`. Run register: `convos/julian/26-10-03_wandb_storage_RUNS.md`.
