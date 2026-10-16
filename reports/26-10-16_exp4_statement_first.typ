#import "template.typ": *

#show: report.with(
  title: "Exp 4: decomposing where the truth judgement is formed",
  subtitle: "A causal layer scan and three targeted parameter decompositions of Qwen2.5-7B-Instruct's MLPs 11–17 on a statement-first truth prompt",
  date: "2026-10-16",
  author: "Written by Claude Opus 5.5 for Julian Bitterlich",
  summary: [
    - *Prompt.* `<statement>\nThe amount of truth in this statement is `, ending in a space token; the model answers with the digit 1 (true) or 0 (false), correctly for 97.8% of the 1,794 test statements.
    - *Layer scan.* No single MLP layer is needed for the judgement, but replacing the outputs of MLPs 11–17 at the statement's tokens by their average takes the answer to chance (accuracy 0.54) while the model still answers with a digit. The judgement is spread over the statement's tokens: doing this at the statement's last token alone keeps 82% of the true/false separation.
    - *Three decompositions of MLPs 11–17.* *4a*: the training loss looks only at the answer. It reproduces the answer, but through a layer-11 write 13 times the size of the real MLP output, cancelled by the rest of the weights — a re-implementation, not a decomposition. *4b*: adds a term requiring the components to reproduce each MLP's output; at the statement's last token the components now point along the real MLP output (cosine 0.88). *4c*: 4b at every statement token; the components are a (smaller) part of the MLP output everywhere, including the tokens with giant activations.
    - *Which components carry truth (4c).* One component, L17:5, is switched off on true and on for false statements in most families, writes 22% of the truth signal at the statement's last content word, and its removal costs 15% of the answer's true/false separation. A group of seven components fires on false negated statements in all four negated families. Most other components detect the task family, not truth. Most of the direct writing along the truth direction (73%) still sits in the decomposition's unexplained remainder.
    - *Caveats.* One seed per run; selectivity read from importance values; effects measured as direct writes and single removals.
  ],
)

= Background

== Method in brief

*Parameter decomposition* writes each decomposed weight matrix $W$ as a sum of rank-1 *components* plus a full-rank remainder: $W = sum_c U_c V_c^top + Delta$. A small *causal-importance (CI) network* outputs, at every token, a value in $[0, 1]$ per component: how much the component is needed there. A component is *active* where its CI exceeds 0.01. Training asks that the model behave like the original when components are masked by their CI values (randomly, and adversarially in the last 20% of training), and that few components be active. *Targeted* PD (tPD; Vigouroux & Sharkey 2026, arXiv:2607.13047) trains this on a chosen *target* dataset with Δ switched off at the *decomposed positions*, so the components must carry the computation there; on general web text (the Pile) Δ stays on. All other positions run on the original weights.

== Data and prompt

The 20 Truth-is-Universal datasets (*tiu*; Bürger et al. 2024): 10 families of simple statements in true and false versions, e.g. "The city of Krasnodar is in Russia.", "Fifty-one is larger than fifty-eight.", "The Spanish word 'uno' does not mean 'one'." Five families are negated versions of the other five. 6,560 training and 1,794 test statements (subjects disjoint). All numbers are on the test statements.

The prompt is `<statement>\nThe amount of truth in this statement is ` (ending in a space token), tokenized as one string: the statement's final period and the newline merge into one token `'.\n'`, the *end token*. The model's next token is the digit after the final space. *Verdict score*: log P('1') − log P('0'); *separation*: mean score of true minus mean score of false statements (8.65 for the original model); *digit mass*: P('1') + P('0') (0.984). The prompt was chosen from 14 statement-first candidates in a screen on 546 statements (accuracy 0.963 there).

== Why layers 11–17

The previous decompositions (Exp 0–3, report of 2026-10-14) used MLPs 15–19 at the statement's last token. A steering sweep on this prompt found that the judgement can be changed by pushing along a truth direction at the statement's tokens in layers about 13–17, and at the answer tokens from layer 18 on; a direct-write analysis put the start of the truth writing at layers about 11–12. Steering shows where the judgement is *used*, not where it is computed, so the layer choice was made with the causal scan below.

The *truth direction* used throughout: the difference between the mean residual stream of true and of false training statements after layer 17, at a given token (unit length), measured on this prompt. *Truth gap*: the true-minus-false difference of the mean projection on it (after layer 17: 12.84 at the end token, 17.90 at the *object token*, the token holding the statement's last letter or digit, e.g. " Russia").

= Causal layer scan

Each MLP's output is replaced at the statement's tokens by its average over the training statements (*mean ablation*; averaged separately for the first token, the end token and the tokens in between), for single layers and for blocks of layers. Setting the output to zero (tPD's "everything off" state) was run as a reference.

#tbl(
  columns: 6,
  [MLPs replaced by their mean], [accuracy], [separation share], [digit mass], [KL (nats)], [truth gap share after L17: end / object],
  [none (original)], [0.977], [1], [0.984], [0], [1 / 1],
  [any single layer 8–14], [0.957–0.977], [0.86–0.97], [≥ 0.979], [≤ 0.05], [0.67–0.82 / 0.73–0.89],
  [single layer 15], [0.915], [0.89], [0.985], [0.10], [0.66 / 0.78],
  [single layer 17], [0.904], [0.77], [0.980], [0.08], [0.76 / 0.48],
  [block 6–10], [0.715], [0.18], [0.952], [0.44], [0.13 / 0.16],
  [block 11–14], [0.876], [0.38], [0.979], [0.23], [0.22 / 0.22],
  [*block 11–17*], [*0.541*], [*0.05*], [0.977], [0.68], [*0.03 / 0.01*],
  [block 15–19 (the old layers)], [0.662], [0.34], [0.982], [0.49], [0.32 / 0.15],
  [block 11–17, end token only], [0.924], [0.82], [0.984], [0.11], [0.19 / 1.00],
)
#verified[`layer_scan.py`, all 1,794 test statements, one run on an H100]

#finding(title: "The judgement is spread over layers and tokens; MLPs 11–17 together carry it")[
  No single MLP layer is necessary (each keeps ≥ 77% of the separation). Removing MLPs 11–17 at the statement's tokens takes the verdict to chance for affirmative and negated statements alike while the model keeps answering with a digit, so the effect is on the judgement, not on the answer format. The old layers 15–19 cover about two-thirds of it. Removing the block at the end token alone keeps 82% of the separation: the object token carries the judgement too. #concluded
]

Zero ablation gives the same picture for these layers and blocks (block 11–17: accuracy 0.531, separation share 0.08). *Artifact:* mean ablation of layers 3, 4 and 25–27 is misleading: a few tokens per statement (the first token in 40% of statements) have MLP outputs with norms up to about 11,800 there ("massive activations", median about 10), which dominate the averages, so mean ablation writes a huge vector at every token. Layers 12–19 are unaffected.

= Three decompositions of MLPs 11–17

All three decompose the `down_proj` matrices of layers 11–17 (96 components each, 672 in total), with the same settings as Exp 0–3 (5,000 steps, 64 statements + 16 Pile sequences per step, learning rate 5e-4, seed 0). The *target loss* is the KL divergence of the model's answer distribution after the final space. They differ in two choices:

#tbl(
  columns: (auto, auto, 1fr, auto),
  align: (left, left, left, left),
  [run], [WandB id], [decomposed positions; extra loss], [training time],
  [*4a*], [#runid("s-4cd6abe5")], [the end token `'.\n'`; none (the loss of Exp 3, on other layers and prompt)], [48 min],
  [*4b*], [#runid("s-717b50c9")], [the end token; *reconstruction term*], [58 min],
  [*4c*], [#runid("s-5ecb5791")], [every statement token, the first included; reconstruction term], [58 min],
)

*Reconstruction term* (new): at the decomposed positions, each decomposed MLP's output under the training's random masks must match the original output; per position the squared error is divided by the squared norm of the original output (1 = as far off as an all-zero output), so tokens with giant outputs don't dominate. Weight 1. It was added because in Exp 0–3 the components had reproduced the target output without reproducing the MLP outputs (report of 2026-10-14, section "Are the components a part of the MLP outputs?").

== The verdict with parts of each decomposition

States, with the weights replaced only at the decomposed positions: *components only* = active components, Δ off; *all off* = the seven MLP outputs set to zero there; *recovered* = share of the all-off KL that components only removes.

#tbl(
  columns: 8,
  [run], [active components per statement], [accuracy (affirmative / negated)], [separation share], [KL (nats)], [recovered], [all off: accuracy], [all off: separation],
  [4a], [2.2], [0.980 (0.985 / 0.965)], [0.94], [0.011], [94%], [0.855], [0.73],
  [4b], [10.8], [0.968 (0.984 / 0.924)], [0.92], [0.023], [88%], [0.855], [0.73],
  [4c], [91 (≈ 13 positions)], [0.919 (0.920 / 0.914)], [0.48], [0.171], [74%], [0.540], [0.08],
)
#verified[`exp4_analysis.py`; the all-off rows reproduce the scan's zero ablation with independent code]

In 4a and 4b the end token alone carries only part of the judgement (all off keeps 73% of the separation). In 4c, where every statement token is decomposed, the components bring the verdict from chance (0.540) back to 0.919.

== Are the components a part of the MLP outputs?

At each decomposed position the MLP output splits exactly as $y = A + I + D$: $A$ = write of the active components, $I$ = of the inactive ones, $D$ = of Δ. The goal is not that $A$ reproduces all of $y$; the hope is $y = t + r$ with $t$ the part relevant to the truth judgement, and that the components find $t$. What can be checked is whether the rest of the output *undoes* what the components write: $|y - A| \/ |y| > 1$ holds exactly when $y - A$ points against $A$ by more than half of $A$'s length. Error below 1 is compatible with $A$ being a part of $y$.

#tbl(
  columns: 7,
  [run], [error $|y - A| \/ |y|$], [cosine A, y], [$|A| \/ |y|$], [$|D| \/ |y|$], [cosine I, D], [active per layer and position],
  [4a], [*2.80* (layer 11: 12.7)], [0.004], [layer 11: *12.9*; layers 13–17: ≤ 0.08], [3.5–13.7], [−0.36 to −0.97], [0.08–1.34],
  [4b], [*0.44*], [*0.88*], [0.81–0.96], [0.35–0.44], [−0.10 to 0.04], [1.2–2.1],
  [4c], [0.81], [0.48], [0.29–0.66], [0.63–0.75], [−0.07 to −0.01], [0.8–1.7],
  [Exp 0–3, arm A (for comparison)], [1.07–1.54], [≤ 0.12], [], [], [], [],
  [Exp 1 (residual lens)], [0.83], [0.56], [], [], [], [],
)

4c, by position type (pooled over layers; "massive" = output norm above 5 times the layer's median, 2.8% of positions):

#tbl(
  columns: 6,
  [positions], [mean $|y|$], [mean $|A|$], [error], [cosine A, y], [cosine A+I, y],
  [first token], [65.5], [46.7], [0.58], [0.63], [0.80],
  [middle tokens], [28.1], [13.3], [0.83], [0.46], [0.60],
  [end token], [22.5], [10.8], [0.88], [0.44], [0.57],
  [massive], [250.7], [173.9], [0.67], [0.74], [0.86],
  [not massive], [24.8], [11.8], [0.82], [0.47], [0.61],
)
#verified[`exp4_analysis.py`, all decomposed positions of the 1,794 test statements]

#finding(title: "The reconstruction term turns a re-implementation into a decomposition")[
  4a and 4b differ only in this term. Without it (4a), the answer is reproduced best of all three runs, through a layer-11 write 12.9 times the norm of the real MLP output and orthogonal to it; the inactive components and Δ, each 3–14 times the output, cancel it in the full model. With it (4b), about 1.5 components per layer write along the real output (cosine 0.88) and nothing cancels. 4c, covering all statement tokens with the same 96 components per layer, reaches a smaller part of the output (error 0.81), but handles the giant activations of the first token and a few others well (error 0.67, better than ordinary tokens). There, unlike in Exp 0–3, the inactive components add further parts of $y$ (cosine of A+I with y 0.60 vs 0.46 for A alone). #concluded
]

The term costs sparsity (10.8 active components per statement in 4b vs 2.2 in 4a) and buys robustness: adversarially chosen masks change the answer by 0.13 nats in 4b vs 1.20 in 4a, and the general-text (Pile) KL is 0.021 vs 0.083.

= Which components carry the truth part?

For 4b and 4c, per component active on ≥ 1% of statements: its activity by label within each family; its *direct write* along the truth direction at the end and object tokens (as a share of the truth gap; later layers ignored); and the effect of removing it from the full model (all components + Δ = the original weights; meaningful here because nothing cancels): change of the separation, accuracy, truth gaps. "On/off by label" = within some family, the share of true and of false statements on which it is active differs by at least 0.5.

*Direct writes along the truth direction after layer 17* (share of the truth gap; MLPs 11–17 together write 0.76 of it at the end token and 0.98 at the object token):

#tbl(
  columns: 5,
  [], [active components], [inactive components], [Δ], [whole MLP output],
  [4c, end token], [0.13], [−0.10], [0.73], [0.76],
  [4c, object token], [0.25], [0.01], [0.73], [0.98],
  [4b, end token], [0.16], [0.23], [0.38], [0.76],
)

== Exp 4c (all statement tokens)

Of 200 components active on ≥ 1% of statements, 68 are on/off by label in at least one family.

#tbl(
  columns: (auto, 2.4fr, 0.9fr, 1.3fr),
  align: (left, left, left, left),
  [component], [activity: share of true / false statements on which it is active], [direct write, end / object], [removed: separation change, accuracy, object-token gap change],
  [*L17:5*], [cities 0.04 / 0.63, animal classes 0.56 / 1.00, translations 0.44 / 1.00; on in all number comparisons and negated statements, there with stronger importance on false ones], [+5% / *+22%*], [*−1.30* (−15%), *0.880*, *−3.96* (−22%)],
  [L11:70], [every statement], [≈ 0], [−1.22, 0.979, −3.56],
  [L11:29], [nearly every statement], [≈ 0], [−1.09, 0.952, −0.42],
  [L15:52], [number comparisons: larger-than true 0.98 / 0.30, smaller-than 0.59 / 1.00], [≈ 0], [−0.56, 0.929, −1.09],
  [*L15:51*], [false negated statements in all four negated families (e.g. negated cities 0.00 / 0.92, negated translations 0.06 / 1.00); almost never on affirmative statements], [+1% / 0%], [−0.27, 0.968, end-token gap −0.65],
)
#verified[`component_analysis.py`, all 1,794 test statements; separation 8.65, accuracy 0.978 with nothing removed]

- *L17:5 is the only component with both label selectivity and a sizeable effect.* It is switched off on true and active on false statements of the affirmative fact families, and more strongly active on false statements in five further families; it writes 22% of the truth gap at the object token, and removing it costs 15% of the separation and 22% of the object-token gap. A plausible reading is a "doesn't fit / false" writer at the object token #assumed.
- *A "false negated statement" group*: L15:51 and six others with the same profile (L16:28, L14:34, L15:84, L13:60, L16:20, L11:91). For a negated statement, "false" means the relation *does* fit ("X is not in Y" while X is in Y), so they fire on negation combined with a fitting relation, the combination that makes a negated statement false. Each removal alone is small (−0.05 to −0.27 separation), as expected for a redundant group #concluded.
- *Number comparisons get comparison-result components, not truth components*: L15:52 and L14:56 fire on "larger-than true" and "smaller-than false", i.e. whenever the first number is larger, whatever the claim.
- *Most components are task or family detectors*, active on every statement of a family regardless of truth; some matter causally (L11:70, L11:29: −1.2 and −1.1 separation). They are machinery the judgement needs, not the judgement.

== Exp 4b (end token only)

Of 45 components, the large ones are task detectors: five fire on the number comparisons and almost nothing else, one on cities and animal classes, one on two negated families, one on everything; removing any one changes the separation by at most −0.51. There are within-family truth switches, e.g. L16:90 (cities: true 1.00 / false 0.03) and L17:36 (1.00 / 0.26) for true cities, and L15:39 for false cities, animal classes and translations, which writes 9.5% of the end-token truth gap. None has a sizeable single effect on the verdict, consistent with the end token carrying only part of the judgement in 4b (the object token runs on the original weights).

#finding(title: "Part of the truth computation is in components; most of the direct truth writing is not")[
  The decompositions with the reconstruction term isolate components that follow truth across families (4c: L17:5, the "false negated" group) and within families (4b: true/false city components), and L17:5 is causally relevant for the answer. But the active components write only 13–25% of the truth gap directly; Δ writes 73%. In terms of $y = t + r$: part of $t$ is found, most of it is not yet in components. #concluded
]

= Caveats

#caveat[
  - *One seed per run.* Whether L17:5 and the "false negated" group reappear with another seed is the most direct test that they are not artefacts of one run.
  - *Selectivity is read from importance values*, and an importance network reads truth from the activations even untrained (earlier result); the removals are the causal part.
  - *Single removals underestimate redundant groups*; "small effect" does not mean "unused".
  - *Direct writes ignore later layers.* A component can matter through how later layers use its write.
  - *The loss weighs long statements more in 4c*: the answer's KL counts once per decomposed position (6–20 per statement).
  - *Object-word shortcut*: in the number comparisons and animal classes, the object word alone predicts the label in about 70–75% of statements, so some within-family switches may track the object word.
]

= Next

- L17:5 in detail: at which statement tokens it is active, the angle between its write and the truth direction, its removal per family.
- Remove the "false negated" group together.
- A second seed of 4c.
- A weaker form of the reconstruction term that matches the $t + r$ goal: forbid the rest of the output from undoing the components' write, instead of asking the components to reproduce all of $y$.

= Files

#text(size: 9pt)[#set par(justify: false)
Runs: WandB `bitt-j-personal/spd/<id>`; configs `spd/experiments/lm/honesty_targeted_decomposition/config_truth_sf_verdict{,_recon,_recon_alltok}.yaml` (4a, 4b, 4c). Code: commit `8165035` (joint tokenization, `TokenPositions.start`, position-masked relative `StochasticHiddenActsReconLoss`). Outputs: `~/spd_out/26-10-10_bolder_experiments/layer_scan/` (scan), `exp4/full/` (training metrics, `analysis/`: verdict states and component-vs-output test, `components/`: component analysis). Project record: `convos/julian/26-10-10_bolder_experiments_LOG.md`.
]
