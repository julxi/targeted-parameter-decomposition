#import "template.typ": *

#show: report.with(
  title: "What the target loss selects: four tPD runs on true/false statements",
  subtitle: "Targeted parameter decomposition of Qwen2.5-7B-Instruct at the end of simple statements, with four different target losses",
  date: "2026-10-14, updated 2026-10-16",
  author: "Written by Claude Opus 5.5 for Julian Bitterlich",
  summary: [
    - *Setup.* Four decompositions of the same five MLP layers (15–19) of Qwen2.5-7B-Instruct, all at the final period of simple true/false statements, differing only in what the training loss compares: the next token (Exp 0), the residual stream after layer 23 (Exp 1), the model's answer to an appended " Is this true or false? Answer:" (Exp 2), and that answer with the question's tokens decomposed too (Exp 3). One seed each.
    - *The plain next-token loss misses what the verdict needs.* Switching the five layers off at the period costs the verdict 0.116 nats. Exp 0’s components alone recover 8% of that; Exp 1 63%, Exp 2 72%. On negated translations ("The Spanish word 'uno' does not mean 'one'."), where switching the layers off reverses nearly every verdict (accuracy 0.86 → 0.06), Exp 0’s components reach 0.19, Exp 1’s 0.64, Exp 2’s 0.95.
    - *The residual loss puts the truth direction into components.* Only in Exp 1 do components, rather than the unexplained remainder, write most of the true/false separation at layer 19 (71%; other runs ≤ 22%).
    - *Exp 3 found the output end of the verdict.* Its components alone reproduce the verdict over the period and question (KL 0.036 nats; all off 4.19). Among them are components at the answer token that write "True", active almost only on statements the model judges true (component 447: on 97% of true, 1% of false statements). They follow the model's own judgement; they do not compute it. Caveat: the components alone do not reconstruct the MLP outputs; they produce the answer their own way (large writes at the answer token), so these are directions that set the answer, not shown to be what the model's MLPs write.
    - *Where the components write, the rest of the weights undoes it* (added 2026-10-16). The components need not reproduce the whole MLP output, only its target-relevant part; but except partly in Exp 1, the rest of the output points against what they write. Exp 0 and arm A write vectors of the output's size nearly orthogonal to it (cosine 0.08–0.12); at Exp 3's answer token the components write 5.3 times the output's norm at cosine 0.06. Exp 1 is compatible with its components being a part of the output (cosine 0.56).
    - *No truth hidden from the output.* Within each negated family the original model already ranks true above false almost perfectly (e.g. AUC 1.00 at accuracy 0.73). Its errors are a bias towards "False". Exp 3’s components alone answer more accurately than the model (0.977 vs 0.953) only by shifting that bias, through two of the "True" components.
    - *Open:* where the judgement that gates these components is computed, and whether the negated-statement effects are truth or negation. Comparing components across runs found nothing beyond their shared initialisation.
  ],
)

= What was run

== Method in brief

*Parameter decomposition* writes each decomposed weight matrix $W$ as a sum of rank-1 *components* plus a full-rank remainder: $W = sum_c U_c V_c^top + Delta$. A small *causal-importance (CI) network* outputs, at every token position, a value in $[0, 1]$ per component: how much the component is needed there. Training asks for two things: with components masked by their CI values (randomly, and adversarially in the last 20% of training), the model must behave like the original; and few components may be active at once. *Targeted* PD (tPD; Vigouroux & Sharkey 2026, arXiv:2607.13047) trains this on a chosen *target* dataset with Δ switched off, so the components must carry the target computation. On general text (the *non-target* stream, here web text from the Pile dataset) Δ stays on, so the decomposition need not explain everything else the model does.

All runs decompose the MLP output matrices (`down_proj`) of layers 15–19 of Qwen2.5-7B-Instruct (28 layers), 96 components each, 480 in total. Components replace the weights only at selected token positions (the *decomposed positions*); every other position runs on the original weights.

== Data

The 20 Truth-is-Universal datasets (*tiu*; Bürger et al. 2024): 10 families of simple statements, each in a true and a false version, e.g. "The city of Krasnodar is in Russia.", "Fifty-one is larger than fifty-eight.", "Zinc does not have the symbol Fr." Five families are negated versions of the other five. 6,560 training statements; 1,794 test statements whose subjects never occur in training. All numbers below are on the test statements.

== Why four losses

Earlier decompositions on these statements, trained to preserve the model's next-token predictions at every token, carried little truth-specific structure. One explanation: a truth computation that changes no next-token prediction costs nothing to drop. The four runs keep everything fixed (data, layers, the decomposed position = the statement's final period, training settings) and vary only the *target lens*, i.e. what the target loss compares between the original and the masked model:

#tbl(
  columns: (auto, auto, 1fr),
  align: (left, left, left),
  [run], [WandB id], [target loss],
  [Exp 0, next-token lens], [#runid("s-aeaa70d8")], [KL divergence of the next-token distribution at the period],
  [Exp 1, residual lens], [#runid("s-cd129592")], [squared error of the residual stream after layer 23 at the period, each of the 3,584 channels divided by its standard deviation over the training statements],
  [Exp 2, stimulus lens], [#runid("s-05db01ad")], [" Is this true or false? Answer:" is appended to every statement (on the original weights); KL of the next-token distribution at the final ":", i.e. of the model's verdict],
  [Exp 3, stimulus lens, question decomposed], [#runid("s-161f8a7b")], [as Exp 2, but the 8 question tokens are decomposed too (9 positions per statement), so the verdict cannot be recomputed on the original weights],
)

Common settings: 5,000 steps, 64 statements + 16 Pile sequences per step, learning rate 5e-4, adversarial masking in the last 1,000 steps, seed 0. The question " Is this true or false? Answer:" was chosen in a screen of five phrasings: it put the most probability on true/false-type answers (0.74) and separated true from false statements best. Without "Answer:" the model mostly starts reasoning instead of answering.

Two reference decompositions of the same layers and size are used in the analyses: *arm A* (#runid("s-bd23f0d1")), trained with the next-token KL at every statement token, and *untrained* (#runid("s-7fad0c14")), the initial state of such a decomposition.

= Training

At step 5,000, on 256 held-out statements. "Target, rounded": the target loss with components whose CI exceeds 0.01 fully on, all others off, Δ off; in nats for KL lenses, normalised squared error for Exp 1. "Δ only": all components off, Δ on. "Adversarial": the worst masks a 20-step search finds within the CI-allowed range.

#tbl(
  columns: 7,
  [run], [active components per decomposed position], [target, rounded], [target, Δ only], [adversarial], [Pile KL, rounded], [time],
  [Exp 0], [7.7], [0.107], [6.63], [0.63], [0.033], [24 min],
  [Exp 1], [10.3], [0.188 (all components and Δ off: 2.26)], [5.01], [0.21], [0.016], [55 min],
  [Exp 2], [0.47], [0.029], [0.014], [0.60], [0.040], [47 min],
  [Exp 3], [3.4 (≈ 31 per statement)], [0.028], [15.1], [1.62], [0.054], [29.5 min],
)
#verified[run metrics; times on different H100 rentals, not comparable]

- In Exp 0, 1 and 3, Δ alone cannot replace the components: they carry the target computation. In Exp 2, Δ alone reproduces the verdict *better* than the components (0.014 vs 0.029): with the question running on the original weights, the verdict loss puts little pressure on the period.
- Exp 3 reconstructs the verdict as well as Exp 2 while the components now carry it (Δ alone: 15 nats). Its adversarial and Pile losses are the highest of the four.

= The verdict under parts of each decomposition

*Verdict score*: log P(true-like tokens) − log P(false-like tokens) at the final ":" after " Is this true or false? Answer:" (true-like: single-token spellings of true/correct/yes/right; false-like: false/incorrect/wrong/no). The verdict is "true" when the score is positive. *Separation*: mean score of true minus mean score of false statements. *AUC*: area under the ROC curve of the score for true vs false (1 = perfect ranking, 0.5 = chance).

== Replacing the weights at the period only

States: *components only* = rounded CIs, Δ off; *all off* = the five MLP outputs set to zero at the period. "Recovered" = the share of the all-off KL that the components-only state removes. Original model: accuracy 0.953, separation 12.04. All off: accuracy 0.918, separation 10.21, KL 0.116 nats.

#tbl(
  columns: 6,
  [components only of], [active per statement], [KL to original (nats)], [recovered], [separation], [accuracy],
  [Exp 0, next token], [7.8], [0.106], [*8%*], [9.98], [0.907],
  [Exp 1, residual], [10.5], [0.043], [63%], [11.32], [0.936],
  [Exp 2, stimulus], [0.8], [0.033], [72%], [10.88], [0.958],
  [Exp 3, stimulus + question], [2.0], [0.043], [63%], [11.29], [0.953],
  [arm A], [9.3], [0.092], [21%], [9.84], [0.927],
  [untrained], [220], [0.115], [1%], [10.31], [0.918],
)
The untrained decomposition’s 220 "active" components per statement come from its untrained CI network, which marks almost everything as needed; its component weights do nothing useful yet.
#verified[`verdict_accuracy.py`, all 1,794 test statements]

The whole effect sits in negated statements. Accuracy of the components-only state per family (affirmative families: 0.96–1.00 in every state, omitted):

#tbl(
  columns: 9,
  [family (test statements)], [original], [all off], [Exp 0], [Exp 1], [Exp 2], [Exp 3], [arm A], [untrained],
  [neg. translations (64)], [0.86], [0.06], [0.19], [0.64], [*0.95*], [0.91], [0.50], [0.17],
  [neg. element symbols (56)], [0.55], [0.36], [0.43], [0.39], [0.59], [0.52], [0.50], [0.23],
  [neg. animal classes (37)], [0.73], [0.68], [0.62], [0.62], [0.76], [0.68], [0.62], [0.70],
  [neg. cities (332)], [0.93], [0.94], [0.88], [0.90], [0.94], [0.94], [0.91], [0.94],
)

#finding(title: "The loss decides what the components carry")[
  Same layers, same position, same data: the next-token lens (Exp 0) yields components that carry almost none of what the verdict needs from the period, even with 7.8 active per statement. The verdict lens (Exp 2) gets most of it with under one. The residual lens (Exp 1), which never sees a verdict, gets nearly as much. Matching the period's internal state carries the verdict-relevant computation; matching its next-token output does not.
]

Exp 3 was trained with the question decomposed; at the period alone it recovers as much as Exp 1. Exp 0 does not settle whether the negated-statement effect is about truth or about negation: it recovers nothing in either kind of family.

== Exp 3: replacing the weights at the period and the question

Switching the five layers off at all 9 positions destroys most of the verdict (separation 3.49, accuracy 0.770, KL 4.19 nats). Here Exp 2 and the untrained decomposition, which were never trained at the question tokens, serve as references.

#tbl(
  columns: 6,
  [components only of], [active per position], [KL to original (nats)], [separation], [accuracy], [AUC],
  [*Exp 3*], [3.6], [*0.036*], [9.55], [*0.977*], [0.999],
  [Exp 2], [7.9], [4.20], [3.25], [0.824], [0.955],
  [untrained], [227], [3.38], [4.71], [0.890], [0.980],
  [original model], [–], [0], [12.04], [0.953], [0.999],
)
#verified[`verdict_accuracy.py --decompose_stimulus`]

Exp 3’s components alone answer *more* accurately than the original model, mainly on negated statements (negated animal classes 0.86 vs 0.73, negated cities 0.98 vs 0.93, negated translations 0.95 vs 0.86). Ten of its components are active on ≥ 99% of statements. Removing any one of them from the full model (all components + Δ) moves the answer distribution at ":" by 3.0–11.0 nats KL, comparable to switching all five layers off (4.19). How much of the verdict survives varies: from component 28 (accuracy 0.896, AUC 0.993: the ranking survives, the calibration shifts) to component 402 (accuracy 0.446, AUC 0.511: no verdict left). Most of these removals take the model out of the answer format rather than to the other answer: after "Answer:" it continues the text (" for", " the", " &", " (") with 74–99% of its probability outside the true/false tokens (original model: 41%, mostly " To", the start of reasoning). Removing all components, leaving Δ alone, yields garbage tokens (KL 15.3), far worse than zeroing the five MLP outputs (4.18, still fluent text). The reason is cancellation: at these positions the components' part and Δ's part of the MLP output are each 2.0–2.9 times as large as the output they sum to, so Δ alone writes a large vector the original model never produces there #verified[norms over all 9 positions]. A single component's write is smaller (0.2–0.8 times its layer's MLP output), so what breaks the format is its direction, not its size. Removing a component therefore means "original weights plus the negative of its write", a state the original model never sees; this caveat applies to every single-component ablation in this report. Flipping a component's sign instead of removing it breaks the format further (KL 6.6–15.4), and flipping "True" writer 447 does not make it write "False": true statements lose "True" and move to a bare space token. Their CIs do not depend on the statement, so they are not gated by truth; whether what they write carries the verdict was not tested #assumed.

= Exp 3’s answer-token components

Per statement, the CIs of every Exp 3 component at each of the 9 decomposed positions, and the verdict with single components ablated from the full model (all components + Δ = the original weights).

*"True" writers.* Of 221 components active on ≥ 1% of statements, four are clearly selective for the label, all for *true*, all at the final ":":

#tbl(
  columns: 5,
  [component], [active on true statements], [active on false statements], [label AUC of its CI], [ablated from the full model],
  [447 (layer 19)], [97.0%], [1.2%], [0.983], [true statements' score −8.76, false −1.43; every answer "false"],
  [319 (layer 18)], [93.1%], [0.8%], [0.964], [–],
  [371 (layer 18)], [82.4%], [11.6%], [0.889], [accuracy 0.840, share answered "true" 0.454 → 0.341],
  [396 (layer 19)], [72.4%], [21.4%], [0.806], [accuracy 0.878, share "true" 0.380],
)
#verified[per-statement CIs and ablations; reproduces the aggregate numbers of the verdict analysis exactly]

The most false-selective components are weaker (e.g. active on about 25–28% of false and 0.3% of true statements). The answer looks asymmetric: dedicated "True" components, with "False" as the default #assumed.

*They follow the model's judgement, not the truth.* Component 447’s CI predicts the model's own answer (AUC 0.997) better than the label (0.983). On the 83 true statements the model calls false, its CI is weak (mean 0.27, vs 0.95 on true statements answered "true").

*Another kind: task components.* Component 380 (layer 18) is active on all number comparisons ("Fifty-one is larger than fifty-eight."), true and false alike, and rarely elsewhere. Ablating it raises false comparisons' scores by about 10, i.e. it is needed to answer "false" to them.

== Why the components beat the original model

Accuracy, AUC and share answered "true", over all statements and over the negated ones:

#tbl(
  columns: 6,
  [state], [accuracy], [AUC], [share "true"], [negated: accuracy], [negated: AUC],
  [original model], [0.953], [0.999], [0.454], [0.861], [0.998],
  [Exp 3 components only], [0.977], [0.999], [0.478], [0.920], [0.998],
  [… without 371], [0.967], [0.998], [0.468], [0.885], [0.997],
  [… without 396], [0.968], [0.998], [0.469], [0.892], [0.998],
  [… without 371 and 396], [0.941], [0.996], [0.442], [0.796], [0.994],
)

#finding(title: "A calibration shift, not hidden truth")[
  The original model already ranks true above false almost perfectly within each negated family (AUC 1.00 on negated animal classes, where its accuracy is 0.73; the exception is negated element symbols, AUC 0.91). Its errors on negated statements are a bias towards "False", not missing information. Exp 3’s components alone answer "true" more often, and two "True" writers (371, 396) carry that shift: without them, accuracy falls below the original while the AUC barely moves. #concluded
]

*Caveat: the components-only state is a re-implementation.* At the 9 positions, what Exp 3's active components write is largely undone by the rest of the MLP output (next section). At 6 of the 8 question tokens none is active, so the MLP outputs are zero there. At ":", about 28 active components (5.6 per layer) write a vector five times the norm of the real MLP output (169 vs 32) and nearly orthogonal to it (cosine 0.06); in the full model, inactive components and Δ carry its negative. The target loss only asked for the right answer at ":", so the decomposition writes the answer directly. The "True" writers are therefore shown to control the answer and to be gated by the model's judgement. They are not shown to be what the model's own MLPs write #concluded. A check: project the original MLP output at ":" onto component 447's write direction.

Caveat on the method: CI values come from a trained network that reads the activations; even an untrained CI network reads truth from them (earlier result: a probe on untrained CIs reaches 0.998). CI selectivity alone is therefore weak evidence. The ablations above are the causal part.

= Are the components a part of the MLP outputs?

_Added 2026-10-16; measured 2026-10-15._ At every decomposed position, each decomposed MLP's output $y$ splits exactly into three parts, because components and Δ sum to the original weights: $y = A + I + D$, with $A$ the write of the *active* components (CI above 0.01, the "components only" state used throughout this report), $I$ that of the inactive components, and $D$ that of Δ.

*What to expect.* The goal is not that $A$ reproduces all of $y$. At these positions the MLPs may also compute things the target computation doesn't use. The hope is $y = t + r$, with $t$ the part that matters for the target (e.g. the truth check) and $r$ an unrelated rest, and that the components recover $t$. Then $A$ need not be aligned with $y$. What the measurements can test is whether the rest $y - A$ behaves like unrelated other work or *undoes* what the components write. At a single position, $|y - A| > |y|$ holds exactly when $y - A$ points against $A$ by more than half of $A$'s length ($A dot (y - A) < -|A|^2 \/ 2$). An unrelated rest gives no reason for that: if $t$ and $r$ are orthogonal, the error is $|r| \/ |y| < 1$ and the cosine between $A$ and $y$ equals $|A| \/ |y|$. An error above 1, the more so with a large $|A|$, means the rest of the weights cancels the components' write, i.e. the components produce the target output through writes the model itself does not make there.

Measures, per layer and position, then averaged:
- *relative error* $|y - A| \/ |y|$: below 1 = compatible with $A$ being a part of $y$; exactly 1 where no component is active ($A = 0$), the same as switching the MLP off; above 1 = the rest points against $A$ (see above);
- *cosine* between $A$ and $y$: 1 = same direction, 0 = orthogonal; likewise between $A + I$ (all components, Δ off) and $y$;
- *norm ratios* $|A| \/ |y|$, $|I| \/ |y|$, $|D| \/ |y|$ (sums over layers 15–19 of the mean norms, divided by the sum of the mean $|y|$), the cosine between $I$ and $D$ (−1 = they cancel each other), and the relative error of the Δ-only state, $|y - D| \/ |y|$.

At the period, all 1,794 test statements (mean $|y|$ = 30.5, the same for every run since it is the original model's output):

#tbl(
  columns: (1.6fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  [run], [active per layer], [relative error], [cosine A, y], [cosine A+I, y], [$|A| \/ |y|$], [$|I| \/ |y|$], [$|D| \/ |y|$], [cosine I, D], [error of Δ only],
  [Exp 0, next token], [1.56], [1.26], [0.12], [0.11], [0.85], [0.53], [1.36], [−0.36], [1.02],
  [*Exp 1, residual*], [2.09], [*0.83*], [*0.56*], [*0.58*], [0.76], [0.19], [0.82], [−0.12], [0.79],
  [Exp 2, stimulus], [0.17], [1.18], [0.00], [−0.01], [0.21], [1.65], [2.07], [−0.78], [1.78],
  [Exp 3, at the period], [0.40], [1.07], [0.00], [0.01], [0.11], [1.95], [2.24], [−0.84], [2.05],
  [Exp 3, all 9 positions], [0.73], [1.54], [0.01], [0.02], [0.67], [2.01], [2.55], [−0.79], [2.36],
  [arm A], [1.86], [1.24], [0.08], [0.08], [0.77], [0.58], [1.36], [−0.41], [0.99],
  [untrained], [44.0], [1.03], [0.00], [0.00], [0.25], [0.28], [1.07], [−0.26], [0.37],
)
#verified[`cancel_split.py` on an RTX 4090, all test statements, per-layer logs; the active counts reproduce the verdict analysis. Cosine A+I, y added 2026-10-16 in a rerun on an H100 that reproduces all other numbers within bf16 noise]

Exp 3 per position over its span (pooled over the five layers):

#tbl(
  columns: 6,
  [position], [active per layer], [$|A|$], [$|y|$], [cosine A, y], [cosine A+I, y],
  [period], [0.40], [3.3], [30.5], [0.00], [0.01],
  [" Is", " this", " true", " or", "?", " Answer"], [0.00–0.01], [0.0], [29.4–34.7], [–], [≤ 0.02],
  [" false"], [0.49], [16.1], [31.8], [0.01], [0.01],
  [":" (answer token)], [5.6], [*169.2*], [31.7], [0.06], [0.05],
)

#finding(title: "Where the components write, the rest of the weights undoes it (except partly in Exp 1)")[
  - *Exp 0 and arm A* write a vector of about $y$'s size (0.77–0.85 of its norm), nearly orthogonal to it (cosine 0.08–0.12), with error 1.24–1.26: on average the rest $y - A$ cancels more than half of the components' write. That is not what a truth part $t$ next to an unrelated rest would look like.
  - *Exp 2 and Exp 3 at the period* write little (0.17 and 0.40 active components per layer; at about 90% of position–layer pairs none is active, where the error is exactly 1). Their mean errors (1.18, 1.07) then imply errors of about 2.7–2.9 and 1.6–1.7 where components are active (arithmetic from the logged means): the little they write is also undone. Separately, their inactive components and Δ are each about twice the size of $y$ and point in nearly opposite directions (cosine −0.78 / −0.84). Both are off in the components-only state, so this doesn't affect it, but the full model is then a near-cancellation of large terms, which matters for every ablation from the full model.
  - *Exp 3 at the answer token ":"*: the components write 5.3 times the norm of $y$ (169 vs 32) at cosine 0.06, so the rest of the weights must carry nearly all of $-A$.
  - *Exp 1* is the only run compatible with "the components are a part of $y$": error 0.83, cosine 0.56, little cancellation between $I$ and $D$ (cosine −0.12).
  - *All components together are no closer to $y$ than the active ones*: the cosine of $A + I$ with $y$ is within 0.02 of that of $A$ in every run (0.58 in Exp 1, 0.11 or less elsewhere). So the inactive components are not the rest $r$ of the MLP output either. Outside Exp 1, since $y = (A + I) + D$ with $A + I$ nearly orthogonal to $y$, Δ's write carries the whole component along $y$, and its remaining, orthogonal part is what cancels the components ($y - (A + I) = D$).
  #concluded
]

*Why.* Every training loss in these runs compares only the model's output at the lens (next token, layer-23 residual, verdict); nothing constrains what the components write inside the MLP, only its effect downstream. Since components + Δ = the original weights by construction, whatever the active components write beyond the MLP's own output must be undone by the inactive components and Δ in the full model. The residual lens constrains the MLP writes most, because layer 23 is only four layers downstream and all of its channels count; a KL on a next-token distribution can be met by many different writes (reasoned from the training objective, not tested by changing the loss) #concluded.

*Consequences.*
- Except partly for Exp 1, the "components only" numbers in this report describe the decomposition's own way of producing the target output, not a part of the computation the MLPs perform. The same holds for arm A, whose single-component ablations were the basis of an earlier truth-writing result.
- The cancellation explains why removing all components (Δ only) is so destructive: in Exp 2 and 3, Δ alone is a write 2.1–2.6 times the size of the MLP output (relative error 1.78–2.36, against 1 for switching the MLPs off).
- A fix under test (Exp 4b, 2026-10-16): a reconstruction term that asks the components to reproduce each decomposed MLP's whole output at the decomposed positions. This asks for more than the goal $A approx t$ requires: it also makes the components cover the rest $r$. A weaker condition matching the goal would only forbid the rest from undoing the components' write, e.g. a penalty on negative $A dot (y - A)$ (an untested idea) #assumed.

= Where truth is written

The *truth direction* at layer $l$: the difference between the mean residual stream of true and of false statements after layer $l$ at the period (fit on training statements). Its *separation* is the difference of the two class means projected on it. Analyses on the bare statements (no question), at the period.

*Who writes it.* Share of the layer-19 separation written by the components' part of the weights vs the remainder Δ, summed over layers 15–19 (the rest is written by other layers):

#tbl(
  columns: 7,
  [], [Exp 0], [Exp 1], [Exp 2], [Exp 3], [arm A], [untrained],
  [components], [0.22], [*0.71*], [−0.02], [0.00], [0.15], [0.00],
  [Δ], [0.60], [0.11], [0.83], [0.81], [0.67], [0.82],
)

*Single components.* Largest drop of the layer-23 separation (total 70.1) when one component is removed from the weights: arm A 19.0 (two components, 27% each), Exp 3 17.3, Exp 0 11.7, Exp 2 11.8, Exp 1 6.8, untrained 1.7. #verified[`truth_writers.py`]

Reading #concluded: Exp 1, whose loss is the layer-23 residual, is the only run where components write most of the truth direction, which is what a residual-matching loss should produce. The verdict-trained runs (Exp 2, 3) write none of it through components, so what they carry from the period is not the projection on this direction #assumed (their components are also mostly inactive on bare statements, which a question-free analysis cannot see).

= Do the runs share components?

For each component active on ≥ 1% of statements in run A: the most similar component of run B by cosine of the rank-1 weight $U_c V_c^top$, per layer, all ten pairs of Exp 0–3 and arm A. Every match above 0.1 pairs a component with the *same index* in the other run. Different-index best matches reach at most 0.069, mostly below 0.01 (random baseline: 99th percentile below 0.001). All runs start from identical initial weights (seed 0), so same-index matches may be leftovers of the shared initialisation. The comparison shows no shared component beyond that #concluded. Telling the two apart needs each component’s cosine to its own initial weights, or a second seed.

= Caveats

#caveat[
  - *One seed per run.*
  - *Components only ≠ part of the MLP computation.* Except partly in Exp 1, the rest of the MLP output undoes what the active components write (section "Are the components a part of the MLP outputs?"), so every components-only and single-ablation result describes the decomposition's own way of producing the output.
  - *Truth or negation?* The verdict effects at the period concentrate in negated statements, where the five layers reverse the answer. The components may implement "negation flips the answer" rather than truth; Exp 0 does not separate the two.
  - *Small families.* The negated families have 37–64 test statements each, except negated cities (332).
  - *One question, one scoring.* All verdict numbers depend on " Is this true or false? Answer:" and on the token sets used to score it.
  - *bf16.* Rounding alone moves single verdict scores by up to 0.37 (period) and 0.50 (period and question), so single-component effects at the period (at most 1.3 in separation) are small.
]

= Next

- Where the judgement that gates Exp 3’s "True" writers is computed: attribute their CI and input at ":" to earlier positions and layers.
- Truth vs negation: compare per-family single ablations in Exp 2/3 with a negation-only control (statements where "not" does not flip the truth value).
- Component overlap against the shared initialisation, to make the comparison between runs informative.

= Files

#text(size: 9pt)[#set par(justify: false)
Runs: WandB `bitt-j-personal/spd/<id>` (configs `spd/experiments/lm/honesty_targeted_decomposition/config_truth_{last_token,residual_lens,stimulus_lens,stimulus_span}.yaml`). Local outputs: `~/spd_out/26-10-10_bolder_experiments/` (`full/`: training; `full/v/`, `analysis/out/v/`: verdict; `analysis/out/{tw_tiu,overlap,lead}/`: truth direction, overlap, answer-token components; `dist_shift/cancel_all/`: components vs MLP outputs). Project record: `convos/julian/26-10-10_bolder_experiments_LOG.md`.
]
