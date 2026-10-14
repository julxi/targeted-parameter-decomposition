#import "template.typ": *

#show: report.with(
  title: "Exp 2: a decomposition trained on the model's true/false verdict",
  subtitle: "Stimulus-lens tPD of Qwen2.5-7B-Instruct on simple true/false statements",
  date: "2026-10-13",
  author: "Written by Claude Opus 5.5 for Julian Bitterlich",
  summary: [
    - *What Exp 2 is.* A targeted parameter decomposition (tPD) of five MLP layers of Qwen2.5-7B-Instruct, applied only at the final period of simple true/false statements. Its training loss is the model's *verdict*: the next-token distribution after the appended question " Is this true or false? Answer:".
    - *It is very sparse.* On held-out statements, 0.8 of the 480 components are active at the period on average; 13 components are active on at least 1% of statements.
    - *Those few components carry much of the verdict-relevant work of the five layers.* Switching all five layers off at the period costs 0.116 nats of verdict KL; keeping only Exp 2’s active components costs 0.033. An earlier decomposition with the same layers, trained on ordinary next-token prediction ("arm A"), costs 0.092 with 9.3 active components; an untrained decomposition costs 0.115, the same as switching everything off.
    - *The effect sits in negated statements.* In "The Spanish word 'uno' does not mean 'one'."-type statements, switching the layers off flips the verdict (accuracy 0.86 → 0.06); Exp 2’s components alone restore it (0.95); arm A's reach 0.50, the untrained ones 0.17.
    - *Not yet established:* whether this is a truth mechanism or a negation mechanism, and how much of it any lens would find at this position. The reference run with the plain next-token loss at the same position (Exp 0) is still training. One seed.
  ],
)

= What was run

== Method in brief

*Parameter decomposition* writes each decomposed weight matrix $W$ as a sum of rank-1 *components* plus a full-rank remainder: $W = sum_c U_c V_c^top + Delta$. A small *causal-importance (CI) network* outputs, at every token position, a value in $[0, 1]$ per component: how much the component is needed there. Training asks for two things: with components masked by their CI values (randomly, and adversarially in the last 20% of training), the model must behave like the original; and few components may be active at once. *Targeted* PD (tPD; Vigouroux & Sharkey 2026, arXiv:2607.13047) trains this on a chosen *target* dataset, with Δ switched off there, so the components must carry the target computation, while on general text (the *non-target* stream, here general web text from the Pile dataset) Δ stays on, so the decomposition need not explain everything else the model does.

Everything here decomposes the MLP output matrices (`down_proj`) of layers 15–19 of Qwen2.5-7B-Instruct (28 layers), 96 components each, 480 in total. Components are replaced only at selected token positions (the *decomposed positions*); every other position runs on the original weights.

== Exp 2: the stimulus lens

Earlier decompositions on these statements were trained to preserve the model's next-token predictions. Their components turned out to carry little truth-specific structure. One explanation: a truth computation that changes no next-token prediction costs nothing to drop. Exp 2 asks the model for its verdict and trains on that instead.

#tbl(
  columns: (auto, 1fr),
  align: (left, left),
  [Setting], [Exp 2 (#runid("s-05db01ad"))],
  [Target data], [The 20 Truth-is-Universal datasets (*tiu*; Bürger et al. 2024): 10 families of simple statements, each with true and false versions, e.g. "The city of Krasnodar is in Russia.", "Fifty-one is larger than fifty-eight.", "Zinc does not have the symbol Fr."; 6,560 training statements, 1,794 test statements whose subjects never occur in training],
  [Stimulus], [" Is this true or false? Answer:" appended to every statement, tokenized separately so the statement's tokens are unchanged],
  [Decomposed position], [The statement's final period only. The stimulus tokens run on the original weights.],
  [Target loss], [KL divergence between the original and the masked model's next-token distribution at the last stimulus token (":"), i.e. at the model's verdict],
  [Non-target stream], [Pile text, Δ on, next-token KL at all positions],
  [Training], [5,000 steps, 64 statements + 16 Pile sequences per step, learning rate 5e-4; 47 min on one H100],
)

*Why this stimulus.* In a screen of five stimuli on 546 test statements, " Is this true or false? Answer:" put the most probability on true/false-type answer tokens (0.74 on average) and separated true from false statements best (AUC 0.998: the area under the ROC curve of the true-vs-false score, 1 = perfect ranking, 0.5 = chance). Asked only " Is this true or false?", the model mostly starts reasoning ("To determine whether…"). With a trailing space after "Answer:", the most likely next token is the digit "1". #verified[forward passes, 546 statements]

= Results

== Training

#tbl(
  columns: 7,
  [step], [active components per statement], [verdict KL, rounded CIs], [verdict KL, Δ only], [adversarial KL], [Pile KL, rounded CIs], [],
  [1,000], [1.7], [0.036], [0.035], [1.98], [0.084], [],
  [4,000], [1.0], [0.037], [0.022], [0.93], [0.056], [adversarial phase starts],
  [5,000], [0.47], [0.029], [0.014], [0.60], [0.040], [end],
)
Evaluated on 256 held-out statements every 500 steps; KL in nats. "Rounded CIs": components with CI > 0.01 fully on, the rest off, Δ off. "Adversarial": the worst masks a 20-step search finds within the CI-allowed range. #verified[run metrics]

The adversarial loss was the one worry from a 300-step trial (9.7 nats); it falls to 0.60 in the full run. The Pile side effect ends at 0.040, the same level as a next-token-lens trial at the same position (0.041).

== The verdict under parts of the decomposition

*Verdict score*: log P(true-like tokens) − log P(false-like tokens) at the last stimulus token, where true-like = single-token spellings of true/correct/yes/right and false-like = false/incorrect/wrong/no. The verdict is "true" when the score is positive. *Separation*: mean score of true statements minus mean score of false statements. All 1,794 test statements; replacements at the period only.

States: *components only* = the components with rounded CI on, Δ off; *Δ only* = all components off, Δ on; *all off* = the five MLP outputs set to zero at the period. Reference decompositions with the same layers and size: *arm A* (#runid("s-bd23f0d1")), trained with the ordinary next-token KL at every statement token; *untrained* (#runid("s-7fad0c14")), the initial state of such a decomposition.

#tbl(
  columns: 7,
  [], [active comp. per statement], [accuracy], [AUC], [separation], [KL to original (nats)], [Δ only: KL],
  [original model], [–], [0.953], [0.999], [12.04], [0], [–],
  [*Exp 2* components only], [0.8], [0.958], [0.998], [10.88], [*0.033*], [0.030],
  [arm A components only], [9.3], [0.927], [0.992], [9.84], [0.092], [0.044],
  [untrained components only], [220], [0.918], [0.984], [10.31], [0.115], [0.008],
  [all off], [0], [0.918], [0.983], [10.21], [0.116], [–],
)
#verified[`verdict_accuracy.py`, one H100 pass per state]

Reading #concluded:
- At the period, the five layers matter for the verdict, but not most of it: removing them all costs 0.116 nats and 15% of the separation. The rest of the verdict reaches the stimulus through other layers, or the stimulus recomputes it from the statement.
- Exp 2’s single active component, on average, recovers about 70% of that KL (0.116 → 0.033) and a third of the lost separation. Arm A's components, nine on average, recover a fifth of the KL and *lose* separation compared with switching everything off (9.84 vs 10.21). The untrained components do nothing: their state costs the same as switching everything off, presumably because component weights start small #assumed. Their 220 "active" CIs come from an untrained CI network.
- This is what Exp 2 was trained for, so recovering the verdict is expected. What is informative is how little it takes, and that it holds on test statements whose subjects never occurred in training.

== Where the effect sits: negated statements

Verdict accuracy per family (true and false statements together; families with no difference between the states omitted: affirmative cities, animals, element symbols, translations, number comparisons are at 0.96–1.00 in every state):

#tbl(
  columns: 6,
  [family (test statements)], [original], [all off], [*Exp 2* components], [arm A components], [untrained components],
  [negated Spanish translations (64)], [0.86], [0.06], [*0.95*], [0.50], [0.17],
  [negated element symbols (56)], [0.55], [0.36], [0.59], [0.50], [0.23],
  [negated animal classes (37)], [0.73], [0.68], [0.76], [0.62], [0.70],
  [negated cities (332)], [0.93], [0.94], [0.94], [0.91], [0.94],
)
#verified[`verdict_accuracy.py` per-family output]

#finding(title: "The clearest single result")[
  For negated translations ("The Spanish word 'uno' does not mean 'one'."), the five layers at the period are needed for the verdict: without them the model gets 6% right, i.e. it reverses nearly every verdict. Exp 2’s components alone restore 95% — above the original model's 86%. The same layers matter less for the other negated families, and not at all for affirmative statements.
]

*Single components.* Ablating one of Exp 2’s 13 frequently active components from the full model changes the separation by at most 0.38 (component 116, active on 12% of statements); four change it by more than 0.2. For arm A (55 components) the largest change is 0.33, three exceed 0.2; for the untrained decomposition (475) none exceeds 0.07. Rounding in bf16 alone moves individual verdict scores by up to 0.25 (the separation stays at 12.04), so single-component effects of this size are small. #verified[`verdict_accuracy.py`]

= Caveats

#caveat[
  - *One seed, one run.* No repeat of Exp 2 exists.
  - *The missing reference.* Exp 0, the same decomposition at the same position trained with the model's own next-token KL instead of the verdict, is still training. Arm A is decomposed at every token and evaluated here at the period only, so it is not a like-for-like comparison. Whether the verdict recovery is specific to the stimulus lens needs Exp 0.
  - *Truth or negation?* The effect concentrates in negated statements, where the verdict reverses without the layers. The components may implement "negation flips the answer" rather than anything about truth. The per-family data shown cannot separate the two; per-component, per-family ablations can.
  - *Small families.* The negated families have 37–64 test statements each.
  - *The stimulus is fixed.* All verdict numbers depend on " Is this true or false? Answer:" and on the token sets used to score it.
]

= Next

- Compare with Exp 0 (next-token lens at the period, training now) and Exp 1 (a residual-stream lens: the loss compares the residual stream after layer 23 at the period, also training now) under the same analysis.
- Exp 3, queued: the same verdict loss, but with the stimulus tokens decomposed too, so the verdict cannot be recomputed on the original weights.
- For Exp 2’s top components: which families each one affects, and whether the effect on negated translations comes from one component.

= Files <sec-files>

Run: WandB `bitt-j-personal/spd/s-05db01ad` (config `spd/experiments/lm/honesty_targeted_decomposition/config_truth_stimulus_lens.yaml`, commit `ae7f041`). Analysis outputs (local): `~/spd_out/26-10-10_bolder_experiments/full/` (training metrics, `v/verdict_*.json`). Stimulus screen and zero-ablation measurements: `~/spd_out/26-10-10_bolder_experiments/r1_stimulus_screen.out`, `r2r3_zero_ablation.out`. Project record: `convos/julian/26-10-10_bolder_experiments_LOG.md`.
