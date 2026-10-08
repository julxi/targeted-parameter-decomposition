#import "template.typ": *

#show: report.with(
  title: "tPD on Qwen2.5-7B for truthfulness: state of the experiments",
  subtitle: "What was tested, what the probes take as input, what came out, and the open options",
  date: "2026-10-06",
  author: "Written by Claude Opus 5.5 for Julian Bitterlich",
  summary: [
    Unless noted, every number is the test accuracy of a logistic-regression probe on features read at the *last token* of each sample; chance is 0.50. "Simple statements" = the Truth-is-Universal data (*tiu*), tested on held-out subjects; "diverse data" = the Universal Truthfulness Hyperplane data (*UTH*), tested on task categories the probe never saw.
    - *Probes on causal-importance (CI) values detect truth, but for a trivial reason.* An *untrained* CI network reads statement truth as well as the raw activations do (0.998 on simple statements). Training adds nothing measurable: when each statement keeps only its k largest CI values, the trained network beats the untrained one by 1–2 points, about the spread (1.4 points) between five trained arm-A runs that differed only in batch size, steps and learning rate.
    - *On diverse data the trained CIs carry no truth that transfers to unseen task categories* (0.488, chance; untrained CIs 0.717; activations 0.806). The information is there within each new task (within-task probes 0.62–0.92), but the readout learnt on the training tasks does not carry over.
    - *Most of the direct truth writing in the decomposed layers stays in the remainder Δ (the part of each weight matrix not captured by components), not in the components* (components 15% vs Δ 67% on simple statements; 7% vs 61% on diverse data).
    - *The one positive lead:* in both trained decompositions, ablating a *single* component removes 27–31% (simple statements) or about 45% (unseen tasks) of the true/false separation along the layer-23 truth direction (mean true minus mean false of the residual stream after layer 23). In the untrained and code-trained decompositions no component comes within a factor of 10. These ablations have *no controls yet*, so the lead is unconfirmed.
    - *Recommended next step:* the controlled ablation experiment (called "B": the same ablations with random-component and non-truth controls, @sec-next), minutes of GPU time. Further CI-probe accuracy comparisons are unlikely to tell us anything new.
  ],
)

= Scope and how to read this report

This report collects the tPD truthfulness experiments of 2026-10-02 to 2026-10-06 in one place, so that the next steps can be chosen with the whole picture in view. It covers four experiments on three trained decompositions and one untrained reference:

+ CI probes on the Truth-is-Universal statements (*tiu*), with the untrained baseline (@sec-exp1).
+ A control decomposition trained on lines of code (@sec-exp2).
+ CI probes across task categories on the Universal Truthfulness Hyperplane data (*UTH*) (@sec-exp3).
+ Component-level analyses: which components write into the truth direction, what happens when one is ablated, and whether the model's next-token output depends on truth (@sec-exp4).

Every number was re-read from the result files listed in @sec-files, not copied from memory. Each run is a single seed unless stated otherwise. Status tags mark how solid each claim is: #verified[evidence] means a measurement was made, #concluded means it follows from stated reasoning, and #assumed means a working assumption or untested prediction. Abbreviations are expanded in the glossary (@sec-glossary).

= Background <sec-background>

== tPD in brief

*Parameter decomposition (PD)* writes a weight matrix $W$ as a sum of rank-1 *components* $U_c V_c^top$. A small *causal-importance (CI) network* looks at the model's activations at each token position and outputs one value $"CI"_c in [0, 1]$ per component: how much the component matters at that position. Training asks two things at once:
- *reconstruction*: with the components masked according to their CI values (randomly and adversarially perturbed), the model's next-token predictions must stay close (in KL) to the original model's;
- *importance minimality*: few components may have non-zero CI at any token.

*Targeted PD (tPD)* (Vigouroux & Sharkey 2026, arXiv:2607.13047) aims to decompose only what a chosen *target* dataset uses. Each decomposed matrix gets a full-rank remainder $Delta = W - sum_c U_c V_c^top$. On target batches Δ is scaled down (by a random factor in $[0,1]$ per position, or adversarially), so the components must carry the target computation. On general text (the *non-target* stream, here the Pile) Δ stays fully on.

This is a trade-off, not a guarantee. A target mechanism that would need many active components can stay in Δ, at the price of a worse target reconstruction. The target reconstruction loss measures how much was left there. So "tPD decomposes what the target uses" is the design intent, not an established property of our runs.

One more property matters for everything below #concluded: tPD sees the model only through the *next-token predictions* on the training text. A computation that changes no next-token prediction there (for example, truth computed "for later use" that nothing in the text reads) costs nothing to lose, so tPD has no reason to put it into components.

== The question and the null hypothesis

*Goal* #decided: find out to what extent tPD on Qwen recovers mechanisms related to statement truthfulness, and find evidence for them. The starting point was someone else's unreplicated result: tPD on Qwen's middle layers, with target statements from the UTH domains and padding positions included in the training loss; linear probes (logistic regression, mass-mean) on the CI values, probably averaged over positions, detected statement truth. That result had no baseline.

*Null hypothesis* #concluded: the CI network is a learned nonlinear readout of mid-layer activations, and truth is known to be linearly readable from those activations. So CI values may carry truth for this mundane reason, whether or not tPD found anything truth-specific. Every experiment below is built around separating these two explanations.

= The decompositions

== Shared setup

All decompositions share one architecture and training recipe. Only the target data differs.

#tbl(
  columns: (auto, 1fr),
  align: (left, left),
  [Setting], [Value],
  [Model], [Qwen2.5-7B-Instruct, frozen, loaded in bf16; 28 decoder layers, residual width 3,584, MLP hidden width 18,944],
  [Decomposed matrices], [MLP `down_proj` of layers 15–19 (0-indexed), 96 components each, *480 components in total*. Each component $c$ has an input direction $V_c$ (18,944-dim, reads the MLP hidden vector $h$) and an output direction $U_c$ (3,584-dim, writes into the residual stream)],
  [CI network], [One shared MLP for all five layers, applied to each token position separately: input = the five `down_proj` inputs at that position concatenated ($5 times 18,944 = 94,720$ dims), one hidden layer of 512, output 480 values, then clamped to $[0, 1]$. No attention, so the CI at a position depends only on that position's activations],
  [Loss positions], [All real tokens of each sample. On tiu this is called *arm A*; two further arms were designed but never run: the *last-token arm* (loss on the final token only) and the *padded arm* (loss on all positions including right-padding, more than half of them padding, as in the motivating setup). Positions outside the loss run on the original weights],
  [Losses], [Stochastic-mask reconstruction (components masked by their CI values plus random noise, Δ randomly scaled; KL on the next-token output) ×1; persistent adversarial reconstruction (masks optimised to maximise the KL, PGD) ×0.5 in the last 20% of steps; importance minimality ×1e-3 (×2 on non-target data)],
  [Non-target stream], [The Pile (`monology/pile-uncopyrighted`), 64-token chunks, batch 16, Δ fully on],
  [Training], [Batch 16, 5,000 steps, LR 5e-4 cosine, about 35–40 min on one H100. Tuned on arm A only, for reconstruction quality and speed, one seed per setting],
  [Never tuned], [The choice of layers 15–19 (inherited from an earlier config whose own comment calls it "an unvalidated guess"), C = 96 per layer, the loss coefficients],
)

== Datasets

- *tiu* (Truth-is-Universal, Bürger et al., NeurIPS 2024): short templated statements in 10 families, each true and false: cities ("The city of Krasnodar is in Russia."), animal classes, chemical element symbols, Spanish→English translations, larger-than and smaller-than number comparisons, and negated versions of four of these. 20 datasets, *6,560 train / 1,794 test* statements, 50% true, split by subject (a city, word, element, animal or number pair never appears in both splits). Mean length about 10 tokens. The last token is "." (or "'." for the translations).
- *UTH* (Liu et al., "On the Universal Truthfulness Hyperplane Inside LLMs", EMNLP 2024): 49 datasets in 17 task categories, each sample a text with a true/false label, e.g. `"Question: …\nAnswer: …"` with a correct or a plausible wrong answer.
  - *Training tasks*: 29 datasets in 10 categories (only datasets where at least 80% of samples fit in 128 tokens), *20,292 train / 5,694 held-out* samples.
  - *Test tasks*: the paper's 3 held-out categories (sentence completion: copa, hellaswag, `story_cloze`; short-answer QA: `nq_re`, `triva_qa_re`, sciq; summarization: `cnn_dailymail_re`, `xsum_re`). 8 datasets, *7,200* samples, never seen by any decomposition or probe fit, and uncapped in length.
  - Two label bugs in the source data were fixed by keeping only intact true/false pairs (`story_cloze`, `definite_pronoun_resolution`).
- *Code lines*: 8,354 single lines of code from the Pile's GitHub subset, with the same length distribution and split sizes as tiu. Used only as a control target.

== Runs

All runs are on WandB (`bitt-j-personal/spd`). Final evaluation on held-out target data. "Target KL" values are in nats per token with Δ switched off, i.e. they measure the components alone. Active components are averaged over all tokens here; at the last token the numbers are higher (arm A 9.3, UTH run median 12).

#tbl(
  columns: (auto, auto, auto, auto, auto, auto),
  [Run], [Target], [Active comps per token (all tokens)], [Target KL, active comps only], [Target KL, all comps off], [Target KL, all 480 on],
  [#runid("s-bd23f0d1") arm A], [tiu], [7.0], [0.136], [3.54], [0.135],
  [#runid("s-b6cce5de") code control], [code lines], [2.2], [0.315], [0.41], [0.359],
  [#runid("s-d2ded461") UTH], [29 UTH datasets], [7.8], [0.433], [2.80], [0.479],
  [#runid("s-7fad0c14") untrained], [none (0 steps)], [about 220 (last token, on tiu)], [–], [–], [–],
)

How to read the KL columns: "all components off" is how much the five matrices' decomposed part matters on that data; "active components only" is what remains unreconstructed. Arm A leaves about 4% of the components' effect unreconstructed (0.136 of 3.54). The UTH run leaves about 15% (0.433 of 2.80), and even all 480 components without Δ reproduce the layers poorly (0.479). So the UTH decomposition is limited by capacity or training, not by sparsity #concluded. The code control is small because layers 15–19 matter little for isolated code lines (0.41).

Five further arm-A runs come from the tuning of batch size, steps and learning rate, before the frozen model was switched from fp32 to bf16 loading (which changes only rounding; bf16 is faster and halves memory): run 1 #runid("s-c2562082") (batch 4, 10k steps), e2b #runid("s-ba5f9dd5") (batch 16, 5k steps, the final settings), e2c #runid("s-b3401a03") (batch 4, 10k steps), e3a #runid("s-a77ae04e") (batch 16, 2.5k steps, LR 1e-3), e3b #runid("s-83033a96") (batch 16, 5k steps, LR 1e-3), plus an fp32 untrained reference #runid("s-286aa6a9"). They give the run-to-run spread used below. The probe script needs one dtype per pass, so fp32 and bf16 runs were probed in separate passes.

= What the probes take as input <sec-inputs>

== One forward pass, one position

Every probe feature comes from *one forward pass of the original, unmodified model*. The CI network of each decomposition is then applied to the inputs of the five `down_proj` matrices (layers 15–19) captured in that pass, at the readout position only (the CI network sees one position at a time). So CI values describe how a decomposition *would* gate its components on that input; the components themselves are not in the forward pass. All runs share the frozen model, so one pass serves all of them.

The *readout position* is the *last real token* of each sample (tiu: the final "."; UTH: the last token of the text). Experiments 1–3 also probed the *mean over all real tokens* (the feature vector, e.g. the 480 CI values, averaged over the sample's positions). On tiu it was slightly better for trained CIs (arm A 0.994 vs 0.991 at the last token; fp32 runs 0.979–0.996 vs 0.970–0.984) and slightly worse for the untrained network (0.993 vs 0.998). On UTH it was worse for every feature set except the trained CIs, which stayed near chance (residual L19 0.705 vs 0.806; untrained CIs 0.596 vs 0.717; trained CIs 0.511 vs 0.488, test tasks). It changes none of the conclusions, so the tables below show the last token only.

== Feature sets

#tbl(
  columns: (auto, auto, 1fr),
  align: (left, center, left),
  [Feature set], [Dims], [What it is],
  [CI values], [480], [The CI network's output at the readout position, clamped to $[0,1]$. A component counts as *active* if its CI exceeds 0.01],
  [Top-k CI (matched sparsity)], [480], [Per sample, keep only its k largest CI values and set the rest to 0 (k = 1 … 50). Which components survive differs per sample and is chosen by size, not by label. This makes trained (about 10 active) and untrained (about 220 active) networks comparable],
  [Binary on/off], [480], [1 if CI > 0.01, else 0],
  [Top-5 by label], [5], [The same 5 components for every sample: those with the largest |mean(true) − mean(false)| / standard deviation on the training split (tiu only)],
  [Residual stream after layer $l$], [3,584], [The model's hidden state at the output of decoder block $l$ (after its attention and MLP), the "probe the activations" baseline. Layer 19 (the last decomposed layer) by default; UTH also swept layers 7, 11, 15, 19, 23, 27],
  [Inner activations $a_c$], [480], [$a_c = V_c dot h$: how strongly each component's input direction is present, *ungated* by CI. For the untrained run these are 480 random projections of $h$],
  [CI-gated activations], [480], [$a_c times "CI"_c$],
  [Output log-probs], [1,001], [Next-token log-probabilities at the readout position, over the 1,000 tokens most likely there on average plus one bucket for the rest],
  [Mean log-probability], [1], [The model's mean log-probability of the sample's own tokens (a likelihood baseline)],
)

== Probe types

- *Logistic regression*: each feature standardised to mean 0, variance 1 on the training split, then L2-regularised logistic regression. On tiu (Experiments 1–2) the strength was the scikit-learn default C = 1, untuned. On UTH and in Experiment 4, C was chosen from {1e-4, 1e-3, 1e-2, 0.1, 1} on a held-out split of the *training* data, never on the test data. Because of the standardisation, a weak but consistent feature counts as much as a strong one; only exact zeros carry no information.
- *Mass-mean*: direction = mean(true) − mean(false) on the training split, threshold at the midpoint of the projected class means. No scaling, no hyperparameter. Usually a few points below logistic regression.

== Evaluation protocols

- *tiu, in-distribution*: fit on the train split, score on the test split (held-out subjects, same families). Chance 0.50. Standard error at 1,794 samples: about 0.004 at 97% accuracy, 0.012 near chance.
- *UTH, cross-task*: fit on the training tasks, C chosen on their held-out samples, score on the 8 test datasets from unseen categories. Reported as the *unweighted mean over the 8 datasets* (the paper's measure; standard error about 0.006).
- *Within-task cross-validation*: 5-fold CV inside each test dataset, true/false pairs kept in the same fold, logistic regression with the C chosen for the cross-task probe. Separates "no truth information in the features" from "information present, but the readout learnt elsewhere does not transfer".

== Overview: what each experiment probes

#tbl(
  columns: (2.3cm, 2.3cm, 1fr, 3.2cm, 2.6cm),
  breakable: true,
  align: (left, left, left, left, left),
  [Experiment], [Position], [Features], [Probe], [Fit → scored on],
  [Exp. 1: tiu CI probes], [last token (also mean)], [CIs of arm A (bf16 and 5 fp32 runs) and untrained; top-k CIs, binary, top-5; residual L19; mean log-prob], [logistic regression (C = 1), mass-mean], [tiu train → tiu test (held-out subjects)],
  [Exp. 2: code control], [last token (also mean)], [CIs of the code-trained decomposition, on tiu], [same], [same],
  [Exp. 3: UTH cross-task], [last token (also mean)], [CIs of the UTH run and untrained; top-k CIs, binary; residual L7–L27; mean log-prob], [logistic regression (C on training tasks' held-out split), mass-mean], [training tasks → 8 unseen test datasets],
  [Exp. 4: component analyses], [last token (ablations: all positions)], [CIs, inner activations $a_c$, CI-gated $a_c$, output log-probs, residual L15–19, L23, per-sublayer outputs; component weights $U_c, V_c$], [logistic regression (C on tune split); attribution and ablation (no probe)], [tiu: fit → tune → test; UTH: as in 3, plus within-task CV],
)

= Experiment 1: CI probes on tiu <sec-exp1>

*Question*: do the arm-A CI values read statement truth, and better than an untrained CI network?

#tbl(
  columns: (1fr, auto, auto),
  [Feature set (last token)], [Logistic regression], [Mass-mean],
  [Residual stream after layer 19], [0.998], [0.981],
  [Untrained CIs (#runid("s-7fad0c14"))], [0.998], [0.981],
  [Arm A CIs (#runid("s-bd23f0d1"), bf16)], [0.991], [0.918],
  [Arm A CIs, 5 fp32 runs (run 1, e2b, e2c, e3a, e3b)], [0.970–0.984], [0.802–0.924],
  [Mean log-probability], [0.575], [–],
)
#verified[`probe/results.json` of 26-10-05 and 26-10-03; one seed per run]

*Sparsity.* At the last token arm A has 9.3 active components on average (median 7, 10th–90th percentile 5–15); the untrained network 220. The on/off pattern alone reads truth at 0.941, the 5 most label-related components at 0.936. So the truth signal survives in the few active components, and it is not a constant "end of statement" code.

*Matched sparsity* (each statement keeps its k largest CIs):

#tbl(
  columns: (auto, auto, auto, auto),
  [k], [Arm A], [Untrained], [Gap],
  [2], [0.946], [0.926], [\+2.0],
  [7], [0.991], [0.979], [\+1.2],
  [10], [0.994], [0.985], [\+0.9],
)
#verified[`followup.out`, 26-10-05]

#finding(title: "Reading")[
  The motivating result is fully explained by the null hypothesis: an untrained CI network already reaches the ceiling #concluded. Training makes the CIs sparse and costs 0.7 (bf16 run: 0.998 − 0.991) to 2.8 points (fp32 runs, down to 0.970). The trained network's 1–2-point lead at matched sparsity (bf16 runs) is about as large as the spread between the five fp32 runs, which differed only in batch size, steps and learning rate (0.970–0.984, 1.4 points), so it is no evidence that tPD picked truth-specific components #concluded. Because every readout sits at the ceiling, this in-distribution test cannot discriminate between hypotheses.
]

Not done on tiu: generalisation tests (affirmative → negated, leave one family out); the last-token and padded arms.

= Experiment 2: a code-trained control <sec-exp2>

*Question*: does the *target* shape what the CIs carry? A decomposition with arm A's exact settings, but trained on lines of code, probed on the tiu statements.

The code lines carry little of tiu's truth feature: projected onto the layer-19 truth direction, they spread over about a tenth of the true/false gap #verified[projection check, 26-10-05].

#tbl(
  columns: (1fr, auto, auto, auto),
  [On tiu test (last token)], [Arm A], [Code control], [Untrained],
  [CI probe, logistic regression], [0.991], [0.654], [0.998],
  [Active components per statement (mean)], [9.3], [1.4], [220],
  [Statements with no active component (all CIs ≤ 0.01)], [0%], [29%], [0%],
  [Statements with all 480 CIs exactly 0], [0%], [18.5%], [0%],
  [Components ever non-zero], [385], [66], [480],
  [Probe on the 1,462 statements where the control has ≥ 1 non-zero CI], [0.991], [0.656], [0.998],
  [Top-2 CIs per statement (matched sparsity)], [0.946], [0.654], [0.926],
)
#verified[`probe/features.npz`, recomputed 26-10-06]

Per family, the control reads about 0.95 on affirmative cities and animal classes and 0.50–0.59 on the negated families.

#finding(title: "Reading")[
  Where its components are on, the code control still reads truth at only 0.656, against 0.926 for two random (untrained) CIs per statement. (The control rarely has more than 2 non-zero CIs, so its top 2 are nearly all it has.) So training on an unrelated target *removes* truth from the CIs, while training on truth statements *keeps* it #concluded. The target shapes what the CIs read; that tPD *learned* truth beyond a random readout is still not shown (that is the 2-point gap of Experiment 1). The untrained network's 0.998 is a property of a dense readout: a sparse trained CI network carries truth only through components that are active on the probed inputs.
]

= Experiment 3: cross-task probes on UTH <sec-exp3>

*Question*: tiu put every readout at the ceiling. UTH's cross-task protocol (probes tested on unseen task categories) leaves room. Do the trained CIs transfer better than untrained ones?

*Success criterion fixed in advance*: trained CIs beat untrained CIs cross-task at matched sparsity, by clearly more than the run-to-run spread. Otherwise the series ends.

#tbl(
  columns: (1fr, auto, auto),
  [Feature set (last token, logistic regression)], [Held-out samples of training tasks], [Unseen test tasks],
  [Residual stream after layer 7 / 11 / 15], [0.673 / 0.725 / 0.804], [0.585 / 0.651 / 0.768],
  [Residual stream after layer 19 (best)], [0.850], [*0.806*],
  [Residual stream after layer 23 / 27], [0.828 / 0.819], [0.771 / 0.741],
  [Untrained CIs], [0.794], [*0.717*],
  [Trained UTH CIs (#runid("s-d2ded461"))], [0.706], [*0.488*],
  [Mean log-probability], [0.572], [0.561],
)
#verified[`stage1/results.json`, 26-10-05; values are means over datasets]

*Per test dataset* (residual layer 19 / untrained CIs / trained CIs):

#tbl(
  columns: (auto, auto, auto, auto),
  [Test dataset], [Residual L19], [Untrained CIs], [Trained CIs],
  [copa (n 200)], [0.965], [0.960], [0.285],
  [`story_cloze`], [0.959], [0.939], [0.401],
  [sciq], [0.842], [0.836], [0.438],
  [hellaswag], [0.781], [0.570], [0.500],
  [`nq_re`], [0.631], [0.584], [0.540],
  [`triva_qa_re`], [0.721], [0.661], [0.544],
  [`cnn_dailymail_re`], [0.738], [0.552], [0.642],
  [`xsum_re`], [0.809], [0.633], [0.550],
)

*Matched sparsity* (test tasks): trained vs untrained at k = 2: 0.504 vs 0.558; k = 5: 0.504 vs 0.585; k = 10: 0.499 vs 0.593; k = 50: 0.498 vs 0.649; on/off pattern 0.530 vs 0.728. Only at k = 1 are they equal (0.540 vs 0.538). The trained network has a median of 12 active components per sample; 8% of test samples have none (0.4% of training samples).

#finding(title: "Reading")[
  The success criterion is not met, and the effect goes the other way: the trained CIs read truth within the training tasks (0.706) but are at chance on unseen categories, and worse than the untrained network at every k ≥ 2 #concluded. On copa, `story_cloze` and sciq the probe is systematically *wrong* (4–6 standard errors below chance); Experiment 4 traces this to an overfit probe (@sec-a6). Caveat: one seed, and this decomposition fits the data poorly (15% unreconstructed). Whether a better-fitting one (more components per layer) transfers was not tested.
]

= Experiment 4: component-level analyses <sec-exp4>

*Question*: CI probes only use the gating values. Do the components themselves write into the truth direction, where in the model is truth written, and does removing a component change the truth signal? Script `truth_writers.py`, run on tiu with arm A, the code control and the untrained decomposition, and on UTH with the UTH run and the untrained decomposition. Splits: tiu train divided by subject into *fit* (5,260) and *tune* (1,300, for choosing C), test 1,794; UTH: fit = training tasks' train split, tune = their held-out samples, test = the 8 unseen test datasets. The analyses carry the labels A1–A7 and G of the analysis plan; B is the proposed follow-up (@sec-next).

The script re-extracted all features in a new forward pass, so its probe numbers can differ from Experiment 3's by up to about 1.5 points (trained UTH CIs cross-task 0.490 here vs 0.488; copa 0.300 vs 0.285). Likely causes are numerical differences of the new pass and, on tiu, the smaller fit split and the tuned C #assumed. Components are named by layer and index: L17:33 is component 33 of layer 17's `down_proj`.

== Truth direction and separation

The *truth direction* $w$ after layer $l$ is the mass-mean direction (mean true − mean false of the residual stream at the last token, fit on the fit split), normalised to length 1. Called `mm_L19` and `mm_L23` below. Seven global directions were built (mass-mean after each of layers 15–19 and 23, and the logistic-regression direction after layer 19), plus one mass-mean direction per dataset; A1, A2 and A5 use them.

The *separation* $S$ of any vector along $w$ is mean over true samples minus mean over false samples of its projection onto $w$, in raw residual-stream units. Because the residual stream is a sum of sublayer outputs, $S$ splits exactly into one term per sublayer, and within a decomposed layer exactly into one term per component plus one for Δ:
$ S_("mlp"_l) = sum_c S_c + S_Delta, quad S_c = ("mean"_"true" a_c - "mean"_"false" a_c) (U_c dot w). $
$S_c$ is the *direct write* of component $c$ along $w$, computed in the unmodified model (every component fully on, so no CI enters).

== A1: are output directions aligned with truth? No

The largest |cos($U_c$, $w$)| over arm A's 480 components is 0.082 for `mm_L19`. The null: the same maximum against 200 mass-mean directions built from randomly permuted labels, median 0.057. p = 0.03 for `mm_L19`, which does not survive a correction for the 7 global directions tested; the other six are not significant. UTH: none significant. Static alignment says little either way #verified[`report.txt`].

== A2: where is truth written, and by components or by Δ?

Fractions of $S$ along each direction, tiu test split, all statements:

#tbl(
  columns: (1fr, auto, auto),
  [Contribution], [Along `mm_L19`], [Along `mm_L23`],
  [Layers 0–14 (attention + MLP)], [5.4%], [0.9%],
  [Attention, layers 15–19], [13.2%], [2.1%],
  [MLPs, layers 15–19], [81.5%], [13.1%],
  [#h(1em) of which MLP 18 / MLP 19], [25.0% / 41.1%], [3.4% / 7.7%],
  [#h(1em) of which arm A components / Δ], [14.6% / 66.9%], [5.5% / 7.5%],
  [#h(1em) of which code-control components], [−0.4%], [0.3%],
  [#h(1em) of which untrained components], [−0.2%], [−0.2%],
  [Layers 20–23], [–], [83.9%],
)
#verified[exact decomposition; built-in check: terms sum to $S$ within bf16 rounding]

With each tiu family's own direction, arm A's components carry 4–13% and Δ 59–73%. On UTH (along `mm_L19`): components 7% (training tasks) / 8% (test tasks), Δ 61% / 59%; untrained components 0%.

#finding(title: "Reading")[
  - Most of the direct truth writing in layers 15–19 stays in Δ #verified[A2]. The trained decompositions capture a real but minor part of it; untrained and code-trained components capture none.
  - Each layer's truth direction is mostly written by the MLPs just before it (`mm_L19` by MLPs 18–19, `mm_L23` by MLPs 20–23). If truth were written once in early layers and then carried along, both directions would credit the same early layers; instead each credits its own recent layers. So truth looks rewritten layer by layer #concluded (from two directions only), and direct attribution cannot single out "the" layers that construct truth: it only sees the last write. Causal measurements (A7) also count indirect effects.
]

== A3 and A4: probes on component features, and within-task probes

On tiu, every feature set built on the ungated inner activations $a_c$ reads truth at 0.997–0.998, *including the untrained and code-control decompositions*: the MLP hidden vector carries truth, and such probes cannot tell decompositions apart.

On UTH (test tasks, mean over datasets): trained CIs 0.490, trained $a_c$ 0.773, untrained CIs 0.717, residual L19 0.807, output log-probs 0.747. *Within each test dataset* (5-fold CV) the trained CIs do read truth #verified[A4]:

#tbl(
  columns: (auto, auto, auto, auto, auto),
  [Test dataset], [Trained CIs], [Untrained CIs], [Trained $a_c$], [Residual L19],
  [copa], [0.920], [0.945], [0.970], [0.965],
  [`story_cloze`], [0.920], [0.998], [1.000], [1.000],
  [sciq], [0.784], [0.890], [0.917], [0.917],
  [hellaswag], [0.666], [0.896], [0.894], [0.915],
  [`cnn_dailymail_re`], [0.778], [0.981], [0.989], [0.996],
  [`xsum_re`], [0.800], [0.962], [0.963], [0.973],
  [`nq_re`], [0.616], [0.603], [0.693], [0.691],
  [`triva_qa_re`], [0.618], [0.704], [0.762], [0.784],
)

#finding(title: "Reading")[
  The UTH cross-task failure is a readout that does not transfer, not missing information #concluded. The trained CIs carry truth on the unseen tasks, though less than the untrained ones (by up to 0.23 on hellaswag and 0.20 on `cnn_dailymail_re`; about equal on `nq_re`), but the direction that separates true from false in CI space differs between tasks.
]

== A5: which components write, per kind of statement (tiu)

Activity = share of samples with CI > 0.01; write = $S_c$ as a share of the dataset's $S$ along `mm_L19` #verified[A5].
- L18:62 is active only on number comparisons (`larger_than` 51%, `smaller_than` 90%, the other 8 families 0%).
- L15:84, L16:69 and L18:15 are active on factual recall and translation, and at 0–1% on numbers.
- L19:34, the largest direct writer (about 6% of $S$ on its own), is active on most statements; its write is concentrated on negated statements (29–30% of $S$ for negated element symbols and translations, 7–9% for negated cities and animal classes, about 0 for affirmative cities and animal classes).
- For each dataset, the 480 values $S_c$ form a "write vector". Mean correlation of write vectors between datasets: arm A 0.50, untrained 0.26, code control 0.19. Two random halves of the *same* dataset correlate at 0.98–1.00, so differences between datasets are real, not noise. Arm A shares more writers across families than random components do, but far from all. On UTH: trained 0.26, untrained 0.20 (halves 0.90).

Caveat #assumed: a component specific to number statements may encode the topic (numbers vs entities) rather than truth. Only a per-family ablation (B) can tell.

== A6: why the UTH probe inverts <sec-a6>

#tbl(
  columns: (auto, auto, auto, auto, auto),
  [Test dataset], [CI probe, selected C = 1], [CI probe, C = 1e-4], [Binarised CIs (range over the 5 values of C)], [Correlation of per-component true−false differences: training tasks vs this task],
  [copa], [0.300], [0.77], [0.52–0.54], [\+0.24],
  [`story_cloze`], [0.395], [0.765], [0.54–0.58], [\+0.13],
  [sciq], [0.447], [0.66], [0.52–0.66], [−0.02],
)
#verified[A6]

The inversion is driven by a few heavily weighted components whose true−false difference flips sign between the training tasks and the test task: L15:10, L17:47 and L17:26 for copa and `story_cloze`, L15:21 for sciq. C selection chose the weakest regularisation because it fit the training tasks best. Nothing deeper than overfitting is needed to explain the loud failure #concluded.

== A7: what happens when one component is removed

For each component, a first-order estimate (attribution patching: the gradient of the layer-23 separation with respect to the component's mask, summed over all positions) predicts how much $S$ along `mm_L23` changes when the component is switched off. For the 5 largest estimates per run, the change was then *measured by real ablation*: mask 0 at all positions, Δ on, i.e. the weight becomes $W - U_c V_c^top$. Subsample: at most 200 samples per dataset.

#tbl(
  columns: (auto, auto, auto, auto, auto),
  [Component], [Estimate (fit)], [Actual (fit)], [Actual (test)], [Direct write $D$],
  table.cell(colspan: 5, align: left)[*tiu arm A* (fit: n 1,594, $S$ = 62.5; test: n 1,114, $S$ = 70.1)],
  [L17:33], [−17.1], [−19.7], [−19.0], [\+0.01],
  [L19:34], [−14.3], [−18.2], [−18.8], [\+1.27],
  [L16:69], [−5.5], [−10.5], [−10.8], [\+0.14],
  [L18:15], [−4.0], [−8.5], [n/c], [\+0.39],
  [L16:75], [−3.2], [n/c], [−10.0], [0.00],
  [Code control, largest], [], [±1.1–1.3], [±1.2–1.6], [],
  [Untrained, largest], [], [±0.6–1.5], [±0.6–1.7], [],
  table.cell(colspan: 5, align: left)[*UTH* (fit = training tasks: n 5,734, $S$ = 22.6; test = unseen tasks: n 1,600, $S$ = 13.0)],
  [L16:5], [−3.2], [−4.5], [−5.9], [\+0.03],
  [L15:10], [−1.6], [−5.8], [−5.8], [0.00],
  [L18:36], [−1.9], [−1.9], [−1.6], [\+0.05],
  [Untrained, largest], [], [±0.1–0.2], [±0.2–0.3], [],
)
#verified[real ablations in `report.txt`; "n/c" = not among that split's 5 checked components; one seed]

- On tiu, removing L17:33 or L19:34 alone cuts 27–31% of the layer-23 separation. On UTH, L16:5 or L15:10 alone cut about 45% of the separation on the *unseen* tasks (20–26% on the training tasks).
- The effects are mostly *indirect*: L17:33 writes almost nothing along the truth direction itself ($D$ = 0.01) and acts through later layers. Of its first-order effect (17.1 on fit), 8.8 comes from ablating it at the last token, the rest through attention from earlier positions.
- L15:10 is also the main driver of the UTH inversion (A6): it matters for truth on the unseen tasks, but its CI relates to truth differently there.
- The first-order estimate underestimates arm A's real effects by 13–55%, so only the real ablations count.

#caveat(title: "Why this is a lead, not a result")[
  These ablations have no controls yet. Three alternative explanations are open #assumed:
  - *Generic disruption.* L19:34 has CI > 0.01 on a median of 95% of each family's statements. Removing a component that matters for many inputs may shrink the residual stream or the computation generally, and $S$ is measured in raw units, not relative to the class spread. A non-truth separation (family or dataset identity) under the same ablation would show this.
  - *Weak reference.* The untrained components are random rank-1 pieces at their initial size, so removing one may simply be a much smaller perturbation than removing a trained component. The fair reference is random *trained* components of the same decomposition with matched activity and matched overall effect (e.g. the same output KL change when ablated). The code control is a better reference than the untrained run, and its largest effects are also more than 10× smaller, but it was trained on different inputs.
  - *Using vs computing truth.* A component downstream of truth (for example, part of the model's reaction to a false statement, see G) would also reduce the separation at layer 23 when removed. Ablation shows involvement, not that the component computes truth.
]

== G: does the next-token output depend on truth?

This matters because tPD only sees next-token predictions (@sec-background). Paired KL between the next-token distributions after the true and after the false version of the same item, against the KL between two *different* true items of the same dataset (content baseline) #verified[G]:

#tbl(
  columns: (auto, auto, auto),
  [Dataset (test split)], [Paired KL, true vs false], [Baseline, two true items],
  [tiu cities], [3.89], [1.39],
  [tiu `larger_than`], [2.39], [0.40],
  [tiu `animal_class`], [1.12], [0.62],
  [tiu `neg_cities`], [0.71], [1.09],
  [UTH copa], [0.38], [0.30],
  [UTH `story_cloze`], [0.60], [0.44],
  [UTH sciq], [1.97], [3.92],
)

On tiu the shifted tokens are visibly reactions to truth: after a false city statement the model continues with " Is" at probability 0.58 (after the true version 0.005); after a false larger-than statement " (" at 0.95 (true 0.39). A probe on the output log-probs reads tiu truth at 0.996. Negated statements show a weaker effect. In the UTH "Answer: X." format truth changes the output less than content does (cross-task output probe 0.747).

#finding(title: "Reading")[
  For affirmative simple statements the model's output *does* depend on truth (paired KL well above the content baseline for cities, larger-than and animal classes), so a decomposition trained on the next-token loss gets real truth pressure there #concluded. For negated statements it does not (`neg_cities` 0.71 vs a baseline of 1.09). In the UTH format the pressure is weak: the paired KL is near or below the content baseline.
]

= Where this leaves us <sec-state>

#tbl(
  columns: (1fr, auto),
  align: (left, left),
  [Claim], [Status],
  [CI probes read truth because any CI network reads truth-laden activations (untrained 0.998)], [#verified[tiu probes]],
  [Training on truth adds truth information to the CIs beyond a random readout], [not shown (+1–2 points, within run spread)],
  [Training on an unrelated target removes truth from the CIs], [#verified[code control, one seed]],
  [Trained UTH CIs carry truth within tasks but the readout does not transfer across tasks], [#verified[A3, A4]],
  [Most direct truth writing in layers 15–19 is in Δ, not in components], [#verified[A2, both datasets]],
  [Truth writers differ by kind of statement (numbers vs facts vs negation)], [#verified[A5] (topic confound open)],
  [Single trained components causally carry a large share of the truth separation], [lead; controls missing],
  [The next-token output depends on truth for tiu statements], [#verified[G]],
  [tPD recovers truth-specific mechanisms], [no evidence for, none against],
)

Overall: *no evidence that tPD recovers truth mechanisms, and none that it cannot.* The CI-probe route is exhausted: on in-distribution data everything is at the ceiling, and on cross-task data the trained CIs are worse than random ones. The only result that points towards truth-specific components is causal (A7), and it is uncontrolled.

= Caveats that apply throughout

- *One seed per run.* No decomposition has been repeated with a different seed; consistency of components across seeds (one of tPD's claimed properties) is untested here.
- *Untuned architecture.* Layers 15–19, C = 96 per layer and the loss coefficients were inherited, not tuned. A2 suggests truth along the layer-23 direction is mostly written by layers 20–23, which are not decomposed.
- *Poor fit on UTH.* 15% of the components' effect is unreconstructed; conclusions about UTH concern this decomposition.
- *Next-token loss only.* A truth computation that changes no prediction in the training text is invisible to tPD. On tiu, G shows the output does depend on truth; on UTH only weakly.
- *Last-token readout.* All analyses read the last token. Truth computed elsewhere and not moved to the last token would be missed.
- *Subsamples.* A7 uses at most 200 samples per dataset; a few G cells rest on very few pairs.

= Options for next steps <sec-next>

Costs assume a rented GPU (an RTX 4090 with 48 GB suffices for analysis; training needs an H100 or A100). My recommendations are marked; none of them has been agreed.

#tbl(
  columns: (2.6cm, 1.5fr, 2.1cm, 1fr),
  breakable: true,
  align: (left, left, left, left),
  [Option], [What it is], [Cost], [What it would tell us],
  [*B. Controlled ablations* (recommended first)], [Ablate the candidates (tiu: L17:33, L19:34, L16:69, L18:62, L18:15 and per-family writers; UTH: L16:5, L15:10, L18:36). Controls: about 20 random components of the same decomposition per candidate, matched on how often they are active (and ideally on the output KL change their ablation causes); the same ablation's effect on a non-truth separation (family / dataset identity); $S$ in units of the class spread; a double dissociation by family (number writers should hurt number statements but not factual ones, and factual writers the reverse). A third subcommand of `truth_writers.py`], [Coding a few hours; minutes of GPU], [Whether the one positive lead survives. If it does, tPD has found components with a truth-specific causal role, the first such evidence. If not, the project has no positive evidence left],
  [*Second seed of arm A* (recommended if B holds)], [Same config, seed 1; rerun A5/A7 and match components across seeds], [About 40 min H100 + analysis], [Whether high-effect truth components recur, or are an accident of one run],
  [Characterise the candidates (if B holds)], [What drives L17:33 (its input direction $V_c$, which tokens and earlier features activate it); which later layers carry its indirect effect (path patching through layers 20–23)], [Analysis only], [Whether the component computes truth or reacts to it],
  [New decomposition, different loss], [(a) loss on the last k tokens only (the never-run last-token arm; G says tiu gives truth pressure there); (b) append "Is this statement true? Answer:" and train on the answer position; (c) reconstruct the decomposed layers' hidden outputs at the last tokens instead of the next-token output (the code has such a loss, `StochasticHiddenActsReconLoss`, but it probably ignores the loss-position mask, so it needs a check or a small change)], [Config / code + H100 runs (40 min each)], [(a) focuses capacity on the statement end; (b) decomposes the model's truth judgement; (c) is the only option that sees truth computed but not used for the output],
  [New decomposition, other layers], [Include layers 20–23 (where A2 places the writing of the layer-23 direction), or more matrices at once. A 150-step memory trial on the H100 measures how many fit (bf16 training peaks at about 27 of 80 GB today)], [Trial minutes; run 40+ min], [Whether more truth writing moves out of Δ into components],
  [Better-fitting UTH decomposition], [Larger C, more steps], [H100 hours], [Whether cross-task failure is a capacity artefact; lower priority, because the well-fitting tiu decomposition gives cleaner tests of the same questions],
  [Not recommended], [More CI-probe accuracy comparisons; the padded arm (it only reproduces the original setup, whose result the untrained baseline already explains); a "form-matched" control decomposition trained on sentences that mention the same entities without making a claim ("Anna wrote a postcard about Krasnodar and Russia."; declined by Julian when the tiu series was closed)], [–], [Little: CI probes are at the ceiling or below random],
)

*My view on ordering* #assumed: run B before any new training. It is the cheapest experiment with the most decision value, because it decides whether the existing decompositions contain anything worth characterising. If B holds, a second seed and the characterisation come next, and the choice of a new loss can build on what those components do. If B fails, the existing decompositions have nothing truth-specific to offer, and the remaining question is whether a different objective ((b) or (c) above) gives tPD a reason to decompose truth at all. On tiu the last-k option (a) is viable, because G found real truth pressure in the next-token output after affirmative statements.

#pagebreak()

= Appendix

== Glossary <sec-glossary>

#tbl(
  columns: (auto, 1fr),
  align: (left, left),
  [Term], [Meaning],
  [tPD], [Targeted parameter decomposition (Vigouroux & Sharkey 2026)],
  [Component, L17:33], [A rank-1 piece $U_c V_c^top$ of a decomposed weight matrix; L17:33 = component 33 of layer 17's `down_proj`],
  [CI], [Causal importance: the per-token value in $[0,1]$ the CI network predicts for each component. Active = CI > 0.01],
  [Δ], [The full-rank remainder $W - sum_c U_c V_c^top$ of each decomposed matrix],
  [$a_c$], [Inner activation $V_c dot h$ of a component, $h$ = the MLP hidden vector feeding `down_proj`],
  [tiu], [Truth-is-Universal statements (Bürger et al. 2024)],
  [UTH], [Universal Truthfulness Hyperplane datasets (Liu et al. 2024)],
  [Arm A], [The tiu decomposition trained on all real tokens of each statement],
  [Untrained baseline], [A decomposition saved at step 0: random components and random CI network],
  [Matched sparsity / top-k], [Each sample keeps only its k largest CI values before probing],
  [Cross-task], [Probe fit on some task categories, scored on unseen ones],
  [`mm_L19`, `mm_L23`], [Mass-mean truth directions of the residual stream after layers 19 and 23],
  [$S$], [True/false separation: mean projection of true minus false samples onto a unit truth direction],
  [Fit / tune / test], [Training split / held-out split of the training data (choosing C) / evaluation split],
)

== Runs, scripts and result files <sec-files>

#tbl(
  columns: (auto, 1fr),
  align: (left, left),
  [Item], [Location],
  [Configs], [`spd/experiments/lm/honesty_targeted_decomposition/config_truth_all_tokens.yaml` (arm A), `config_code_control.yaml`, `config_uth_all_tokens.yaml`, `config_truth_untrained.yaml`; never run: `config_truth_last_token.yaml`, `config_truth_padded.yaml`],
  [Probe scripts], [Same folder: `probe_ci.py` (tiu, Experiments 1–2), `probe_uth.py` (UTH, Experiment 3; subcommands `run`, `probe`, `sparsity`), `truth_writers.py` (Experiment 4; `extract` on GPU, `analyze` on CPU; not yet committed)],
  [Data], [`data/tiu/*/v1`, `data/uth/*/v1`, `data/pile_code/github_lines/v1` (generators `build_tiu.py`, `build_uth.py`, `build_pile_code.py`)],
  [Experiment 1 results], [`~/spd_out/26-10-05_no_truth_baseline/probe/results.json`, `followup.out` (bf16); `~/spd_out/26-10-03_test_accuracy_analysis/results.json` (fp32 runs)],
  [Experiment 2 results], [Same folder as Experiment 1 (bf16); projection check `projection_report.json`],
  [Experiment 3 results], [`~/spd_out/26-10-05_uth_experiments/stage1/results.json`, `sparsity.log`],
  [Experiment 4 results], [`~/spd_out/26-10-06_truth_writers/{tiu,uth}/report.txt`, `results.json`],
  [Detailed records], [`convos/julian/`: topics of 26-10-03 (test accuracy), 26-10-05 (no-truth baseline, UTH experiments), 26-10-06 (truth-writing components)],
)

All result folders under `~/spd_out/` are local to the laptop and not in git.
