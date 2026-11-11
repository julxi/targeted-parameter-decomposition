#import "template.typ": *
#import "26-11-04_tpd_truth_paper_draft_diagram.typ": target_panel, nontarget_panel, restriction_panels

// Pre-version of the paper: a very detailed description, written section by section on Julian's
// instructions. Topic: convos/julian/26-11-04_paper_draft_LOG.md. Not the paper's prose yet.

#show: report.with(
  title: "Trying to decompose a truth mechanism with tPD",
  subtitle: "Pre-version of the paper (working title): a detailed description, written section by section",
  date: "2026-11-04",
  author: "Julian Bitterlich; draft text by Claude Opus 5.5",
  summary: [
    - *Status.* Two sections exist: the three decomposition methods (@methods) and how they are evaluated (@evaluation: the behavioural tests, symmetric in $P$ and $R$, with their baselines and axes, a test of whether $P$ computes the truth value, inspection tests, a small causal model of the truth computation, a definition of when a part of the weights captures the truth value, what the tests can and cannot detect and how they are validated, the evaluation setup, the status of each test, and a glossary of the analyses run). The results section (@results) is a skeleton of placeholders: the planned tables and figures, with marks for which data exist. Data, the results' numbers and introduction follow in later steps.
    - *What this document is.* A pre-version of the paper: it carries every detail a reader needs to follow what was done, without the final paper's selection and prose.
  ],
)

= The three decomposition methods <methods>

We try to use targeted parameter decomposition (tPD; Vigouroux and Sharkey 2026) to get a partial machanisitic decomposition of a truth-value mechanism in an LLM.

The basic setup of tPD is shown in @fig-setup

#figure(
  block(width: 100%)[
    #set align(left)
    #text(size: 8pt, weight: "bold")[Target data]
    #target_panel
    #v(0.6em)
    #text(size: 8pt, weight: "bold")[Non-target data]
    #nontarget_panel
  ],
  caption: [Blue: layers where some weight matrices are partially decomposed using the tPD method $W' = sum_c m_c U_c V_c^top + m_Delta Delta$; grey: original weights; dashed light blue: original weights which carry the decompositions indirect effect; orange: the outputs compared with the original model by KL; dashed orange line indicates potential extra losses on the components writeouts (@fig-restrictions).],
) <fig-setup>

Plain tPD constrains only outputs: the verdict, and the next-token predictions on the Pile. Recon and ortho each add a constraint on what the components write at the statement tokens (@fig-restrictions).

#figure(
  restriction_panels,
  caption: [What the extra terms ask of one decomposed matrix at one statement token. $y = W x$ is the original output, $C$ the write of all components at full strength, $D = Delta x$ the write of the remainder.],
) <fig-restrictions>

== Shared training setting

*Model and prompt.* The model is Qwen2.5-7B-Instruct, frozen. The target data consists of a statement followed by a fixed question,

#align(center)[`<statement>\nThe amount of truth in this statement is␣`]

and we study the mechanism behind the model's prediction of the next token. That (greedy) token is almost always the digit `1` (true) or `0` (false) (see digit mass in @tab-verdict-accuracy). We call the whole input the *verdict prompt* and the digit the *verdict*. Statement and question are tokenized as one string, as the model would read the text, and the *statement tokens* are all tokens up to and including the one that contains the statement's final period.#footnote[Usually this token is `.\n` as the period merges with the line break that follows it. A character just before the period can merge as well. In the two translation families every statement ends in a quoted word, so the token is `'.\n`, and a few out-of-distribution `facts` statements end in `).\n`, `.'\n` or `."\n`.] We score a statement by the *logit difference* $s = log P(1) - log P(0)$ (equal to the difference of the two digits' logits) at the answer position; the model calls it true if $s > 0$.

The model does this task well (@tab-verdict-accuracy): 97.9% of the test statements are judged correctly, and the two digits together get 98.4% of the probability mass on average, so the model answers in the format.

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    [family (test split)], [$n$], [accuracy], [acc. true], [acc. false], [AUC], [digit mass],
    [`animal_class`], [37], [1.000], [1.000], [1.000], [1.000], [0.976],
    [`cities`], [332], [1.000], [1.000], [1.000], [1.000], [0.988],
    [`element_symb`], [56], [0.875], [0.750], [1.000], [1.000], [0.949],
    [`sp_en_trans`], [64], [0.984], [0.969], [1.000], [1.000], [0.988],
    [`larger_than`], [408], [0.995], [0.990], [1.000], [0.997], [0.999],
    [`smaller_than`], [408], [0.980], [1.000], [0.961], [0.998], [0.996],
    [`neg_animal_class`], [37], [1.000], [1.000], [1.000], [1.000], [0.984],
    [`neg_cities`], [332], [0.961], [1.000], [0.922], [1.000], [0.964],
    [`neg_element_symb`], [56], [0.929], [0.893], [0.964], [0.976], [0.931],
    [`neg_sp_en_trans`], [64], [0.953], [0.969], [0.938], [0.997], [0.989],
    [*all*], [1,794], [0.979], [], [], [], [0.984],
  ),
  caption: [The model on the verdict prompt. Accuracy: share with the right sign of $s$; AUC: of $s$ for true vs false within the family; digit mass: $P(1) + P(0)$, averaged. Each family holds a true and a false set of equal size. Run B2 (2026-10-26).],
) <tab-verdict-accuracy>

*What is decomposed.* The `down_proj` matrix of each of the layers 11 to 17. `down_proj` is the MLP's output projection, so its output is everything the MLP writes into the residual stream. Each of these seven matrices is written as
$ W = sum_(c=1)^(96) U_c V_c^top + Delta, $
96 rank-1 *components* per layer (672 in all) plus a full-rank remainder $Delta$. A single *causal-importance (CI)* network, shared by the seven layers, reads the inputs of all seven matrices at a token and gives every component a CI value in $[0, 1]$ at that token. $Delta$ has no CI value.

*Why these layers.* We replaced the outputs of MLP layers at all statement tokens by their mean over the training statements and measured what is left of the verdict (@tab-layer-scan). The measure is the *separation share*: the gap between the mean logit difference $s$ of true and of false statements, as a fraction of the original gap (8.66). The question tokens are never ablated.
- No single MLP layer is necessary: from layer 5 to 24, removing one layer leaves at least 0.77 of the separation (the minimum is at layer 17). The verdict is built up across many layers.
- The block 11–17 is sufficient: removing these seven MLPs leaves 0.05 of the separation (accuracy 0.538, chance level) while the model still answers with a digit (digit mass 0.977). So the block carries the judgement, not the answer format. It is the smallest block in the scan with this property; the blocks 11–14 and 15–19 each leave about a third.
- The block 6–10 also matters (0.18 left). We left it out: it would make the decomposition 12 layers long, and its role may be processing the statement's content (recognising the entities, recalling the fact) rather than judging it. This was not tested #assumed.
- All statement tokens are needed: removing the block 11–17 at the statement's last token `.\n` only leaves 0.82 of the separation (accuracy 0.924; layer scan of 2026-10-16, same test split, separate run), because the answer also reads the other statement tokens.

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto),
    [mean-ablated at all statement tokens], [accuracy], [separation share], [digit mass], [KL (nats)],
    [nothing (original)], [0.977], [1], [0.984], [0],
    [single MLP layers 5–24], [0.881–0.978], [0.77–1.02], [0.979–0.988], [0.00–0.24],
    [MLP 6–10], [0.707], [0.18], [0.952], [0.44],
    [MLP 11–14], [0.881], [0.38], [0.979], [0.23],
    [MLP 15–19], [0.663], [0.33], [0.983], [0.49],
    [*MLP 11–17*], [*0.538*], [*0.05*], [*0.977*], [0.68],
    [attention 11–14], [0.742], [0.44], [0.984], [0.63],
    [attention 11–17], [0.655], [0.21], [0.977], [0.66],
    [MLP + attention 11–17], [0.518], [0.02], [0.983], [0.66],
  ),
  caption: [Causal scan of the original model on the verdict prompt (run B1, 2026-10-26, test split, 1,794 statements). KL: of the answer distribution against the original, full vocabulary. Mean ablation of the single layers 3, 4, 25 and 26 is left out: there a few tokens have outputs of very large norm, so the mean itself is an unusual write. The original row's accuracy is 0.977 here and 0.979 in @tab-verdict-accuracy; the two runs differ in the third digit for a reason we did not trace.],
) <tab-layer-scan>

*Why only `down_proj`.* Mean-ablating an MLP's output is the same as replacing its `down_proj` output, so the scan above is a scan of `down_proj` outputs. Decomposing only this matrix has two limits. First, the MLP's input side (`gate_proj`, `up_proj`) is not decomposed, so the components show which hidden directions the MLP reads out and where it writes them, not how those hidden directions are computed. Second, attention matters too: mean-ablating the attention outputs of layers 11–14 leaves 0.44 of the separation, about as little as the MLPs of the same layers (0.38). We still do not decompose attention, because removing MLPs 11–17 alone already takes the verdict to chance: every route to the verdict passes through these MLPs, with attention in series. (The restriction to `down_proj` was originally made for memory; the tPD paper decomposes the attention and MLP input matrices as well.)

*Positions.* The components replace the weights only at the *statement tokens*: every token of the statement, the first included, up to and including `.\n`. The question tokens always run on the original weights. So whatever the decomposition changes, it changes inside the statement, and it reaches the verdict only through the attention of the later question tokens to the statement.

*Data streams.* Each training step uses one batch of *target* data and one batch of *non-target* data; the gradients of both are summed before the optimizer step (as in the tPD paper).
- Target data: 64 statements per batch from the training split of the Truth-is-Universal statement sets (the 10 families of @tab-verdict-accuracy, each in a true and a false version; at most 130 statements per set, 1,962 in all). Details of the data in a later section.
- Non-target data: 16 windows of 64 tokens from the Pile (uncopyrighted version), every position used.



*Notation.* At a decomposed position and layer, with $x$ the matrix's input there, the original output is $y = W x$. It splits exactly into the write of all components at full strength, $C = sum_c U_c V_c^top x$, and the write of the remainder, $D = Delta x = y - C$.

*Masks.* In a masked forward pass each component's contribution at a position is multiplied by a mask value $m_c$ and $Delta$'s by $m_Delta$. A component's mask never goes below its CI: $m_c in [upright("CI")_c, 1]$, so a component with CI 1 is always fully on and a component with CI 0 may be anywhere between off and on. On the statements $m_Delta in [0, 1]$; on the Pile $m_Delta = 1$.

== What the losses ask for

tPD looks for few components that carry the target behaviour on their own. On the statements, the verdict must come out right when the components are masked anywhere between their CI and fully on and $Delta$ is partly or fully removed, while as few components as possible have a non-zero CI. So the components with high CI must carry everything the verdict needs, and $Delta$ must not be needed for it.

The Pile stream decides what goes into $Delta$. There $Delta$ stays fully on, and every non-zero CI is penalised. A mechanism the model uses on the Pile but not on the statements therefore costs CI on the Pile if it sits in a component and nothing if it sits in $Delta$, so it ends up in $Delta$. Components that are active on the statements are shaped to disturb the Pile as little as possible (as argued in the tPD paper).



== Method 1: plain tPD (the terms shared by all three)

These terms make up *plain tPD*. The two other methods add one term each.

+ *Verdict loss under random masks* (target data). One masked forward pass per batch. At every statement token, every component gets $m_c = upright("CI")_c + (1 - upright("CI")_c) u$ with $u$ uniform in $[0, 1]$, drawn independently per component and position, and $Delta$ gets $m_Delta$ uniform in $[0, 1]$, drawn per layer and position. In addition, at each position a random number $k$ of the seven layers (uniform in 1 to 7) uses the masked decomposition and the other layers use their original weights. This routing is the SPD codebase's standard stochastic reconstruction loss, which we inherited unchanged; it generalises the SPD paper's two losses ($k = 1$ is its layerwise loss, which masks one layer at a time, and $k = 7$ its full loss). Its purpose is not clear to us, and it looks like an engineering choice; we would have preferred results without it. We kept it because all experiments had already been run with it.

  The loss is the KL divergence between the original model's next-token distribution at the last question token (the verdict distribution, over the full vocabulary) and the masked model's. *The loss signal comes only from the verdict*: the model's next-token predictions at the statement tokens (what it expects as the statement's next word) are not compared, and neither are the matrices' outputs. Implementation detail: the verdict KL is counted once per decomposed position, so a statement weighs in proportion to its number of tokens.
+ *Verdict loss under adversarial masks* (target data, last 20% of training only). The same verdict KL, but with mask values chosen to maximise it (components within $[upright("CI")_c, 1]$, $Delta$ within $[0, 1]$, all seven layers masked), using persistent adversarial masks: they are kept across training steps and moved by a few gradient-ascent steps in each one.
+ *Importance minimality* (target data). A penalty on the CI values at the statement tokens (an $L_p$ norm, $p$ annealed from 2 to 1), which pushes towards few components with non-zero CI per token.
+ *Non-target loss* (Pile). The model's next-token predictions at every position of the Pile windows, compared by KL with the original model's, under masks with $Delta$ fixed at 1, the components' masks random in $[upright("CI")_c, 1]$ and the same random layer subset as in loss 1. Plus importance minimality on the Pile CI values, at twice the target coefficient, which moves mechanisms that the statements do not use into $Delta$.
+ *Weight decay* of 0.1 on the component vectors, scaled by one minus the component's largest CI in the batch, so that components that never become important shrink towards zero.
// that is something that we should keep in mind and check in the actual decompositions

== Method 2: recon (plain tPD + per-layer reconstruction)

Recon is meant to prevent a failure that plain tPD can fall into, in which the components re-implement the verdict instead of decomposing what the MLPs compute (we come back to this in the results). It adds one term on the target data, at the statement tokens. In a masked forward pass with masks drawn as in the verdict loss (components in $[upright("CI")_c, 1]$, $Delta$ uniform in $[0, 1]$, but all seven layers masked), each decomposed matrix's output at each statement token must match the original model's output of that matrix at that token:
$ cal(L)_"recon" = "mean over statement tokens and layers of" |hat(y) - y|^2 \/ |y|^2, $
with $hat(y)$ the matrix's output in the masked pass and $y$ its output in the original model. The division by $|y|^2$ per position keeps positions with very large outputs from dominating. Since $hat(y)$ is computed in the masked pass, the matrix's input already carries the changes of the masked layers before it.

*Where the signal comes from.* From the matrices' own outputs, not from the verdict. Because $Delta$'s mask is random, the components with high CI have to reproduce the output themselves. The term therefore asks for a decomposition of the whole MLP output at the statement tokens, the verdict-relevant part and everything else alike. Expected consequence: many active components per token, and a decomposition that is faithful but not targeted at the verdict. The term is not applied to the Pile.

== Method 3: ortho (plain tPD + global orthogonality)

Ortho addresses the same failure as recon without giving up the targeting. It rests on a different assumption: that the part of each MLP output that matters for the verdict is roughly orthogonal to everything else these layers write #assumed. If so, the output splits into two orthogonal parts, one carried by the components and one by $Delta$, and asking for such a split lets the components carry only the verdict-relevant part.

The term acts on the target data, at the statement tokens. Per statement token and layer, with $C$ and $D$ as in the notation above (all components at full strength, not weighted by CI):
$ cal(L)_"ortho" = "mean over statement tokens and layers of" (chevron.l C, D chevron.r \/ |y|^2)^2. $
$C$ and $D$ are computed from the matrix's input in the original model's forward pass, one matrix at a time, so the term needs no extra forward pass and masks no layer.

*What it asks.* $chevron.l C, D chevron.r = 0$ with $C + D = y$ means that $C$ lies on the sphere whose diameter is $y$: $C$ is the orthogonal projection of $y$ onto $C$'s own direction. Then $|C| <= |y|$ and $|D| <= |y|$, so the components and $Delta$ cannot write large vectors that cancel each other. The term does not say *which* part of $y$ the components carry: $C = 0$ and $C = y$ both satisfy it, and only the verdict losses push components on. So unlike reconstruction, it leaves the method targeted.

*Design choices.* The condition is on the sum of all components, not one per component, so components may partly cancel each other as long as their sum is a projection of $y$. Because it uses all components at full strength, the term does not depend on the CI values and trains only the component vectors $U$, $V$. It is not applied to the Pile.

== Settings <settings>

The three methods share every setting except the extra term and, for ortho, the learning rate. Values were tuned on the validation split (procedure in a later section): learning rate and importance-minimality coefficient for plain tPD, then only the extra term's coefficient for recon and ortho, plus ortho's learning rate, because at plain tPD's learning rate ortho's components recovered much less of the verdict in tuning.

#tbl(
  columns: (auto, 1fr, 1fr, 1fr),
  [], [plain], [recon], [ortho],
  [extra term], [none], [per-layer reconstruction, coefficient 10], [global orthogonality, coefficient 10],
  [learning rate (cosine decay to 10%)], [$5 dot 10^(-4)$], [$5 dot 10^(-4)$], [$2 dot 10^(-3)$],
  [importance minimality (target; Pile ×2)], table.cell(colspan: 3)[$10^(-3)$, with a warm-up to 4× in the first 20% of training and a decay back],
  [verdict loss, random masks], table.cell(colspan: 3)[coefficient 1; per position a random $k in {1, ..., 7}$ of the layers masked],
  [verdict loss, adversarial masks], table.cell(colspan: 3)[weight 0.5 in the total loss (the random-mask version has weight 1); switched on only for the last 20% of training (steps 4,000–5,000); all 7 layers masked],
  [Pile loss], table.cell(colspan: 3)[coefficient 1; $m_Delta = 1$; per position a random $k in {1, ..., 7}$ of the layers masked],
  [layers masked by the extra term], [–], [all 7], [none (each matrix on its original input)],
  [components per layer], table.cell(colspan: 3)[96 (7 layers, 672 in all)],
  [steps; batch], table.cell(colspan: 3)[5,000; 64 statements + 16 Pile windows],
  [seeds], [1, 2], [1, 2], [1, 2, 3],
)

= How the decompositions are evaluated <evaluation>

== The question <capture>

=== Intuition

The basic question is "Has a decomposition captured the model's internal truth computation?" Making this precise is very hard and we'll have to work with many simplifications and most likely the underlying models of our reasoning is not how the llm works internally, but it's the best we got.

One basic assumption is that the model computes truth values of statements at all: in its forward pass it builds a quantity that says whether it takes the statement to be true (its belief), and it uses this belief in coherent ways. This is a particular instance of the believe that llm's generally extracts meaningful features from the input and produces its output from these features only. One of the features, $T$, plays the role of the truth value.#footnote[A model may compute several truth values, e.g. for several statements in one input; we restrict to inputs with one statement. This is already a major simplification of the task of uncovering the model's truth mechanism.]

=== A simple model of the truth computation <ca-narrow>

A very simple model of the truth computation has two variables

- $X$: the tokens of the statement.
- $T$: the internal truth value attributed to the statement.

We don't know what form $T$ has or where it lives (at which token position or layer). So we use readouts to read $T$ off the model's own outputs. We don't use probes or truth directions for the definition, as to not bake further assumptions in our setup. Some assumptions that we do make for the design of the following test are
+ *(A1) One truth value.* On a statement there is one $T$
+ *(A2) Truth verdicts mainly depend on $T$.* Every verdict prompt urges the model to surface it's truth value.
+ *(A3) Other-quality readouts don't depend on $T$.* The other readouts we choose like hosety, or humor do not depend on $T$ or are only weakly influenced.

*Why two kinds of readout.* The $Y_j$ make "uses the belief coherently" testable: if there is one $T$, one edit at the statement moves every $Y_j$ together. Because attention is causal, the activations at the statement tokens do not depend on the question that follows, so the same edit can be tested under every question. The $Q_k$ make specificity testable: an edit meant to change $T$ must leave them alone. With truth readouts only, every prediction of $cal(H)$ is about $T$, and a split can pass every test while disturbing everything the truth readouts do not read (@ca-failures).

*Which $T$.* Two readings of $T$ make different predictions for one readout, recall. If $T$ is the *judgement* "the statement agrees with what I know", recall reads $Z$, not $T$, and an edit of $T$ leaves recall unchanged. If $T$ is the *stored fact* itself, recall reads it, and an edit of $T$ changes recall too. Steering the original model along a truth direction at the statement (B3, @glossary) flipped the verdict and left recall intact, which fits the first reading. Both readings are legitimate high-level models; the tests below report recall separately, so a decomposition can be judged against each.


=== The tests <tests>

Each site is decomposed into a component part $P$ and the rest $R$

$ W = P + R quad p = P x, r = R x quad y = p + r $

We want to see if the $P$-parts capture the truth value computation. Ideally we would understand the mechanisms behind $P$ (or could conlude that $P$ does not represent a coherent mechanism) but understanding them by hand is tricky. So we rather analyse if how $P$ and $R$ influence the $T$ value. Still it's is tricky to measure "influence" for that we look at a variety of tests.

The tests come in mirror pairs. What we expect of $P$ on the truth value, we expect of $R$ on everything else, and vice versa; and most interventions on $P$ have a counterpart on $R$ (@tab-tests).

 We use three kinds of intervention on the $P$-writes (or the $R$-writes) at the statement tokens:
- *removing or averaging* them should change the value;
- *swapping* them in from another statement should transcribe the value;
- *editing* them should move the value in a predictable way.
We can't measure the truth value directly, so we have to resort to the model's answers to truth questions (*truth readouts*), best in different formats:
- the verdict prompt from training;
- other 0/1 prompts;
- Yes/No prompts;
- another language.
The other features are read off answers that should not depend on the truth value (*other-quality readouts*): digit questions about other properties of the statement (humour, emotion, surprise, word count), recall of the fact about the statement's subject (but see *Which $T$* in @ca-narrow), and the next-token predictions on the statement itself and on non-target text (the Pile).

All interventions act on the decomposed matrices at the statement tokens; the question tokens run on the original weights. Because attention is causal, the same edited statement can therefore be read under every question. All these tests look only at the model's outputs, so we call them *behavioural tests*. Tests that look inside the decomposition follow in @tests-inspect.

==== The behavioural tests <tests-behav>

The next tests use *twins*: two statements of the same sentence form and domain that disagree in their truth value. Usually they differ only in the last word, like "The city of Lu'an is in China." and "The city of Lu'an is in Turkey." (details in @eval-setup).

- *Influence of $P$ on the truth verdict.*
  - *Only $P$* on the statement (the same run as "$R$ removed"): we measure the verdict separation. The goal is no change, as $P$ contains all relevant computation for the truth value.
  - *$P$ removed* on the statement: under the assumption that the verdict largely depends on $T$, we expect removing $P$ to affect the verdict about as much as removing the entire sites. The effect might be a bit less pronounced than the full ablation, because the LLM could have some fallback mechanism. On the other hand, this is quite a counterfactual scenario, so these are just heuristic expectactions.
  - *$P$ averaged over twins:* each $P$-write is replaced by the average of the two twins' $P$-writes. This removes what varies with truth between the twins and keeps what they share, so it is less out of distribution than removal. Expected: the verdict no longer separates the true twin from the false one (separation share and within-twin AUC at chance).
  - *$P$-swap:* on twins $x$ and $x'$, we patch the $P$-writes of $x'$ into the run on $x$. This should be a bit less out of distribution than removal, but the reasoning is similar. We measure the *interchange intervention accuracy* (IIA), the share of swaps after which the verdict is the verdict on $x'$ (0 without a swap, since the verdicts differ), and the graded *swap fraction*
    $ nu = (s_"swap" - s(x)) / (s(x') - s(x)), $
    1 if the score $s$ moved all the way to $x'$, 0 if not at all. Expected: as for the site swap.
  - *Fixed shift:* add one fixed vector to the $P$-write at each site and position: the mean difference of the $P$-writes between the false and the true twin in the training split (added for statements the model calls true, subtracted for those it calls false), scaled by one factor chosen on validation. Expected: the verdict flips.
  - *Hand edits of $P$.* This is a tricky one, a shot in the dark as long as we don't understand the mechanisms that $P$ represents, so these edits have to be designed by hand. Edits of single components, reversed (mask $-1$) or amplified (mask 2), have been run (M-core, M-twin, M-wide in @glossary).
- *Influence of $P$ on other qualities.* Here $P$ should be irrelevant.
  - *$P$ removed, $P$ replaced by its median over the statements* (less out of distribution than removal) *and $P$ averaged over twins*, read by the other-quality questions and by the next-token predictions on the statement; on the Pile, $P$ removed. Expected: little change, at least on predictions that don't need the truth mechanism. Under the twin average, only the qualities the twins share are expected to stay.
  - *$P$-swap and fixed shift,* read by the other-quality questions whose answer the twins share (the subject's fact, the word count). Expected: no more change than under the site swap or the site shift. This is the behavioural test that a re-implementation fails (@ca-failures). Next-token predictions on the statement can't be used here: before the first differing token the twins' activations are identical, so a swap changes nothing there, and after it the twins' predictions differ anyway.
- *Influence of $R$ on the truth verdict.* Ideally $R$ should not affect it.
  - *$R$ removed* is the run "only $P$" above.
  - *$R$ averaged over twins:* expected, the verdict separates true from false as in the original model.
  - *$R$-swap:* the $R$-writes of $x'$ patched into the run on $x$. Expected: the verdict stays with $x$ (IIA and $nu$ near 0).
- *Influence of $R$ on other qualities.* $R$ should carry everything else.
  - *Only $P$* (= $R$ removed), read by the other-quality questions and by the next-token predictions on the statement and on the Pile. Expected: about as much change as removing the whole sites. This test is weak: $R$ contains the remainder $Delta$, i.e. most of the matrix, so removing it breaks almost everything whatever the split. It mainly shows that $P$ is not the whole matrix.

#figure(
  tbl(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: (left, left, left, left, left),
    [intervention], [$P$, truth readouts], [$P$, other qualities], [$R$, truth readouts], [$R$, other qualities],
    [removed], [verdict degrades as under site removal], [little change], [no change (run "only $P$")], [change as under site removal],
    [averaged (over twins; median)], [no separation between twins], [shared qualities unchanged], [separation kept], [–],
    [swapped], [verdict follows $x'$ as under the site swap], [shared qualities unchanged], [verdict stays with $x$], [–],
    [fixed shift], [verdict flips], [unchanged], [–], [–],
    [hand edits], [as designed], [as designed], [–], [–],
  ),
  caption: [The behavioural tests and their expected outcomes. Mirror cells: what $P$ should do on the truth readouts, $R$ should do on the other qualities, and vice versa. Each cell is read against the comparison runs of @tests-compare and run over the axes of @tests-grid. "–": not run.],
) <tab-tests>

==== How an effect is compared with its baselines <tests-compare>

Since we only decompose some matrices of some layers, an intervention on $P$ can't do more than the same intervention on the whole sites, and the size of an effect alone says little. Each measurement $m$ is therefore read against the same intervention, on the same statements, with the same readout and the same ablation type (zero or mean; the layer scan used means, "all off" in @eval-setup is zero), applied to:
- *nothing*, i.e. the original model: $m_0$;
- *the whole sites* (the *site version*; $P = W$): $m_"site"$. It is the ceiling for a test on $P$, and it checks the readout: if the site version doesn't move a readout, no part of the sites can, and the test can't judge the split at these layers (@ca-validate);
- *random component sets of the same size* within the decomposition: $m_"rand"$. A result that a random part of the same size also produces says nothing about the split;
- *the control bases* (the untrained decomposition and the data-aligned basis, @eval-setup), with the same rule for choosing $P$.
The *effect fraction*
$ phi = (m - m_0) / (m_"site" - m_0) $
is the share of the site intervention's effect that the intervention on the part produces. Ideally it is 1 for $P$ on the truth readouts, 0 for $R$ on the truth readouts and 0 for $P$ on the other qualities. Examples:
- removal, with the separation share as $m$: $phi = (1 - "sep"_P) \/ (1 - "sep"_"site")$.
- swap, with the score $s$ as $m$: $phi = nu_P \/ nu_"site"$;
- other-quality readouts: $m$ is the KL of the answer distribution from the original's, so $m_0 = 0$. Where the site effect is itself close to 0, as for qualities the twins share, $m$ and $m_"site"$ are compared directly instead of divided.
$phi$ is reported with its bootstrap interval, next to $phi_"rand"$, the control bases' $phi$ and the size of $P$.

==== The tests as a cross-product <tests-grid>

Most tests are a cross-product of several axes, and we run (most of) it:
- *the test:* the intervention (removed, averaged, swapped, fixed shift, hand edit) on $P$ or on $R$;
- *the quality:* the truth value, or one of the other qualities;
- *the verdict style:* for the truth value, the format of the question (training verdict prompt, other 0/1 prompt, Yes/No, another language);
- *the size of $P$:* the top $k$ components (ranked by the effect of removing each alone, T1 in @glossary), the active set, or all components; reported as the share of components and of parameters;
- *the statements:* in distribution (test split) or out of distribution (the OOD sets);
- *the decomposition:* method (plain, recon, ortho) and seed, plus the control bases;
- *the comparison runs* of @tests-compare.
Not every combination exists: swaps and twin averages need twins of equal token length, which so far are counted only for the test split (@eval-setup); the Pile has no truth readout; and "another language" is not defined yet (the question asked in another language, or statements in another language).

==== Inspection tests <tests-inspect>



The behavioural tests see only what reaches an output and is also limited by the contexts we provide. A split can differ from the model's mechanism in ways none of our behavioural test detect. For that we also look inside the decomposition to see how they act within the model. None of the following decides whether $P$ captures $T$; they explain behavioural results and provide further data to judge the decompositions.
- *Cancellation.* Per layer at the statement tokens, the norms $|C|$ and $|D|$ and $chevron.l C, D chevron.r \/ |y|^2$ (notation of @methods). Large opposing writes mean that the components and $Delta$ cancel each other, the signature of a re-implementation; this is what ortho penalises. For twins $x$ and $x'$, with $delta_P = p(x') - p(x)$ and $delta_R = r(x') - r(x)$, the cancellation ratio $kappa = (|delta_P|^2 + |delta_R|^2) \/ |delta_P + delta_R|^2$ is 1 for no cancellation and $>> 1$ for cancellation between the twins; then the $P$-swap is no longer a part of the site swap. Both show cancellation, not unfaithfulness: the model itself may contain cancelling parts.
- *Structure of the decomposition.* How many components are active per statement (the active-set curve), whether the CI values have a gap between "off" and "on" (valley check), and whether the seeds find the same components (seed comparison).
- *Alignment and decodability.* The cosine of each component's write with the truth directions, the share of the true–false gap along these directions carried by the active components, the inactive ones and $Delta$ (Dir), and per-component selectivity for truth within a family and its transfer to the OOD sets (K1, K2). These describe the representation. That truth can be read from a part doesn't mean the model uses it (K0 in @glossary, the reason probe accuracy is not evidence).

= Results <results>
// Results skeleton (26-11-10): placeholders only, no numbers yet. Red ⟨…⟩ text describes what goes
// in a slot; in the tables ● = the data exist (phase D or the G2 runs; still to be re-expressed as
// below), ○ = still to run, – = does not apply. The helpers below go once the numbers are in.

#let ph-color = rgb("#b03030")
#let ph(body) = text(fill: ph-color, size: 0.9em)[⟨#body⟩]
#let have = text(fill: rgb("#2e7d4f"))[●]
#let torun = text(fill: ph-color)[○]
#let phfig(height: 5cm, body) = block(
  width: 100%, height: height, inset: 10pt, radius: 3pt,
  stroke: (paint: ph-color, dash: "dashed", thickness: 0.6pt),
)[#set text(size: 8.5pt, fill: ph-color); #align(center + horizon)[*Placeholder figure.* #body]]
#let status(have-text, run-text) = note(title: "Data status")[*Have:* #have-text \ *To run:* #run-text]
#let bases = ("plain s1", "plain s2", "recon s1", "recon s2", "ortho s1", "ortho s2", "ortho s3", "untrained s1", "untrained s2", "data-aligned")
#let _cell(s) = if s == "h" { have } else if s == "r" { torun } else if s == "n" { text(fill: ph-color)[·] } else { [–] }
#let rows(names, pattern) = names.map(b => ([#b], ..pattern.map(_cell))).flatten()

All results are reported per decomposition, never pooled over seeds: plain and recon with seeds 1 and 2, ortho with seeds 1 to 3, next to the two untrained decompositions and the data-aligned basis (@eval-setup). Test split unless stated, 95% bootstrap intervals in every cell. Each effect is read against its site version as the effect fraction $phi$ (@tests-compare), next to $phi_"rand"$ for random component sets of the same size.

*Which $P$.* Unless stated, $P$ is the decomposition's active set (the smallest top-$m$ set by mean CI that reaches 95% of the components' recovery on validation; @glossary) and $R$ is everything else, $Delta$ included. The dependence on the size of $P$ is shown separately, along the top-$k$ ranking of T1 (@fig-res-size). #assumed (choice of the default $P$ open; the data-aligned basis has no CI values and therefore no active set: there $P$ is the top-$k$ by T1 at the median active-set size of the decompositions.)

After an overview of all outcomes (@res-overview), the section checks which tests can judge a split at these layers (@res-validity), goes through the four groups of @tab-tests (@res-p-truth to @res-r-other), then the other axes of @tests-grid (@res-transfer), and ends with the inspection tests (@res-inspect).

== Overview <res-overview>

#ph[one paragraph per method: which tests it passes and fails, and the one result that best separates it from the others and from the controls]

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    breakable: true,
    [test (@tab-tests)], [plain], [recon], [ortho], [untrained], [data-aligned],
    table.cell(colspan: 6)[_$P$ on the truth readouts_],
    ..rows(("only P", "P removed", "P averaged over twins", "P-swap", "fixed shift", "component edits (M-core)"), ("n", "n", "n", "n", "n")),
    table.cell(colspan: 6)[_$P$ on the other qualities_],
    ..rows(("P removed / median", "P averaged over twins", "P-swap", "fixed shift"), ("n", "n", "n", "n", "n")),
    table.cell(colspan: 6)[_$R$ on the truth readouts_],
    ..rows(("R averaged over twins", "R-swap"), ("n", "n", "n", "n", "n")),
    table.cell(colspan: 6)[_$R$ on the other qualities_],
    ..rows(("only P (= R removed)",), ("n", "n", "n", "n", "n")),
    table.cell(colspan: 6)[_other axes and inspection_],
    ..rows(("second prompt", "Yes/No", "OOD statements", "cancellation (norms, κ)"), ("n", "n", "n", "n", "n")),
  ),
  caption: [Scorecard, filled in last (· = not filled yet). Each cell: ✓ (the expectation of @tab-tests holds within the interval, and beyond the random sets of the same size), ✗ (it fails), \~ (the seeds disagree; the cell then lists them), n/a (the site version already fails, so the test cannot judge the split; @res-validity). The thresholds behind ✓ are fixed before the outputs are read (@eval-setup).],
) <tab-res-scorecard>

#figure(
  phfig(height: 6cm)[
    The mirror plot, one panel per intervention (removal, twin average, swap). x-axis: $phi$ on the truth readouts; y-axis: $phi$ on the other qualities. One filled marker per decomposition for $P$, one open marker for $R$, coloured by method, shape by seed; a grey cloud for random component sets of the same size; crosses for the control bases. Ideal: $P$ at (1, 0), $R$ at (0, 1). A re-implementation sits high on the y-axis for $P$; a split where $P$ is (nearly) the whole matrix sits near (1, 1).
  ],
  caption: [Headline figure: the $P$/$R$ symmetry of @tab-tests in one picture. #ph[one sentence on where each method lands]],
) <fig-res-mirror>

== Can the tests judge a split at these layers? <res-validity>

=== Site versions

Before any split is judged, each intervention is applied to the whole sites (all seven `down_proj` outputs at the statement tokens) and read by every readout. A readout that the site version does not move cannot judge a split; a truth readout that the site swap moves while another does not means that A2 (one $T$ read by every truth question) fails at these layers (@ca-validate).

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    [site version], [verdict (training)], [second prompt], [Yes/No], [digit questions], [recall], [next token],
    ..rows(("removed, zero (= all off)",), ("h", "h", "h", "h", "r", "h")),
    ..rows(("removed, mean",), ("h", "r", "r", "r", "r", "r")),
    ..rows(("averaged over twins", "swapped"), ("r", "r", "r", "r", "r", "r")),
    ..rows(("fixed shift",), ("h", "h", "r", "r", "h", "r")),
  ),
  caption: [The site versions. Cell: the effect $m_"site" - m_0$ (separation share for the truth readouts, $nu$ for the swap, KL from the original for the other qualities), with interval. The mean removal of the verdict is the layer scan (@tab-layer-scan, 0.05 of the separation left); the zero removal is the all-off state of every analysis. The existing fixed-shift cells are B3, a shift along a truth direction rather than the mean difference of the site writes. #ph[which readouts the site swap moves, and whether all truth readouts move together]],
) <tab-res-site>

#ph[result: which tests remain usable; for each unusable one, which assumption failed]

=== Seeded defects

Each test is run on splits known to be wrong in the way it should catch (@ca-validate). The scaling dial turns a decomposition into a re-implementation of strength $k$ without changing the model ($W_c |-> (1 + k) W_c$, $Delta |-> Delta - k sum_c W_c$).

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    [test], [$k = 0$], [$k = 1$], [$k = 3$], [$k = 10$], [Exp 4a], [$P = W$],
    ..rows(("P-swap, truth readouts", "P-swap, shared other qualities", "fixed shift, other qualities", "P removed (vs random sets)", "R on the other qualities", "cancellation norms, κ"),
      ("r", "r", "r", "r", "r", "r")),
  ),
  caption: [Does each test catch the defect it is meant to catch? Cell: ✓ caught / ✗ missed, with the measured value. Dial values $k$ are placeholders, to be fixed with the thresholds. Expected (@ca-validate): the $P$-swap on the other qualities fails increasingly with $k$, the $P$-swap on the truth readouts may get *easier* (overshoot); $P = W$ is exposed only by its size and by $R$ on the other qualities.],
) <tab-res-defects>

#figure(
  phfig(height: 4cm)[x-axis: dial strength $k$ (log); y-axis: $phi$ of the $P$-swap, on the truth readouts and on the shared other qualities (two lines per decomposition, one panel per method). Shows that the other-quality readout catches what the truth readout misses.],
  caption: [The scaling dial. #ph[result]],
) <fig-res-dial>

== $P$ on the truth readouts <res-p-truth>

=== Only $P$, and $P$ removed

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    [basis], [$|P|$ (components; share of params)], [only $P$: KL recovered], [only $P$: separation share], [$P$ removed: separation share], [$phi_"remove"$], [$phi_"rand"$],
    ..rows(bases, ("h", "h", "h", "h", "h", "h")),
    ..rows(("site version",), ("–", "–", "–", "h", "–", "–")),
  ),
  caption: [Removal tests on the training verdict prompt. Only $P$: the active set at mask = CI, everything else off (R-active). $P$ removed: the active set set to 0, everything else kept (T2 at the active-set size); $phi_"remove" = (1 - "sep"_P) \/ (1 - "sep"_"site")$ against the zero site removal; $phi_"rand"$: the same for 5 random subsets of the active set of the same size (for the controls: the same rule on their bases).],
) <tab-res-removal>

#figure(
  phfig(height: 5cm)[x-axis: size of $P$ as the share of components, top-$k$ by T1 ($k$ = 1, 2, 5, …, 200, log scale); y-axis: $phi_"remove"$. One line per decomposition (colour = method, dash = seed), the random-$k$ band per method, the control bases dashed grey, the active-set size marked on each line. A second panel: the same with the share of parameters on the x-axis.],
  caption: [Removal against the size of $P$ (T2 re-expressed as $phi$). #ph[result: is the top-$k$ removal above random at every size, and at which size does it reach the site removal]],
) <fig-res-size>

#ph[result: only $P$ and $P$ removed, per method; how plain's top-$k$ compares with its random sets]

=== $P$ averaged over twins, $P$-swap, fixed shift

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    [basis], [twin avg: separation share], [twin avg: within-twin AUC], [IIA], [$nu$], [$phi_"swap"$], [fixed shift: flips], [$phi_"shift"$],
    ..rows(bases, ("r", "r", "r", "r", "r", "r", "r")),
    ..rows(("site version", "random, same size"), ("r", "r", "r", "r", "r", "r", "r")),
  ),
  caption: [The interventions that stay closer to the network's own states, on the twins of equal token length (476 pairs of twins in the test split before filtering by the model's verdicts; @eval-setup). IIA and $nu$ as in @tests-behav; $phi_"swap" = nu_P \/ nu_"site"$. Fixed shift: share of verdicts that flip, at the scale chosen on validation.],
) <tab-res-swap>

#figure(
  phfig(height: 4.5cm)[Per family with twins (`cities`, `neg_cities`, `larger_than`, `smaller_than`, `element_symb`, `neg_element_symb`): the distribution of $nu$ for the $P$-swap, the $R$-swap and the site swap, one row per method (seeds side by side). Ideal: $P$-swap like the site swap, $R$-swap at 0.],
  caption: [Swap fractions per family. #ph[result]],
) <fig-res-swap>

=== Component edits

The reversal (mask $-1$) and amplification (mask 2) of single components and of the top-10 group (M-core, M-twin) are already run; their figures `m_share_shift` and `m_twin` from the analysis tables go here, re-expressed against the site version where one exists (B3 for the steering of the unsplit model).

#phfig(height: 4cm)[M-core: separation share and shift under reversal and amplification of the top-10 group and of the label-selective group, per decomposition, against 10 random active components alone and the control bases; both prompts, test and the 13 OOD files.]

#status[only $P$ and $P$ removed (R-active, T2, random subsets, control bases); component edits (M-core, M-twin).][the twin builder; twin averages, $P$-swap and site swap ($"IIA"$, $nu$); the fixed shift on $P$, on random sets, on the control bases and on the sites; $phi$ for all removals.]

== $R$ on the truth readouts <res-r-truth>

Removing $R$ is the run "only $P$" (@tab-res-removal).

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto),
    [basis], [$R$ twin avg: separation share], [$R$-swap: IIA], [$R$-swap: $nu$], [$phi_(R"-swap")$],
    ..rows(bases, ("r", "r", "r", "r")),
    ..rows(("site version",), ("r", "r", "r", "r")),
  ),
  caption: [$R$ on the truth readouts. Expected: separation as in the original model under the twin average; IIA, $nu$ and $phi$ near 0 under the $R$-swap. A used second route in $R$ shows here (@ca-catches).],
) <tab-res-r-truth>

#ph[result: does any method leave a second route to the verdict in $R$]

#status[only $P$ (= $R$ removed).][$R$ twin average and $R$-swap.]

== $P$ on the other qualities <res-p-other>

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    [basis], [next token: prefix], [next token: recall position], [next token: object + end], [digit questions], [recall question], [Pile],
    ..rows(bases, ("h", "h", "h", "h", "r", "h")),
    ..rows(("site version (all off)",), ("h", "h", "h", "h", "r", "h")),
  ),
  caption: [$P$ removed, read by the other qualities. Cell: KL from the original, as a share of the site removal's KL (ideal: near 0). The existing numbers (G2, SE-Pile) remove *all* components ($Delta$ alone), not the active set, and the Pile numbers are on validation only. Digit questions: humour, emotion, surprise, word count (one column each in the appendix).],
) <tab-res-p-other-removal>

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto),
    [basis], [$P$ median: KL], [$P$ twin avg: shared qualities], [$P$-swap: shared qualities], [fixed shift: other qualities],
    ..rows(bases, ("r", "r", "r", "r")),
    ..rows(("site version", "random, same size"), ("r", "r", "r", "r")),
  ),
  caption: [The other-quality tests that a re-implementation should fail (@ca-catches). Shared qualities: those the twins share (the subject's fact, the word count; list still open, @ca-status). Cell: KL from the original next to the site version's, compared directly where the site effect is near 0 (@tests-compare).],
) <tab-res-p-other-swap>

#ph[result: does plain tPD's $P$-swap disturb the shared qualities beyond the site swap, and do recon and ortho stay at the site level]

#status[next-token and digit-question KL for all components removed and for the reversed top-10 group (G2); Pile KL with all components off (validation).][the same for the active set; the recall question; the median version; twin average, $P$-swap and fixed shift on the shared qualities, with site and random-set versions.]

== $R$ on the other qualities <res-r-other>

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto),
    [basis], [next token on the statement], [digit questions], [recall question], [Pile],
    ..rows(bases, ("r", "r", "r", "r")),
    ..rows(("site version (all off)",), ("h", "h", "r", "h")),
  ),
  caption: [Only $P$ (= $R$ removed), read by the other qualities. Cell: KL as a share of the site removal's (ideal: near 1). Weak test: $R$ holds $Delta$, so its removal breaks nearly everything whatever the split; it mainly separates $P$ from the whole matrix.],
) <tab-res-r-other>

#status[the site removal (all off) for the next-token, digit-question and Pile readouts.][only $P$ under every other-quality readout.]

== Other verdict styles and out-of-distribution statements <res-transfer>

#figure(
  tbl(
    columns: (auto, ..range(8).map(i => 1fr)),
    [basis], [only $P$, training], [only $P$, second], [only $P$, Yes/No], [only $P$, OOD], [$phi_"remove"$, training], [$phi_"remove"$, second], [$phi_"remove"$, Yes/No], [$phi_"remove"$, OOD],
    ..rows(bases, ("h", "h", "h", "h", "h", "h", "h", "h")),
  ),
  caption: [The removal tests across verdict styles (training verdict prompt, the second 1/0 prompt, the Yes/No question) and on the OOD statements (only $P$: all 24 OOD files; removal: the 13 files of T2, training and second prompt). Existing only-$P$ numbers use all components at mask = CI (R), not the active set. "Another language" is not defined yet (@tests-grid).],
) <tab-res-transfer>

#figure(
  phfig(height: 5cm)[Heat map per method (columns: verdict style × in/out of distribution; rows: removal, twin average, $P$-swap, $R$-swap, fixed shift; cell colour: $phi$, with ✗ where the site version fails). Shows at a glance whether a decomposition's effect transfers beyond the prompt it was trained on.],
  caption: [Transfer of each test across the axes of @tests-grid. #ph[result: which methods transfer to Yes/No and to OOD; ortho s2 and s3 against s1]],
) <fig-res-transfer>

#status[R under the training and second prompts and on OOD (phase D), under Yes/No (G2); T2 under all three styles and on 13 OOD files.][every swap, twin-average and shift test under the other styles and on OOD twins (OOD twins not yet counted).]

== Inside the decompositions <res-inspect>

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    [basis], [active set], [$|C| \/ |y|$], [$|D| \/ |y|$], [$chevron.l C, D chevron.r \/ |y|^2$], [$kappa$ (twins)], [max $|cos|$ with truth direction], [K2: share that transfers],
    ..rows(bases, ("h", "h", "h", "h", "r", "h", "h")),
  ),
  caption: [Inspection tests (@tests-inspect). Norms: mean over statement tokens, range over the seven layers. $kappa$: cancellation between twins, 1 = none. Truth direction: mass-mean direction after layers 11–17 (Dir); expected value of max $|cos|$ for random vectors in the caption once filled. K2: share of the validation-selective components that keep their sign on the OOD sets of the same form.],
) <tab-res-inspect>

#figure(
  phfig(height: 4cm)[Left: $|C|$ and $|D|$ per layer (11–17), one line per decomposition. Right: CI histograms per method (valley check). Existing figures: `ci_histograms`, `active_set_curve`, `d_alignment`.],
  caption: [Cancellation and CI structure. #ph[result: plain's components and $Delta$ write against each other, recon's and ortho's do not; whether the CI values have a gap]],
) <fig-res-inspect>

#ph[result: seed comparison: do the seeds of one method find the same components]

#status[cancellation norms, active set, valley check, seed comparison, Dir, K1, K2.][$kappa$ on the twins.]


= Discussion

== What the tests can and cannot detect <ca-tests>


=== Ways a split can look right without capturing $T$ <ca-failures>

+ *A second route.* $R$ supplies $T$ too, by a route of its own.
+ *A re-implementation.* The components write a large vector $c$ that is not a part of the layer's real output $y$, and $R$ writes $y - c$, cancelling it. On every input $p + r = y$, so no test on the unedited model can see it. Under an edit that separates $P$ from $R$ (e.g. $P$-writes from $x'$, $R$-writes from $x$), whatever in $c(x') - c(x)$ the truth readouts do not read is no longer cancelled. The truth readouts cannot notice, because they read only $T$. It shows only in other outputs that read those directions, i.e. in the other-quality readouts.
+ *A lever.* $P$ moves the verdict without carrying $T$: a bias towards "1" or "0", or a feature of the sentence form.
+ *A different route.* $P$ produces the verdict from a cue that agrees with the model's own computation on the training data, and the two come apart elsewhere (other questions, other sentence forms).
+ *Passing on.* $P$ carries a $T$ computed elsewhere (before layer 11, or in attention).
+ *Everything.* $P$ is (nearly) the whole matrix.

=== Which test catches which <ca-catches>

- *A second route:* the $R$-swap and the $R$ twin average (the verdict must not move). The $P$-swap catches it too, if the second route is used: the $R$-writes are recomputed under the $P$-swap, so a route of their own from $x$ to $T$ would pull the verdict back towards that of $x$ #concluded. A copy of $T$ in the $R$-writes that nothing reads does not matter.
- *A re-implementation:* the $P$-swap and the fixed shift read by the other qualities the twins share, and internally the cancellation norms and $kappa$ (@tests-inspect). The truth readouts alone may even look *better* under a re-implementation (overshoot, @ca-validate).
- *A lever:* the twin average (a bias survives it but cannot separate the true twin from the false one), the $P$-swap (a bias does not transcribe the verdict of $x'$), and the other truth formats (a bias towards the digit "1" does not move a Yes/No answer).
- *A different route:* the other truth formats and the OOD statements.
- *Passing on:* only; every behavioural test is passed by a $P$ that passes $T$ on.
- *Everything:* the size of $P$, the random sets of the same size, and the tests of $R$ on the other qualities.

=== Limits

*What each test assumes.* The swaps need A2 (@ca-narrow) only for the states the network reaches. Reading the other qualities adds A3. The twin average and the fixed shift produce states no statement produces, so they need $cal(H)$ to hold off the network's states as well; for the fixed shift this is where superposition bites (): it sets $T$ against the content, which may be impossible or only approximate if $T$ is stored entangled with it. The removals have no counterpart in $cal(H)$ at all: they show *whether* a part is used, not *how* (Geiger et al., Sec. 3.5). The more a test assumes, the less a failure says about the split alone, which is why every test has a site version (@tests-compare, @ca-validate).

*Routes around the sites.* Only some layers are split, so $T$ may also travel by routes that bypass the sites; then even the site version moves the verdict less than fully. Reading every result against the site version (@tests-compare) accounts for this.

*What no behavioural test sees.* A split can differ from the model's mechanism in ways no output reads. Only the inspection tests (@tests-inspect) can see such differences, and they show cancellation or structure, not unfaithfulness.

== Validating the tests <ca-validate>

A test counts as evidence only if it could have failed. Besides the comparison runs of @tests-compare (random sets of the same size, control bases), two checks make sure it could.

*Assumption checks with the site version.* Every behavioural test on $P$ has a site version (@tests-compare), run first: the same intervention on the whole layer outputs (site swap, site shift, twin average of $y$, site removal). Under A1–A3 the site swap moves every truth readout together and leaves the shared other qualities in place. If the site version fails a test, that test cannot judge the split at these layers. Example: if the site swap moves the 1/0 verdict but not the Yes/No answer, then a split whose effect does not transfer to the Yes/No question has not failed; A2 (one $T$ read by every truth question) has, at these layers.

*Seeded defects.* Each test is run on splits that are known to be wrong in the way it is meant to catch:
- *Re-implementation by construction:* for any decomposition, $W_c |-> (1 + k) W_c$ and $Delta |-> Delta - k sum_c W_c$ keep the model exactly and turn the components into a re-implementation of growing strength $k$. The $P$-swap and the fixed shift read by the other qualities must fail increasingly with $k$; the $P$-swap read by the hard verdict may get *easier* (overshoot), which is why the truth readouts alone are not enough.
- *A known re-implementation:* Exp 4a, an earlier tPD run with the verdict loss only, whose components were shown to cancel against $Delta$ instead of decomposing the MLP outputs.
- *Everything:* $P = W$ passes every test on $P$ trivially; only its size and the tests of $R$ on the other qualities expose it. It fixes the top of the scale.

== Evaluation setup <eval-setup>

*Statement sets.* The validation and test splits of the Truth-is-Universal statement sets (test: 10 families, 1,794 statements, @tab-verdict-accuracy) and 24 out-of-distribution (OOD) files; T2 and M-core use 13 of the OOD files. The model is weakest on the negated families and on `element_symb`, where it calls a quarter of the true statements false #verified[run B2, original model, test split].

*Twins.* Two statements of the same sentence form and domain that disagree in their truth value. In our sets they are the true and the false statement about the same subject ("The city of Lu'an is in China." / "… in Turkey."; in `larger_than`, the same two numbers in the opposite order). In the test split each subject has exactly these two statements, except in `sp_en_trans` and its negation, where the false statements mostly use another Spanish word #verified[count by subject, 26-11-10]. The swaps and the twin averages replace the writes at each token position by those of the other twin at the same position (or by the mean of the two), so only twins with the same number of tokens are used. The swaps also use only twins on which the model's verdicts differ, so that they are balanced with respect to the model's belief. The test split holds 476 pairs of twins of equal token length: 115 each in `cities` and `neg_cities`, 103 each in `larger_than` and `smaller_than`, 18 each in `element_symb` and its negation, 2 each in `animal_class` and its negation, none in `sp_en_trans` and its negation #verified[count with the Qwen tokenizer, 26-11-06; before filtering by the model's verdicts]. More twins can come from the training and validation splits; OOD twins have not been counted.

*Prompts.* The training verdict prompt; the held-out *second prompt* "Is this statement true (1) or false (0)? Answer: ", a different truth question with the same 1/0 format; two Yes/No truth questions; four digit questions about other properties of the statement (humour, emotion, surprise, word count). G0 checks that the model answers the last two kinds in format. Every edit acts at the statement tokens; the appended question runs on the original weights.

*States of the analyses run so far.* These analyses replace the output of each decomposed matrix at the statement tokens by
$ y' = m_Delta D + sum_c m_c U_c V_c^top x $
(notation of @methods). The named states: *original* (all masks 1); *components only* ($m_c = upright("CI")_c$, $m_Delta = 0$); *$Delta$ alone* ($m_c = 0$, $m_Delta = 1$); *all off* ($m_c = 0$, $m_Delta = 0$: the decomposed matrices write nothing at the statement tokens); *removals* (chosen components from 1 to 0, everything else kept); *reversal* and *amplification* (chosen components at $m_c = -1$ or $2$). Decompositions and controls run through the same code, so a difference between them cannot come from the intervention.

*Measures,* at the answer position:
- *KL recovered* $= 1 - "KL"("original" || "state") \/ "KL"("original" || "all off")$, each KL averaged over statements: 1 = the original verdict distribution, 0 = as bad as all off.
- *Accuracy* (against the labels) and *agreement* (against the model's own verdict).
- *Separation share* (@methods) and *shift*: the signed move of the midpoint between the true and the false class means of $s$, in units of the original gap (> 0 = towards "1"). Together they fix both class means. The shift was added after the outputs were read; it is descriptive.
- *Flips*: the share of the model's "1" verdicts that become "0", and of its "0" verdicts that become "1".
- *Digit mass* (@tab-verdict-accuracy) and *AUC* of a score for true against false within a family.
- For swaps: $"IIA"$ and the swap fraction $nu$ (@tests-behav).
- *Effect fraction* $phi$: the share of the site version's effect (@tests-compare).
Metrics are computed per run, never pooled over seeds; pooled over statements and macro-averaged over families, with 95% bootstrap intervals (1,000 resamples within family).

*Controls.* Two bases that nobody trained, run through the same code:
- the *untrained decomposition*: a run's checkpoint at step 0, where $Delta$ holds essentially the whole matrix;
- the *data-aligned basis*: per layer, read vectors $V_c$ = the 96 top principal directions of the matrix's input (uncentred, over the statement tokens of 512 training statements), write vectors $U_c = W V_c$, remainder $W (1 - P_V)$. It has no CI network, so analyses that need CI values do not apply to it.
Within a decomposition, random choices of the same size: random subsets of the active set (T2, 5), random subspaces of the same dimension (T3, 3), random active components (M-core, 10).

*Pre-registration.* The analyses in @glossary were fixed in a plan before the test and OOD sets were read, except G0–G2, which were added after the first results, and the shift. The tests of @tests that are not yet run (@ca-status) will be fixed, with their pass thresholds, before their outputs are read.

== Status: what has been run and what has to be added <ca-status>

Our analyses were planned before these tests, and most of them are removals (components switched off, reversed or amplified), not averages or swaps. They cover the removal tests on the truth verdict; for the other tests they are at best removal versions, whose states lie off the network's natural states and whose reference (all off) is itself a broken model. None of them is yet expressed as the effect fraction $phi$ against the site version (@tests-compare).

#tbl(
  columns: (auto, 1.2fr, 1.3fr, 1.6fr),
  align: (left, left, left, left),
  breakable: true,
  [test (@tab-tests)], [run so far (@glossary)], [what it covers], [to add or redo],
  [$P$ removed, truth], [T0, T1, T2, T3 (A8), Dir-remove; T2 under the second prompt, Yes/No and on OOD], [against random sets of the same size and the control bases], [*add:* $phi$ against the site removal],
  [only $P$ (= $R$ removed), truth], [R, R-active, R-mean, active set; under the second prompt, Yes/No (G2) and on OOD], [complete], [*add:* $phi$],
  [$P$ removed, other qualities], [G2 next-token, G2 digit questions ($Delta$ alone and the reversed top-10 group, against all off); SE-Pile ($Delta$ alone)], [removal against all off; no random sets], [*add:* recall question; median of $P$; random-set and site versions],
  [only $P$, other qualities], [none], [–], [*add:* digit questions, next-token on the statement and on the Pile],
  [$P$ / $R$ averaged over twins], [none (T0 sets the components to zero instead of averaging)], [–], [*add:* both, truth and shared other qualities, site version],
  [$P$-swap / $R$-swap, truth], [none], [–], [*add:* twin builder (equal length, different verdicts); $P$-, $R$- and site swap; $"IIA"$, $nu$, $phi$, $kappa$],
  [$P$-swap, other qualities], [none], [–], [*add:* choose the qualities the twins share; KL against the site swap],
  [fixed shift], [B3 on the unsplit model (steering along a truth direction)], [a site-like version for the verdict and recall], [*add:* the shift on $P$, random-set, control-base and site versions; truth and other readouts],
  [hand and component edits], [M-core, M-twin, M-wide (masks −1 and 2)], [whether reversing or amplifying components moves the verdict, against random components; G2 digit questions under the reversed top-10 group], [hand edits, if we find a hypothesis to edit],
  [does $P$ compute $T$], [none], [–], [*add:* the projection protocol of ],
  [inspection], [cancellation norms, active set, valley check, seed comparison, Dir, K1, K2], [complete for the decompositions], [*add:* $kappa$ on the twins],
  [site versions], [layer scan, B1, B3], [removal versions on the original model (which layers the verdict needs)], [*add:* the site version of every new test, first of all the site swap under the Yes/No prompts],
  [seeded defects], [none], [–], [*add:* scaling dial $k$, Exp 4a, $P = W$],
)

*Analyses not tied to a test.* K0 (probes over all features of one kind) is reported as the reason probe accuracy is not evidence. K3 (on statements the model gets wrong, does a selective component follow the label or the model's verdict) is descriptive: the decompositions are trained on the model's own verdict distribution, never on labels.

*Proposed order* #assumed: the site swap under every readout first (it decides which tests can work at these layers); then the $P$- and $R$-swaps, read by the truth and the other qualities, together with the scaling dial, since they share the twins and the swap code; then the twin averages, the fixed shift, and the remaining axes of @tests-grid.

*Open points.*
- *Which other qualities.* The swaps need qualities that the twins share, read off the model in a format it answers reliably.
- *Twins.* Twins differ in content as well as truth; the swap carries that content along. This is allowed (the $P$-writes carry a bundle containing $T$), but restricts which qualities the twins share.
- *Another language:* the question in another language, or the statements.
- *Thresholds* $q$ and the statement distribution $D$ for each test.

== Glossary of experiments <glossary>

All analyses run on every decomposition (plain, recon, ortho; all seeds) and, where they apply, on the controls. Test split unless stated; *OOD* = the out-of-distribution statement sets; *both prompts* = the training verdict prompt and the second prompt. Removals and edits act at every statement token. Which test of @tests each analysis bears on is listed in @ca-status.

#tbl(
  columns: (auto, 1fr),
  align: (left, left),
  breakable: true,
  [name], [what is done],
  [*layer scan*], [Original model. The output of single MLP layers or blocks replaced by its mean over training statements, at all statement tokens or at the last one only; verdict accuracy, separation share, digit mass. Chose layers 11–17 (@tab-layer-scan).],
  [*B1*], [Original model. The layer scan repeated for MLP and attention outputs, single layers and blocks, mean and zero ablation; why attention is not decomposed.],
  [*B2*], [Original model on every test and out-of-distribution statement set under three candidate prompts: accuracy, AUC, digit mass per set. Chose the second prompt and which OOD sets are usable.],
  [*B3*], [Original model. Steering along the in-context truth direction at the statement tokens flips the verdict under both prompts, but the model still recalls the fact correctly afterwards.],
  [*G0*], [Original model, validation subset. Checks that the extra prompts of G2 (Yes/No questions, digit questions) are answered in format.],
  [*R* (recovery)], [Components only: every component at mask = CI, $Delta$ off. KL recovered, accuracy, agreement with the model, separation share, digit mass; test, all 24 OOD files, both prompts.],
  [*R-active*], [As R, but only the active set at mask = CI, all other components and $Delta$ off.],
  [*R-mean*], [As R, at the training's expected mask $upright("CI") + (1 - upright("CI")) \/ 2$.],
  [*active set*], [Components ranked by mean CI; the curve of KL recovered when only the top $m$ are on ($m$ = 1, 2, 4, …, 512); the active set is the smallest such set reaching 95% of R's KL recovered (validation). No CI threshold defines "active" anywhere.],
  [*G2 Yes/No transfer*], [R under two Yes/No truth questions ("… Yes"/"No" instead of 1/0): does the decomposition transfer to an answer format it was not trained on? Added after the plan.],
  [*T0*], [All components removed from the full model ($Delta$ alone). KL recovered (ideal: 0, i.e. $Delta$ alone carries nothing of the verdict) and separation share.],
  [*R-inactive*], [The full model with every component outside the active set removed; plus the norms of the active and inactive writes and their cosine.],
  [*cancellation norms*], [At the statement tokens: $|C|$, $|D|$ and $chevron.l C, D chevron.r \/ |y|^2$ per layer. Large opposing writes = the components and $Delta$ cancel each other.],
  [*T1*], [Each candidate component removed alone from the full model (mask 1 → 0, everything else kept). On validation for all candidates (this ranking feeds T2 and M-core), on test for the active set.],
  [*T2*], [The top $k$ candidates by T1 removed together, $k in {1, 2, 5, 10, 20, 50, 100, 200}$ plus the active-set sizes; against 5 random size-$k$ subsets of the active set, and the same rule on the untrained decomposition and the data-aligned basis. Test and 13 OOD files, both prompts (the random controls on OOD on at most 50 statements per file).],
  [*T2 Yes/No*], [T2 under the Yes/No prompts (G2).],
  [*T3 (A8)*], [No decomposition in the forward pass: one layer at a time, the *original* MLP output loses its projection onto the span of (a) all 96 write vectors $u_c$ of that layer, (b) the active set's write vectors in it. Controls of the same dimension: 3 random subspaces and the top principal directions of the output.],
  [*Dir-remove*], [The mass-mean or logistic-regression truth direction projected out of the original MLP output, per layer and in all seven layers together; compared with removing the components.],
  [*SE-Pile*], [Side effect on general text: next-token KL on the Pile with all components off and $Delta$ on, from the training's evaluation.],
  [*G2 next-token*], [Side effect within the statement: next-token KL against the original at the statement tokens, split into prefix (before the object; truth not yet determined), recall position (predicts the object) and object + end; for $Delta$ alone and the reversed top-10 group, compared with all off.],
  [*G2 digit questions*], [Same-format control judgements: four questions about other properties of the statement (humour, emotion, surprise, word count), answered with a digit, under $Delta$ alone and the reversed top-10 group.],
  [*M-core*], [The top 10 components by T1, each alone and as a group, and the label-selective group (K2), reversed (mask −1) and amplified (mask 2); 10 random active components alone; the same rule on the controls. Flips, separation share and shift, digit mass; test and 13 OOD files, both prompts. Masks −1 and 2 lie outside the trained range.],
  [*M-twin*], [The one prediction fixed in advance: under an edit, does a negated statement's verdict follow the original verdict of its affirmative twin ("the 'not' is ignored")?],
  [*M-wide*], [Exploratory, validation only: every active component reversed alone, per family and label.],
  [*Dir*], [Mass-mean and logistic-regression truth directions after layers 11–17 at the statement's last token (fit on training statements). Per component the cosine of its write with each direction, against random vectors; the share of the true–false gap along the direction carried by the active components, the inactive ones and $Delta$; the rank correlation of T1 with the gap share.],
  [*valley check*], [Histograms of all CI values (validation): is there a gap between "off" and "on" that would define "active"?],
  [*seed comparison*], [Every table shows all seeds side by side; no method is ranked above another where its seeds overlap.],
  [*K0*], [Logistic probes over all features of one kind (all CI values; all coefficients; the residual stream after layer 17), fit on training statements, scored on test and OOD, for the label and the model's verdict. Reported as the reason probe accuracy is not evidence.],
  [*K1*], [Per component the AUC for truth within each family; probes on the top $k$ components by training selectivity, $k in {1, 2, 5, 10}$, against the controls at the same $k$.],
  [*K2*], [Components selective on validation within a sentence form ($|"AUC" - 0.5| >= 0.2$) keep their sign on the OOD sets of the same form ($|"AUC" - 0.5| >= 0.1$): share that transfers, against the controls.],
  [*K3*], [On the statements the model gets wrong: does a selective component follow the label or the model's verdict?],
)
