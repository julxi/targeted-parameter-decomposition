#import "template.typ": *
#import "26-11-04_tpd_truth_paper_draft_diagram.typ": target_panel, nontarget_panel, restriction_panels

// Paper draft written from scratch on 2026-11-17 at Julian's request (C: comment at the bottom of
// convos/julian/26-11-15_open_research_LOG.md): S = all components, the minimal test set in the main
// part, minor findings in the appendix. Sources of every number: the generated tables in
// ~/spd_out/26-11-15_open_research/analysis/ (session2.md, score.md, inspect.md, tables.md,
// truth_linked_transfer.md), the phase D tables scratch/26-10-29_paper_writing/tables/tables.md, and
// the site and dial tables of reports/26-11-04_tpd_truth_paper_draft.typ.

#show: report.with(
  title: "Trying to decompose a truth mechanism with tPD",
  subtitle: "Paper draft: three targeted parameter decompositions of Qwen2.5-7B-Instruct's truth verdict, and five tests that each of them fails in a different way",
  date: "2026-11-17",
  author: "Julian Bitterlich; draft text by Claude Opus 5.5",
  summary: [
    - *Question.* Can targeted parameter decomposition (tPD) isolate, in the MLP output matrices of layers 11–17 of Qwen2.5-7B-Instruct, the part of the weights that carries the model's truth value of a statement? We compare plain tPD with two variants that add a constraint on what the components write: per-layer reconstruction (*recon*) and global orthogonality (*ortho*).
    - *Split.* Every test splits each decomposed matrix into the sum of all its components ($C$) and the remainder $Delta$. This is the split the method defines; the "active" components alone are not a self-contained part (removing the others costs 17–60% of the verdict's true–false separation).
    - *Tests.* Five tests on pairs of a true and a false statement about the same subject (*twins*): (1) where the truth value sits, by swapping $C$ or $Delta$ between twins with the other part held fixed; (2) whether $C$ is a part of the model's output; (3) whether swaps damage answers that should not shift; (4) the cost on ordinary text; (5) whether two seeds find the same thing.
    - *Results.* Each method fails a different test. Plain tPD re-implements the verdict with writes the model does not have (test 2: 0.01–0.02 against 0.73), damages recall (test 3) and finds unrelated solutions per seed (test 5). Recon is a faithful part of the output and carries most of the truth value (test 1: 0.86–0.89 of the site swap, $Delta$ 0.09–0.10), but it rebuilds much of what the layers do on any text (test 4). Ortho's components carry little of the truth value (0.11–0.33; $Delta$ 0.61–0.84) and its seeds disagree (test 5). A subspace fitted to twin differences, without any decomposition, beats all three on test 1 (0.94 / 0.01).
    - *Status.* Every number in the main text is measured (one run per cell). Not run: test 1 with components weighted by their causal importance.
  ],
)

= Introduction <intro>

Parameter decomposition methods split a network's weights into parts that are meant to correspond to the mechanisms the network uses. Targeted parameter decomposition (tPD; Vigouroux and Sharkey 2026) restricts this to the mechanisms that process a chosen subset of inputs: a few rank-1 *components* must carry the network's behaviour on the target inputs, while a high-rank remainder $Delta$ absorbs everything else.

We ask whether tPD can isolate a high-level mechanism in a large language model: the computation behind the model's judgement that a statement is true or false. The target behaviour is the answer of Qwen2.5-7B-Instruct to the question how true a statement is. We compare three methods: plain tPD, and two variants that add a constraint on what the components write at the statement's tokens. We evaluate them with five behavioural tests built on *twins*, pairs of a true and a false statement about the same subject.

The main result is negative, and specific. Each method fails at least one test, and each fails a different one:
- plain tPD builds its own route to the verdict that the model does not have (a *re-implementation*);
- recon decomposes the layers faithfully, but much of what it decomposes is the layers' general computation, not the truth judgement;
- ortho stays targeted, but its components carry a minority of the truth value, and its seeds find different things.

A subspace of the model's own outputs, fitted directly to the differences between twins, does better than all three on the central test. A secondary result is about the tests themselves. Several natural tests mislead: removal tests, swaps in which the other part is recomputed, and swaps read only by truth questions. We explain why, and which tests replace them.

= Setting <setting>

#figure(
  block(width: 100%)[
    #set align(left)
    #text(size: 8pt, weight: "bold")[Target data]
    #target_panel
    #v(0.6em)
    #text(size: 8pt, weight: "bold")[Non-target data]
    #nontarget_panel
  ],
  caption: [The setup of tPD. Blue: layers whose `down_proj` matrix is decomposed, $W' = sum_c m_c U_c V_c^top + m_Delta Delta$, at the statement tokens; grey: original weights; dashed light blue: original weights whose input carries the decomposition's indirect effect; orange: the outputs compared with the original model by KL. The dashed orange outline marks the extra loss of recon and ortho on the components' writes (@fig-restrictions).],
) <fig-setup>

== Model, task and score

The model is Qwen2.5-7B-Instruct, frozen. It reads a statement followed by a fixed text that asks about it,

#align(center)[`<statement>\nThe amount of truth in this statement is␣`]

and its next token is almost always the digit `1` (true) or `0` (false). We call such an appended text a *probe*, and the answer to a probe about truth the *verdict*. This probe, the one used in training, is *truth rating 1*. We score a statement by the logit difference $s = log P(1) - log P(0)$ at the answer position; the model calls it true if $s > 0$.

The statements come from the Truth-is-Universal sets: ten families (cities, element symbols, animal classes, Spanish–English translations, number comparisons "larger than" and "smaller than", and the negations of the four fact families), each in a true and a false version. The model does the task well (@tab-verdict-accuracy).

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
  caption: [The original model under truth rating 1, test split. Accuracy: share with the right sign of $s$; AUC: of $s$ for true against false statements within the family; digit mass: $P(1) + P(0)$, averaged.],
) <tab-verdict-accuracy>

== What is decomposed, and why there

We decompose the `down_proj` matrix (the MLP's output projection) of each of the layers 11 to 17. Each of these seven matrices is written as
$ W = sum_(c=1)^(96) U_c V_c^top + Delta, $
96 rank-1 components per layer (672 in all) plus a full-rank remainder $Delta$. One *causal-importance* (CI) network, shared by the seven layers, gives every component a value in $[0, 1]$ per token. The decomposition replaces the weights only at the *statement tokens* (every token of the statement up to and including the one that holds its final period); the probe always runs on the original weights. So whatever a decomposition changes reaches the verdict only through the probe tokens' attention to the statement.

*Why these layers.* We replaced the MLP outputs at all statement tokens by their mean over the training statements and measured what is left of the verdict (@tab-layer-scan). The measure is the *separation share*: the gap between the mean scores of true and false statements, as a fraction of the original gap.
- No single layer is necessary: removing one MLP from layer 5 to 24 leaves at least 0.77 of the separation.
- The block 11–17 is sufficient: removing it leaves 0.05 (accuracy at chance), while the model still answers with a digit (digit mass 0.977). It is the smallest block in the scan with this property.
- All statement tokens are needed: removing the block at the last statement token only leaves 0.82 of the separation.

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
  caption: [Causal scan of the original model under truth rating 1 (test split, 1,794 statements). KL: of the answer distribution against the original's. Single layers 3, 4, 25 and 26 are left out: a few tokens there have outputs of very large norm, so their mean is itself an unusual write.],
) <tab-layer-scan>

*Limits of this choice.* Only the MLP's output side is decomposed, so the components show which hidden directions the MLP reads out and where it writes them, not how those directions are computed. Attention matters too (removing attention 11–14 leaves 0.44 of the separation); we do not decompose it, because removing MLPs 11–17 alone already takes the verdict to chance, so every route to the verdict passes through these MLPs.

== The three methods <methods>

*Shared terms (plain tPD).* Each training step uses a batch of 64 statements from the training split (target data) and 16 windows of 64 tokens from the Pile (non-target data).
+ *Verdict loss under random masks* (statements). Each component's contribution at each statement token is scaled by a mask $m_c = upright("CI")_c + (1 - upright("CI")_c) u$, $u$ uniform in $[0, 1]$; $Delta$'s by $m_Delta$ uniform in $[0, 1]$. At each position a random number of the seven layers (uniform in 1 to 7) uses the masked decomposition, the others the original weights; this routing is the SPD codebase's standard stochastic loss, inherited unchanged. The loss is the KL divergence between the original and the masked model's answer distribution at the last probe token. The only signal is the verdict.
+ *Verdict loss under adversarial masks* (statements; last 20% of training): the same KL with masks chosen to maximise it, all seven layers masked.
+ *Importance minimality*: an $L_p$ penalty on the CI values ($p$ annealed from 2 to 1), on the statements, and at twice the coefficient on the Pile.
+ *Non-target loss* (Pile): the next-token predictions at every position, compared by KL with the original's, with $Delta$ fully on and the components' masks random in $[upright("CI")_c, 1]$. A mechanism the model uses on the Pile but not on the statements costs CI if it sits in a component and nothing if it sits in $Delta$, so it ends up in $Delta$.
+ *Weight decay* on the component vectors, scaled by one minus the component's largest CI in the batch.
So the components with high CI must carry everything the verdict needs, and $Delta$ must not be needed for it.

*Recon* adds a per-layer reconstruction term at the statement tokens: under the same random masks (all seven layers masked), each matrix's output $hat(y)$ must match the original output $y$, $cal(L)_"recon" = "mean" |hat(y) - y|^2 \/ |y|^2$. Because $Delta$'s mask is random, the components with high CI must rebuild the output themselves. Expected consequence: a faithful decomposition of the whole MLP output at the statement tokens, not only of its verdict-relevant part.

*Ortho* adds a global orthogonality term at the statement tokens. With $C = sum_c U_c V_c^top x$ the write of all components at full strength and $D = Delta x = y - C$ the remainder's write,
$ cal(L)_"ortho" = "mean over statement tokens and layers of" (chevron.l C, D chevron.r \/ |y|^2)^2. $
With $C + D = y$, $chevron.l C, D chevron.r = 0$ puts $C$ on the sphere whose diameter is $y$: the components and $Delta$ cannot write large vectors that cancel each other, but the term does not say *which* part of $y$ the components carry, so the method stays targeted. It rests on the assumption that the verdict-relevant part of each MLP output is roughly orthogonal to the rest #assumed.

#figure(
  restriction_panels,
  caption: [What the extra terms ask of one decomposed matrix at one statement token. $y = W x$: the original output; $C$: the write of all components at full strength; $D = Delta x$: the remainder's write. Plain: no condition. Recon: $C approx y$. Ortho: $C$ on the circle with diameter $y$.],
) <fig-restrictions>

#figure(
  tbl(
    columns: (auto, 1fr, 1fr, 1fr),
    [], [plain], [recon], [ortho],
    [extra term], [none], [per-layer reconstruction, coefficient 10], [global orthogonality, coefficient 10],
    [learning rate (cosine decay to 10%)], [$5 dot 10^(-4)$], [$5 dot 10^(-4)$], [$2 dot 10^(-3)$],
    [importance minimality (statements; Pile ×2)], table.cell(colspan: 3)[$10^(-3)$, with a warm-up to 4× in the first 20% of training and a decay back],
    [adversarial verdict loss], table.cell(colspan: 3)[weight 0.5, steps 4,000–5,000, all 7 layers masked],
    [components per layer], table.cell(colspan: 3)[96 (672 in all)],
    [steps; batch], table.cell(colspan: 3)[5,000; 64 statements + 16 Pile windows],
    [seeds], [1, 2], [1, 2], [1, 2, 3, 4],
    [components alone recover (KL recovered, test)], [0.961, 0.958], [0.974, 0.971], [0.906, 0.893, 0.875, 0.903],
  ),
  caption: [Settings. Learning rate and importance-minimality coefficient were tuned on the validation split for plain tPD, then only the extra term's coefficient for recon and ortho, plus ortho's learning rate. Last row: KL recovered $= 1 - "KL"("original" || "components only") \/ "KL"("original" || "all off")$ at the verdict, with every component at its CI and $Delta$ off (1 = the original verdict distribution, 0 = as bad as removing the seven outputs). All three methods do what their verdict loss asks.],
) <tab-settings>

= How the decompositions are evaluated <evaluation>

== The split: all components against the remainder <split>

At a decomposed layer and statement token, the output splits exactly into
$ y = C + D, quad C = sum_(c=1)^(96) U_c V_c^top x quad "(all components at full strength)", quad D = Delta x. $
Every test below edits $C$ or $D$ at every statement token of all seven layers. We use all components, not the components that are *active* (high CI) on a statement, for three reasons:
- *It is the split the method defines.* tPD writes $W$ as components plus $Delta$, and its losses compare the components with $Delta$: the components must suffice without $Delta$. "Active" needs a threshold or a cutoff that appears in no loss, and ortho's term constrains only the sum of all components.
- *The inactive components are not inert.* Removing the components outside the active set from the full model costs 17–60% of the verdict's separation in every method, and in ortho these components cancel 60–87% of what the active ones write (@app-active). So the active set alone is not a self-contained part of the model.
- *Equal size.* $C$ has 672 components in every decomposition; active sets range from 27 to 432 components, which would mix a size effect into every comparison between methods.
What this leaves out: whether the CI values pick out the right components for a given statement. That would need the same tests with components weighted by their CI, which we have not run (@limits).

== Twins, swaps and readouts <swaps>

*Twins.* Two statements about the same subject with opposite truth values, usually differing only in the object: "The city of Lu'an is in China." / "… is in Turkey."; in the comparisons, the same two numbers in the opposite order. A swap replaces writes position by position, so we use only twins with the same number of tokens, and only those on which the model's verdicts differ. The test split has 449 such pairs (898 rows; each pair is used in both directions). We report three groups: *affirmative facts* (cities, element symbols, animal classes; 264 rows), *negated statements* (238), *comparisons* (396). The Spanish–English families have no twins of equal length.

*Swap.* On twins $x$ and $x'$, a part's writes on $x$ are replaced by its writes on $x'$. The *swap fraction*
$ nu = (s_"swap" - s(x)) \/ (s(x') - s(x)), $
pooled over rows, is 0 if the score stays with $x$ and 1 if it moves all the way to the twin's.

*Clamped swap.* When one part is swapped, the other part is held at its value on $x$ at every layer, instead of being recomputed from the edited run. The seven outputs are then exactly $q(x') + (y(x) - q(x))$ for the swapped part $q$. Clamping matters: unclamped, the recomputed part at later layers reads what the swapped part wrote earlier and partly follows it, so the effects of swapping $C$ and of swapping $Delta$ do not add up to the effect of swapping everything (@app-clamp). Clamped, they do, within 0.05 in all eight decompositions (@tab-test1).

*Site swap.* The same swap of the whole seven outputs ("the sites") is the reference: no part can do more. It brings over $nu = 0.94$ [0.94, 0.95] of the twin's verdict under truth rating 1 and 0.91 [0.90, 0.93] under Yes/No 1.

*Readouts* (@tab-probes):
- *truth readouts*: truth rating 1 (training) and *Yes/No 1*, an answer format no decomposition was trained on, as a check that a result concerns the truth value and not the digit;
- *damage readouts* (test 3): *object recall* (the probability that the model repeats the statement's own object when asked for it) and the *word count* rating.

#figure(
  tbl(
    columns: (auto, 2.4fr, 1.2fr),
    align: (left, left, left),
    [probe], [text after the statement], [answer],
    [truth rating 1 (training)], [`\nThe amount of truth in this statement is␣`], [1 / 0],
    [Yes/No 1], [`\nQuestion: Is the statement above true? (Yes/No)\nAnswer:`], [Yes / No],
    [word count], [`\nThe number of words in this statement is␣`], [digit 0–9],
    [object recall], [`\nThe {noun} in the statement above is␣"` ({noun}: country, number, …, per family)], [the stated object, scored over its spellings],
  ),
  caption: [The probes used in the main text. `␣` marks a space. Every edit acts at the statement tokens; the probe runs on the original weights, and because attention is causal the same edited statement can be read under every probe.],
) <tab-probes>

== Five tests <tests>

#figure(
  tbl(
    columns: (auto, 1.5fr, 1.4fr, 1.2fr),
    align: (left, left, left, left),
    [test], [what is measured], [a perfect decomposition], [what it detects],
    [*1. Where is the truth value?*], [$nu$ of the clamped $C$-swap and of the clamped $Delta$-swap], [$C$ at the site swap (0.94), $Delta$ at 0], [a truth value that stays in $Delta$],
    [*2. Is $C$ a part of the model's output?*], [$nu$ of the swap of the *original* output inside the span of $C$'s write vectors, next to the $C$-swap; random subspace of the same size as control], [about equal; random near 0], [re-implementation],
    [*3. Damage*], [mean change (*bias*) of object recall and word count under the clamped swaps], [as under the site swap], [writes that break unrelated answers],
    [*4. Off-target cost*], [$C$ removed at every position of Pile text: next-token KL as a share of removing the seven outputs], [near 0], [a decomposition of everything, not of the verdict],
    [*5. Seed agreement*], [cosine between two seeds' $C$-writes on twin differences], [near 1], [arbitrary solutions],
  ),
  caption: [The tests reported in the main text. All use $C$ = all components at full strength and $D = Delta x$, at the statement tokens of layers 11–17.],
) <tab-tests>

*Test 1* asks where the model's truth value sits after the split. If $C$ carried it and $Delta$ did not, swapping $C$ between twins would move the verdict as far as swapping everything, and swapping $Delta$ would not move it. The two clamped numbers add up to the site swap, so they share it out between the parts. Swaps are used instead of removals because removing a part puts the network in a state far from any it reaches on real inputs: in our runs, removing the components left little of the verdict's separation, which reads as "the components are necessary, $Delta$ carries nothing", while the $Delta$-swap shows that $Delta$ does carry the value (@app-clamp).

*Test 2* checks a failure that test 1 cannot see. Since $C + D = y$ on every input, a decomposition can give its components any write $C$ as long as $Delta$ writes $y - C$. Such components can move the verdict when swapped without being anything the model computes. The test compares $C$ with a part of the *original* matrix that has $C$'s directions but none of its gains: per layer, with $Q_U$ an orthonormal basis of the span of the 96 write vectors $U_c$, the part $Q_U Q_U^top y$. Inside this span the model's own twin difference is $C$'s plus the remainder's: $Q_U Q_U^top delta y = delta C + Q_U Q_U^top delta D$. If $C$ is a part of the model's output, the remainder adds other content in these directions but does not systematically undo $delta C$, and the two swaps move the verdict about equally. If the write-span swap moves it much less, the remainder writes against $C$ in $C$'s own directions: the verdict effect of the $C$-swap comes from a write that the model, as a whole, does not have. We call this a *re-implementation*. The comparison needs no threshold: both are swaps of the same kind, read by the same probe. (Both are unclamped.)

*Test 3* catches a consequence of re-implementation that is visible from the outputs alone. Twins differ in content as well as in truth, so a swap legitimately changes many answers: the site swap brings over much of the twins' differences in word count, recall and other ratings (@app-quality). The test therefore uses the *bias*, the mean change of a readout over all swapped rows. Each pair is swapped in both directions, so whatever is brought over from the twin cancels in the mean, and what remains is a shift in one direction: damage. It is read against the site swap's bias, and only on readouts that are not truth readouts (under the truth readouts even faithful parts shift the score slightly towards "false"; @app-quality).

*Test 4* is where "targeted" is tested. The truth tests cannot show it: a decomposition of the whole layer output passes them. On Pile text there is no truth judgement to make, so whatever removing $C$ costs there is computation that the verdict was not the reason for. We use 256 Pile validation windows of 64 tokens, with $C$ removed at every position ($Delta$ kept).

*Test 5* asks whether the method finds one answer. Two seeds that found the same mechanism write the same thing on the same input, whatever their individual components look like, so we compare them as functions: the cosine between their twin differences $delta C$, pooled over the object and end tokens and the seven layers. No matching of components is needed.

== Baseline <baseline>

A part of the model's output can be fitted without any decomposition. Per layer, the top 96 principal directions of the site output's difference between twins of the training split give a subspace $Q$ of the same size as $C$; the baseline split is $C_"base" = Q Q^top y$ and the rest. It is a part of the model by construction (test 2 does not apply), and it is supervised by which statements are twins, so it is a reference for what a subspace of this size can carry, not a competing method.

= Results <results>

All results are on the test split, one run per cell, with 95% bootstrap intervals over pairs (1,000 draws) where shown. Each decomposition is reported separately; nothing is pooled over seeds. Tests 1 and 3 were run on an RTX 4090, test 2 partly on an H100 (the GPUs keep slightly different twin sets, 898 and 906 rows, because a few near-tie verdicts differ under their arithmetic; @app-facts).

== Overview <res-overview>

#figure(
  tbl(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    [test], [plain (s1, s2)], [recon (s1, s2)], [ortho (s1–s4)], [baseline],
    [1. clamped $C$ / $Delta$ swap (site 0.94)], [0.63–0.66 / 0.32–0.35], [0.86–0.89 / 0.09–0.10], [*0.11–0.33 / 0.61–0.84*], [0.94 / 0.01],
    [2. write-span swap against $C$-swap], [*0.01–0.02 against 0.73*], [0.88 against 0.88], [0.32–0.43 against 0.28–0.61], [n/a],
    [3. object-recall bias, clamped (site 0.00)], [*−0.63 to −0.68*], [0.00 to +0.01], [−0.01 to +0.01], [0.00],
    [4. Pile cost of removing $C$], [0.11], [*0.19*], [0.03], [–],
    [5. seed agreement], [*0.02*], [0.82], [*0.07–0.11*], [n/a],
  ),
  caption: [Each method against the five tests, ranges over seeds. Bold: the cell where the method fails. No pass threshold was fixed before the data; "fails" is our reading of each cell against the site swap, the baseline and the other methods.],
) <tab-overview>

#finding(title: "Each method fails a different test")[
  - *Plain tPD re-implements.* Its components move the verdict, but the model's own output in the same directions does not (test 2). The mismatch shows in other answers: swapping either part erases the recall of the statement's object in the number comparisons (test 3). Its two seeds write unrelated things (test 5).
  - *Recon is faithful and carries most of the truth value*, the best of the three on test 1, and it passes tests 2, 3 and 5. But removing its components costs 19% of everything the seven layers do on ordinary text (test 4): it decomposes what the layers do on statements, of which the truth value is a part.
  - *Ortho stays targeted* (test 4) and does no damage (test 3), but its components carry only 0.11–0.33 of the truth value and $Delta$ the rest (test 1), and its four seeds write different things (test 5).
  - *A fitted subspace does better.* The top 96 directions per layer of the output's twin differences carry the whole site swap with nothing left in the rest (0.94 / 0.01).
]

== Test 1: where is the truth value? <res-test1>

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    [split], [$C$ clamped], [$Delta$ clamped], [sum], [Yes/No 1: $C$ / $Delta$], [$C$ by group: affirm. / negated / compar.], [$Delta$ by group],
    [site (unclamped)], [0.94 [0.94, 0.95]], [–], [–], [0.91 / –], [0.86 / 0.89 / 0.99], [–],
    [baseline], [0.94 [0.94, 0.95]], [0.01 [0.01, 0.02]], [0.96], [0.90 / 0.01], [0.85 / 0.88 / 1.00], [0.01 / 0.04 / 0.00],
    [plain s1], [0.66 [0.65, 0.68]], [0.32 [0.31, 0.33]], [0.99], [0.72 / 0.25], [0.72 / 0.58 / 0.67], [0.22 / 0.38 / 0.34],
    [plain s2], [0.63 [0.62, 0.64]], [0.35 [0.34, 0.37]], [0.98], [0.70 / 0.26], [0.71 / 0.64 / 0.60], [0.24 / 0.33 / 0.40],
    [recon s1], [0.86 [0.85, 0.87]], [0.10 [0.09, 0.11]], [0.96], [0.70 / 0.19], [0.79 / 0.95 / 0.86], [0.08 / −0.03 / 0.15],
    [recon s2], [0.89 [0.87, 0.90]], [0.09 [0.07, 0.10]], [0.97], [0.73 / 0.17], [0.79 / 0.96 / 0.89], [0.07 / −0.03 / 0.13],
    [ortho s1], [0.11 [0.10, 0.12]], [0.84 [0.82, 0.85]], [0.95], [0.16 / 0.70], [0.12 / 0.04 / 0.13], [0.68 / 0.90 / 0.87],
    [ortho s2], [0.33 [0.31, 0.36]], [0.61 [0.58, 0.63]], [0.94], [0.21 / 0.64], [0.13 / 0.59 / 0.32], [0.69 / 0.31 / 0.68],
    [ortho s3], [0.14 [0.13, 0.16]], [0.80 [0.78, 0.82]], [0.94], [0.13 / 0.73], [0.11 / 0.21 / 0.13], [0.75 / 0.69 / 0.85],
    [ortho s4], [0.19 [0.18, 0.21]], [0.75 [0.73, 0.76]], [0.94], [0.20 / 0.68], [0.15 / 0.27 / 0.18], [0.67 / 0.62 / 0.82],
  ),
  caption: [Test 1: swap fraction $nu$ under truth rating 1 (ideal: $C$ at the site's 0.94, $Delta$ at 0), 898 twin rows. Clamped: the other part held at its value on the original statement. Sum: $C$ + $Delta$ clamped, against the site's 0.94. Yes/No 1: the same clamped swaps under a probe no decomposition was trained on (site 0.91). Groups: affirmative facts 264 rows, negated statements 238, comparisons 396.],
) <tab-test1>

- *The clamped parts share out the site swap.* $C$ and $Delta$ add up to 0.94–0.99 against the site's 0.94 (Yes/No 1: 0.85–0.97 against 0.91), so the two numbers are a split of one effect.
- *Recon carries the most*: $C$ 0.86–0.89, $Delta$ 0.09–0.10, in every group (0.79 affirmative, 0.95–0.96 negated, 0.86–0.89 comparisons). Under Yes/No 1 $C$'s share drops to 0.70–0.73, and $Delta$ carries 0.17–0.19.
- *Plain splits it about two to one* (0.63–0.66 / 0.32–0.35). Test 2 shows that $C$'s share here is not a part of the model's computation.
- *Ortho leaves most of it in $Delta$* (0.61–0.84). Its components carry 0.11–0.33, little in every group; only seed 2 carries more for negated statements (0.59).
- *The baseline isolates it*: 0.94 / 0.01, at the same size as $C$.

== Test 2: is $C$ a part of the model's output? <res-test2>

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    [], [plain s1], [plain s2], [recon s1], [recon s2], [ortho s1], [ortho s2], [ortho s3], [ortho s4],
    [$C$-swap, truth rating 1], [0.73], [0.73], [0.88], [0.88], [0.47], [0.61], [0.28], [0.39],
    [write-span swap, truth rating 1], [*0.02*], [*0.01*], [0.88], [0.88], [0.39], [0.43], [0.32], [0.35],
    [$C$-swap, Yes/No 1], [0.77], [0.80], [0.72], [0.70], [0.35], [0.24], [0.14], [0.29],
    [write-span swap, Yes/No 1], [*0.02*], [*0.02*], [0.74], [0.72], [0.31], [0.25], [0.26], [0.30],
    [GPU (twin rows)], [H100 (906)], [4090 (898)], [4090], [4090], [H100], [H100], [H100], [4090],
  ),
  caption: [Test 2: swap fraction $nu$ of the $C$-swap and of the swap of the original layer output inside the span of $C$'s 96 write vectors per layer (both unclamped). Control: a random subspace of 96 dimensions per layer gives 0.01. Ideal: the two swaps about equal.],
) <tab-test2>

- *Plain fails in both seeds:* its components move the verdict (0.73), the model's own output in the same 96 directions per layer does not (0.01–0.02, the level of a random subspace). Inside the layers this is visible directly: all components together write 1.9 (before the object) to 16 (at the end token) times the output's squared norm, almost orthogonal to it, and $Delta$ writes the negative (@app-plain).
- *Recon passes:* 0.88 against 0.88, and under Yes/No 1 0.72–0.74 against 0.70–0.72. A trained decomposition can pass this test.
- *Ortho passes* in every seed (0.32–0.43 against 0.28–0.61). Its full component sum is a part of the output, as its loss asks.

== Test 3: damage <res-test3>

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto),
    [split], [object recall: $C$], [object recall: $Delta$], [comparisons only: $C$ / $Delta$], [word count: $C$], [word count: $Delta$],
    [site (unclamped)], table.cell(colspan: 2)[+0.00 [−0.00, 0.01]], [−0.00], table.cell(colspan: 2)[+0.09 [0.07, 0.11]],
    [baseline], [+0.00 [−0.00, 0.01]], [−0.00 [−0.01, −0.00]], [−0.00 / −0.00], [+0.09 [0.07, 0.11]], [+0.02 [0.00, 0.03]],
    [plain s1], [*−0.68* [−0.71, −0.65]], [*−0.65* [−0.68, −0.62]], [*−0.97 / −0.97*], [*−0.49* [−0.53, −0.45]], [*−0.53* [−0.58, −0.48]],
    [plain s2], [*−0.63* [−0.66, −0.59]], [*−0.65* [−0.68, −0.62]], [*−0.98 / −0.96*], [*−0.53* [−0.59, −0.48]], [*−0.65* [−0.71, −0.60]],
    [recon s1], [+0.01 [0.00, 0.01]], [+0.00 [−0.00, 0.00]], [0.00 / −0.00], [+0.05 [0.04, 0.07]], [−0.02 [−0.04, −0.01]],
    [recon s2], [+0.00 [−0.00, 0.01]], [+0.00 [−0.00, 0.00]], [0.00 / −0.00], [+0.04 [0.03, 0.06]], [−0.01 [−0.03, 0.01]],
    [ortho s1], [+0.00 [0.00, 0.01]], [−0.01 [−0.01, 0.00]], [−0.01 / −0.00], [−0.05 [−0.06, −0.03]], [+0.03 [0.01, 0.06]],
    [ortho s2], [+0.01 [0.00, 0.01]], [−0.01 [−0.01, 0.00]], [−0.00 / −0.01], [−0.05 [−0.07, −0.04]], [−0.05 [−0.07, −0.03]],
    [ortho s3], [−0.00 [−0.01, 0.00]], [−0.01 [−0.02, −0.00]], [0.00 / −0.00], [−0.02 [−0.04, −0.01]], [+0.01 [−0.01, 0.03]],
    [ortho s4], [+0.00 [−0.00, 0.01]], [−0.01 [−0.02, −0.01]], [−0.00 / −0.00], [−0.07 [−0.08, −0.06]], [−0.02 [−0.05, −0.00]],
  ),
  caption: [Test 3: bias (mean change over all 898 swapped rows) under the clamped swaps. Object recall: probability of repeating the statement's own object, original mean 0.937. Word count: the rated number of words, original mean 6.01, sd 0.90. Since each pair is swapped in both directions, what is brought over from the twin cancels; the site swap's bias is the reference. Subject recall shows the same pattern, weaker (plain −0.04 to −0.06, the others within ±0.01).],
) <tab-test3>

- *Plain does gross damage.* Under either swap the recall of the statement's own object falls by 0.63–0.68 from 0.94; in the number comparisons it falls to about zero. The word-count rating drops by half a word or more (site +0.09). This is what re-implementation predicts: $C$ from one statement and $Delta$ from the other no longer cancel, and the leftover write is several times the size of the output.
- *Recon and ortho are clean* on object recall (within 0.01 of zero) and within 0.16 words of the site on word count.

== Test 4: off-target cost <res-test4>

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    [], [plain s1], [plain s2], [recon s1], [recon s2], [ortho s1], [ortho s2], [ortho s3], [ortho s4],
    [$C$ removed: share of the site removal's KL], [0.112], [0.113], [*0.191*], [*0.189*], [0.028], [0.025], [0.025], [0.027],
    [mean $|C|^2 \/ |y|^2$ on the Pile], [0.73], [0.72], [0.08], [0.08], [0.12], [0.12], [0.12], [0.12],
  ),
  caption: [Test 4 on 256 Pile validation windows of 64 tokens, $C$ removed at every position. Removing all seven outputs costs a next-token KL of 0.742 nats; the first row is the share of that.],
) <tab-test4>

- *Recon's components cost 19% of the layers' whole effect on ordinary text*, although they write only 8% of the output's energy there; at the statement tokens they write 76–88% of it (@app-recon). Recon is targeted at statements as an input distribution and rebuilds what the layers do on such sentences, of which the truth judgement is one part.
- *Ortho's cost is small* (0.025–0.028).
- *Plain's components write 72–73% of the output's energy on the Pile too*, cancelled by $Delta$; removing them costs 0.11.

== Test 5: do two seeds find the same thing? <res-test5>

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    [], [plain s1], [plain s2], [recon s1], [recon s2], [ortho s1], [ortho s2], [ortho s3], [ortho s4],
    [plain s1], [1], [0.02], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
    [plain s2], [], [1], [0.01], [0.01], [0.02], [0.01], [0.02], [0.02],
    [recon s1], [], [], [1], [*0.82*], [0.16], [0.16], [0.14], [0.17],
    [recon s2], [], [], [], [1], [0.16], [0.16], [0.14], [0.17],
    [ortho s1], [], [], [], [], [1], [0.08], [0.08], [0.11],
    [ortho s2], [], [], [], [], [], [1], [0.10], [0.07],
    [ortho s3], [], [], [], [], [], [], [1], [0.08],
    [ortho s4], [], [], [], [], [], [], [], [1],
  ),
  caption: [Test 5: $cos(delta C_i, delta C_j)$ of the twin differences of all components' writes, pooled over the object and end tokens and the seven layers (906 twin rows, H100).],
) <tab-test5>

- *Recon's seeds agree* (0.82).
- *Plain's and ortho's do not* (0.02; 0.07–0.11). For ortho the raw writes of the active sets fall into two groups (seeds 1 and 4, seeds 2 and 3; @app-ortho), but on the twin differences, which carry the truth value, no pair agrees.

= Discussion <discussion>

== What each method's failure means

*Plain tPD.* Nothing in plain tPD's losses asks the components to be a part of what the layers write. The verdict loss only needs the components to produce the verdict, and $Delta$ can absorb anything, because its mask is random and the Pile loss keeps it on. The decompositions use this freedom: the components write vectors several times the size of the output, nearly orthogonal to it, and $Delta$ writes their negative. A truth readout cannot tell this apart from a decomposition. Plain's $C$-swap carries 0.73 of the verdict, more than ortho's, and the scaling of a decomposition into a re-implementation is invisible to it (@app-validate). The failure shows only in tests anchored in the original model (test 2), in other answers (test 3), and in the arbitrariness of the solution (test 5).

*Recon.* The reconstruction term fixes this: the components are a faithful part of the output (test 2), the seeds agree (test 5), and most of the truth value is in them (test 1). The price is targeting. The term weights every direction of the output by its size, so the components rebuild most of the output at the statement tokens, whatever its role. What makes recon targeted at all is the Pile stream: on ordinary text its components are quiet. So recon finds the computation of the layers *on statements*, not the truth judgement within it. The truth value sits in that computation, but test 1 alone cannot show this restriction to the computation on statements. Test 4 shows it.

*Ortho.* Orthogonality of the full component sum prevents re-implementation (test 2) and keeps the components small (test 4), as intended. But the term does not say which part of the output the components should carry, and in our runs they carry a minority of the truth value: most of it stays in $Delta$, and the swapped components move the verdict mainly because the recomputed $Delta$ at later layers follows them (unclamped 0.28–0.61, clamped 0.11–0.33; @app-clamp). The assumption behind the method, that the verdict-relevant part of the output is orthogonal to the rest, gives the optimiser no reason to move that part into the components. The seeds land on different solutions.

*The baseline.* A subspace of 96 directions per layer, fitted to twin differences, carries the whole site swap and leaves nothing in the rest. So a part of these layers' output with this property exists, at the size of $C$, and none of the three methods finds it. The baseline is supervised by which statements are twins, and it is a subspace of the output, not a mechanism: it says nothing about how the layers compute what it carries. It also mixes truth and content: in negated statements it brings over the content-linked differences between twins along with the verdict (@app-negcities).

== What we learned about the tests

- *Removals mislead.* Removing the components leaves little of the verdict and suggests that $Delta$ carries nothing; the $Delta$-swap shows that it does. A removal tests whether a part is used, in a state the network never reaches; a swap tests whether it carries the value.
- *Unclamped swaps are not additive.* The recomputed part reads the swapped part's earlier writes. Clamping gives each part's own share; the unclamped numbers mix it with the other part's response, and per family they give the opposite picture (@app-clamp).
- *A truth readout rates the re-implementer well.* Every test read only by truth questions, swap or shift, rates plain tPD at least as well as the others. Tests anchored in the original model (test 2) or in other answers (test 3) are needed.
- *"Other qualities unchanged" is the wrong criterion.* Many answers we intended as truth-independent differ between true and false statements in the original model, some because of the truth value and some because of the content (@app-quality). A swap must move them as far as the site swap does. The bias, which cancels whatever is brought over from the twin, isolates damage.
- *Seed agreement as functions is cheap and decisive.* One forward pass and a cosine separate "the same thing found twice" from "two unrelated answers".

== Limits <limits>

- *One model, one task, one run per cell.* The intervals cover the sampling of statements only. Plain and recon have two seeds, ortho four.
- *Components at full strength.* Every test uses all components at mask 1. The method scales each component by its CI per token, and a test 1 with CI-weighted components (what the CI network picks out per statement) was not run.
- *In-distribution twins of equal length.* 898 rows, weighted towards cities and number comparisons; no translation families, no out-of-distribution statements.
- *No pass thresholds were fixed in advance.* The reading "fails" in @tab-overview is ours.
- *Two GPUs.* Tests 1 and 3 ran on an RTX 4090, test 2 partly and tests 4–5 entirely on an H100; their twin sets differ by four pairs.
- *The decomposed sites.* Only `down_proj` of layers 11–17 at the statement tokens. The truth value may also travel by routes outside them; reading every result against the site swap accounts for this.

#pagebreak()

= Appendix: further findings <appendix>

== The active set is not a self-contained part <app-active>

For each decomposition the *active set* $S$ is the smallest top-$m$ set of components, ranked by mean CI on validation statements, whose CI-weighted components-only state recovers 95% of what all components recover: plain 27 and 44 components, recon 387 and 432, ortho 29, 31, 58 and 38. "Inactive" below means the components outside $S$.

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    [run], [all at CI], [only $S$ at CI], [mean mask], [outside $S$ removed: separation share], [1→0 flips], [0→1 flips],
    [plain s1], [0.961], [0.909], [0.964], [0.446], [0.045], [0.327],
    [plain s2], [0.958], [0.915], [0.964], [0.405], [0.113], [0.310],
    [recon s1], [0.974], [0.928], [0.974], [0.675], [0.193], [0.017],
    [recon s2], [0.971], [0.934], [0.973], [0.830], [0.055], [0.036],
    [ortho s1], [0.906], [0.868], [0.959], [0.602], [0.122], [0.031],
    [ortho s2], [0.893], [0.861], [0.960], [0.682], [0.014], [0.216],
    [ortho s3], [0.875], [0.832], [0.948], [0.735], [0.160], [0.160],
    [ortho s4], [0.903], [0.860], [0.959], [0.671], [0.092], [0.016],
    [untrained s1 / s2], [−0.002 / −0.003], [–], [0.006 / 0.049], [–], [–], [–],
  ),
  caption: [Active against inactive components (test split, truth rating 1, all 1,794 statements). Left three columns: KL recovered with $Delta$ off, components at their CI, only the active set at its CI, or all at the training's average mask $upright("CI") + (1 - upright("CI")) \/ 2$. Right three: the full model ($Delta$ kept) with every component outside $S$ removed (mask 1 → 0); separation share and flips macro over families. Ortho s3 and s4 ran on H100s, the others on an H200. Untrained: a run's checkpoint at step 0.],
) <tab-app-active>

- *Removing the inactive components costs 17–60% of the separation in every method.* A decomposition whose CI values were honest would lose nothing. Plain's removal pushes false statements to "true" (0→1 flips 0.31–0.33).
- *At their CI they add little:* all components at CI recover 0.03–0.05 more than the active set alone. Their role is as part of the full sum at mask 1: in plain they are part of the large writes that $Delta$ cancels, in recon real parts of the output, in ortho a counterweight to the active set.
- *Ortho's inactive components cancel its active set.* At mask 1 the active set's raw write is 60–87% cancelled in the output, by the other components and not by $Delta$ (seed 1: $chevron.l p, "other components" chevron.r \/ |y|^2$ from −0.13 to −0.32 per position class, $chevron.l p, D chevron.r \/ |y|^2$ from −0.001 to 0.02). Its active set's swap moves the verdict by 0.31–0.67, the model's own output in the active set's write span by 0.03–0.11: for the active set alone, ortho would fail test 2.
- *The training's average mask helps ortho* (+0.05 to +0.07 KL recovered), which fits the idea that the components rely on the inactive ones' random contributions during training. The evidence is weak: an untrained decomposition gains 0.049 too, and the mean mask also raises active components whose CI lies strictly between 0 and 1, which in ortho are at least as common as CI values of exactly 1.

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    [decomposition], [$|S|$], [$S$-swap], [rest-swap], [$S$-swap, rest held], [rest-swap, $S$ held], [write span of $S$, swap],
    [plain s1], [27], [0.92], [0.54], [0.65], [0.32], [0.00],
    [plain s2], [44], [0.85], [0.51], [0.70], [0.26], [0.00],
    [recon s1], [387], [0.79], [0.41], [0.75], [0.19], [0.80],
    [recon s2], [432], [0.81], [0.33], [0.81], [0.14], [0.83],
    [ortho s1], [29], [0.31], [0.68], [0.12], [0.80], [0.03],
    [ortho s2], [31], [0.67], [0.47], [0.34], [0.60], [0.11],
    [ortho s3], [58], [0.40], [0.65], [0.21], [0.72], [0.10],
    [ortho s4], [38], [0.43], [0.62], [0.23], [0.72], [0.04],
  ),
  caption: [The swap tests with $P$ = the active set $S$ and the rest = everything else (H100, 906 twin rows, truth rating 1; site swap 0.93). With the active set, every method has a rest that carries a large part of the truth value, and ortho fails the write-span test.],
) <tab-app-active-swap>

== Clamped against unclamped swaps; removals <app-clamp>

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto),
    [split], [$C$-swap], [$Delta$-swap], [sum], [$C$-swap by group: affirm. / negated / compar.],
    [site], [0.94], [–], [–], [0.86 / 0.89 / 0.99],
    [baseline], [0.89], [0.03], [0.92], [0.64 / 0.88 / 0.98],
    [plain s1 / s2], [0.73 / 0.73], [0.56 / 0.52], [1.29 / 1.25], [0.71, 0.76 / 0.77, 0.68 / 0.72, 0.73],
    [recon s1 / s2], [0.88 / 0.88], [0.29 / 0.26], [1.17 / 1.14], [0.52, 0.50 / 0.89, 0.91 / 0.99, 0.99],
    [ortho s1 / s2 / s3 / s4], [0.46 / 0.61 / 0.28 / 0.39], [0.69 / 0.43 / 0.47 / 0.66], [1.15 / 1.04 / 0.75 / 1.05], [0.10, 0.09, 0.01, 0.07 / 0.14, 0.65, 0.34, 0.41 / 0.70, 0.77, 0.35, 0.50],
  ),
  caption: [Unclamped swaps (the other part recomputed), truth rating 1, same 898 rows as @tab-test1.],
) <tab-app-unclamped>

- *Unclamped, the two parts do not add up*: their sum ranges from 0.75 to 1.29 against the site's 0.94. Clamped it is 0.94–0.99.
- *Clamping reverses the per-group picture.* Unclamped, recon's $C$ carries only 0.50–0.52 for affirmative facts and ortho's $C$ 0.35–0.77 for comparisons, which reads as "the components carry the truth value of comparisons and negations, not of plain facts". Clamped, recon's $C$ carries 0.79 for affirmative facts, and ortho's $C$ carries little in every group (@tab-test1). The unclamped numbers mix in how $Delta$ at later layers responds to the swapped writes.
- *Removals.* With the active set removed from the full model, the rest alone leaves little of the separation (at most 0.31 in our runs), while the rest's swap moves the verdict by 0.33–0.68 (@tab-app-active-swap). The removal state is far from any the network reaches on real inputs; the verdict breaks for that reason, not because the rest lacks the truth value.

== Inside plain tPD <app-plain>

- *Raw writes.* All components together write $|C|^2 \/ |y|^2$ = 1.9 before the object, 7.8–8.2 at the object and 14.8–16.2 at the end token, with $chevron.l C, D chevron.r \/ |y|^2$ equal to minus the same number: $chevron.l C, y chevron.r approx 0$.
- *Twin differences.* Only 0.002–0.003 of $C$'s twin difference is present in the model's ($chevron.l delta C, delta y chevron.r \/ |delta C|^2$); $|delta C|^2$ is 7.9–8.4 times $|delta y|^2$.
- *Weight space, no data.* $chevron.l P, W chevron.r_F \/ |P|_F^2$ is 0.000 in every layer for the sum of all components $P$, and $|P|_F^2$ is 0.95–0.97 of $|W|_F^2$: the remainder is the original matrix minus a matrix of its own size that the layer does not have.

== Inside recon <app-recon>

- *Faithful on the write side:* 1.05 of $C$'s twin difference is present in the model's; $|C|^2 \/ |y|^2$ is 0.76–0.88 at the statement tokens and 0.08 on the Pile.
- *Read side undetermined.* For the active sets, swapping what $W$ itself writes in response to the input directions the components read moves the verdict by only 0.13 / 0.16, and the two seeds' read subspaces overlap by 0.06 while their write subspaces overlap by 0.59 (share of the smaller subspace; mean over layers). The two seeds write nearly the same thing from nearly disjoint input directions: on the statements, a read vector only has to correlate with the pattern the matrix responds to. The components say what the layers write, not which inputs make them write it.
- *Weight space:* $chevron.l P, W chevron.r_F \/ |P|_F^2$ is 0.10–0.18 per layer. Outside the directions the statements excite, the component matrices are unconstrained by any loss.

== Inside ortho <app-ortho>

- The full component sum is orthogonal to $Delta$'s write ($chevron.l C, D chevron.r \/ |y|^2$ between −0.010 and 0.001), as trained.
- On the twin differences only 0.27–0.29 of $delta C$ is present in the model's difference: the raw writes are dominated by what twins share, and that is all the term constrains.
- *Two groups of seeds in the raw writes.* The active sets' raw writes agree between seeds 1 and 4 (cosine 0.48) and between seeds 2 and 3 (0.42), and 0.02–0.12 across the groups. On the twin differences of all components (test 5) no pair exceeds 0.11.

== Other-quality readouts depend on truth and on content <app-quality>

In the original model, many answers we had intended as truth-independent differ between true and false twins. The negated cities separate truth from content: they use the same cities and countries as the affirmative cities, but "The city of Lu'an is not in Turkey." is true and names the wrong country.

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto),
    [readout], [cities], [neg_cities], [larger_than], [smaller_than], [follows],
    [verdict (truth rating 1)], [+1.55], [+1.62], [+2.55], [+1.79], [truth],
    [emotion rating], [+2.69], [−0.01], [+1.19], [+0.80], [truth, affirmative only],
    [surprise rating], [+0.39], [+0.38], [+0.34], [+0.16 (none)], [truth (small)],
    [humour rating], [+0.41], [−0.79], [+0.56], [+0.47], [content],
    [word count], [+0.60], [−0.99], [+1.10], [−0.12], [content],
    [subject recall], [−0.76], [+0.89], [−0.96], [−0.66], [content],
  ),
  caption: [Mean gap between the true and the false twin in the original model, in standard deviations of the readout over the 906 twin rows (H100). A sign that flips from cities to negated cities means the readout follows whether the named country is the city's own, not whether the statement is true.],
) <tab-app-quality>

- A truth swap must therefore move these readouts too, by as much as the site swap moves them; "other qualities unchanged" cannot be met even by a perfect decomposition. Test 3's bias handles this, since it cancels whatever is brought over from the twin.
- Under the truth readouts, the faithful parts are the ones with a bias: recon's active set shifts the verdict score slightly towards "false" when swapped, as does the baseline. A state mixed from two statements may read as false #assumed. So a bias counts as damage only on readouts that are not truth readouts.
- The scaling dial (@app-validate) confirmed that these readouts react when a part's edit is inflated.

== Does any part separate truth from content? The negated cities <app-negcities>

Criterion, fixed before the clamped runs were read: in `neg_cities`, under the clamped swaps, $C$ brings over at least half of the verdict's true–false gap, and of the humour, word-count and subject-recall gaps a share whose interval includes 0 or lies below it, while $Delta$ brings over the larger share of each.

#figure(
  tbl(
    columns: (auto, auto, auto, auto, auto, auto),
    [split], [verdict: $C$], [humour: $C$ / $Delta$], [word count: $C$ / $Delta$], [subject recall: $C$ / $Delta$], [holds],
    [site], [0.89], [1.01], [0.51], [0.41], [–],
    [baseline], [0.87], [1.01 / 0.03], [0.44 / 0.08], [0.39 / 0.00], [no],
    [plain s1 / s2], [0.58 / 0.63], [−0.23, −0.04 / 1.10, 1.07], [−0.34, 1.77 / 0.67, −1.40], [0.53, 0.81 / 0.30, −0.02], [no],
    [recon s1 / s2], [0.96 / 0.98], [0.17, 0.13 / 0.90, 0.94], [−0.40, −0.00 / 0.83, 0.42], [0.34, 0.42 / 0.02, −0.14], [no],
    [ortho s1–s4], [0.04, 0.57, 0.22, 0.28], [0.12–0.22 / 0.79–0.95], [−0.02–0.58 / −0.08–0.42], [0.13–0.33 / 0.12–0.34], [no],
  ),
  caption: [Share of each readout's true–false gap that the clamped swap brings over, `neg_cities` (206 rows, RTX 4090).],
) <tab-app-negcities>

No split meets the criterion. Recon's $C$ comes closest: it carries the verdict (0.96–0.98) and leaves most of the humour and word-count gaps to $Delta$, but brings over 0.13–0.17 of the humour gap (interval above 0) and the subject-recall gap goes with it. The baseline, fitted to twin differences, carries the content-linked gaps along with the verdict, as the site does.

== Smaller baselines <app-baseline>

The twin-difference subspace at the sizes of the active sets (H100, unclamped swaps, truth rating 1): with plain s1's per-layer sizes (27 directions) the part / rest swaps give 0.76 / 0.56, with ortho s1's (29) 0.80 / 0.37, with recon s1's (387) 0.86 / 0.05. The input principal directions at recon s1's size give 0.70 / 0.62, and random subspaces 0.00 at every size. So a small truth-specific subspace exists in the output, but at small sizes it does not empty the rest either.

== Validating the readouts <app-validate>

*Site versions.* Every truth readout moves under the site swap (truth rating 1: 0.93, truth rating 2: 0.94, Yes/No 1: 0.95 on the H100), so all truth probes read what the sites write in the same way. The site swap leaves the word-count answer almost in place (digit KL 0.08).

*Seeded defects.* For any decomposition, $W_c |-> (1 + k) W_c$ and $Delta |-> Delta - k sum_c W_c$ leave the model unchanged and turn the components into a re-implementation of strength $k$; every test on the components then sees $(1 + k)$ times their own edit. On seed 1 of each method (active set, $k = 0, 1, 3, 10$):
- the word-count KL under the components' swap rose with $k$ in all three, to 0.22–0.79 at $k = 10$ against the site swap's 0.081, and subject recall fell by 0.34–0.39;
- the swap read by truth rating 1 stayed at or below the site's 0.93 at every $k$, so a truth readout alone does not notice the defect.

== Run facts <app-facts>

- *Decompositions* (WandB project `bitt-j-personal/spd`): plain #runid("s-7152548c"), #runid("s-46efc525")\; recon #runid("s-eeac663f"), #runid("s-edc17041")\; ortho #runid("s-0c2fa0d7"), #runid("s-4186f973"), #runid("s-6acf4d80"), #runid("s-22600dd8") (seed 4, trained on an H100 with the finals' configuration; checkpoint not on WandB).
- *GPUs.* Tests 1 and 3 and the write-span cells of plain s2, recon and ortho s4: RTX 4090, 898 twin rows (449 pairs). Test 2 for plain s1 and ortho s1–s3, tests 4 and 5, and the active-set tables: H100, 906 rows. Twins are kept only where the model's own verdicts differ, and four near-tie pairs come out differently under the two GPUs' bf16 arithmetic. @tab-app-active: H200 (ortho s4: H100).
- *Code* (`spd/experiments/lm/honesty_targeted_decomposition/`): `paper_tests.py` (swaps, clamped swaps, write-span and twin-difference targets), `paper_interventions.py`, `paper_inspect.py` (twin differences, seed agreement), `paper_pile.py` (test 4), `paper_recovery.py` (@tab-app-active).
