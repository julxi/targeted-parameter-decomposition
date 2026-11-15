#import "template.typ": *
#import "26-11-15_method_flaws_tables.typ": *

// Agent-led research report (topic convos/julian/26-11-15_open_research_LOG.md). The tables are
// generated from the run outputs by ~/spd_out/26-11-15_open_research/analysis/report_tables.py
// (file 26-11-15_method_flaws_tables.typ); the prose quotes them.

#show: report.with(
  title: "Where the three targeted decompositions fail",
  subtitle: "Swap tests on both sides of each split, model-anchored versions of each component set, and agreement between seeds, for plain tPD, recon and ortho on the truth verdict of Qwen2.5-7B-Instruct",
  date: "2026-11-15",
  author: "Research and text by Claude Fable 5.2, on an open brief from Julian Bitterlich",
  summary: [
    - *Question.* The paper draft compares three targeted parameter decompositions of the MLP outputs of layers 11–17 (plain tPD, recon, ortho) by asking whether a set of components $P$ captures the model's truth value. This report runs the draft's swap tests on both sides of every split ($P$ and the rest $R$) for all final decompositions, adds tests of whether $P$'s writes are something the model itself writes, and compares seeds as functions. One GPU session, test split, 906 statements with a twin.
    - *No decomposition isolates the truth value.* Swapping the rest between a statement and its twin moves the verdict in all eight decompositions (swap fraction 0.33–0.68; ideal 0). With the other part held fixed, the two effects add up to the swap of the whole outputs, and the components' own share is 0.65–0.81 for plain and recon and 0.12–0.34 for ortho. Removal tests had suggested the opposite ("the remainder carries nothing").
    - *Plain tPD re-implements.* Its components write vectors the layers' outputs do not contain (0.4% of their twin difference is present in the model's), larger than the outputs themselves; swapping the model's own output in their write directions moves the verdict by 0.00, against 0.85–0.92 for the components' swap; two seeds are unrelated (cosine 0.05).
    - *Recon is faithful and broad.* Its components are an orthogonal piece of the output and its seeds agree (0.72), but the piece is 70–79% of the output at every statement token: it rebuilds what the layers do on statements, not the verdict. Its read vectors are not determined (two seeds: write subspaces overlap 0.59, read subspaces 0.06).
    - *Ortho is orthogonal only where it is trained.* All components together are a part of the output, but the active set is cancelled to 60–87% by the other components and fails the write-span test like plain (0.03–0.11 against 0.31–0.67). Its components carry the verdict for negations and comparisons and almost nothing for plain facts (0.02–0.08), and its four seeds form two groups.
    - *A baseline does better.* A subspace of the outputs fitted by PCA on twin differences of the training split, with the size of an ortho active set, carries 0.80 of the swap with smaller side effects than the whole outputs, and is a part of the model by construction.
    - *Tests.* The swap of $P$ read by a truth probe rates the re-implementation best; the write-span swap, the present share and agreement between seeds expose it. Limits: one run per cell, components at full strength (not weighted by causal importance), in-distribution twins only.
  ],
)

= Setting <setting>

This report continues the paper draft "Trying to decompose a truth mechanism with tPD" (`reports/26-11-04_tpd_truth_paper_draft.typ`). Everything needed to read it is repeated here.

*Model, prompt, score.* Qwen2.5-7B-Instruct reads a statement followed by the fixed text `\nThe amount of truth in this statement is␣` and answers with the digit 1 (true) or 0 (false). The *score* of a statement is $s = log P(1) - log P(0)$ at the answer position. The appended text is called a *probe*; this one, used in training, is *truth rating 1*. A second probe with another answer format, *Yes/No 1* (`\nQuestion: Is the statement above true? (Yes/No)\nAnswer:`, scored as $log P("Yes") - log P("No")$), checks whether a result is about the truth value or about the digit. Statements come from ten families of the Truth-is-Universal sets (cities, negated cities, element symbols, animal classes, Spanish–English translations, number comparisons and their negations); the test split has 1,794 statements.

*What is decomposed.* The `down_proj` matrix (the MLP's output projection, 18,944 inputs, 3,584 outputs) of each of the layers 11 to 17, at the statement's tokens only; the probe's tokens always run on the original weights. Each matrix is written as $W = sum_(c=1)^(96) u_c v_c^top + Delta$: 96 rank-1 *components* per layer with read vector $v_c$ and write vector $u_c$, plus a full-rank remainder $Delta$. A network trained with the decomposition gives each component a *causal importance* (CI) in $[0, 1]$ per token. Removing all seven outputs at the statement tokens takes the verdict to chance while the model still answers with a digit, which is why these seven matrices were chosen.

*The three methods.* All are trained so that the model's answer distribution is preserved when each component is scaled anywhere between its CI and 1 and $Delta$ anywhere between 0 and 1, while as few components as possible get a non-zero CI; on non-target text (the Pile) $Delta$ stays on and every non-zero CI is penalised.
- *plain*: only these terms (targeted parameter decomposition as in Vigouroux and Sharkey 2026).
- *recon*: plus a term that makes each matrix's output under the same random masks match its original output at the statement tokens.
- *ortho*: plus a term that makes the write of all components together, $C = sum_c (v_c dot x) u_c$, orthogonal to the write of the remainder, $D = Delta x$, at the statement tokens.
The decompositions studied are the paper's final runs: plain seeds 1 and 2, recon seeds 1 and 2, ortho seeds 1 to 3, and a fourth ortho seed trained for this report with the same configuration (#runid("s-22600dd8"), @res-ortho).

*A split, $P$ and $R$.* At a statement token and a layer, with $x$ the matrix's input and $y = W x$ its output, a component set $S$ gives a part $p = sum_(c in S) (v_c dot x) u_c$ and the rest $r = y - p$; over the seven layers these are the parts $P$ and $R$ of the split. Components enter at full strength (mask 1, not weighted by CI), so $p + r = y$ exactly. Unless stated, $S$ is a decomposition's *active set*: its components ranked by mean CI on the validation statements, cut at the smallest number that reaches 95% of what all components recover of the verdict. $R$ then holds the other components and $Delta$.

*Twins and the interventions.* *Twins* are the true and the false statement about one subject ("The city of Lu'an is in China." / "… in Turkey."). The tests use the #n_twin_rows test statements whose twin has the same number of tokens and gets the opposite verdict from the model, each once as $x$ with its twin as $x'$. An intervention replaces one part $q$ ($p$ or $r$) at every statement token of all seven layers and leaves the other part to be recomputed:
- *swap*: $q$ takes the value it has on the twin. The *swap fraction* $nu = (s_"swap" - s(x)) \/ (s(x') - s(x))$, pooled over statements, is 0 if the score did not move and 1 if it moved all the way to the twin's.
- *removal* (zero), *mean removal* (the training-split mean per token role), *twin average* (the mean of the two twins' values), *fixed shift* (the training-split mean difference between false and true statements, added with the sign that should flip the verdict). For these the measure is the *separation share*: the gap between the mean scores of true and false statements as a fraction of the original gap (1 = untouched, 0 = no separation, below 0 = inverted).
The same intervention on the whole outputs ($P = W$, the *site version*) is the reference: no part can do more. The site swap gives $nu = 0.93$ under truth rating 1 and 0.95 under Yes/No 1.

*What a perfect split would show.* If $P$ carried the truth value and $R$ carried everything else, the $P$-swap would move the verdict as far as the site swap, the $R$-swap would leave it alone, and neither edit of $P$ would change answers that do not depend on truth. The draft's evaluation section derives this from a definition; here it is the yardstick.

= Tests added in this report <tests>

== The clamped swap

In the swap above the other part is recomputed. At later layers it then reads what the swapped part wrote earlier, so the two parts are not independent routes: the measured effect of swapping $R$ includes $P$ following $R$ downstream. The *clamped swap* holds the other part at its clean value on $x$ at every layer, so the seven outputs are exactly $q(x') + [y(x) - q(x)]$ and only the part's own writes act.

== Model-anchored versions of a component set <tests-anchored>

A $P$-swap that moves the verdict shows that $P$'s writes _can_ move it. It does not show that the model writes them. Since $p + r = y$ on every input, a decomposition can give its components any write $c$ as long as the rest writes $y - c$; the unedited model never shows the difference. To test this without trusting the decomposition, each active set is compared with parts of the *original* matrix $W$ that have the set's directions and dimensions but none of its gains. Per layer, let $Q_U$ and $Q_V$ be orthonormal bases of the span of the set's write vectors and of its read vectors:
- *write span*: $P_"w" = Q_U Q_U^top W$, the part of the layer's own output that lies in the directions the set writes to;
- *read span*: $P_"r" = W Q_V Q_V^top$, what the layer itself writes in response to the input directions the set reads;
- controls of the same per-layer size: a *random* output subspace; the top principal directions of the matrix's input (*input PCA*; the draft's "data-aligned basis"); and the top principal directions of the output's difference between twins of the _training_ split (*twin-difference PCA*), the simplest truth-specific subspace one can fit without any decomposition.
Swapping such a part between twins is an interchange intervention on a subspace of the unsplit model.

*How to read the write-span swap.* Inside its own write span the model's twin difference is the set's own plus the rest's: $Q_U Q_U^top delta y = delta p + Q_U Q_U^top delta r$ (with $delta$ the difference between twins). If the set's writes are a part of what the model computes, the rest adds other content in the same directions but does not systematically undo $delta p$, and the write-span swap moves the verdict about as far as the $P$-swap. If the write-span swap moves the verdict much less than the $P$-swap, the rest writes against $P$ in $P$'s own directions: the verdict effect of the $P$-swap comes from a write that the model, as a whole, does not have. That is what the draft calls a re-implementation.

*What it cannot show.* Agreement of the two swaps does not prove that the components are the model's mechanism, and the read-span version is not a requirement at all: a component may legitimately read a direction that is merely correlated, on the statements, with the pattern the matrix actually responds to (@res-recon).

== One inspection pass: cancellation between twins and agreement between seeds

One clean forward pass per twin gives every decomposition's $p$ at every statement token. From the twin differences at the tokens where twins differ (the object and the end token), pooled over the seven layers:
- the *present share* $chevron.l delta p, delta y chevron.r \/ |delta p|^2$: 1 if $P$'s twin difference is an orthogonal piece of the model's, 0 if the model's difference contains none of it (the rest cancels it);
- the energy ratio $|delta p|^2 \/ |delta y|^2$, and $kappa = (|delta p|^2 + |delta r|^2) \/ |delta y|^2$, 1 without cancellation between $P$ and $R$ (proposed in the draft, not computed before);
- the *agreement* of two decompositions, $cos(delta p_i, delta p_j)$ over the same positions. Two seeds that found the same mechanism write the same thing on the same input, whatever their individual components look like.
The same pass gives the raw writes: $|p|^2 \/ |y|^2$ and $chevron.l p, r chevron.r \/ |y|^2$ per token.

== Removal on non-target text

$P$ is removed at every position of #pile_windows validation windows of 64 tokens from the Pile (the runs' non-target data), and the change of the model's next-token predictions (KL from the original, mean over positions) is given as a share of the same removal of the whole seven outputs. The Pile contains no truth judgement, so this share is computation that the verdict was not the reason for. The same share at the statement tokens _before the object_, where a statement and its twin are still identical and truth is not yet determined, is the *prefix share*.

= Results <results>

All numbers are from the test split, one run per cell, on one rented H100; twin tests on the #n_twin_rows twin rows. Intervals in brackets are 95% bootstrap intervals over statements within family. Every table lists each decomposition separately; nothing is pooled over seeds.

== Overview

#figure(
  tab_score,
  caption: [One line per decomposition, training probe (truth rating 1). $|P|$: components in the active set (of 672). $P$-swap, $R$-swap and write-span swap: swap fraction $nu$ (site swap: 0.93; ideal for a split that isolates the truth value: $P$-swap = site swap, $R$-swap = 0, write-span swap ≈ $P$-swap). Present share: $chevron.l delta p, delta y chevron.r \/ |delta p|^2$ over the twin differences at the object and end tokens, all seven layers (1 = an orthogonal piece of the model's difference). Agreement: $cos(delta p_i, delta p_j)$ with each other seed of the same method. Prefix share and Pile share: next-token KL after removing $P$, as a share of the KL after removing the whole seven outputs, at the statement tokens before the object and on #pile_windows Pile windows (0 = $P$ does nothing there).],
) <tab-score>

#finding(title: "Four results")[
  + *No decomposition isolates the truth value.* Swapping the rest $R$ between twins moves the verdict in all eight ($nu$ 0.33–0.68 under the training probe). With the other part held fixed the two effects add up to the site swap, and $P$'s own share is 0.65–0.81 for plain and recon and 0.12–0.34 for ortho (@res-rest).
  + *Plain tPD's components are not part of the layers' computation.* The model's own output inside their write directions carries none of the verdict swap ($nu = 0.00$, the level of a random subspace, against 0.85–0.92 for the $P$-swap), 0.4% of their twin difference is present in the model's, and two seeds have nothing in common (@res-plain).
  + *Recon's components are a faithful part of the output, of nearly everything.* Present share 1.03, write-span swap equal to the $P$-swap, seeds agree; but the active set is 70–79% of the output's energy at every statement token and its removal costs 77–80% of the site removal before the object (@res-recon).
  + *Ortho is orthogonal only where it is trained.* All its components together are a part of the output, but the active set is not: the other components cancel 60–87% of what it writes, and its $P$-swap moves the verdict (0.31–0.67) while the model's own twin difference in the same directions moves it by 0.03–0.11. Its components carry the verdict for negations and comparisons and almost nothing for plain facts, and its four seeds fall into two groups (@res-ortho).
  A subspace of the model's own output with the size of an ortho or plain active set, fitted by a PCA on the twin differences of the training split, carries more of the verdict swap than ortho's components and is part of the model by construction (@res-base).
]

== The rest carries the truth value in every decomposition <res-rest>

#figure(
  tab_swap_tr1,
  caption: [Swap fractions $nu$ under the training probe. "Held": the clamped swap, the other part fixed at its clean value. Write span and read span: the model-anchored versions of the active set (@tests-anchored). All components / $Delta$: the split with $P$ = all 672 components and $R$ = the remainder $Delta$'s write. "–": not run (the last two columns were cut for time for some decompositions).],
) <tab-swap>

#figure(
  tab_swap_yn1,
  caption: [The same under Yes/No 1, a truth probe with another answer format that no decomposition was trained on.],
) <tab-swap-yn>

*The $R$-swap is far from zero.* A split that isolated the truth value would leave the verdict with $x$ when the rest's writes come from the twin. The verdict moves by 0.33–0.68 of the way under the training probe and 0.43–0.78 under Yes/No 1 (@tab-swap, @tab-swap-yn). For ortho seeds 1, 3 and 4 the rest moves it further than the components do.

*This is not an artefact of the components following the rest.* Under the swap as the draft defines it, the part that is not swapped is recomputed, and for plain and recon the two swap fractions sum to far more than the site's 0.93 (plain seed 1: $0.92 + 0.54$). With the other part held fixed they add up to the site swap within 0.04 in all eight decompositions, under both probes: $P$'s own writes carry 0.65–0.70 (plain), 0.75–0.81 (recon) and 0.12–0.34 (ortho) of the swap, the rest's own writes 0.26–0.32, 0.14–0.19 and 0.60–0.80. The clamped pair is the cleaner measure of who carries the value; the recomputed pair shows in addition that each part reads the other's earlier writes.

*Removals had hidden this.* With $P$ removed, the rest alone leaves little of the verdict separation (at most 0.31 here, @tab-other; the draft's analysis T0 reads this as "the components are necessary and the remainder carries nothing"). The swap shows that the rest does carry the value. Removing $P$ breaks the verdict for another reason: the state it produces is far from any the network reaches. The training objective explains why no method prevents a copy in the rest: the verdict must survive when $Delta$ is masked out, so $P$ must suffice; nothing asks that $Delta$'s writes be uninformative.

#figure(
  tab_family,
  caption: [Swap fraction $nu$ per family under the training probe (number of twin rows in brackets; the two animal families have 4 rows each and are noise). Site: all seven outputs swapped.],
) <tab-family>

*What the parts carry depends on the kind of statement* (@tab-family). In the affirmative fact families (cities, element symbols) ortho's $P$-swap is at 0.02–0.08 in every seed and the rest carries the value; ortho's components carry the verdict for comparisons and negated statements, a different subset per seed. Recon's $P$, which is most of the output, carries 0.31–0.42 of the swap for the affirmative facts and about all of it for comparisons. So what these components hold is closer to the handling of negation and comparison than to a truth value that is shared by all statements.

== Plain tPD: components the layers do not contain <res-plain>

#figure(
  tab_raw,
  caption: [Raw writes at the statement tokens of the twin rows, mean over tokens and layers of per-token ratios. $p$: write of the active set at full strength; $r = y - p$; $C$: write of all 96 components per layer; $D = y - C$, the remainder's write. $chevron.l p, r chevron.r = -|p|^2$ means $chevron.l p, y chevron.r = 0$: the output contains nothing of $p$.],
) <tab-raw>

#figure(
  tab_delta,
  caption: [Twin differences at the object and end tokens, pooled over the seven layers. Present share: $chevron.l delta p, delta y chevron.r \/ |delta p|^2$. $kappa = (|delta p|^2 + |delta r|^2) \/ |delta y|^2$. Fourth column: share of the model's twin-difference energy that lies in the active set's write span.],
) <tab-delta>

- *The components' write is orthogonal to the layer's output.* All components together write 5.5–5.9 times $|y|^2$ on average over the statement tokens (the active set alone 0.12–0.14 before the object and 3.6–4.1 at the end token), and the inner product with the rest's write is minus the same number (@tab-raw): $chevron.l C, y chevron.r approx 0$. The remainder is the original matrix minus a matrix the layer never had.
- *The model's twin difference contains none of the components'.* Present share 0.004; $|delta p|^2$ is 2.7–2.9 times $|delta y|^2$; $kappa$ 6.3–6.8 (@tab-delta).
- *Behaviourally:* swapping the model's own output inside the write span does nothing ($nu = 0.00$ in both seeds, @tab-swap; a random subspace of the same size also gives 0.00, @tab-base), and removing it leaves the separation at 0.99. The $P$-swap reaches the site swap because an un-cancelled write several times the size of the model's whole twin difference is added.
- *The side effects exceed the site's:* the $P$-swap disturbs the word-count answer more than swapping the whole outputs does in seed 1 (KL 0.113 against 0.079, @tab-side), and keeping only $P$ damages the statement's next-token predictions more than removing all seven outputs (prefix KL 2.60 and 2.58 against 2.28, rows "$R$" of @tab-side: removing $R$ leaves only $P$).
- *The two seeds are unrelated:* $cos(delta p_1, delta p_2) = 0.05$ (@tab-agree).
- In weight space the picture is the same without any data: $chevron.l P, W chevron.r_F \/ |P|_F^2$ is 0.000–0.002 in every layer and the components' squared Frobenius norm is 0.95–0.97 of the matrix's own (@tab-frob).
Plain tPD at these settings therefore adds a second, self-made route to the verdict and subtracts it again in $Delta$. Every test that only reads the truth verdict rates it best of the three ($P$-swap 0.85–0.92).

== Recon: a faithful part of the output, of nearly everything <res-recon>

- *Faithful on the output side.* Present share 1.03, $kappa = 0.98$ (@tab-delta); the write-span swap equals the $P$-swap (0.80 and 0.83 against 0.79 and 0.81, @tab-swap); the seeds agree as functions ($cos = 0.72$ on twin differences, 0.96 on raw writes; @tab-agree, @tab-agree-raw). Recon is the positive control of the write-span test: a trained decomposition can pass it.
- *Not targeted at the verdict.* The active set (387 and 432 of 672 components) writes 70–79% of $|y|^2$ at every statement token, the tokens before the object included (@tab-raw), and removing it costs 77–80% of what removing the whole outputs costs there (@tab-score). On the Pile the same components are quiet: 6% of the output's energy (@tab-pile). So recon is targeted at the statements as an input distribution: it rebuilds whatever these layers do on such sentences. What it does not do is single out the verdict.
- *The truth value is under-represented in it.* Although $P$ is most of the output, its own writes carry 0.75–0.81 of the swap and the small rest 0.14–0.19 (clamped swaps); in the affirmative fact families $P$ and $R$ carry about the same (@tab-family). A reconstruction loss weights directions by their size, and the truth-relevant difference is small.
- *The read side is not determined.* Swapping what $W$ itself writes in response to the input directions the active set reads moves the verdict by 0.13–0.16 (@tab-swap), and the two seeds' read subspaces overlap by 0.06 while their write subspaces overlap by 0.59 (share of the smaller subspace's dimensions; mean over layers). The two seeds write nearly the same thing from nearly disjoint input directions. On the statements a component's read vector only has to be correlated with the pattern the matrix responds to; in an 18,944-dimensional input, many vectors are. The components say what the layers write, not which inputs make them write it.

== Ortho: orthogonal where it is trained, not where the truth value is <res-ortho>

#figure(
  tab_all,
  caption: [All 96 components per layer as $P$ (672 in all), training probe. Second column: the model's own output inside the span of all their write vectors (at most 96 of 3,584 dimensions per layer). Last two rows: subspaces of the same size without a decomposition. Present share and energy ratio as in @tab-delta, for the full component sum $C$.],
) <tab-all>

The ortho term is trained on one thing: the raw write of all components together should be orthogonal to the remainder's. That holds ($chevron.l C, D chevron.r \/ |y|^2 = -0.01$, @tab-raw). Three things it does not give.

- *The active set is not a self-contained part.* The active set's write is cancelled to 60–87% in the model's output, and by the _other components_, not by $Delta$: at the first seed, $chevron.l p, "other components" chevron.r \/ |y|^2$ is $-0.13$ to $-0.32$ while $chevron.l p, D chevron.r \/ |y|^2$ stays between $-0.001$ and $0.02$; the other components' write is larger than the active set's own (0.21–0.57 of $|y|^2$ against 0.15–0.34; all from the inspection tables, `inspect.md`). So components with a low mean causal importance are not negligible in the model: they undo most of what the important ones write. The phase D analysis saw the behavioural side of this without the cause: removing every component outside the active set from the full model left 0.60–0.74 of the verdict separation (seeds 1–3; read from table 1c of the phase D analysis tables, `scratch/26-10-29_paper_writing/tables/tables.md`, an earlier run on another GPU).
- *The behavioural consequence:* the active set's $P$-swap moves the verdict by 0.31–0.67, the model's own twin difference in the same write directions by 0.03–0.11 (@tab-swap). For the active set, ortho fails the write-span test as plain does. For all components together it roughly passes: their swap gives 0.26–0.58 and the swap of their write span 0.30–0.40, where a random subspace of that size gives 0.01 (@tab-all). The full component sum is a part of the model's output; the part that the method marks as important is not.
- *Orthogonality of the raw writes does not carry over to the twin differences.* Even for all components, only 0.27–0.29 of $|delta C|^2$ is present in the model's twin difference (@tab-delta). The raw writes are dominated by what a statement and its twin share, and that is all the term constrains.

What ortho's components carry:
- *A minority of the truth value.* The active set's own writes carry 0.12–0.34 of the swap (clamped), the rest 0.60–0.80; with all components as $P$, swapping $Delta$ alone still moves the verdict by 0.40–0.67 (@tab-swap). The fixed shift shows where the average difference between false and true statements lives: added to $P$ it leaves 0.51–0.95 of the separation, added to $R$ it removes or inverts it ($-0.05$ to $-0.43$, @tab-other).
- *Sentence form more than truth.* In the affirmative fact families (cities, element symbols) the $P$-swap is 0.02–0.08 in all four seeds; it is large only for number comparisons and negated statements, and for a different subset of them in each seed (@tab-family).
- *Not the same thing under another probe.* Seed 2's $P$-swap falls from 0.67 to 0.36 under Yes/No 1; the others from 0.31–0.43 to 0.25–0.31 (@tab-swap-yn).

*Seeds.* The raw writes fall into two groups: seeds 1 and 4 agree with each other (cosine 0.48), seeds 2 and 3 with each other (0.42), and across the groups the agreement is 0.02–0.12 (@tab-agree-raw). This is the "at least two modes" seen earlier, now with a fourth seed that lands in the first seed's group. On the twin differences, the part that matters for truth, no two seeds agree by more than 0.20 (@tab-agree). The fourth seed (38 active components) behaves like the others in every table.

What ortho does better than the other two: its edits disturb other answers least ($P$-swap word-count KL 0.011–0.030, site 0.079), and its components are nearly silent before the object (prefix share 0.02–0.12) and on the Pile (0.006–0.009, @tab-score).

== A baseline without a decomposition <res-base>

#figure(
  tab_base,
  caption: [Parts of the original matrices with no decomposition behind them, same tests, training probe unless marked. Each row is the original matrix restricted to a subspace. Sizes: the per-layer numbers of components of the named active set, or a fixed number per layer (total in brackets). Twin-difference PCA: top principal directions of the output's difference between twins of the training split, per layer. Input PCA: top principal directions of the matrix's input on training statements (the draft's data-aligned basis). Random: a random subspace of the output space. Separation left and prefix share: after removing the part (prefix share: of the site removal's KL).],
) <tab-base>

- *A small truth-specific subspace exists in the model's own output.* With the per-layer sizes of ortho seed 1's active set (29 directions over the seven layers), the twin-difference PCA part gives a $P$-swap of 0.80 under the training probe and 0.77 under Yes/No 1, with side effects below the site's (word-count KL 0.031 against 0.079; prefix share 0.06), and its fixed shift inverts the verdict (separation $-0.23$). Two directions per layer (14 in all) already give 0.76; one per layer gives 0.12. Ortho's active set of the same size gives 0.31, with a write-span swap of 0.03.
- *It does not isolate the value at that size either:* its complement's swap is 0.37 (0.54 for two per layer). With recon's sizes (387 directions) the part carries 0.86 and the complement 0.05, the only split in this report that comes close to "the rest carries nothing"; it is then as broad as recon (prefix share 0.69 against 0.77).
- *Recon against the baselines of its size:* recon's active set (0.79, rest 0.41) is better than input PCA (0.70, rest 0.62), so its verdict loss does some targeting within the reconstruction, and worse than the twin-difference PCA (0.86, rest 0.05).
- *Generic variance is not truth:* input PCA with ortho's sizes moves the verdict by 0.01 while its removal costs 0.55 of the site removal's prefix KL. Random subspaces give 0.00 at every size.
- *The baseline is weakest where the decompositions are:* 0.34–0.50 in the affirmative fact families, 0.93–1.00 for comparisons (@tab-family). The truth value of plain facts is spread over more directions than 29, or over sites outside these seven matrices.
The baseline is supervised by which statements are twins, and it says nothing about how the layers compute what it carries. It is a reference: this much of the truth value can be had from a subspace of this size that is a part of the model by construction. No trained decomposition offers that at the size of a plain or ortho active set: ortho's components carry less, and plain's are not a part of the model.

== What the tests showed about the tests <res-tests>

- *The $P$-swap read by a truth probe cannot tell a decomposition from a re-implementation.* It rates plain best. The write-span swap can: it is 0.00 for plain and equal to the $P$-swap for recon, and both are interchange interventions with the same readout, so the comparison needs no threshold and no extra assumption. It complements the draft's test of the same failure, the side effects of the $P$-swap on other answers, which catches plain seed 1 (word-count KL 0.113 against the site's 0.079) but not seed 2 (0.076) and not ortho (0.011–0.030, @tab-side).
- *Removal and swap can disagree, in both directions.* Removing $P$ suggests that the rest carries nothing; swapping the rest shows that it does (@res-rest). Removing the model's output inside ortho's write span costs a large part of the separation (0.19–0.50 left) while swapping it between twins moves the verdict by 0.03–0.11: these directions hold something the verdict needs that is the same for a statement and its twin. Removals show that a part is used; only the swaps show whether it carries the value.
- *Swaps of the two parts are not additive unless the other part is held.* The recomputed part follows the swapped one at later layers. The clamped swaps add up to the site swap within 0.04 and give each part's own share.
- *The read-span swap is not a criterion.* The faithful method fails it too (0.13–0.16). It is reported as a description of the read side (@res-recon).
- *Agreement between seeds as functions is cheap and decisive.* One forward pass per statement and a cosine separate "the same mechanism found twice" (recon, 0.72) from "two unrelated answers" (plain, 0.05) without matching components.
- *Weight-space similarity to $W$ is not a test of faithfulness.* $chevron.l P, W chevron.r_F \/ |P|_F^2$ is only 0.10–0.18 for recon, which passes every data-based test (@tab-frob): outside the directions the statements excite, the component matrices are unconstrained by any loss. The components are parts of the weights only as far as the statements can see.

= What follows for the paper <paper>

These are proposals for the draft (`reports/26-11-04_tpd_truth_paper_draft.typ`); none is decided.

+ *The three expected results hold, and each can now be stated with a test behind it.* Plain re-implements (now with a behavioural test and with numbers that leave no room: present share 0.004). Recon is faithful and broad, and "broad" can be stated precisely: targeted at the statement distribution, not at the verdict. Ortho is "better, with clear weaknesses", and the weaknesses can be named: its full component sum is a part of the output, but the active set is not (the unimportant components cancel it), its components carry a minority of the truth value and mostly sentence form, and its seeds form two groups.
+ *A fourth result for all methods:* the rest carries the truth value. It belongs next to the removal results it corrects (T0, "only $P$").
+ *Add to the evaluation section:* the clamped swap (as the additive version of the $P$/$R$ swaps), the write-span swap with its random-subspace control, the present share (next to $kappa$), and agreement between seeds. Drop the read-span idea or keep it as a description.
+ *Report a baseline.* The twin-difference PCA part of the same size is the reference a targeted decomposition has to beat; at present none does (@res-base).
+ *The per-family table should be in the main text.* Pooled numbers hide that the affirmative fact families behave differently from negations and comparisons.
+ *Seeds:* for plain and ortho, "which seed" changes what $P$ is as a function. The paper's seed comparison can state this with the agreement table instead of component matching.

= Limits <limits>

- *One run per cell, one model, one task.* The intervals cover statement sampling only. Plain and recon have two seeds, ortho four (the fourth trained here). A few cells were cut for GPU time ("–" in the tables): all components as $P$ for plain seed 2, recon and ortho seed 4, and the mirrors of the anchored parts.
- *Components enter at full strength.* Every test uses the active set at mask 1, the draft's convention ($p + r = y$ exactly). The method itself scales each component by its causal importance per token. For ortho the difference matters: the components outside the active set write against it at full strength (@res-ortho), and a CI-weighted $P$ was not tested. The numbers for all components together (@tab-all) are the closest substitute.
- *Twins of equal length only,* which leaves out the translation families and weights the pooled numbers towards cities and number comparisons (@tab-family). No out-of-distribution statements.
- *The baseline is supervised by truth labels* through the choice of twins, on the families it is tested on. The decompositions are supervised by the model's verdict on the same families, so the comparison is fair in kind, but no claim is made that the baseline transfers to other families.
- *The write-span test detects cancellation in the set's own write directions.* A set whose writes agree with the model's in those directions passes it without being the model's mechanism (the test is necessary, not sufficient), and it says nothing about the read side.
- *The fourth ortho seed was trained on another GPU type and next to other jobs* (an H100 instead of the RTX PRO 6000 of seeds 1 and 2); earlier work in this project found the active set to be sensitive to GPU numerics.

= Further tables <appendix>

#figure(
  tab_other,
  caption: [The other interventions on $P$ and $R$ (active set), training probe and Yes/No 1. Separation share left after the zero removal, the mean removal and the fixed shift (1 = untouched, below 0 = inverted); $nu$ of the twin average (the site version gives 0.45: halfway).],
) <tab-other>

#figure(
  tab_side,
  caption: [Side effects on answers that should not depend on truth. Word count: KL of the digit distribution under the probe `\nThe number of words in this statement is␣` from the original's. Subject recall: change of the probability of the statement's subject under a recall probe. Prefix / object + end KL: next-token KL at the statement tokens before the object / at the object and end tokens.],
) <tab-side>

#figure(
  tab_agree,
  caption: [Agreement between decompositions as functions: $cos(delta p_i, delta p_j)$ of the active sets' twin differences, pooled over the object and end tokens and the seven layers.],
) <tab-agree>

#figure(
  tab_agree_raw,
  caption: [The same for the raw active-set writes at all statement tokens (each token's vectors divided by $|y|$).],
) <tab-agree-raw>

#figure(
  tab_pile,
  caption: [Removal on the Pile: #pile_windows validation windows of 64 tokens, the part removed at every position. KL of the next-token distribution from the original's, mean over positions; its share of the site removal's; mean $|p|^2 \/ |y|^2$ over positions and layers.],
) <tab-pile>

#figure(
  tab_frob,
  caption: [Weight space, no data. $chevron.l P, W chevron.r_F \/ |P|_F^2$: 1 if the component matrix $P$ is a Frobenius-orthogonal piece of $W$, 0 if $W$ has no gain along it.],
) <tab-frob>

= Run facts <facts>

- *Machine.* One rented H100 SXM (vast.ai), 2026-11-15, 18:11 to 21:03 UTC; destroyed afterwards. All test numbers in this report come from this machine, the site version included.
- *Decompositions* (WandB run ids, project `bitt-j-personal/spd`): plain #runid("s-7152548c"), #runid("s-46efc525")\; recon #runid("s-eeac663f"), #runid("s-edc17041")\; ortho #runid("s-0c2fa0d7"), #runid("s-4186f973"), #runid("s-6acf4d80"), and seed 4 #runid("s-22600dd8") (trained here; checkpoint not on WandB, local copy `~/spd_out/runs/spd-s-22600dd8/`).
- *Code* (in `spd/experiments/lm/honesty_targeted_decomposition/`): `paper_tests.py` (targets `writespan`, `readspan`, `bothspan`, `randspan`, `pcaspan`, `diffspan`, intervention `swap_clamped`), `paper_interventions.py` (`span_split`), `paper_inspect.py`, `paper_pile.py`.
- *Outputs and scripts:* `~/spd_out/26-11-15_open_research/` (`vm/test/` per-statement arrays and summaries, `analysis/` the scripts that produce every table here, `scripts/` the job lists as run).
- *Record:* `convos/julian/26-11-15_open_research_LOG.md`, with the predictions written before the outputs were read.

