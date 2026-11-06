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
    - *Status.* Two sections exist: the three decomposition methods (@methods) and how they are evaluated (@evaluation: categories, controls, measures, the experiment-to-category table and a glossary). Data, results and introduction follow in later steps.
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

== What it means for a decomposition to capture the truth value <capture>

This section builds, step by step, the notion against which a decomposition is judged. It starts from how we think about the model, not from any decomposition method; tPD enters only at the end, as one way of splitting the weights. We use the language of causal abstraction (Geiger et al., "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability", JMLR 2025; definition numbers refer to that paper).

=== Intuition

The basic question we want to answer is "Has a decomposition captured the model's internal truth computation?" Making this intuition precise is surprisingly hard.

One assumption this intuition presumes is that the model computes truth values of statements: in its forward pass it builds a quantity that says whether it takes the statement to be true, we may say its belief, and it uses this belief in coherent ways. A stronger claim, which makes this even clearer, is that the model can be understood as extracting features from the input and producing its output from these features only. One of the features, $T$, plays the role of the truth value.

"Where is $T$ computed in the weights?" then splits into two questions that need different tools:
- *Where is $T$ represented?* A question about activations: in which activations can $T$ be read, and how can it be changed there? (@ca-link, @ca-superposition)
- *Which weights compute $T$?* A question about mechanisms: which part of the weights produces what that readout reads? (@ca-weights)
The second presupposes an answer to the first, or a stand-in for it.

=== Causal models

A *causal model* is an acyclic graph whose nodes are *variables*, with one *mechanism* per variable: a function that gives the variable's value from the values of its parents. Variables without parents are inputs. Running the model on an input computes every variable in the order of the graph.

An *intervention* changes how some variables are computed and recomputes everything downstream. The simplest kind fixes variables to chosen values (a hard intervention). More general ones replace a variable's mechanism by another, for instance "add this vector to the variable's value" (Geiger's *interventionals*, Def. 11). We call an intervention on the network an *edit*.

=== Two views of the model

*The high-level model $cal(H)$* (the feature view):
- an input variable $X$, the tokens;
- feature variables $F_1, ..., F_k$, each computed from $X$ or from other features. $T$ is one of them. With one statement per input there is one $T$;
- an output variable $Y$, the logits of the next-token prediction, computed from the features only.

$cal(H)$ is the *claim* about the model.

*The low-level model $cal(L)$* (the network). The variables are the activations: every weight matrix's output and every nonlinearity's output, at every position. The mechanisms are the matrix multiplications and the nonlinear functions. The *weights sit inside the mechanisms*; they are not variables. $cal(L)$ makes no claim; it is the network written as a causal model.

"The network can be understood as $cal(H)$" needs a link between the two. The obvious link, "each feature is a group of activations", does not work, because features do not sit on neurons. The next subsection gives a link that needs no such grouping.

=== Linking the views: readouts and edits <ca-link>

The link has two parts (Geiger's *exact transformation*, Def. 25):
- a *readout* $tau$: from a state of the network (all its activations on one run) to a state of $cal(H)$ (a value for every feature). It says what the features are in the network. It may be *partial*, defined only on states the network actually reaches, or close to them;
- a set $Psi$ of *edits* on the network, and a map $omega$ that says which intervention on $cal(H)$ each edit implements. Example: "add $v$ to the activation at layer 14 of the last statement token" implements "set $T$ to true".

*$cal(H)$ describes the network on $Psi$* if, for every edit $e in Psi$,
$ tau("run of" cal(L) "under" e) = "run of" cal(H) "under" omega(e): $
editing the network and then reading out gives the same as intervening in $cal(H)$. In words: whatever we do to the network through the edits in $Psi$, $cal(H)$ predicts the outcome. The set $Psi$ is part of the claim: $cal(H)$ can describe the network for some edits and fail for others. In practice the equality holds only approximately; the graded version (Def. 41) measures how often, or how closely, it holds.

Formal detail: Geiger require $Psi$ to be an *intervention algebra*. Edits of different variables must commute, and a later edit of a variable overrides an earlier one (Def. 16). Edits along overlapping directions can violate this (@ca-superposition).

=== The special case: a basis

The usual form of the link is an *alignment* (Def. 31) and gives a *constructive abstraction* (Def. 33). Each feature is assigned its own group of network variables, possibly after an invertible change of coordinates such as a rotation of the activation space (Def. 28; distributed alignment search finds such rotations). The readout reads the group; an edit overwrites it; edits of different features cannot disturb each other, because the groups are disjoint. This is the precise sense of "a basis aligned with the high-level view".

The basis is a strong requirement. Every feature needs its own coordinates at the same time, so at a site with $d$ dimensions at most $d$ features fit. This is also what gives it teeth: the feature has to sit in a definite place, and overwriting that place has to work.

=== Superposition <ca-superposition>

Superposition is the hypothesis that the model represents more features than it has dimensions, in overlapping directions, relying on most features being inactive most of the time. Then no basis in the sense above exists. What remains possible is easiest to see in the smallest example.

*Example.* Two features $F_1, F_2 >= 0$ are never non-zero together. The model stores both in one number, $h = F_1 - F_2$, and later reads $F_1 = "ReLU"(h)$ and $F_2 = "ReLU"(-h)$, exactly, on every input that occurs.
- *Readouts exist:* $tau(h) = ("ReLU"(h), "ReLU"(-h))$.
- *Edits exist only for some interventions.* "Set $F_1 := a$" on an input with $F_2 = 0$ is the edit $h := a$. "Set $F_1 := a$" on an input with $F_2 = b > 0$ has no edit: $cal(H)$ says $(a, b)$, and no value of $h$ encodes both.

So $cal(H)$ with $F_1$ and $F_2$ as two independent variables describes the network *on the edits that keep them exclusive, and on no others*. The theory allows two honest descriptions, and both state a fact about the model:
- *restrict $Psi$* to the edits that keep $F_1$ and $F_2$ exclusive;
- *merge* $F_1$ and $F_2$ into one high-level variable with values "$F_1 = a$", "$F_2 = b$", "neither" (Geiger's value and variable merge, Prop. 40). Then $h$ is a basis for it again.

What the theory does not allow is the claim that $F_1$ and $F_2$ can be manipulated separately in this model, and that is correct: they cannot. With nearly orthogonal directions instead of opposite ones, edits work approximately, with an error that grows with the number of other active features; the graded version measures it.

*Consequence for $T$* #assumed. $T$ is a function of the statement's content (subject, object, recalled fact). Any intervention that sets $T$ against the content ("content of a false statement, $T$ = true") creates a combination that occurs naturally only where the model errs. If $T$ is stored entangled with the content, such edits may be impossible or only approximate, as "$F_1 := a$ with $F_2 = b$" above. The interventions we need for $T$ sit exactly where superposition bites.

=== What keeps the general link from being empty

Without the grouping, the link is too permissive. A strong enough readout extracts $T$ from almost any activation (the usual objection to probes), and an edit that rewrites the whole activation can force almost any outcome. The general version therefore needs constraints, and they have to be stated, because the theory does not supply them:
+ *Locality:* the readout and the edits for $T$ act at stated sites (layers, positions).
+ *Specificity:* an edit of $T$ leaves the readouts of the other features unchanged, within the edits in $Psi$.
+ *Simplicity:* readouts and edits come from a small class (linear, or along few directions).
Which constraints are chosen is part of the claim "$T$ is represented here".

=== Narrowing to the truth question

The general $cal(H)$ has every feature the logits read, so describing the network by it needs a readout and edits for all of them. The truth question needs less. Keep only
$ X -> Z -> T -> (Y_1, ..., Y_r), $
with $Z$ the content bundle, $T$ the truth value, and $Y_1, ..., Y_r$ *readouts of $T$ by the model itself*: the verdicts under different truth questions appended to the statement (the verdict prompt, the second prompt, the Yes/No prompts). Removing the other features is a *marginalization* (Prop. 40). It is allowed if these outputs read the features only through $T$; that is the assumption "the verdict measures $T$", stated as a causal model.

The several outputs make "uses the belief in coherent ways" testable: one $T$ means one edit at the statement moves every $Y_j$ as $cal(H)$ predicts. Because attention is causal, the activations at the statement tokens do not depend on the question that follows, so the same edit can be tested under every question. B3 shows why this matters: steering along a truth direction at the statement flipped the verdict under both prompts, while the model still recalled the fact correctly. The verdict prompt may read a truth *assessment* that is not the model's knowledge.

=== Which weights compute $T$: splitting the mechanisms <ca-weights>

The weights sit inside the mechanisms, so "these weights compute $T$" cannot be stated in $cal(L)$ as it is. It needs a *refinement* of $cal(L)$: split a linear layer's weight matrix into parts, $W = P + R$, and insert their outputs as new variables,
$ p = P x, quad r = R x, quad y = p + r. $
Three properties make this well-defined:
+ *Summing gives back the network exactly.* $p + r = y$ on every input, so the refinement claims nothing by itself.
+ *The split happens at a linear map.* $W x$ splits into summed parts only because $W$ acts linearly on $x$. The same split could not be made across a nonlinearity.
+ *If the split depends on the input, it is a function of the network's own activations* (for instance of $x$). Then $p$ and $r$ have mechanisms like every other variable.

The refinement is not a change of basis: $p$ and $r$ cannot be recovered from $y$, and they may overlap in activation space. So superposition in the activations does not block it; the split is made in the weights, not along directions. What it buys is new edits: "replace only the $P$-part of this layer's output" has no counterpart in $cal(L)$.

With the split, "$P$ computes $T$" has three readings of increasing strength:
- *(R1) $T$ is carried by $P$'s writes.* Setting the $P$-writes (at all split layers and positions) to their values on another statement $x'$ makes the outputs behave as for $T(x')$.
- *(R2) Nothing outside them is needed.* Setting the $R$-writes to their values on $x'$ leaves the outputs behaving as for $T(x)$. R2 is not implied by R1: if $T$ were produced by the $R$-writes of early layers and only passed on by the $P$-writes of later ones, R1 could hold while swapping the $R$-writes also moves the outputs.
- *(R3) $P$ computes $T$ from the content.* $T$ depends on $Z$ only through $P$: change $Z$ only in what $P$ reads, and $T$ moves as $cal(H)$'s mechanism $Z -> T$ says. This needs a readout of $Z$ at the input side of $P$, and with several layers it is hard to set up, because later layers read the writes of earlier ones.

R1 and R2 are about where $T$ is carried and who supplies it; R3 is about computation. A method that names "the weights that do it" is a choice of split: a circuit takes $P$ = a subset of attention heads or neurons, a weight decomposition takes $P$ = some of its components. For tPD, $P = sum_(c in S) U_c V_c^top$ for a set $S$ of components, and $R$ = the other components plus $Delta$; we call their writes the *C-writes* and the *$Delta$-writes*.

*Two readings that are too strong.*
- *"$P$'s writes are $T$ and nothing else."* $P$ is a part of the model's real weights, and the writes that carry $T$ also carry the content it is computed from. So $P$'s writes align with the bundle $Z$ (which contains $T$), not with $T$ alone. Swapping all of $Z$ from $x'$ still gives $T(x')$, so R1 is the right claim: "$P$'s writes contain $T$".
- *Minimality* (no part of $P$ can be dropped). A targeted decomposition may take along parts that have nothing to do with $T$. The price: without minimality, $P = W$ passes trivially. How small $P$ is must therefore be reported alongside (the share of components or parameters).

=== Measuring with the verdict

R1 and R2 are interchange interventions (Def. 44): a *swap* is an edit that sets some variables to their values on another input.

*Pairs.* A pair $(x, x')$ of statements has the same number of tokens and aligned positions (the writes are per position) and different verdicts.

*Swaps.* The *P-swap* (for tPD: C-swap) runs the network on the base statement $x$ and sets the $P$-write at every split layer and position to its value in the run on $x'$; everything else, the $R$-writes included, is recomputed as usual. The *R-swap* ($Delta$-swap) does the same with the $R$-writes. The *site swap* replaces the split layers' whole outputs $y$.

Because of the recomputation, the P-swap also catches a used route through $R$: the $R$-writes are recomputed from $x$ and the changed activations, so if they had a route of their own from $x$ to $T$ that the outputs read, they would pull the verdict back to that of $x$ #concluded. A copy of $T$ in the $R$-writes that nothing reads does not matter.

*Interchange intervention accuracy.* For a swap, $"IIA"$ is the share of pairs on which the verdict after the swap equals the verdict on $x'$ (Def. 48). Without any swap it is 0, since the verdicts of a pair differ.

*All layers split.* $P$ *captures $T$* (R1 + R2) if $"IIA"(P) = 1$ and $"IIA"(R) = 0$, under every readout $Y_j$.

*Only some layers split* (our case: `down_proj` of layers 11–17 at the statement tokens). $T$ may also travel by routes that bypass the split layers, so even the site swap need not move the verdict, and no split of these layers can do better. The definition is taken relative to the site swap:
- $"IIA"(P) \/ "IIA"("site")$, ideally 1;
- $"IIA"(R) \/ "IIA"("site")$, ideally 0;
- the size of $P$, reported alongside.

*Swaps and superposition.* A swap between statements of different truth puts $x'$'s $P$-writes, which carry $x'$'s content and $T(x')$, next to $x$'s content elsewhere in the run. If the outputs also read $x$'s content by another route (attention to the tokens, the $R$-writes, unsplit layers), they judge a mix. Through $R$ that is a failure of the split, as it should be. Through unsplit layers, the site swap absorbs it.

=== What this says about our analyses

- *Representation was never tested on its own.* We have readouts of $T$ (probes, the mass-mean and logistic-regression truth directions: K0, Dir) and one edit (B3), but no test that a readout and an edit for $T$ satisfy the link condition.
- *Most of our analyses are removals, not swaps* (R, T0, T1, T2, M-core: components switched off, reversed or amplified). A removal produces activations no statement produces, so the readout $tau$, defined on states the network reaches, says nothing about them. In Geiger's treatment of ablation (Sec. 3.5) the high-level model then needs an extra "was it removed" variable, and the result says *whether* something is used, not how. The swaps of R1/R2 were not run.
- *Not measured at all:* R3 (computation rather than passing on), and how far the swaps leave the states the model reaches (the consequence for $T$ in @ca-superposition).

=== Open points

- *Which question the paper's criterion answers:* representation (@ca-link) or the weights (@ca-weights). The swaps answer the second, with the verdicts as a stand-in for a readout of $T$.
- *Which constraints* (locality, specificity, simplicity) make a readout or edit for $T$ admissible.
- *Hard verdict or logit difference.* With the hard verdict, a near-tie that tips over counts as a full flip. Measuring with the logit difference $s$ instead would grade this.
- *The set $S$* for tPD: all components or an input-dependent set of active ones.
- *Pairs.* Two statements with different truth value, the same token layout and otherwise little difference are hard to come by; twins that differ in one word differ in content too, and the swap carries that content along.

== Glossary of experiments <glossary>

All analyses run on every decomposition (plain, recon, ortho; all seeds) and, where they apply, on the controls. Test split unless stated; *OOD* = the out-of-distribution statement sets; *both prompts* = the training verdict prompt and the second prompt "Is this statement true (1) or false (0)? Answer: ". Removals and edits act at every statement token.

#tbl(
  columns: (auto, 1fr),
  align: (left, left),
  breakable: true,
  [name], [what is done],
  [*layer scan*], [Original model. The output of single MLP layers or blocks replaced by its mean over training statements, at all statement tokens or at the last one only; verdict accuracy, separation share, digit mass. Chose layers 11–17 (@tab-layer-scan).],
  [*B1*], [Original model. The layer scan repeated for MLP and attention outputs, single layers and blocks, mean and zero ablation; why attention is not decomposed.],
  [*B2*], [Original model on every test and out-of-distribution statement set under three candidate prompts: accuracy, AUC, digit mass per set. Chose the second prompt and which OOD sets are usable.],
  [*B3*], [Original model. Steering along the in-context truth direction at the statement tokens flips the verdict under both prompts, but the model still recalls the fact correctly afterwards: the verdict prompt reads a truth assessment, not the model's knowledge.],
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


// neeeds to go into evaluation setup
 It is weakest on the negated families and on `element_symb`, where it calls a quarter of the true statements false #verified[run B2, original model, test split].
