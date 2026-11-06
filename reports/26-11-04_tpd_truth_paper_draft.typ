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
    - *Status.* Two sections exist: the three decomposition methods (@methods) and how they are evaluated (@evaluation: what it means for a decomposition to capture the truth value, a ladder of tests derived from that definition with our analyses placed on it, the evaluation setup and a glossary). Data, results and introduction follow in later steps.
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

This section says what we test and why. @capture defines what it means for a decomposition to capture the model's truth value, under stated assumptions. @ca-ladder derives the tests from that definition, orders them into a ladder from sanity checks to control of the truth value, and places our analyses on it. @eval-setup gives the statement sets, the states, the measures and the controls; @glossary describes every analysis.

== What it means for a decomposition to capture the truth value <capture>

We use the language of causal abstraction (Geiger et al., "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability", JMLR 2025; definition numbers refer to that paper) to make this precise.

=== Intuition

The basic question we want to answer is "Has a decomposition captured the model's internal truth computation?" Making this intuition precise is surprisingly hard; we attempt it here.

One basic assumption is that the model computes truth values of statements at all: in its forward pass it builds a quantity that says whether it takes the statement to be true, we may say its belief, and it uses this belief in coherent ways. A stronger claim, often the north star of mechanistic interpretability, is that the model can be understood as extracting meaningful features from the input and producing its output from these features only. One of the features, $T$, plays the role of the truth value.#footnote[A model may compute several truth values, e.g. for several statements in one input; we restrict to inputs with one statement. This is already a major simplification of the task of uncovering the model's truth mechanism.]

The question "Where is $T$ computed in the weights?" splits into two questions:
- *Where is $T$ represented?* A question about activations: in which activations can $T$ be read, and how can it be changed there? (@ca-link, @ca-superposition)
- *Which weights compute $T$?* A question about mechanisms: which part of the weights produces what that readout reads? (@ca-weights)
The second presupposes an answer to the first, or a stand-in for it. Our criterion answers the second, with the model's own answers as the stand-in (@ca-narrow).

=== Two views of the model

A *causal model* is an acyclic graph whose nodes are *variables*, with one *mechanism* per variable: a function that gives the variable's value from the values of its parents. Variables without parents are inputs. Running the model on an input computes every variable in the order of the graph.

*The low-level model $cal(L)$.* The network itself, written as a causal model. The variables are the activations: every weight matrix's output and every nonlinearity's output, at every position.#footnote[A causal model has a fixed set of variables, so inputs of different lengths need some care. We treat this as a technicality which should be resolvable with better modeling. E.g. for $cal(L)$ we can simply think that all inputs are padded to the maximal context window of the model.] The mechanisms are the matrix multiplications and the nonlinear functions. The weights define part of the mechanisms and are not variables.

*The high-level model $cal(H)$.* A causal model of the platonic idea of what the net "actually does":
- an input variable $X$, the tokens;
- feature variables $F_1, ..., F_k$, each computed from $X$ or from other features. $T$ is one of them;
- an output variable $Y$, the logits of the next-token prediction, computed from the features only.

The claim is that the network can be understood as $cal(H)$. This needs a link between the two.

=== Linking the views: readouts and edits <ca-link>

For a causal model an *intervention* changes how some variables are computed and recomputes everything downstream. The simplest kind fixes variables to chosen values (a hard intervention). More general ones replace a variable's mechanism by another, for instance "add this vector to the variable's value" (Geiger's *interventionals*, Def. 11). We call an intervention on the network $cal(L)$ an *edit*.

The link has two parts (Geiger's *exact transformation*, Def. 25):
- a *readout* $tau$: from a state of $cal(L)$ (a value for every activation, i.e. one run, possibly under an edit) to a state of $cal(H)$ (a value for every feature). It says what the features are in the network. It may be *partial*, defined only on states the network actually reaches, or close to them;
- a set $Psi$ of *edits* on $cal(L)$, and a map $omega$ that says which intervention on $cal(H)$ each edit implements. Example: "add $v$ to the activation at layer 14 of the last statement token" implements "set $T$ to true".

*$cal(H)$ describes the network on $Psi$* if, for every edit $e in Psi$,
$ tau("run of" cal(L) "under" e) = "run of" cal(H) "under" omega(e), $
i.e. editing the network and then reading out gives the same as intervening in $cal(H)$. In words: whatever we do to the network through the edits in $Psi$, $cal(H)$ predicts the outcome. The set $Psi$ is part of the claim: $cal(H)$ can describe the network for some edits and fail for others. Two mechanisms that agree on every edit in $Psi$ cannot be told apart at this level; they come apart only on an edit outside $Psi$. In practice the equality holds only approximately; the graded version (Def. 41) measures how often, or how closely, it holds.

Formal detail: Geiger require $Psi$ to be an *intervention algebra*. Edits of different variables must commute, and a later edit of a variable overrides an earlier one (Def. 16). Edits along overlapping directions can violate this (@ca-superposition).

=== Bases and superposition <ca-superposition>

*The special case: a basis.* The usual form of the link is an *alignment* (Def. 31) and gives a *constructive abstraction* (Def. 33). Each feature is assigned its own group of network variables, possibly after an invertible change of coordinates such as a rotation of the activation space (Def. 28; distributed alignment search finds such rotations). The readout reads the group; an edit overwrites it; edits of different features cannot disturb each other, because the groups are disjoint. This is the precise sense of "a basis aligned with the high-level view". It is a strong requirement: every feature needs its own coordinates at the same time, so at a site with $d$ dimensions at most $d$ features fit. This is also what gives it teeth: the feature has to sit in a definite place, and overwriting that place has to work.

*Superposition* is the hypothesis that the model represents more features than it has dimensions, in overlapping directions, relying on most features being inactive most of the time. Then no basis in the sense above exists. What remains possible is easiest to see in the smallest example. Two features $F_1, F_2 >= 0$ are never non-zero together. The model stores both in one number, $h = F_1 - F_2$, and later reads $F_1 = "ReLU"(h)$ and $F_2 = "ReLU"(-h)$, exactly, on every input that occurs.
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
Which constraints are chosen is part of the claim "$T$ is represented here". For the truth question, specificity becomes testable through readouts of the content (@ca-narrow).

=== The truth question: content, truth value and two kinds of readout <ca-narrow>

The general $cal(H)$ has every feature the logits read, so describing the network by it needs a readout and edits for all of them. The truth question needs less. Keep only
$ X -> Z -> T -> (Y_1, ..., Y_r), quad Z -> (Q_1, ..., Q_m), $
with $Z$ the content bundle (subject, object, recalled fact) and $T$ the truth value, and two kinds of outputs, both read off the model itself:
- *truth readouts* $Y_1, ..., Y_r$: the verdicts under different truth questions appended to the statement (the verdict prompt, the second prompt, the Yes/No prompts);
- *other-quality readouts* $Q_1, ..., Q_m$: outputs about the statement that do not depend on its truth, e.g. recalling the fact ("The city of Paris is in"), judgements of other properties of the statement (humour, emotion, surprise, word count), next-token predictions within the statement before its truth is determined.

Removing the other features is a *marginalization* (Prop. 40). It is allowed under three assumptions, and everything below is conditional on them:
+ *(A1) One truth value.* On a statement there is one $T$, computed from $Z$.
+ *(A2) Truth readouts read only $T$.* Every $Y_j$ depends on the statement only through $T$, the same $T$ for all $j$. This is "the verdict measures $T$", stated as a causal model.
+ *(A3) Other-quality readouts read only $Z$.* Every $Q_k$ depends on $Z$ but not on $T$.

*Why both kinds of readout.* The $Y_j$ make "uses the belief in coherent ways" testable: one $T$ means one edit at the statement moves every $Y_j$ as $cal(H)$ predicts. Because attention is causal, the activations at the statement tokens do not depend on the question that follows, so the same edit can be tested under every question. The $Q_k$ make specificity testable: an edit that is meant to change $T$ must leave them alone. Without them, every prediction of $cal(H)$ is about $T$, and a split can pass every test while disturbing everything the verdicts do not read (@ca-reimpl).

*Which $T$.* B3 steered along a truth direction at the statement: the verdict flipped under both prompts, while the model still recalled the fact correctly. If $T$ is the judgement "the statement agrees with what I know", recall reads $Z$, not $T$, and this is what an edit of $T$ should do. If $T$ is meant as the stored fact itself, recall reads it, and B3's edit did not change it. Both readings are legitimate; they are two high-level models with different predictions for the same edit, and a decomposition can be tested against each. Twins such as "Paris is in France" / "Paris is in Spain" keep the stored fact and change the claim, so swaps between them test the first reading.

=== Which weights compute $T$: splitting the mechanisms <ca-weights>

The weights sit inside the mechanisms, so "these weights compute $T$" cannot be stated in $cal(L)$ as it is. It needs a *refinement* of $cal(L)$: split a linear layer's weight matrix into parts, $W = P + R$, and insert their outputs as new variables,
$ p = P x, quad r = R x, quad y = p + r. $
Three properties make this well-defined:
+ *Summing gives back the network exactly.* $p + r = y$ on every input, so the refinement claims nothing by itself.
+ *The split happens at a linear map.* $W x$ splits into summed parts only because $W$ acts linearly on $x$. The same split could not be made across a nonlinearity.
+ *If the split depends on the input, it is a function of the network's own activations* (for instance of $x$). Then $p$ and $r$ have mechanisms like every other variable.

The refinement is not a change of basis: $p$ and $r$ cannot be recovered from $y$, and they may overlap in activation space. So superposition in the activations does not block it; the split is made in the weights, not along directions. What it buys is new edits: "replace only the $P$-part of this layer's output" has no counterpart in $cal(L)$. The price is that the new variables are ours, not the network's: the network only ever sees their sum.

With the split, "$P$ computes $T$" has three readings of increasing strength:
- *(R1) $T$ is carried by $P$'s writes.* Setting the $P$-writes (at all split layers and positions) to their values on another statement $x'$ makes the outputs behave as for $T(x')$.
- *(R2) Nothing outside them is needed.* Setting the $R$-writes to their values on $x'$ leaves the outputs behaving as for $T(x)$. R2 is not implied by R1: if $T$ were produced by the $R$-writes of early layers and only passed on by the $P$-writes of later ones, R1 could hold while swapping the $R$-writes also moves the outputs.
- *(R3) $P$ computes $T$ from the content.* $T$ depends on $Z$ only through $P$: change $Z$ only in what $P$ reads, and $T$ moves as $cal(H)$'s mechanism $Z -> T$ says. This needs a readout of $Z$ at the input side of $P$, and with several layers it is hard to set up, because later layers read the writes of earlier ones.

A method that names "the weights that do it" is a choice of split: a circuit takes $P$ = a subset of attention heads or neurons, a weight decomposition takes $P$ = some of its components. For tPD, $P = sum_(c in S) U_c V_c^top$ for a set $S$ of components, and $R$ = the other components plus $Delta$; we call their writes the *C-writes* and the *$Delta$-writes*.

*Two readings that are too strong.*
- *"$P$'s writes are $T$ and nothing else."* $P$ is a part of the model's real weights, and the writes that carry $T$ also carry the content it is computed from. So $P$'s writes align with the bundle $Z$ (which contains $T$), not with $T$ alone. Swapping all of $Z$ from $x'$ still gives $T(x')$, so R1 is the right claim: "$P$'s writes contain $T$".
- *Minimality* (no part of $P$ can be dropped). A targeted decomposition may take along parts that have nothing to do with $T$. The price: without minimality, $P = W$ passes trivially. How small $P$ is must therefore be reported alongside (the share of components or parameters).

=== The definition <ca-def>

*$P$ captures $T$* if $cal(H)$ (@ca-narrow) describes the refined network (@ca-link), with $T$, together with the part of $Z$ that $P$'s writes carry, read from $P$'s writes and nothing of $T$ from $R$'s writes, *for every intervention on these variables, every readout ($Y_j$ and $Q_k$) and every statement in the domain*.

This is one sentence. @ca-ladder splits its "for every" into separate tests.

*What passing the tests can establish.* Every test is graded (Def. 41): a share or a distance on a sample of statements from a stated distribution $D$, with a confidence interval. A passed set of tests then says: if A1–A3 hold, then with confidence $1 - alpha$ each clause of the definition holds to accuracy at least $q$ on $D$. Three limits are part of this statement:
- *Nothing outside $D$.* Each clause is a statement about all statements, which finite tests can refute but not prove. $D$ should therefore be chosen where failures are likely.
- *Not unique.* Several splits may pass (non-identifiability); each of them is then *a* truth mechanism in this sense.
- *Conditional on A1–A3.* A failed test may be the assumptions' fault rather than the split's; @ca-acheck separates the two.

== From the definition to tests: a ladder <ca-ladder>

=== The grid

The definition's "for every" has three parts:
- *Interventions.* (i) *Swaps:* values that another statement produces. Which intervention on $cal(H)$ a swap implements follows from the inputs: the swapped values mean on $x$ what they meant on $x'$ (Def. 44, 47). (ii) *Settings:* values that no statement produces: $T$ set to the opposite value with $Z$ kept, or $T$ set to "absent". Here $omega$ is an extra assumption, that $cal(H)$ still holds off the states the network reaches, which is where superposition bites (@ca-superposition). (iii) *Interventions on what $P$ reads:* change $Z$ only in $P$'s input (R3).
- *Readouts.* The training truth readout $Y_1$; the other truth readouts $Y_2, ..., Y_r$; the other-quality readouts $Q_1, ..., Q_m$.
- *Statements.* In-distribution pairs; out-of-distribution forms and statements the model gets wrong.

=== The ladder

Each test is a cell of this grid. Ordered by how much of the assumptions they rely on, the cells form a ladder:

#tbl(
  columns: (auto, 1.1fr, 1.9fr, 1.4fr),
  align: (left, left, left, left),
  breakable: true,
  [rung], [cell], [prediction of $cal(H)$], [alternative ruled out],
  [*0* sanity], [precondition of the split], [$p + r = y$; removing the inactive components changes nothing, on statements and on general text], [a broken split],
  [*1* used], [outside the definition (removals)], [removing $P$ degrades the verdict], [$P$ is irrelevant],
  [*2* carries $T$], [(i) × $Y_1$ × in-distribution], [P-swap: $Y_1$ as on $x'$. R-swap: $Y_1$ as on $x$ (R1, R2)], [$R$ supplies $T$ by a second route],
  [*3* carries only $T$], [(i) × $Q$ × in-distribution], [P-swap: each $Q_k$ stays between its values on $x$ and $x'$, and changes no more than under the site swap], [a re-implementation whose uncancelled writes have side effects],
  [*4a* flip], [(ii) $T :=$ opposite × all readouts], [an edit of $P$'s writes from a small stated class flips every $Y_j$ and leaves every $Q_k$; the same edit class on a random split does not], [$P$ is only a lever on the verdict (a bias, the sentence form)],
  [*4b* oblivious], [(ii) $T :=$ absent × all readouts], [$P$'s write replaced by its mean over a twin-balanced set: every $Y_j$ independent of truth, every $Q_k$ unchanged], [$R$ supplies $T$; $P$ is only a lever],
  [*5* coherent use], [(i), (ii) × $Y_2, ..., Y_r$ × out-of-distribution, model errors], [rungs 2–4 hold under every truth question and on every statement set], [$P$ serves only the training question, or computes the verdict from a cue that correlates with truth on the training data],
  [*6* computes], [(iii)], [$T$ moves as $Z -> T$ says when $Z$ is changed only in $P$'s input (R3)], [$P$ passes on a $T$ computed elsewhere],
)

*Why this order.* Each rung relies on more of the assumptions than the one below. Rung 2 needs A2 for one readout, and only on states the network reaches. Rung 3 adds A3. Rung 4 needs $cal(H)$ to hold off the states the network reaches. Rung 5 widens the readouts and $D$. Rung 6 needs $cal(H)$'s mechanism $Z -> T$ at $P$'s input. So the higher the rung, the less a failure says about the split alone. Rung 0 is part of the decomposition itself. Rung 1 does not follow from the definition at all: a removal produces states no statement produces and has no counterpart in $cal(H)$, so it shows *whether* $P$ is used, not *how* (Geiger's treatment of ablation, Sec. 3.5). It stays as a sanity check.

=== Swaps in practice <ca-measure>

*Pairs.* A pair $(x, x')$ of statements has the same number of tokens and aligned positions (the writes are per position) and different verdicts. A pair with equal verdicts cannot test anything: $cal(H)$ predicts the same output with and without the swap.

*Swaps.* The *P-swap* (for tPD: C-swap) runs the network on the base statement $x$ and sets the $P$-write at every split layer and position to its value in the run on $x'$; everything else, the $R$-writes included, is recomputed as usual. The *R-swap* ($Delta$-swap) does the same with the $R$-writes. The *site swap* replaces the split layers' whole outputs $y$; it does not test the split, but it is a state the network can reach, and it serves as the reference.

Because of the recomputation, the P-swap also catches a used route through $R$: the $R$-writes are recomputed from $x$ and the changed activations, so if they had a route of their own from $x$ to $T$ that the outputs read, they would pull the verdict back to that of $x$ #concluded. A copy of $T$ in the $R$-writes that nothing reads does not matter.

*Interchange intervention accuracy.* For a swap, $"IIA"$ is the share of pairs on which the verdict after the swap equals the verdict on $x'$ (Def. 48). Without any swap it is 0, since the verdicts of a pair differ. With the hard verdict, a near-tie that tips over counts as a full flip; the graded version uses the logit difference $s$ instead.

*Only some layers split* (our case: `down_proj` of layers 11–17 at the statement tokens). $T$ may also travel by routes that bypass the split layers, so even the site swap need not move the verdict, and no split of these layers can do better. Rung 2 is therefore taken relative to the site swap:
- $"IIA"(P) \/ "IIA"("site")$, ideally 1;
- $"IIA"(R) \/ "IIA"("site")$, ideally 0;
- the size of $P$, reported alongside.
Rung 3 uses the site swap the same way: per $Q_k$, the change under the P-swap against the change under the site swap.

*Swaps and superposition.* A swap between statements of different truth puts $x'$'s $P$-writes, which carry $x'$'s content and $T(x')$, next to $x$'s content elsewhere in the run. If the outputs also read $x$'s content by another route (attention to the tokens, the $R$-writes, unsplit layers), they judge a mix. Through $R$ that is a failure of the split, as it should be. Through unsplit layers, the site swap absorbs it.

=== Checking the assumptions with the site swap <ca-acheck>

Each cell of the ladder has a counterpart that tests A1–A3 instead of the split: the same test with the site swap. Under the assumptions, the site swap moves every truth readout together and leaves in place the other qualities that the two statements of a pair share. If the site swap fails a cell, that cell cannot test the split. Example: if the site swap moves the 1/0 verdict but not the Yes/No answer, then a split whose effect does not transfer to the Yes/No question has not failed; A2 (one $T$ read by every truth question) has, at these layers.

=== Re-implementation: why the verdicts alone are not enough <ca-reimpl>

A split can pass rung 2 without being the model's mechanism. Example, a *re-implementation*: the components write a large vector $c$ that is not a part of the layer's real output $y$, and $R$ writes $y - c$, cancelling it. On every input $p + r = y$, so the model is unchanged, and no test on the unedited model can see the difference. In the P-swap, however, $c$ comes from $x'$ and the cancelling $r$ from $x$; whatever in $c(x') - c(x)$ the verdict does not read is no longer cancelled. The truth readouts cannot notice this, because they read only $T$. A re-implementation can therefore show only in edits that separate $P$ from $R$, and in two forms:
- *Uncancelled writes:* large writes in directions the verdict does not read become visible when an edit breaks the cancellation, provided some other output reads those directions: rungs 3 and 4, through the $Q_k$.
- *A different route:* $P$ produces the verdict signal from a cue that agrees with the model's own computation on the training data. It shows where the two come apart: rung 5.
If neither shows, the split differs from the model's mechanism in ways no output reads. Only internal measures (cancellation norms, @glossary) see this; whether it counts as a failure depends on whether the decomposition is meant as an account of the mechanism or as a handle on the behaviour.

=== Our analyses on the ladder <ca-analyses>

Our analyses were fixed in a plan before the ladder existed (@eval-setup), and most of them are removals (components switched off, reversed or amplified), not swaps. Placed on the ladder:

#tbl(
  columns: (auto, 1.3fr, 1.6fr, 1.3fr),
  align: (left, left, left, left),
  breakable: true,
  [rung], [analyses run], [what they show here], [missing],
  [*0*], [R, R-active, R-mean, R-inactive, active set, SE-Pile], [the components alone give the verdict; the inactive components can be removed; the side effect on the Pile; the size of $P$], [–],
  [*1*], [T0, T1, T2, T3 (A8), Dir-remove], [whether $P$ is used for the verdict; T3 and Dir-remove act on the original model], [– (rung outside the definition)],
  [*2*], [none], [–], [P-swap and R-swap on pairs],
  [*3*], [G2 next-token, G2 digit questions], [a removal version: other readouts under $Delta$ alone and the reversed top-10 group, against all off. Both states lie off the network's natural states, and the reference is itself a broken model], [the swap version, against the site swap],
  [*4a*], [M-core, M-twin, M-wide; B3 on the unsplit network], [whether reversing or amplifying components moves the verdict, against random components and the controls; masks −1 and 2 lie outside the trained range], [the $Q_k$ under the same edits (only the G2 digit questions under the reversed top-10 group exist)],
  [*4b*], [T0], [the verdict half, with zero instead of mean], [the mean version; the $Q_k$],
  [*5*], [R and T2 under the second prompt, on OOD; G2 Yes/No transfer, T2 Yes/No], [removal versions of transfer to other questions and statement sets], [swaps under every truth question],
  [*6*], [none], [–], [a path test],
  [A-checks], [layer scan, B1], [removal versions on the original model: which layers the verdict needs], [the site swap under every readout],
)

*Analyses that are not on the ladder.* K0–K3 and Dir ask whether truth can be *read* from the components (probes, selectivity, alignment with truth directions); they describe the representation, not a clause of the definition. K0 is reported as the reason probe accuracy is not evidence. K3 comes closest to a rung: on the statements the model gets wrong, it asks whether a selective component follows the model's verdict or the label, i.e. whether it tracks $T$ rather than the truth, a correlational form of rung 5. The valley check and the active set describe the decomposition's structure; the cancellation norms are the internal diagnostic of @ca-reimpl.

=== Open points

- *Which other qualities $Q_k$.* Rung 3 needs qualities that the two statements of a pair share, read off the model in a format it answers reliably.
- *Pairs.* Two statements with different truth value, the same token layout and otherwise little difference are hard to come by. Twins that differ in one word differ in content too; the swap carries that content along, which is allowed under the broad reading (the bundle $Z$), but restricts which $Q_k$ the pair shares.
- *The set $S$* for tPD: all components or an input-dependent set of active ones.
- *The edit class* for rung 4a and the set over which the mean of rung 4b is taken.
- *Which $T$:* the judgement or the stored fact (@ca-narrow).

== Evaluation setup <eval-setup>

*Statement sets.* The validation and test splits of the Truth-is-Universal statement sets (test: 10 families, 1,794 statements, @tab-verdict-accuracy) and 24 out-of-distribution (OOD) files; T2 and M-core use 13 of the OOD files. The model is weakest on the negated families and on `element_symb`, where it calls a quarter of the true statements false #verified[run B2, original model, test split].

*Prompts.* The training verdict prompt; the held-out *second prompt* "Is this statement true (1) or false (0)? Answer: ", a different truth question with the same 1/0 format; two Yes/No truth questions; four digit questions about other properties of the statement (humour, emotion, surprise, word count). G0 checks that the model answers the last two kinds in format.

*States.* Every state replaces the output of each decomposed matrix at the statement tokens by
$ y' = m_Delta D + sum_c m_c U_c V_c^top x $
(notation of @methods), and the question tokens run on the original weights. The named states: *original* (all masks 1); *components only* ($m_c = upright("CI")_c$, $m_Delta = 0$); *$Delta$ alone* ($m_c = 0$, $m_Delta = 1$); *all off* ($m_c = 0$, $m_Delta = 0$: the decomposed matrices write nothing at the statement tokens); *removals* (chosen components from 1 to 0, everything else kept); *reversal* and *amplification* (chosen components at $m_c = -1$ or $2$). Decompositions and controls run through the same code, so a difference between them cannot come from the intervention.

*Measures,* at the answer position:
- *KL recovered* $= 1 - "KL"("original" || "state") \/ "KL"("original" || "all off")$, each KL averaged over statements: 1 = the original verdict distribution, 0 = as bad as all off.
- *Accuracy* (against the labels) and *agreement* (against the model's own verdict).
- *Separation share* (@methods) and *shift*: the signed move of the midpoint between the true and the false class means of $s$, in units of the original gap (> 0 = towards "1"). Together they fix both class means. The shift was added after the outputs were read; it is descriptive.
- *Flips*: the share of the model's "1" verdicts that become "0", and of its "0" verdicts that become "1".
- *Digit mass* (@tab-verdict-accuracy) and *AUC* of a score for true against false within a family.
Metrics are computed per run, never pooled over seeds; pooled over statements and macro-averaged over families, with 95% bootstrap intervals (1,000 resamples within family).

*Controls.* Two bases that nobody trained, run through the same code:
- the *untrained decomposition*: a run's checkpoint at step 0, where $Delta$ holds essentially the whole matrix;
- the *data-aligned basis*: per layer, read vectors $V_c$ = the 96 top principal directions of the matrix's input (uncentred, over the statement tokens of 512 training statements), write vectors $U_c = W V_c$, remainder $W (1 - P_V)$. It has no CI network, so analyses that need CI values do not apply to it.
Within a decomposition, random choices of the same size: random subsets of the active set (T2, 5), random subspaces of the same dimension (T3, 3), random active components (M-core, 10).

*Pre-registration.* All analyses were fixed in a plan before the test and OOD sets were read, except G0–G2, which were added after the first results, and the shift.

== Glossary of experiments <glossary>

All analyses run on every decomposition (plain, recon, ortho; all seeds) and, where they apply, on the controls. Test split unless stated; *OOD* = the out-of-distribution statement sets; *both prompts* = the training verdict prompt and the second prompt. Removals and edits act at every statement token. *Rung*: the place on the ladder (@ca-analyses); *setup* = chooses the setting; *–* = descriptive, not on the ladder.

#tbl(
  columns: (auto, auto, 1fr),
  align: (left, left, left),
  breakable: true,
  [name], [rung], [what is done],
  [*layer scan*], [setup, A-check], [Original model. The output of single MLP layers or blocks replaced by its mean over training statements, at all statement tokens or at the last one only; verdict accuracy, separation share, digit mass. Chose layers 11–17 (@tab-layer-scan).],
  [*B1*], [setup, A-check], [Original model. The layer scan repeated for MLP and attention outputs, single layers and blocks, mean and zero ablation; why attention is not decomposed.],
  [*B2*], [setup], [Original model on every test and out-of-distribution statement set under three candidate prompts: accuracy, AUC, digit mass per set. Chose the second prompt and which OOD sets are usable.],
  [*B3*], [4a (unsplit)], [Original model. Steering along the in-context truth direction at the statement tokens flips the verdict under both prompts, but the model still recalls the fact correctly afterwards.],
  [*G0*], [setup], [Original model, validation subset. Checks that the extra prompts of G2 (Yes/No questions, digit questions) are answered in format.],
  [*R* (recovery)], [0, 5], [Components only: every component at mask = CI, $Delta$ off. KL recovered, accuracy, agreement with the model, separation share, digit mass; test, all 24 OOD files, both prompts.],
  [*R-active*], [0], [As R, but only the active set at mask = CI, all other components and $Delta$ off.],
  [*R-mean*], [0], [As R, at the training's expected mask $upright("CI") + (1 - upright("CI")) \/ 2$.],
  [*active set*], [0], [Components ranked by mean CI; the curve of KL recovered when only the top $m$ are on ($m$ = 1, 2, 4, …, 512); the active set is the smallest such set reaching 95% of R's KL recovered (validation). No CI threshold defines "active" anywhere.],
  [*G2 Yes/No transfer*], [5], [R under two Yes/No truth questions ("… Yes"/"No" instead of 1/0): does the decomposition transfer to an answer format it was not trained on? Added after the plan.],
  [*T0*], [1, 4b], [All components removed from the full model ($Delta$ alone). KL recovered (ideal: 0, i.e. $Delta$ alone carries nothing of the verdict) and separation share.],
  [*R-inactive*], [0], [The full model with every component outside the active set removed; plus the norms of the active and inactive writes and their cosine.],
  [*cancellation norms*], [diagnostic], [At the statement tokens: $|C|$, $|D|$ and $chevron.l C, D chevron.r \/ |y|^2$ per layer. Large opposing writes = the components and $Delta$ cancel each other.],
  [*T1*], [1], [Each candidate component removed alone from the full model (mask 1 → 0, everything else kept). On validation for all candidates (this ranking feeds T2 and M-core), on test for the active set.],
  [*T2*], [1, 5], [The top $k$ candidates by T1 removed together, $k in {1, 2, 5, 10, 20, 50, 100, 200}$ plus the active-set sizes; against 5 random size-$k$ subsets of the active set, and the same rule on the untrained decomposition and the data-aligned basis. Test and 13 OOD files, both prompts (the random controls on OOD on at most 50 statements per file).],
  [*T2 Yes/No*], [1, 5], [T2 under the Yes/No prompts (G2).],
  [*T3 (A8)*], [1], [No decomposition in the forward pass: one layer at a time, the *original* MLP output loses its projection onto the span of (a) all 96 write vectors $u_c$ of that layer, (b) the active set's write vectors in it. Controls of the same dimension: 3 random subspaces and the top principal directions of the output.],
  [*Dir-remove*], [1], [The mass-mean or logistic-regression truth direction projected out of the original MLP output, per layer and in all seven layers together; compared with removing the components.],
  [*SE-Pile*], [0], [Side effect on general text: next-token KL on the Pile with all components off and $Delta$ on, from the training's evaluation.],
  [*G2 next-token*], [3 (removal)], [Side effect within the statement: next-token KL against the original at the statement tokens, split into prefix (before the object; truth not yet determined), recall position (predicts the object) and object + end; for $Delta$ alone and the reversed top-10 group, compared with all off.],
  [*G2 digit questions*], [3 (removal)], [Same-format control judgements: four questions about other properties of the statement (humour, emotion, surprise, word count), answered with a digit, under $Delta$ alone and the reversed top-10 group.],
  [*M-core*], [4a], [The top 10 components by T1, each alone and as a group, and the label-selective group (K2), reversed (mask −1) and amplified (mask 2); 10 random active components alone; the same rule on the controls. Flips, separation share and shift, digit mass; test and 13 OOD files, both prompts. Masks −1 and 2 lie outside the trained range.],
  [*M-twin*], [4a], [The one prediction fixed in advance: under an edit, does a negated statement's verdict follow the original verdict of its affirmative twin ("the 'not' is ignored")?],
  [*M-wide*], [4a], [Exploratory, validation only: every active component reversed alone, per family and label.],
  [*Dir*], [–], [Mass-mean and logistic-regression truth directions after layers 11–17 at the statement's last token (fit on training statements). Per component the cosine of its write with each direction, against random vectors; the share of the true–false gap along the direction carried by the active components, the inactive ones and $Delta$; the rank correlation of T1 with the gap share.],
  [*valley check*], [–], [Histograms of all CI values (validation): is there a gap between "off" and "on" that would define "active"?],
  [*seed comparison*], [all], [Every table shows all seeds side by side; no method is ranked above another where its seeds overlap.],
  [*K0*], [–], [Logistic probes over all features of one kind (all CI values; all coefficients; the residual stream after layer 17), fit on training statements, scored on test and OOD, for the label and the model's verdict. Reported as the reason probe accuracy is not evidence.],
  [*K1*], [–], [Per component the AUC for truth within each family; probes on the top $k$ components by training selectivity, $k in {1, 2, 5, 10}$, against the controls at the same $k$.],
  [*K2*], [–], [Components selective on validation within a sentence form ($|"AUC" - 0.5| >= 0.2$) keep their sign on the OOD sets of the same form ($|"AUC" - 0.5| >= 0.1$): share that transfers, against the controls.],
  [*K3*], [– (5)], [On the statements the model gets wrong: does a selective component follow the label or the model's verdict?],
)
