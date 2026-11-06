C: (transcribed from chat) I need another go at the categorization. Let's discuss this in a new conversation and also independent of any results so far.

C: (transcribed from chat) The approaches using RepE or the mediator taxonomy has mostly confused me instead of helped.

C: (transcribed from chat) I'll propose some tests / categories and you can comment on them:
Let's start from the perfect decomposition W = C + Δ (and also perfect separation of target / non-target data). Everything the model needs to determine the truth value lives in C all the rest in Δ (I'm ignoring the CI-values for now).
We can check C or Δ alone on target/non-target data. Basically by definition of "perfect decomposition" we sohuld have:
- C + target data: perfect recovery
- Δ + non-target data: KL = 0
Now what should happen if we swap them?
- Δ + target data: constant (not sure though)
- C + non-target data: even less sure what we should expect here

C: (transcribed from chat) Also I would say that:
- C + target data measures sufficency
but I'm less sure what is selectiviy or completeness are.

C: (transcribed from chat) I'd like to sort out these bits first and then keep going from there

---

26-11-05, Claude Opus 5.5 — *four-cell table: C and Δ alone on target and non-target data*; I have read INDEX.md, the paper-draft SUMMARY (`convos/julian/26-11-04_paper_draft_SUMMARY.md`) and the draft's methods section, for the masking setup only. As asked, I did not reread results or the earlier category schemes (RepE four, mediator-derived S0/A–F).

**Setup.** The *site* is whatever is decomposed (here `down_proj` of layers 11–17, at the statement tokens). W = C + Δ at the site. There are four variants of the model per input: **W** (original), **C alone** (Δ removed), **Δ alone** (C removed), **none** (the whole site removed). "Removed" needs one convention (zero or mean ablation), used the same way in all four.

### 1. The missing reference point: "none"

Your two uncertain cells become clear once "none" is used as the second reference next to W [concluded: argument below]. On each data type, one part should behave like W and the other like none:

| | target, truth readout (verdict) | target, everything else | non-target |
|---|---|---|---|
| **C alone** | ≈ W → *sufficiency* | no requirement | ≈ none → *C is silent* (diagnostic) |
| **Δ alone** | ≈ none → *completeness* | ≈ W → *selectivity, target side* | ≈ W → *selectivity, non-target side* |

Target: C ≈ W, Δ ≈ none (for truth). Non-target: Δ ≈ W, C ≈ none. The two halves of the table mirror each other.

### 2. Δ alone on target: "carries no more truth than none", not "constant"

- Whether Δ-alone's verdict is *constant* is a property of the **model**, not of the decomposition. It is whatever the model says when the truth computation is missing: a fixed default ("true"), or answers driven by surface features (negation, length). The decomposition can't influence that, so the criterion shouldn't predict it. The right criterion is that the verdict is **truth-independent**, judged against none [concluded].
- Why compare to none and not to chance: if the model has routes to the truth value that bypass the site (attention, other layers), then even a perfect decomposition leaves some truth signal in Δ-alone. The same amount is left in none. So completeness means "Δ alone carries no truth signal beyond none". Whether the site is a bottleneck is a separate question about the model.
- **Necessity and completeness are the same cell here.** Removing C from W leaves exactly Δ. Necessity only becomes separate when (a) CI comes back (removing the *active* components leaves Δ + the inactive ones), or (b) subsets are removed. Necessity of *individual* components is not a requirement of a perfect decomposition: with redundant routes inside C, each part can be dispensable on its own, as long as removing all of C kills the signal [concluded].
- This cell is where cancellation shows up. If C writes a large verdict signal s and Δ contains −s + (the original truth write), then Δ alone carries (anti-)truth.

### 3. C alone on non-target: "≈ none", a diagnostic rather than a requirement

- Expectation: C contributes nothing that isn't already missing, so C-alone ≈ none.
- It does not follow strictly from the non-target selectivity cell. That cell says adding C *to Δ* changes nothing, and the downstream is nonlinear. Both cells follow from the simpler property **C x ≈ 0 on non-target inputs** (C is silent). They come apart when C writes something on the Pile that the downstream ignores in context.
- Practical problem: on non-target data both C-alone and none are badly broken models, and a KL between two broken models is hard to read. Better to measure it directly at the site: |C x| / |y| on non-target inputs.
- This cell is the weight-level counterpart of "CI ≈ 0 on non-target". Once CI comes back, CI takes over its role.

### 4. Sufficiency (C alone on target): two caveats

- Compare against **W, not the labels**. A perfect decomposition reproduces the model's mistakes as well.
- The readout is one bit, so C can produce the verdict by a route of its own (re-implementation). Cell 1 alone can't rule that out. Cells 1 and 2 together, with C + Δ = W, catch the cancelling kind (section 2). They can't catch a C whose write on the target inputs *equals* the original truth write but comes from different weights. That case shows up only off the target distribution, or in the components' structure. It's an open question whether it should count as a failure at all.

### 5. Selectivity: two halves

"All the rest is in Δ" means that removing C removes **only** truth:
- non-target side: Δ alone ≈ W on non-target data (C isn't used for anything else);
- target side: Δ alone ≈ W on the target inputs, for everything that doesn't depend on truth (C carries nothing else the model does with the statements). This needs readouts that don't depend on truth, e.g. other questions about the statement, or the format of the answer.

### 6. What the idealisation assumes (worth knowing before CI comes back)

- **Shared mechanisms.** Without CI, a mechanism at the site that is used both for truth and on non-target text can't satisfy both sufficiency and non-target selectivity. Fact recall in these MLPs is a candidate. If the mechanism sits in C, Δ alone breaks the Pile. If it sits in Δ, C alone lacks it on the statements. With fixed weights, W = C + Δ doesn't allow it in both. CI exists to resolve exactly this: a shared component can be active on both data types. So the table holds exactly only for a mechanism used on target data alone. Once CI is back, "C" in each cell becomes "the components active on this input" [concluded].
- **Separation of the data.** The Pile contains factual statements. If the model evaluates truth there too, the mechanism is used on non-target data, and non-target selectivity can fail even for a perfect decomposition. That's a property of the non-target data, not of the decomposition.

### 7. Possible next step (not part of the four cells)

Swap test: run statement x with the site's write C x' + Δ x, where x' is x's opposite-truth twin. If C holds the truth computation and Δ doesn't, the verdict follows x'. This tests sufficiency and completeness at once and doesn't need to know what none outputs.

### Open points for Julian

1. Use "none" as the second reference (table above)?
2. Removal convention: zero or mean ablation, the same for C, Δ and none?
3. Is the target side of selectivity worth a column? It needs readouts that don't depend on truth.
4. Next: bring CI back, i.e. how the shared-mechanism case (section 6) changes the cells.

---
---
C: Alright. You've got good points. I think I have to start even more conceptuall simplified. Let's say we have a full targeted decomposition of a supposed truth function of the model f(W). As you said, part of the decomposition is actually the CI that selects the active components.

What are the properties of this decomposition? (Simplified I'll use W = Δ + C here where C is the C weighted part of W and W is actually the tuple of all weight matrices)
- Δ produces the same result as W on data without truth stimulus
- C computes all relevant activations relevant to use the truth value

Already on this level I have problems to express what a perfect decomposition should look like.

Answer (26-11-05, Claude Opus 5.5) — *perfect targeted decomposition with CI: properties P1–P6, defined by swaps*; for this answer I also read the masking and loss description in the paper draft's methods section (`reports/26-11-04_tpd_truth_paper_draft.typ`, *Masks* and the loss list). Nothing else is new, and still no results.

Your two properties are the right ones. They are hard to make precise because each hides a choice. Below: the choices, then a definition that makes them, then how it relates to the four-cell table and to what tPD actually trains.

**Notation.** At the site, the components are W_c with CI g_c(x). For a *perfect* decomposition I take CI to be 0 or 1. An intermediate value ("half needed") has no clean meaning. Active set A(x) = {c : g_c(x) = 1}, active part C_x = Σ_{c ∈ A(x)} W_c, inactive part I_x = the other components. Then for every input W = Δ + C_x + I_x. So with CI, "W = Δ + C" no longer holds as an identity. There are three parts, and the split depends on the input.

### 1. What property 1 hides

"Δ produces the same result as W on data without truth stimulus" is two statements:
- (a) **CI is honest:** on any input, removing the inactive components changes nothing, f(x; Δ + C_x) = f(x; W). This is SPD's core property, and it applies to all data, not only to non-target data.  
C: yes, that is something tangible and good
Answer (26-11-05, Claude Opus 5.5): kept as the first property (B1) of the simpler basis in the answer to your last comment, at the bottom.
- (b) **Targeting:** on inputs without truth stimulus nothing is active, A(x) = ∅.  
C: hmm, not sure about this. See below.
Answer (26-11-05, Claude Opus 5.5): agreed, (b) is too strong. Reason in the answer under P3; the basis at the bottom replaces it with "every component is needed for the verdict on some statement" (B3) and "nothing active that isn't needed" (B4).

Together they give your property 1, f(x; Δ) = f(x; W). It's worth keeping them apart because they fail for different reasons: (a) when CI is wrong, (b) when the components are used for other things.

### 2. What property 2 hides

"C computes all activations relevant to use the truth value" has three soft spots.
- **Whose truth value.** It has to be the model's truth value τ(x), not the label: a perfect decomposition reproduces the model's errors. So τ needs a definition before decomposing anything, e.g. the model's verdict under the verdict prompt. A "belief" that holds across prompts is a stronger τ, and then generality becomes part of the definition.  
C: the model's believe makes more sense, yes. So one assumption is that the model has a truth value τ(x). The model's verdict under the verdict prompt is something concrete that we use to measure τ.
Answer (26-11-05, Claude Opus 5.5): agreed. This is assumption A of the basis at the bottom: τ(x) exists, and the verdict *measures* it rather than *defining* it. One consequence: where verdict and belief come apart (prompt sensitivity, or a first digit that isn't an answer), that's measurement error in the readout, not a property of the decomposition. Whether the first digit is an answer was checked in-distribution in `convos/julian/26-11-05_verdict_digit_trust_SUMMARY.md`.
- **"Relevant": narrow or broad.** Everything the model does with a statement is *used* when it judges truth: reading the subject, recalling the fact, comparing it with the stated object. Only the result of the comparison *varies with τ*. Narrow reading: C holds only what carries τ.  
C: Is there a somewhat more concrete definition of "what carries τ"?  
Answer (26-11-05, Claude Opus 5.5): the most concrete one I know uses twin statements. Compare the runs on "Paris is in France" (true) and "Paris is in Spain" (false):
- recalling Paris → France happens in both runs, identically: it doesn't carry τ;
- encoding the object (France vs Spain) differs, but that's content: in another pair ("Rome is in Italy" true, "Rome is in France" false) "France" now comes with the opposite τ;
- the result of comparing the recalled fact with the stated object differs the same way in every pair: it carries τ.

So, observationally: a part of the computation carries τ if, across many pairs, its difference between the twins depends on τ and not on what the statements are about. That's "contains τ", what a probe finds. To say the model *uses* it you need a counterfactual: putting that part's value from x' into the run on x moves the verdict to τ(x'). I don't think "used" has a non-counterfactual definition, because two parts can both correlate with τ while the downstream reads only one of them [concluded].
Broad reading: C holds the whole computation from the input to τ. Under the broad reading, (b) is unattainable as soon as a step (fact recall, say) is also used on general text, because those components then have to be active there. I recommend the narrow reading [concluded: (b) and the narrow reading can hold together; (b) and the broad reading can't when steps are shared]. The shared steps then sit in Δ, or in components active on both data types, which are then not part of "the truth mechanism".
C: It would be nice to have the narrow reading, but I'm afraid we don't have the luxury to use the narrow reading as a basis because the experiments, clearly don't fall in there.
OUTDATED (26-11-05): the recommendation of the narrow reading in the bullet above is withdrawn; the broad reading is the basis now (answer below and basis at the bottom).
Answer (26-11-05, Claude Opus 5.5): agreed, and there's a reason that doesn't need the results: tPD's own objective asks for the broad reading. On the statements, the active components with Δ removed must produce the verdict (masks in [CI, 1], Δ's mask allowed down to 0, per the draft's methods section). So everything the verdict needs from the site has to be in the active components: reading the subject, recalling the fact, and even generic contributions of the MLPs that the downstream relies on [concluded]. Judging tPD by the narrow reading would grade it on something it doesn't aim at. How generic "everything the verdict needs" gets depends on the removal convention: with zero ablation the generic part has to be in C, with mean ablation the mean supplies most of it. The narrow question stays, but as a question asked about C ("which of its components carry τ"), not as the definition: Q2 in the basis at the bottom.
- **"C computes".** C can't compute anything alone. Removing Δ breaks everything else the site does, and every decomposed layer after the first then sees inputs the model never produces. So "C alone does X" is a statement about a hybrid model, not about the model. A cleaner way to say "C holds τ" keeps the model on its own inputs: swap writes between two statements.  
C: yes, that's the notion I'm all the time struggeling with, how to isolate C's effect from the rest of the model
Answer (26-11-05, Claude Opus 5.5; I missed this comment in the previous round): see the answer to your comment at the bottom (*holistic definition D1–D4*). For a *definition* I'd accept the C-only network after all: "the truth function is the sub-network C" is exactly the claim, and the C-only network is that sub-network. The isolation problem comes back when we *measure*, because then we read something off a network that never occurs naturally.

### 3. Proposed definition (narrow reading)

Fixed before decomposing: the site; the inputs with truth stimulus (approximated by the statements); the inputs without (approximated by the Pile); and τ.

A decomposition (components W_c, CI g, remainder Δ) is perfect if:
- **P1 exact:** W = Δ + Σ_c W_c. True by construction.
- **P2 honest CI:** f(x; Δ + C_x) = f(x; W) for every input, on all outputs.
- **P3 targeted:** A(x) = ∅ on inputs without truth stimulus. P2 + P3 = your property 1.  
C: This really depends on the narrow definition and also exactly what the narrow definition actually means. Does that mean that narrow definition says none of the computation on  "The city of Paris is" has nothing that carries τ? Because the sentence could be continued in a low truth stimulus way "The city of Paris is my home town"
Answer (26-11-05, Claude Opus 5.5): yes. Under the narrow reading nothing at the prefix "The city of Paris is" carries τ. The model reads left to right, so at those tokens it can't know whether a claim follows, and τ isn't determined before the object. So under the narrow reading "truth stimulus" is a property of positions, not of whole inputs, and P3 holds at the prefix trivially. Under the broad reading the prefix computation (reading "Paris", recalling its facts) is part of the computation that leads to τ, and it runs identically in "The city of Paris is my home town". The components doing it must then be active there too, and P3 fails. That's the shared-step case of section 6 of the four-cell entry, and your example shows it isn't a corner case: every statement's prefix is also ordinary text. So with the broad reading, targeting can't mean "nothing active on other text". The basis at the bottom uses B3/B4 instead.
- **P4 C carries τ, Δ doesn't:** take two statements x, x' with τ(x) ≠ τ(x') and the same token layout. Run x, but at every decomposed layer and position use the active components' write from the run on x' (and Δ's write from x). The verdict should follow τ(x'). Swap the other way (Δ's write from x', the components' write from x): the verdict should stay τ(x). The first half is "C has all of it", the second "Δ has none of it": your property 2.  
OUTDATED (26-11-05): P4 as written fails for a perfect decomposition when several layers are decomposed (Δ at a later layer passes on the τ that C wrote earlier); see the answer below.
C: This is good, yes. But it feels quite involved and it feels like theer is a step missing from perfect decomposition => P4. I mean you are basically describing a counterfactual setting here.
Answer (26-11-05, Claude Opus 5.5): right, nothing derives P4 from "perfect decomposition". P4 is a *definition* of "C carries τ", and it's a counterfactual. I think that's unavoidable: "C carries τ" is a causal claim (the verdict depends on τ through C), and a causal claim is about what happens when something is changed. Every test of "C does X" is a counterfactual, ablations included: "C alone" means "what if Δ's write were zero". So the choice isn't between counterfactual and not, only which counterfactual: removal (simple, needs no pairs, but runs a model on activations it never produces) or swap (stays on the model's own activations, needs aligned pairs).

Your unease also points at a real flaw in P4 as I wrote it. With several decomposed layers, Δ at layer 12 can pass on the τ that C wrote at layer 11. Taking "Δ's write from x" in the first swap then re-inserts τ(x), and taking Δ's write from x' in the second swap inserts τ(x'). A perfect decomposition fails both halves [concluded]. Computing the unswapped part live in the mixed run fixes the first half; the second half gets complicated. The basis at the bottom avoids the problem by starting from a single layer.
OUTDATED (26-11-05): the "flaw" in this paragraph (and the marker above Julian's P4 comment) comes from reading the swap's result at the verdict instead of at the model's internal truth value; read there, it disappears. See *holistic definition D1–D4* at the bottom.
- **P5 C carries only τ:** in the first swap of P4, outputs that don't depend on τ (what the statement is about, its next-token predictions, other questions about it) should still follow x.  
C: yes, this as well.
Answer (26-11-05, Claude Opus 5.5): under the broad reading P5 applies only to the τ-carrying components. The whole C also carries content (subject, recalled facts), so swapping all of C moves more than τ by design. In the basis at the bottom it sits under Q2.
- **P6 minimal:** as few active components as P2–P5 allow. Not needed for "perfect" in the behavioural sense, but needed for the components to be readable.  
C: Ideally minimality should mean something else, or maybe minimality should appear later in an argument for necessity or something like that
Answer (26-11-05, Claude Opus 5.5): yes, that works. Take minimality in the inclusion sense, not the count sense: no active component can be switched off without breaking another property. Then every active component is necessary where it's active, by definition, and that's the argument for necessity [concluded: switching off a component whose removal breaks nothing would give a smaller active set]. It's B4 in the basis at the bottom. Counting components ("as few as possible") isn't needed for that.

Two remarks:
- **P4 and P5 alone don't need a parameter decomposition.** If the model writes τ along one direction, the projection of the site's output onto that direction (as a weight matrix, P·W) would satisfy them. P2 and P3 are what make the definition a statement about *weights*: the same weights must be removable, and inactive, on text without truth stimulus. A direction the model also writes on general text fails P3. So P2–P3 are where tPD can claim more than a probe direction [concluded].
- **Twins differ in content as well as in τ** ("…in France" vs "…in Spain"). That's what P5 is for: if C also carries which country was stated, the swap moves that too.

### 4. Relation to the four-cell table

The cells of the earlier entry are ablation versions of these properties:

| ablation cell (entry *four-cell table*) | property |
|---|---|
| C alone on target, verdict ≈ W (sufficiency) | P4, first half |
| Δ alone on target, verdict ≈ none (completeness) | P4, second half |
| Δ alone on target, other readouts ≈ W (selectivity, target side) | P5 |
| Δ alone on non-target ≈ W (selectivity, non-target side) | P2 + P3 |
| C alone on non-target ≈ none (C silent) | replaced by P3 (CI = 0 there) |

Ablations need no pairs but run hybrid models. Swaps stay on the model's own inputs but need aligned pairs. Proposal: define perfection by the swaps, measure both, and read the ablations as cheaper proxies [assumed: that the swaps can be built from the tiu twin pairs; not checked].

### 5. What tPD's training asks for, against this definition

From the draft's masking description: on the statements the verdict must come out right for component masks in [CI, 1] and Δ's mask in [0, 1]. On the Pile, Δ's mask is 1, the components' masks are in [CI, 1], and every non-zero CI is penalised.
- Statements: P2 for the verdict only (the inactive components may be off), and sufficiency of the active part with Δ off. Not P2 for the other outputs at the statement.
- Pile: P2 and P3.
- Never inside the training range on the statements: the active components switched off. So the second half of P4 (Δ carries no τ) isn't trained, and neither is P5. A decomposition at zero loss can still have Δ write an anti-τ term that cancels a matching extra term in C, the cancellation case of section 2 of the earlier entry [concluded: C alone and C + Δ then both give the right verdict, and so does every mask in between, so this configuration meets every training condition listed above].  
C: I would push back on this. Obviously we can't just train P1-P5 directly. The goal of tPD is to have a proxy to train a gooddecomposition, the arguments for that are in the paper.
Answer (26-11-05, Claude Opus 5.5): agreed, and it wasn't meant as criticism of tPD. The use is for evaluation: properties the loss enforces only need a check that they hold on held-out data; properties it doesn't enforce are the ones whose presence is an actual finding. In the basis at the bottom that's the split between B1–B4 (what a tPD decomposition aims at) and Q1–Q3 (what it leaves open).

### Open points for Julian

1. Narrow reading (C holds only what carries τ) or broad (the whole computation leading to τ)? I recommend narrow.
OUTDATED (26-11-05): recommendation withdrawn, broad reading adopted as the basis (answers under Julian's comments on the narrow reading and at the bottom).
2. τ = the verdict under this prompt, or a belief across prompts?
3. Define perfection by the swaps (P4–P5) and treat the ablation cells as proxies?

C: I want to keep talking about the conceptual simple case. For me your open pointsare jumping ahead. Let's try to find a good simple basis on which we can base our investigation on.

Answer (26-11-05, Claude Opus 5.5) — *simple basis: one layer, broad reading, B1–B4 and Q1–Q3*; built from your comments above, still no results read.
OUTDATED (26-11-05): the one-layer simplification, B2 (stated with the verdict) and Q3 (left open instead of part of the definition) are replaced by *holistic definition D1–D4* in the answer to Julian's next comment.

Agreed, the open points jumped ahead. Here's a smaller basis. It takes over your comments: τ is the model's belief and the verdict measures it, the broad reading, minimality as the route to necessity.

**Simplifications.** One weight matrix W (one layer), binary CI. Several layers add the pass-on problem (answer under P4) and come later.

**Given:** the model; the site W; target inputs (statements) and non-target inputs (other text).

**Assumption A:** on a statement x the model has a truth value τ(x); the verdict measures it.

**Decomposition:** W = Δ + Σ_c W_c, with CI g_c(x) ∈ {0, 1}; C_x = the sum of the components active on x.

**What a perfect decomposition satisfies** (proposal):
- **B1 honest CI:** f(x; Δ + C_x) = f(x; W) on every input and every output. Switching off the inactive components changes nothing.
- **B2 C suffices for the verdict:** on statements, the verdict with only C_x at the site (Δ removed) equals the verdict with W. Everything the verdict needs from the site is in the active components (broad reading). This is what tPD trains, so the basis uses the method's own aim.
- **B3 targeted:** every component is needed for the verdict on some statement. Whatever the verdict never needs belongs to Δ.
- **B4 minimal:** on every input, switching off any one active component breaks B1 or B2. This makes each active component necessary where it's active (answer under P6).

B3 and B4 replace "nothing active on other text": a component may be active on "The city of Paris is my home town" if it is part of the verdict computation (B3) and that text uses it (B4).

**What B1–B4 leave open, i.e. the evaluation questions:**
- **Q1 sharing:** how much of C is active on non-target text? Your property 1 (Δ alone = W) holds exactly on text that uses none of the components. Given B1–B4, how much is shared is a fact about the model (how general-purpose the verdict computation at the site is), not a defect of the decomposition.
- **Q2 what in C carries τ:** the narrow question, asked inside C instead of as a definition (twin pairs, answer under "what carries τ"). Under it: P5, whether the τ-carrying components carry only τ.
- **Q3 does Δ carry τ too:** B2 says C suffices, not that Δ is free of τ. A redundant copy of the τ computation in Δ, or cancellation (C writes s, Δ writes −s plus the original write), meets B1–B4 [concluded: B1 holds because Δ + C_x = W at the site; B2 holds because C_x alone gives the verdict]. That's completeness.

Each Q needs a counterfactual (removal or swap), which is the next thing to settle, but only once the basis holds.

Does this match what you have in mind as the basis? In particular B2: it defines the truth mechanism at the site as "what the verdict needs from the site", which is the method's own definition, broad, and depends on the removal convention (answer under the narrow-reading comment).

---
---

C: I'm not happy yet. What I would like is a definition of "perfect decomposition". B1 does sound like a good starting point. B2 however, can't be part of the definition as it's already using a proxy for τ. Also, I don't know what I should think of the move to say, we just look at single layers. Does that make sense? You can convince me. So is the idea to say decomposition is perfect if we have B1-B4 on every site? Don't we miss something holistic here?

Answer (26-11-05, Claude Opus 5.5) — *holistic definition D1–D4, stated with τ instead of the verdict*; nothing new read.

Three points, then the definition.

**B2 and the proxy.** Agreed. The fix is to state it with τ itself. That needs your assumption one step more precise: τ is not only a property of the statement but a variable *inside* the model. Call it T: a quantity in the model's activations through which every truth-dependent part of its behaviour passes, with T = τ(x) on statement x. Then "C computes τ" can be said without any readout: in the network with only C, T takes the value τ(x). The verdict comes back only when we measure. This also removes something B2 smuggled in: to give a verdict, C would also have to contain the machinery that turns τ into "1"/"0". Stated with T, it doesn't.

**Single layers.** I withdraw it. It made the swap tests easy to write down, but it isn't a definition. The truth function can span layers (one layer recalls, a later one compares), and "B1–B4 at every site" splits it into pieces that can each look fine while the whole doesn't. Removing Δ at one layer with the others intact is a different statement from removing Δ everywhere at once. Your original setup, W as the tuple of all weight matrices, is the right one.

The single layer was my workaround for the flaw I claimed in P4: Δ at layer 12 passing on the τ that C wrote at layer 11. With T that flaw disappears [concluded]:
- If Δ passes τ on *before* T is formed, that step is part of computing τ, the decomposition isn't perfect, and P4 failing is correct.
- If it passes τ on *after* T, it doesn't touch T. It only mattered because I read the swap's result at the verdict, downstream of T.

**Holistic.** Yes. What makes the definition below holistic is that D2 and D3 talk about all matrices at once: C as a whole is a sub-network that computes τ, Δ as a whole is a network that doesn't. Something else could count as "holistic": requiring C to be *one* mechanism rather than a set of parts. I'd leave that out. Redundant routes inside C are legitimate, and D4 already keeps unrelated parts out. If you meant something else by "holistic", say so.

### Definition (proposal)

**Setting.** W = (W¹, …, W^L), all weight matrices of the model. A decomposition writes each W^l = Δ^l + Σ_c W_c^l, with CI g_c(x) ∈ {0, 1}. C_x = the tuple of the parts active on x; Δ = the tuple of remainders. "The network with only C_x" means every matrix replaced by its C_x part; the same for Δ.

**Assumption A (the model has a truth value).** There is a variable T in the model's activations through which all truth-dependence of its behaviour passes; on statement x, T = τ(x). The verdict is a measurement of T.

A decomposition is **perfect** if:
- **D1 honest CI:** the network with Δ + C_x gives f(x; W), for every input and every output (= B1).
- **D2 C computes τ:** for every statement x, T = τ(x) in the network with only C_x.
- **D3 Δ doesn't:** in the network with only Δ, T doesn't depend on τ.
- **D4 nothing superfluous:** no component can be moved into Δ, and no active component can be switched off on an input, without breaking D1–D3. Then every part of C is necessary (your remark under P6).

**Where your two original properties went.** Your property 2 ("C computes everything relevant to τ, the rest is in Δ") is D2 + D3. Your property 1 ("Δ gives the same as W on text without truth stimulus") isn't an axiom any more; it follows from D1 + D4 on any text that uses no part of the truth function. On text that does use part of it ("The city of Paris is my home town"), exactly the parts it uses are active, which is how it should be. The target data enters only through τ, i.e. D2 and D3 are statements about statements, and D4 keeps C restricted to what computing τ needs.

**What this definition costs:**
- **It is broad.** D2 asks the C-network to compute τ from the input on its own, so everything τ needs (reading the subject, recalling the fact) is in C. You argued we need the broad reading anyway.
- **It uses the C-only and Δ-only networks**, the hybrids you struggle with (your comment under "C computes"). For a definition I think that's fine: "the truth function is the sub-network C" is the claim, and the C-only network *is* that sub-network. The difficulty returns when we measure T in a network that never occurs naturally.
- **Idealisation: every weight matrix is decomposed.** In the experiments only some are, and the rest of the model stays in both networks and may compute part of τ itself. Then "T doesn't depend on τ" in D3 has to become "no more than with the decomposed matrices removed entirely" (the "none" reference of the four-cell entry). That's for later.

Does this work for you as the definition?

C: Yes, this looks much better. Under one clarification though. What does "T doesn't depend on τ" mean in D3? That's important for me to understand to judge these definitions


Answer (26-11-05, Claude Opus 5.5) — *what "T doesn't depend on τ" means in D3*; nothing new read.

Good question: as written it's underdefined, and the obvious readings don't work.

**Why it isn't trivial.** τ(x) is itself a function of the statement x (the model's belief about x). In the Δ-only network, T is also just a function of x; call it T_Δ(x). So "T_Δ doesn't depend on τ" can't mean "T_Δ doesn't depend on x". Any feature that fully describes x ("subject Paris, object Spain") also determines τ.

**Two readings that don't work:**
- *T_Δ is constant.* Sufficient, but too strong. Without the truth computation, the network may answer from surface features (negation, length, particular words), or T may not even be a clean value in that network. That's how the model behaves when the truth computation is missing, not a defect of the decomposition (same argument as for "Δ alone on target" in the four-cell entry).
- *T_Δ is uncorrelated with τ on our dataset.* Too dependent on the dataset. If false statements are negated more often, a Δ that only reads negation correlates with τ without computing anything about truth. And uncorrelated is weaker than independent: a non-linear dependence would slip through.

**Proposal.** D3 means T_Δ carries **no information about τ** (statistical independence, not just zero correlation) on a set of statements built so that nothing except the truth computation itself predicts τ:
- **twin-balanced:** every statement has a twin of the same form with opposite τ ("Paris is in France" / "Paris is in Spain"), so form, negation and length tell you nothing about τ;
- **crossed:** every subject and every object occurs in true and in false statements (France is true for Paris, false for Rome), so no single word tells you τ either.

On such a set, the only route to τ is relating subject and object, i.e. the truth computation. "T_Δ independent of τ" then says exactly what we want: Δ doesn't do that computation. Balance is with respect to the model's τ, not the labels, because the model's errors count as its truth value.
OUTDATED (26-11-05): "the truth computation" in the paragraph above conflates belief and truth; the only route to τ on such a set is the model's own belief computation, whatever it relates, heuristics included. See the answer under Julian's comment on belief vs truth below.

**Concretely**, independence means: no rule of any form, reading T_Δ, predicts τ better than chance. For twin pairs it can be put more simply: the difference T_Δ(x) − T_Δ(x') shows no systematic relation to which twin is the true one.

**What it catches:**
- Δ computes τ itself (a redundant copy): T_Δ predicts τ.
- Cancellation (C writes s plus the original write, Δ writes −s): T_Δ predicts τ with flipped sign, which counts as dependence too.

**What it adds to the setup:** like τ and the site, the statement set has to be fixed before decomposing, and the definition depends on it. That's unavoidable, I think: "doesn't carry τ" only has a meaning relative to which variation in the statements counts as τ and which as form or content [concluded: τ is a function of x, so independence from τ is only defined relative to a distribution over x].

In the partial case (only some matrices decomposed), D3 becomes: T_Δ carries no more information about τ than T does with the decomposed matrices removed entirely.

C: Ok. I'm fine with D3 being defined in words and then we admit that it's hard to define concretely what this actually means and then use something like twin-balance to get proxies. It's a bit funny to ask, what remains if you take out the believe function of a model and I see that you agree that this is a hard question. I'm a bit concerned that the independence answer conflates believe and truth? (I hope not)

Answer (26-11-05, Claude Opus 5.5) — *does D3's independence conflate belief and truth?*; besides this LOG I used only the verdict-trust results (`convos/julian/26-11-05_verdict_digit_trust_SUMMARY.md`) and the in-distribution / OOD accuracies of B2 re-read this session.

Agreed on the plan: D3 in words, twin balance as the proxy.

**"What remains if you take out the belief function"**: D3 is stated negatively, so it doesn't have to answer that. It doesn't say what the Δ-network does, only what it lacks: whatever it computes carries no information about τ. That's why I'd keep the negative form even in words.

**Belief vs truth: the definition doesn't conflate them, but two things around it do.**

1. *The definition.* D2 and D3 are about τ, the model's belief, and nothing in them mentions the label. A C that reproduces the model's errors meets D2. A Δ that holds correct knowledge the belief doesn't use is not ruled out by D3 as such [concluded: D3 asks for independence from τ only]. A Δ that judges truth by its own route does fail D3, because its output agrees with τ wherever the model is right. That's intended: it's a second judging route.

2. *My wording.* I wrote that on a balanced set "the only route to τ is relating subject and object, i.e. the truth computation". That conflates them, and I've marked it OUTDATED. The belief function is whatever the model does to arrive at τ, including heuristics that track plausibility or familiarity rather than truth. If the model uses such a heuristic, it is part of the belief function and belongs in C. On a balanced, crossed set the only route to τ is the model's own belief computation, whatever it relates.

3. *The proxy.* Twins are built from ground truth ("Paris is in France" / "Paris is in Spain"), so a twin set is balanced with respect to the *labels*. Where the model gets one twin wrong, both twins have the same τ and the pair is not balanced with respect to the belief. A label-balanced set therefore tests dependence on truth, which equals dependence on τ only where the model is right. Fix: balance with respect to the verdict, i.e. keep the pairs whose two verdicts differ. The cost is that the verdict only *measures* τ (the verdict-trust check found the first digit reliable in-distribution, with "0.5" hedges at near-ties). In-distribution the difference is small (verdict accuracy 0.979, B2); OOD it is large (0.801) [verified: B2 accuracies, re-read this session].

**The real limit is in the data, not the definition.** On statements where the model is right, belief and truth coincide, and no measurement can tell "C computes the belief" from "C computes the truth". Only the model's errors separate them. So any test of whether a decomposition follows belief rather than truth needs statements the model gets wrong, mostly OOD. That's the same reason the paper's truth-vs-verdict test (K3, `convos/julian/26-10-22_tpd_truth_paper_SUMMARY.md`, OOD budget) keeps all OOD files.

---
---
Now that we have some sort of working definition for the "perfect decomposition". Can you do a search for other works that have tried to come up with a definition?

26-11-05, Claude Opus 5.5 — *literature: who else has defined a perfect decomposition or mechanism*; 16 web searches (abstracts, search summaries and paper pages; no full paper read). Claims about what a paper says are tagged [verified: abstract/summary] when the search result stated them, [assumed: from memory] otherwise.

**Short answer.** Nobody I found defines a perfect *targeted parameter* decomposition with an internal truth variable. But each of D1–D4 and assumption A has a close relative, and two bodies of work could give the definition a ready-made formal language: **causal abstraction** for D2/D3, and **concept erasure** for D3's "no information about τ". The combination (weights instead of activations, a belief variable, a remainder that must be free of it) looks like ours [assumed: limited search].

### Mapping onto D1–D4 and assumption A

| ours | closest prior work | what it adds / where it differs |
|---|---|---|
| **A** the model has a truth value T, measured by the verdict | Herrmann & Levinstein, *Standards for Belief Representations in LLMs* (Minds and Machines 2025): four adequacy conditions for calling a representation a belief: accuracy, coherence, uniformity, use [verified: abstract/summary]. Levinstein & Herrmann, *Still no lie detector* (2023): truth probes fail to generalize in basic ways, plus conceptual reasons why probing alone can't find beliefs [verified: abstract] | Gives criteria for *when positing T is justified*: we currently just assume it. "Use" is a causal requirement, like our D2/D3. "Uniformity" (one mechanism across topics) is a candidate for your "holistic" worry |
| **D1** honest CI | SPD (Bushnaq, Braun, Sharkey 2025): causal importance = how ablatable a subcomponent is on an input; important ones not ablatable, ablatable ones unimportant [verified: summary]. APD (Braun et al. 2025): faithfulness = components sum to the parameters [verified: abstract] | D1 is SPD's own core property; P1 (exact sum) is APD's faithfulness. Nothing new needed |
| **D2** C-network computes τ | Circuit faithfulness: Wang et al., IOI (2022): the circuit alone recovers the model's behaviour [verified: summary]. Hanna et al., *Have Faith in Faithfulness* (COLM 2024): ablating everything outside the circuit leaves task behaviour intact [verified: abstract]. Ayonrinde & Jaburi (AIES 2025) separate *explanatory* faithfulness (matches the model's internal explanation) from *behavioural* faithfulness (same outputs) [verified: abstract] | The circuit literature states faithfulness at the *output*; Ayonrinde & Jaburi's distinction is exactly why we moved from the verdict (B2) to T (D2) |
| **D2 + D3** stated with an internal variable T; the swaps (P4) | **Causal abstraction** (Geiger et al., *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*, arXiv 2301.04709): a high-level causal model (here: one with a variable T) is aligned to the network; interchange interventions set internal variables to their values on another input; interchange intervention accuracy (IIA) measures agreement; distributed interventions target subspaces [verified: summary] | Our assumption A *is* an alignment claim: a high-level variable T realised in the network. Our P4 swaps *are* interchange interventions. Difference: causal abstraction aligns T with *activations* (a subspace), tPD with *weights* (a sub-network that computes it). Adopting its vocabulary would make D2/D3 precise and give a "perfect" criterion (IIA = 1) for free |
| **D3** Δ-network carries no information about τ | **LEACE** (Belrose et al., NeurIPS 2023): *linear guardedness*: no linear classifier beats the majority-class baseline at predicting the concept; equivalent to zero cross-covariance with the concept [verified: summary]. Amnesic probing (Elazar et al., TACL 2021): remove a concept, check whether the task still works [verified: summary, via LEACE] | Our "no rule predicts τ above chance" is guardedness with all rules instead of linear ones; linear guardedness is the obvious measurable proxy. Amnesic probing is your "what remains if you take out the belief function" asked as an experiment. Its known weakness (INLP removes hundreds of dimensions, so effects can't be pinned on the concept) is the same as "Δ-only is a broken network" |
| **D3, holistic version** | Causal scrubbing (Chan et al., Redwood, 2022): a hypothesis defines which activations can be *resampled* from other inputs without changing behaviour; tested on the whole network at once [verified: summary] | A behavioural, resampling version of D3: replace Δ's activity on statement x with its activity on a random other statement; if T doesn't move, Δ carries no τ. Stays closer to the model's own activations than zero ablation |
| **D4** nothing superfluous, every part necessary | Wang et al.: *minimality* (every circuit node matters for some subset) and *completeness* (circuit and model respond alike to removing parts) [verified: the three criteria and that they were measured; exact definitions assumed: from memory]. APD: minimality = as few components as possible [verified: abstract]. Craver's *mutual manipulability* (philosophy of science, 2007): a part is a component of a mechanism iff intervening on the part changes the phenomenon and intervening on the phenomenon changes the part [verified: summary] | APD's minimality is the count sense we dropped; Wang's minimality and Craver's criterion are the inclusion sense of D4. Craver is the standard philosophical answer to "what makes a part a component of the mechanism", the question D4 answers |

### Two warnings from the literature

- **Non-identifiability** (Méloux et al., ICLR 2025): on small networks, several circuits reproduce the same behaviour, one circuit has several interpretations, one algorithm aligns with several subspaces [verified: abstract]. So D1–D4 may be met by several different decompositions; D4 (inclusion-minimal) doesn't make the perfect decomposition unique [concluded: inclusion-minimal sets need not be unique]. Worth deciding whether "perfect" should be unique or only "one of the admissible ones".
- **Ablation choice decides the result** (Miller et al., *Transformer Circuit Faithfulness Metrics are not Robust*, COLM 2024): faithfulness scores change with seemingly minor ablation choices, and which task a circuit performs depends on the ablation [verified: abstract]. "The network with only C" in D2/D3 silently picks zero ablation. Same open point as zero vs mean ablation in the four-cell entry.

### Not checked

- The tPD paper itself (Vigouroux & Sharkey, arXiv 2607.13047): whether it states a target for a perfect targeted decomposition. Search found it but not its text [verified: listing only]. Worth reading before claiming novelty.
- *Open Problems in Mechanistic Interpretability* (Sharkey et al., TMLR 2025) has a decomposition section on whether current methods lack theoretical foundations; its text wasn't returned [verified: table of contents only].
- *The Quest for the Right Mediator* (already used in `convos/julian/26-11-04_paper_draft_SUMMARY.md`) not revisited.

### Suggestion

[assumed: my judgement] Restate D2/D3 in causal-abstraction terms: a high-level model with variable T, and the claim that T is realised by the C-network (D2) and that the Δ-network's state is guarded with respect to T (D3, linear guardedness as the first proxy). That links the definition to an established theory, makes the swap tests standard interchange interventions, and leaves tPD's actual novelty clear: the variable is realised by weights, not by a subspace of activations. Reading next, in this order: Geiger et al. (definitions of abstraction and interchange interventions), Herrmann & Levinstein (when A is justified), the tPD paper.

Sources:
- Geiger et al., Causal Abstraction: https://arxiv.org/pdf/2301.04709
- Braun et al., APD: https://arxiv.org/abs/2501.14926
- Bushnaq, Braun, Sharkey, SPD: https://arxiv.org/html/2506.20790v2
- Wang et al., IOI circuit: https://arxiv.org/abs/2211.00593v1
- Hanna, Pezzelle, Belinkov, Have Faith in Faithfulness: https://arxiv.org/abs/2403.17806v2
- Ayonrinde & Jaburi, Mathematical Philosophy of Explanations in MI: https://arxiv.org/abs/2505.00808
- Belrose et al., LEACE: https://arxiv.org/pdf/2306.03819
- Chan et al., Causal Scrubbing: https://lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing
- Herrmann & Levinstein, Standards for Belief Representations: https://arxiv.org/pdf/2405.21030
- Levinstein & Herrmann, Still No Lie Detector: https://arxiv.org/pdf/2307.00175.pdf
- Méloux et al., Is MI Identifiable?: https://arxiv.org/abs/2502.20914v1
- Miller, Chughtai, Saunders, Faithfulness Metrics not Robust: https://arxiv.org/abs/2407.08734v1
- Csordás et al., Are Neural Nets Modular? (weight masks): https://arxiv.org/abs/2010.02066
- Craver, mutual manipulability (overview and critiques): https://philarchive.org/archive/CRACRA-3 ; https://philsci-archive.pitt.edu/23510/1/Amin,%20The%20Philosopher's%20Error,%20PSA2024.pdf
- Sharkey et al., Open Problems in MI: https://arxiv.org/abs/2501.16496
- Vigouroux & Sharkey, tPD: https://arxiv.org/pdf/2607.13047

---
---

C: can you check the Geiger paper in detail (first couple of pages). It looks mathy which can be good or just be general abstract nonsees. I don't know, haven't checked.
Answer (26-11-05, Claude Opus 5.5) — *Geiger et al. read: useful core, heavy machinery, one sharp point for us*; read myself from the PDF (JMLR 26 (2025) 1–63, arXiv v4): pp. 1–10 (abstract, intro, causal models, interventions, start of the intervention algebras), pp. 13–26 (exact transformation, alignment, constructive abstraction, approximate transformation, interchange interventions, IIA, the worked neural-network example), pp. 37–41 (ablation, concept erasure, sub-circuit analysis, causal scrubbing). Skipped: rest of 2.2 (intervention algebras), 2.7 (bubble sort), 3.1–3.4, 3.6–4. Page and definition numbers below are from that version.

**Verdict: real maths, not abstract nonsense, but most of it is machinery we don't need.** The part we need is about four pages and concrete. About a third of the formal content (the intervention algebras of 2.2, Defs 15–24 with normal forms and lemmas) exists to generalize from fixing variables to editing mechanisms and rotating coordinates, so that subspaces of activations can count as variables. That's a sound goal, but for us it's background. Section 3 recasts existing methods (patching, ablation, concept erasure, circuits, causal scrubbing) in the framework. Correct, but mostly translation: for ablation and concept erasure the high-level model is a trivial three-variable model with a binary "was it ablated?" variable. The authors say so themselves: "Ablation studies attempt to determine *whether* a concept is used by a model; they do not characterize *how* that concept is used" (p. 39).

**The core, in plain terms** [verified: read]:
- *Causal model* (Def 5, p. 7): variables, each with a mechanism that computes its value from the others. In the neural-network example (p. 24) the variables are the neurons; the **weights sit inside the mechanisms** (F_H(n) = ReLU(nW)), they are not variables.
- *Intervention* (Defs 9–11, p. 8): hard = fix variables to values; soft = replace a variable's mechanism; *interventional* = any map from old mechanisms to new ones.
- *Alignment* (Def 31, p. 16): each high-level variable gets a cell of low-level variables plus a map π from that cell's values to the high-level value. Leftover low-level variables go into a "forgotten" cell Π⊥.
- *Constructive abstraction* (Def 33 via Def 25, pp. 13, 17): the high-level model H abstracts the network L if "intervene on L, then translate" equals "translate, then intervene on H" for every intervention in the domain (a commuting diagram).
- *Interchange intervention* (Def 44, p. 21): run the base input, but set some variables to the values they take on a source input. *Distributed* version (Def 46, p. 22): rotate first, swap in the rotated coordinates, rotate back.
- *IIA* (Def 48, p. 23): the share of interchange interventions on which L and H give the same output; IIA = 1 is exact abstraction under interchange interventions. A graded version (Def 41, p. 20) uses any distance and any statistic.
- Worked example (2.6, pp. 23–26): hierarchical equality ("are both pairs equal or both unequal?"), a tree algorithm, and a hand-built 3-layer ReLU network whose weights are written out. The alignment is explicit (first four hidden neurons ↔ "pair 1 equal", next four ↔ "pair 2 equal"), and interchange interventions on those neurons produce the algorithm's counterfactual outputs. The example is concrete and a good model for what our assumption A claims.

**What it means for D1–D4** [concluded unless tagged]:

1. *Assumption A becomes a high-level model.* Input = the statement, intermediate variable T, output = the verdict, plus an alignment Π_T: which activations carry T and how to read it (π_T). That's a real cost: our D2/D3 assumed T is "somewhere"; causal abstraction makes us say where.  
C: What is "Assumption A"? Why is that a real cost?
Answer (26-11-05, Claude Opus 5.5): Assumption A is from the entry *holistic definition D1–D4*: "the model has a truth value": there is a variable T in the model's activations through which all truth-dependence of its behaviour passes, with T = τ(x) on statement x, and the verdict measures T. Why a cost: D2 and D3 talk about "T in the C-network". Causal abstraction only gives that a meaning once we say *which* activations carry T and how T's value is read from them (the alignment). Finding that location is a separate empirical problem, and the test depends on it: put T in the wrong place and even a perfect decomposition fails. In the suggestion below the cost largely goes away: the decomposition itself proposes where T lives (in the active components' writes), so we test that proposal instead of locating T first.


2. **The sharp point: hybrid networks are outside what the theory can interpret.** For intermediate variables the alignment map π is induced from inputs: a low-level setting gets the high-level value it has on the input that produces it (Def 47, p. 22). The authors: "π will only be defined for values of intermediate partition cells that are realized when some input is provided to the low-level model ... we are only able to interpret low-level *interchange interventions*" (p. 22). The C-only and Δ-only networks produce activations no input produces, so T *cannot be read* there. That's a formal version of your problem of isolating C from the rest of the model. It also says D2/D3 as written (T in the C-only / Δ-only network) are not well defined unless T is given an extension beyond natural activations. It's a strong argument for stating D2/D3 with swaps rather than removals.

3. *Weights as interventionals.* Replacing W by C_x everywhere is an interventional in their sense (Def 11: mechanisms → new mechanisms). So "the network with only C" is expressible, but by point 2 its T isn't readable. For the swaps (P4) one needs a low-level model in which C's write and Δ's write at each site are separate variables that are then summed. That's a refinement of the network, not a rotation, so it doesn't fit Def 46 directly [concluded: the sum C_x·a + Δ·a cannot be inverted back into its two summands, so it isn't a bijective change of coordinates].

4. *Their completeness is causal, not informational.* In a constructive abstraction, swapping only the cell Π_T must move the output as H predicts. If the forgotten variables (here Δ's activity) carried a redundant copy of τ *that the output read*, swapping Π_T alone would fail and the abstraction would fail. A τ copy that the output never reads is allowed [concluded: from Defs 25, 33 and 44: only the commuting of outputs under interventions on aligned cells is required]. So their analogue of D3 is "Δ's τ information is not *used*", weaker than our "no information" (guardedness). That is the "contains vs uses" split from the answer under "what carries τ". Which one D3 should be is now a choice.

5. *Causal scrubbing as a holistic D2 + D3 test* (3.5.3, pp. 40–41). Every connection is resampled. Connections outside the hypothesized circuit get values from random inputs; connections inside get values from inputs that agree with the base on the high-level variable. Translated: compute Δ's contribution on a random other statement and C's contribution on a statement with the same τ; if the verdict survives, C carries what is needed and Δ carries nothing needed. It stays on activations the model produces, and it is about the whole network at once. Of the measurements I've seen, it's the closest to your "holistic".

**What I'd take from it:** the vocabulary (high-level model, alignment, interchange intervention, IIA), point 2 as the reason to define with swaps, the D3 choice in point 4, and causal scrubbing as the template for a holistic test. Sections to read if you look yourself: 2.3.2 and 2.5 (pp. 16–17, 21–23) and the example in 2.6 (pp. 23–26); skip 2.2.

Open question for Julian: D3 informational ("Δ carries no information about τ") or causal ("nothing downstream uses Δ's information about τ", as in causal abstraction)? The causal version is weaker, measurable with swaps, and is what tPD's loss could plausibly reach. The informational version is stronger and needs guardedness-type probes.

---
---

C: give me your suggestion wtith new language and causal D3.

Answer (26-11-05, Claude Opus 5.5) — *perfect decomposition in causal-abstraction language: the decomposition proposes the alignment*; builds on D1–D4 and on the Geiger et al. reading above; nothing new read.

**Idea.** Don't locate T first and then ask whether C computes it. Let the decomposition *be* the claim about where T lives: **the writes of the active components carry T**. Causal abstraction then says exactly what that claim requires, and the requirement is a swap test that stays on the model's own activations.

### Setting

- *The model as a low-level causal model* (Geiger's L): variables are the activations; the weights sit in the mechanisms. At each decomposed site l and position t, split the site's output into two variables that are then summed: the **C-write** c_l,t(x) = Σ_{active components} W_c^l a and the **Δ-write** d_l,t(x) = (Δ^l + inactive components) a. This refines the network without changing what it computes (D1 makes the inactive components irrelevant).
- *Assumption A as a high-level model H:* statement X → truth value T → verdict Y, with Y = T. In words: the verdict depends on the statement only through T. That's the old assumption A, now as a causal model.
- *Alignment proposed by the decomposition:* X ↔ the input tokens, Y ↔ the verdict, **T ↔ all C-writes (every decomposed site, every statement position)**. Δ-writes and everything else go into the "forgotten" cell. T's value on the C-writes is read as in Geiger's Def 47: the C-writes produced by statement x mean τ(x).
- *Statement distribution P:* pairs of statements with the same token layout, balanced with respect to the verdict (from the answer on belief vs truth).

### Definition (proposal)

A decomposition (components, CI, Δ) is **perfect** if:

- **E1 honest CI** (= D1): replacing the inactive components by nothing changes no output on any input.
- **E2 the C-writes are T** (replaces D2 and D3): for every pair (x, x') in P, run x but set all C-writes to their values on x'. Everything else, Δ-writes included, is recomputed by the model as usual. The verdict equals the verdict on x'. In Geiger's terms: H is a constructive abstraction of the model under this alignment, i.e. interchange intervention accuracy (IIA) = 1.
- **E3 nothing superfluous** (= D4): no component can be moved from the C-cell into Δ, and no active component switched off on an input, without breaking E1 or E2.

**Why E2 contains both D2 and the causal D3.**
- *C suffices* (D2): the swapped-in C-writes alone carry the verdict over from x'.
- *Δ isn't used* (causal D3): Δ-writes are recomputed from the base statement x's tokens and the mixed activations. If Δ carried a route of its own from x to τ(x) that the verdict read, the verdict would be pulled back towards τ(x) and E2 would fail [concluded: E2 asks the verdict to follow x' for every base x, including bases with the opposite τ]. A τ-copy in Δ that nothing reads is allowed. That's the causal D3 you asked for.
- *No hybrid networks.* Only C-writes are fixed, to values the model actually produces on x'. Everything downstream, including the machinery that turns T into "1"/"0", runs as usual. So C doesn't need to contain that machinery (your objection to B2), and T is never read off activations no input produces (the problem from the Geiger reading, point 2).
- *The pass-on problem across layers* (answer under P4) disappears for the same reason: later Δ-writes are recomputed from the mixed state rather than taken from x or x'.

**Graded version and the partial case.** In practice IIA < 1. In the experiments only some sites are decomposed (MLPs 11–17 at the statement tokens), and attention or other layers may carry τ by routes that bypass them. Then even a perfect decomposition can't reach IIA = 1. The reference is the whole site: swap *all* writes at the decomposed sites (C + Δ) from x' and call that IIA_site. Perfect in the partial case:
- IIA(C-writes) = IIA_site: C carries everything the sites contribute;
- IIA(Δ-writes alone) at the no-swap level: Δ contributes nothing that is used (the causal D3, measured directly).
IIA can use the verdict or the score s (Geiger's Def 41 allows any similarity), so near-ties don't have to count as full flips.

**What this changes against D1–D4:**
- One condition (E2) instead of two (D2, D3), and it is a definition and a test at once.
- T doesn't have to be located separately; the decomposition's claim is what gets tested. The remaining assumption is H itself: the verdict depends on the statement only through one variable.
- The verdict is back as the readout, but only as H's output under interventions, the standard in causal abstraction. C no longer has to *produce* the verdict on its own, which was the problem with B2.
- Dropped: P5 (C carries only τ). Under the broad reading the C-writes also carry content (subject, recalled facts), so swapping them moves content too, by design. The narrow question stays as Q2: which part of the C-cell can be swapped alone, i.e. a finer alignment with intermediate variables in H.
- The informational D3 (guardedness) becomes an optional, stronger property, not part of "perfect".

**Costs and open points:**
- *Same-layout pairs.* C-writes are per position, so x and x' need the same token count and aligned positions, e.g. twins differing in one object token [assumed: that tiu twins can be paired this way; not checked].
- *Content in the swap.* Twins differ in the object ("France" vs "Spain"). Swapping all C-writes also carries x''s object into the run. The verdict then judges "subject of x with object of x'" rather than τ(x'), unless the object enters only through C. A cleaner pair design would keep the content and vary only τ, e.g. affirmative vs negated forms of the same fact. But "not" adds a token, which clashes with the same-layout requirement unless positions are aligned by hand. Same content, opposite τ and same layout don't come together easily [assumed: not checked against the datasets].
- *Non-identifiability.* Several decompositions may meet E1–E3. The definition picks out a class, not a unique decomposition.

Does this version work for you as the basis? Then the next step is to check which pairs the datasets allow.

---

26-11-06, Claude Opus 5.5 — *local copy of the Causal Abstraction paper*; I have read INDEX.md and grepped this LOG for the paper; I have not read the rest of this LOG in this session

C: (transcribed from chat) I need to discuss the "Causal Abstraction" paper. Can you store a local copy of it?

C: (transcribed from chat) I don't care much abouth the pdf. I rather want you to have a referencen so I can ask you about the paper

Stored Geiger et al., *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability* (JMLR 26, 2025; arXiv:2301.04709 v4; the paper cited in the *literature* entry above) as `papers/Causal_Abstraction/ca_paper.md` plus `figures/` (18 PNGs), following the format of the other `papers/` copies.

How it was made: arXiv v4 LaTeX source → pandoc 3.9 (gfm) → a post-processing script (in the session scratchpad, not kept). Citations are resolved to (Author, year) from the `.bbl`, and all 172 references are listed at the end. Definition, theorem, section and figure numbers match the PDF [verified: spot-checked Def 24/25 on p. 13, Figures 1–5 and Eqs (1)–(4) against the PDF text]. The paper's custom macros are expanded to plain LaTeX. The TikZ diagrams could not be compiled (no LaTeX here), so they are crops of the arXiv PDF. Under each inline diagram I added a short text description I wrote from the image, marked as such; the one for path patching is approximate. Algorithm 1 (causal scrubbing) is transcribed as pseudocode. Known gaps: only the four referenced equations carry their numbers; the two HTML tables (the alignment Π in §2.6 and Table 1 of methods) keep pandoc's HTML, with degraded math in the first.

---

26-11-06, Claude Opus 5.5 — *stuck on the basis: narrow the output, the decomposition is the basis, superposition*; I have read INDEX.md, this topic's SUMMARY, this LOG from the Geiger reading on, the paper-draft SUMMARY, and Julian's rework of the draft's `<capture>` section (`reports/26-11-04_tpd_truth_paper_draft.typ`, subsections *Intuition* and *Modeling with Causal Models*, with his `//` comments); in `papers/Causal_Abstraction/ca_paper.md` I re-read §3.1 (polysemantic neurons, modular features) and §3.6.1 (sparse autoencoders)

C: (transcribed from chat) I'm trying to find a principled evaluation criterion for the different tPD outputs. I'm still stuck on the basis. You have writter a proposal in @reports/26-11-04_tpd_truth_paper_draft.typ . The marked part is my rework of that. I think however I'm stuck again in the basics.

Julian's two `//` comments in the rework, which this entry answers:
- after "A decomposition should show where this quantity is computed in the model's weights": "This is a critical point. I don't have a good intuition here right now... Maybe more causal modeling will make it clearer."
- after the alignment paragraph: "This doesn't work as described here. We have to find a right basis to support the abstraticnon. Also how does this alignment thing work with supperposition or isn't that a problem?"

### Diagnosis: two changes in the rework make the problem harder than it needs to be [concluded]

1. **The output became all logits, and H became a full feature model** (features F_1 … F_k, Y the logits, a function of all features). To show that such an H abstracts the network, *every* feature that the logits read needs its own cell of network variables. That is the full superposition problem: a basis for all features at once. The truth question doesn't need it.  
C: you are right that this is not what we need or what our setup is designed for. But I want to start as general as possile and see what how to simplify this to what we're actually doing.
Answer (26-11-06, Claude Opus 5.5): Agreed. The general version is in the answer to your last comment at the bottom (G1–G3); narrowing to the verdict then becomes the last, simplifying step instead of the first.
2. **"Where T is computed in the weights"** mixes two claims, where T is *represented* and which weights *compute* it. They need different tests (below).  
C: this is a conceptual problem for me how to express this.
Answer (26-11-06, Claude Opus 5.5): See G4 at the bottom. In short: "where T is represented" is a claim about activations (a readout of T), "which weights compute T" is a claim about mechanisms, and it can only be stated after splitting a layer's mechanism into parts.

### 1. Keep the big feature model as intuition, test a marginalized one

The feature model is a good *intuition* ("the model extracts features, one of them is T"). The model we *test* should keep only what the truth question needs: X → T → Y with Y = the verdict, or X → Z → T → Y with Z the content bundle (subject, object, recalled fact). Going from the big model to the small one is marginalization, one of the three operations that constructive abstraction is built from (Geiger Prop. 40). It is allowed if the verdict reads the features only through T, and that is the assumption "the verdict measures T", stated as a causal model. Everything else in the network goes into the forgotten cell. Then **only T needs a cell**, and superposition matters only insofar as it affects that one cell (section 3).

Julian's "uses this belief in coherent ways" fits this well: give H *several* outputs that all read the same T, e.g. the verdict prompt, the second prompt ("Is this statement true (1) or false (0)? Answer: ") and the Yes/No prompts of G2. Because attention is causal, the activations at the statement tokens don't depend on the question appended afterwards. So one decomposition and one swap can be tested under every readout. "Coherent use" then means: the same swap moves every readout as H predicts [concluded]. This matters because of B3 (the earlier finding that steering at the statement flips the verdict under both prompts while the model still recalls the fact correctly): the verdict prompt may read a *truth assessment* that is not the model's knowledge. Several readouts don't remove that, but they test whether there is one T rather than one per prompt.

### 2. "We have to find a right basis": the decomposition is the proposed basis [concluded]

C: Also here I want to start with the conecpt. I'm not interested about what tPD does at this moment. I want to describe how we think about the model. So how do we think about it? We have a high-level view and a low-level view. But the low-level cannot be the activations as they don't neatly cluster into the features. I want to formalize the idea that there is a basis that is aligend with the high level view. And still I don't know what to do about superposition, did you answer that?
Answer (26-11-06, Claude Opus 5.5): You're right, I argued from the method backwards. The answer at the bottom (G1–G4) doesn't use tPD. It drops "basis" in favour of something weaker that also works under superposition: per feature a *readout* and an *edit* (G2). Superposition was not answered properly above; G3 does it with your example.

Causal abstraction's answer to the basis problem (Geiger §3.1, §3.6): you may first change variables, then align. Neurons, a rotation (DAS), or SAE features are all allowed. Which change of variables is right is an *empirical hypothesis*, tested by whether the alignment passes interchange interventions. Their words (§3.1): "Whether a particular decomposition of a deep learning system into modular features is useful for mechanistic interpretability should be understood as an empirical hypothesis that can be falsified through experimentation."

tPD's decomposition is such a change of variables, in parameter space rather than activation space: every decomposed output y is split into a C-write c and a Δ-write d, with y = c + d. This is the draft's existing *Step 3: the decomposition as a change of variables*, which is still in the file below the rework. The rework stops right before it. So the plan is: H has T; the network has no variable that is T; the decomposition proposes one (the C-writes); the swaps test the proposal.

Formal status: the split is not a rotation (two summands can't be recovered from their sum). But it is a refinement: the old model is recovered by summing, and every old intervention is still available. Geiger's own treatment of SAEs has the same status: an SAE with zero reconstruction error counts as a bijective translation there (§3.6.1), although with k ≫ n features it is invertible only on the activations the model actually produces. **Δ plays the role of the SAE's error term**: it is what makes the split exact on every input. Keeping the error term as an explicit node is also how SAE circuit work stays faithful [assumed: from memory, Marks et al. 2024 *Sparse Feature Circuits* include SAE error nodes; not re-checked].

### 3. Superposition: what it breaks and what it doesn't [concluded unless tagged]

- *It doesn't break the definition.* A cell can be any set of variables; the map π from cell values to T is induced from inputs (Def. 47). Nothing requires one feature per dimension.
- *It breaks the existence of a clean* fixed *linear cell.* With more features than dimensions, T's direction overlaps with other features. Swapping a fixed subspace then drags those features along, and IIA drops *if the verdict reads them*. That's why DAS-type subspaces are often imperfect.  
C: This might be important for my understanding. Does that mean that the causal model theory as I want to use it is not able to deal with superposition properly: we have two features F1 and F2, and the model can conflate them but usually doesn't as they are mutually exclusive.
Answer (26-11-06, Claude Opus 5.5): Short version: the theory handles it, but honestly. With F1 and F2 superposed and mutually exclusive, H can treat them as two variables only for interventions that keep them exclusive; otherwise they have to be merged into one variable. It does not let you claim they are separately manipulable when the model can't hold both. Worked example in G3 at the bottom.
- *Our cell is not a fixed subspace.* The C-write is the output of the components that are active on this input, so different statements can write T through different directions. That is the parameter-decomposition answer to superposition: features are sparse, so on any given input few are active [assumed: the SPD/APD motivation; whether it holds here is exactly what the evaluation measures].
- *Where superposition still shows up: the bundle.* The cell carries a bundle Z ⊃ T (T plus the content it was computed from). Aligning the cell with Z is fine, because swapping all of Z from x' gives T(x'). It fails when the verdict *also* reads x's content by a route outside the cell (attention to the tokens, Δ-writes, undecomposed layers): then the verdict judges a mix. For routes that bypass the decomposed sites, the partial-decomposition reference (the site swap) absorbs this. For routes through Δ, it is a real failure of the decomposition, as it should be.

### 4. "Where in the weights": three claims of increasing strength

- **R1 represented:** T sits in the outputs of these components (the C-swap moves the verdict; draft step 4 (a), E2). The weights come in because the variable is *defined by* them: these weight pieces are the ones that write T.
- **R2 not supplied elsewhere:** the Δ-writes add nothing of T that the verdict uses (draft step 4 (b), the causal D3).
- **R3 computed:** the components implement the high-level mechanism Z → T: they read content and produce T, rather than copying a T that is already in their input. Testing this needs a cell for Z at the components' input side and a path test (change only what the components read). With seven layers that is messy, because layer-12 components read layer-11 writes.

R1 + R2 is what a swap can establish; it is the current definition. R3 is the stronger claim already listed under the draft's open point *Representation or computation*. Suggested replacement for the sentence Julian marked: "A decomposition should single out a part of the weights whose outputs carry this quantity, such that the rest of the weights does not supply it. Whether that part also computes the quantity from the statement's content is a further question."

### 5. Small points on the rework

- Acyclic models with the graph read off the parent sets: agreed, nothing is lost (our network is acyclic; Geiger allow cycles only for generality).
- "T_1, …, T_2" is presumably T_1, …, T_m. With one statement per input there is one T; several T's would only be needed for inputs with several statements.
- "The low model": the variables are the activations, including every matrix output, as written. The decomposition then adds c and d at the decomposed matrices (section 2).

### Proposed order for the section

Intuition (with the R1/R2 wording) → big feature model as motivation → marginalized H (X → Z → T → readouts) → the network as L, and that no network variable is T → the basis is a hypothesis (Geiger) → the decomposition as the proposed basis (existing step 3) → what capturing T requires and the swaps (existing steps 4–5). On this reading the rework misses no piece; it lacks the narrowing of the output, and the realization that the existing step 3 *is* the basis.

Open for Julian: does this unblock the basis? If yes, I can rewrite the `<capture>` subsections in this order, keeping his text where it fits.

C: as you see, I'm still confused. I want to have a solid beginnig of the derivation of our analysis methods (this might even show that our methods are not right or inadequate, or we don't know how to measure certain properties. Pleaes don't try to shift the basis in favor of the resulting methods)

Answer (26-11-06, Claude Opus 5.5) — *the general picture without tPD: readouts and edits instead of a basis; superposition limits the interventions*; built from the model alone, as asked; the method appears only in G5, to check what it does and doesn't measure. Geiger references re-checked in `papers/Causal_Abstraction/ca_paper.md`: exact transformation must commute "for all I in Domain(ω)" (Def. 25); merges and marginalization (Prop. 40); graded version (Def. 41).

### G1 The two views

- **Low level L: the network.** Variables: the activation vector at each layer and position. Mechanisms: the layers; the weights sit inside the mechanisms. This makes no claim; it is just the network written as a causal model.
- **High level H: the feature view.** Input X; features F_1, …, F_k, each with a mechanism that computes it from X or from other features; output Y, computed from the features. T is one feature. H is the claim about the model.

"L can be understood as H" needs a link between the two. Your comment is right: the link can't be "each feature is a group of activation coordinates", because features don't sit on coordinates.

### G2 The link: a readout and an edit per feature, not a basis

What we need for each feature F_i, at the place where it lives (layer, position):
- a **readout** r_i: activation → value of F_i. It says what F_i is in L;
- an **edit** e_i: change the activation so that F_i takes a chosen value and the other features stay. It gives an intervention on F_i in H its counterpart in L.

**H abstracts L, relative to a set 𝓘 of interventions,** if for every intervention in 𝓘, "edit in L, run, read out" gives what "intervene in H, run" gives. That is Geiger's commuting diagram (Def. 25), written with readouts and edits. The set 𝓘 is part of the claim; Geiger's definition carries it too (Domain(ω)).

**A basis is the special case** where some invertible change of coordinates gives every feature its own coordinates. Then readout = read those coordinates, edit = overwrite them, and edits of different features can't disturb each other, because they touch disjoint coordinates. Geiger's "bijective translation, then alignment" is this case. It allows at most d features at a site (d = the width of the activation vector), all at the same time. Superposition is the hypothesis that there are more features than that. So the question "is there a basis aligned with H?" is generally answered *no*, and the right question is "are there readouts and edits for which the diagram commutes, and for which interventions?" [concluded]

### G3 Superposition, with your example

F1, F2 ≥ 0, mutually exclusive (never both non-zero). The model stores both in *one* dimension: h = F1 − F2. Downstream it reads F1 = ReLU(h), F2 = ReLU(−h), exactly, on every input that occurs.
- *Readouts exist:* r1(h) = ReLU(h), r2(h) = ReLU(−h).
- *Edits exist, but not for every intervention.* Setting F1 := a on an input with F2 = 0 is easy: h := a. Setting F1 := a on an input with F2 = b > 0 is not possible: H says (a, b), but no h encodes both; h = a − b is read as one of them only.

So H with F1, F2 as two independent variables abstracts L **for the interventions that keep them exclusive, and for no others.** The theory offers two honest descriptions, and both state a true fact about the model [concluded]:
- (a) *restrict 𝓘*: exclude interventions that make F1 and F2 co-occur;
- (b) *merge*: one high-level variable F with values "F1 = a", "F2 = b", "neither" (Geiger's value/variable merge, Prop. 40). Then h is a basis for F again.

What it does *not* allow is the claim that F1 and F2 are separately manipulable in the model. That isn't a flaw of the theory: in this model they aren't. With nearly orthogonal directions instead of opposite ones, edits work approximately, with an error that grows with interference from the other active features; that is the graded version (Def. 41).

**Consequence for us** [assumed: an argument, not checked]: T is a function of the content Z. Any intervention that sets T against the content ("content of a false statement, T = true") produces a combination that occurs naturally only where the model errs. If T is stored entangled with content, as superposition makes plausible, such edits may be impossible or approximate, like F1 := a with F2 = b above. So the interventions we need for T sit exactly in the region where superposition bites. That's a real limit, not a technicality.

### G4 Where the weights come in: "T is computed in these weights"

In H, T has a mechanism f_T: Z → T (content → truth value). "Computed in these weights" is a claim about **mechanisms**, not about variables. To state it:
1. T must be readable somewhere (a readout r_T at a site s_out), and its inputs Z at an earlier site s_in (readout r_Z). Then the layers between s_in and s_out implement f_T; that already follows from the abstraction.
2. "*Which weights*" asks for more: which *part* of those layers does it. That needs a split of each of these layers' mechanisms into parts, e.g. W = P + R, whose outputs add up to the layer's output. The split adds new variables to L (the write of P, the write of R): a refinement of L, not a change of basis.
3. The claim "P computes T" then has two halves:
   - **(i)** r_T depends on Z only through the writes of P: edit Z, let only P see the edit, and T moves as f_T says;
   - **(ii)** the writes of R carry no dependence of T on Z that r_T uses.

Without a readout r_T one can use the output as the readout, provided the output reads the features only through T (the narrowing from G1 to "X → Z → T → verdict").

Two points that answer your earlier worry:
- *Representation vs computation* is now two separate questions: "is there a readout and edit for T?" (G2, activations) and "which part of the weights produces what r_T reads?" (G4, mechanisms). The second presupposes the first, or the output as a stand-in.
- *Superposition in activations doesn't block G4.* The writes of P and R may overlap in activation space; the split is made in the weights, not in directions. Any method that says "these weights do it" chooses some split: a circuit (a subset of heads or neurons), or a weight decomposition.

### G5 What this says about our methods (as a check, not a basis)

- **Never done: the representational question (G2) on its own.** We have readouts of T (probes, the mass-mean and logistic-regression truth directions, analyses Dir and K0) and one edit (B3, steering along the in-context truth direction). B3 flipped the verdict, but recall stayed correct, so it's unclear what that edit sets. No test checks a readout/edit pair for T by the commuting diagram.
- **The swap test from step 5 (E2) is a G4 test with the verdict as readout.** It tests a split P/R without locating T in activations first. It's legitimate only under the assumption "the verdict reads the features only through T"; that assumption isn't tested, but it can be made testable with several readouts (verdict prompt, second prompt, Yes/No; see section 1 above).
- **Not measured:** (i) in G4, "P computes T rather than passing it on", which needs a readout of Z at the input side; (ii) the off-distribution problem of G3: swapping T between statements of different truth creates content/T combinations the model rarely sees.

Open for Julian: does G1–G4 work as the solid beginning? If it does, the next step is to decide which of G2 (representation) and G4 (mechanism) the paper's criterion is about. My view: G4, with G2's limits (G3) stated as a caveat.
