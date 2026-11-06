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

---

C: (transcribed from chat) I'm currentlty understanding and rewritiren "how the decompositions are evaluated". I have a question about the high-level model. How does the variable input length fit into the picture here? Or the variable features   dependingon the input?

---

26-11-06, Claude Opus 5.5 — *variable input length and input-dependent features in H and L*; I have read INDEX.md (head), the draft's `<capture>` section as Julian saved it (`reports/26-11-04_tpd_truth_paper_draft.typ`), the tails of this LOG and of `convos/julian/26-11-04_paper_draft_LOG.md`. A grep of `papers/Causal_Abstraction/ca_paper.md` for "variable length", "sequence length", "input length", "padding", "positions" found no treatment of variable-length inputs [verified: grep, 26-11-06; only these terms searched]. Geiger's examples I remember use fixed-length inputs [assumed: from memory].

**The rule that answers both questions:** a causal model has a fixed set of variables, but a variable's *values* can come from any set. So whatever varies with the input has to go into the values, or the input domain has to be restricted, or one model per input shape is needed. [concluded: Geiger's Def. of a causal model fixes the variable set; nothing else follows]

**1. Variable length, in H: no problem.** X is one variable whose values are token strings of any length. Z, T and Y are statement-level variables, not per-position ones (Y = the next-token logits at the answer position). The narrowed H (X → Z → T → Y_1…Y_r) has the same variables for every input. [concluded]

**2. Variable length, in L: here it bites.** The network on n tokens has n positions of activations, so a different set of variables for each n. Two equivalent standard fixes [concluded; equivalence holds because with a causal mask and right-padding, padded positions influence no earlier position]:
- (a) a *family* L_n, one low-level model per length, all with the same weights. "H describes the network" then means: for every n there are τ_n, Ψ_n, ω_n that satisfy the link condition. H is shared.
- (b) one L padded to a maximum length N; padded positions hold a fixed value and influence nothing.

Consequences:
- *Sites must be named independently of the length*: relative to anchors (the statement's last token `.\n`, the answer position, "all statement tokens"), not by absolute index. This is what *locality* has to mean in our setting.
- *Swaps between inputs need a map between positions.* The draft avoids it by requiring pairs of equal length with aligned positions; that is why good pairs are hard to find (open point *Pairs*). An alternative would be to swap only at anchored positions (e.g. only `.\n`), which tests a weaker claim. The layer scan shows the verdict also reads the other statement tokens (removing block 11–17 at `.\n` alone leaves 0.82 of the separation), so swapping at `.\n` alone would miss most of it.
- The tPD split itself does not depend on length: P = Σ_{c∈S} U_c V_c^T acts per position with the same weights at every position.

**3. Input-dependent features: fixed variables, variable values.**
- *A feature absent on an input* takes a value ⊥ ("not present"), and its mechanism outputs ⊥. The sparsity behind superposition is exactly this (inactive = 0 / ⊥).
- *A variable number of features of one kind* (one T per statement with several statements; one entity feature per token) cannot be a fixed list unless it is bounded. Options: bound the count and pad with ⊥, or restrict the input domain. The draft does the second ("With one statement per input there is one T"): the claim holds only on verdict prompts with one statement, and this domain restriction is part of the claim, like the edit set Ψ.
- *Features of different kind across families* (cities: city and country; larger_than: two numbers): Z's value set is a union of these, which H allows. The difficulty moves to the readout τ, which must read Z in every family. We never read Z, so this costs nothing yet; it matters for R3 (does P compute T from Z).
- *Position-indexed features* (a "subject" feature at the subject token): a general H that has these inherits the length problem from point 2, with the same fixes. The narrowed H avoids it, because Z and T are statement-level; where in the network they are read (possibly spread over positions) is τ's business.
- *Input-dependent active components* (the set S changes from statement to statement): this is the refinement's property 3 in the draft ("if the split depends on the input, it is a function of the network's own activations"), so it is allowed; with CI as that function, it is natural for tPD.

**Proposed sentences for the draft** (not inserted; Julian is rewriting the section):
- after the H bullets: "The variables of $cal(H)$ are fixed; what varies between inputs is their values. $X$ takes token strings of any length as values, and a feature that does not apply to an input takes the value 'absent'. The claim is made only for verdict prompts with one statement, so there is one $T$."
- after the L paragraph: "The network on $n$ tokens has $n$ positions of activations, so there is one low-level model $cal(L)_n$ per length, all with the same weights. $cal(H)$ describes the network if it describes every $cal(L)_n$; the sites of readouts and edits are therefore named relative to the statement's end and the answer position, not by absolute position."

Whether H's general version ("feature variables F_1, …, F_k") needs to say more than this is open: with arbitrary text inputs the number of features is unbounded, and the fixed k is a simplification that only the narrowed H gets rid of. [assumed]

---

C: (transcribed from chat) what is Phi that is going to be relevant for our evaluations?

---

26-11-06, Claude Opus 5.5 — *which edit set Ψ our evaluation uses*; the draft has no Φ, so I read the question as the edit set Ψ of the draft's subsection *Linking the views*; context: the draft's `<capture>` section as of today

C: yes I meant Ψ

**Ψ for the weight question (R1/R2)** [concluded; agent proposal]: the swaps of the draft's *Measuring with the verdict*, on the refined network (down_proj 11–17 split into P = chosen components, R = the rest plus Δ), at all seven layers and all statement tokens at once, for pairs (x, x') of equal length with aligned positions and different verdicts:

| edit e (base x, source x') | ω(e) in H = X → Z → T → (Y_1…Y_r) | prediction if P captures T |
|---|---|---|
| P-swap | set T := T(x') (strictly: Z := Z(x'), which gives T(x')) | verdicts as for x' |
| R-swap | no intervention (identity) | verdicts as for x |
| site swap (P and R) | set T := T(x') | verdicts as for x' (reference, because routes outside layers 11–17 exist) |

C: what does the first row mean with ω(e)?

Answer (26-11-06, Claude Opus 5.5): ω is the second half of the link in the draft's *Linking the views*. For each edit e on the network, ω(e) names the intervention in H that e is claimed to carry out. The link condition then says: "edit the network with e, then read out" gives the same as "intervene in H with ω(e), then run H".

The first row in words: the P-swap (run x, replace P's writes by their values on x') is claimed to carry out "set T to its value on x'". In H the outputs Y_j depend only on T, so H predicts that every verdict comes out as on x'. That prediction is the third column.
- *ω is not measured; choosing it is the claim.* ω(P-swap) = "set T := T(x')" is the claim "P's writes carry T". ω(R-swap) = identity is the claim "R's writes carry nothing of T that the outputs use". The test checks whether the network agrees with these claims.
- *"Strictly Z := Z(x')".* P's writes carry the content as well as T (the draft's *Two readings that are too strong*). So the swap sets the whole bundle Z to its value on x', and H's mechanism Z → T then gives T(x'). Because the outputs read only T, both versions predict the same verdicts.

- τ needs no readout of T: the network's outputs are H's outputs Y_j (the verdict under each truth question), and τ reads them directly. This is where the assumption "the verdict measures T" enters.
- Intervention-algebra check comes free: P-swap ∘ R-swap from the same x' = site swap, and ω(P) ∘ ω(R) = "set T(x')" ∘ identity = ω(site). Consistent [concluded].
- Not in Ψ: the removals we actually ran (R, T0, T1, T2, M-core: components set to 0, −1 or 2). They produce states no statement produces, and H has no intervention they implement. They could enter only if T gets an extra value ("no truth information") with a stated effect on the verdict, which H does not specify [assumed]; mean ablation would be the closest to such an edit.  
C: I don't understand what you mean by this comment

Answer (26-11-06, Claude Opus 5.5): Restated. Ψ may only contain edits for which ω can say which intervention in H they carry out. Without that, H makes no prediction for the edit, and the edit tests nothing. Our removal analyses have no such ω:
- The analyses meant: T0 (all components switched off, Δ alone), T1/T2 (single components or the top k switched off), M-core (components reversed with mask −1 or amplified with mask 2), R (recovery: the components alone).
- A swap puts in writes that a real statement x' produces. Those writes mean what they mean on x' (Z(x'), hence T(x')), so ω follows from the inputs (Geiger Def. 47: the readout is defined on the network's own states). A removal puts in writes (zero, negated, doubled) that no statement produces. T in H has only the values true and false, and neither is what "P's writes switched off" means. So there is no ω(removal), and H predicts nothing about the verdict after a removal.
- To bring removals in, H would need an extra value of T, say "unknown", plus a rule for what the readouts do when T is unknown (verdict 50/50? the model's default "true"?). H says neither. Picking one would be an extra assumption.

In short: removals answer "is P used for the verdict?", not "does P carry T the way H says?". They remain useful evidence, just not tests of the link condition. The draft says the same in *What this says about our analyses*, second bullet.
- Open choices that change Ψ: S = all components or the input-dependent active set; whether to also allow swaps restricted to some layers or positions (finer Ψ = stronger claim).  
C: These are future questions for experiments but not important right now in the conceptual phase

Answer (26-11-06, Claude Opus 5.5): Agreed. They stay in the draft's *Open points* (the set S) and are left out of the conceptual discussion here.

**Ψ for the representation question** (not the paper's main criterion, per my earlier view): edits on the unsplit network that set T, e.g. B3's steering along a truth direction with ω = "set T := true/false". The superposition caveat applies to exactly these.

C: What is the representation question? Also I think edits on the network altering T should be one criterion even if we don't meet it (is it the first row in the table). It would be strange to say, we have the mechanism for T but we can't do anything with it.

Answer (26-11-06, Claude Opus 5.5):

**The representation question** is the first of the two questions in the draft's *Intuition*: "Where is T represented?" Is there a site in the *unsplit* network with a readout of T and an edit that sets T to a chosen value (`@ca-link`)? Example edit: "add v at layer 14 of the last statement token", with ω = "set T := true". The other question, "Which weights compute T?" (`@ca-weights`), needs the split W = P + R and edits on P's writes. In the entry above I treated only the second as our criterion.

**Edits that alter T as a criterion: agreed. The first row is one, but a weak one.** The P-swap is an edit on P's writes that changes T, and R1 says the verdicts follow. So "we can do something with it" is already in the definition: if P carries T in the R1 sense, swapping P's writes moves T. What the P-swap cannot do is set T to a value *of our choosing while keeping the content*. It carries over x''s whole bundle Z(x') and only ever produces T(x') = f(Z(x')), a combination that occurs on a real statement. The strong sense of "we can do something with the mechanism" would be:

- **R4 control (proposal):** there is an edit of P's writes, from a stated small class (e.g. a fixed vector per component, or a rescaling of components), that carries out "set T := t, Z unchanged": every readout Y_j follows t.

Why R4 is a separate criterion and not implied by R1/R2 [concluded]: R1/R2 only use edits that lead to states the network reaches on some statement. R4 asks for states where T contradicts Z, which the network reaches only where it errs. That is where superposition bites (draft, *Consequence for T*): if T is stored entangled with Z, no such edit may exist even when R1/R2 hold perfectly. So a decomposition can carry T in the R1/R2 sense and fail R4. I agree it belongs on the list. My suggestion: state it together with this reason, so that failing it is not read as the method's fault alone.

R4 needs two things R1/R2 did not:
- *Its ω cannot come from the inputs*, since no statement has "content of x, T flipped". H has to be trusted off the network's natural states. That is an extra assumption and should be stated.
- *A control.* Some edit always flips the verdict, e.g. a large enough push towards "0". R4 says something about P only if the edit class is restricted (the simplicity and specificity constraints of the draft's *What keeps the general link from being empty*) and the same edit class is tried on a random or untrained split. Relevant precedent: in Exp 4e an untrained data-aligned basis swapped verdicts about as well as the trained components (`convos/julian/26-10-16_faithful_targeting_SUMMARY.md`).

**Side point on B3, possibly relevant to the draft's *Narrowing* paragraph** [concluded, from H's graph]. In H = X → Z → T → Y, recalling the fact ("Paris is in …") reads Z, not T. An edit that sets T against Z should therefore move the verdicts and *leave recall unchanged*. Two earlier results show exactly this: B3 (steering along the truth direction flipped the verdict, the model still recalled the fact) and the gradient-mask search (`convos/julian/26-10-16_component_steering_SUMMARY.md`: verdicts inverted within some families, direct-question recall unchanged). Under H, that is what an R4 edit should do, not evidence against one. The draft reads B3 as "the verdict prompt reads an assessment, not the model's knowledge", and the steering topic concluded "no belief lever". Both readings are right if T is meant as the stored fact (part of Z). They are not objections if T is the judgement "the statement agrees with what I know", which is how the draft's *Intuition* currently words it ("whether it takes the statement to be true"). Worth pinning down which of the two T is.
C: also how did you come up with these swaps, why do we choose these as the criteria?

Answer (26-11-06, Claude Opus 5.5): Where they came from: step 5 (entry *perfect decomposition in causal-abstraction language*, E2). There I took Geiger's standard test, the interchange intervention, and applied it to the split. I did not derive it there; here is the derivation:

1. *The edits must act on P and R separately.* The claim under test is about the split (R1: P's writes carry T; R2: R's writes don't supply it). Edits on the unsplit network (steering, the site swap) predict the same outcome whatever the split is, so they cannot test it.
2. *Each edit needs an ω* (answer to the removal comment above). The only edits whose ω follows from the inputs alone put in values that another input x' produces. Those values mean what they mean on x', so ω = "set Z, and with it T, to its value on x'". Those are interchange interventions (Geiger Def. 44).
3. *One edit per claim.* Applied to the two parts of the split, this gives two edits: the P-swap tests R1 (prediction: verdicts of x'), the R-swap tests R2 (prediction: verdicts of x). The site swap does not test the split. It is the reference for the partial case: how far swapping these layers can move the verdict at all, i.e. the most any split of them can reach.
4. *Pairs with different verdicts.* For a pair with equal verdicts, H predicts the same output with and without the swap, so the test cannot fail.

What the derivation does not show [concluded]:
- That interchange interventions are the only reasonable edits. They are the weakest edits that can test the split: they stay on the network's natural states and need no extra assumption for ω, at the price of saying nothing about states off them (R4 above).  
OUTDATED (26-11-06): "stay on the network's natural states" holds for each refined variable (P's write, R's write) separately, not for the run as the network sees it. With cancelling writes, c(x') + r(x) can be far from anything the network produces. See the answer to the comment on re-implementations at the bottom (*re-implementation under the swaps*).
- That passing R1/R2 means P *computes* T rather than passing it on (R3, untested).

The criteria list then reads: R1 and R2 (swaps; ω from the inputs), R3 (computation; path test), R4 (control; ω assumed, needs a control split), plus the size of P. Ψ for the weight question = the P- and R-swaps, the site swap as reference, and R4's edit class if R4 is adopted. Open for Julian: adopt R4 as a criterion in this form, and which T (the judgement or the stored fact) the draft means.
C: we might adopt R4 as criterion, but let's keep talking
C: T could be both I suppose. The intersting thing is too make predictions about decompositions of either of them.

Answer (26-11-06, Claude Opus 5.5): Agreed. Then they are two high-level models that predict different outcomes for the same edit:
- **H_J (judgement):** X → Z → T → verdicts, with recall reading Z. An edit of T moves the verdicts and leaves recall.
- **H_F (stored fact):** the fact B is part of Z, recall reads B, and T = f(B, the statement's claim). An edit of B moves recall *and* the verdicts together; an edit of T alone exists only in H_J's sense.

A decomposition can be tested against both: its swaps or R4 edits either move recall along with the verdicts or they don't. The tiu twins ("Paris is in France / Spain") keep B fixed and change the claim, so swaps between them test H_J. Testing H_F needs pairs that differ in the fact, not the claim, which is hard with one model's fixed knowledge [assumed: not thought through].

C: So we haven't really though about which edits should we test? One question is what do we expect what the current tests predict on a reimplementation (what we most likely see in plain tPD). In my understanding I think plain tPD will ace these tests.

Answer (26-11-06, Claude Opus 5.5) — *re-implementation under the swaps*; context: this LOG, the cancellation results in `convos/julian/26-10-10_bolder_experiments_SUMMARY.md` (Exp 4a) and `convos/julian/26-10-29_paper_writing_SUMMARY.md` (*Cancellation in plain tPD*). Nothing run.

**I think you're right: a re-implementation passes R1 and R2, possibly better than a faithful decomposition** [concluded under a linear-readout simplification; not measured].

Recall what re-implementation means (from those topics): the active components write a vector c that is not a part of the real output y (Exp 4a: 12.9 × |y| at cosine 0.03 with y), and the inactive components plus Δ write r ≈ y − c, cancelling it. Plain tPD in the paper does this moderately (C and Δ each 1.2–2.3 |y|, writing against each other).

Simplify the readout: the verdict depends on one linear function v of the site's output (its truth signal), with sign v = the verdict. tPD's loss makes c alone give the right verdict, so sign v(c(x)) = τ(x), and |v(c)| is free; with large writes it can exceed |v(y)|. For a pair with τ(x) ≠ τ(x'):
- P-swap: the site writes y(x) + [c(x') − c(x)], so v = v(y(x)) + v(c(x')) − v(c(x)). The last two terms both have the sign of τ(x'). If |v(c)| ≥ |v(y)| the verdict follows x': **R1 passes.**
- R-swap: the site writes y(x) + [r(x') − r(x)] = y(x') − [c(x') − c(x)], so v = v(y(x')) − v(c(x')) + v(c(x)). With |v(c)| ≥ |v(y)| the verdict stays at x: **R2 passes.**

A faithful split (c a real part of y, |v(c)| ≤ |v(y)|) passes only if c carries nearly all of v(y); a re-implementation passes by overshooting. The hard-verdict IIA can't see the overshoot. With the logit difference s instead, the overshoot would show as a P-swap that moves s *beyond* x''s value, which is a symptom but not a criterion.

**Why the test is blind, and which assumption of mine it breaks.** The two decompositions differ in the part of c(x') − c(x) that v doesn't read. In a re-implementation that part can be large, and in the full model it is cancelled by r; in the swap it is not cancelled, because c comes from x' and r from x. H can't see it, because we marginalized H down to T and the verdicts: nothing in H reads the other directions. So my earlier claim that swaps "stay on the network's natural states" was wrong at the level that matters (marked OUTDATED above). Each refined variable takes a value some input produces, but the network only sees their sum, and the sum can be foreign. The refined variables are our hypothesis, not the network's; naturalness has to be judged in L, the real network.

Whether the swap state actually is foreign depends on how much c *varies* between twins. If the cancelling part is mostly constant across statements (e.g. always-active components), c(x') − c(x) is small and close to y(x') − y(x), and then the swap is natural. That case is arguably fine: for swaps only the variation matters, and P does carry the variation. Re-implementation hurts swaps only if the cancellation sits in the *differences*: c(x') − c(x) and r(x') − r(x) large and opposite. Not measured for any run [assumed].

**What follows for which edits to test.**
1. *Design principle:* a test belongs in the evaluation only if we know what a re-implementation predicts for it and the prediction differs from a faithful split. R1/R2 with the hard verdict fail this as stated. I hadn't checked that.
2. *Anchor the swap in the real network* (a specificity condition, the draft's constraint 2): written as an edit of the real activations, the P-swap is y ↦ y + [c(x') − c(x)]. Require it to be a part of the site swap y ↦ y + [y(x') − y(x)]:
   - in the writes: |c(x') − c(x)| not larger than |y(x') − y(x)|, and no cancellation between c(x') − c(x) and r(x') − r(x) (the cancellation test, applied to differences);
   - in behaviour: readouts other than the verdict (next-token predictions at the statement tokens, the answer format) stay within what the x and x' runs show. This puts back some of what the marginalization removed.
3. *R4 does not help here*: a re-implementation has a strong verdict handle and likely passes R4 too [assumed]. R4 stays a criterion about controllability, not a re-implementation test.
4. *Seeded defect for the evaluation:* Exp 4a is a known strong re-implementation (checkpoint kept on WandB per `convos/julian/26-10-03_wandb_storage_SUMMARY.md` [assumed: not re-checked today]). Any proposed test can be run on 4a first. If 4a passes, the test can't distinguish re-implementation from decomposition. Recon (Exp 4b-style, components a part of y) is the opposite reference.

So far the edits were derived from H alone. This shows they also have to be derived from the failure modes we want to catch, re-implementation first among them. Open: whether to list the failure modes (re-implementation; Δ carrying T; P = W; passing on instead of computing) and derive one test per failure mode, with the anchoring condition in point 2 as the candidate for re-implementation.

C: What kind of test could expose a reimplemenatition. And how does such a test fall in the causal model framework?

Answer (26-11-06, Claude Opus 5.5) — *exposing re-implementation: P-swaps that leave the network's states*; context: this LOG, the draft's `<capture>` section and its ortho method description (`reports/26-11-04_tpd_truth_paper_draft.typ`), Geiger Defs. 28 and 46 in `papers/Causal_Abstraction/ca_paper.md`. Nothing run.

**Short answer.** H can't see a re-implementation. In the framework it is a failure of the *refinement* (the split y = p + r), not of the link between H and the network. It shows up as P-swaps that put the network in a state it never reaches. There the readout τ is outside its domain, so a "pass" there means nothing. The verdict hides this, because read off the network's output it is defined on every state. The test that exposes it asks whether each P-swap is, at every split site, a genuine interchange intervention of the *network*, not only of the refined model. That has an exact geometric form: the P-difference and the R-difference between the twins are orthogonal.

### 1. Where the hole is [concluded]

- The draft's τ is partial: "defined only on states the network actually reaches, or close to them" (*Linking the views*). The link condition can only be checked on edits that keep the run in τ's domain.
- In *Measuring with the verdict*, τ is the verdicts, read from the outputs. That readout is defined on any state, so it silently extends τ to states the network never reaches. That is the loophole a re-implementation uses.
- Why ordinary interchange interventions don't have this problem: in Geiger's setting the edit acts on the network's own variables, possibly after a bijective translation (Def. 28; a rotation for distributed interchange interventions, Def. 46). Each coordinate of the swapped state then takes its value from x or from x'. The joint state is new, but every coordinate is natural. As an edit of the site's output, a distributed swap is y(x) ↦ y(x) + Π[y(x') − y(x)], with Π an orthogonal projection. It can never move the state further than the site swap, nor in a direction the site swap doesn't go.
- The split is not a bijective translation (draft, *Which weights compute T*: "p and r cannot be recovered from y"). Written as an edit of the network, the P-swap is y(x) ↦ y(x) + [p(x') − p(x)], and nothing forces p(x') − p(x) to be a part of y(x') − y(x). The draft names the hole itself: "What it buys is new edits: 'replace only the P-part of this layer's output' has no counterpart in L". An edit without a counterpart in L gets its ω only from the refined model, i.e. from our own hypothesis. That is the reason removals were excluded from Ψ (answer under the removal comment above), and it applies to cancelling P-swaps as well.

### 2. The test: is each P-swap an interchange intervention of the network?

Per split site and position, for a pair (x, x'):
- Δ_site = y(x') − y(x), the site swap's edit;
- Δ_P = p(x') − p(x) and Δ_R = r(x') − r(x), so Δ_P + Δ_R = Δ_site.

**Fact** [concluded: two lines of linear algebra]: Δ_P is the orthogonal projection of Δ_site onto some direction (equivalently: in some orthonormal basis the P-swap state takes, coordinate by coordinate, the value of x or of x') **if and only if ⟨Δ_P, Δ_R⟩ = 0.** Proof: if ⟨Δ_P, Δ_R⟩ = 0, take u = Δ_P/|Δ_P|; then ⟨Δ_site, u⟩ = |Δ_P|, so Δ_P = ⟨Δ_site, u⟩ u. Conversely, a projection Π gives ⟨ΠΔ, (1 − Π)Δ⟩ = 0.

As a graded number, summed over split layers and positions:

κ = (|Δ_P|² + |Δ_R|²) / |Δ_site|² = 1 − 2⟨Δ_P, Δ_R⟩ / |Δ_site|².

- κ = 1: the P-swap is a part of the site swap, as for a rotation.
- κ ≫ 1: the P- and R-differences cancel. The P-swap pushes the network where the site swap doesn't go: re-implementation in the differences.
- κ < 1: P and R carry overlapping parts of the variation (both move the same way). Overlap, not cancellation.

Predictions: a re-implementation whose cancellation varies between twins gives κ ≫ 1; a faithful split gives κ near 1 (or below with overlap). A re-implementation with a *constant* cancelling part (the harmless case in the previous answer) also gives κ ≈ 1, which is right: its swaps are natural.

**Where it sits in the framework.** κ does not test H against the network; it decides which edits are admissible in Ψ. A P-swap (and likewise an R-swap, since Δ_R = Δ_site − Δ_P) tests R1/R2 only where κ is near 1. Elsewhere its ω has no basis in the network. Equivalently, κ ≈ 1 is a condition on the refinement that Geiger's bijective translations meet automatically. It could join the draft's list of properties of the split, but it is a property of the split *together with the edits*, not of the split alone. Proposal: report κ next to IIA(P) and IIA(R), and count a high IIA(P) as evidence only where κ is near 1. Filtering pairs by κ is the other option, but it would hide failures.

Remarks:
- Positions before the first token where the twins differ have identical activations (causal attention), so all three differences are 0 there and drop out.
- κ comes from the clean runs on x and x'; no swap run is needed. In the actual swap run, later layers see changed inputs, so per layer the clean-run κ is an approximation.
- The direction u differs from pair to pair, while Def. 46 uses one rotation for all inputs. A fixed subspace would bring back the superposition problem (G3). The per-pair version asks only for no cancellation: weaker, on purpose [concluded].
- κ ≈ 1 shows absence of cancellation, not faithfulness. Conversely, the model itself may contain sub-mechanisms that write against each other, and a faithful split that separates them fails κ. Telling "the decomposition invented the cancellation" from "the model has it" needs a reference, e.g. κ for splits of the layer along its own hidden neurons [assumed: not thought through].

### 3. Two weaker tests

- **Other outputs (specificity, the draft's constraint 2).** Under the P-swap, next-token predictions at the statement tokens after the differing token should lie between the base run and the site-swap run, e.g. KL(base, P-swap) ≤ KL(base, site). This brings back outputs the marginalization removed. H says nothing about them; the condition comes from the same requirement as κ (the edit stays a part of the site swap), checked by behaviour instead of geometry. It catches the uncancelled residue only if something downstream reads it; κ catches it regardless.
- **Overshoot of the score s (graded IIA, Def. 41).** With the logit difference s and the site swap as reference, a P-swap that moves s further than the site swap does is suspicious. Cheap, but only a symptom. A faithful split shows it when R's difference pushes s the other way, and a re-implementation whose verdict signal happens to match |v(y)| doesn't show it.

### 4. Calibration: seeded re-implementations with a dial

Every test should first be shown to fail on a known re-implementation. Besides Exp 4a (previous answer), there is a construction that works on any decomposition and has a dial k: scale the components and let Δ absorb the difference,

W_c ↦ (1 + k) W_c, Δ ↦ Δ − k Σ_c W_c.

W is unchanged. On every input p ↦ (1 + k) p and r ↦ r − k p, so Δ_P ↦ (1 + k) Δ_P and Δ_R ↦ Δ_R − k Δ_P. Starting from κ = 1, κ becomes 1 + 2k(1 + k) |Δ_P|² / |Δ_site|² [concluded: algebra]. Meanwhile the hard-verdict R1/R2 get *easier* to pass as k grows (overshoot, previous answer). So the dial shows directly that the hard-verdict swaps can't detect it and κ can. Honest CI is kept approximately: the inactive components' writes scale too, which matters only if they weren't negligible. Recon is the opposite reference (components ≈ the whole output, Δ_R ≈ 0, so κ ≈ 1 trivially).

### 5. Side observation on ortho (a consequence, not an argument for the method)

κ = 1 has the same geometry as ortho's loss term (C on the sphere with diameter y), but for the *differences* between twins and for the active set rather than all components at full strength. Ortho's term does not imply it: ⟨C(x), D(x)⟩ = 0 and ⟨C(x'), D(x')⟩ = 0 leave the cross terms free. They bound |Δ_P| only by |y(x)| + |y(x')|, which can be far above |Δ_site| for twins. So κ would be a held-out finding for ortho too, not something it was trained to pass [concluded].

### 6. Back to the failure-mode list

With this, the failure modes from the previous answer each have a test: re-implementation → κ (calibrated with the dial and Exp 4a); Δ carrying T → R-swap; P = W → the size of P; passing on instead of computing → R3, still without a practical test. Open for Julian: adopt κ as the admissibility condition for the swaps in Ψ.

C: kappa is some internal measurement. I would have liked to have a real behavioural difference between a a real decomposition that captures T and a re-implementation of T. Somehow a reimplementation should hurt some other parts of the computation in some way.
You are on a quite abstract level, which is great and can provide great benefits. But somehow theory and expectation are quite far away. I'd expected something like: if we have a good decomposition we can e.g. (assuming certain conditions on how the model works internally)
- change the verdict, while keeping verdicts of other qualities unchanged
- make the model oblivious to the truth of a statement
Let's say that I'm aiming for these kind of things, at least as the final best tests for a decomposition. So there are some basic sanity checks but then some tests for which my decompositions fail.
It feels a bit like that from the theory that you have extracted we are at the lower end of the spectrum. What I have in mind is on the higher end of the spectrum and we're missing something in the middle.
Also it feels to me that we don't have (or I'm overlooking something) that brings the high level H down to the decomposition level. We talked about the basis and stuff, but I don't see this anywhere. But the technical reality of the model has to enter the description somewhere. What I mean maybe best understood in an example, let's say we have a high level model H that does not capture what the net actually does but still predicts everything perfectly. We might even have readuouts and edits that work great for the bits we are interested, but all of this is not grounded on he computation itself but on a different mechanism that produces the same outputs.
My comment is a bit rambly. The meta problem is that:
- I don't fully understand all the theory you are providing and I asked you to setup
- I don't know if there's still a gap or a theory mismatch with what we have and what we actually need to informatively talk about tPD
- If the theory leads to a usefull class of evaluations for the tPD decompositions
- I don't know if all this theory actually brings any benefit (like showing us blind spots in our evaluation setup. Have a suit of evaluations that within the theory guarantees that a decomposition that aces them all, can be called a decomposition of the/a truth mechanism)

Answer (26-11-06, Claude Opus 5.5) — *the missing middle: a ladder of alternatives, and where a re-implementation can show at all*; context: this whole LOG re-read; `convos/julian/26-10-29_paper_writing_SUMMARY.md` (sections *Where plain tPD fails*, *Why Δ alone isn't worse than all-off*, the side-effect test G0–G2); `convos/julian/26-10-16_component_steering_SUMMARY.md` (mask search, recall test); the T0 table in `scratch/26-10-29_paper_writing/tables/tables.md` §2a. Nothing run.

Your comment is fair. Short answers to your four meta questions first, then the substance.

**Short answers**

1. *Is there a gap between the theory and what we need?* Yes, and I made it. In the entry *stuck on the basis* (section 1) I narrowed H to "statement → T → verdict" and dropped every other output. You wanted to start general, and you were right. A narrowed H can only predict the verdict, so every test derived from it is a verdict test, and a re-implementation that gets the verdict right passes all of them. Your high-end tests ("flip the verdict, other qualities unchanged"; "make the model oblivious to truth") need H to have outputs besides the verdict. The theory has room for them; my version of it didn't.
2. *Where does H get grounded in the model?* Only through the edits it has to predict. In causal abstraction, "H describes the network" means: for every edit in a chosen set Ψ, editing the network and running it gives what the matching intervention in H gives. Your example (an H that predicts everything, but via a different mechanism) can exist only relative to a Ψ: two mechanisms that agree on every edit in Ψ can't be told apart at that level, and they come apart only on an edit outside Ψ. The re-implementation is exactly your example, for Ψ = {P-swap, R-swap} with the verdict as the only readout. The model's technical reality enters in three places: which weights (the split P/R), which edits are applied to the real network, and which outputs are read. If those three are thin, H floats.
3. *Does it lead to useful evaluations?* Only when used as a recipe: each test is an edit, H's prediction, and what each named alternative predicts. A test is worth running only if some alternative predicts something different. On its own the theory generates nothing; the list of alternatives does the work.
4. *Can it give a guarantee?* Only a conditional one: "passes every test in the suite ⇒ H describes the network for every edit in the suite". No framework gives "this is *the* truth mechanism" (non-identifiability, Méloux et al., *literature* entry: several decompositions can pass the same tests). Realistic goal: a suite that rules out a named list of alternatives, plus a statement of what it doesn't rule out.  
C: I disagree. The goal might not be tenable for me now, or maybe I have a wrong understanding but you are right. But I sill don't see why we can't have the following methodogical setup:
   (1) assumption about how the model works
   (2) theory about decomposition
   (3) theory of uncertainty from measurments
   (4) a theory of a suit of tests that show that something captuers what we want under the assumptions (1) & (2) to some degree of certainty under the assupmiton (3)
However, I don't need to follow this ideal any longer as I don't feel any closer to it.

Answer (26-11-06, Claude Opus 5.5): I think your setup holds, and my short answer 4 was too pessimistic. It mixed two things. No framework gives "this is the truth mechanism" *unconditionally*, and that stays true. But your (4) is conditional by construction ("under the assumptions (1) & (2)", "under (3)"), and a guarantee of that shape is possible [concluded]. How the pieces map:
- (1) = the high-level model H with its readouts: X → Z → T; truth readouts read only T; other-quality readouts read only Z; one T for all truth readouts. The guarantee is conditional on this.
- (2) = the split W = P + R (with CI) as a refinement of the network, and "P captures T" defined as: H abstracts the refined network with T aligned to P's writes, for a stated set of interventions, readouts and inputs.
- (3) = per-test statistics: each test is an accuracy (graded IIA) on a sample from a stated distribution D, with an interval. We already do this (bootstrap intervals).
- (4) = the ladder, derived from (1) and (2): answer under your comment on the proposal below (*ladder derived from the definition*).

The resulting statement: "if (1) holds, then with confidence 1 − α the split meets each clause of 'captures T' with accuracy ≥ q on D". It keeps two limits, which I'd state rather than count as failure: it says nothing outside D (your forall point under (a)/(b)), and other splits may pass as well (non-identifiability), so what it certifies is *a* truth mechanism. One thing it needs that we don't have: tests on the unsplit network that check (1) itself, so that a failure can be attributed either to the assumption or to the decomposition (the A-checks, below).

What the theory has bought so far: removals answer "is P used", not "does P carry T" (so T0/T1/T2 can't be the main criterion); swaps need pairs with different verdicts; control edits (R4) live where superposition bites, so failing them isn't automatically the method's fault; in H recall reads content, not T, so "verdict flipped, recall intact" is what an edit of T should do. What it did *not* buy: it didn't find the re-implementation blind spot (your intuition did), it only explained it. Beyond "edit, prediction, alternative", the formal machinery adds little for us.
OUTDATED (26-11-06): "it didn't find the re-implementation blind spot" and "the formal machinery adds little" are wrong: rung 3 of the ladder is the earlier P5 and rung 4b is close to the informational D3, both dropped earlier in this LOG; the definition produces the ladder. See *ladder derived from the definition*, under Julian's comment on the proposal below.

### 1. Where can a re-implementation show at all? [concluded, partly from existing data]

A fact that constrains everything: in the unedited model a re-implementation can't hurt anything, because W = P + R exactly, so the model is unchanged. Any behavioural difference must come from an edit that separates P from R. "Hurts other parts of the computation" therefore means *collateral effects of edits on P*.

What plain tPD's re-implementation looks like in the verdict channel. The T0 table (components removed, Δ only; separation share 1 = unchanged true-false gap of the verdict score, 0 = gap gone; test, training prompt): plain s1 0.052, plain s2 0.037, recon s1/s2 0.167/0.164, ortho s1/s2/s3 0.193/0.217/0.238 [verified: `tables.md` §2a, re-read today]. Plain's Δ carries *less* verdict separation than the other methods' Δ. The paper-writing topic also found that plain's write vectors align with the truth direction only at the random level. Reading [concluded, under the linear-readout simplification of the previous answers]: in the direction the verdict reads, plain's components carry the truth signal as a faithful split would, and the cancelling writes sit in directions the verdict doesn't read. So the overshoot scenario of the previous answers (|v(c)| > |v(y)|) is probably not what plain does, and plain would likely pass R1/R2 *legitimately* in the verdict channel.

That leaves two places where plain's re-implementation can show in behaviour. They are two different failure modes, which I had lumped together:
- **(a) Junk writes.** Large cancelled writes in directions the verdict ignores. Visible only if some *other* output reads those directions, and only under an edit that breaks the cancellation. → side effects on other qualities.
- **(b) A different route.** P's verdict signal is computed from different input features than the model's own (e.g. a cue that correlates with truth on tiu). On the training distribution P's write and the model's agree in the verdict direction, so nothing shows. They come apart where the cue and the model's route disagree. → transfer failures: other truth prompts, OOD forms, statements the model gets wrong.  
C: I like these. I like them conceptually and I also like them because they are tricky. They are something like a forall statement. So these might result in a test that is not practical or does not give us any conclusion, wich is ok.

Answer (26-11-06, Claude Opus 5.5): Agreed. Both are statements about all inputs, so finite tests can refute them but not prove them. Under (3), a pass reads "no failure at a rate above q on D, with confidence 1 − α", and its value depends on D. So D should be where a failure is most likely. (b) lives where a shortcut cue and the model's own route disagree: OOD forms, other truth prompts, statements the model gets wrong. (a) lives under edits that break the cancellation, read on outputs nothing was trained on. An active search for failures (like the gradient mask search) is a stronger D than a random sample, at the price of a guarantee that is harder to state [assumed].

If neither shows, plain's extra writes are invisible to the model's behaviour, and only internal measures (κ, write norms) see them. Whether that is a failure then depends on what the decomposition is for: as an account of the model's mechanism, yes; as a behavioural handle, no. I'd demote κ accordingly: a diagnostic that explains a behavioural failure, not a criterion (your objection holds).

### 2. The ladder, with your high-end tests at the top

Each rung is defined by the alternative it rules out. H gets back the outputs it lost: statement X → content Z → T, with *truth readouts* (verdict prompt, second prompt, Yes/No) that read T, and *other-quality readouts* (recall "Paris is in …", the humor/emotion/surprise/word-count digit questions, next-token predictions within the statement) that read Z, not T.

| rung | claim | alternative ruled out | edit and H's prediction | existing data |
|---|---|---|---|---|
| 0 sanity | it is a decomposition | broken CI; P = W | inactive off: no change on Pile; components alone give the verdict; size of P | done (recovery, Pile KL) |
| 1 used | P matters for the verdict | P irrelevant | remove P: the verdict loses truth | done (T0/T1/T2). Doesn't separate methods: Δ alone ≈ all-off for all of them (first-order argument in the paper-writing topic) |
| 2 carries T | P's writes carry T, R's don't supply it | Δ holds a second route to T | P-swap: truth readouts follow x'; R-swap: they stay at x | not run; plain probably passes (section 1) |
| 3 carries only T *(the missing middle)* | the P-swap changes nothing that H doesn't predict | junk writes, failure (a) | same swaps; other-quality readouts that the twins share stay as in the x run. Reference: the site swap (all of layers 11–17's output from x'), a state the network actually reaches | removal version done (G2, Δ alone against all-off): plain above all-off at recall / object + end in both seeds (2.16/1.55 and 1.34/1.94 × all-off), but so is ortho s3 (1.87/1.37), which has no cancellation: inconclusive. Swap version not run |
| 4a control: flip | an edit of P flips T and nothing else | P is only a verdict lever (bias, sentence form) | a small fixed edit of P's writes: every truth readout flips, other-quality readouts unchanged; the same edit class on a random or untrained split as control | partly: mask search on 4c (an earlier decomposition, one seed): inversion in some families (cities, number comparisons, negated families), collapse in others, recall intact |
| 4b control: oblivious | an edit of P removes T and nothing else | Δ holds a second route; P is only a lever | replace P's write by its mean over a twin-balanced set: truth readouts independent of truth; other-quality readouts unchanged | verdict half ≈ T0 (zero instead of mean ablation): plain is the closest to oblivious. Other-quality half: G2 above. Mean version not run |
| 5 coherent use | the same weights carry T for every reader and every input | P is a verdict-prompt mechanism; failure (b) | rungs 2–4 under the second prompt, Yes/No, OOD forms | partly, and every method fails: components-only recovery under Yes/No falls from 0.89–0.97 (1/0 prompt) to 0.24–0.85 (G2) |
| 6 computes | P computes T from Z rather than passing it on | P copies T from upstream | path test (R3) | not run, no practical design yet |

Notes on the table:
- *Your high-end tests are rungs 4a and 4b, and they are well defined in H.* For removals I objected earlier that H doesn't say what the verdict does once T is gone (50/50? the default "true"?). Your "oblivious" sidesteps that: the prediction is not a value but an *independence*. Across a twin-balanced set, the truth readouts no longer depend on truth, and the other-quality readouts don't move. That's a statistical prediction, which Geiger's graded version allows (Def. 41), and the erasure work (LEACE, amnesic probing; *literature* entry) has the same form. Cost, stated once: 4a and 4b put the network into states no statement produces, so their ω is an extra assumption rather than derived from inputs. For a top rung that's acceptable if stated.
- *Why the middle rung uses swaps, not removals.* G2 measured side effects under Δ alone, against all-off. All-off is itself a broken model, so every method lands near or above it, and the comparison is between two broken models (the same problem as the "C alone on non-target" cell in the first entry). With swaps between twins the reference is the site swap, a state the network reaches. Qualities the twins share should stay put, and "how much more does the P-swap disturb them than the site swap" is a clean number [assumed: that this separates the methods better than G2 did; it may not].
- *Your grounding question, at the level of weights.* What is specifically weight-level about a tPD claim is rung 5: the same P must work on every input and under every truth prompt, with the CI it computes itself. A probe direction claims something only about the activations it was fit on. Rung 5 is where "the weights do it" gets tested, and there the existing data already shows every method failing (Yes/No).

### 3. Conclusions and proposal

- [concluded] The tests you have in mind are not beyond the theory; they need the outputs the narrowed H dropped. With other-quality readouts back in H, rungs 3–5 come from the same recipe as the swaps.
- [concluded, from the table's data column] The decompositions already fail a high rung (5, Yes/No transfer), and the removal version of rung 3 is inconclusive. So the suite isn't trivially aced; the spectrum you describe shows up in the data.
- [assumed] If plain's re-implementation shows anywhere in behaviour, it is at rung 3 (junk writes, under swaps) or rung 5 (different route). Rung 2, and the verdict half of 4b, will likely not separate it.
OUTDATED (26-11-06): the proposal in the next bullet (stop extending the formal theory) is replaced by *ladder derived from the definition*, under Julian's comment below it.
- Proposal: stop extending the formal theory. Use the ladder as the structure of the evaluation section: each rung = claim, alternative, edit, prediction. The paper's theory part then needs only H with the two kinds of readouts, the edit/prediction idea, and the conditional guarantee (short answer 4).  
C: My hope was to have a formal theory that basically produces the ladder as a consequence of it as of validating a decomposition against it

Answer (26-11-06, Claude Opus 5.5) — *ladder derived from the definition*: I think it nearly does, and most of the pieces are already in this LOG. Derivation [concluded unless tagged]:

**Definition.** "P captures T" = H (your (1)) abstracts the refined network, with T (more exactly: T together with the part of the content Z that P's writes carry) aligned to P's writes and nothing of T to R's writes, *for every intervention on the aligned variables, every readout and every statement*. That is one sentence. The ladder is what you get by splitting its "for every" into parts.

**Three axes**, one per "for every":
- *Interventions:* (i) values that another statement produces (interchange: P-swap, R-swap; ω follows from the inputs); (ii) values that no statement produces (T := not T with Z kept; T := "absent"; ω is an extra assumption taken from (1)); (iii) interventions on what P reads (Z at P's input; needs H's mechanism Z → T).
- *Readouts:* the training truth readout; the other truth readouts (second prompt, Yes/No); the other-quality readouts.
- *Inputs:* in-distribution twins; OOD forms and statements the model gets wrong.

**The rungs are cells of this grid:**

| rung | interventions | readouts | inputs |
|---|---|---|---|
| 2 carries T | (i) | training truth readout | in-distribution |
| 3 carries only T | (i) | other-quality readouts the twins share | in-distribution |
| 4a flip, 4b oblivious | (ii) | all truth and other-quality readouts | in-distribution |
| 5 coherent use | (i), (ii) | other truth readouts | OOD, model errors |
| 6 computes | (iii) | truth readouts | in-distribution |

Rung 0 (P + R = W, honest CI) is the precondition that makes the split a refinement at all, i.e. part of (2). Rung 1 (removals) does *not* come out of the definition: removals have no ω. It stays as a sanity check outside the theory, consistent with the earlier answer on removals.

**The order also follows.** Each step up relies on more of (1): (i) on natural states needs the least; other-quality readouts additionally need "they read only Z"; (ii) needs H to hold off the natural states; more inputs widen D; (iii) needs a finer H. So the higher a rung, the more of its failure can be the assumption's fault rather than the decomposition's. That is also why the next point is needed.

**A-checks: one column the ladder still lacks.** Each cell has a counterpart on the unsplit network that tests (1) itself, using the site swap (all of layers 11–17's output taken from x'). Does it move every truth readout together (one T)? Does it leave the other qualities that the twins share in place (they read only Z)? If the site swap fails a cell, that cell can't test the decomposition. Example: the Yes/No failure at rung 5 has to be checked this way first. If the site swap doesn't move the Yes/No answer the way it moves the 1/0 verdict, the failure belongs to (1) (no single T read by both prompts at these layers), not to the decomposition [assumed: not checked].

**Re-reading this LOG with the grid.** The clauses were all on the table before:
- rung 2 = E2 (step 5);
- rung 3 = P5 ("outputs that don't depend on τ … should still follow x"). I dropped it under the broad reading, because twins differ in content too. Restricting it to the qualities the twins share repairs that;
- rung 4b is close to the informational D3 ("the Δ-network carries no information about τ"), demoted to optional in step 5;
- rung 4a = R4; rung 6 = R3.

So the re-implementation blind spot came from clauses we dropped or deferred, not from a gap in the theory. That corrects my "the theory didn't find it" above (marked OUTDATED).

**What is still missing for your (1)–(4):**
- (1) written down once, as the fixed list of assumptions. The readout split (truth readouts read only T, other qualities read only Z) is new and unchecked; the A-checks test it.
- (3) per cell: an accuracy with an interval on a stated D. Cells are combined as a conjunction, so the result is a profile over the cells, not one score.
- The A-check column.

Proposal: I write this up as one section (definition → grid → ladder → A-checks). It can go into a DRAFT file in this topic or straight into the paper draft's `<capture>` section; your choice.

Open for Julian:
1. Is the ladder the shape you had in mind (sanity checks → middle → your high-end tests)? Rungs to add or drop?  
C: I really really like the ladder.

Answer (26-11-06, Claude Opus 5.5): Noted. With the derivation above, the ladder and the theory are now one thing: the rungs are cells of the definition's "for every". Question 2 (which other qualities) stays open.
2. Which other qualities? The twins must share them for rung 3. Already validated in G0: the digit questions (humor, emotion, surprise, word count) and the Yes/No prompts; recall and next-token KL within the statement are used in G1/G2.

---

26-11-06, Claude Opus 5.5 — *ladder written into the paper draft*

C: (transcribed from chat) I'll ignore question 2. for now, because I want to read your writedown into the paper draft under 1 first

Written as the subsection *From one definition to a ladder of tests* (`<ca-ladder>`) of the draft's `<capture>` section (`reports/26-11-04_tpd_truth_paper_draft.typ`); details of the edit in `convos/julian/26-11-04_paper_draft_LOG.md`, entry *ladder subsection added to the capture section*. It follows the two answers above (*the missing middle*, *ladder derived from the definition*), with the result numbers left out. Question 2 (which other qualities) deferred by Julian.

