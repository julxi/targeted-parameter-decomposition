# SUMMARY: Evaluation categories from first principles

**Last updated:** 26-11-06 (step 7: general picture without tPD, readouts and edits instead of a basis, superposition as a limit on the interventions; awaiting Julian)

Julian asked for a fresh go at the evaluation categories, independent of results so far. The RepE categories and the mediator taxonomy (used in `convos/julian/26-11-04_paper_draft_SUMMARY.md`) had mostly confused him. Starting point, his: a perfect decomposition W = C + Δ, with perfect separation of target and non-target data. He wants to settle a simple conceptual basis first, and not jump ahead to measurement choices.

## Step 1: four ablation cells, CI ignored (agent proposal, not ratified) [concluded]

Add "none" (the whole decomposed site removed) as a second reference next to W. Then:

| | target, truth readout | target, everything else | non-target |
|---|---|---|---|
| C alone | ≈ W: sufficiency | no requirement | ≈ none: C silent (diagnostic; better measured as \|Cx\|/\|y\|) |
| Δ alone | ≈ none: completeness | ≈ W: selectivity (target side) | ≈ W: selectivity (non-target side) |

- Δ alone on target should be *truth-independent relative to none*, not "constant". Whether it's constant is a property of the model.
- Without CI, necessity and completeness coincide (W − C = Δ). Individual necessity isn't required (redundancy).
- Sufficiency compares against W, not labels. Cancellation shows up in the completeness cell.

## Step 2: with CI, properties P1–P6 (agent proposal; partly superseded by step 3)

Julian restarted with CI: "Δ produces the same result as W on data without truth stimulus; C computes all activations relevant to use the truth value". The answer split this into P1 exact sum, P2 honest CI (inactive components removable on all inputs), P3 targeted (nothing active without truth stimulus), P4 swap test (C's write from an opposite-τ twin moves the verdict, Δ's doesn't), P5 that swap moves nothing else, P6 minimal. It recommended the narrow reading ("C holds only what carries τ").

Julian's comments and what changed:
- P2 honest CI: accepted by Julian ("tangible and good"); kept as B1.
- τ: Julian prefers the model's belief; assumption: the model has τ(x), and the verdict *measures* it.
- Narrow reading: Julian: not a usable basis. The agent agreed and withdrew the recommendation [superseded by the broad reading, step 3]. Reason independent of results: tPD's loss requires the active components with Δ removed to give the verdict, so it aims at the broad reading (everything the verdict needs from the site) [concluded].
- P3: Julian's "The city of Paris is …" example. Under the broad reading the prefix computation (reading "Paris", recalling facts) also runs on ordinary text ("… my home town"), so "nothing active on non-target text" can't hold [concluded]. Replaced by B3/B4.
- P4: Julian: involved, and a counterfactual not derived from "perfect". Answer: it's a definition, and any claim "C carries τ" is causal, so some counterfactual (removal or swap) is unavoidable. P4 as written was also flawed for several layers: Δ at a later layer passes on the τ C wrote earlier, so a perfect decomposition fails both halves [concluded]. Step 3 therefore starts from one layer.
- "What carries τ", made concrete with twins ("Paris is in France / Spain"): recall is identical in both, the object encoding is content, the comparison result differs with τ in every pair. "Contains τ" is observational (probe); "used" needs the counterfactual.
- P6: Julian wants minimality tied to necessity. With inclusion-minimality (no active component can be switched off without breaking another property) every active component is necessary where active. Kept as B4.
- tPD's loss vs the properties: Julian pushed back (tPD is a proxy, argued in the paper). Agreed; the point is only that properties the loss doesn't enforce are where evaluation finds something.

## Step 3: simple basis, one layer [superseded by step 4]

One weight matrix, binary CI. Assumption A: the model has τ(x) on statements; the verdict measures it. W = Δ + Σ_c W_c, C_x = active components on x.

- **B1 honest CI:** f(x; Δ + C_x) = f(x; W) on all inputs and outputs.
- **B2 C suffices for the verdict:** on statements, C_x alone (Δ removed) gives W's verdict (broad reading; tPD's own aim; how broad depends on zero vs mean ablation).
- **B3 targeted:** every component is needed for the verdict on some statement; everything else is Δ.
- **B4 minimal:** switching off any active component breaks B1 or B2 (each active component necessary).

Evaluation questions B1–B4 leave open:
- **Q1 sharing:** how much of C is active on non-target text (Julian's property 1 holds exactly where none is); a fact about the model, given B1–B4.
- **Q2 what in C carries τ:** the narrow question, asked inside C (twin pairs); P5 belongs here.
- **Q3 does Δ carry τ too:** completeness; a redundant copy or cancellation meets B1–B4.

Julian's objections: B2 uses a proxy for τ (the verdict); restricting to single layers is unconvincing; B1–B4 per site miss something holistic.

## Step 4: holistic definition (agent proposal, not ratified) [concluded]

W = all weight matrices of the model; each split into Δ^l + Σ_c W_c^l with binary CI. **Assumption A:** the model has an internal truth value T, a variable in its activations through which all truth-dependence of its behaviour passes, with T = τ(x) on statements; the verdict measures T.

- **D1 honest CI:** Δ + C_x reproduces W's outputs on every input.
- **D2 C computes τ:** in the network with only C_x, T = τ(x).
- **D3 Δ doesn't:** in the network with only Δ, T is independent of τ.
- **D4 nothing superfluous:** no component can be moved into Δ, and no active one switched off, without breaking D1–D3 (gives necessity).

Consequences and reasons:
- Stated with T, C doesn't need the machinery that turns τ into "1"/"0" (B2 forced that in).
- Holistic: D2/D3 quantify over all matrices at once; per-site properties can each hold while the whole fails. The single-layer simplification is withdrawn.
- The P4 "flaw" (Δ passing on τ between layers) disappears when read at T [concluded]: passing on before T means the decomposition isn't perfect; after T doesn't touch T. It came from reading at the verdict.
- Julian's property 2 = D2 + D3; his property 1 follows from D1 + D4 on text that uses no part of the truth function. Target data enters only through τ.
- Costs: broad reading; the C-only / Δ-only networks are used (fine for a definition, a problem for measurement); idealisation that every matrix is decomposed (otherwise D3 needs the "none" reference from step 1).
- "One mechanism" as an extra holistic requirement: proposed to leave out (redundancy is legitimate, D4 excludes unrelated parts).

**D3 made precise** (Julian asked what "T doesn't depend on τ" means): τ is a function of x, so the phrase only has meaning relative to a set of statements. Proposal: T in the Δ-only network is statistically independent of τ (no readout of it predicts τ above chance) on a statement set that is **twin-balanced** (each statement has a same-form twin of opposite τ) and **crossed** (every subject and object occurs in true and false statements), balanced with respect to the model's τ, not the labels. Rejected readings: "constant" (too strong; the Δ-network may answer from surface features) and "uncorrelated on our dataset" (confounded by form, misses non-linear dependence). Catches a redundant τ copy in Δ and cancellation. The statement set becomes part of the fixed setup [concluded]. Partial decomposition: no more information about τ than with the decomposed matrices removed.

**D3 in words, twin balance as the proxy** (Julian agreed). His worry: does independence conflate belief and truth? Answer [concluded]: the definition doesn't (D2/D3 mention only τ, the belief; errors reproduced by C are fine). The earlier wording "the only route to τ is the truth computation" did and is corrected: the belief function includes any heuristic the model uses. The proxy does: twins built from labels are balanced w.r.t. truth, not belief, wherever the model errs; fix: keep pairs whose verdicts differ (verdict accuracy 0.979 in-distribution, 0.801 OOD, so this matters mostly OOD). Belief and truth can only be told apart on statements the model gets wrong. D3's negative form also avoids saying "what remains" without the belief function.

## Literature (26-11-05, 16 web searches, abstracts only)

No prior definition of a perfect *targeted parameter* decomposition with an internal truth variable found [assumed: limited search]. Closest relatives: A ↔ Herrmann & Levinstein's belief standards (accuracy, coherence, uniformity, use); D1 ↔ SPD's causal importance, P1 ↔ APD's faithfulness; D2 ↔ circuit faithfulness (Wang et al.; Hanna et al.), with Ayonrinde & Jaburi's explanatory vs behavioural faithfulness motivating T over the verdict; D2/D3 with T and the swaps ↔ **causal abstraction** (Geiger et al.: high-level variable, interchange interventions, IIA), but aligned with activations, not weights; D3 ↔ **LEACE's linear guardedness** and amnesic probing; holistic D3 ↔ causal scrubbing (resampling); D4 ↔ Wang's minimality and Craver's mutual manipulability (inclusion sense), not APD's count sense. Warnings: non-identifiability (Méloux et al.: several decompositions may be "perfect"), and ablation choice changes faithfulness results (Miller et al.). Suggestion [assumed]: restate D2/D3 in causal-abstraction terms with linear guardedness as D3's first proxy; read Geiger et al., Herrmann & Levinstein, and the tPD paper itself (not yet read). Sources in the LOG entry *literature: who else has defined a perfect decomposition*.

**Geiger et al. read in detail** (JMLR 2025, pp. 1–10, 13–26, 37–41) [verified: read]: real maths, but the needed core is about four pages (alignment Def 31, constructive abstraction Def 33, interchange interventions Def 44/46, IIA Def 48, worked hand-built network in 2.6); the intervention algebras (2.2) and the method unification (section 3) are machinery or translation. Weights are inside mechanisms, not variables. Consequences [concluded]: (1) assumption A becomes a high-level model with an alignment saying *where* T is read; (2) **T cannot be read in the C-only or Δ-only network**: the alignment map is only defined on activations some input produces (p. 22), so D2/D3 as written are not well defined; an argument for defining by swaps; (3) their completeness is causal (Δ's τ information not *used*), weaker than our informational D3; (4) causal scrubbing (resample Δ from random statements, C from same-τ statements) is the template for a holistic D2+D3 test. Open: D3 informational or causal. Local copy for reference (26-11-06): `papers/Causal_Abstraction/ca_paper.md` (Markdown from the arXiv v4 LaTeX source; numbering matches the PDF; diagrams as PDF crops with text descriptions).

## Step 5: causal-abstraction version with causal D3 (agent proposal, not ratified) [concluded]

Julian asked for the definition in the new language with a causal D3. Idea: the decomposition *proposes the alignment*: T lives in the writes of the active components, so T need not be located separately (this answers his question why assumption A was "a cost"). Low-level model: activations, with each decomposed site's output split into a C-write and a Δ-write. High-level model H: statement → T → verdict (assumption A as a causal model). Alignment: T ↔ all C-writes.
- **E1 honest CI** (= D1).
- **E2 the C-writes are T** (replaces D2 + D3): for every verdict-balanced, same-layout pair (x, x'), running x with all C-writes set to their values on x' (everything else, Δ-writes included, recomputed as usual) gives the verdict of x'. That is IIA = 1, H a constructive abstraction. It contains C sufficing and the causal D3: a Δ route from x to τ(x) that the verdict reads would pull the verdict back. No hybrid networks; the τ → "1"/"0" machinery stays live, so C needn't contain it (the B2 objection), and the pass-on problem across layers disappears.
- **E3 nothing superfluous** (= D4).
- Partial decomposition: compare with swapping all site writes (IIA_site): perfect means IIA(C-writes) = IIA_site and Δ-writes alone at the no-swap level. IIA can use s instead of the verdict.
- Dropped: P5 (broad reading: C-writes carry content too; the narrow question stays as Q2, a finer alignment). Informational D3 becomes optional.
- Costs: same-layout pairs needed; twins differing in the object also carry content through the swap; same content + opposite τ + same layout don't come together easily (negation adds a token) [assumed: datasets not checked]; non-identifiability.

## Step 6: the basis and superposition (26-11-06, agent answer, not ratified) [concluded]

Julian reworked the draft's `<capture>` section (intuition; causal models; a high-level model with many features F_1…F_k and all logits as output; the network as the low-level model; alignment) and got stuck: what "where T is computed in the weights" means, how to find "a right basis", and whether superposition breaks the alignment. Answer (LOG entry *stuck on the basis: narrow the output, the decomposition is the basis, superposition*):
- *Narrow the tested model.* With all logits as output, every feature the logits read needs a cell: the full superposition problem. Keep the feature model as intuition and test its marginalization (Geiger Prop. 40) X → Z → T → readouts, with Z the content bundle; only T needs a cell. Several readouts (verdict prompt, second prompt, Yes/No) can share one swap, because the statement tokens' activations don't depend on the appended question. That makes "uses the belief coherently" testable.
- *The decomposition is the proposed basis.* In causal abstraction the change of variables before alignment is an empirical hypothesis (Geiger §3.1, §3.6). tPD's split y = c + d is such a change in parameter space (the draft's existing step 3). It is a refinement, not a rotation, with the same status as an SAE with zero reconstruction error; Δ plays the role of the SAE error term.
- *Superposition* doesn't break the definition. It rules out clean *fixed* linear cells, but the C-write is input-dependent. What remains is that the cell carries a bundle Z ⊃ T; the swap fails if the verdict also reads x's content by a route outside the cell (through Δ: a real failure; bypassing the sites: absorbed by the site-swap reference).
- *"Where in the weights"*, in three claims of increasing strength: R1 T sits in the components' outputs; R2 Δ doesn't supply it (both testable by swaps, the current definition); R3 the components compute T from content (needs path tests; the draft's open point *Representation or computation*). Suggested intuition wording and a section order are in the LOG.

## Step 7: the general picture, without tPD (26-11-06, agent answer, not ratified) [concluded]

Julian, still confused by step 6, wants to start as general as possible, with no argument from the method ("Pleaes don't try to shift the basis in favor of the resulting methods"). His questions: activations don't cluster into features, so how to formalize "a basis aligned with the high-level view"; does causal abstraction handle superposition (two mutually exclusive features that the model may conflate); how to express "where T is computed". Answer (LOG, answer under his last comment, *the general picture without tPD*):
- **G1** low level = the network (activations; weights inside the mechanisms); high level = the feature view (features with mechanisms, T one of them).
- **G2** the link is a *readout* and an *edit* per feature; H abstracts L *relative to a set of interventions* (Geiger Def. 25, Domain(ω)). A basis (each feature owns coordinates) is the special case, impossible with more features than dimensions.
- **G3** superposition, e.g. h = F1 − F2 read by ReLU: readouts exist, but edits exist only for interventions that keep F1, F2 exclusive. Honest options: restrict the interventions, or merge them into one variable. Consequence for us [assumed]: setting T against the content is the kind of edit superposition may make impossible.
- **G4** "computed in these weights" is a claim about mechanisms: a readout of T (or the output as stand-in) and of its inputs Z, plus a split of the layers' mechanisms (W = P + R); P computes T if T depends on Z only through P's writes and R's writes carry nothing of it that is used. Superposition in activations doesn't block this.
- **G5** check against our methods: the representational question was never tested on its own (B3 is the only edit); the step-5 swap is a G4 test with the verdict as readout; not measured: computation vs passing on, and the off-distribution problem of G3.

**Open:** whether E1–E3 work as the basis (then: which pairs the datasets allow); earlier: whether D1–D4 work as the definition (Julian: "much better"; D3 accepted in words); then how to measure T, and the partial-decomposition case (only some matrices decomposed).

## See also

- See also: [convos/julian/26-11-04_paper_draft_SUMMARY.md] — the paper's current evaluation section (S0, A–F), which this topic may replace.
- See also: [convos/julian/26-11-05_verdict_digit_trust_SUMMARY.md] — whether the verdict digit is the model's answer ("1" as "10%", "0" as "0.9"); matters for the verdict as the measurement of τ.
