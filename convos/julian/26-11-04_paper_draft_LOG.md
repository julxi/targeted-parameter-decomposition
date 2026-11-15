C: (transcribed from chat) It's time to start writing the paper (for now in the report folder). I want to do this bit by bit. I'll tell you what to write and you add it in. The output should not be the real paper yet but a pre-version that has as many details as necessary to be basically a very detailed description.

First you have to identify a good conversation for this (maybe a new one)

Then we start with a basic description of the three methods (not the extact details, but which positions are trained and where does the loss come from)

---

26-11-04, Claude Opus 5.5 — *topic opened; methods section drafted*

I have read INDEX.md, the SUMMARYs of `convos/julian/26-10-29_paper_writing` and `convos/julian/26-10-22_tpd_truth_paper`, the three paper configs and the final runs' stored configs (`~/spd_out/runs/spd-s-{7152548c,eeac663f,0c2fa0d7}/final_config.yaml`), and the training code for the parts the methods section describes (`spd/run_spd.py` target and non-target blocks, `spd/models/output_lens.py`, `spd/utils/component_utils.py` mask sampling, `spd/routing.py` subset routing, `spd/metrics/hidden_acts_recon_loss.py`, `spd/metrics/orthogonality_loss.py`, `spd/persistent_pgd.py` header). I have not read the 26-10-29 LOG.

**Which conversation.** A new topic, this one. The paper-writing topic (26-10-29) planned writing as its stage 4, but it is 679 lines of analysis (tables, explanations, GPU list) and still has open analysis threads; bit-by-bit drafting produces its own back-and-forth, which would bury both [assumed: agent's choice, Julian may merge them]. Both SUMMARYs link each other. Analysis questions that come up while writing go to the 26-10-29 topic; this topic holds the text decisions.

**Draft file.** `reports/26-11-04_tpd_truth_paper_draft.typ` (Typst, uses `reports/template.typ`; PDF gitignored, build with `typst compile reports/26-11-04_tpd_truth_paper_draft.typ`). It plays the role of a DRAFT file: edited in place; this LOG records what changed and why. Working title from the experiments topic: "Trying to decompose a truth mechanism with tPD".

**Methods section written** (section 1 of the draft). Content, all from code and final configs [verified: code read 26-11-04]:
- Shared setting: Qwen2.5-7B-Instruct frozen; verdict prompt `<statement>\nThe amount of truth in this statement is `, tokenized jointly ('.' + '\n' → '.\n'); `down_proj` of layers 11–17, 96 rank-1 components each plus Δ = W − ΣUVᵀ (no free parameter); one global CI network (`global_shared_mlp`, reads the inputs of all seven matrices).
- Positions: components replace the weights only at the statement tokens (first token through '.\n'); the question tokens run on the original weights (`decompose_stimulus: false`). The decomposition reaches the verdict only through attention from the question tokens.
- Plain tPD's loss sources: (1) verdict KL (original vs masked next-token distribution at the last question token, full vocabulary) under random masks: components in [CI, 1], Δ uniform in [0, 1] per layer and position, plus subset routing (per position a uniform k ∈ 1..7 of the layers masked, the rest original); counted once per decomposed position, so longer statements weigh more (`StimulusOutputLens` repeats the verdict logits at every position). (2) Same KL under persistent adversarial masks, last 20% of training, coefficient 0.5. (3) Importance minimality on statement-token CIs. (4) Pile: next-token KL at all positions with Δ forced to 1, components random in [CI, 1], importance minimality ×2. (5) CI-scaled weight decay 0.1 on component vectors. The loss signal on target data comes only from the verdict.
- Recon: `StochasticHiddenActsReconLoss`, relative, coefficient 10; masked forward with all layers masked (`AllLayersRouter`, no subset routing), Δ random; each matrix's output at each statement token vs the original model's output, squared error / |y|²; the masked pass's input already carries upstream changes. Signal from the matrices' outputs, not the verdict; not on Pile (excluded in `run_spd.py`).
- Ortho: `ComponentOrthogonalityLoss`, `all_components`, coefficient 10; (⟨C, D⟩/|y|²)² per position and layer, C = all components at mask 1, from cached original inputs (no extra forward); trains U, V only (no CI gradient); sum condition, not per component; not on Pile. Geometric reading (C on the sphere with diameter y, so |C|, |D| ≤ |y|) from the code's docstring, re-derived [concluded: two-line algebra].
- Settings table: plain/recon LR 5e-4, ortho 2e-3 (tuned because ortho at 5e-4 recovered less in tuning: 0.27–0.52 per the 26-10-22 SUMMARY); importance minimality 1e-3 with warm-up to 4× in the first 20% (`coeff_peak_multiplier 4`, `coeff_warmup_frac 0.2`); 5,000 steps; batch 64 + 16 Pile windows; seeds 1, 2 (ortho 1, 2, 3).

Choices in the text the user may want to change [assumed: agent's choices]:
- One sentence of motivation for the two extra terms, citing the earlier pilot (Exp 4a, plain tPD at the end token only) as a re-implementation with cancellation, without numbers. The paper's own plain runs show only moderate cancellation (26-10-29 SUMMARY, *Cancellation in plain tPD*); that belongs in results, not here.
- No numbers from results in the methods section.
- The summary box on page 1 holds the draft's status rather than a bottom line.

Not yet in the draft: data section, evaluation states and metrics (components only, Δ only, all off, KL recovered, CI-weighted vs removal), tuning procedure, results, introduction.

---

26-11-04, Claude Opus 5.5 — *where the per-layer subset routing comes from*

Julian asked where the random layer subset in the verdict loss comes from; he had not heard of it.

- What it is [verified: code]: the target verdict loss is `StochasticReconSubsetLoss` with `routing: {type: uniform_k_subset}` (`spd/metrics/stochastic_recon_subset_loss.py`, `spd/routing.py`). At each position k is drawn uniformly from 1 to 7, k of the 7 decomposed matrices use the masked decomposition and the rest return the original output (`torch.where(routing_mask, components_out, output)` in `spd/models/component_model.py`). The Pile loss uses the same class (with Δ forced to 1), so the routing applies there too. The recon term uses `AllLayersRouter` (all 7 masked).
- Where it comes from [verified: git log]: the loss class comes from upstream SPD (consolidated in `f931fb2`, Dan Braun, 2025-10-03). It generalises the SPD paper's two reconstruction losses (`papers/Stochastic_Parameter_Decomposition/spd_paper.md`: stochastic-recon, all layers masked, and stochastic-recon-layerwise, one layer at a time): k = 1 is the layerwise case and k = L the full case. The routing entered this project's honesty configs with Madina's first honesty commit (`5b49341`, 2026-09-01), and every later config, the paper configs included, inherited it. The tPD paper's text does not mention routing (grep for "rout" in `tpd_paper.md`: no hit).
- It was recorded once before: in the tPD method reference, `convos/julian/26-10-02_overview_of_goal_LOG.md` (the `StochasticReconSubsetLoss` bullet).
- The draft sentence now says it is the codebase's inherited standard loss and how it relates to SPD's full and layerwise losses.

Whether the paper should keep it as a stated detail or justify it is Julian's call. It is the same in all three methods, so it does not affect the comparison between them [concluded: shared setting].

---

26-11-04, Claude Opus 5.5 — *Julian's draft comments on the methods section addressed*

Julian left `//` comments in the draft (chat: "I've made some comments in the draft"). They are removed from the draft as they are addressed, so they are recorded here verbatim (Julian's words, copied by the agent from the draft file), each with what was done. Sources read for the numbers: run B1 (`~/spd_out/26-10-22_tpd_truth_paper/b1_full/b1_full.log`), run B2 (`…/b2_full/b2.json`, `b2_full.log`), the causal-layer-scan entry in `convos/julian/26-10-10_bolder_experiments_LOG.md`, the arm A config comment (`config_truth_all_tokens.yaml`), the opening of `convos/julian/26-10-16_faithful_targeting_LOG.md`, `spd/configs.py` and `spd/persistent_pgd.py` (routing of the adversarial loss).

1. > "Here I want the data of the experiment that shows that" (after "answers with the digit 1 (true) or 0 (false)")

   Added the score definition (s = log P(1) − log P(0), true if s > 0) and Table 1: the original model per family on the test split, from B2 [verified: re-read `b2.json`; pooled accuracy 0.979 and digit mass 0.984 recomputed from it]. Weakest: element_symb (true statements 0.750) and the negated families (0.929–0.961 accuracy).

2. > "can you also include a picuter (preferibly not a figure) that shows the setup in an example? I want some errors that show: flow of information (through attention and loss)." and "also show the difference in training between target-data and non-target data"

   Drawn in Typst with cetz (helper file `reports/26-11-04_tpd_truth_paper_draft_diagram.typ`), two panels: a statement under the verdict prompt (decomposed `down_proj` row blue at the statement tokens, grey at the question, attention arrows from statement tokens to the answer position, only the answer's output into the loss, the other outputs marked "—") and a Pile window (decomposed at every position, KL at every position, m_Δ = 1). I read "errors" as arrows. Open: I read "preferably not a figure" as "a diagram drawn in the document rather than a data plot", and put it in a numbered figure with a caption (Figure 1). If you meant no floating figure environment, the wrapper is easy to drop.

3. > "Here I want the numbers why these layers and down projection have been choosen."

   Added *Why these layers* (four points) and Table 2 from B1 (mean ablation at all statement tokens, test split) [verified: re-read `b1_full.log`; single-layer ranges recomputed with awk]: single layers 5–24 leave 0.77–1.02 of the separation; MLP 11–17 leaves 0.05 (accuracy 0.538, digit mass 0.977); 11–14 0.38, 15–19 0.33, 6–10 0.18; attention 11–14 0.44, 11–17 0.21; MLP + attention 11–17 0.02. End token only: 0.82 (26-10-16 scan, verified there). Why 6–10 is out stays `#assumed` in the text (from the 26-10-16 choice: 12 layers, possibly content processing). Added *Why only `down_proj`*: its output is the MLP's whole output, so the scan is a scan of it; limits (MLP input side not decomposed; attention matters as much at 11–14, but every route passes through MLPs 11–17); the restriction was originally made for memory (arm A config: "Narrowed for memory … Layers 15-19 down_proj are an unvalidated guess"). The original row differs between B1 (0.977) and B2 (0.979); said so in the caption, cause not traced.

4. > "this here is redundant:" (the sentence that Δ is not a free parameter)

   Removed.

5. > "Here I would rather write that to me the reasons for this is not quite clear. If I could I'd rather had the results without this change (it seems like engineering). However, we stick with this implementation detail because all the tests had already been run."

   Written into item 1 in paper voice: purpose not clear to us, looks like engineering, we would have preferred results without it, kept because all experiments had already been run with it.

6. > "what does that mean? what is the model's preditcion within the statement?"

   Answer: the model's next-token prediction at each statement token, i.e. what it expects as the statement's next word (after "The city of" it predicts a city name). tPD on ordinary text compares exactly these; our verdict lens does not. Rewritten in the text as "the model's next-token predictions at the statement tokens (what it expects as the statement's next word) are not compared"; the diagram shows them as "—".

7. > "a motivation of these losses should go into the general overview not here. The single ',which' above is a good call back though"

   New subsection 1.2 *What the losses ask for* (before Method 1) holds the motivation of the target losses and the Pile stream. In the list only ", which …" call-backs remain (importance minimality; Pile).

8. > "that is something that we should keep in mind and check in the actual decompositions" (weight decay scaled by 1 − max CI)

   Left in the draft as Julian's to-do and added to the SUMMARY's open items. Why it may matter [assumed]: the decay shrinks components that are never important in a batch, so the inactive components' write (and with it any cancellation they carry) is pushed towards Δ over training.

9. > "let's keep this discussion out of here. also I think it's too simplistic" (*What plain tPD leaves free*)

   Removed.

10. > "first the name then the description" (heading of Method 2; "as above" for Method 3)

    Headings now "Method 2: recon (plain tPD + per-layer reconstruction)" and "Method 3: ortho (plain tPD + global orthogonality)".

11. > "incorporate this into a proper sentence / paragraph or leave it out. You can foreshadow that this is meant to deal with a reimplementation problem from Method 1" (and "as above, make this text" for ortho)

    Both methods now open with a paragraph: recon is meant to prevent the components re-implementing the verdict instead of decomposing the MLPs ("we come back to this in the results"); ortho addresses the same failure without giving up targeting.

12. > "I would include the idea behind this: it's based on another assumption that the truth relevant direction is relatively orthogonal to all other features these layers compute"

    Added as the ortho opening, tagged `#assumed`: the verdict-relevant part of each MLP output is roughly orthogonal to everything else these layers write, so the output splits into two orthogonal parts, one for the components, one for Δ. (Julian's first description of the idea, 26-10-16, called it "a middle approach … not very principled but more like a band aid"; the new framing is his.)

13. > "can you also include the uniform affectely layer's. It's also not there for recon, right?"

    Right [verified: code]: the recon term masks all 7 layers (`AllLayersRouter`); the adversarial verdict loss also masks all 7 (`PersistentPGDReconLoss`, not the `…Subset…` variant, so `_get_router_for_ppgd_config` returns `AllLayersRouter`); the ortho term masks nothing (each matrix on its original input). Only the random-mask verdict loss and the Pile loss use the random layer subset. Added rows to the settings table and "all seven layers masked" to item 2 and the recon paragraph; item 4 now names the subset.

Draft compiles; pages checked by eye at 80 ppi.

---

26-11-04, Claude Opus 5.5 — *settings cell of the adversarial loss reworded*

Julian asked (chat) what "coefficient 0.5, from 80% of training" in the settings table means. It is the adversarial verdict loss (`PersistentPGDReconLoss`): weight 0.5 in the total loss, against 1 for the random-mask verdict loss; skipped entirely (not computed, masks not optimised) while the training fraction is below `start_frac` 0.8, i.e. active only for steps 4,000–5,000 of 5,000 [verified: `spd/losses.py` (`if current_frac_of_training < cfg.start_frac: continue`), `spd/run_spd.py` (`active_ppgd_configs`), final configs]. The table cell now says this in words.

---

26-11-04, Claude Opus 5.5 — *diagram: KL label, attention from layer 12, conceptual alternative*

Julian (chat), on the setup diagram:

C: (transcribed from chat) I would just write KL like for non-target data since it's the same mechanism, right? I also have a question about the arrows for the attentition. Already layetrs 11-27 of the verdict prompt are influenced by the results of the W'. Also I would have liked a somewhat more conceptual picture. The tokenized view is technically very accurate but a bit harder to get. Can you show me an alternative?

- KL: yes, the same KL(original ‖ masked) of the next-token distribution over the full vocabulary (`calc_kl_divergence_lm`), only at the answer position instead of every position. The answer position's output box now says "KL", like the Pile panel's boxes. The line above says "KL to the original model's next-token distribution, at the answer position only".
- Attention: Julian is right that the old arrows (into layers 18–27 of the last token only) were wrong. In a Qwen2 block attention comes before the MLP, so W' at layer 11 changes the statement's residual stream, and the question tokens read it through attention from layer 12 on [concluded: block order]. Option A now draws the arrows into the question tokens' layer 11–17 row, labelled "attention, layers 12–27". Every box with original weights but a changed input is dashed light blue: the question's layers 11–27 and the statement's 18–27. In the Pile panel this is layers 18–27. Precise point: the question's layer-11 block itself is unaffected, so the label says 12–27. The draft caption of Figure 1 was updated to match.
- Alternative: `conceptual_panel` in `reports/26-11-04_tpd_truth_paper_draft_diagram.typ` (Option B). It shows blocks instead of tokens: statement | question | Pile as three columns, rows input / layers 0–10 / 11–17 / 18–27 / output. Masks are written in the W' blocks, with attention arrows from the statement's 11–17 block to the question's 11–17 and 18–27 blocks, and the outputs "next-word predictions: not used" vs "answer 1/0: KL". Both options were rendered side by side for Julian (session scratchpad, `diagram_options.pdf`). The draft keeps Option A until he chooses.

---

26-11-04, Claude Opus 5.5 — *diagram: Option A kept, arrows removed*

C: (transcribed from chat) I like A more. So we stick with that. I'd remove the blue arrows though.

Done. The attention arrows and their label are removed from the token diagram, and the row gap they used is closed. The dashed light-blue shading (original weights, changed input) stays, and the Figure 1 caption still explains it via attention from layer 12 on. Option B's code (`conceptual_panel`) was deleted from the helper file; its design is described in the previous entry if it is ever wanted again.

---

26-11-04, Claude Opus 5.5 — *the four evaluation categories, recalled for the methodology section*

Julian (chat): "Next in the writup we should describe the experiments methodology / categories. I don't know if I still stand by the chategories that I've suggested previously. What were they?"

Sources re-read: the opening request of `convos/julian/26-10-22_tpd_truth_paper_LOG.md` (Julian's category text, adapted from Zou et al. 2023, *Representation Engineering*), section *4. The four categories* there with his C: comments, *D3 proposal*, *D3 plan, revision 2* and *revision 3*.

- The categories as Julian proposed them (26-10-22): correlation (neural correlates, prediction accuracy in- and out-of-distribution), manipulation (stimulate/suppress against a baseline, causal), termination (remove, measure degradation: necessity, lesion studies), recovery (remove everything, reintroduce the identified activity: sufficiency, rescue experiments).
- Julian's comments then: CI probes count as correlation but give no evidence (an untrained CI network reads truth as well), so "we have to find something better"; manipulation may give weak evidence; no fixed definition of "mechanism", only heuristics, "stay honest that we don't really understand what's going on" [decided: C: comments there].
- What they became in the frozen plan (D3): recovery = components only (Δ off) on test, OOD and the second prompt; termination = T0 Δ only (all components removed), T1 single removals, T2 top-k curves vs random/controls, T3 = A8 (removing the components' write span from the *original* MLP output); correlation = K0 all-CI probes shown as non-evidence, K1 single-component selectivity and sparse probes vs controls, K2 OOD transfer of selectivity, K3 truth vs verdict on the model's errors; manipulation = version A (top-10 by T1 plus the label-selective group, ×−1 and ×2, effect-selected controls, paired-twin test, exploratory hand-reading). Outside the four: side effects (Δ-only Pile KL), active vs inactive components / functional active set, truth directions (section 5), and later the GPU list (next-token side effects, digit questions, Yes/No transfer).

Agent's view on whether they still fit [assumed: proposal for Julian]: in the chat reply (summarised): the categories mix two kinds of test that the analysis topic separated: tests inside the decomposition (recovery, T0–T2, manipulation of components) and tests anchored in the original model or controls (A8, write alignment with the truth direction, random/data-aligned controls, side effects). Candidate lesson L4 (`convos/julian/26-10-29_paper_writing_SUMMARY.md`, *Where plain tPD fails*) is exactly that the first kind can't tell decomposing from re-implementing. An organisation by question may serve the paper better than the RepE categories. Awaiting Julian's choice.

---

26-11-04, Claude Opus 5.5 — *fresh agent asked to cut the evaluation categories*

C: (transcribed from chat) The four categories are designed for a signal. We are rating a potential mechanisms. Maybe we should cut the categories differently. Can you ask a fresh non-biased agent how they would cut the categories for our situation?

Launched a fresh general-purpose sub-agent (not a fork, so it does not inherit this conversation). Its prompt: the setup (task, verdict prompt, down_proj 11–17, components + Δ = W, CI, training losses, the three method variants in one line each, what interventions and baselines are possible), the goal (rate how well a decomposition recovered a candidate truth mechanism), the RepE categories named as the framework in question, and the team's suspicion that they cut the problem in the wrong place. It asks for categories, failure modes and controls per category, essential vs nice-to-have, order, and the mapping to RepE, under ~1,200 words. To keep it unanchored, it was told not to read INDEX.md, `convos/`, `reports/`, `scratch/`, FUTURE_WORK.md, FEEDBACK.md or earlier evaluation plans, and it was given no results and neither our D3 categories nor the agent's "two kinds of test" proposal. Allowed: the tPD and SPD papers in `papers/`. Not hidden: the RepE categories themselves (Julian's question is about them), and the three method variants (which hint at the re-implementation concern). Its answer goes in the next entry.

---

26-11-04, Claude Opus 5.5 — *the fresh agent's categories, and how our analyses map onto them*

**The sub-agent's answer** (condensed by me, structure and claims kept; it read only the Method and "incomplete circuits" sections of the tPD paper). Its diagnosis of why RepE cuts wrongly: we rate an optimiser-found set of weight edits, not a read-out signal. Three problems follow. There are two models (the hybrid with Δ removed, and the original). The output is one bit, so the optimiser can build a classifier instead of finding one. Δ is a hiding place for whatever the components leave out.

- **S0 Scope** (framing, measured first): what the decomposition can see at all (down_proj, layers 11–17, statement tokens). Is truth already decodable at the layer-11 input (then 11–17 may only relay it)? Does the verdict form mostly at the question tokens? Probe each site and patch site by site. This sets the ceiling on any claim.
- **A Faithful sufficiency**: the components alone reproduce the verdict (held-out, OOD, second prompt) *and* the original's internal states (residual after layer 17 and at the question tokens). Failure modes: an invented classifier off the original activation distribution, reading surface cues (most likely in plain tPD; ortho blocks cancellation, not invented writes). Controls: an activation-match number, a shortcut probe set (lexically matched pairs), matched-rank baselines (same pipeline on a same-format control task; randomly initialised decomposition). Caveat: masks outside [CI, 1] (negative, > 1) and untrained per-token patterns are outside what training guarantees.
- **B Necessity and selectivity in the original model**: remove the components from W (keep Δ and inactive components). Report separately (i) true-vs-false accuracy, (ii) the shift towards always "1" or "0", (iii) Pile KL and a same-format control judgement. Controls: matched-norm rank-1 directions that are random, top principal components of the down_proj outputs, and a difference-of-means truth probe turned into a rank-1 weight edit (the most important: if the probe edit does as well, tPD added nothing). A weak ablation effect with strong sufficiency = redundancy in Δ, i.e. incompleteness.
- **C Completeness (accounting for Δ)**: the agent calls it the most important and says it has no RepE counterpart. Δ alone and full-minus-components: do verdicts survive? Probe truth from Δ's writes vs the components'. The tPD paper's redundancy test (an ablation hurts more in components-only than in the full model). Confound: Pile-active parts of the mechanism are pushed into Δ.
- **D Counterfactual content**: interchange interventions on minimal pairs (Paris/France vs Paris/Spain; with/without "not"; 51 > 58 vs 58 > 51). Swap the CI-weighted writes from the donor and check whether the verdict follows the donor's truth value. Separate a statement-dependent truth variable from a statement-independent push towards "1" (an output bias). Dose-response, transfer to German, conjunctions and the second prompt. Prefer swaps over added vectors (a 1-bit output flips under almost any large write).
- **E Structure and interpretation**: read/write vectors, logit lens on writes, where CI fires, sub-mechanisms and composition, stability across seeds and variants (matched subspaces), every story turned into a prediction tested on new statements. Nice-to-have.
- **F Decodability (demoted)**: CI/write correlation with truth is nearly automatic.
- Essential: S0, A, B, C, D. Order: S0, A, B, C, D, E; OOD as a column in every table; all variants plus baselines at matched rank in every figure. Mapping: correlation → F, manipulation → D, termination → B, recovery → A, new: S0 and C. Its own uncertainties: activation matching may be too strict (report it as a continuous number); a fair rank-1 probe-edit baseline is unclear; if truth is decodable before layer 11, the honest result may be "relay and integration step".

**Mapping onto what we already have** [concluded: from the frozen plan and the analysis SUMMARY; nothing re-run]:
- S0: partly. The causal layer scan and the B1 attention scan say where the verdict *depends* on (MLP 11–17 necessary). Whether truth is already decodable at the layer-11 input was not measured in the paper set [assumed: not found in the 26-10-29 SUMMARY; worth a grep before claiming it]. Earlier truth-direction work used layers 15–19 and 23.
- A: recovery (components only, test, OOD, second prompt). The activation-match part is missing. The recon method's reconstruction error is a per-matrix version for recon only.
- B: T1/T2 removals act on the full model (keep Δ). T3/A8 removes the write span from the original MLP output. Controls: random, data-aligned basis (input principal directions), untrained decomposition. The separation *shift* (added 26-10-30) is the agent's (ii). Side effects: Pile Δ-only KL, the G2 next-token readouts. Missing: a probe-derived rank-1 edit baseline, and a same-format control judgement (the G0–G2 digit questions come close).
- C: T0 (Δ alone), the components' vs Δ's share of the truth gap and write alignment (directions section), cancellation norms. Missing: the explicit redundancy test (ablation in components-only vs full).
- D: manipulation (×−1, ×2) and the paired-twin test. These are added-vector edits outside the trained mask range, the agent's extrapolation caveat. True interchange swaps on minimal pairs were not run; tiu's affirmative/negated twins would make some of them possible.
- E: the exploratory per-component tables (wide tables, hand-reading). Seed stability is known to be poor for ortho s2 vs s1/s3 on transfer.
- F: K0–K3.

My view [assumed: for Julian]: the scheme fits the "rating a mechanism" framing better than RepE, and C and S0 name things our story already depends on (Δ alone, cancellation, "the components re-implement"). Most of our analyses map onto it without new runs. Real gaps: S0 decodability before layer 11, A's activation match, B's probe-edit baseline, C's redundancy test, D's interchange swaps. Each needs a GPU. Whether to run any is Julian's call; the section can also present the gaps as limitations. One caveat on the agent's scheme: "two models" and C overlap with candidate lesson L4 (within-decomposition tests can't tell decomposing from re-implementing), which it arrived at independently. That agreement is weak evidence, because the method variants in its prompt hinted at the concern.

---

26-11-04, Claude Opus 5.5 — *evaluation section: categories, experiment table, glossary*

C: (transcribed from chat) Can you add the categories that we have to the draft and also include a table which experiment falls in which one. Then also give e glossary of experiment name + experiment description

Reading: "the categories that we have" = the fresh agent's scheme (S0, A–F), which the conversation had just arrived at. To cover the other reading too, the table also gives each experiment's category in the frozen plan (the RepE four). If Julian meant the RepE four as the structure, the section has to be reorganised [assumed].

New section 2 *How the decompositions are evaluated* in the draft:
- 2.1 why not the RepE categories (two models, one-bit output, Δ as hiding place). It states that the categories were proposed by an independent reviewer and adopted afterwards (`#assumed`). The analyses were fixed in the plan before test/OOD were read, except G0–G2, which were added after the first results.
- 2.2 a table of the six categories: question / what a positive result shows / the main way to pass without a real mechanism and the guards (our controls, with the not-run tests marked).
- 2.3 controls (untrained decomposition, data-aligned basis, random sets and subspaces) and measures (KL recovered, separation share and shift, flips, digit mass, accuracy/AUC).
- 2.4 Table 3: experiments by category, with the frozen-plan category as the last column, and the gaps as "not run" rows: decodability before layer 11 (S0), internal-state match (A), redundancy test (C), interchange swaps (D). One gap is only "partly covered": a probe-derived baseline edit (B). Dir-remove projects the mass-mean / logistic-regression direction out of the original MLP output, which is close but not a matched-rank weight edit.
- 2.5 glossary, 31 entries (layer scan, B1–B3, G0, R, R-active, R-mean, active set, G2 Yes/No, T0, R-inactive, cancellation norms, T1, T2, T2 Yes/No, T3, Dir-remove, SE-Pile, G2 next-token, G2 digit questions, M-core, M-twin, M-wide, Dir, valley check, seed comparison, K0–K3). The descriptions come from the scripts' docstrings and constants (`paper_{recovery,termination,correlation,manipulation,directions,side_effects}.py`, `paper_common.py`: K_GRID, N_RANDOM 5 (T2) / 10 (M), N_RAND_SUB 3, TARGET_SHARE 0.95, M_GRID, K_SPARSE (1, 2, 5, 10), SELECT 0.2, TRANSFER 0.1, TOP 10, N_BASIS_STATEMENTS 512) and from the SUMMARYs for B1–B3 and the layer scan [verified: docstrings and constants read 26-11-04]. Names like R-active, R-mean, R-inactive, Dir-remove, SE-Pile are new labels for states/analyses that had no short name.

Placement choices [assumed]: K3 in F (correlational, although it is about truth vs verdict); Dir split into gap shares (C) and per-component cosines (E); G2 digit questions as B's same-format control judgement; the valley check in E. Draft compiles; pages checked by eye.

---

26-11-05, Claude Opus 5.5 — *recon term on the Pile? checked in code*

Julian asked in chat whether tPD + recon also applies the reconstruction term to non-target (Pile) data. Informational question, not transcribed.

Answer: no [verified: `spd/run_spd.py`, the `nontarget_loss_configs` filter in *Nontarget setup*, read 26-11-05]. The Pile loss list is built from `loss_metric_configs` and drops `StochasticHiddenActsReconLossConfig` (recon), `ComponentOrthogonalityLossConfig` (ortho), `UnmaskedReconLossConfig` and the PPGD losses. The code comment says these terms "constrain the components' share of each module's output, which is the target data's job; on nontarget data delta carries the computation". On the Pile recon therefore gets the same terms as plain tPD: output KL with Δ at 1, and importance minimality at 2× the target coefficient. The draft's sentence "The term is not applied to the Pile" (Method 2) is correct, and so is the same sentence for ortho.

Side finding: `config_paper_recon.yaml` and `config_paper_ortho.yaml` carry `coeff: 1` for the method term. The finals used 10 [verified: `final_config.yaml` of `spd-s-eeac663f`, `spd-s-edc17041` (recon) and `spd-s-0c2fa0d7` (ortho) in `~/spd_out/runs/`, all `coeff: 10.0`]. The run driver overrides the yaml values (see `convos/julian/26-10-22_tpd_truth_paper_LOG.md`, "The tuning runs override LR, importance-minimality and the method coefficient"). So the repo configs do not reproduce the finals as they stand. Not changed; Julian to decide whether to set the final values in the yamls before release.

---

26-11-05, Claude Opus 5.5 — *optional-loss marker in Figure 1, new restriction figure*

C: (transcribed from chat) I want to change this figure a bit. For the Target Data you should include something that indicates that we optionally base a loss on the decomposition like recon or ortho.
C: (transcribed from chat) I also want another figure explaining how these restrictions look like. What I had in mind is three pannels each showing y, C, D and what the losses want them to be: 1) no restriction C can be very large and needs hughe canelation D 2) C is close to y 3) C on thales circle (but don't mention thales)
C: (transcribed from chat) The problem I see here is that this is not quite true, since ortho is based on the unmasked components.

Changes (helper `reports/26-11-04_tpd_truth_paper_draft_diagram.typ`, draft `reports/26-11-04_tpd_truth_paper_draft.typ`):
- Figure 1, target panel: a dashed orange outline around the statement tokens' `down_proj` row, with the label "optional (recon, ortho), dashed: a loss on the decomposed matrices' outputs at the statement tokens"; the caption names it.
- New Figure 2 (`restriction_panels`, label `fig-restrictions`) in 1.2 *What the losses ask for*, referenced from the sentence "Recon and ortho each add a constraint…". Three panels of y, C, D at one token and layer: plain (C far larger than y, D cancelling), recon (C ≈ y, D ≈ 0), ortho (C on the dashed circle whose diameter is the segment 0→y, right-angle mark between C and D). No mention of Thales. Compiles; pages 1 and 4 checked by eye.

On Julian's concern (ortho uses the unmasked components). My reading [concluded, reasoning below]: if C in the figure means the full-strength sum of all components, as in the draft's *Notation* paragraph, all three panels are correct, and the mismatch is elsewhere.
- Plain: C + D = y holds by construction (D = Δx = y − C), and no term relates C to y. Correct.
- Recon: the term is on masked outputs, but with the same input ŷ − y = Σ_c (m_c − 1) C_c + (m_Δ − 1) D. It vanishes for all draws (m_Δ uniform in [0, 1], low-CI components' masks in [CI, 1] ≈ [0, 1]) only if D ≈ 0 and the low-CI components write ≈ 0. Then the full-strength C ≈ y as well, so the masked term constrains the unmasked C too (and even more: it also says which components carry y). Caveats: soft penalty; in the masked pass the input already carries upstream layers' changes, so the identity holds only approximately.
- Ortho: directly on the full-strength C. Correct.
- What the panels do not show is the split of C into the components active at the token (C_a) and the inactive ones (C_i). For ortho only the sum C_a + C_i lies on the circle; C_a alone can be larger, with C_i partly cancelling it. The verdict loss (inactive masks random) only requires C_i to be irrelevant to the verdict, not small, so nothing except the CI-scaled weight decay works against this [assumed: not checked in the decompositions]. Recon's term does rule it out. The caption of Figure 2 now says this.
- Check worth doing, if Julian wants it: in the ortho decompositions, measure |C_a|/|y| and ⟨C_a, y − C_a⟩/|y|² at the statement tokens (C_a = CI-thresholded active components), to see whether the active part itself is near the circle. Not run.

---

26-11-05, Claude Opus 5.5 — *statement-end merged tokens checked*; I have read the draft around the verdict prompt (section *Shared setting*) and paper_common.py's statement loading

C: (transcribed from chat) Can you add a space indicator in this string?

Done: the verdict prompt line now ends in `␣` (U+2423) instead of an invisible trailing space. DejaVu Sans Mono has the glyph; compiles.

C: (transcribed from chat) I need help with this. For completness the text says that the newline symbol merges with the dot. Sometimes also a closing quotation + dot + newline merge

Checked which token ends each statement when statement + verdict stimulus (`"\nThe amount of truth in this statement is "`) are tokenized as one string with the Qwen2.5-7B-Instruct tokenizer, over all v3 jsonl files in `data/tiu/` (the paper configs use v3) [verified: scratchpad script, 26-11-05, all splits]:
- In distribution: `'.\n'` everywhere except `sp_en_trans` and `neg_sp_en_trans`, where every statement (351 each) ends in a quoted word and the last token is `'.\n` (closing quote merged).
- OOD: `sp_en_trans_conj/_disj/_de` and `neg_sp_en_trans_de` likewise all end in `'.\n`. `facts`, `neg_facts`, `facts_conj`, `facts_disj` have a handful of other merges: `).\n` (3/3/6/2), `.'\n` (2/2/0/0), `."\n` (1/1/0/0); the rest `'.\n'`.
Draft sentence rewritten accordingly ("usually `'.\n'` … every statement of the two translation families ends in a quoted word, giving `"'.\n"`, and a few out-of-distribution `facts` statements end in …"). Lines that later say "up to and including `'.\n'`" (*Positions* paragraph) still name only the common token; fine as shorthand given the definition, but could say "the merged end token".

---

26-11-05, Claude Opus 5.5 — *evaluation categories motivated by the mediator criteria*; I have read the draft's evaluation section, this LOG from the methods diagram on, and the topic SUMMARY

C: (transcribed from chat) In a previous session we have developed these categories. I'd like to motivate them based on these criteria from "The Quest for the right mediator". So the idea is that we list these and say whihc ones apply to our situation and how some of them split into some subcategories

Julian pasted the criteria passage of Mueller et al., *The Quest for the Right Mediator* (arXiv 2408.01416): four criteria (sparsity, generality, selectivity, faithfulness; "a good causal explanation explains the most general effects from the fewest non-trivial causes", ideal mediator = Pareto optimum), plus one per goal: human understandability (explaining a behaviour), counterfactual faithfulness / IIA (verifying a hypothesis), downstream performance (localisation and editing, where faithfulness is "trivially satisfied if the model's behavior changes after the editing operation"). The paper is not in `papers/`; I worked from the pasted passage only.

New subsection 2.2 *Criteria for a good mediator* (label `criteria`) in `reports/26-11-04_tpd_truth_paper_draft.typ`, between *Why not the usual four categories* and *The categories* (now labelled `categories`). The intro sentence of 2.1 points to it. Content:
- The four criteria and the three goal-specific ones, with the survey's wording quoted.
- OUTDATED (26-11-05): the goal was reframed as a method question at Julian's objection; see entry *"which goal is ours" reframed as a method question*.
- *Which goal is ours*: explaining a behaviour (are the components the model's route to the verdict). Counterfactual faithfulness applies only weakly, since the hypothesis is coarse (components carry truth; M-twin is the one prediction fixed in advance). Understandability is secondary. Editing is not a goal, and the survey's "trivially satisfied" remark is why a flipped one-bit verdict is weak evidence (D).
- *Why the criteria split here* [concluded: argument stated in the draft]: the survey's mediators are parts of the model's own computation; tPD components are optimiser-made weight matrices with only components + Δ = W fixed. (1) The trivial mediator, which the survey excludes as "non-trivial", is components + Δ, which is perfectly faithful by construction, so every test must remove Δ or components. (2) Faithfulness splits into sufficiency in the hybrid (A), necessity in the original (B), completeness (C), with Wang et al. 2023 (IOI: faithfulness vs completeness) as precedent. (3) Sparsity (importance minimality) and selectivity (Pile loss) are partly trained, so they are not independent evidence, and trained selectivity works against completeness (parts also used on general text go to Δ).
- New Table 3 (`tab-criteria`): criterion | applies here? | splits into | category | measured by. Faithfulness → output sufficiency (A), internal-state sufficiency (A, not run), necessity (B), completeness (C). Selectivity → towards other text/judgements (B), within the task truth vs push to one answer (B, D: shift next to share). Generality → new statements / new statement types / new prompts and answer formats, a column, not a category. Sparsity → active set and the KL-recovered-vs-m curve as the sparsity–faithfulness trade-off; reported, not tested. Counterfactual faithfulness → D (interchange swaps not run). Understandability → E. Downstream performance → no.
- Closing paragraph: S0 (choice of mediator type and place, the survey's own subject) and F (correlational, which no causal criterion rewards, hence demoted) do not come from the criteria.
The experiments-by-category table moved from Table 3 to Table 4 (no literal "Table 3" references in the draft). Compiles; page 8 checked by eye after widening the category column.

Points for Julian [assumed: agent's view]:
- The motivation is post hoc: the categories came first (fresh agent), the criteria were mapped onto them afterwards. The draft text says "relates them to" rather than "derived from"; the paper should not claim the categories were derived from the survey.
- Sparsity and generality do not get categories. That is consistent with the survey's Pareto framing (they are axes along which every test is reported), but a reviewer who knows the survey may expect a sparsity result. The active-set curve is the natural figure for it.
- Citation year/venue to check (comment in the draft): I cited it as Mueller et al. 2024 from the arXiv date. The pasted passage itself cites a different "Mueller et al. 2025" for minimality, which is a separate paper.

---

26-11-05, Claude Opus 5.5 — *generality rows of Table 3 no longer name categories*

Julian (chat) was confused that Table 3 lists "A, B, D, F" as the categories for generality, since A is sufficiency. Cause: in the generality rows the category column meant "categories whose tests are repeated on wider data", while in every other row it means "the category that tests this criterion". I proposed a fix; Julian replied:

C: (transcribed from chat) yes

Done in `reports/26-11-04_tpd_truth_paper_draft.typ`, Table 3: the generality rows' category cells are now "–". The criterion cell says "yes; not a category but the range of data on which the tests of A, B, D, F are run". The "measured by" cells say "every test of A, B, D, F, run on the test split", "… repeated on the out-of-distribution sets; K2", "the tests of A, B, D repeated under the second prompt and the Yes/No prompts (G2)". Compiles; page 8 checked by eye.

Still open from my earlier chat answer, not acted on: give each category in the 2.3 table its criteria (the reverse mapping), add sparsity to A there, and say that seed stability (now filed under E / understandability) is a criterion the survey does not name [assumed: agent's proposal].

---

26-11-05, Claude Opus 5.5 — *"which goal is ours" reframed as a method question*

C: (transcribed from chat) I don't know if I agree with these goals. This project has been driven by the question: can we make a targeted decomposition of a high level mechanism work at least in parts.

My earlier text (entry *evaluation categories motivated by the mediator criteria*) put our goal under the survey's "explaining a behaviour". OUTDATED (26-11-05): replaced by the method framing below, at Julian's objection. I agreed with the objection: the survey's three goals use a mediator to learn about a model, whereas this project uses one model and one behaviour to test a method. I proposed a replacement paragraph in chat; Julian replied:

C: (transcribed from chat) ok

Done in `reports/26-11-04_tpd_truth_paper_draft.typ`, 2.2 *Which goal is ours*: none of the survey's goals fits exactly. The question is whether a targeted decomposition can recover a high-level mechanism (the model's judgement of truth across many facts and sentence forms) at least in part. The four general criteria define "recover"; "in part" = some criteria met and others failed, or met on some families only. The goal-specific criteria enter as tests: counterfactual faithfulness (weak, coarse hypothesis, M-twin), understandability (what a decomposition promises, no reliable measure, exploratory E), downstream performance not a goal (edits as evidence; "trivially satisfied" caveat → D). Table 3's understandability cell changed from "secondary" to "exploratory: no reliable measure". Compiles.

Open: "high-level" should be defined in the introduction. A contrast with the mechanisms decomposed in the original tPD paper (Vigouroux and Sharkey) would need a check of that paper first [assumed].

---

26-11-05, Claude Opus 5.5 — *section 2 restructured: criteria first, RepE subsection removed*

C: (transcribed from chat) Can you rewrite section 2. I don't want "why not the usual four categories" anymore and can you start "Criteia for a good mediator" with the passage that we are using the goals as our eval

Done in `reports/26-11-04_tpd_truth_paper_draft.typ` (Julian's own edit of the faithfulness bullets into nested sub-bullets kept as is):
- Subsection *Why not the usual four categories* deleted. Section 2 now opens with 2.1 *Criteria for a good mediator*.
- 2.1 opens with the study's question (can a targeted decomposition recover a high-level mechanism at least in part) and the statement that the decompositions are evaluated by the survey's criteria for a good causal mediator, "recover" meaning meeting them. Then the four criteria, "in part" = some met, some failed, or on some families only, and a forward pointer: letters (S0, A–F) refer to the categories in 2.2. The goal-specific criteria follow ("none of these goals is ours exactly … enter as tests, not as goals") with the three bullets unchanged. The separate *Which goal is ours* lead-in is gone; its content is in these two paragraphs.
- Content of the deleted subsection kept where still needed: the hybrid model (removing Δ or components gives a hybrid next to the original; a result in it need not hold in the original) is now in *Why the criteria split here*. The Δ-as-hiding-place sentence is in the completeness sub-bullet. The one-bit-output point was already in the sufficiency sub-bullet.
- 2.2 *The categories* now opens with: six categories, each a question about a candidate mechanism, Table 3 gives the criterion each serves; proposed by an independent reviewer, adopted afterwards (`#assumed`); analyses fixed in the plan before test/OOD were read, except G0–G2.
- 2.4 *Which experiment falls in which category*: the frozen-plan column is now explained in place (RepE categories correlation, manipulation, termination, recovery, Zou et al. 2023, made for rating a signal, not a candidate mechanism in weight space).
Compiles; pages 6–7 checked by eye.

---

26-11-06, Claude Opus 5.5 — *capture section rewritten in the readout/edit framework*; context: this session's answers in `convos/julian/26-11-05_evaluation_categories_LOG.md` (steps 6 and 7, the answer *the general picture without tPD*); Geiger et al. definitions re-checked in `papers/Causal_Abstraction/ca_paper.md` (Defs 11, 16, 25, 28, 31, 33, 41, 44, 48; Prop. 40; Sec. 3.5)

C: (transcribed from chat) can you update @reports/26-11-04_tpd_truth_paper_draft.typ  using this new framework, so i can go through that and try to understand it there

Replaced the subsection `<capture>` (Julian's rework of *Intuition* and *Modeling with Causal Models*, and the earlier steps 3–5 plus open points) with a method-free derivation. New order:
1. *Intuition*: Julian's text, typos fixed; his sentence "A decomposition should show where this quantity is computed in the model's weights" is replaced by the split into *where T is represented* (activations) and *which weights compute T* (mechanisms).
2. *Causal models*: acyclic, graph via parents (his `//` suggestion); interventions incl. interventionals; "edit" = intervention on the network.
3. *Two views*: H (features, T one of them, logits from features only) and L (activations; weights inside the mechanisms).
4. *Linking the views: readouts and edits*: Geiger's exact transformation (Def. 25) in words: readout τ (partial), edits Ψ with ω, commuting condition, Ψ part of the claim, graded version, intervention-algebra caveat.
5. *The special case: a basis*: alignment / constructive abstraction; at most d features; locality gives teeth.
6. *Superposition*: h = F1 − F2 example; restrict Ψ or merge; consequence for T [assumed].
7. *What keeps the general link from being empty*: locality, specificity, simplicity as stated choices.
8. *Narrowing to the truth question*: marginalization to X → Z → T → (Y_1…Y_r) with several truth questions as readouts; B3.
9. *Which weights compute T*: refinement W = P + R (the old step 3's three properties, now for a general split), R1 carried / R2 not supplied elsewhere / R3 computed; tPD as one choice of split (C-writes, Δ-writes); the two too-strong readings (bundle Z, minimality) moved here.
10. *Measuring with the verdict*: the old step 5 in P/R terms, plus swaps and superposition.
11. *What this says about our analyses*: representation never tested alone; most analyses are removals, which the readout can't interpret (Geiger Sec. 3.5: ablation says whether, not how); swaps not run; R3 and the off-distribution issue not measured.
12. *Open points*: which question the criterion answers, which constraints, hard verdict vs s, the set S, pairs.

Julian's two `//` comments in the old text ("This is a critical point…", "This doesn't work as described here…") were removed; they are quoted and answered in the evaluation-categories LOG (entry *stuck on the basis…* and the answer *the general picture without tPD*). Compiles without errors [verified: `typst compile`, 26-11-06]. Everything in the new section is an agent proposal, not ratified.

---

26-11-06, Claude Opus 5.5 — *variable length and absent features added to "Two views of the model"*; context: my answer in `convos/julian/26-11-05_evaluation_categories_LOG.md` (entry *variable input length and input-dependent features in H and L*)

C: (transcribed from chat) can you add this in the .typ?

Added two paragraphs to subsection *Two views of the model* in `reports/26-11-04_tpd_truth_paper_draft.typ`: *Fixed variables, varying values* (after "$cal(H)$ is the claim"; X ranges over strings of any length, absent features take the value "absent", the one-statement domain restriction is part of the claim, the fixed F_1…F_k is a simplification the narrowed model doesn't need) and *Variable length* (after the L paragraph; one L_n per length with the same weights or padding, sites named relative to the statement end and answer position, swaps need a position map, pairs of equal length). New labels `<ca-narrow>` and `<ca-measure>` on the subsections *Narrowing to the truth question* and *Measuring with the verdict*. Compiles [verified: `typst compile`, 26-11-06]. Agent proposal, not ratified.

---

26-11-06, Claude Opus 5.5 — *variable-length paragraphs shortened to a footnote*

C: (transcribed from chat) can you change this into a short footnote. The details doen't matter much I think

Replaced the two paragraphs *Fixed variables, varying values* and *Variable length* (previous entry) by one footnote on "at every position" in the L paragraph: fixed variable set; in H, X ranges over strings of any length and inapplicable features are "absent"; in L, one model per length with the same weights, sites named relative to the statement's end and the answer position. Labels `<ca-narrow>`, `<ca-measure>` left in place (unused, harmless). Compiles [verified: `typst compile`, 26-11-06].

---

26-11-06, Claude Opus 5.5 — *proofread of Julian's unstaged edits to the capture section*; read `git diff` of `reports/26-11-04_tpd_truth_paper_draft.typ` (subsections Intuition, Two views, Linking the views; labels)

C: (transcribed from chat) Can you go through the currently unstaged changes of @reports/26-11-04_tpd_truth_paper_draft.typ  and proofread them / flag anything that might be wrong?

Findings (nothing edited; reported in chat):
- Substantive: the footnote in *Intuition* uses `^[…]` (Pandoc syntax); Typst prints it literally — needs `#footnote[…]` [concluded: `^` and `[` are not markup in Typst text mode]. Its content overlaps the H bullet "With one statement per input there is one T", and it ends in a note to self.
- Substantive: the readout τ is now "from L to H (a value for every feature)"; τ maps a *state* of L (all activations on one run) to a state of H, not model to model, and the parenthesis now dangles.
- Wording: missing punctuation after the bold headings of L and H; "nets computations", "paddet", "a models computation", "platonic idea what", "on the L", capital "The" inside the quoted claim, "hard but we give an attempt", "basic … basic north star".
- Minor: labels `<ca-narrow>`, `<ca-measure>` now unreferenced (harmless). The removed sentence about grouping does not break "Without the grouping…" later, because the basis subsection introduces the grouping first.

---

26-11-06, Claude Opus 5.5 — *proofreading fixes applied*

C: (transcribed from chat) ok to "#footnote[A model may compute several truth values, e.g. for several statements in one input; we restrict to inputs with one statement.]" but add the messy message, because this can already be seen as a mayor sipmlification from uncovering the models truth mechanism
C: (transcribed from chat) About 2. In Geiger I think the models were stateless, so a different state is always an intervention. But you are right that's better.
C: (transcribed from chat) You can apply the fixes

Applied in `reports/26-11-04_tpd_truth_paper_draft.typ`: the Intuition footnote as `#footnote[…]` with the added sentence "This is already a major simplification of the task of uncovering the model's truth mechanism."; the duplicate "With one statement per input there is one T" removed from the H bullet; τ now "from a state of L (a value for every activation, i.e. one run, possibly under an edit) to a state of H (a value for every feature)"; the wording fixes listed in the previous entry. Labels `<ca-narrow>`, `<ca-measure>` kept. Compiles [verified: `typst compile`, 26-11-06].

On Julian's point about states: in Geiger, τ maps *total settings* (a value for every variable) of the low-level model to total settings of the high-level one (Def. 25, `τ: Val_V → Val_V*`; notation table: "A map from total settings of M to total settings of M*") [verified: `papers/Causal_Abstraction/ca_paper.md`, 26-11-06]. There is no hidden state: a run on an input, edited or not, yields one total setting, which is what "state" means in the draft. So the two readings agree [concluded].

---

26-11-06, Claude Opus 5.5 — *ladder subsection added to the capture section*; content from `convos/julian/26-11-05_evaluation_categories_LOG.md` (answers *the missing middle* and *ladder derived from the definition*); read the draft's `<capture>` section as it is on disk today

C: (transcribed from chat) I'll ignore question 2. for now, because I want to read your writedown into the paper draft under 1 first

New subsection *From one definition to a ladder of tests* `<ca-ladder>` in `reports/26-11-04_tpd_truth_paper_draft.typ`, after *Measuring with the verdict* and before *What this says about our analyses*. Content: why the verdicts alone can't catch a re-implementation (uncancelled writes in the P-swap; the narrowed H has no output that reads them); three assumptions (one T computed from content Z; truth readouts read only T; other-quality readouts Q read only Z); the definition of "P captures T" as one "for every" sentence; measurement and the form of the conditional guarantee (confidence 1 − α, accuracy ≥ q on D; nothing outside D; not unique); the grid (interventions: swaps / settings / at P's input; readouts: Y_1 / other Y / Q; statements: in-distribution / OOD and model errors); the ladder table (rungs 0–6 with cell, H's prediction, alternative ruled out, which of our analyses touch it, named by glossary entries); why this order; A-checks with the site swap; where a re-implementation can show. One sentence added at the end of *Narrowing to the truth question* pointing to `@ca-ladder`. Julian's text elsewhere unchanged. No result numbers in the subsection (the data column names analyses only). Table columns set to fractional widths after the first render collapsed two `auto` columns. Compiles; pages 10–11 rendered and checked by eye [verified: `typst compile`, 26-11-06].

Not yet adjusted to the new subsection [assumed: left for Julian]: the *Open points* list (its first bullet, representation vs weights, is partly answered by the ladder) and the document summary at the top, which still lists "categories, controls, measures".

---

26-11-06, Claude Opus 5.5 — *evaluation section redone around the definition and the ladder*; read the whole section 2 as on disk, `spd/experiments/lm/honesty_targeted_decomposition/paper_common.py` (docstring, `state_metrics`, `data_aligned_basis`, all-off assertion), `scratch/26-10-29_paper_writing/tables/tables.md` (header notes on shift, bootstrap, bases)

C: (transcribed from chat) can you redo the whole of "How the decompositions are evaluated"?

Reading: restructure section 2 so that it runs from the definition to the tests, now that the ladder exists, instead of the ladder sitting as an appendix to the capture subsection. Julian's own text (*Intuition*, *Two views of the model*, *Linking the views*) is kept verbatim apart from one added sentence each in *Intuition* ("Our criterion answers the second…") and *Linking* ("Two mechanisms that agree on every edit in Ψ cannot be told apart…"). Backup of the previous version: session scratchpad `draft_before_eval_redo.typ` (not kept; the staged git version also has it).

New structure of section 2:
- Intro paragraph (roadmap).
- 2.1 *What it means…* `<capture>`: Intuition; Two views; Linking `<ca-link>`; *Bases and superposition* `<ca-superposition>` (the former subsections *The special case: a basis* and *Superposition* merged, text kept); *What keeps the general link from being empty* (+ pointer: specificity becomes testable via content readouts); **rewritten** *The truth question: content, truth value and two kinds of readout* `<ca-narrow>`: H = X → Z → T → (Y_1…Y_r) plus Z → (Q_1…Q_m), assumptions A1–A3 numbered here, why both kinds of readout, and *Which T* (B3 read under the judgement vs stored-fact versions of T, replacing "the verdict prompt may read a truth assessment that is not the model's knowledge"); *Which weights compute T* `<ca-weights>` (kept; one sentence added: the new variables are ours, the network only sees their sum); new *The definition* `<ca-def>` with the conditional guarantee and its three limits.
- 2.2 *From the definition to tests: a ladder* `<ca-ladder>`: the grid; the ladder table (rung, cell, prediction, alternative; the analyses column moved to a separate table); why this order (removals = whether, not how, Geiger Sec. 3.5); *Swaps in practice* `<ca-measure>` (the former *Measuring with the verdict*, plus: equal-verdict pairs can't test, site swap as reference also for rung 3, hard verdict vs s); *Checking the assumptions with the site swap* `<ca-acheck>`; *Re-implementation: why the verdicts alone are not enough* `<ca-reimpl>`; *Our analyses on the ladder* `<ca-analyses>` (table rung → analyses run → what they show → missing; replaces *What this says about our analyses*; plus analyses not on the ladder: K0–K3, Dir, valley check, active set, cancellation norms); *Open points* (updated: which Q_k, pairs, the set S, the 4a edit class and 4b mean set, which T; dropped: representation vs weights (answered in Intuition), the constraints question, hard verdict vs s (moved into the text)).
- 2.3 new *Evaluation setup* `<eval-setup>`: statement sets (with Julian's `// neeeds to go into evaluation setup` sentence on the model's weak families, moved here from the end of the file, subject "It" → "The model"); prompts; states (the edit formula y' = m_Δ D + Σ m_c U_c V_c^T x; original, components only, Δ alone, all off, removals, reversal/amplification) [verified: `paper_common.py` docstring; all-off writes exactly 0]; measures (KL recovered, accuracy, agreement, separation share, shift (descriptive, added after reading), flips, digit mass, AUC; per run, pooled and macro, 95% bootstrap with 1,000 resamples within family) [verified: `state_metrics`, `tables.md`]; controls (untrained = step-0 checkpoint; data-aligned basis = 96 top principal directions of the input, U = W V, remainder W(1 − P_V), no CI; random subsets/subspaces/components 5/3/10) [verified: `data_aligned_basis` docstring; counts from the glossary]; pre-registration (frozen plan; G0–G2 and the shift added after).
- 2.4 Glossary: descriptions unchanged; new column *rung*; B3's description drops its interpretation ("the verdict prompt reads a truth assessment…"), now discussed under *Which T*.
- Document summary at the top updated to the new section content.

Compiles; pages 6–14 rendered and checked by eye (tables laid out with fractional columns) [verified: `typst compile`, 26-11-06]. Everything new is an agent proposal, not ratified.


---

26-11-07, Claude Opus 5.5 — *section 2 restructured: question, definition, tests, status*; read INDEX, both SUMMARYs (paper draft, evaluation categories), the whole section 2 as on disk

C: (transcribed from chat) I think you can clenup 2. even more. It doesn't read well right now and feels confusing. You can rearrange the structure also you don't have to tie the tests of the ladder to the tests we already have. I'm pretty sure there are some tests that we neend to redo
C: (transcribed from chat) or rather add

Backup of the previous version: session scratchpad `draft_before_sec2_cleanup.typ` (not kept; the staged git version also has it).

What I found confusing in the old version [concluded]: the causal-abstraction machinery (intervention algebra, basis vs superposition, constraints) came before the reader knew what the model of the truth computation was; the small model H (X → Z → T → Y, Q) appeared only in the sixth subsection; the ladder cells were written in grid code ("(i) × Y_1 × in-distribution"); the failure modes were spread over the re-implementation subsection and the "rules out" column; and our existing analyses were mixed into the ladder ("Our analyses on the ladder"), so the tests looked like a re-labelling of what we had.

New structure of section 2:
- Intro: four steps; the tests follow from the definition alone, and only 2.6 meets the existing analyses.
- 2.1 *The question*: Intuition (Julian's text kept, the "two questions" list dropped, one closing sentence on needing only the part through T); *A small model of the truth computation* `<ca-narrow>` (H with a bullet per variable, A1–A3, why two kinds of readout, which T: now stated as different predictions for recall, reported separately, so no reading is chosen).
- 2.2 *When a part of the weights captures the truth value*: Two causal models (Julian's causal-model and L paragraphs kept; H now points to 2.1 instead of the generic feature list); Readouts and edits `<ca-link>` (τ, Ψ, ω and the equation kept; the "three consequences" list absorbs "two mechanisms agreeing on Ψ", the graded version and the old *What keeps the link from being empty*; the intervention-algebra detail dropped; *Bases and superposition* compressed into one paragraph with the F1/F2 example); Splitting the weights `<ca-weights>` (shortened; *sites* defined; R1–R3); The definition (two too-strong readings, conditional guarantee).
- 2.3 *From the definition to tests*: *What the tests must catch* `<ca-failures>` (six named failures: second route, re-implementation, lever, different route, passing on, everything; the old re-implementation subsection lives here); the grid; the ladder table (test, edit/readouts/statements, passes if, rules out); *The tests in detail* (a protocol paragraph per test).
- 2.4 *Validating the tests* (new grouping): assumption checks with the site version of every test; seeded defects (scaling dial, Exp 4a, P = W); controls; κ as internal diagnostic (renamed differences δ_P, δ_R to avoid a clash with tPD's Δ).
- 2.5 *Evaluation setup*: as before, plus a *Pairs and twin groups* paragraph, the swap measures, and a pre-registration sentence for the not-yet-run tests.
- 2.6 *Status: what has been run and what has to be added* (new; replaces "Our analyses on the ladder"): per test what exists, what it covers, what to add or redo; analyses not tied to a test; proposed order; open points.
- 2.7 Glossary: the rung column removed (2.6 holds the mapping); descriptions unchanged.

Test protocols that are new or changed against the old ladder [concluded unless tagged; none run]:
- Test 2 gets a graded measure, the swap fraction ν = (s_swap − s(x)) / (s(x') − s(x)), next to IIA, both relative to the site swap.
- Test 3: Q_k must be shared by the pair; measure = KL of Q_k under the P-swap from its clean value, against the same under the site swap. For same-subject twins, recall of the subject's fact is a shared Q (under the judgement reading of T). In-statement next-token predictions dropped as a test-3 readout: positions before the first differing token are untouched by any swap, and after it the twins' predictions differ anyway.
- Test 4a made concrete: one fixed vector (mean P-write difference false − true over training pairs, one scale factor chosen on validation), with random-set, control-basis and whole-output (ceiling, assumption check) versions.
- Test 4b changed: the old version (P-writes replaced by their mean over a twin-balanced set) would also remove the content the P-writes carry, so "Q_k unchanged" could not be predicted. Now: P-writes replaced by their average over the statement's *twin group* (same subject and form, both truth values), which removes what varies with truth inside the group and keeps what it shares; mirror on the R-writes as a second form of R2.
- Test 5 (c), pairs with model errors: the P-swap should carry the model's verdict, not the label.
- Test 6: still no protocol; stated what it would need (same-truth, different-content swaps at the first site's input plus a Z readout).
- Seeded defects as a validation step, run per test. I dropped a "bias split" defect I first wrote, because a fixed bias is not a part of W and so not a split.

Status table (2.6), what has to be added or redone: test 0 on the Pile with only the inactive components removed (SE-Pile is Δ alone); test 2 entirely (pair builder, P-/R-/site swap, IIA, ν, κ); test 3 redone as swaps; 4a with Q_k and controls; 4b twin-group averages; 5 redone as swaps under every truth question, OOD pairs, model-error pairs; 6 a protocol; the site versions of all tests (first: site swap under Yes/No); the seeded defects. Proposed order [assumed]: site versions first, then tests 2 + 3 + scaling dial (shared code), then 4b, 4a, 5.

Pair count (new fact): disjoint label-twin pairs in the test split with the same subject and the same Qwen token length: 476 of 1,794 statements; cities 115, neg_cities 115, larger_than 103, smaller_than 103, element_symb 18, neg_element_symb 18, animal_class 2, neg_animal_class 2, sp_en_trans 0, neg_sp_en_trans 0 [verified: script `pairs.py` in the session scratchpad over `data/tiu/*/v3/test.jsonl`, Qwen2.5-7B-Instruct tokenizer from the local HF cache, 26-11-06; pairs by label, before filtering by the model's verdicts]. In larger_than/smaller_than a twin is the same two numbers in the opposite order. OOD and model-error pairs not counted. sp_en_trans has 0 with the subject key; I did not check whether its `subject` field holds the Spanish word or something else.

Document summary at the top updated. Compiles; pages 10 and 13 (ladder, status table) rendered and checked by eye [verified: `typst compile`, 26-11-07]. Everything here is an agent proposal, not ratified.

---

26-11-10, Claude Opus 5.5 — *does Julian's 2×2 outline cover the tests?*; read INDEX, this SUMMARY, the uncommitted diff of the draft, section 2 and the glossary as on disk

C: (transcribed from chat) I'm rewriting the paper. I have thought about the experiments and for me it looks like they are described by the summary that I've selected

The selection is Julian's new text in the draft, section 2.1 *The tests*: two assertions (P captures the truth computation, R captures the rest), split into four: (PT) P influences T, (P¬O) P does not influence other non-truth mechanisms, (R¬T) R does not influence T, (RO) R influences all other features; three ways to measure influence (remove/average, swap, edit); readouts in several formats (training verdict, other 0/1 prompts, Yes/No, other language).

**Verdict [concluded]: mostly yes.** The four assertions × three interventions generate tests 1–4b and 5(a) of the ladder, and the existing analyses fall into the cells. It is a cleaner organising principle than the ladder. Seven things are missing or need a qualifier.

Where the tests and existing analyses sit (names from the glossary; "planned" = in the ladder, not run):

| | remove / average | swap | edit |
|---|---|---|---|
| PT | test 1: T0, T1, T2, T3 (removal); test 4b twin-group average of the P-writes (planned) | test 2, P-swap (planned) | test 4a fixed vector (planned); M-core, M-twin, M-wide (masks −1, 2) |
| P¬O | G2 next-token, G2 digit questions under Δ alone (removal version); SE-Pile (Δ alone on the Pile) | test 3 (planned) | test 4a, its Q_k half (planned) |
| R¬T | R (components only = R removed: the verdict should survive); test 4b mirror on the R-writes (planned) | test 2, R-swap (planned) | none: a new cell (fixed vector added to the R-writes should not flip) |
| RO | nothing run (components only on the Pile would be it) | nothing | nothing |

Gaps and qualifiers:
1. **The readout list has only truth readouts.** P¬O and RO need readouts of the *other* features (the Q_k: recall, digit questions about humour/emotion/surprise/word count, Pile next-token). As written, two of the four assertions have no readout.
2. **P¬O is too strong as stated** and conflicts with the current definition. The P-writes carry T together with the content it is computed from; a P-swap between twins ("… in China" / "… in Turkey") moves that content too. The current draft solves this by restricting test 3 to Q_k *shared by the pair*. The assertion needs that qualifier, e.g. "P does not influence features other than T and its inputs". Recall sits on the boundary (*Which T*).
3. **RO is weak and not required by the definition.** R = the other components plus Δ, which holds most of the matrix, so removing R breaks almost everything whatever the split; this only rules out failure 6 (P ≈ W). And a targeted decomposition does not need R to carry all non-truth content, since P may carry content too. I would drop it and report the size of P instead, or keep it only as "P is not the whole matrix".
4. **The statement axis is missing.** Readouts vary, but statements do not: OOD sets (5b) and pairs where the model errs (5c, K3) are absent. 5c is the only test that tells belief from label, so I would not lose it.
5. **Test 0 (faithfulness / honest CI) is not an influence claim.** P + R = W, and removing the inactive components changes nothing. It is a precondition and belongs before the grid.
6. **"Captures the computation" vs "influences".** The first assertion says computation, but all four sub-assertions test influence; failure 5 (passing on a T computed before layer 11) passes all of them. Either weaken the headline to "carries T" or keep test 6 as an explicitly open item. Test 6 has no protocol yet, so the first is more honest.
7. **Controls are a cross-cutting layer, not a cell:** random part of the same size, the site version (assumption check), untrained and data-aligned bases, seeded defects. The text should say that every cell is read against them; the table's first row already does this for removal.

Smaller points:
- "Removing / averaging" lumps two edits of different status: a removal produces states no statement produces (it shows *whether* P is used, not *how*), while the twin-group average stays near natural states and removes only what varies with truth. Worth one sentence.
- "Other language" is ambiguous: a truth question asked in another language (a readout; the model's in-format answering would need a G0-style check first) or statements in another language (the statement axis). Open question for Julian.
- Analyses outside the grid stay outside: K0–K2 and Dir (decodability), cancellation norms, valley check, active set. These describe the decomposition and do not test an assertion; the draft already says so.
- The table under the selection still mixes the new first row with the old numbering and "rules out" column; a natural rewrite is rows = the four assertions, columns = the three interventions, as above.

Nothing in the draft changed in this entry.

---

26-11-10, Claude Opus 5.5 — *what Julian's detailed experiment list leaves out*; read INDEX, this SUMMARY, the previous entry, the whole draft as on disk (section 2 incl. ladder, validation, setup, status, glossary)

C: (transcribed from chat) I'm describing the expermintes that we want to report myself. Which ones have I forgotten?

The list in question is Julian's new text in the draft after "Now we describe what we measure in more detail" (influence of P on the truth verdict: only P, P deactivated, P-patching with IIA and ν, hand edits of P; influence of P on other verdicts: only P, no P, median P; influence of P on next-token prediction: Pile, statement; influence of R on the verdict: unfinished). Compared against his own outline just above it (four assertions × remove/average, swap, edit × readout formats), the ladder (tests 0–6), the validation subsection and the glossary [concluded, all items below are agent proposals]:

**Cells of the outline the list leaves empty**
1. *R on the verdict* (unfinished): R-swap (verdict should stay with x; ideally 0 relative to the site swap; the other half of test 2); twin-group average of the R-writes (separation should survive; 4b mirror); a fixed vector added to the R-writes (should not flip; new). R removal = "only P", already listed under P; say so to avoid counting it twice.
2. *Averaging on the verdict:* the twin-group average of the P-writes (test 4b). Averaging is one of the three named ways but appears only under "other verdicts". It is the least off-distribution removal.
3. *Swaps read by other verdicts* (test 3): P-swap, read by Q_k shared by the pair, against the site swap. This is the only behavioural test that catches a re-implementation, the failure recon and ortho were built against. Without it nothing in the list separates plain tPD from ortho on that point.
4. *Non-hand edits:* the fixed mean-difference vector (4a) is automatic; M-core, M-twin, M-wide (reversal at mask −1, amplification at 2) already ran and are edits. If edits are "by hand" only, these results drop out of the paper.
5. *Other truth formats:* the readout list names other 0/1 prompts, Yes/No, other language, but no experiment in the list uses them. Existing: R and T2 under the second prompt, G2 Yes/No transfer, T2 Yes/No.

**Axes missing entirely**
6. *Statements:* OOD sets (R and T2 ran on them) and statements the model gets wrong (K3; test 5c: a swap should carry the model's verdict, not the label). The model-error check is the only one that tells belief from label.
   OUTDATED (26-11-10): the belief-vs-label argument is moot. The decompositions are trained on the model's own verdict distribution, never on labels, so a split cannot track the label (Julian's correction, LOG entry *tests section cleaned up around Julian's list*). OOD statements stay as an axis.
7. *Controls:* random component sets of the same size (T2, M-core), untrained decomposition, data-aligned basis. The site comparison (the sentence below the list) gives a ceiling but not the "any part this size would do it" baseline.
8. *Size of P* reported with every test (P = W passes everything).
9. *Validation of the tests:* the site version is there; seeded defects are not (scaling dial k, Exp 4a, P = W).
10. *Sanity (test 0):* P + R = W numerically; inactive components removed changes nothing, on statements (R-inactive, ran) and on the Pile (not run). Plus which P each experiment uses: all components at CI, or the active set.
11. *Internal diagnostics:* cancellation norms (ran) and κ (planned). The direct check of what ortho optimises, and the one place where a re-implementation shows without a readout.
12. *Computes (test 6):* still nothing tests "computation" rather than "carries"; flagged in the previous entry.

**Ran, not a test; report or drop explicitly:** K0–K3, Dir, Dir-remove, T3 (A8), active-set curve, valley check, seed comparison; on the original model B3 (steering; the "which T" argument) and G0 (format check, a precondition for the other-verdict readouts).

**Problems in the listed items**
- *Verdict recovered = separation under P-ablation / separation under site ablation* has a near-zero denominator: mean-ablating MLP 11–17 leaves 0.05 of the separation (Table 2, layer scan) [verified: draft table re-read 26-11-10]. Suggest the damage ratio (1 − sep_P) / (1 − sep_site), the same form as ν; and the same ablation type (zero or mean) for P and the site, since the layer scan used mean and "all off" is zero.
- The IIA bullet breaks off mid-sentence ("We can also measure the separat").
- "Other verdicts" is ambiguous: other-quality digit questions (Q_k), or other truth formats. The bullet's expectation (P irrelevant) fits only the first.

Nothing in the draft changed in this entry.

---

26-11-10, Claude Opus 5.5 — *tests section cleaned up around Julian's list*; context: the previous entry, the whole draft as on disk

C: (transcribed from chat) OK. I'd like you to clean the mess a bit up. Here some comments first:
- first, I like my presentation of the tests so far, so try to keep it similar
- the test I have described are something like "behavioural tests"
- there is some symmetrie between R and P that we could use in presenting the tests to make it more systematic 
- things that need to be added:
   - R on verdict, yes the first tw
   - Averaging P on the verdict yes
   - p-swaps on other verdicts, yes
   - fixed mean-difference, yes
- the presentation should make clear that a lot of the tests are kind of a cross-product: test x quality x verdict style x size of P x in/out of distribution (x maybe others I forgot)
- about 6. There is no way the decomposition reports the label. The training is unsupervised
- yes, the controls should be described in the tests as well
- the discussion about what the test cannot detect should come later
- Sanity is not important. P + R is real output by construction. ci-masked, or active-masked P might break, but this is more of a problem of the method than showing thet the decomposition is bad
- about 11. yes, we also need a paragraph about other inspection tests
- you can also decsrie test 6
- you are right my general description of how we compare effect vs base was too simplistic. you can fix that

Backup of the draft before this edit: session scratchpad `draft_before_tests_cleanup.typ` (not kept; the git index also has an earlier version).

**Section 2.1.3 *The tests* `<tests>`, rewritten** (from the heading to the small-model subsection). Julian's text kept where possible (assertions, three interventions, readout formats, his wording of the P-on-verdict items). Removed from it: the old half-converted ladder table with the "rules out" column, the duplicated old test paragraphs (2, 3, 4a, 4b, 5, 6) and the duplicated pair definition, Julian's drafting notes "It's not clear what capturing the computation means…", "*Functionality*…" and the "We cluster the test as following" fragment (their content is covered by the new text). New structure:
- Intro: the two assertions and their four parts, stated as mirror pairs (what P should do on truth, R should do on the rest); three interventions; truth readouts in four formats; other-quality readouts (digit questions, recall, next-token on statement and Pile); "behavioural tests".
- `<tests-behav>` bullet list in Julian's style, four groups: P on truth verdict (only P, P removed, twin-group average [added], P-swap with IIA and ν, fixed mean-difference shift [added], hand edits; M-core etc. named as the component edits already run); P on other qualities (removed / median / twin average; P-swap and fixed shift read by qualities the pair shares [added]); R on truth verdict (R removed = only P; twin average [added]; R-swap [added]); R on other qualities (only P; stated as weak, it mainly shows P ≠ W). Table 3 `<tab-tests>`: interventions × {P, R} × {truth, other}, expected outcomes, "–" for cells not run (no R edit, per "the first two").
- `<tests-compare>` (replaces Julian's one-line "compare with the whole site"): each measurement against the original, the site version (ceiling and readout check), random sets of the same size, and the control bases; effect fraction φ = (m − m_0)/(m_site − m_0); for removal φ = (1 − sep_P)/(1 − sep_site), with the reason the plain ratio fails (site removal leaves 0.05, Table 2); for swaps φ = ν_P/ν_site; for other qualities KL, compared directly when the site effect is ≈ 0; same ablation type for part and site.
- `<tests-grid>`: the cross-product axes (test, quality, verdict style, size of P, statements in/out of distribution, decomposition and seed, comparison runs) and the combinations that don't exist.
- `<tests-compute>` (test 6): projection protocol [assumed]: project the truth directions (Dir) out of the residual entering layer 11 at the statement tokens and repeat the twin average and P-swap; the site version = original model under the projection; limits (only the linear part of T is removed; later layers of P may pass on what earlier ones computed).
- `<tests-inspect>`: cancellation norms and κ; structure (active set, valley check, seed comparison); alignment and decodability (Dir, K1, K2; K0 as the reason probe accuracy is not evidence).

**Section 2.3 renamed *What the tests can and cannot detect*** (the later discussion Julian asked for): the six failure modes (model errors dropped from "different route"); new *Which test catches which* (failure → test); *Limits* (assumptions each test needs, incl. superposition for the fixed shift and "removal shows whether, not how"; routes around the sites; what no behavioural test sees). Removed: the grid, the ladder fragment, the per-test protocols (now in 2.1.3).

**Section 2.4 Validating:** site version and seeded defects rewritten with the new test names; P = W "passes every test on P; only its size and the tests of R on other qualities expose it". Controls and κ paragraphs removed (now in `<tests-compare>`, `<tests-inspect>`).

**Status (2.6):** the table now has one row per test of Table 3, plus does-P-compute, inspection, site versions, seeded defects. Test 0 (sanity) dropped per Julian. Test 5(c) (model-error pairs carry the model's verdict, not the label) dropped: Julian pointed out training never sees labels; K3 is kept as descriptive under "not tied to a test". Proposed order and open points updated ("another language" added; "the set S" now the size-of-P axis).

**Evaluation setup:** "OOD pairs and model-error pairs" → "OOD pairs"; φ added to the measures; swap measures point to `<tests-behav>`. Document summary at the top updated.

Not touched, flagged for Julian: the small-model subsection `<ca-narrow>` now lists only X and T as variables, but A1–A3 and the following paragraphs use Z, Y_j, Q_k; and the simplified picture X → C → T → Y uses C for the content, while C is already the write of all components in section 1 (the rest of section 2 uses Z).

Compiles without warnings; pages 8, 9 and 15 (Table 3, comparison and cross-product, status table) rendered and checked by eye [verified: `typst compile`, 26-11-10]. Everything new is an agent proposal, not ratified.

---

26-11-10, Claude Opus 5.5 — *pair vs twin vs twin group*; context: this session's edits of section 2, the test-split jsonl files

C: (transcribed from chat) I'm confused by this. What is the difference between a pair, a twin and a twin group?

(About the paragraph opening `<tests-behav>` in the draft.)

New fact: in the test split every twin group (statements with the same `subject` field within a family) has exactly one true and one false statement: animal_class 18 such groups + 1 singleton, cities 166, element_symb 28, larger_than 204, smaller_than 204, negated families the same; sp_en_trans and its negation almost no groups (1 matched group; the subject is the Spanish word, which the false statements mostly don't share) [verified: count over `data/tiu/{family}_{true,false}/v3/test.jsonl` by `subject`, 26-11-10]. So "twin group" = "the two twins", and the term adds nothing for this data.

Answer [concluded]: twins = the true and the false statement about the same subject in the same form (always two here). A pair (x, x') = twins that additionally have the same token length and different verdicts *of the model*, ordered (x receives, x' donates). Twins are defined by the labels, pairs by the model's verdicts; they differ only where the model errs on one twin or the tokenizations differ in length (cities: 166 twin groups, 115 equal-length). The twin-group average also needs aligned positions, i.e. equal token length, which the draft does not say. Proposed to Julian: drop "twin group", call the average the *twin average*, and state that it runs on equal-length twins. Draft not changed yet.

---

26-11-10, Claude Opus 5.5 — *twins only: pairs and twin groups dropped*; context: previous entry

C: (transcribed from chat) "The writes are compared"? Are they compared?

Answer: no, wrong word in my proposed text: swaps and averages replace the writes position by position, which needs matching positions; equal token length is the simplest way (our choice, not a necessity).

C: (transcribed from chat) We should only describe what twins are. Statement from the same sentence form / domain that disagree on their truth value, which usually means on their last token (let's ignore numbers for now).

Done in the draft:
- The paragraph opening `<tests-behav>` now only defines twins: two statements of the same sentence form and domain that disagree in their truth value, usually differing only in the last word (Lu'an example). I wrote "last word" instead of "last token": the statement's last token is `.\n` for both twins, and the object can span several tokens.
- "Pair" and "twin group" removed throughout: "averaged over the twin group" → "averaged over twins" / "twin average"; "on a pair" → "on twins x and x'"; IIA = share of swaps after which the verdict is that of x'; fixed shift from the false and true twin in the training split; Table 3, 2.3, 2.4, status table, open points, κ, the compute protocol likewise.
- The evaluation-setup paragraph *Pairs and twin groups* → *Twins*: definition, each subject has exactly two statements in the test split (except sp_en_trans), equal token length needed for swaps and twin averages, swaps only on twins with different model verdicts, the 476 count (now "pairs of twins").
- Not changed: the glossary's M-twin, where "twin" means a negated statement and its affirmative counterpart, a different sense.

Compiles [verified: `typst compile`, 26-11-10].

---

26-11-10, Claude Opus 5.5 — *results section skeleton with placeholders*; context: the whole draft as on disk (incl. Julian's own edits since the last entry: section 2.3 onwards now under `= Discussion`, `<tests-compute>` and `<ca-def>` removed), `convos/julian/26-10-29_paper_writing_SUMMARY.md`, the file list of `scratch/26-10-29_paper_writing/tables/`

C: (transcribed from chat) The current write down has now a decently good section describing all the tests. I'd like to include the results of the tests right after it. Some tests still need to be run. I'd like you to include the test at `//Results should go in here`. Before you provide all the data (at least for the tests for which we have data). I'd like you to fill the section first with placeholders for how you would imagine this part to look like.

Done: `= Results <results>` replaces the `// Results should go in here` line. No numbers anywhere; only the planned tables and figures. Placeholder conventions (helpers defined at the top of the section, to be removed when the numbers come in): red ⟨…⟩ = what goes in a text slot; dashed red box = placeholder figure with its axes described; in tables ● = data exist, ○ = to run, – = does not apply; a grey *Data status* box per subsection (Have / To run).

Structure [concluded: agent proposal, not ratified]:
- Intro: per-decomposition reporting (plain s1/s2, recon s1/s2, ortho s1–s3, untrained s1/s2, data-aligned), φ against the site version next to φ_rand. **Default P = the active set** [assumed: open choice for Julian]; the size dependence shown along T1's top-k; the data-aligned basis has no active set, so its P is the top-k by T1 at the median active-set size (also a proposal).
- 3.1 *Overview*: scorecard table (Table 3's cells × plain/recon/ortho/untrained/data-aligned; ✓/✗/~/n/a; filled last) and a headline "mirror plot" (φ on truth readouts vs φ on other qualities; P ideal at (1,0), R at (0,1); one panel per intervention).
- 3.2 *Can the tests judge a split at these layers?*: site versions table (intervention × readout) and seeded defects table (tests × dial k, Exp 4a, P = W) + dial figure. Placed before the P/R results because a test whose site version fails can't judge any split.
- 3.3 *P on the truth readouts*: removal table (only P, P removed, φ_remove, φ_rand) + size-of-P figure (T2 as φ); twin average / P-swap / fixed shift table + per-family ν figure; component edits (M-core, M-twin).
- 3.4 *R on the truth readouts*: R twin average, R-swap.
- 3.5 *P on the other qualities*: removal table (next-token prefix/recall/object+end, digit questions, recall question, Pile, relative to the site removal) and the median / twin average / swap / shift table.
- 3.6 *R on the other qualities*: only P under the other-quality readouts.
- 3.7 *Other verdict styles and OOD*: only-P recovery and φ_remove under training / second / Yes/No / OOD + a heat-map figure for all tests.
- 3.8 *Inside the decompositions*: norms, κ, alignment, K2 + cancellation/CI figure; seed comparison slot.

The ●/○ marks come from the SUMMARYs (paper-writing topic, the draft's own status table `<ca-status>`), not from opening each output file [assumed]; check each cell when filling in. Known caveats written into the captions: the existing other-quality numbers (G2, SE-Pile) remove *all* components, not the active set, and the Pile numbers are validation only; the existing only-P transfer numbers are R (all components at mask = CI), not the active set; the existing "fixed shift" site cells are B3 (shift along a truth direction), not the mean difference of the site writes.

Compiles; result pages 10–17 rendered and checked by eye [verified: `typst compile` + pdftoppm, 26-11-10].

Open for Julian: the default P (active set vs all components vs a fixed k); whether the overview/scorecard should lead or close the section; whether 3.2 (test validity) belongs in Results or in an appendix.

---

26-11-11, Claude Opus 5.5 — *can the site-version tests (3.2.1) be run individually?*; context: draft section 3.2.1 (`tab-res-site`), `paper_common.py` (edit hooks), `paper_side_effects.py` (g1), docstrings of the other `paper_*.py`, file lists of `~/spd_out/26-10-22_tpd_truth_paper/{phase_b,b1_full,b3_v2}` and `~/spd_out/26-10-10_bolder_experiments/layer_scan/`; I have not run anything

Julian asked (chat): "I want to do the tests for 3.2.1. Are the scripts set up that this can be done individually?"

Answer: no [verified: code read]. Per row of `tab-res-site`:
- *Zero removal (all off).* Exists as the `all_off` state in `paper_recovery.py` (verdict, second prompt) and `paper_side_effects.py g1` (Yes/No, digit questions, next-token KL by position class). It cannot be run alone: `g1` always also runs `full`, `delta_only`, `components_ci`, `reverse_top10`, needs a decomposition `--run` and its `--termination` JSON, and asserts the JSON matches the run. All-off itself does not depend on the basis, so the G2 outputs already hold these cells. Recall question: no in-repo readout.
- *Mean removal.* Only the verdict, from the 26-10-16 layer scan (`~/spd_out/26-10-10_bolder_experiments/layer_scan/layer_scan.py`) and B1 (`~/spd_out/26-10-22_tpd_truth_paper/phase_b/b1_attn_scan.py`), both outside the repo. `paper_common.edited_outputs` can only rescale basis writes (y' = d_on (y − C) + Σ mask_c (v_c·x) u_c), so a mean replacement needs a new hook (simple, like `projected_out`).
- *Twin average, swap.* Nothing exists: no twin builder for same-subject, equal-length, different-verdict twins (`paper_manipulation.twin_pairs` pairs negated with affirmative statements, a different sense), and no hook that caches one statement's site outputs and writes them into another's forward pass.
- *Fixed shift.* B3 (`phase_b/b3_belief_steering.py`, outside the repo) steers the residual stream along a truth direction; the draft's test shifts the `down_proj` outputs by the mean difference of the site writes. Not the same intervention; nothing implements the latter.
- *Recall readout.* B3's recall is greedy generation with string matching, cities and elements only. A recall readout for all families has to be designed.

Proposal [assumed: not discussed]: one new script, `paper_site_versions.py`, that needs only the model and the run config's data and positions (no decomposition), takes `--intervention {zero,mean,twin_average,swap,shift}` and `--readouts` (subset of verdict, second, yes_no, digit, next_token, recall), so each cell of the table can be run alone.  
C: Will you use a good architecture so that there are not two completely independent scripts for site ablations & component ablations?
Answer (26-11-11, Claude Opus 5.5): Yes. The site is the special case of a split with P = the whole output (P = W, which is also the seeded defect "everything" of 2.4), so one code path covers both. Design [assumed: proposal]:
  - *Split*: per module, a function that returns the P-part of y at the decomposed positions; R-part = y − P-part. Site: P-part = y. Component set S of a basis: Σ_{c∈S} (v_c·x) u_c (the existing `Basis`); P and R tests use the same split with the roles exchanged.
  - *Intervention* on the chosen part, one forward hook: zero, mean (a stored train mean of that part), twin average / swap (the part from the twin's forward pass, cached in a first pass), shift (part + α · a stored direction). The existing `edited_outputs` (mask rescaling) is the zero/scale case and stays as is.
  - *Readouts*: one module with verdict, second prompt, Yes/No, digit questions, next-token KL per position class, recall; each takes any edit. Moved out of `paper_side_effects.run_prompt`, not copied.
  - *One driver*: `--split site|components --part P|R --intervention ... --readouts ...`, so each cell of the site table and of the P/R tables is one call.
  The frozen phase D scripts (`paper_recovery/termination/...`) stay untouched, because their outputs are final; the new code is checked against them: its zero removal of the site must reproduce G2's stored all-off KLs and scores per statement, and its zero removal of a T2 top-k set the stored T2 scores.
It reuses `paper_common` (statements, batching, prompts, bootstrap) and `paper_side_effects` (Yes/No and digit prompts, position classes). The twin builder and the swap hook would be written once here and reused for the P-/R-swaps (the status section's proposed order puts the site swap first anyway). Open before coding: the recall question's form;  
C: we should have recall of subject and object
Answer (26-11-11, Claude Opus 5.5): Agreed. The data allow it: every statement has a `subject` field, and `paper_side_effects.object_char` already locates the object (the part where twins differ) in every family. Proposed readout [assumed]: after the statement, one question per family for the subject and one for the object it stated, e.g. cities "<statement>\nThe city in this statement is" / "…\nThe country in this statement is"; scored as the teacher-forced log-prob of the answer string (handles multi-token names such as "New South Memphis" without generation and string matching), reported as the change against the original and as accuracy within a fixed candidate set (the family's subjects / objects). Two things to settle: (1) *stated vs true object*: for a false statement they differ ("Lu'an … in Turkey"). I would score the stated one (in-context recall, a content quality the edit should keep) and add the true one (the model's knowledge, what B3 read) as a third readout;  
C: What do you mean exactly? We ask for the country and then chec if the readout gives (a) the country in the statement or (b) the cities country? Or are you talking about swaps here?
Answer (26-11-11, Claude Opus 5.5): Not about swaps. I meant two different questions after the statement, each scored against one answer: (a) *in-context recall*, "…\nThe country in this statement is" → scored on the country the statement names (Turkey for "Lu'an is in Turkey"); (b) *knowledge recall*, a question about the subject that doesn't refer to the statement, e.g. "…\nThe city of Lu'an is located in the country of" → scored on its real country (China), as B3 did by generation. For true statements the two answers coincide; for false ones they differ, and only the matching answer is scored under each question. Why (b) as well [assumed]: fact lookup is usually located in mid-layer MLPs at the subject tokens, i.e. inside our sites (MLPs 11–17 at the statement tokens), so the site removal may break knowledge recall even if it leaves in-context recall intact; that is a non-truth quality the removal damages and worth seeing. Subject recall is (c): "…\nThe city in this statement is" → Lu'an.  
C: I don't want (b). We have enough tests already
Answer (26-11-11, Claude Opus 5.5): OK, dropped: `paper_tests.py` has only subject and object recall (entry *site-version code written and smoke-tested* below).
only cities and the comparisons store the correct object, the other families get it from the true twin. (2) Under the swap the twins share the subject but not the object, so subject recall is a shared other quality, while the object travels with the bundle the swap carries (the twin caveat in 2.3). As for every new prompt, first a G0-style check that the original model answers these in format.
C: I would not do many readouts for the twin swaps. Just some selected view. So I wouldn't do recall for them
Answer (26-11-11, Claude Opus 5.5): Fine: no recall under the swaps (and twin averages). Proposed selected view [assumed]: the three truth readouts (training verdict, second prompt, Yes/No), since whether they move together is the A2 check, plus one other quality the twins share, the digit question with the cleanest format from G0 (e.g. word count), to see whether the site swap carries more than truth. Recall and next-token KL stay with the removals and the shift.  
C: ok
mean over which statements (train split, all statement tokens per position?);  
C: mean over train. This is about `removed, mean`, right? Mean would be easier but I'm unsure what effects come from that, e.g. the giant activations at the first tokens. At least for cities mean over position could work, but on the other hand sentences like this "The city of New South Memphis is in the United States" and "The city of Zibo is in China" have quite different token counts. I'm unsure
Answer (26-11-11, Claude Opus 5.5): Yes, `removed, mean`. A precedent exists: the 26-10-16 layer scan (whose 0.05 the draft cites in Table 2) took the train mean separately for three position groups, for exactly your worry: the statement's first token (attention sink, massive activations), its end token '.\n', and all tokens in between (`~/spd_out/26-10-10_bolder_experiments/layer_scan/layer_scan.py`, docstring). Its norm check (`mean_norms.out`, 1,000 train statements) shows the worry is not limited to position 0: at layer 11 the max |down_proj output| is ~320 against a median of 16 at the first token and 23 in the middle group, and 7.6% of the *middle* tokens exceed 10× the median, so the middle mean is pulled by a few massive-activation tokens [verified: read `mean_norms.out`, layer 11 rows; not checked which tokens these are]. Your token-count point means a mean per absolute position would mix roles across statements of different lengths. Proposal [assumed]: (a) default = the layer scan's three groups, so the cited 0.05 and the new site removal are the same intervention; (b) on val, one alternative that groups by role instead (first token, massive-activation tokens, subject, relation words, object, end; the object located by `object_char`), with the off-distribution damage read directly as the next-token KL at the prefix positions, where truth is not yet determined: the variant with less prefix damage at a similar verdict effect is the cleaner removal. If the two agree on the verdict, the choice doesn't matter for the paper and (a) stays.
C: yes, lLet's test it out first.
Answer (26-11-11, Claude Opus 5.5): OK: both groupings (the layer scan's first/middle/end, and the role groups) are computed on train and compared on val by verdict effect and prefix next-token KL; the default is fixed from that before test runs.
the shift's direction (mean true − mean false site writes on train, per layer and token position?) and its size.
C: maybe here we could test for the right parameters for the sites?
Answer (26-11-11, Claude Opus 5.5): Yes, a good use of the site version: it depends only on the model, so tuning on it tunes nothing towards a decomposition. Proposal [assumed]: direction per layer and position group (the groups of the mean, answer above) = mean site write of the false minus the true statements on train; on val, sweep the size α and fix the smallest α at which the site shift flips a target share of the verdicts (e.g. as many as the site swap, which makes shift and swap comparable), checking that the digit mass stays up. The P-shift then uses P's own mean difference at the same α; random sets and control bases likewise their own. α, the direction and the target share are fixed on val before test is read, as the draft's pre-registration paragraph requires. The same val sweep can settle the mean-removal grouping (answer above).

C: ok, so there is a script to be build and also some experiments to run. 
Answer (26-11-11, Claude Opus 5.5): Yes. Order [assumed: proposal]:
  1. *Code* (no GPU): shared module (split × intervention × readout) and one driver as in the first answer; twin builder (same subject, equal token length, different model verdicts); recall prompts per family (subject, stated object, knowledge); train statistics (group means, false − true mean differences per layer and group). CPU smoke test on a few statements if the 7B model fits in RAM, else the first GPU minutes.
  2. *GPU, val*: format check of the new recall questions on the original model (G0-style: probability mass on the answer, accuracy among candidates); mean grouping comparison; site swap; shift α sweep against the site swap's flip share.
  3. *Fix on val*: mean grouping, α, readouts per intervention; recorded here before step 4.
  4. *GPU, test*: the full site table; regression check of the new code against G2's stored all-off outputs (needs one decomposition checkpoint; G2 ran on an H100).
  Steps 2 and 4 need only the model, not the decompositions, except for the regression check.

---

26-11-11, Claude Opus 5.5 — *site-version code written and smoke-tested*; context: this entry's thread above (Julian's comments up to "I don't want (b)" and "ok"), `docs/CODING.md`, `paper_common.py`, `paper_side_effects.py`, the draft's *The tests* section, the layer scan's `mean_norms.out`; CPU only, no GPU yet

**Code** (new, uncommitted): `spd/experiments/lm/honesty_targeted_decomposition/paper_interventions.py` (splits, position groupings, train statistics, the edit hook) and `paper_tests.py` (statements and twins, readouts, the per-batch engine, summaries, CLI `run` and `check_g2`). One code path for sites and components, as answered under Julian's architecture comment: a split (site, P = whole output; or a component set, part P or R) × interventions × readouts, one call per cell group. Design details [concluded: implemented as proposed above]:
- *Edit*: y' = y − q + q' at the statement tokens of all seven sites, q = the edited part. Replacement values for twin average and swap come from clean forward passes (the twin's value at each site whatever happens upstream: an interchange intervention); the shift adds to the current q.
- *Twins*: true and false statement with the same record `subject` field in one family (for the comparisons that field is the number pair, so the two orders pair up), equal token layout, and, for the twin interventions, different training-prompt verdicts.
- *Mean and shift statistics*: on all train statements, under each readout's prompt (the statement's last token differs: '.\n' vs '.'); recall uses the training prompt's. Groupings `scan3` (layer scan) and `role` (massive / first / pre-object / object / end, "massive" = clean norm > 10× the site's train median). Shift direction = train mean of the edited part over false minus true statements that have a twin, per group; sign by the model's training-prompt verdict, as the draft defines it.
- *Readouts*: verdict (+ next-token KL per position class for non-twin interventions), second, yes_no:<key>, digit:<key>, recall:subject:<form>, recall:object:<form>. Recall is teacher-forced (summed log-prob of the answer as written, and exact = every answer token is the argmax). No recall under twin interventions (Julian). Knowledge recall dropped (Julian).
- Deviation from my architecture answer above: I said `paper_side_effects.run_prompt` would be *moved*. It stays, because G2's frozen outputs come from it; the new engine imports the side-effects helpers (prompts, digit readout, object locator, position classes) instead of copying them.

**Checks** [verified: CPU, 26-11-11]:
- `ruff` and `basedpyright` clean.
- Twin count: first 270 equal-length pairs on test against the draft's 476; cause: I had keyed twins by the parsed first word, which for comparisons is only the first number. Keyed by the record's `subject` field now: **476 pairs, per family exactly the draft's 115/115/103/103/18/18/2/2**; val 272, train 430 equal-length pairs (before the verdict filter).
- Edit algebra on random tensors: site, P and R × the five interventions give exactly the intended q' at the statement positions and leave all other positions unchanged; position groups correct on a hand-built example.
- Recall tokenization: one example per family and readout printed and read; statement tokens, question and answer split as intended.
- CPU smoke test on the real model (run `s-25d2cca7`'s config; 26 test statements = 4 cities twin pairs + one per dataset, train statistics from 44 train statements; outputs in the scratchpad, not a result): all paths run. Against G2's stored H100 outputs for the same statements, the site zero removal reproduces G2's all-off state within CPU-vs-GPU bf16 noise (mean |difference| of the verdict log-probs 0.03, max 0.25 nats; statement-prefix KL 2.24 vs 2.22 stored). Twin path on the 5 pairs: site swap IIA 1.0, ν 0.90 (verdict); twin average ν 0.42. Smoke-only observations, to be checked on val: mean:role disturbs the prefix next-token predictions less than mean:scan3 (prefix KL 0.55 vs 1.26; zero 2.24); shift:role at α = 4 flips every verdict (separation share −1.04) and leaves the prefix untouched (KL 0.002, as expected: twins share the prefix, so the direction is 0 there), so α = 4 is too large.
- `fire` passes a comma list as one string when its items contain ':'; the CLI now splits strings on commas (found by testing the parsing, not by the smoke test, which called the function directly).

**Recall phrasing problem** [verified: smoke, 26 statements]: under "The animal named in this statement is", the model often continues with a judgement (' not' 0.30, ' incorrect' 0.24, ' actually' 0.05 on "The aardvark is a fish.") instead of the name; exact subject recall 0.35. So this form partly reads truth, which an other-quality readout must not. A second form, "The <noun> mentioned in the statement above:", is in the code; the val format check compares both, and the form is fixed before test.

**GPU plan** (script `~/spd_out/26-11-04_paper_draft/site_versions/val.sh`, run from the instance; site only, `--run s-0c2fa0d7`, a final run with stored G2 outputs): smoke (2 per dataset) → recall format check on val (originals, both forms) → zero, mean:scan3, mean:role and shift α ∈ {0.25, 0.5, 1, 2, 4} for both groupings on val, read by verdict, second, yes_no:question, digit:words → site twin average and swap on val, same readouts → zero on test with every G2 readout, then `paper_tests.py check_g2` locally against `side_s-0c2fa0d7_decomposition_test.npz`. Then, from the val outputs, the choices to fix before test [assumed: proposal]: recall form, mean grouping, α.

---

26-11-11, Claude Opus 5.5 — *recall wording screen*; context: the previous entry; Julian in chat (transcribed below); screen run on the rented RTX 4090 (48 GB)

C: (transcribed from chat) for the recall wording we should try to find a question that gets the right subject / object with high probiblity. For example the animal question could be "The name of the animal mentioned in the statament above is:" or something like that.

C: (transcribed from chat) Can you give me an overview over the questions and their logits?

Screen [verified: one run, original model, 26-11-11; script `~/spd_out/26-11-04_paper_draft/site_versions/recall_screen.py`, output `~/spd_out/26-11-04_paper_draft/site_versions/vm/recall_screen/recall_screen.json`]: seven wordings, each scored teacher-forced on the answer span as written in the statement (leading space, except after the opening quote of `quote`). Shown as probabilities rather than logits: P(answer span) = exp(summed log-prob of its tokens).

Wordings ({noun} = city / element / animal / Spanish word / first number for the subject; country / symbol / animal class / English word / second number for the object):

- `named`: '\nThe {noun} named in this statement is'
- `colon`: '\nThe {noun} mentioned in the statement above:'
- `name_of`: '\nThe name of the {noun} mentioned in the statement above is:'
- `which_q`: '\nWhich {noun} is mentioned in the statement above? Answer:'
- `question`: '\nQuestion: Which {noun} does the statement above mention?\nAnswer:'
- `repeat`: '\nRepeat the {noun} from the statement above:'
- `quote`: '\nThe {noun} in the statement above is "'

Overall, mean over 190 val statements (10 per dataset) of P(answer span) / exact (every answer token the argmax):

| wording | subject P | subject exact | object P | object exact |
|---|---|---|---|---|
| `named` | 0.12 | 0.14 | 0.19 | 0.23 |
| `colon` | 0.20 | 0.26 | 0.41 | 0.46 |
| `name_of` | 0.53 | 0.57 | 0.55 | 0.66 |
| `which_q` | 0.59 | 0.62 | 0.48 | 0.55 |
| `question` | 0.11 | 0.06 | 0.07 | 0.05 |
| `repeat` | 0.25 | 0.29 | 0.12 | 0.14 |
| `quote` | 0.66 | 0.70 | 0.77 | 0.83 |

Per family, mean P(answer span), subject:

| family | `named` | `colon` | `name_of` | `which_q` | `question` | `repeat` | `quote` |
|---|---|---|---|---|---|---|---|
| animal_class | 0.01 | 0.05 | 0.80 | 0.15 | 0.04 | 0.00 | 0.54 |
| cities | 0.59 | 0.69 | 0.95 | 0.98 | 0.35 | 0.02 | 0.91 |
| element_symb | 0.10 | 0.28 | 0.55 | 0.63 | 0.10 | 0.51 | 0.44 |
| larger_than | 0.00 | 0.01 | 0.13 | 0.66 | 0.01 | 0.74 | 0.53 |
| neg_animal_class | 0.01 | 0.09 | 0.76 | 0.06 | 0.01 | 0.00 | 0.46 |
| neg_cities | 0.31 | 0.51 | 0.98 | 0.97 | 0.37 | 0.02 | 0.84 |
| neg_element_symb | 0.16 | 0.27 | 0.62 | 0.91 | 0.11 | 0.22 | 0.47 |
| neg_sp_en_trans | 0.00 | 0.02 | 0.21 | 0.09 | 0.00 | 0.07 | 0.99 |
| smaller_than | 0.00 | 0.01 | 0.19 | 0.92 | 0.04 | 0.82 | 0.36 |
| sp_en_trans | 0.00 | 0.03 | 0.19 | 0.31 | 0.00 | 0.01 | 1.00 |

Per family, mean P(answer span), object:

| family | `named` | `colon` | `name_of` | `which_q` | `question` | `repeat` | `quote` |
|---|---|---|---|---|---|---|---|
| animal_class | 0.09 | 0.07 | 0.47 | 0.41 | 0.02 | 0.00 | 0.63 |
| cities | 0.33 | 0.41 | 0.94 | 0.91 | 0.12 | 0.03 | 0.94 |
| element_symb | 0.08 | 0.71 | 0.68 | 0.88 | 0.29 | 0.18 | 0.89 |
| larger_than | 0.35 | 0.83 | 0.92 | 0.31 | 0.00 | 0.30 | 0.99 |
| neg_animal_class | 0.02 | 0.06 | 0.11 | 0.01 | 0.00 | 0.00 | 0.25 |
| neg_cities | 0.35 | 0.39 | 0.84 | 0.24 | 0.03 | 0.08 | 0.84 |
| neg_element_symb | 0.27 | 0.46 | 0.48 | 0.62 | 0.16 | 0.04 | 0.86 |
| neg_sp_en_trans | 0.00 | 0.00 | 0.04 | 0.51 | 0.00 | 0.16 | 0.68 |
| smaller_than | 0.37 | 0.95 | 0.92 | 0.32 | 0.00 | 0.29 | 0.99 |
| sp_en_trans | 0.01 | 0.03 | 0.00 | 0.47 | 0.00 | 0.01 | 0.48 |

Quote form, families below 0.6, the first example statement and the model's top first tokens:

- subject animal_class (0.54): 'The kangaroo is a crustacean.' answer 'kangaroo' | top: 'k' 0.61, 'the' 0.29, 'The' 0.07, 'a' 0.01
- subject element_symb (0.44): 'Molybdenum has the symbol Np.' answer 'Molybdenum' | top: 'm' 0.76, 'M' 0.13, 'N' 0.08, 'n' 0.01
- subject larger_than (0.53): 'Fifty-three is larger than fifty-four.' answer 'Fifty-three' | top: 'f' 0.59, 'F' 0.28, '5' 0.13, 'five' 0.00
- subject neg_animal_class (0.46): 'The lobster is not a crustacean.' answer 'lobster' | top: 'the' 0.48, 'lob' 0.26, 'not' 0.10, 'The' 0.07
- subject neg_element_symb (0.47): 'Francium does not have the symbol Fr.' answer 'Francium' | top: 'fr' 0.68, 'Franc' 0.22, 'Fr' 0.08, ' franc' 0.01
- subject smaller_than (0.36): 'Sixty-six is smaller than sixty-one.' answer 'Sixty-six' | top: 'six' 0.57, '6' 0.27, 'Six' 0.16, ' Six' 0.00
- object neg_animal_class (0.25): 'The lobster is not a crustacean.' answer 'crustacean' | top: 'lob' 0.45, 'cr' 0.45, 'L' 0.02, 'not' 0.02
- object sp_en_trans (0.48): "The Spanish word 'cerca' means 'dog'." answer 'dog' | top: 'cer' 0.66, 'near' 0.13, 'Spanish' 0.05, 'dog' 0.05

Reading [concluded: from the tables and the printed top tokens]:
- `quote` is best overall, and best or near-best in most families. Julian's `name_of` is second for the subject (strong for animals and cities), weaker for the Spanish families. The two forms that end in "Answer:" on a new line (`question`) or invite commentary (`named`) are the worst.
- P(answer) ≈ P(first token) for every wording (totals in the log), so the misses are decided at the first answer token.
- Most of `quote`'s remaining misses are surface form, not recall: the model starts the right word lowercased ('k'angaroo, 'm'olybdenum, 'f'ifty-three, 'fr'ancium) or with an article ('the' lobster). A single scored string undercounts these.
  OUTDATED (26-11-11): 'k'angaroo is not a case miss: the answer is lowercase "kangaroo" and 'k' (0.61) is its correct first token; that statement's miss is the article ('the' 0.29). The case misses are subjects capitalised only because they open the sentence (Molybdenum, Fifty-three, Francium). Found while answering Julian's question about the quote form (entry *quote form explained* below).
- Real confusions are rarer: the class question on "The lobster is not a crustacean." answers the lobster ('lob' 0.45 vs 'cr' 0.45); the English-word question on "The Spanish word 'cerca' means 'dog'." answers the Spanish word ('cer' 0.66) or the true translation ('near' 0.13), not the stated 'dog' (0.05).

Proposal [assumed: not yet discussed]: use `quote`, and score the summed probability of the answer's surface variants (as written, first letter lowercased, first letter capitalized; for the animal subject also with "the "), so a readout measures whether the model still recalls the word, not its spelling. Re-screen that once before fixing it; the val runs' recall format step used the two old forms (`named`, `colon`) and will be superseded.

---

26-11-11, Claude Opus 5.5 — *quote form explained*; context: the recall wording screen above

Julian asked in chat whether the quote form starts with a quote token. No [verified: tokenizer, 26-11-11]: the question `\nThe animal in the statement above is "` ends with the token ' "'; the scored answer follows it (kangaroo → 'k', 'ang', 'aroo'; Turkey → 'Turkey'), and the question's tokens are identical with and without the answer appended, so the quote never merges into the answer. Why it works best [assumed: reading of the screen]: inside quotes the model is expected to copy the literal word, which leaves little room for ' not', ' actually' or an article. Correction to the screen entry marked OUTDATED there (kangaroo).

---

26-11-11, Claude Opus 5.5 — *quote wording adopted, spelling variants, G2 reproduced*; context: the entries above; runs on the rented RTX 4090; outputs copied to `~/spd_out/26-11-04_paper_draft/site_versions/vm/`

C: (transcribed from chat) Yes, the quote form looks strong so we take that. Would you just check `{noun} in lowercase({answer})` or what is the plan?

**Recall readout now** [decided: quote form, Julian above; the scoring is my proposal, implemented, not yet ratified]: `recall:subject` / `recall:object` use `\nThe {noun} in the statement above is "`. Scored on a fixed rule set of spellings (`paper_tests.recall_variants`), not one string: as written; first letter's case flipped; for the animal subject also both with "the ". Their probabilities are summed per statement (disjoint strings; the code asserts the sum ≤ 1), and *recalled* = sum > 0.5. Julian asked how the set of spellings is chosen: by this rule, derived from the misses in the screen's top tokens; it does not cover other forms (digits for number words, abbreviations).

**Re-screen through `paper_tests.py`** [verified: one run, val, 986 statements, original model, `tests_site_P_val_s-0c2fa0d7_original_recall-subject+recall-object.json`]:
- Subject: mean summed probability 0.867, recalled 0.987. Mass by spelling: as written 0.642, case flipped 0.217, "the …" 0.009 (so the case rule matters, the article rule hardly).
- Object: 0.901, recalled 0.939; case flipped adds 0.002.
- Weak object families (mean summed probability): neg_animal_class 0.25, sp_en_trans 0.55, animal_class 0.64, neg_sp_en_trans 0.66; all other families ≥ 0.81. These are confusions, not spelling (screen entry: the animal named instead of its class; the Spanish word or its true translation instead of the stated English word). For the readout this means: in those families the object question measures less; reported per family.

**G2 reproduction** [verified: test, 1,794 statements, `paper_tests.py check_g2` against `side_s-0c2fa0d7_decomposition_test.npz`]: the new code's site zero removal reproduces G2's all-off state. All 24 compared means (originals and zero removal; verdict, both Yes/No prompts, four digit questions, next-token KL per position class) agree within 1.1% relative (largest: the emotion digit KL); e.g. verdict KL 0.650 vs 0.647, prefix next-token KL 2.281 vs 2.284. Per statement the values scatter (mean |difference| 0.03–0.10 nats) as much as the unedited originals do between the two GPUs (0.046 for the verdict log-probs; G2 ran on an H100, this on an RTX 4090), and the all-off verdict sign agrees on 97.7% of statements. So `check_g2` now tests the means (relative tolerance) and prints the scatter; the first version asserted per-statement maxima, which cannot pass across GPU types.

**Val runs done** (smoke, recall format with the old forms [superseded by the re-screen above], removal + shift sweep, site twin average / swap): outputs in `vm/val/`, not analysed yet. Next: a table script over them (removal with both mean groupings incl. the prefix next-token KL; shift sweep vs the site swap's flip share), then the choices to fix before test.

---

26-11-14, Claude Opus 5.5 — *val analysis: mean grouping and shift size*; context: INDEX, this SUMMARY, the entries from *can the site-version tests (3.2.1) be run individually?* on, `val.sh`, the val JSON/NPZ outputs in `~/spd_out/26-11-04_paper_draft/site_versions/vm/val/`, the summary code in `paper_tests.py` (`truth_summary`, `summarize`)

Julian asked (chat) whether there is something to decide before the test runs or whether only a VM is needed. Answer: two choices to fix first (mean grouping, shift grouping and α); everything else is ready. Table script: `/tmp/claude-1000/-home-jx-projects-targeted-parameter-decomposition/26fb2e24-5d5d-4f4e-82f2-2e3ef4e398c2/scratchpad/val_tables.py` (scratchpad, not kept).

**Val, site version, 986 statements, original model unless stated** [verified: read from the val outputs, one run, 26-11-11 on the RTX 4090]. sep = separation share (mean score of true minus false statements, relative to the original; 1 = untouched, 0 = no separation, < 0 = inverted); flips = share of statements whose training-prompt verdict changes; KL = KL of the answer distribution against the original; prefix KL = mean next-token KL at the statement positions before the object (where truth is not yet determined: a clean removal should leave it near 0).

| state | verdict sep | verdict flips | verdict KL | second sep | Yes/No sep | digit KL (word count) | prefix KL | object/end KL |
|---|---|---|---|---|---|---|---|---|
| zero | 0.069 | 0.47 | 0.67 | 0.146 | 0.135 | 0.266 | 2.25 | 2.47 |
| mean:scan3 | 0.042 | 0.48 | 0.60 | 0.064 | 0.060 | 0.102 | 1.01 | 1.31 |
| mean:role | 0.051 | 0.47 | 0.83 | 0.029 | 0.024 | 0.094 | 0.58 | 1.14 |
| shift:scan3:1 | 0.218 | 0.38 | 0.99 | 0.279 | 0.137 | 0.017 | 0.005 | 0.23 |
| shift:scan3:2 | −0.569 | 0.74 | 3.13 | −0.205 | −0.405 | 0.057 | 0.018 | 0.76 |
| shift:scan3:4 | −1.060 | 0.93 | 4.66 | −0.592 | −0.951 | 0.112 | 0.061 | 1.78 |
| shift:role:0.5 | 0.180 | 0.40 | 1.13 | 0.249 | 0.178 | 0.010 | 0.001 | 0.21 |
| shift:role:1 | −0.642 | 0.78 | 3.45 | −0.429 | −0.392 | 0.035 | 0.001 | 0.65 |
| shift:role:2 | −1.073 | 0.98 | 4.59 | −1.026 | −1.278 | 0.079 | 0.001 | 1.30 |

Twin interventions (506 rows: equal-length twins whose verdicts differ): swap sep −0.889 (verdict), −0.909 (second), −0.915 (Yes/No); verdict flips 0.955, IIA 0.955, ν 0.944; digit KL 0.069. Twin average: verdict sep 0.094, IIA 0.443, ν 0.453 (≈ 0.5 expected for a halfway point); digit KL 0.020. The shift's flip shares on twin rows equal those on all rows within 0.03 (checked from the NPZ), so the comparison with the swap is not a subset artefact.

Bootstrap 95% intervals (within family) of the verdict separation: zero [0.053, 0.084], mean:scan3 [0.028, 0.055], mean:role [0.036, 0.067].

**Reading.**
- *Mean grouping* [concluded]. The two groupings remove the truth separation equally (0.042 vs 0.051, intervals overlap); scan3 reproduces Table 2's 0.05 from the layer scan (same intervention, val instead of the scan's split). They differ in side effects: role halves the prefix damage (0.58 vs 1.01), but pushes the verdict to "false" (true verdicts flipped 0.87, false ones 0.05; scan3 0.63 / 0.34; zero pushes to "true", 0.23 / 0.72) and has the larger answer KL on the second and Yes/No prompts (2.63 vs 1.63, 2.81 vs 2.24). Neither is clean on every measure. The rule agreed on 26-11-11 (answer under Julian's "yes, lLet's test it out first"): if both agree on the verdict, the choice doesn't matter for the paper and scan3 stays. By that rule: **scan3**.
  OUTDATED (26-11-14): Julian chose role for the mean removal as well, with massive at 5× instead of 10×; entry *role for mean and shift, massive at 5x* below.
- *Shift grouping* [concluded]. role is better on every count: it reaches a given flip share at half the α of scan3, leaves the prefix untouched (prefix KL ≤ 0.001 at every α; twins share the prefix, so role's direction is 0 there, while scan3's middle group averages prefix and object tokens and smears the direction onto the prefix), and moves the three truth readouts more evenly (role:2: sep −1.07 / −1.03 / −1.28 vs scan3:4: −1.06 / −0.59 / −0.95). Using role for the shift and scan3 for the mean is not inconsistent: the mean removal keeps the layer scan's intervention, the shift needs a direction that is 0 where twins agree.
- *Shift size* [concluded, choice open]. The 26-11-11 proposal was "the smallest α that flips as many verdicts as the site swap" (0.955): on the grid that is role α = 2 (0.98, sep −1.07, slightly past the swap's −0.89). An alternative needs no tuning: α = 1 is by construction the *average* twin displacement (direction = train mean of false minus true site writes over statements with a twin), i.e. a swap with each statement's own twin difference replaced by the mean one. It flips 0.78 against the swap's 0.955; the gap is what the statement-specific part of the twin difference adds. α = 1 also keeps the site effect away from the ceiling: at α = 2 nearly every verdict flips, so a P carrying, say, 70% of the site's difference would flip almost as many, and the flip share would hardly separate P from the site (the continuous ν and sep still would). I recommend **role, α = 1** [assumed: recommendation, for Julian to decide].
- *Side effects of the shift on a non-truth quality*: digit KL (word count) at role:1 0.035, below the swap's 0.069 and far below the zero removal's 0.266. So the site shift is a fairly selective truth edit at the site level.

**Not covered by the val runs**: recall under interventions (the val recall runs scored only the originals; the quote form came after `val.sh` was written). Recall under zero / mean / shift is first measured on test. No choice depends on it.

**What the test run will be** [assumed: proposal], one VM session, `--run s-0c2fa0d7 --target site --part P --split test`: (1) zero, mean:scan3, shift:role:<α> with verdict, second, yes_no:question, yes_no:same_line, the four digit questions, recall:subject, recall:object (next-token KL per position class comes with the verdict readout); (2) twin_average, swap with verdict, second, yes_no:question, digit:words (Julian's selected view). mean:role could go along as a side row (cheap) if Julian wants the role-vs-scan3 contrast in the paper. Zero on test already exists from the G2 reproduction, without recall.

---

26-11-14, Claude Opus 5.5 — *scan3 vs role on concrete statements*; context: the previous entry, `paper_interventions.group_ids`, a CPU forward pass of the original model on six val statements (script `/tmp/claude-1000/-home-jx-projects-targeted-parameter-decomposition/26fb2e24-5d5d-4f4e-82f2-2e3ef4e398c2/scratchpad/group_example.py`, scratchpad, not kept), the train group counts stored in the val NPZ

Julian asked (chat) for a concrete example of how scan3 and role group the tokens.

Statement tokens = the statement including its end token '.\n' (training prompt; the statement comes first, no BOS). Example, 'The city of Nanchong is in Nigeria.' [verified: CPU run, 26-11-14]:

| pos | token | scan3 | role, layer 11 | role, layers 12–17 | norm / train median, layers 11, 12, 13 |
|---|---|---|---|---|---|
| 0 | 'The' | first | first | first | 0.7, 1.0, 0.7 |
| 1 | ' city' | middle | **massive** | pre_object | 14.2, 8.0, 3.4 |
| 2–7 | ' of', ' N', 'anch', 'ong', ' is', ' in' | middle | pre_object | pre_object | 0.8–1.2 |
| 8 | ' Nigeria' | middle | **object** | object | 1.2, 1.0, 1.0 |
| 9 | '.\n' | end | end | end | 0.9, 0.8, 1.2 |

In 'Fifty-one is larger than fifty-seven.' the massive token is position 0 ('F', 14.1 at layer 11); the object is ' fifty', '-seven'. In 'The lobster is not a crustacean.' it is position 1 again (' lobster', 14.3).

Observations [verified: the six statements, and the train counts below]:
- The two groupings differ in two places only: role splits scan3's *middle* into pre_object and object, and at layer 11 it moves one token per statement into *massive*.
- The massive token is a positional effect, not a word: its layer-11 norm is the same (14.1–14.3 × median) whether the token is ' city', ' lobster', ' arm' or 'F'. It sits at position 1 after a leading 'The', otherwise at position 0. Train counts at layer 11: 1,962 massive positions for 1,962 statements, 724 of them at position 0 (first: 1,238 = 1,962 − 724) and 1,238 at position 1 (pre_object 11,978 vs 13,216 at the other layers). At layers 12–17 no position passes the 10× threshold (massive count 0), although the same token is still elevated at layer 12 (8.0×) and 13 (3.4×). So the role grouping's massive group acts only at layer 11; this matches the layer scan's norm check (40% of first tokens, 7.6% of middle tokens at layer 11), now located: it is this one token.
- Consequences for the two uses: in a scan3 mean, layer 11's middle mean averages one ~14× token per statement into ~8 ordinary ones, so every middle token gets a part of that massive value, and the massive token itself gets an ordinary-sized middle mean. That plausibly explains scan3's larger prefix damage (1.01 vs 0.58) [assumed: not isolated]. For the shift, the split of middle into pre_object and object is what matters: twins share all pre-object tokens, so role's pre_object difference is ≈ 0 and the shift lands on object and end only; scan3's middle difference is the object's difference diluted over all middle tokens and added to the prefix too.

---

26-11-14, Claude Opus 5.5 — *role for mean and shift, massive at 5x*; context: the two entries above; `paper_interventions.py`; the draft's site table (`tab-res-site`)

C: (transcribed from chat) Since role is clearly better in shift and we have to use it anyway, I want to use role for mean-ablation and shift. Also massive can be 5x median.

Decided [decided: Julian, above]: the role grouping for the mean removal and the shift; "massive" = clean site-output norm > 5 × the site's train median (was 10). This overrides the 26-11-11 rule "same verdict effect → keep scan3" (entry *val analysis: mean grouping and shift size*). The consequence that rule protected: the paper's mean removal is no longer the layer scan's intervention, so Table 2's 0.05 and the site table's mean row come from different groupings (on val they agree on the verdict: scan3 0.042, role 0.051).

Changes:
- `paper_interventions.py`: `MASSIVE_FACTOR = 5.0`, with the reason in the comment; module docstring records role as the paper default (scan3 stays selectable). `ruff` clean.
- Draft, `tab-res-site` caption: it said the mean removal *is* the layer scan; it now describes the role groups and says the layer scan's grouping is coarser, with a `#ph` for the per-layer count of massive tokens. Compiles.

Consequence for the runs [concluded]: the val numbers for mean:role and shift:role (entry *val analysis*) used factor 10, so they are stale. At 5× the one massive token per statement should also count at layer 12 (8.0× there on six val statements), maybe more tokens elsewhere (on those six, everything else stayed ≤ 3.1×) [assumed: from six statements, not the train set]. So the VM session starts with a val re-run of mean:role and the shift:role sweep at 5× (minutes), which also prints the new group counts, and only then the test runs. Still open: the shift size α (my recommendation α = 1, the average twin displacement, no tuning; alternative α ≈ 2, matching the swap's flip share).

C: Alright, let's take your recommendation. I'm currently renting a machine.
Answer (26-11-14, Claude Opus 5.5): Fixed: shift size α = 1 (the average twin displacement, no tuning) [decided: Julian's comment above]. With that every val choice is made (role grouping, massive at 5×, α = 1). Prepared for the machine:
  - Run script `~/spd_out/26-11-04_paper_draft/site_versions/test.sh` (same pattern as `val.sh`: copy it to `/root/spd_out/site_versions/` after `spd-vast --mode sync`, run with `bash`, progress in `progress`). Steps: (0) smoke, 2 statements per dataset, zero / mean:role / shift:role:1 with every readout incl. recall (recall under an intervention has not yet run on a GPU), stops on failure; (A) val re-run of mean:role and shift:role at α 0.5 / 1 / 2 at 5×, same readouts as the 26-11-11 val run, into `val5x/`; (B1) test, zero / mean:role / mean:scan3 / shift:role:1 with all readouts (verdict, second, both Yes/No, four digit questions, subject and object recall); (B2) test, twin average and swap with the selected view (verdict, second, Yes/No question, digit words). The 26-11-11 val session took 7 minutes on the RTX 4090; this one is of similar size plus recall [assumed: not timed].
  - mean:scan3 runs along on test as one side row, my addition: it does not depend on the massive factor, costs one more intervention, and lets the paper say whether the role mean and the layer scan's grouping (behind Table 2's 0.05) also agree on test, as they did on val. Drop it from B1 if you'd rather not.
  - Code: the outputs now record `massive_factor` (JSON field and NPZ array), because the file names don't show it and the 26-11-11 role outputs (10×) would otherwise be indistinguishable from the new ones (`paper_tests.py`; ruff and basedpyright clean). The group counts at 5× are printed in each log under "train statistics" (`describe_stats`).
  - Sync: HEAD is on `origin/main`, so `spd-vast` can clone it; the uncommitted code changes go up with the working tree.
  After the run: copy `/root/spd_out/site_versions/` back to `~/spd_out/26-11-04_paper_draft/site_versions/vm/` (the new outputs land in `smoke5x/`, `val5x/`, `test/` and do not overwrite the 26-11-11 files, except `progress` and `logs/`, whose new tags differ), then I check the 5× group counts and val rows against the 10× ones and fill the site table.

---

26-11-14, Claude Opus 5.5 — *site table on test: VM run and results*; context: the answer above, `test.sh`, the run's logs and JSON outputs, the draft's `tab-res-site`

Julian (chat): "vm is up". I synced the working tree (`spd-vast --mode sync`; the instance had HEAD `779c1ee` from GitHub, the uncommitted 5× and `massive_factor` changes went up with the working tree, checked by grep on the instance), copied `test.sh` and ran it detached on the RTX 4090 (48 GB). All four steps exited 0 (smoke 15:03, val5x 15:06, test removal/shift 15:13, test twins 15:14 UTC). Outputs copied to `~/spd_out/26-11-04_paper_draft/site_versions/vm2/` (15 files, 36 MB, sha256 identical to the instance's); the 26-11-11 outputs in `vm/` are untouched. The VM is no longer needed. Table script: session scratchpad `site_tables.py` (not kept).

**Group counts at 5×** [verified: train statistics in `logs/val5x.log`, 1,962 train statements]: the massive group holds exactly one position per statement at layers 11 and 12 (1,962 each), none at layers 13–17, as expected from the six-statement example in entry *scan3 vs role on concrete statements*.

**Val, 5× against 10×** [verified: both val outputs, 986 statements]: the shift rows are unchanged within GPU noise (role α = 1: verdict sep −0.642 both, flips 0.777 vs 0.779), as expected since the massive token's twin difference is small (|diff| 0.27 at layer 11, 0.14 at layer 12, against 2.9–4.2 at the object). Mean:role changes little: verdict sep 0.055 [0.037, 0.072] (was 0.051), prefix next-token KL 0.52 (was 0.58), true → false flips 0.81 (was 0.87), false → true 0.13 (was 0.05).

**Test, site version** [verified: `vm2/test/*.json`, one run; 1,794 statements; twin interventions on 898 statements = 449 twin pairs of equal length and opposite verdicts]. sep = separation share left; ν for twin interventions; intervals 95% bootstrap within family.

| state | verdict | second | Yes/No question | Yes/No same line | digit KL (4 questions) | recall subj / obj p | prefix / object-end KL |
|---|---|---|---|---|---|---|---|
| original | 1 | 1 | 1 | 1 | 0 | 0.869 / 0.893 | 0 / 0 |
| zero | 0.082 [0.071, 0.095] | 0.160 | 0.147 | 0.162 | 0.081–0.297 | 0.874 / 0.804 | 2.28 / 2.53 |
| mean:role | 0.066 [0.052, 0.079] | 0.030 | 0.025 | 0.059 | 0.035–0.110 | 0.953 / 0.906 | 0.47 / 1.09 |
| mean:scan3 | 0.050 [0.039, 0.060] | 0.066 | 0.068 | 0.059 | 0.036–0.109 | 0.943 / 0.905 | 1.07 / 1.25 |
| shift:role:1 | −0.551 [−0.586, −0.517] | −0.395 | −0.369 | −0.411 | 0.036–0.128 | 0.876 / 0.891 | 0.001 / 0.62 |
| twin average (ν) | 0.445 [0.427, 0.463] | 0.492 | 0.482 | – | 0.022 (words) | – | – |
| swap (ν) | 0.933 [0.925, 0.940] | 0.938 | 0.948 | – | 0.081 (words) | – | – |

Swap IIA (share of twins whose verdict becomes the twin's): 0.945 (verdict), 0.938 (second), 0.933 (Yes/No). Flip directions (true → false / false → true): zero 0.18 / 0.72 (pushes to "true"); mean:role 0.80 / 0.11, and 1.00 / 0.00 under second and both Yes/No prompts (pushes to "false"); shift 0.79 / 0.73 under the training prompt, 0.97–0.99 / 0.23–0.46 under the other three; twin average 0.54 / 0.39 under the training prompt, 0.95–0.99 / ≤ 0.06 under second and Yes/No. Test agrees with val (shift α = 1: val sep −0.64, flips 0.78; test −0.55, 0.76; swap ν val 0.944, test 0.933). Zero removal's verdict KL 0.650 equals the G2 reproduction's value.

**Reading** [concluded: from the table; one run, site only]:
- All three truth prompts move together under the swap (ν 0.93–0.95) and the twin average (≈ 0.45–0.49): what the sites write at the statement tokens is read the same way by every truth question, so A2 holds at the site level and every truth readout can judge a split.
- The swap and the fixed shift are selective at the site level: prefix untouched, recall of subject and stated object unchanged under the shift, digit KL ≤ 0.13. The removals are not: prefix KL 2.28 (zero) / 0.47 (mean), and each pushes the verdict to one side (zero → "true", mean → "false"). The zero removal lowers object recall (0.89 → 0.80), the mean removal doesn't. So a part's removal has to be read against these side effects; the swap and shift are the cleaner tests.
- New observations, not predicted: (1) the mean removal *raises* subject recall (0.87 → 0.95; mainly the case-flipped spelling and the weak families);
  OUTDATED (26-11-14): the parenthesis was a guess, checked right after: the gain is split evenly between spellings (as written 0.643 → 0.682, case flipped 0.211 → 0.250) and comes from larger_than (0.86 → 0.98), smaller_than (0.83 → 0.97), neg_cities (0.86 → 0.95) and element_symb (0.81 → 0.88); the animal families are unchanged [verified: test JSON]. (2) under the second and Yes/No prompts the shift and the twin average act mostly in one direction (towards "false"), so the twin average's ν ≈ 0.5 there averages "moved fully" and "not moved" rather than a halfway score per statement. Under the training prompt both directions move. Possible reading [assumed]: those prompts have a stronger "false" default when the statement-token evidence is weak or mixed.
- Role vs layer-scan grouping on test: verdict separation 0.066 vs 0.050 (intervals touch: [0.052, 0.079] vs [0.039, 0.060]); role halves the prefix damage (0.47 vs 1.07) but pushes harder towards "false" under the other prompts. Same picture as on val.

**Draft** (`reports/26-11-04_tpd_truth_paper_draft.typ`, 3.2.1): `tab-res-site` filled with the test numbers (raw values instead of the placeholder's "effect m_site − m_0", stated in the caption), the layer-scan mean as a side row, the massive-token count and the shift's definition and α in the caption; the old caption sentence on B3 as the "existing fixed-shift cells" removed (superseded by this run); the result placeholder below the table replaced by a five-point reading (the one above, in paper voice, tagged [concluded]). Compiles; page 11 checked by eye.

Not done: per-family breakdown (swap ν and shift flips per family), and recall under the twin interventions (excluded by Julian's choice). The seeded-defect table and every split (P/R) run are next.

---

26-11-14, Claude Opus 5.5 — *naming the appended questions*; context: the draft's sections 2 and Evaluation setup, `YES_NO_PROMPTS` / `DIGIT_QUESTIONS` in `paper_side_effects.py`, `SECOND_STIMULUS` in `verdict_lens_analysis.py`, `RECALL_TEMPLATE` in `paper_tests.py`

C: (transcribed from chat) Alright I think we need names for the different "prompts" and a name for these type of prompts themselves

(Prompted by Julian asking what "(question / same line)" in the 3.2.1 site table means: the two Yes/No prompts kept by G0, keys `question` and `same_line`.)

**Proposal** [assumed: proposal, not yet ratified].

*Category name: **question*** = the fixed text appended after the statement; *prompt* = statement + question; a *readout* = a question plus how its answer is scored (the next-token readout is the one readout without a question). Reason: the draft already uses "question" this way ("a statement followed by a fixed question", "truth questions", "digit questions", "the appended question runs on the original weights"), so this only makes it official. Weak point: `amount` and the recall questions are sentence stems, not grammatical questions; the definition should say so. Alternatives considered: *probe* (collides with linear probes, which the paper discusses), *query* (collides with attention queries), *stimulus* (the code's term; odd in a paper), *suffix* (accurate but colourless).

*Names of the individual questions* (monospace in the paper):

| name | kind | text after the statement (code constant) | answer | line |
|---|---|---|---|---|
| `amount` | truth, training | `"\nThe amount of truth in this statement is "` (training stimulus) | 1 / 0 | new |
| `true-or-false` | truth, held out | `" Is this statement true (1) or false (0)? Answer: "` (`SECOND_STIMULUS`) | 1 / 0 | same |
| `yes-no` | truth, held out | `"\nQuestion: Is the statement above true? (Yes/No)\nAnswer:"` (`YES_NO_PROMPTS["question"]`) | Yes / No | new |
| `yes-no-inline` | truth, held out | `" Is this true? Yes or No? Answer:"` (`YES_NO_PROMPTS["same_line"]`) | Yes / No | same |
| `humour`, `emotion`, `surprise`, `words` | other quality (digit questions) | `"\nThe amount of humor in this statement is "` etc. (`DIGIT_QUESTIONS`) | digit 0–9 | new |
| `recall-subject`, `recall-object` | other quality | `'\nThe {noun} in the statement above is "'` (`RECALL_TEMPLATE`) | teacher-forced spellings | new |
| (none) | other quality | next-token predictions on the statement (positions prefix / object, end) | – | – |

Code keys stay as they are (`question`, `same_line`, `second`): existing output files are keyed by them; the paper maps them through this table.

*Where in the draft.* One table like the above in *Evaluation setup → Prompts*, replacing the prose list; tables and captions then use the names (e.g. the 3.2.1 header "Yes/No (question / same line)" → `yes-no` / `yes-no-inline`). "Verdict" can stay as the name for the answer to any truth question; "the verdict prompt" (Methods) becomes "statement + `amount`".

**Flag found while checking** [verified: `SECOND_STIMULUS` in `verdict_lens_analysis.py:68`; docstring of `paper_tests.py`]: the second prompt starts with a space, so like `same_line` it continues on the statement's line. The draft quotes it without the leading space and calls only the Yes/No prompt "same line". This matters beyond wording: with a same-line question the statement ends in a bare `.` token, while with a new-line question the period merges into `.\n`. So under `true-or-false` and `yes-no-inline` the edits act on a different last statement token than under `amount`. The "line" column above makes that visible.

---

26-11-14, Claude Opus 5.5 — *general names from scratch: question, rating, Yes/No*; context: the entry above

C: (transcribed from chat) Let's talk about the general names first. I want good names and this shouldn't be influenced by what we have right now.

statement and prompt are good. I'm not sure about question though. I also want an easy way to reference yes/no-question and digit question (the second one feel clunky)

**Two levels need names** [assumed: proposal]: the general thing (the text after the statement that asks the model about it) and its kinds. The kinds are best told apart by how the model answers, because that also decides how the answer is scored:
- the model completes "The amount of X in this statement is " with a digit 0–9;
- the model picks one of two labelled options ("true (1) or false (0)");
- the model answers Yes or No;
- the model completes a word from the statement (recall).

**Recommended scheme:**
- General: *question*. Every alternative collides with a term the paper needs: *probe* (linear probes; the draft says explicitly that it does not use them), *query* (attention queries), *cue* ("truth cues" reads as surface features that give the truth away), *item* (statements are the data items), *suffix* (says where, not what for). Fault of *question*: some are sentence openings, not questions; one definition sentence fixes that.
- Digit questions → *ratings*: the model rates an amount on a 0–9 scale ("the humour rating", "the ratings"). Side effect: the training question has exactly this form ("The amount of truth in this statement is "), so it becomes the *truth rating*, which tells the reader that it is the same template as the other-quality ratings, answered in practice with 1 or 0. The word-count question is a count, not a rating; it can stay in the group with that remark, or be named *word count* on its own.
- Labelled options → *choice* ("the truth choice").
- Yes/No → *Yes/No question*, short *Yes/No* in tables and running text ("under Yes/No"). *Polar question* (the linguists' term) rejected: next to the negated families it reads as polarity of the statement.
- Recall → *recall* (subject / object).

So the truth questions would be: truth rating (training), truth choice, Yes/No (two wordings: one starting on a new line, one on the statement's line), and the other-quality questions: the ratings (humour, emotion, surprise), word count, recall. The individual names in the entry above (`amount`, `true-or-false`, `yes-no`, ...) are set aside until the general names are fixed.

Open, not asked by Julian but clunky in the same way: *other-quality readouts / questions*. No good short name found yet: *control* collides with the control bases, *off-target* with tPD's target data.

C: I don't like question. I like probe a lot actually. Maybe we can call it a "verdic probe"?
Answer (26-11-14, Claude Opus 5.5): *Probe* works, with one rename elsewhere in the paper. I rejected it above too quickly. *Verdict probe* is a good name, but only for the truth probes, not as the general term.

  *Why probe is defensible* [concluded, except where tagged]. A probe is an instrument that measures without changing what it measures. That fits well: the appended text runs on the original weights and only reads out what the statement tokens hold. Prompt-based probes also have precedent: the LAMA "probe" (Petroni et al. 2019, *Language Models as Knowledge Bases?*) is a set of cloze sentences [assumed: from memory, check before citing].

  *The collision is real but local* [verified: grep of the draft, 26-11-14]. "Probe" currently means a linear classifier on activations in four places: the setup sentence "We don't use probes or truth directions for the definition" (*A small model of the truth computation*), *Inspection tests* ("the reason probe accuracy is not evidence"), the status section's "Analyses not tied to a test" (K0), and the glossary (K0 "Logistic probes over all features…", K1 "probes on the top k components"). Fix: call those *linear classifiers* throughout, and define *probe* once in Evaluation setup ("a probe here is a fixed text appended to the statement, not a classifier on activations"). Without the rename, the first sentence would read "we don't use probes" a few lines before the paper starts using probes.

  *Why "verdict probe" rather than "truth probe".* Our statements come from the Truth-is-Universal paper, and in that literature a "truth probe" is a linear classifier for truth. *Verdict probe* avoids this, and it matches the draft's own term: the *verdict* is already defined as the model's answer to a truth question.

  *Why not as the general name.* The ratings for humour, emotion, surprise and word count, and the recall questions, give no verdict. "Verdict probe" for them would be wrong, so the general word stays bare *probe*. Proposed scheme [assumed: proposal]:
  - *probe*: the text appended after the statement. *Prompt* = statement + probe. *Readout* = probe + how its answer is scored. The next-token readout is the one readout without a probe.
  - *verdict probes* (truth): the training probe ("The amount of truth in this statement is "), the *choice probe* ("true (1) or false (0)"), and the two *Yes/No probes*. The training probe needs its own short name once "verdict probe" means the whole class. "Training probe" is the plainest option.  
  - *content probes* (the other qualities): *rating probes* (humour, emotion, surprise; word count is a count, which one remark covers) and *recall probes* (subject, object). Why "content": in the small model H these readouts are the Q_k, which read the content Z and not T (assumption A3). So "verdict probe / content probe" names the split that the tests use, and it also gives the clunky "other-quality readouts" a replacement. Weak point: humour or surprise is "content" only in the broad sense of "what the statement says, apart from whether it is true". If that reads oddly, *quality probes* is the fallback.
  OUTDATED (26-11-14): the *verdict / content probe* scheme above is replaced by names by format with "truth" in each truth probe's name, built on Julian's quality / recall grouping (answer beneath his next comment).
  C: Ok, I like probe. The different probes need better names. So I've just come to realise that the training probe and the second probe are actually qualitavely different, since the first one allows for a gradet answer like 0.8 or 75%, but the second just allows a binary answer. I don't want to make a big dealt out of this though, because I don't think it matter much. What do you think about my suggestions:
  - quality probes:
   - rating probe 1
   - tating probe 2
   - yes/no probe 1
   - yes/no probe 2
   - humor probe (etc)
- recall probes
   - subject/object probe

  Answer (26-11-14, Claude Opus 5.5): **Graded vs binary: agreed on both counts.** The difference is real: "The amount of truth in this statement is " allows "0.8" or "75%", while "true (1) or false (0)" offers two options. It also matters little in practice. The verdict-digit check (`convos/julian/26-11-05_verdict_digit_trust_SUMMARY.md`) sampled continuations on 200 in-distribution test statements. After '1', 99.6% meant "true". After '0', 96.1% were plain "0"/"0%", 2.4% were "0.5" hedges (at near-ties), and none were "0.7"–"0.9" [verified: that SUMMARY, re-read 26-11-14; one run, out-of-distribution statements and interventions not covered]. So the model answers the graded probe as if it were binary. One sentence in the paper with that number covers it.

  **Your grouping (quality vs recall) is a good cut** [concluded]: it separates probes that ask the model to judge the statement from probes that ask it to copy a word. Three points in the list:
  1. *"Rating probe 2" contradicts your own observation.* The "true (1) or false (0)" probe is the binary one, so in a format-based scheme it belongs with the Yes/No probes as a choice, not with the ratings.
  2. *The humour probe is a rating probe.* Its text is "The amount of humor in this statement is ", the same template as the training probe, answered with a digit 0–9. So the rating probes are truth, humour, emotion, surprise and word count; only the quality rated differs.
  3. *The truth / other split is missing from the names.* The tests are organised by it (Table 3: truth readouts vs other qualities), but in your list the four truth probes and the humour probe sit in one group with nothing marking which are which. Putting "truth" into each truth probe's name fixes that without an extra group level.

  On the numbers 1 / 2: the two Yes/No probes differ in a way that matters. One starts on a new line and one continues on the statement's line, so the statement's last token is '.\n' in one and a bare '.' in the other (flag in the entry *naming the appended questions*). A descriptive name carries that, while a number sends the reader to the table every time.

  Proposed names [assumed: proposal]:

  | group | name | text after the statement | answer |
  |---|---|---|---|
  | quality probes (truth) | *truth rating* (training) | `\nThe amount of truth in this statement is ` | digit; graded answers possible, in practice 1 / 0 |
  | | *truth choice* | ` Is this statement true (1) or false (0)? Answer: ` | 1 / 0 |
  | | *Yes/No* | `\nQuestion: Is the statement above true? (Yes/No)\nAnswer:` | Yes / No |
  | | *inline Yes/No* | ` Is this true? Yes or No? Answer:` | Yes / No |
  | quality probes (other) | *humour / emotion / surprise rating*, *word count* | `\nThe amount of humor in this statement is ` etc. | digit 0–9 |
  | recall probes | *subject recall*, *object recall* | `\nThe {noun} in the statement above is "` | the word, teacher-forced |

  In running text: "the truth probes" (the first four) and "the rating probes" (all five ratings, the truth rating included), so both groupings stay available. "Truth probe" is safe here only because the linear classifiers get renamed (answer above). *Subject recall* rather than *subject probe*, because "subject probe" could be read as a probe for the subject's representation.



  Nothing in the draft is changed until you confirm the scheme. Once you do, the renames are: the four classifier passages, the Evaluation-setup prompts paragraph (as a table), and the table headers and captions (e.g. 3.2.1's "Yes/No (question / same line)" → the two Yes/No probes, whose individual names are still open).

  C: Ok, here is the confirmed scheme:
    | group | name | text after the statement | answer |
  |---|---|---|---|
  | quality probes (truth) | *truth rating 1* (training) | `\nThe amount of truth in this statement is ` | digit; graded answers possible, in practice 1 / 0 |
  | | *truth rating 2* | ` Is this statement true (1) or false (0)? Answer: ` | 1 / 0 |
  | | *Yes/No 1* | `\nQuestion: Is the statement above true? (Yes/No)\nAnswer:` | Yes / No |
  | | *Yes/No 2* | ` Is this true? Yes or No? Answer:` | Yes / No |
  | quality probes (other) | *humour / emotion / surprise rating*, *word count* | `\nThe amount of humor in this statement is ` etc. | digit 0–9 |
  | recall probes | *subject recall*, *object recall* | `\nThe {noun} in the statement above is "` | the word, teacher-forced |

reasoning: I am clusstering by behavioural changes. The decompositions act differently on the digit vs yes/no probes. Also the details of a probe wheather inline / new line are irrelevant. They are simply two probes for checking generalisation.

C: (transcribed from chat) I've decided on a scheme for probes. The rest should be decided as well.

Answer (26-11-14, Claude Opus 5.5): Applied. Decided [decided: the two C: lines above]: the probe names of Julian's table (truth rating 1 / 2, Yes/No 1 / 2, humour / emotion / surprise rating, word count, subject / object recall; groups: quality probes (truth, other) and recall probes). Also decided ("the rest"): *probe* = the appended text, *prompt* = statement + probe, *verdict* = the answer to a truth probe, and the draft's linear classifiers are no longer called probes. Julian's reasoning for the numbering: the clusters follow behaviour (the decompositions act differently on the digit and the Yes/No probes), and new line vs inline is irrelevant, since the two probes of a format only check generalisation.

Changes in `reports/26-11-04_tpd_truth_paper_draft.typ` (and the diagram helper's label "question: original weights" → "probe: original weights"):
- Methods: the training text is introduced as a probe; *probe*, *prompt*, *truth rating 1* and *verdict* are defined there. "Verdict prompt" is gone everywhere (Tables 1 and 2: "under truth rating 1"); "question tokens" → "probe tokens" (positions, layer scan, loss list).
- Section 2: the truth-readout list (truth ratings 1, 2; Yes/No 1, 2; another language), the other-quality readouts (other ratings, recall probes), the cross-product axis "verdict style" → "truth probe", A2 and the "Why two kinds of readout" paragraph, the test bullets ("other-quality questions" → "probes").
- Evaluation setup: the *Prompts* paragraph → *Probes*, with the groups, Julian's graded-vs-binary point (and the verdict-digit numbers: 99.6% of continuations after "1" mean true; after "0" 2.4% "0.5" hedges, none 0.7–0.9) and new Table `tab-probes` (texts copied from `paper_side_effects.py`, `verdict_lens_analysis.py`, `paper_tests.py`).
- Results: headers and captions of the site table, scorecard, other-quality and transfer tables; the reading under the site table names the probes ("under truth rating 2 and both Yes/No probes"); subsection 3.7 is now *Other truth probes and out-of-distribution statements*.
- Linear classifiers: the setup sentence ("We don't use classifiers on activations…"), inspection tests, K0 under status, glossary K0/K1 ("logistic-regression classifiers").
- Glossary: *both prompts* → *both truth ratings*; the entry *G2 digit questions* renamed *G2 ratings*; B2 "chose truth rating 2"; G0 "eight Yes/No wordings tried; G2 uses Yes/No 1 and 2".

Checks [verified: 26-11-14]: G2 ran both Yes/No probes, T2 included (`~/spd_out/26-10-29_paper_writing/gpu/results/g2_tables_test.md`: columns `question` / `same_line`, T2 subsections for both), so "T2 under all four truth probes" holds. Under the twin interventions only Yes/No 1 ran (the site table says so). Compiles; pages 11 (site table) and 19 (probe table) rendered and checked by eye.

Two choices of mine Julian may want to change [assumed]: (1) the probe-table caption still says in one sentence that truth rating 2 and Yes/No 2 continue on the statement's line, so the last statement token is a bare period. That is not part of the names (Julian: irrelevant for them), but a reader re-running the edits needs it. (2) In the other-quality readout list, recall now reads "the recall probes (subject recall and object recall; but see *Which T*)". *Which T* discusses recall of the stored fact, which was dropped as a readout (26-11-11), so that pointer no longer fits the recall probes as they are and should probably go. I left it in, because it is a content change.

---

26-11-15, Claude Opus 5.5 — *original row in the site table*; I have read the site table and the end of this LOG

Julian asked in chat: "Can you add "original" to the table" (the site-version table `tab-res-site`, which his cursor was on). Added a first row *nothing (original)*: separation share 1 (ν = 0) under every truth probe, other-rating KL 0, next-token KL 0 / 0 (all by definition), and recall 0.87 / 0.89, which the caption already gave as "original 0.87 / 0.89" and the entry *site table on test: VM run and results* (26-11-14) confirms (0.87 → 0.95 subject, 0.89 → 0.80 object). The caption's parenthetical moved into the row; the caption now says that every value of the row except recall holds by definition. Compiles.

---

26-11-15, Claude Opus 5.5 — *Yes/No split into two columns in the site table*; same context as the entry above

Julian asked in chat: "Can you split yes/no 1 and yes/no 2?" The column *Yes/No 1 / Yes/No 2* of `tab-res-site` is now two columns (8 in all); values moved unchanged, Yes/No 2 is "–" under the twin average and the swap (not run there). Compiles; page 12 rendered and checked by eye: the table fits the page width, but the interval cells now wrap onto two lines.

---

26-11-15, Claude Opus 5.5 — *how the scaling dial is built, and what it can show*; context: draft sections *Validating the tests* and *Seeded defects*, `convos/julian/26-11-05_evaluation_categories_LOG.md` section *4. Calibration: seeded re-implementations with a dial*

Julian asked in chat: "How exactly is the re-implementation by construction constructed? Does it need a already trained decomposition?"

**Construction.** Take any decomposition of a layer, $W = \sum_c W_c + \Delta$ with $W_c = U_c V_c^\top$, and a number $k \ge 0$. Replace every component by $(1+k) W_c$ (all 96 of the layer, not only $P$) and the remainder by $\Delta - k \sum_c W_c$. The sum is still $W$, so the model is unchanged on every input; the CI network is untouched, so the CI values and the active set $P$ stay the same. Per position the components now write $(1+k) C$ and $\Delta$ writes $D - kC$: a write of the components that is not part of $y$, cancelled by $\Delta$. No training is involved. Code: not written yet (the draft's status table lists it under *add*) [verified: grep of `spd/` for the construction, 26-11-15, nothing found].

**Yes, it needs a decomposition,** but any one: a trained run (plain, recon, ortho), and formally also an untrained one. What the dial can show depends on the base, see below.

**What the dial does to the tests** [concluded: algebra, one layer on its own inputs]. Under the $P$-swap a layer writes
$(1+k) C_P(x') + (1+k) C_{R\text{-comps}}(x) + D(x) - k C(x) = y(x) + (1+k)\,[C_P(x') - C_P(x)]$.
So the dial multiplies $P$'s twin difference by $1+k$, nothing else. Same for every edit of $P$ with $\Delta$ kept (removal: $y - (1+k) C_P$). States with $\Delta$ off (*only P*, *components only*) change more: they write $(1+k) \sum m_c C_c$.

**Consequence for the claim in the draft** [concluded, from the line above]. The draft says the $P$-swap read by the other qualities "must fail increasingly with $k$". It fails only through two routes: (a) $P$'s twin difference of the base already contains content besides $T$, which the dial amplifies; (b) the amplified edit is large enough to disturb the other qualities by its size alone (off-distribution), even if the difference were pure $T$. If the base's $P$ were clean, the dial would be a stronger truth edit, closer to steering than to the re-implementation of the failure list (a write $c$ with content the truth readouts do not read). So the dial validates "the other-quality readouts notice an oversized edit", not "they notice cancelled foreign content". The prediction "fails increasingly" is [assumed], not guaranteed.

**Possible addition** [assumed: proposal, not discussed]: a second seeded defect that writes foreign content by construction: add to $P$ a component $E$ whose write differs between twins in a direction a shared quality reads (e.g. a subject direction), with $\Delta \mapsto \Delta - E$. The model stays the same, the truth readouts under the $P$-swap stay the same, and only the other-quality readouts should move. Exp 4a remains the real-world case.

---

26-11-15, Claude Opus 5.5 — *scaling dial: what is to decide before running it*; context: the entry above, `paper_tests.py` (`run`, docstring), `paper_interventions.py` (`ComponentSplit`, `part_of`, `edit_fn`)

Julian asked in chat: "First and foremost, I want to do the test. What do I have to decide for that?"

**Code** [concluded: from reading `part_of` and `edit_fn`]. `paper_tests.py run --target active` already runs every intervention on a component set: $P$ = the active set's writes $p = (xV)U$, $R = y - p$, edit $y' = y - q + q'$. The dial is then $p \mapsto (1+k)p$, i.e. $U \mapsto (1+k)U$ in the `ComponentSplit`, a few lines (a `--dial k` option, recorded in the output name). Swap, twin average, zero, mean and shift (whose train difference scales with $q$) all follow without further change. Per the entry above, that equals the weight construction exactly. Not written yet.

**Decisions** (my recommendation in brackets, all [assumed] until Julian chooses):
1. *Base decomposition(s).* The dial amplifies the base's own $P$ difference, so the result is a property of the base. [one base first: ortho s1 `s-0c2fa0d7`, the run the site versions used, with the recovery JSON `~/spd_out/26-10-29_paper_writing/gpu/results/rec_check/recovery_s-0c2fa0d7.json`; plain s1 `s-7152548c` next if the first looks informative.]  
C: Can you explain in bit more detail why those?
Answer (26-11-15, Claude Opus 5.5): My reasons were weaker than they read. "The run the site versions used" does not matter: the site versions do not use the decomposition (`--run` only supplies the model and the positions). The real criterion: the dial is informative when the base passes the other-quality tests at $k = 0$, so that a failure at $k > 0$ is the dial's doing. Which base passes is unknown, since no $P$ test has run yet. Ortho is the method trained against cancellation, so it is the most likely to start clean [assumed]; plain is the one suspected of re-implementing, so its $k = 0$ may already fail and the dial would add little there. Since the tests are cheap, the better choice is all six trained decompositions (plain `s-7152548c`, `s-46efc525`; recon `s-eeac663f`, `s-edc17041`; ortho `s-0c2fa0d7`, `s-4186f973`): $k = 0$ is then each one's own $P$ test, which the scorecard needs anyway, and the comparison across bases shows how much of the dial's effect comes from the base. Active sets from `~/spd_out/26-10-22_tpd_truth_paper/phase_d/h200_paper_d/recovery/recovery_<run>.json` (all six there; for ortho s1 the H100 re-run in `rec_check/` gave the same active set, `convos/julian/26-10-29_paper_writing_LOG.md`, entry *G1 results: next-token damage yes, Yes/No no, digits mixed*). I would start the script with ortho s1 so that a crash shows early.
2. *Values of k.* [0, 1, 3, 10, the draft's placeholders; k = 0 is the undialled $P$ test, needed anyway.]  
C: yeah, that's fine
3. *Interventions.* [swap and shift:role:1, the two tests the dial is meant to break; zero as a cheap third.]  
C: why zero and not mean? It's cheap to run both, right?
Answer (26-11-15, Claude Opus 5.5): No good reason; yes, both are cheap. The mean removal needs the train statistics, one extra pass over the 1,962 training statements per probe, which the shift needs anyway. So: swap, shift:role:1, zero, mean:role (role, as for the site table).
4. *The other-quality readout under the swap* — the main open point. The code allows only `digit:words` (word count) under twin interventions and blocks recall there by an assert (Julian 26-11-11: few readouts for the swaps). Subject recall is the cleanest quality the twins share. [lift the block for `recall:subject` in this test; truth readouts `verdict` and `second`.] Without it, the dial's "fail" can only show in the word count.  
C: write the code so the block is generally lifted. I might also rerun the Site versions then. These tests are cheap
Answer (26-11-15, Claude Opus 5.5): Done, see the entry *dial option and recall under the twin interventions* below.
5. *Split.* [val: the dial validates the test, it is not a result, and val keeps test untouched for the thresholds.]  
C: which thresholds?
Answer (26-11-15, Claude Opus 5.5): The ones the draft promises for the ✓/✗ cells: the scorecard caption says "The thresholds behind ✓ are fixed before the outputs are read", and the seeded-defect table's caption "Dial values $k$ are placeholders, to be fixed with the thresholds". With no pass/fail rule (your comment under 6) there are none, and my reason for val falls away. Then test is the better split: the site table is on test, and the $k = 0$ runs are the decompositions' own $P$ tests, reported on test. Those two captions (and the ✓/✗ legend of both tables) then need rewriting; not done, since it changes how results are presented.
6. *Pass/fail rule* for the ✓/✗ cells: not needed to run; the curves over k can be read first and the rule fixed before the decompositions' own test runs.  
C: I don't need a pass/fail rule.
Answer (26-11-15, Claude Opus 5.5): Noted; see the answer under 5 for what it changes in the draft.

---

26-11-15, Claude Opus 5.5 — *dial option and recall under the twin interventions*; context: the entry above with Julian's comments, `paper_tests.py`, `paper_interventions.py`, `docs/CODING.md`

Julian's comments on the entry above: k = 0, 1, 3, 10 fine; mean as well as zero; "write the code so the block is generally lifted. I might also rerun the Site versions then. These tests are cheap"; no pass/fail rule. Answers in place there (bases, mean, thresholds → test split).

**Code** (not run; no GPU on this machine):
- `paper_tests.py run --dial k` (default 0; target active only, asserted): `component_split` (`paper_interventions.py`) now takes `dial` and scales the active set's $U$ by $1+k$. Scaling only the active set is exact for every test, because $R$ is computed as $y$ minus the edited part, so the inactive components' scaling would cancel inside $R$ anyway. Output names carry it (`target` = `active-dial<k>`), the JSON summary has a `dial` field.
- Recall under twin interventions: the assert is gone, for every target. The twin pass runs the twin's statement followed by the row's *own* recall question and answer; only the twin's writes at the statement positions enter the edit, and by causal attention they don't depend on what follows. Consequence: `recall:object` under the swap scores the row's own stated object (it should drop, since the object travels with the swapped writes), not the twin's. `run_cells` asserts that both sequences have the same layout; it will crash loudly if a twin's statement tokenizes to another length under the recall question.
- `make check`: clean (ruff, basedpyright).

**Run script** `~/spd_out/26-11-04_paper_draft/dial/dial.sh` [assumed: plan, not run]: smoke on ortho s1 (k = 3, val, 2 per dataset, all interventions and readouts: the first GPU run of recall under twin interventions), then on test for each of the six trained decompositions (ortho s1 first) and k ∈ {0, 1, 3, 10}: `zero, mean:role, shift:role:1, twin_average, swap` with every readout, part P. `twin_average` is my addition: cheap, and at k = 0 it is a scorecard test. Not included: part R (the defect table's row "R on the other qualities") and $P = W$ (= the site versions, already run). Copy and launch commands in the script's header.

Open for Julian [assumed]: the draft's ✓/✗ legend and the two captions that promise thresholds (scorecard, seeded-defect table) no longer fit without a pass/fail rule; I have not changed them.

---

26-11-15, Claude Opus 5.5 — *no pass/fail rule confirmed; dial needs a GPU*; same context

C: (transcribed from chat) Exactly. I just want the data and analyse them. Do you need a gpu?

So the dial runs produce data for Julian's analysis, with no ✓/✗ rule [decided: the transcribed line above]. A GPU is needed (Qwen2.5-7B; this machine has none). The `vastai` ssh alias points to the old instance, which refuses connections [verified: ssh, 26-11-15]. Time estimate [assumed: extrapolated from the 26-11-14 site run on the RTX 4090, 48 GB: 4 non-twin interventions × 10 readouts on test took about 7 min, the two twin interventions with 4 readouts about 1 min]: about 8–12 min per (decomposition, k) with model loading, so 3–5 h for the 24 runs.

---

26-11-15, Claude Opus 5.5 — *target all added; dial launched on the H100*; same context

C: (transcribed from chat) Yes, add it but don't use 6 seeds but only seed 1 for all three methods. GPU is up

(Answer to my chat question whether to add P = all components as a third target and run both targets.)

**Code** [concluded: `make check` clean]: `paper_tests.py run --target all` = P is every component of every decomposed layer (672), R = Δ's write $y - C$; no `--recovery`; the dial works with it (`all-dial<k>`). Asserted that the split holds every component.

**Run** [decided: the transcribed line above, for the bases]: `~/spd_out/26-11-04_paper_draft/dial/dial.sh` updated to seed 1 of each method (ortho `s-0c2fa0d7`, recon `s-eeac663f`, plain `s-7152548c`) × target {active, all} × k ∈ {0, 1, 3, 10} = 24 test runs, after two smoke runs (ortho, k = 3, val, 2 per dataset, both targets). Interventions zero, mean:role, shift:role:1, twin_average, swap; all ten readouts; part P. Instance: vast.ai H100 80 GB, power limit 700 W [verified: nvidia-smi], rented by Julian. Working tree synced with `spd-vast --mode sync` (instance at `main` 6607c7d6 from GitHub plus the uncommitted dial code, checked by grep on the instance); active sets copied from `~/spd_out/26-10-22_tpd_truth_paper/phase_d/h200_paper_d/recovery/`. Launched detached 10:32 UTC; outputs on the instance in `/root/spd_out/dial/{smoke,test,logs}`.

C: (transcribed from chat) Ok, I'm gone until 12 UTC. You are in charge to fix any problems that might come up until then. When everything is finished kill the instance. If you come to a problem you don't know how I would want it to be resolved you can also kill the instance and we resume from there later

(First test run, ortho s1 active k = 0, took about 3 min on the H100: 10:39:38 → 10:42:30 UTC [verified: progress file].)

---

26-11-15, Claude Opus 5.5 — *dial run finished, outputs copied, instance destroyed*; same context

**Run** [verified: progress file]: both smoke runs and all 24 test runs exited 0; ~2 min 52 s per run on the H100 (first run 10:39:38 → 10:42:30, last ended 11:49:31 UTC). Smoke check: recall under the swap ran on both targets (the twin-layout assert held); the smoke's twin numbers rest on one pair and mean nothing.

**Outputs** [verified: sha256 of all 83 files identical on the instance and locally]: `~/spd_out/26-11-04_paper_draft/dial/vm/` — `test/` (24 × `tests_<target>_P_test_<run>_<interventions>_<readouts>.{json,npz}`, target = `active`, `active-dial{1,3,10}`, `all`, `all-dial{1,3,10}`; 16.7 MB per npz), `smoke/`, `logs/` (one per run), `progress`, `recovery/`. Copy speed from the instance was 130–270 KB/s, so the 412 MB took ~25 min.

**Instance** vast.ai 60773727 destroyed at Julian's instruction (transcribed above) after the checksum match [verified: `vastai show instances` lists none].

**Spot check, not an analysis** [verified: ortho s1 active, test, from the JSONs]: the dial changes the results monotonically where expected. Swap, verdict ν 0.32 / 0.49 / 0.47 / 0.53 at k = 0 / 1 / 3 / 10; word-count KL under the swap 0.011 / 0.033 / 0.110 / 0.405; subject-recall change under the swap −0.007 / −0.018 / −0.034 / −0.340; zero removal, separation left 0.20 / 0.15 / 0.075 / 0.015. Analysis is Julian's.

---

26-11-15, Claude Opus 5.5 — *dial analysis*; context: the dial outputs `~/spd_out/26-11-04_paper_draft/dial/vm/test/` (24 JSON summaries), the site outputs `site_versions/vm2/test/`, the draft's section 2 (failure modes, seeded defects)

Julian asked in chat: "Can you analyse the data?"

**Files.** `~/spd_out/26-11-04_paper_draft/dial/analysis/`: `extract.py` → `long.csv` (every metric of the 24 dial runs and the 2 site runs, one row per run × readout × intervention × metric), `tables.py` → `tables.md` (tables A–D below in full: truth readouts at k = 0 per probe, φ against the site, other qualities at k = 0, the dial over k). All numbers below are read from those files [verified: one run per cell, test split, 1,794 statements; twin interventions on 906 rows]. Truth metrics carry 95% bootstrap intervals within family (in `long.csv`; widths ±0.01–0.03 for the cells quoted); digit KLs and recall changes have **no intervals**, so differences of 0.01–0.02 there are not interpretable. Abbreviations: TR1/TR2 = truth rating 1/2, YN1/YN2 = Yes/No 1/2; sep = separation share left (1 = untouched); ν = swap fraction (1 = the twin's score); φ = effect as a fraction of the site version's ((1 − sep)/(1 − sep_site) for removals and shift, ν/ν_site for twin interventions).

**Comparability with the site table** [verified: base scores of both runs]: the twin set has 906 rows here, 898 in the site run. On the H100 six statements with |s| ≤ 0.25 get the other verdict than on the RTX 4090 (bf16 noise), which adds 4 pairs; the site's 898 rows are all contained. Negligible.

**1. The dial (the seeded re-implementation): do the tests catch it?**

| split | k | swap ν TR1 | swap word-count KL | swap subject-recall Δ | shift sep TR1 | shift word-count KL |
|---|---|---|---|---|---|---|
| site | – | 0.93 | 0.081 | – | −0.55 | 0.036 |
| ortho active | 0 / 1 / 3 / 10 | 0.32 / 0.49 / 0.47 / 0.53 | 0.011 / 0.033 / 0.110 / 0.405 | −0.007 / −0.018 / −0.034 / −0.340 | 0.95 / 0.81 / 0.23 / −0.13 | 0.002 / 0.004 / 0.013 / 0.073 |
| recon active | 0 / 1 / 3 / 10 | 0.79 / 0.93 / 0.84 / 0.68 | 0.033 / 0.069 / 0.146 / 0.216 | 0.008 / 0.020 / 0.025 / −0.386 | 0.04 / −0.66 / −1.08 / −1.18 | 0.013 / 0.039 / 0.092 / 0.229 |
| plain active | 0 / 1 / 3 / 10 | 0.92 / 0.89 / 0.77 / 0.66 | 0.113 / 0.200 / 0.363 / 0.789 | −0.038 / −0.064 / −0.164 / −0.358 | −0.28 / −0.60 / −0.87 / −0.97 | 0.043 / 0.111 / 0.270 / 0.639 |

(P = all: the same pattern; full rows in `tables.md`, table D.)
- *The other-quality readouts catch it on every base* [verified]: under the swap the word-count KL rises with k on all six splits (plain all flattens at k = 10: 0.48 → 0.44), and subject recall drops by 0.34–0.39 at k = 10 on all six. Under the shift the word-count KL rises too. The word count reacts more than the humour rating (ortho active, swap: humour KL 0.020 → 0.219; plain active 0.134 → 0.167).
- *The shift's truth readout overshoots, as the draft warned* [verified]: with k the shift moves the verdict further than the site shift does (recon active sep 0.04 → −1.18, against the site's −0.55; verdict flips 0.99–1.00 in both directions from k = 3). A truth-only reading would rate the dialled split better than the whole site.
- *The swap's truth readout does not overshoot* [verified]: ν stays at or below the site's 0.93 for every split and k, and for recon and plain it falls at large k (plain active 0.92 → 0.66; flips true → false 0.80 → 0.48). The draft's "the P-swap read by the hard verdict may get *easier*" holds for ortho (0.32 → 0.53) and for recon only at k = 1 (0.79 → 0.93), not in general. An oversized swapped write seems to disturb the verdict rather than transcribe it more strongly [assumed: interpretation].
- *Zero removal under the dial* destroys the statement: next-token KL before the object 0.13–1.9 at k = 0, 8.4–9.7 at k = 10, against 2.28 for the site's zero removal. So the dialled removal fails on side effects at once.
- Caveat (entry *how the scaling dial is built*): the dial only amplifies the base's own P difference, so it validates that the other-quality readouts notice an oversized edit, not that they notice foreign content.

**2. The decompositions' own P (k = 0): data for the scorecard.** φ under TR1 (site = 1):

| split | zero | mean | shift | twin average | swap |
|---|---|---|---|---|---|
| ortho active (29 comp.) | 0.87 | 0.57 | 0.03 | 0.16 | 0.34 |
| recon active | 0.77 | 0.71 | 0.62 | 0.79 | 0.85 |
| plain active | 1.07 | 1.07 | 0.82 | 1.44 | 0.99 |
| ortho all | 0.97 | 0.83 | 0.01 | 0.42 | 0.45 |
| recon all | 0.93 | 0.81 | 0.76 | 0.88 | 0.92 |
| plain all | 1.03 | 1.04 | 0.78 | 1.22 | 0.79 |

- *Plain looks like a re-implementation at k = 0* [concluded: overshoot plus side effects beyond the site's, both below]. Its active set does more than the whole site on the truth readouts: the twin average moves ν to 0.64 [0.63, 0.65] against the site's 0.45 [0.43, 0.46], and the mean removal leaves sep −0.00 [−0.01, 0.01] against 0.07. A part can only exceed the whole if the rest pushes back, i.e. R cancels a write of P. And its side effects exceed the site's: swap word-count KL 0.113 against the site's 0.081; mean removal of P = all lowers object recall by 0.52 while the site's mean removal leaves it (+0.01). Removing a part damages the object more than removing everything, which fits Δ writing the negative of something the components write.
- *Recon is the cleanest* [concluded, one seed]: φ 0.6–0.9 under every intervention and probe, with other-quality effects below the site's (recon active: swap word-count KL 0.033 against 0.081; shift digit KL ≤ 0.061 against ≤ 0.128; recall changes ≤ 0.04 for both targets except zero removal).
- *Ortho's P is needed but does not carry the twin difference* [verified for the numbers; reading assumed]: removing it kills most of the separation (φ 0.87 zero, 0.57 mean), but shifting, averaging or swapping it moves little (φ 0.03 / 0.16 / 0.34). Even P = all carries only 0.42–0.45 of the twin interventions' effect, so in ortho most of what distinguishes a true statement from its false twin is in Δ. In the draft's failure list this is closest to *a second route* in R or *a lever* in P; the R tests would tell.
- *All against active*: for ortho and recon P = all moves more than the active set (as expected); for plain the swap moves less with all components (0.74 against 0.92), so plain's inactive components counteract its active ones.
- *Transfer across probes* [verified: tables A, B]: for most cells φ under TR2, YN1 and YN2 lies within about 0.15 of TR1's. Exceptions: plain active's twin average (TR1 1.44, the others 0.94–1.01), ortho active's twin average (0.16 against 0.20–0.33) and plain's shift (0.55–0.72 against 0.78–0.82). So P's effects mostly transfer across answer formats; plain's overshoot under the twin average is strongest under TR1, the training probe.

**3. Methodological points found on the way.**
- *The shift's "prefix untouched" is by construction* [verified: train statistics]. The shift's twin difference before the object has norm 0.04–0.27, against 3.5–11 at the object (layers 11, 14, 17), because twins share every token before the object; and by causal attention an edit at the object cannot reach earlier positions. So the prefix KL ≤ 0.001 for every split and every k carries no information. The site-table reading in the draft lists "neither touches the next-token predictions before the object (prefix KL ≤ 0.001 for the shift ...)" as evidence of selectivity; for the shift it should say "by construction", as it already does for the swap. Not changed in the draft.
- *Missing site references*: the site's twin interventions were run with four readouts only (TR1, TR2, YN1, word count). So under the twin average and the swap there is no site value for YN2, the humour/emotion/surprise ratings or recall, and e.g. plain all's object-recall drop of 0.65 under the swap (the object travels with the swapped writes) cannot be judged yet. A rerun of the site twin interventions with all readouts takes a few minutes on a GPU.
- *Not run*: random component sets of the same size (φ_rand), part R, seeds 2.

---

26-11-15, Claude Opus 5.5 — *which bases to show in the seeded-defect table*; context: the *dial analysis* entry above, the draft's 3.2.2 (`tab-res-defects`, `fig-res-dial`)

Julian in chat: "I'd like to fill the table in 3.2.2 now. I don't think we need the "Exp 4a" column anymore. The question is, should we show the test results for all three or just one or two?"

**Recommendation** [assumed: proposal]: one base in the table, recon's active set; all six splits in the figure and an appendix table.
- *Why recon:* the table validates the tests, not the methods. A failure at k > 0 can only be blamed on the dial if the base passes at k = 0. Recon is the base closest to that (φ 0.6–0.9, side effects below the site's). Plain already fails at k = 0, so its dial row only adds to an existing defect; ortho's P carries almost no twin difference (shift φ 0.03), so its dial amplifies a near-zero edit and the shift barely reaches the site even at k = 3.
- *Why active:* it is the draft's default P (3.2 *Which P*).
- *Against cherry-picking:* the figure `fig-res-dial` already plans one panel per method; it can show all three with both targets, and `tables.md` (table D) becomes the appendix table. One sentence in the text says the pattern holds on all six splits (true for the other-quality readouts; the truth readouts differ by base, which is itself worth a sentence).
- *Columns:* k = 0, 1, 3, 10, and the site version (the existing *P = W* column). Exp 4a dropped as Julian proposes.
- *Rows that can be filled now:* P-swap read by TR1 (ν), P-swap read by the word count and by subject recall, fixed shift read by TR1 (the overshoot row, not in the table yet), fixed shift read by the word count, zero removal read by the prefix next-token KL. *Not run:* removal against random sets, R on the other qualities, cancellation norms / κ. I'd drop those rows until they run (the status table lists them) rather than show empty rows.
- *Cells:* values, no ✓/✗ (no pass/fail rule); the caption's threshold sentence goes.

---

26-11-15, Claude Opus 5.5 — *seeded-defect tables filled (3.2.2)*; context: the entries above

C: (transcribed from chat) Actually show three tables one for each method. The three changes of yours are fine.

(The three changes: a row for the fixed shift read by truth rating 1; the rows not run dropped; values instead of ✓/✗.)

**Draft, 3.2.2** [decided: the transcribed line above, for the layout; contents concluded]: `tab-res-defects` replaced by `tab-res-defects-ortho`, `-recon`, `-plain` (P = each method's seed-1 active set; columns k = 0, 1, 3, 10, site; Exp 4a column dropped at Julian's earlier request). Rows: P-swap read by truth rating 1 (ν), word count (KL), subject recall (change); fixed shift read by truth rating 1 (separation left) and word count; zero removal read by the next-token KL before the object. Cell values generated by `~/spd_out/26-11-04_paper_draft/dial/analysis/defect_tables.py` from `long.csv`, not typed by hand [verified: values match `tables.md`]. The intro paragraph now states what the dial does to every test ($(1+k)$ times P's own edit). The first caption explains every row, the 906 vs 898 twin sets, the missing site recall under the swap and why the shift's prefix KL is not shown (by construction). A reading below the tables [concluded: from the tables, one run each]: the other-quality readouts catch the dial in all three; the shift's truth readout overshoots for recon and plain; the swap's does not (only ortho's ν rises); the zero removal's side effect grows fastest; plain disturbs the word count more than the site already at k = 0; the dial amplifies P's own edit, not foreign content; "the same holds with P = all" with a placeholder for an appendix table. Compiles; pages 13–14 rendered and checked by eye.

Not changed, for Julian [assumed]: section 2.4 (*Validating the tests*) still lists Exp 4a as a seeded defect and predicts that the swap read by the verdict "may get *easier*" (holds only for ortho); the status table still says "*add:* scaling dial $k$, Exp 4a, $P = W$"; `fig-res-dial` is still a placeholder; the appendix table for P = all does not exist yet.
