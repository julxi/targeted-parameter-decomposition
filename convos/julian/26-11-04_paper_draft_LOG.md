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
