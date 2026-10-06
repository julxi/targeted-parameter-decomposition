I'd like to do some follow up experiments and do some clarifications.

First the clarifications. It's not clear how well tPD works. You wrote in the uth summary "[tPD] decomposes only what a chosen target dataset uses". I don't think that is necessarily true, it's something we hope it does. I can imagine that some mechanisms are too complex for tPD to decompose and they get shoved into Δ trading ci sparsity for reconstruction loss. Also the goal of tPD is not just sparsity, it has two forces, reconstruction faithfulness and CI sparsity, usually pulling in two different directions.
Actually I think it is surprising that it's working at all. And I still wonder what it is learning.
On the uth experiment, the decomposition is trained on all tokens. In some cases that's quite a lot of tokens that are not truth value related and still it seems that the decomposition picks up something truth related. Although the random baseline suggests that this could also just be some truth signal that the CI functions leak by chance. However, we have another experiment the "no truth" baseline for the statement experiment, which uses code one liners, and there it seemed that that the decomposition learned quitle litte (few active components, little effect ablating componints). Also that no-truth baseline was quite bad at detecting truth (it worked on some domains but was pretty bad on others).
So generally I wouldn't say tPD can't deconstruct some part of the truth mechanism, or maybe more principled, cannot find predictions for thruthfullness. However, we have not found evidence yet that it does.
There is also an interesting finding in the uth experiment. The trained decomposition probes did really really bad on some held out tasks, worse than chance and untrained. So maybe it has learned something that doesn't generalise.

I have a weak hypothesis that I want to test. The decomposition discovers at least in some parts something meaningful but it's a mechanisms for writing for some tasks into the thruth direction. However, on the held out tasks this mechanism doesn't fire. If this would be the case it would be a great finding. If we subscribe to the mechanism idea then I think we should also reconsider which layer's output matrices we decompose. The layers writing into the truth direction should come before the ones where the activations carry the truth signal. I'm not sure though, if that is what the decomposition finds, maybe it also finds something else that uses the truth information. The uth paper argues that many truthrelated features are written an a universal truthfullness direction, but reading this direction then it's less clear what kind of truth or un-truthfullness triggers the direction.

So I would try out some experiments for this and also build more probes for the decompositions we already have that include the components themselves. Also it would be nice to see if and which components write into the truth direction.

My ideas would be:
- as I said design other probes that doen't ignore the decompositions information
- do decomposition using only the last k tokens, to focus more on the truth value relevant parts
- find other layers that we might want to use for the decomposition (the layers constructing the truth signal)

I also need some help designing an experiment for the decompositions can be used for more specific truthfullness.

---

26-10-06, Claude Opus 5.5 — *response: clarifications, the truth-writer hypothesis, proposed analyses*

What I have read: INDEX.md; the SUMMARYs of the UTH series, the no-truth baseline (code control), the test-accuracy analysis and the overview of goal; the training loop's target/non-target loss code (`spd/run_spd.py`, `spd/utils/component_utils.py`, `spd/persistent_pgd.py`); `probe_uth.py`; the saved UTH stage 1 results (`~/spd_out/26-10-05_uth_experiments/stage1/results.json`). I renamed your file from `follow_up_experiments.md` to this LOG name. Your text above is unchanged.

Abbreviations used below: **tPD** = targeted parameter decomposition. **CI** = causal importance, the per-token value (0–1) that the CI network predicts for each component. **Δ** = the remainder `W − Σ_c U_c V_c^T` of each decomposed matrix. **UTH** = the Universal Truthfulness Hyperplane datasets. **tiu** = the Truth-is-Universal statements. For a component c of a `down_proj` matrix, `V_c` is its input direction (in the 18,944-dim MLP hidden space) and `U_c` its output direction (in the 3,584-dim residual stream). Its *inner activation* at a token is `a_c = V_c · h`, where h is the MLP hidden vector. The component writes `a_c · U_c` into the residual stream. Runs: UTH `s-d2ded461`, tiu arm A `s-bd23f0d1`, code control `s-b6cce5de`, untrained `s-7fad0c14`.

## 1. Clarification: "decomposes only what the target uses" is wrong as stated

You are right. The sentence describes the paper's *design intent*, not what a trained decomposition does. I wrote it as if it were an outcome. The code shows the trade-off you describe:
- On target batches, Δ is scaled by a random factor in [0, 1] per position (stochastic recon, `component_utils.py`) or by an adversarial factor (persistent PGD). The target output must therefore come out right whatever Δ's scale, which pushes target-relevant computation out of Δ. This is a loss term, not a constraint.
- Importance minimality pulls the other way: few active components per token, on both data streams.
- So a target mechanism that would need many active components can stay in Δ. Its price is target reconstruction loss. Our runs show this directly [verified: numbers re-read from the UTH and no-truth-baseline SUMMARYs]. On UTH, the reconstruction with rounded CIs has KL 0.433 nats, and even all 480 components without Δ give 0.479. A sizeable part of what the target uses is therefore in no component at all. On tiu arm A the same two values are 0.136 and 0.135: better, but not zero.

I will rewrite the sentence in the overview SUMMARY and in the two local self-contained summaries (`scratch/uth_experiment_summary.md`, `scratch/arm_A_experiment_summary.md`). The new wording says that tPD *aims* to decompose only what the target uses, names the two opposing loss forces, and points to target reconstruction loss as the measure of how much stays in Δ.

On "surprising that it works at all": what demonstrably works is the reconstruction. On tiu, about 7 active components per token reproduce five `down_proj` layers well (KL 0.136). Whether these components are *mechanisms* in any interpretable sense is untested in this project. I agree with your summary of the evidence: we have no evidence that tPD recovers truth mechanisms, and no evidence that it can't.

One addition [concluded, from the loss definition]: tPD sees truth only through the KL on next-token predictions. A computation gets decomposed only to the extent that it changes the next-token distribution on the target text.
- Records like "Question: … Answer: Yes." contain a strong signal for *knowing the answer*: at the answer position, the next-token distribution is the model's belief.
C: yes, but it also contains the machanisms for verbalising or querying the truthfullness. The idea is that the model automatically constructs this data without any stimulus.
Answer (26-10-06, Claude Opus 5.5): Agreed that both are in the records. Agreed too that the model plausibly computes truth without a prompt: at the final token of bare tiu statements, with no question asked, a residual-stream probe reads truth at 0.998 [verified: `~/spd_out/26-10-05_no_truth_baseline/followup.out`]. My point is narrower. It is about what tPD's loss rewards, not about what the model computes; see the answer under your next comment.
OUTDATED (26-10-06): the next bullet's "is weak" was untested. Gate G later found that on tiu the next-token output after the statement depends strongly on truth (" Is" p 0.58 after false vs 0.005 after true city statements; output log-prob probe 0.996); in the UTH "Answer: X." format the effect is modest. See entries *analysis script truth_writers.py; first results on tiu arm A* (item 5) and *UTH results (cross-task)* (item 6).
- The signal for *evaluating a given statement*, which is what our last-token probes read, is weak: the truth state at the final "." only affects what the model predicts after the statement ends.
C: the hope as above is that on the last tokens of the sentence, where the truthfullness of the statement gets decided (like Paris is the capital of France / Spain) the model does the computation there for possible future reference.
Answer (26-10-06, Claude Opus 5.5): This is the crux of your hypothesis, so I state it carefully. Our runs train only on the KL of the next-token output (`StochasticReconSubsetLoss` and `PersistentPGDReconLoss` with `output_loss_type: kl`, in all our configs).
- Suppose the model computes truth at "Spain." for future reference, but nothing later in the *training text* reads it. Then that computation changes no next-token distribution in the training data, and tPD pays nothing for losing it.
- If it stays in Δ, Δ's random down-scaling on target batches degrades it without raising the KL. Putting it into components would only add importance-minimality cost.
- So a KL-trained tPD is blind to exactly the computation your hypothesis is about, however real it is [concluded, from the loss definition]. "For future reference" counts only if the future is inside the training text.

Two ways to give tPD a reason to keep it:
1. Text after the statement that depends on its truth: a follow-up sentence, or an appended question ("Is this statement true? Answer:").
2. A loss on intermediate activations instead of the output. The code has `StochasticHiddenActsReconLoss`: the MSE between the decomposed `down_proj` outputs under stochastic masks and the original outputs. It is selectable as a training loss (`spd/losses.py`). The call there passes no position mask, so it would likely ignore `loss_positions` [assumed: read, not tested]. It needs a check or a small change before use.

Gate G (entry *revision after Julian's full notes*) measures how much truth the next-token output at the statement end carries. It therefore tells us how blind a KL-trained run is.
- So even a well-fitting decomposition need not contain a component that writes the truth direction at the last token. Ways around this are listed under *Alternatives* (§5).

## 2. What the UTH inversions say about your hypothesis

Your hypothesis, as I read it: some components write into the truth direction on (some) training tasks, but they don't fire on the held-out tasks. That would explain the cross-task failure. The cross-task numbers fit this for some test datasets, but not for the inverted ones [verified: `stage1/results.json`, last-token logistic regression, test accuracy per test dataset]:

| test dataset | n | trained CIs | untrained CIs | residual L19 |
|---|---|---|---|---|
| copa | 200 | 0.285 | 0.960 | 0.965 |
| story_cloze | 1,000 | 0.401 | 0.939 | 0.959 |
| sciq | 1,000 | 0.438 | 0.836 | 0.842 |
| hellaswag | 1,000 | 0.500 | 0.570 | 0.781 |

- A mechanism that does not fire would leave the probe at chance. hellaswag (0.500) fits that.
- copa, story_cloze and sciq lie 4–6 standard errors *below* chance (treating samples as independent). On these, the trained CIs do contain label information; the probe fit on the training tasks just reads it with the opposite sign. In copa and story_cloze each item comes as a true/false pair that shares its context, so the inversion must come from the part that differs, the ending. This looks like task-specific *content*, not silence [concluded].
C: you are right. I still think that something failing so loudly is an interesting thing to investigate.
Answer (26-10-06, Claude Opus 5.5): Agreed. It costs little extra in the same extraction pass, so I added it as **A6** (entry *answers to Julian's comments; plan for the VM session* at the bottom):
- (a) Which features carry the flipped readout: for each component, compare its true − false mean CI difference on the training tasks with that on copa, story_cloze and sciq. The drivers are components whose sign flips and that carry large probe weight.
- (b) Whether the inversion survives a strongly regularised probe (C = 1e-4) and a probe on binarised CIs (on/off). If it vanishes there, a few overweighted features caused it. If it stays, the features themselves flip.
- (c) Read about 20 copa pairs that the probe gets confidently wrong, to see what their endings have in common.
- One probe detail matters here: for the trained CIs, the logistic regression chose the weakest regularisation in the grid (C = 1.0). Strongly weighted features that work in-distribution and flip on new tasks would produce exactly this [assumed: not tested].

Two corrections to statements in your text, for calibration:
- "Still it seems that the decomposition picks up something truth related": even in-distribution, the trained UTH CIs read truth worse than the untrained ones (held-out split of the training tasks: trained 0.706, untrained 0.794, residual L19 0.850). As you say yourself, this is no evidence of truth-specific learning; the CI network reading truth-laden activations explains it.
- The code control was not "bad at detecting truth" so much as inactive: 1.4 active components per tiu statement, 29% of statements with none. Where its few components were on (affirmative cities, animal classes) it reached 0.95. That fits the same read-through explanation.
C: How exaclty where those probes fitted. A probe fitted on all CI functions doesn't care about the strength of the signal, right? So if that's the case, I think my argument still makes sense.
Answer (26-10-06, Claude Opus 5.5): The probe is fitted per run on the tiu train split (`probe_ci.py`). Each of the 480 CI values is first standardised to mean 0 and variance 1 over the training statements (`StandardScaler`). Then a logistic regression with the default L2 strength (C = 1) is fitted. So you are right: the strength of a signal doesn't matter, because a weak but consistent CI value gets scaled up. Standardisation cannot undo an exact zero, though. So I re-checked the saved features (`~/spd_out/26-10-05_no_truth_baseline/probe/features.npz`, last token, tiu test, 1,794 statements) [verified: computed 26-10-06]:

| run | CI values exactly 0 | statements with all 480 CIs exactly 0 | components ever non-zero | probe acc., all statements | probe acc. on the 1,462 statements where the control has ≥ 1 non-zero CI |
|---|---|---|---|---|---|
| arm A (trained on tiu) | 98.0% | 0% | 385 | 0.991 | 0.991 |
| code control | 99.6% | 18.5% | 66 | 0.654 | 0.656 |
| untrained | 51.4% | 0% | 480 | 0.998 | 0.998 |

The sparsity-matched readout keeps only each statement's k largest CIs. At k = 2 it gives arm A 0.946, untrained 0.926 and code control 0.654; the control rarely has more than 2 active anyway [verified: `~/spd_out/26-10-05_no_truth_baseline/followup.out`].

My correction above went too far:
- The all-zero statements are not the main cause. Where the control's components are on, its probe still reads truth at only 0.656. Two CIs of the untrained network read it at 0.926. So with as few active values, the code-trained components carry much less truth than random ones.
- A likely reason [assumed]: the control fires a few generic components (66 ever active, against 385 for arm A) that are nearly the same across statements. Which components are on then says little about the statement.

What your argument shows [concluded, from the numbers above]:
- Training on an unrelated target makes the CIs *lose* truth relative to a random readout. Training on truth statements keeps it, so the target shapes what the CIs carry.
- It does not show that training on truth *adds* truth information beyond a random readout. That question is the arm A vs untrained comparison at matched k: 0.946 vs 0.926 at k = 2, a 2-point gap with one seed each.
- So "the target shapes what the CIs read" is supported; "tPD learned truth" still isn't.

## 3. Layer choice: what the residual sweep already says

Residual-stream probes from UTH stage 1 (last token, logistic regression) [verified: `stage1/results.json`]:

| after layer | 7 | 11 | 15 | 19 | 23 | 27 |
|---|---|---|---|---|---|---|
| in-distribution (held-out split of training tasks) | 0.673 | 0.725 | 0.804 | 0.850 | 0.828 | 0.819 |
| cross-task (test tasks) | 0.585 | 0.651 | 0.768 | 0.806 | 0.771 | 0.741 |

- The largest gain lies between layers 11 and 15: +11.7 points cross-task. The decomposed layers 15–19 add another +3.8 points. So by your argument, layers 12–15 are a natural place to look for writers, and layers 15–19 sit where the signal is already mostly formed [concluded, with caveats].
- Caveat 1: where a probe *reads* truth does not say where it is *written*. A direct attribution is cheap and better (A2 below).
- Caveat 2: we decompose only MLP `down_proj` matrices. The UTH paper probes mainly attention heads. If attention does most of the writing, no choice of MLP layers captures it.
- Caveat 3: the CI network at layers 15–19 reads MLP activations that already carry truth. A decomposition there can contain components that *use* truth (your alternative) as well as ones that write it, and CI probes can't tell these apart. Component-level analyses can (A1–A3).
- Budget: we decompose 5 matrices because the dense Δ tensors OOM'd beyond that (legacy config comment, [assumed]). Moving to earlier layers means giving up 15–19, not adding layers.
C: depending on the results we can also test our budget. Wouldn't the dtype change maybe also change the budget?
Answer (26-10-06, Claude Opus 5.5): OUTDATED (26-10-06): the "Budget" bullet above misreads the config comment. The header of `config_truth_all_tokens.yaml` says that decomposing *all 28 layers × 7 projections* OOM'd, and that layers 15–19 `down_proj` are "an unvalidated guess". It does not say that more than 5 matrices OOM. So 5 is a choice, not a measured limit, and adding layers is possible.

And yes, the dtype change matters. Loading the frozen model in bf16 cut peak memory on the H100 from 49–54 GB to ~27 GB at batch 16 [verified: dtype trials of 26-10-05, `convos/julian/26-10-05_no_truth_baseline_SUMMARY.md`]. That leaves about 50 GB free on an 80 GB H100. An extra `down_proj` matrix adds:
- its components: 96 × (18,944 + 3,584) ≈ 2.2M parameters, negligible;
- its dense Δ: 68M values;
- the activations of the masked forward passes.

I'd guess 1–3 GB per matrix [assumed: unmeasured], so 10 or more matrices may fit. A 150-step trial with, say, 10 matrices would measure it in minutes. Training needs the H100 either way: on a 24 GB RTX 4090 the bf16 model alone takes ~15 GB (7.6B parameters × 2 bytes).

## 4. Proposed analyses on the existing checkpoints (no new training)

One script, one forward pass per dataset. Runs: UTH `s-d2ded461` (cross-task) and tiu arm A `s-bd23f0d1` (the better-fitting decomposition; in-distribution only, or affirmative → negated). Baselines: untrained `s-7fad0c14`, and the code control for tiu. Truth direction **w**: the mass-mean direction (mean(true) − mean(false)) of the residual stream after layer 19 at the last token, fit on the training split. Mass-mean is primary because it is the more natural "direction written into" (a logistic-regression weight vector also bends away from high-variance noise directions); the logistic-regression direction is secondary.

- **A1. Static alignment.** Cosine of each output direction `U_c` with **w**, for all 480 components. Reference: the untrained `U` (random) and random directions. Needs only the checkpoint and **w**, no data pass. Answers "are there components whose output points along truth?" But it says nothing about whether they ever write differently for true and false.
C: we might want to include multiple w's. Maybe they don't write to w but something narrower that then gets used for w. My idea would be to also check task or domain depended "overfitted" diretction
Answer (26-10-06, Claude Opus 5.5): Agreed. A1, A2 and A5 will use a set of directions instead of one:
- the global mass-mean direction (all training data), at layers 19 and 23 and right after each decomposed layer;
- one mass-mean direction per dataset (tiu: the 10 families, negated ones included; UTH: each training dataset). These are your "overfitted" directions;
- the logistic-regression direction.

Two precautions:
- With many directions, a component's best cosine rises by chance. The untrained decomposition's U and random directions get scored against the same set, so the comparison stays fair.
- Directions fitted on few samples are noisy. Split-half reliability (the direction from half A against the one from half B) shows which directions are stable.

Your other point is that a component may write something narrower that later layers turn into w. That is an *indirect* effect. A1 and A2 measure only direct writes along w, so they would miss it. Indirect effects need a causal measurement: change the component, then measure the change along w downstream.
- For all 480 components at once, a first-order estimate is cheap: attribution patching, i.e. the gradient of the true/false separation along w at layer 23 with respect to each component's mask, from one backward pass per batch.
- Real ablations (B) then check the top candidates.
- I added this as **A7** at the bottom.
- **A2. Where truth is written (per-layer attribution).** The residual stream after layer 19 is exactly the embedding plus the sum of all attention and MLP outputs of layers 0–19. Projecting each term onto **w** at the last token splits the true/false separation along **w** exactly into per-layer, per-sublayer contributions. Report this for training tasks and test tasks, also for **w** from layer 23.
C: yes that makes a lot of sense. I don't understand the part below and the conclusion why this answers the three questions below
Answer (26-10-06, Claude Opus 5.5): The paragraph below was too compressed. Here it is step by step.

1. **The residual stream is a sum.** At the last token, the residual stream after layer 19 is `x = e + Σ_{l=0..19} (attn_l + mlp_l)`. Here e is the token embedding, and `attn_l`, `mlp_l` are what the attention and MLP sublayers of layer l add.
2. **Projection and averaging are linear.** The truth separation along w is `S = mean_true(x·w) − mean_false(x·w)`. It splits exactly into one term per sublayer: `S = S_e + Σ_l S_attn_l + Σ_l S_mlp_l`, where for example `S_mlp_17 = mean_true(mlp_17·w) − mean_false(mlp_17·w)`. Each term says how far that sublayer pushes true and false statements apart along w. The terms add up to S exactly; nothing is left over.
3. **Which layers write truth** (your layer question): the layers with large terms. The result is a bar chart over layers 0–19.
4. **Attention or MLP:** compare the two bars of each layer, `S_attn_l` against `S_mlp_l`.
5. **Components or Δ** (your clarification point): this only concerns the decomposed layers 15–19. There `mlp_l = W h`, with h the MLP hidden vector. By the definition of Δ, `W = Σ_c U_c V_c^T + Δ`, so `mlp_l = Σ_c (V_c·h) U_c + Δ h` exactly. Projected onto w and averaged as in step 2, `S_mlp_l = Σ_c S_c + S_Δ`, with `S_c = mean_true(a_c) (U_c·w) − mean_false(a_c) (U_c·w)` and `a_c = V_c·h`.
   - If `S_Δ` dominates, the truth writing in these layers sits in Δ: tPD did not capture it.
   - If a few `S_c` dominate, those components are the writers, and they become the A5 candidates.
   - This is computed in the unmodified model, where every component is fully on, so no CI value enters. CIs come in later, in A3 and A5, which ask whether the CI network marks the writers as active.
6. **Limit:** this measures direct writes only. Suppose a sublayer writes some other direction v that a later layer turns into w. Then the later layer gets the credit. The indirect part needs A7 (see the answer to your comment under A1).
For layers 15–19, split the MLP term further. In the unmodified model it is exactly `Σ_c a_c (U_c · w) + (Δ_l h) · w`: each component's write along **w**, plus Δ's. No CI enters this, because the original model uses every component in full. This answers three questions at once: which layers write truth (your layer question), whether attention or MLP does it, and whether the truth writing in the decomposed layers sits in components or in Δ (your clarification point, measured).
- **A3. Component-level probes.** Features: (a) inner activations `a_c` (480 per run); (b) CI-gated activations `a_c · CI_c`; (c) per-component writes along **w**, `a_c · CI_c · (U_c · w)`. Same cross-task protocol as before, plus matched sparsity. Baseline: the untrained decomposition's `a_c`, i.e. 480 random projections of the same MLP hidden vectors. Without it, a good probe on (a) would only show that the hidden layer carries truth.
- **A4. Within-task probes on the test tasks.** Five-fold cross-validation inside each test dataset, for every feature set (trained CIs, untrained CIs, residual, A3 features). This separates "no truth information in the features" from "information present, but the readout doesn't transfer". The inversions above predict the latter for copa, story_cloze and sciq.
- **A5. Direct test of your hypothesis.** Rank components by their write separation along **w** on the training tasks (from A2). For the top ones, report activity (fraction of samples with CI > 0.01) and write separation per training dataset and per test dataset.
  - Prediction if your hypothesis holds [assumed]: the top writers are active and separating on some training tasks and inactive on the test tasks.
  - If instead they are active on the test tasks with a flipped or no separation, the hypothesis fails in its "doesn't fire" form.

**B (only if A5 finds candidates). Causal check.** Ablate the candidate components (mask 0, Δ on, so the weight becomes `W − Σ_candidates U_c V_c^T`). Measure the drop in true/false separation along **w** and in residual-probe accuracy at layers 19 and 23, on training vs test tasks. Reference: ablating the same number of random components with matched activity. Only this step would make a "great finding" claim defensible. A1–A5 are correlational.

**C (only if A2 points there). New decomposition** at the layers A2 identifies as writers, with the same settings otherwise.

Cost [assumed: from stage 0 timings]:
- UTH: the stage 1 `features.npz` was not kept locally, so UTH needs a GPU re-extraction. Stage 0 took 7.5 min of feature extraction on an RTX 4090; A2 adds hooks on all 28 layers and A3 needs the component weights, so expect roughly 10–20 min, then CPU probing.
- tiu: runs on the laptop CPU (the earlier tiu probe pass took about 55 min there). The existing tiu `features.npz` holds residual L19 and CIs but no component activations, so it must be re-extracted too.
- A1 alone takes seconds and could run first on the laptop.

## 5. Alternatives worth weighing before any new training run

- If A2 shows that the truth writing in layers 15–19 is mostly in Δ, the UTH decomposition never captured it. Then the poor fit (KL 0.433) is the first thing to fix (larger C, more steps), before moving layers.
- Given the KL point in §1: a target that puts truth *evaluation* into the next-token distribution, e.g. "<statement> Is this statement true? Answer:" with the loss on the last position only. That forces the decomposition to reconstruct the model's own truth judgement. The last-token arm config exists, but it was never run. The probe question would then change: the interesting object is the components that carry the judgement, tested causally, not CI probes.

## Open questions for you

1. Run A1–A5 on both UTH and tiu arm A, or UTH only? I recommend both: tiu arm A is the cleaner decomposition, UTH has the cross-task split.
C: both
2. A1 can run on the laptop right away. A2–A5 on UTH need a rented GPU (an RTX 4090 suffices for extraction). Shall I write the script first and tell you when it is ready for a rental, as in stage 0?
C: run everything on the RTX 4090. The rental is cheap enough that order doesn't matter much.
3. Mass-mean direction as the primary **w**: agreed?
C: It's the best guess for know, but we should be open for more complicated setups (multiple directions are important)

---

26-10-06, Claude Opus 5.5 — *revision after Julian's full notes: last-k, layers, specific truthfulness*

C: (transcribed from chat) I might have forgotton to save the latest changes to my intial convo. Can you go through what you have done and base it on this as the intial convo?

Julian pasted the version of his notes he had meant to save. I replaced the opening text of this LOG with it, verbatim. My entry *response: clarifications, the truth-writer hypothesis, proposed analyses* above was written against the shorter unsaved version. Following the append-only rule, I leave it as it is and record here what changes. The full version adds:
- "it's something we hope it does";
- "leak by chance";
- the point about the UTH paper's universal direction;
- an explicit list of three ideas;
- a request for help designing a "more specific truthfulness" experiment.

## What stays valid from the entry above

- §1 (the tPD clarification): it agrees with the full version ("something we hope it does"). The wording fixes in the overview SUMMARY and the two scratch summaries, and the FEEDBACK entry, stand.
- §2 (the inversions argue against "doesn't fire" for copa, story_cloze and sciq), §3 (the residual layer sweep) and §4 (analyses A1–A5, B, C) stand. Your idea 1, "probes that don't ignore the decomposition's information", is A1–A5. Idea 3, "find the layers constructing the truth signal", is A2.

## "Leak by chance": a correction to the reading of the random baseline

The untrained CI network is not a lucky draw [concluded]. It is a random nonlinear readout (5 × 18,944 MLP hidden dims → 512 → 480) of activations from which truth is linearly decodable. Such a readout keeps most of that information for almost any random draw, as random projections do. Two consequences:
- Re-drawing the untrained baseline should not change the picture.
- The question is not whether the trained CIs leak truth, but whether training makes them carry it *differently*: sparser, more task-specific, causally linked. Hence the component-level analyses.

## Idea 2: decomposition on the last k tokens only

How it works in our code [verified: `spd/configs.py` `LastKPositions`, `prompts_dataset.py`, config comments]:
- `loss_positions: {type: last_k, k}` puts the reconstruction loss and the CI training on the last k real tokens only.
- Every other position runs on the original weights (`route_only_selected_positions`). The decomposition therefore only has to reproduce the computation at those positions, and the sparsity penalty only counts there. This removes the dilution by truth-irrelevant tokens that you describe.
- The tiu last-token arm (`config_truth_last_token.yaml`, k = 1) exists and was never run; UTH would need a new config.
- No mode covers "the answer span of each record": only `all`, `tokens` and `last_k` exist.

The catch [concluded: from the loss definition]: the loss at a position is the KL of the *next-token* prediction there. With k = 1 on a statement ending in ".", that is the prediction of whatever follows the statement (newline, end of text, the next sentence). Truth enters it only insofar as the model's continuation depends on whether the statement was true. If that dependence is weak, a last-token decomposition has nothing truth-related to reconstruct. It may end up small and inactive, like the code control. That outcome is a prediction [assumed], and it can be checked cheaply before training:

- **Gate G (frozen model only, no training).** How much does the next-token distribution at the last k positions depend on truth?
  - Probe the last-position output log-probabilities (e.g. over the 1,000 most frequent next tokens) for truth.
  - On paired data (copa, story_cloze), also measure the mean KL between the true and the false version.
  - If the output barely depends on truth, a last-k decomposition gets no truth pressure, and the next variant is the better use of a GPU run.

- **Variant with a truth question.** Append a question so the model's truth judgement *is* the next token: "<statement>\nIs this statement true? Answer:" with `last_k`, k = 1. This keeps your idea (loss only where truth matters) and guarantees that the loss contains the truth judgement.
  - Cost: the decomposed mechanism is "answer a truth question", which may differ from the passive truth representation the probes read. Both are legitimate targets.
  - For a first positive result I would bet on the prompted one: it is the case in which truth causally shapes the output tPD reconstructs [assumed].
- Training note: with last_k, a batch of 16 has 16 × k loss positions instead of ~16 × 47 (UTH) or ~16 × 10 (tiu). The gradient is noisier; the last-token arm was flagged as needing its own tuning in `convos/julian/26-10-02_training_run_SUMMARY.md`.
C: as I said above. My hypothesis is that the truth computation and signal is always present but may be not used for the output.
Answer (26-10-06, Claude Opus 5.5): Understood, and it changes my recommendation for a new decomposition. Suppose the truth signal is always computed but not used for the output. Then a KL loss gives tPD no reason to decompose it, whether the KL is on all tokens or only on the last k (answer under your second comment in §1). Plain last-k would remove the dilution, but it would leave nothing truth-related to reconstruct. Gate G tests your premise directly: if you are right, it should find little truth in the next-token output at the statement end. The training setups that fit your hypothesis are a hidden-activation loss at the last k positions, or text after the statement that uses its truth. The rephrased Q4 at the bottom lays out the choice.

## The universal direction and "more specific truthfulness"

I read your request like this [assumed: please correct]. A probe on the universal truth direction says *that* something is false, not *what kind* of falsehood triggered it. Can a decomposition say which kind? If many truth-related features write into one shared direction, the writers are where the specificity lives. That is your hypothesis turned into an experiment design.
C: yes, but I'm not strongly sold. It's just some idea that might be partically true.

One point first [concluded]: probe accuracy cannot show that a decomposition is "more specific" than the residual stream.
C: true. what I mean is more to find out how these different truth values work together. The uth paper argues that other papers have just over fitted. But most likely the mechanics behind truth is more complex than just one direction. I just want to see if there's something that can be found out with tPD that can't be done with probes. I don't know yet, what that actually is, so that's why I need some experiments to shape my understanding.
Answer (26-10-06, Claude Opus 5.5): Then I'd run step 0 below as an exploration, not as a test of "domain-specific writers". These are the things tPD can give and a probe cannot [concluded: a probe yields a direction in activation space; a component is a piece of the weights, with an input side, an output side and a causal role]:
- **Shared computation vs shared readout.** Two kinds of truth can be readable along the same direction (a probe transfers between them) and still be written by different weights, or by the same ones. The overlap of writer components across kinds tells which. This bears directly on the UTH paper's argument that other probes overfitted: a universal direction is compatible with kind-specific writers.
- **Negation:** whether negated statements reuse the affirmative writers with flipped sign, or have writers of their own.
- **Causal reach:** ablate a component and see which kinds of truth disappear.
- **What drives a writer:** a component's input direction V_c can be traced back to the tokens and earlier features that activate it. A probe direction has no such input side.

The first cheap picture to shape intuition is a kinds × components matrix: activity and write separation along w per dataset, clustered. It comes out of the same extraction pass as A1–A5.
The residual stream contains everything a probe on components could read, including the statement's domain or error type, which is trivially decodable. What a decomposition can add is *causal* specificity: a component set whose removal kills the truth signal for one kind of statement and leaves the others intact. So the success criterion should be a double dissociation under ablation, not a probe score.

Design, in three steps, each gated on the previous one:

1. **Step 0: per-domain writers on tiu arm A (existing checkpoint `s-bd23f0d1`, laptop CPU).** tiu already has kinds:
   - factual recall: cities, element symbols, animal classes;
   - translation: Spanish → English;
   - numeric comparison: larger_than, smaller_than;
   - negated versions of four of these.

   Run A2/A5 per domain: for each domain, rank components by their true/false write separation along **w**. Then compare domains:
   - the overlap of their top writer sets;
   - the correlation of their per-component separation vectors.

   Noise floor: the same comparison between two random halves of one domain. Reference: the untrained decomposition.

   Possible outcomes:
   - shared writers (one universal mechanism);
   - domain-specific writers (your hypothesis);
   - negation reuses the same writers with the sign flipped;
   - nothing beyond the untrained reference.

   tiu arm A fits much better than UTH (target KL 0.136 vs 0.433), so this is the fairest first test.
2. **Step 1: causal double dissociation (same checkpoint).** Ablate domain X's top writers (with Δ on) and measure the drop in true/false separation along **w** and in probe accuracy at layers 19 and 23, for domain X and every other domain. Reference: ablating random components with matched activity. Success: X's writers hurt X clearly more than other domains, beyond the random-ablation spread, for at least two domains in both directions.
3. **Step 2 (only if steps 0–1 show specificity): purpose-built data.**
   - Minimal true/false pairs (one token differs) for kinds of falsehood that tiu lacks:
     - arithmetic ("17 + 5 = 23");
     - temporal order;
     - contradiction with an earlier sentence in the same text (consistency rather than world knowledge);
     - optionally, statements the model can't know (made-up entities with assigned labels), to separate "the model believes it" from "it is true".
   - Train a new decomposition on all kinds, ideally with the truth-question target above, and repeat steps 0–1.
   - This is a larger project: data generation plus at least one H100 run, about 40 min at the UTH config's cost.

Other readings of "more specific truthfulness" lead to different designs:
- the model's belief vs. the statement's truth;
- truthfulness of the model's own generated answers (lying) rather than of statements it reads.

Tell me if you meant one of these.

## Revised order of work (proposal)

1. **tiu arm A, laptop CPU** (no rental). One extraction pass serves several analyses:
   - A1 alignment;
   - A2 per-layer attribution (also answers the layer question for tiu);
   - A3/A5 component analyses per domain, i.e. step 0;
   - gate G on the last-position output (needs the full model and LM head, so expect about the earlier 55 min, or more).
2. **Step 1 ablations** on tiu, if step 0 finds domain-specific writers.
3. **UTH on a rented RTX 4090**: A1–A5 cross-task, plus gate G on copa/story_cloze pairs.
4. **Then decide on new training:**
   - writer layers from A2;
   - target: last-k if gate G passes, otherwise the truth-question variant;
   - or step 2 data.

## Open questions (replacing those of the entry above)

1. Is my reading of "more specific truthfulness" (which *kind* of falsehood triggers the universal direction, made causal via writer components) what you meant?
2. Start with tiu on the laptop (step 1 of the order above) before renting for UTH?
C: I want to do experiments on vm
Answer (26-10-06, Claude Opus 5.5): Noted: everything, tiu and UTH, runs on a rented RTX 4090, nothing on the laptop. The revised order is at the bottom.
3. Mass-mean direction of the layer-19 residual stream as the primary **w**: agreed?
4. For a new decomposition: plain last-k first (if gate G passes), or straight to the truth-question variant?
C: sorry, I don't know what you mean by this
Answer (26-10-06, Claude Opus 5.5): That was unclear; it is rephrased at the bottom (entry *answers to Julian's comments; plan for the VM session*, "Q4 rephrased").

---

26-10-06, Claude Opus 5.5 — *answers to Julian's comments; plan for the VM session*

Context: I re-read this whole LOG, including Julian's new `C:` comments, and answered each in place beneath it. I also checked `probe_ci.py`, `spd/losses.py`, `spd/metrics/hidden_acts_recon_loss.py` and the config headers, and recomputed CI statistics from the saved tiu features (`~/spd_out/26-10-05_no_truth_baseline/probe/features.npz`). This entry collects the decisions and the resulting plan.

## Decisions from the comments

- Analyses on **both** decompositions, UTH `s-d2ded461` and tiu arm A `s-bd23f0d1` [decided: C: comment, "both"].
- **Everything runs on a rented RTX 4090**, nothing on the laptop. The order of work doesn't matter for cost [decided: C: comments, "run everything on the RTX 4090", "I want to do experiments on vm"].
- **Truth direction:** the mass-mean direction is the primary guess. Multiple directions matter, including task- or domain-specific "overfitted" ones [decided: C: comments under open question 3 and under A1].
- "More specific truthfulness" is **exploratory**: how different kinds of truth work together, and what tPD can show that probes can't. It is not a test of one hypothesis [decided: C: comment in *The universal direction and "more specific truthfulness"*]. Answer in place: the shared vs separate writers, negation, causal reach, inputs of a writer.

## What changed in my reading (details in the in-place answers)

- **The code control is evidence after all, but for a narrower claim.** Probes standardise their features, so a weak signal doesn't matter (Julian's point). On the statements where the control's components are on, it still reads truth at only 0.656, against 0.926 for the untrained network's top 2 CIs per statement [verified: recomputed 26-10-06, and `followup.out`]. So training on an unrelated target removes truth from the CIs. Training on truth keeps it, but adds only 2 points over a random readout at matched sparsity (0.946 vs 0.926 at k = 2, one seed each).
- **Under Julian's hypothesis, KL-trained tPD is blind to the truth computation** [concluded, from the loss definition]. The hypothesis says the signal is always computed but not necessarily used for the output. A computation that changes no next-token output in the training text costs tPD nothing to lose. This affects the new-training choice (Q4 below), not the analyses on the existing checkpoints.
- **The 5-matrix budget was my misreading** (OUTDATED marker placed under the budget bullet in §3). The OOM was for all 196 matrices; 5 was a guess. The bf16 runs peak at ~27 GB on the 80 GB H100, so more layers can probably be decomposed at once. A short memory trial would settle it.

## Analysis set for the VM session (one script, existing checkpoints, no training)

Runs: tiu arm A `s-bd23f0d1` and UTH `s-d2ded461`; untrained `s-7fad0c14` as the baseline for both; code control `s-b6cce5de` on tiu. Direction set **D**: global mass-mean directions (layers 19 and 23, and after each of layers 15–19), one mass-mean direction per dataset with split-half reliability, and the logistic-regression direction.

- **A1** Static alignment: cosine of every output direction `U_c` with each direction in D. Reference: untrained U and random directions, scored against the same D.
- **A2** Exact per-layer attribution of the true/false separation along each global direction: attention vs MLP for layers 0–19 (0–23 for the layer-23 direction), and components vs Δ in layers 15–19. Plain-language derivation in the answer under A2.
- **A3** Probes on component-level features (inner activations `a_c`, CI-gated activations, per-component writes), with the untrained decomposition as the baseline.
- **A4** Within-task cross-validated probes on the UTH test tasks, for every feature set.
- **A5** The top writers from A2: activity and write separation per dataset, including the kinds × components matrix for the exploratory step 0 (tiu families; UTH datasets).
- **A6 (new)** Why the UTH probe inverts on copa, story_cloze and sciq: which components flip sign, whether a strongly regularised or binarised probe still inverts, and a read of the confidently wrong pairs.
- **A7 (new)** Indirect effects. Attribution patching: the gradient of the true/false separation along w at layer 23 with respect to each component's mask, giving a first-order total effect for all 480 components. It catches components that write something narrower that later layers turn into w.
- **G** Gate on the frozen model: how much the next-token output at the statement's last positions depends on truth (a probe on output log-probabilities; true/false KL on the copa and story_cloze pairs). Under Julian's hypothesis it should come out low [assumed: a prediction].

Cost and memory [assumed]: UTH stage 0 extraction took 7.5 min on the 4090 and peaked at ~20.5 of 24 GB (`convos/julian/26-10-05_uth_experiments_SUMMARY.md`). The new hooks add little memory. A7 needs a backward pass, so it will need smaller batches. My guess is 30–60 min of GPU time for both datasets, plus CPU probing on the VM.

**Afterwards, in a second session, on the candidates found:** B, causal ablations (domain X's writers hurt X more than other domains, against random ablations matched in activity), i.e. step 1 of the specific-truthfulness design.

## Q4 rephrased: how to set the loss of a new decomposition that focuses on the end of the statement

This only matters once we train a new decomposition; nothing needs deciding now. There are three options:
- **(a) Plain last-k:** the same texts, with the KL loss only on the last k tokens. If the truth signal is computed but not used for the output (your hypothesis), this loss contains little truth, and tPD has no reason to decompose it. This is the weakest option under your hypothesis.
OUTDATED (26-10-06): gate G found that on tiu bare statements the next-token output does depend on truth, so plain last-k would get real truth pressure there; the case against (a) holds less on tiu, more in the UTH format. See entry *analysis script truth_writers.py; first results on tiu arm A*, item 5.
- **(b) Truth question:** append "Is this statement true? Answer:" and put the KL on the answer position. The output then depends on truth by construction. What gets decomposed is "answer a truth question". It likely reuses the passively computed signal, but adds the querying and verbalising part.
- **(c) Hidden-activation loss at the last k tokens:** reconstruct the decomposed layers' outputs there, not the next-token output. A computation that is present but unused then has to be kept. Costs: truth is a small part of everything those MLP layers write, so the dilution moves to a different axis; and `StochasticHiddenActsReconLoss` probably ignores `loss_positions` today [assumed], so it needs code work.

My recommendation: decide after G and A2. If G finds little truth in the output, drop (a). Then (c) targets the passive signal you care about, and (b) targets the model's judgement.
C: You made a good point about the loss only seeing the actual tokens and not anything in the hidden layers. I will have to think about this and this might need a different approach. As you suggested let's follow through on the probes first. 

## Open

1. Shall I write the analysis script now (A1–A7, G), to the standard of `probe_uth.py` (a CPU smoke test before the rental)?
C: yes
2. Should the memory trial for more decomposed layers (it needs an H100, not the 4090) wait until A2 has pointed at layers? I recommend waiting.
C: yes

---

26-10-06, Claude Opus 5.5 — *analysis script truth_writers.py; first results on tiu arm A*

Decisions from Julian's last comments [decided: C: comments under *Q4 rephrased* and *Open*]: follow the probes first and postpone new training; write the analysis script now; delay the H100 memory trial until A2 has pointed at layers.

## The script

`spd/experiments/lm/honesty_targeted_decomposition/truth_writers.py` (new, uncommitted), two subcommands:
- `extract <out_dir> <tiu|uth> <run_id>...` (GPU). One forward pass stores, per sample at the last real token:
  - the residual stream after layers 15–19 and 23;
  - per run the CI values and the inner activations `a_c` of all 480 components;
  - the next-token log-probs for the 1,000 tokens most likely at that position;
  - per (dataset, label) group, sums of every attention and MLP output of layers 0–23 and each decomposed layer's Δ part. A2 needs only class means, so these sums are exact.
  
  Then the A7 pass (forward + backward with component masks) runs on ≤ 200 samples per dataset of the fit and test splits.
- `analyze <out_dir> <n_workers>` (CPU): directions, A1–A7, G → `results.json`, `report.txt`, `a5_matrices.npz`.
- tiu gets a `tune` split carved from its train split by subject (20% of subjects), so that logistic-regression C is never chosen on test.
- Built-in checks, all passing on the real runs [verified: VM logs 26-10-06]:
  - the residual stream equals the sum of sublayer outputs (max relative error 0.9–1.4%, bf16 rounding);
  - components + Δ reproduce the original model in the A7 forward (2–3%);
  - the A2 sums match the stored residual stream;
  - Σ_c component writes + Δ = MLP output, per layer.
- A7 linearity check, added after a smoke test showed suspiciously large first-order effects: for the 5 components with the largest estimated effect per run and split, the separation is recomputed with the component really ablated.
- Runs: CPU smoke tests on the laptop (4 per dataset), then a GPU smoke test and the full runs on Julian's rented RTX 4090 (48 GB). The tiu extraction took 2.4 min. Outputs: `~/spd_out/26-10-06_truth_writers/tiu/` (copied from the VM).

## tiu results (arm A `s-bd23f0d1`, code control `s-b6cce5de`, untrained `s-7fad0c14`; one seed each; source `~/spd_out/26-10-06_truth_writers/tiu/report.txt` and `results.json`)

Abbreviations: **S** = true/false separation of a projection onto a unit truth direction (mean over true samples minus mean over false samples). **mm_L19 / mm_L23** = global mass-mean truth direction of the residual stream after layer 19 / 23, fit on the fit split. Components are named `L<layer>:<index>`.

**1. Most truth writing in the decomposed layers stays in Δ** [verified: A2; the identity is exact].
- Along mm_L19 (test split), the MLPs of layers 15–19 write 81% of S. Arm A's components carry 14.5% of S, its Δ 67%. For the code control and the untrained decomposition the components carry ≈ 0% (−0.4%, −0.2%).
- With each dataset's own direction (tune split), arm A's components carry 4–13% of S and Δ 59–73%.
- So the trained decomposition captured a real but minor part of the direct truth writing. This measures your clarification point: most of what writes truth stays in Δ.

**2. Where truth is written (direct writes at the last token) depends on where it is read** [verified: A2].
- Along mm_L19: layers 0–14 5%, attention 15–19 13%, MLPs 15–19 81%. MLPs 18 and 19 alone write 25% and 41%.
- Along mm_L23: layers 20–23 write 84% (MLPs 20–23: 10%, 15%, 26%, 31%), layers 15–19 only 15%.
- Reading [concluded, but only from these two directions]: each layer's truth direction is mostly built by the few MLPs just before it. Truth is apparently rewritten layer by layer rather than written once early and carried. So direct attribution does not answer "which layers construct truth" cleanly. Total effects (A7) and ablations (B) are the better tools.

**3. Two arm A components have large *indirect* effects on the layer-23 truth separation** [verified: real ablation of each component, A7 linearity check; fit-split subsample n = 1,594, S = 62.5].

| component | first-order estimate of the change (−T) | actual change when ablated | direct write along mm_L23 (D) |
|---|---|---|---|
| L17:33 | −17.1 | −19.7 | +0.01 |
| L19:34 | −14.3 | −18.2 | +1.27 |
| L16:69 | −5.5 | −10.5 | +0.14 |
| L18:15 | −4.0 | −8.5 | +0.39 |

- On the test split, L17:33 and L19:34 give −19.0 and −18.8 out of S = 70.1. So removing one component cuts ~27–30% of the separation along the layer-23 truth direction.
- L17:33 writes almost nothing along the direction itself: its effect is indirect, through later layers. About half of its gradient comes from the last token and half via earlier positions (attention); the first position contributes nothing.
- References: the largest effects in the code control and the untrained decomposition are 1.3–1.7 [verified: real ablations], 10× smaller.
- The first-order estimate underestimates the real effect by 15–50% for arm A's components, and is accurate for the reference runs.
- Caveat [assumed, untested]: ablating a component that is active on most inputs might disrupt the computation generically and shrink the separation for unspecific reasons. B needs the controls: random components with matched activity, and the effect on a non-truth separation (e.g. domain).

**4. Writers differ by kind of statement** [verified: A5 activity = share of samples with CI > 0.01; write = S_c as a share of the dataset's S along mm_L19].
- L18:62 is active only on numeric comparisons (larger_than 51%, smaller_than 90%, all 8 other families 0%). L15:84, L16:69 and L18:15 are active on the factual-recall and translation families and ~0–1% on the numeric ones.
- L19:34, the largest direct writer, is active broadly. Its write is concentrated on negated statements: 29–30% of S for neg_element_symb and neg_sp_en_trans, 7–9% for neg_animal_class and neg_cities, ≈ 0 for affirmative cities and animal_class.
- Correlation of the per-component write vectors between datasets: arm A 0.50, untrained 0.26, code control 0.19. Split-half noise floor 0.98–1.00. So datasets genuinely differ in which components write. Arm A shares more writers across datasets than random components do, but far from all.
- Caveat: a component can be domain-specific because it encodes the topic (numbers vs entities), not truth. Its truth-relatedness comes from S_c; whether it matters causally per domain is the step 1 double dissociation (B).

**5. Gate G: on tiu, the output after the statement does depend on truth** [verified: G; mean next-token probabilities on test pairs]. This contradicts my prediction and, for affirmative statements, your "not used for the output" premise.
- After a false city statement the model predicts " Is" with probability 0.58 (after the true version 0.005): it starts to question the statement. After a true one it continues with " It" (0.33 vs 0.013). After a false larger_than statement " (" has 0.95 (true 0.39).
- Paired KL(true ∥ false) at the last token: cities 3.89, larger_than 2.39, animal_class 1.12 nats. The baselines (two different true items of the same family) are 1.39, 0.40 and 0.62.
- Negated statements are weaker: neg_cities 0.71 against a baseline of 1.09.
- A probe on these output log-probs reads truth at 0.996.
- Caveat: the paired KL also contains the content difference of the last word; the token shifts above are clearly truth reactions, though.
- Consequence for Q4 [concluded]: on tiu, a plain last-k decomposition would get real truth pressure from the KL. The case against option (a) is weaker than I argued, at least for affirmative statements.

**6. Uninformative here, as expected:**
- A1: output directions are barely more aligned with the global truth directions than null directions are. For arm A, max |cos| along mm_L19 is 0.082, against a null median of 0.057 (p = 0.03, one of 7 global directions tested). Arm A reaches p < 0.05 for 5 of the 10 per-dataset directions, untrained for 1, the code control for 0.
- A3/A4: every feature set built on the ungated inner activations reads truth at 0.997–0.998, including the untrained and code-control decompositions. The hidden layer carries truth, and the probes can't tell decompositions apart. CI probes: arm A 0.991 vs untrained 0.997.
- A6: no inverted datasets on tiu.

## Next

- The UTH run (cross-task; A4, A6 and G on copa and story_cloze pairs) is extracting on the VM.
- Then B on tiu: ablate L17:33, L19:34, L18:62 and the per-domain writers, with the controls above. This is the causal double dissociation (step 1).

---

26-10-06, Claude Opus 5.5 — *UTH results (cross-task)*

UTH decomposition `s-d2ded461` and the untrained `s-7fad0c14`, one seed each. Extraction 23 min on the RTX 4090, analysis 6 min on the VM's CPUs. Source: `~/spd_out/26-10-06_truth_writers/uth/report.txt` and `results.json`; all files copied from the VM, checksums match. Splits: fit/tune = training tasks, test = the 8 held-out-task datasets (cross-task). Abbreviations as in the tiu entry.

**1. Components carry even less of the direct truth writing than on tiu** [verified: A2]. Along mm_L19, the UTH components carry 6.6% of S (fit) and 8.4% (test cross-task); Δ carries 61% and 59%. Untrained components carry ≈ 0. With each training dataset's own direction, the components carry −1% to 8%.

**2. The cross-task failure is a readout that does not transfer, not missing information** [verified: A3, A4].
- Cross-task probe, mean over test datasets: trained CIs 0.490 (chance, as in stage 1); untrained CIs 0.717; residual L19 0.807; trained inner activations 0.773.
- Probes fit *within* each test dataset (5-fold CV, pair members in the same fold) on the trained CIs: copa 0.920, story_cloze 0.920, sciq 0.784, hellaswag 0.666. Untrained CIs: 0.945, 0.998, 0.890, 0.896.
- So the trained CIs do carry truth on the held-out tasks, only less than the untrained ones, and the readout learnt on the training tasks doesn't carry over. This is the outcome the inversions predicted (§2 of the first entry).

**3. The inversions are an overfit probe on a few components** [verified: A6]. This is the drill-down Julian asked for.
- With the selected C = 1.0, accuracy is copa 0.300, story_cloze 0.395, sciq 0.447. With the strongest regularisation (C = 1e-4): 0.77, 0.765, 0.66. With binarised CIs: 0.52–0.66, nowhere inverted.
- Correlation between the per-component standardised true − false differences on the training tasks and on the test dataset: copa +0.24, story_cloze +0.13, sciq −0.02. What separates true from false differs by task.
- The inversion is driven by a few heavily weighted components whose difference flips sign: L15:10, L17:47 and L17:26 for copa and story_cloze (the same three); L15:21 for sciq.
- Reading [concluded]: the selection picked the weakest regularisation because it fits the training tasks best. That gives large weights to components whose relation to truth is task-specific. Nothing deeper than overfitting is needed to explain the loud failure.

**4. Two UTH components have large causal effects on the layer-23 truth separation, larger on the held-out tasks** [verified: real ablation, A7 linearity check].

| component | fit: estimate / actual change (S = 22.6, n 5,734) | test: estimate / actual change (S = 13.0, n 1,600) |
|---|---|---|
| L16:5 | −3.2 / −4.5 | −6.5 / −5.9 |
| L15:10 | −1.6 / −5.8 | −4.2 / −5.8 |
| L18:36 | −1.9 / −1.9 | −1.3 / −1.6 |

- Each of L16:5 and L15:10 removes ~45% of the cross-task separation along the layer-23 direction. Untrained: at most 0.28.
- L15:10 is also the top inversion driver for copa and story_cloze. It matters for truth on the held-out tasks, but its CI relates to truth differently there than on the training tasks. That makes it the best single candidate for your "loud failure".
- For L15:10 the first-order estimate on fit is far off (−1.6 vs −5.8). The real ablation is what counts.
- Same caveat as on tiu: unspecific disruption is not excluded until B's controls are run.

**5. Writers per dataset** [verified: A5]. Between-dataset correlation of the per-component write vectors: trained 0.26, untrained 0.20, split-half floor 0.90. Weaker sharing than tiu arm A (0.50). Several top writers are active only on some tasks (e.g. L17:8: median activity 0, maximum 0.14).

**6. Gate G on UTH: the output after "Answer: X." depends on truth less than on content.**
- Paired KL at the last token against the baseline (two different true items): copa 0.38 vs 0.30, story_cloze 0.60 vs 0.44, sciq 1.97 vs 3.92, nq_re_long 1.56 vs 4.32.
- The baseline is large where items differ a lot, so the comparison is crude. Probe on the output log-probs: cross-task 0.725 (residual 0.790 pooled).
- So in the UTH format truth has a modest effect on the next-token output. On tiu bare statements the effect is strong (" Is" after false statements). The format matters for a future last-k decomposition.

## Overall reading after tiu and UTH [concluded, one seed per run, mostly correlational]

- Both trained decompositions contain a few components whose ablation removes a large share (~30% on tiu, ~45% cross-task on UTH) of the true/false separation along the layer-23 truth direction. Their direct writes along the direction are small, so the effect is indirect. Nothing comparable exists in the untrained or code-control decompositions. This is the first result in the project where tPD's components look specifically involved in truth. It needs B (matched random ablations; the effect on non-truth separations; per-domain dissociation) before we claim it.
- Most of the direct truth writing in the decomposed layers remains in Δ (67% vs 15% components on tiu, 60% vs 7% on UTH).
- Writers differ by kind of statement on tiu (numeric vs factual vs negated). On UTH the probe failure across tasks comes from task-specific readouts of components that do matter (L15:10).

## Next (proposal)

B on both decompositions, on the laptop CPU or the next rental:
- candidates tiu L17:33, L19:34, L16:69, L18:62 and the per-domain writers; UTH L16:5, L15:10, L18:36;
- measures: S along mm_L23 and mm_L19, per dataset;
- controls: (i) 20 random components with matched mean activity per candidate; (ii) the separation of a non-truth label (domain/family; on UTH, dataset identity), to tell truth-specific effects from general disruption; (iii) on tiu the double dissociation between numeric and factual writers.

On a GPU this takes minutes (a forward pass per ablation), so it fits into `truth_writers.py` as a third subcommand.

---

C: (transcribed from chat) yes

---

26-10-06, Claude Opus 5.5 — *pkill rule added to CODING.md*; maintenance session (sync), not the analysis

During this topic, an agent killed its own shell with `pkill -f "truth_writers[.]py extract"` followed by a relaunch in the same Bash call (FEEDBACK.md, 26-10-06). The mechanism: the whole `cmd1; cmd2` string is the shell's command line before `cmd2` starts, so the relaunch's plain `truth_writers.py extract` matched the pattern and the shell died. I demonstrated it with `pgrep` [verified: 26-10-06, the pattern matched two shells before the target process existed]. Julian's "yes" above approves adding the sentence to `docs/CODING.md` (*Kill by a bracket pattern or PID*): it applies to local shells too, the bracket protects only a command line that names the target nowhere else, so kill in a call of its own or by PID. Done; the FEEDBACK entry is marked adopted.
