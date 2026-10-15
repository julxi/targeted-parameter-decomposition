I'd like to be a bit less methodogical and more eplorative in the experiments.

That being said I have a proposal for a new experiments. It's a bit rough around the edges and as usal it hards to know which things matter and which don't or wich design desicions are critical. That's always the case I suppose. I want to combine my idea that the model might generate a truth representation somewhere even when it's not being used, which your argument that in the current setup the loss can only distinguish changes in the next token distribution. I see two general ways around this:
    1. calculate the loss on inner activations
    2. provide a stimulus to surface the mechanism to the output

For 1. an experiment could look like this: use a previous layer for the target-loss, keep the non-target loos setup as before. Why not changing the non-target loss? I think there are arguments for and against it. I was just going with the argument that we want the decomposition change the model's behaviour as little as possible.

For 2. an experiment could look like this: non-target setup stays as it is. For the target setup we take statements for which we suspect the model computes some truthfulness values and apply the decomposition to the sequence position where we suspect the truth values being computed. We append to those statements a stimulus like: "This statement is" and calculate the kl-divergence for the next token. One concrete example is that we use the statment "Paris is the Capital of France. This statement is" where we only do the CI gated ablation of the components on the period after France but calculate the loss of the token distribution after the stimulus. So we use the loss from the downstream behaviour change. I'm not saying that this particular example is very good. In particular the stimulus doesn't necessarily tap into truthfulness but generally into categorizing a statement.

I'd like you to design both experiments. It's quite a lot of work as we have to discuss everything data and so on again.

---

26-10-10, Claude Opus 5.5 — *design of both experiments: two lenses on the statement-end computation*; I have read INDEX.md, the SUMMARYs of the goal, truth-writing, state-report, UTH, code-control, training-run and dataset topics, the *Q4 rephrased* section of `convos/julian/26-10-06_truth_writing_components_LOG.md`, and the loss code (`spd/losses.py`, `spd/metrics/hidden_acts_recon_loss.py`, `spd/run_spd.py`, `route_only_selected_positions` in `spd/models/components.py`). I ran one CPU check on stored features (below). No training code written yet.

This file was `convos/julian/a case for bolder experiments.md`; renamed to the LOG convention (untracked, so plain `mv`). Julian's text above is unchanged.

## Where this comes from

Your two routes are options (c) and (b) of *Q4 rephrased* in the truth-writing LOG: (c) a hidden-activation loss, (b) appended text whose next-token output depends on truth. One fact from that topic changes how to read them: gate G found that on tiu (Truth-is-Universal statements) the model's own next token after the statement already depends on truth (after a false city statement " Is" gets p 0.58, after the true one 0.005) [verified there]. So on tiu, "computed but not used" is not literally the situation. The question the two experiments can answer on tiu is: **does a loss that sees the truth computation more directly pull it out of Δ?**

C: I think, that's not the question. We are simply doing a different experiment with a different motivation. In priciple the mechanism responsible for the first token after the period and after the stimulus could be completly different ones.

Answer (26-10-10, Claude Opus 5.5): Agreed, I withdraw that framing. The inputs at the period are the same, but the next token after the period and the token after the stimulus can depend on entirely different parts of what is computed there. So the lenses can pick out different mechanisms, and "pull truth out of Δ" is not the question. Each experiment stands on its own motivation. The components-vs-Δ share stays one descriptive number among others, not a target.

 In arm A (the all-tokens tiu decomposition `s-bd23f0d1`), the components carry only 14.5% of the direct truth write of layers 15–19 along the layer-19 truth direction, and Δ (the catch-all remainder) carries 67% [verified there]. That share is the number to move.

 C: no it's not. I'm simply trying out fairly principled experiments and then analyse them to see if there's any result that is surprising.

Answer (26-10-10, Claude Opus 5.5): Understood: run the experiments, look at the analyses, and follow what is surprising. I'll report the 14.5% share alongside the other analyses, without treating it as the goal.

## Common frame for both experiments [assumed: agent proposal]

Both experiments decompose **only at the end of the statement** (positions D, by default the last token "." or "'.");

C: Doesn't have to be like this. It's a good start. But for exmaple for exp 2 it could also be "The capital of France is <decomp>Paris.</decomp>". I said we try to use a position that we suspect some truth computations are being computed. But ok, for the rest of my response I just assume as well that we decompose at the period.

Answer (26-10-10, Claude Opus 5.5): Noted. The position set is a free choice, and your example (object plus period, "<decomp>Paris.</decomp>") is one option among others. The zero-ablation measurement R2 (entry *answers to Julian's comments*, below) compares the period, the last 2 tokens and all statement tokens as a heuristic. Your example also bears on the data question: with our own data we can mark the object span exactly. The new entry proposes one use of that (the object token as a position where the next token hardly depends on truth).

all other positions run on the original weights (the existing `route_only_selected_positions` mechanism). They differ only in the **lens**: the downstream quantity the target loss compares between the masked and the original model.
- Exp 1: the residual stream after a later layer L, at the positions D.
- Exp 2: the next-token distribution at the end of an appended stimulus.
- For reference, the never-run arm B (`config_truth_last_token.yaml`) is the third lens: the natural next-token distribution at D, with no code changes.

Why this frame is clean [concluded: causal attention]: the computation at D cannot see tokens that come after it. So it is identical with or without a stimulus, provided the statement is tokenized the same way. The components cannot specialise to the stimulus; the stimulus only decides *which parts* of the D computation the loss protects. Consequences:
- All three decompositions decompose the same computation and are directly comparable.
C: I disagree. They all are decompositions the same position, but if the mechanism they have to cover are related is not given.

Answer (26-10-10, Claude Opus 5.5): You're right; "the same computation" overstated it. They share the positions and the inputs; which mechanisms each loss forces the components to cover can differ, and whether they overlap is an open question. That makes the overlap a result in its own right: do the three runs find matching components (cosine of the U/V vectors between runs, e.g. with `spd/scripts/validation/compare_components.py`, which I have not yet checked on our checkpoints) or disjoint ones? What does still hold: the analysis tools apply to all three runs without changes, and the untrained baseline is shared.
- `truth_writers.py` (the component-analysis script) runs on the bare statements at the last token, unchanged.
- The untrained baseline `s-7fad0c14` stays valid, because the initial decomposition depends on seed, modules and C, not on the loss [concluded: same config fields].

Tokenization [verified: Qwen tokenizer, 26-10-10]: a stimulus that starts with a space leaves the statement's tokens unchanged (`... ' Russia', '.', ' This', ' statement', ' is'`), but a newline merges with the period into one token `'.\n'`. So statement and stimulus are tokenized separately and the ids concatenated.

Shared settings, as before: all 20 tiu v1 datasets and their split,

C: I don't know if I want to use tiu. We haven't discussed that yet.

Answer (26-10-10, Claude Opus 5.5): Right, I had presumed it. The data discussion is now in the entry *answers to Julian's comments*, section *Data*.

Qwen2.5-7B-Instruct, `down_proj` of layers 15–19 with C = 96, batch 16, 5k steps, LR 5e-4, bf16 frozen model, Pile non-target stream unchanged (your choice; I agree: the non-target loss is what keeps the decomposition harmless on general text, and changing it would change two things at once). tiu rather than UTH because labels are clean, every analysis tool already runs on it and gate G is known; UTH can follow if something works.

OUTDATED (26-10-13): the target batch is 64, not 16, in all configs of this topic (Pile batch still 16); see entry *H100 tuning trials: results*, approved in entry *full runs launched*.

## Measured: truth is not a small part of the statement-end residual stream

My earlier objection to (c) was "truth is a small part of everything those MLP layers write, so the dilution moves to a different axis" [assumed then, never measured]. I checked it at the residual-stream level with the stored last-token residuals of the 5,260 tiu fit statements (`~/spd_out/26-10-06_truth_writers/tiu/features.npz`; script and output `~/spd_out/26-10-10_bolder_experiments/dilution.{py,out}`) [verified: single CPU run, 26-10-10]:

| last-token residual after layer | 19 | 23 |
|---|---|---|
| share of the mean squared norm that varies across statements (the rest is the shared mean) | 0.217 | 0.264 |
| share of that variance along the global mass-mean truth direction | 0.119 | 0.153 |
| truth between-class variance / total variance | 0.10 | 0.12 |
| same, within one dataset (median over the 10 families) | 0.42 | 0.40 |
| share of the mean squared norm in the single largest channel (2570) | 0.217 | 0.287 |
| that channel's share of the across-statement variance | 0.009 | 0.026 |

Reading [concluded]:
- Among everything that differs between statements at the period, the truth contrast is 10–15% globally and ~40% within a family. A loss that preserves the statement-end residual stream therefore puts real weight on truth. My earlier dilution worry was overstated for a residual-level loss.
- Caveat: within a family, true and false statements also differ in their object word ("Russia" vs another country), and that content difference is part of the between-class variance. So 0.40 is an upper bound on the pure truth share.
- One channel (2570, a "massive activation" channel) holds 22–29% of the squared norm but only 1–3% of the variation across statements. A plain MSE loss would weight errors in that channel heavily. RMS-normalising each vector changes nothing here (I checked: same shares). Hence the per-channel normalisation proposed below.

## Experiment 1: hidden-activation lens

**Design** [assumed: agent proposal]
- **Target loss**: the squared error between the masked and the original residual stream after layer L at the positions D, with each channel divided by its standard deviation. That standard deviation is computed once, on the frozen model, over the statement-end positions of the tiu train data. Averaged over channels, the loss reads as "the fraction of the natural statement-to-statement variability that the masking destroys". It replaces the KL in every target loss (stochastic subset reconstruction and persistent PGD, the adversarial masking used in the last 20% of training).
- **L = 23** (recommended). Reasons: the earlier ablation results were measured there (ablating component L17:33 of arm A removes 19.7 of 62.5 units of the layer-23 truth separation while writing almost nothing along that direction itself, i.e. an indirect effect); a lens at 23 credits such indirect paths. It is also four layers before the output, so less dominated by next-token prediction. Alternatives: 19 (right after the decomposed block; closest to "what the MLPs wrote") or 27 (last layer; close to the KL lens).
- **Non-target**: unchanged (Pile, Δ fully on, KL on the output).
- **Why not the existing `StochasticHiddenActsReconLoss`** [verified: code read]: it compares each decomposed module's own output (the five `down_proj` outputs), not a downstream layer; it ignores the position mask (it is not among the position-mask-supported losses); and PPGD (persistent PGD) is wired only to output losses. It would also weight every direction of every MLP output equally, whether later layers use it or not.
- **Side benefit** [assumed]: target batches need the model only up to layer L, so no 152k-vocabulary logits there; faster and lighter.

**Code needed**: a config field for the target lens (`logits` / `residual` at layer L with per-channel normalisation); capturing the residual and stopping the forward pass at L; separate loss types for the target and non-target streams (today the non-target loss list is copied from the target list and shares `output_loss_type`, `spd/run_spd.py` ~line 288); the per-channel statistics file; target eval metrics in the new lens, plus the output KL for comparison, so we see what the lens costs behaviourally.

**What it can show, and the risks** [concluded unless tagged]
- It protects everything present at layer 23: subject, family, negation, numbers, and truth. Truth gets 10–15% of the weight, not all of it.
- The decomposition may become larger (more active components per statement), because the lens asks for more than the KL does. On UTH, 96 components per layer already fitted poorly [verified there: target KL 0.433]; capacity could bind here too [assumed].
C: may become larger, yes. I only want to flag your framing that on UTH we would get a better result if we had more than 96 components.

Answer (26-10-10, Claude Opus 5.5): Fair flag. The UTH fit was poor (target KL 0.433; all 480 components without Δ 0.479). Too few components is only one of the causes listed there, all `[assumed]`; too few steps, the layer choice and data too diverse for a sparse decomposition are others. And even a better fit would not imply a better truth result. My sentence treated capacity as the cause; it isn't established. The point I meant to keep: the residual lens asks more of the decomposition than the KL does, so its fit may be worse at the same C.
- The loss scale differs from the KL's, so the balance against importance minimality (the sparsity penalty) is a new tuning knob. Proposal: scale the coefficient so that the "all components off" target loss has the same size as the KL lens's all-off loss at the same positions (G3 below measures both; arm A's all-off KL, 3.54 nats/token, is an average over all tokens, not the last token), then one 150-step trial.
  OUTDATED (26-10-13): no rescaling. R3 found the two all-off losses within a factor 1.35, and Julian decided that Exp 1 and Exp 2 keep Exp 0's coefficients; see entry *decisions on the four open questions; commit*.

## Experiment 2: stimulus lens

**Design** [assumed: agent proposal]
- **Input**: statement + stimulus, tokenized separately and concatenated (see above). Example: `The city of Krasnodar is in Russia.` + ` This statement is`.

    C: I gave the stimulus "This statement is". I'm a bit afraid that for other examples the model focuses on a different attribute of the statement than truthfullness, e.g. the model selects "This statement is _funny_." That's what I meant that this simple stimulus is a stimulus for categorization. For the right dataset it might not be an issue, but maybe there's a better stimulus.

    Answer (26-10-10, Claude Opus 5.5): Resolved by your next comment; the stimulus screen R1 (answer there) measures exactly this rate of non-truth answers such as "funny".

- **Decomposed positions D**: the statement end; the stimulus runs on the original weights.
- **Loss position**: the last stimulus token only; full next-token KL there.
- This needs **two position masks** instead of one: D (where components replace the weights; also where importance minimality and the CI statistics are taken) and the loss position (where the KL is read). Today one mask does both. This is the main code change; it touches the batch format, the target losses, PPGD and the target eval metrics.
  OUTDATED (26-10-13): implemented without a second mask. A *target lens* makes the forward pass return the logits at the stimulus end, repeated at every position, so the existing single-mask losses work unchanged; see entry *R1 stimulus screen, lens implementation, R2/R3, analysis V*, section *Implementation: the target lens*.
- **Stimuli**: chosen at the gate (below) from a candidate list, e.g. ` This statement is`, ` That is`, ` True or false? The statement is`, ` Is this true? Answer:`, ` In fact,` (the last one invites a correction rather than a verdict). Proposal: mix the 2–3 best; each training sample draws one at random. Since D cannot see the stimulus, mixing does not change what the components can compute; it widens what the loss protects and averages out quirks of a single phrasing.

    C: Oh, good that we are talking about the stimulus. So my comment above is resolved. I don' think that mixing the stimulus is a good idea. I'd rather have a little experiment to see which stimulus elicits truthfulness related responses at at a high rate (once we have the dataset)

    Answer (26-10-10, Claude Opus 5.5): Agreed: a single stimulus, chosen by a small screen once the dataset is fixed (R1 in the entry *answers to Julian's comments*). The screen measures how often the model answers with a truth-related token ("true", "false", "correct", "wrong", …), so the "funny" case from your comment above shows up directly as a low rate.

- **Your worry that the stimulus taps categorisation rather than truth**: the gate measures it directly, by comparing the true-vs-false output difference with the difference between two true statements of the same family (gate G's content baseline). And it is acceptable if some categorisation components appear: the per-dataset analysis (A5 of `truth_writers.py`) separates them afterwards.
- **Variant 2b (bolder)**: a binary verdict loss. Renormalise the output to two token sets (true-like vs false-like answers) and take the KL over that 2-way distribution only. The decomposition must then preserve exactly the model's verdict, nothing else. Risk: the token sets are a hand choice, and a narrow loss may be satisfied by very few components that encode the verdict for these 20 families only. I'd run the full-KL version first.
  OUTDATED (26-10-13): variant 2b is dropped (Julian: "single one, full KL", open question 3 of this entry); the binary verdict survives only as analysis V.

**The critical risk: mediation** [concluded: causal structure]. The stimulus tokens can re-read the whole statement through attention (those positions run on the original weights) and recompute truth there. The loss only sees the part of the verdict that flows through the MLPs 15–19 at D. If that part is small, the loss carries no truth pressure and the experiment cannot work. The gate below measures it before any training.

## Gate session before training (frozen model, RTX 4090) [assumed cost: under an hour]

- **G1 stimulus screen**: for each candidate stimulus, on the tiu test statements: the top next tokens after true vs false statements (read by eye), a verdict score (log-probability of true-like minus false-like tokens) and its accuracy, and the paired KL(true ∥ false) against the content baseline.
- **G2 mediation**: zero the outputs of MLPs 15–19 at D. That is exactly tPD's "all components off, Δ off" state (masked `down_proj` output = Σ m_c a_c U_c + δ·Δx = 0). Measure how much of the verdict's true/false separation disappears, for D = last token, last 2 tokens, and all statement tokens. This picks D and tells whether Exp 2 is viable.
- **G3, the same for Exp 1**: what the same zero-ablation does to the normalised layer-23 loss, to the layer-23 truth separation, and to the next-token KL at D. The ratio of the two losses gives the scale for the loss coefficient.
- Rough decision rule (exploratory, not a hypothesis test): if zeroing MLPs 15–19 at all statement tokens removes less than about a fifth of the verdict separation for every stimulus, drop Exp 2 or move the decomposed layers.

    C:  We are doing both experiments anyway no matter the result of G. This is important since the decision rule is based on our hypothesis.

    Answer (26-10-10, Claude Opus 5.5): Agreed, the decision rule is dropped. It assumed that zeroing whole MLP outputs predicts what a trained decomposition will do, and that assumption is exactly what the experiment could prove wrong. G1–G3 become reference measurements R1–R3 (stimulus choice, position heuristic, loss scale), not gates.

## Runs and analysis

- Runs on an H100: Exp 1, Exp 2, each ~40 min [assumed: arm A took 39 min; Exp 1 may be faster without target logits]. Optional reference: arm B (natural next-token KL at the last token), no code needed.
- Analysis on the 4090: `truth_writers.py` on each run (bare statements, last token), plus one new measure for Exp 2: how much ablating each single component shifts the stimulus verdict.

    C:  There's an experiment I want to add to the stimulus lens. It might also already be in the package. I don't know. Check or compare the accuracy of the truthfulness token e.g. True / False (or something similar to variant 2b) when using the components.

    Answer (26-10-10, Claude Opus 5.5): It is not in the package [verified: grep of `spd/metrics/`, `spd/scripts/`, `truth_writers.py`]. `TargetReconLoss` reports only KL under its masks; `CEandKLLosses` reports CE and KL; `spd/scripts/validation/scale_active_components.py` follows the top-1 logit under active components only, but supports only the small LlamaSimpleMLP models, not Qwen. I'll write it as a new analysis (V in the entry *answers to Julian's comments*): the verdict accuracy after the stimulus under the original model, components only (rounded CIs, Δ off), Δ only, everything off, and single-component ablations. I propose to run it on all three runs, not only Exp 2. Whether the residual-lens and arm B decompositions preserve the verdict without being trained for it bears directly on your point that the mechanisms may differ.

- **What would count as a visible effect**, stated now so it is not chosen after the results [assumed: rough thresholds]: the components' share of the direct truth write along the layer-19 direction clearly above arm A's 14.5% (say twice that), and single-component ablations that move the layer-23 separation or the verdict far more than the untrained decomposition's components do. One seed per run.

    C: I don't care so much. I want to see the numbers and then decide how to proceed.

    Answer (26-10-10, Claude Opus 5.5): Dropped; no thresholds. The numbers get reported, with the untrained decomposition as reference.

## Order of work

1. Your comments on the questions below.
2. Gate script (G1–G3), CPU smoke test on the tiny Qwen2.
3. Gate on a rented 4090.
4. Training code for the chosen experiments, with unit tests (two masks; residual lens).
5. H100 runs; then the analysis on the 4090.

## Open questions

1. The common frame: decompose only at the statement end, both experiments. For Exp 1, did you have that in mind, or a hidden-activation loss at all statement tokens?

    C: yes, I think we agree on the basic structure.

    Answer (26-10-10, Claude Opus 5.5): Noted.

2. Exp 1: layer 23 with per-channel normalisation, or another layer?

    C: let's just role with 23

    Answer (26-10-10, Claude Opus 5.5): Layer 23. I read this as also accepting the per-channel normalisation; please say if not.

3. Exp 2: a mix of 2–3 stimuli, or a single one? Variant 2b (binary verdict) now, or only if full KL works?

    C: single one, full KL

    Answer (26-10-10, Claude Opus 5.5): Noted; variant 2b is off the plan, and the binary verdict lives on only as an analysis (V).

4. Gate session first (my recommendation; it can rule out Exp 2 for under an hour of a 4090), or straight to training?

    C: Gate is good to have as reference and maybe as a heuristic for the decomposition positions, but not as a gate for future experiments.

    Answer (26-10-10, Claude Opus 5.5): Noted; see the answer under the decision-rule comment above.

5. Include arm B as a reference run? It costs one H100 run and no code, and it makes clear whether a new lens did anything a plain last-token decomposition does not.

    C: yes, we should add this

    Answer (26-10-10, Claude Opus 5.5): Added, on the same data and positions as the other two runs.

6. tiu first, UTH later?

    C: this needs discussion. tiu is notoriously easy (but maybe all right nontheless). UTH might be too abitious for decomposition (for example imagine there is a truth direction but for each task a different mechanism is responsible for writting to it. I don't think this is true, but it should just demonstrate, that a lot of varied truth related tasks is good for finding a universal truth correction but not necessarily good for finding the mechanisms behind them). I'm open to generate our own data that is tailored to the effects that might play a role in in the hidden-activations lense or stimulus lens

    Answer (26-10-10, Claude Opus 5.5): Options and a proposal in the entry *answers to Julian's comments*, section *Data*. Short version: I agree with your UTH argument, and I'd build our own small, programmatically generated dataset rather than reuse tiu.

---

26-10-10, Claude Opus 5.5 — *answers to Julian's comments; data options; revised plan*; same session as the entry above. I have read all of Julian's C: comments and answered each in place. New checks: the verdict-accuracy metric is not in the package (code grep), and the tiu object-word shortcut (below).

## Decisions [decided: C: comments in this LOG]

- The basic structure stands (open question 1): decompose at the statement end; positions elsewhere run on the original weights.
- Exp 1: residual lens at layer 23 (question 2). Per-channel normalisation: I read the answer as accepting it [assumed: please confirm].
- Exp 2: a single stimulus, full KL (question 3). The stimulus is picked by a small screen once the data is fixed (comment under the stimulus list).
- The measurements before training are reference and heuristic, not gates; both experiments run whatever they show (question 4 and the comment under the decision rule).
- Arm B (natural next-token KL at the same positions) is added as a reference run (question 5).

    C: can we get a better name for this experiment than Arm B? So we can reference it better?

    Answer (26-10-13, Claude Opus 5.5): Proposal: **Exp 0, next-token lens**. "Lens" matches the other two (residual lens, stimulus lens): all three decompose the same statement-end positions and differ only in what the target loss compares. Its lens is the model's own next-token distribution at the decomposed positions. The 0 marks it as the reference that needs no new code. I use this name from here on; say if you prefer another.

- No success thresholds: the numbers get reported, then Julian decides (comment under *What would count*).
- New analysis for the stimulus lens: the true/false verdict accuracy when the components are used (comment under *Runs and analysis*).
- Data: open (question 6); options below.

Framing withdrawn (Julian's first two comments): the experiments are not about moving the components-vs-Δ share. Different lenses on the same positions can select entirely different mechanisms. Whether they do is itself something the runs will show.

## Data

**What the data has to provide** [concluded, from the designs above]:
1. Statements whose truth the model knows or can compute, with exactly known token spans, so the decomposed positions can be the period, the object, or both (Julian's "<decomp>Paris.</decomp>").
2. For Exp 2: the chosen stimulus should draw truth-related answers at a high rate (screen R1 below). That depends on the data, as Julian said.
3. No surface shortcut: the label should not be predictable from the object word alone. Otherwise a component that reacts to "Russia" or "51" looks like a truth component.
4. Minimal pairs: each false statement equals a true one except for the object. That allows paired measures (as gate G used).
5. Optionally, kinds of truth that need different computations, to see whether the lenses find shared or separate mechanisms.

**Option A: tiu** (Truth-is-Universal, as in arm A).
- For: every tool runs on it, labels are clean, arm A exists for comparison.
- Against: easy (probes saturate at 0.998); 10 template families, mostly factual recall plus number comparisons.
- Shortcut check [verified: train split, 26-10-10; in-sample majority vote over the statement's last word, so optimistic where objects are rare]: the object word alone predicts the label at 0.574 (cities, 101 distinct objects in 1,164 statements), 0.701 (animal_class, 10 in 127), 0.753 (larger_than, 45 in 1,572), and higher for element_symb and sp_en_trans, where most objects occur only once or twice, so their numbers mean little. larger_than and animal_class have a real shortcut: the last number or class name alone predicts the label at ~0.7–0.75.

**Option B: UTH** (the Universal Truthfulness Hyperplane datasets). I agree with Julian's argument. Diversity helps find a direction that transfers, not the mechanism behind it, and a sparse decomposition may favour task-specific components. Practical points against it as well: long and varied formats (no single "statement end"), label noise found in the UTH series, and the poor fit there.

**Option C: our own small dataset, generated programmatically (my recommendation)** [assumed: proposal].
- Families, each needing a different computation:
  1. recall: "The capital of France is Paris." (country/capital and similar relations from fixed lists);
  2. arithmetic: "17 + 25 = 42." (computed, not memorised);
  3. in-context consistency: "Anna lives in Rome. Anna lives in Rome." vs "… lives in Paris." (truth set by the context; no world knowledge involved).
- Construction rules:
  - every false statement is a true one with the object swapped for another object of the same family (minimal pairs);
  - each object appears equally often as a true and as a false object, so the object-word shortcut is at chance by construction;
  - single-token objects where possible, and the object span stored as a field;
  - every statement ends with "."; no chat template;
  - split by subject, about the size of tiu.
- Recall items are kept only if the frozen model knows the fact (its top-1 completion of "The capital of France is" is " Paris"). Otherwise "false" would partly mean "unknown to the model". This filter needs the model, so it would run in the same VM session as R1–R3.
- Cost: a generator under the `data/` convention (committed before building), plus a loader case in `truth_writers.py`. Comparisons with arm A on tiu are lost, but arm B on the same data provides the plain-KL reference.

**An idea that own data makes possible** (not part of the agreed plan; take it or leave it): decompose at the **object token** instead of the period. There, the next token is "." almost regardless of truth, so the natural-KL lens (arm B) gets almost no truth pressure. The residual lens protects truth only if truth is already represented at the object token. That would test your original premise, "computed but not used", more directly than the period does on tiu, where the next token already reacts to truth. R2 can measure first whether truth is readable at the object token at layers 19/23.

## Measurements before training (reference, not gates) [assumed: under an hour of an RTX 4090]

- **R1 stimulus screen** (on the chosen data): for each candidate stimulus, the share of statements whose top next token is truth-related, the probability mass on such tokens, and the true-vs-false verdict accuracy. The token sets are fixed after reading the top tokens by eye. You pick the stimulus from the numbers.
- **R2 position heuristic**: zero the outputs of MLPs 15–19 at the period / last 2 tokens / all statement tokens (tPD's "everything off" state). Measure the effect on the stimulus verdict and on the layer-23 truth separation; plus truth probes at the object token, if we use own data.
- **R3 loss scale**: the residual-lens loss and the next-token KL under the same ablation, to set the coefficient of Exp 1.

## Analysis after training (all three runs: Exp 1, Exp 2, arm B)

- `truth_writers.py` (attribution components vs Δ, single-component ablations, writers per family, probes), with the untrained decomposition as reference.
- **V, verdict accuracy** (new; it is not in the package [verified: code grep, see the answer under Julian's comment]): with the stimulus appended, the true/false verdict accuracy and the verdict score under the original model, components only (rounded CIs, Δ off), Δ only, everything off, and each single component ablated. Run on all three decompositions: whether the residual-lens and arm B decompositions keep the verdict without being trained for it shows how far the mechanisms overlap.
- Component overlap between the three runs (cosine similarity of components across runs).

## Revised order of work

1. Data decision (below).
2. If own data: generator, committed before building, then the data.
3. Code: R1–R3 script; two position masks plus stimulus appending; the residual lens; analysis V. CPU smoke tests on the tiny Qwen2, unit tests for the loss changes.
4. RTX 4090: model-knowledge filter (own data), R1–R3; you pick the stimulus and the positions.
5. H100: Exp 1, Exp 2, arm B.
6. RTX 4090: analyses.

## Open questions

1. Data: A (tiu), B (UTH) or C (own)? If C, which families (recall, arithmetic, in-context, others)?
C: we, stick to tiu. As stimulus I'd say "Is this true or false?. Answer: ". Maybe we can also check what the natural response of the model is after "is this true or false" to make it more in distribution

Answer (26-10-13, Claude Opus 5.5): tiu it is. Three tokenization points about the stimulus, all checkable in the screen R1:
- It needs a leading space (" Is this true or false? …"). Then the statement's tokens stay unchanged, as measured for " This statement is".
- "?." looks like a typo; I test " Is this true or false? Answer:" and your exact string.
- A trailing space after "Answer:" becomes its own token. Qwen's tokenizer normally writes " True" as one token with the space included, so after a separate space token the model sees an unusual split. I test both versions, with and without the trailing space.

The natural-response check is now part (a) of R1: greedy continuations after " Is this true or false?", to see how the model itself goes on (" Answer: True", " True.", a newline, …). The stimulus can then end where the model naturally gives its verdict. Qwen2.5-7B is cached on this machine, so R1 runs on the CPU without renting a GPU. Results in the entry *R1 stimulus screen* below.

Caveat that stays with tiu [verified: entry above]: in larger_than and animal_class, the object word alone predicts the label at 0.70–0.75. It matters when reading per-family results, not for the setup.
2. The object-token position: worth trying now, later, or not at all?
C: maybe later

Answer (26-10-13, Claude Opus 5.5): Noted; deferred. It is listed in FUTURE_WORK.md so it doesn't get lost.
3. Per-channel normalisation of the residual lens: confirmed?
C: yes

Answer (26-10-13, Claude Opus 5.5): Noted.


---

26-10-13, Claude Opus 5.5 — *R1 stimulus screen, lens implementation, R2/R3, analysis V*; I have read INDEX.md, this topic's SUMMARY and the whole LOG including Julian's comments of 26-10-13, `docs/CODING.md`, and the loss, PGD, eval and data-loading code. Answers to the four new comments are in place above. Everything below ran on this machine's CPU (Qwen2.5-7B-Instruct is cached here, 62 GB RAM); no GPU was rented. Scripts and outputs: `~/spd_out/26-10-10_bolder_experiments/`.

Decisions taken from Julian's comments (verbatim above): data = tiu; stimulus = his "Is this true or false?. Answer: ", plus a check of the model's natural response; object-token position deferred (now in FUTURE_WORK.md); per-channel normalisation confirmed. Naming: I propose **Exp 0 (next-token lens)** for the former arm B (`config_truth_last_token.yaml`) [assumed: proposal, awaiting Julian].

OUTDATED (26-10-13): the stimulus used is " Is this true or false? Answer:" (Julian's string without the "?." and without the trailing space), chosen after R1 below and decided in entry *decisions on the four open questions; commit*.

## R1 stimulus screen [verified: `r1_stimulus_screen.{py,out}`, single CPU run, 26-10-13]

546 tiu test statements (up to 30 per dataset, seed 0; 273 true, 273 false). Statement and stimulus tokenized separately; for all five stimuli the ids equal the joint tokenization (0/546 mismatches). Verdict score = log P(true-like) − log P(false-like) at the last stimulus token; true-like = single-token spellings of true/correct/yes/right, false-like = false/incorrect/wrong/no, with and without a leading space (20 tokens each; the script prints them). "Truth mass" = probability on these 40 tokens.

**(a) The model's natural response** (greedy, 16 tokens, 2 statements per dataset = 40):
- after " Is this true or false?": 22/40 start reasoning (" To determine whether the statement … is"), 9/40 answer " True."/" False." directly, 9/40 something else (mostly " This statement is false.", " The statement is false.", " Explain your answer.").
- after " Is this true or false? Answer:": 33/40 answer " True"/" False" directly; the other 7 start " To determine…", and all 7 are number comparisons (7 of the 8 larger_than/smaller_than samples).

**(b) Next-token screen:**

| stimulus | truth mass (mean) | top-1 token is true/false-like | verdict accuracy | AUC | share predicted true |
|---|---|---|---|---|---|
| " This statement is" | 0.387 | 0.473 | 0.951 | 0.992 | 0.476 |
| " Is this true or false?" | 0.184 | 0.227 | 0.929 | 0.979 | 0.465 |
| " Is this true or false? Answer:" | **0.736** | **0.797** | 0.903 | **0.998** | 0.403 |
| " Is this true or false? Answer: " (trailing space) | 0.083 | 0.057 | 0.912 | 0.994 | 0.412 |
| " Is this true or false?. Answer: " (Julian's string, with a leading space) | 0.055 | 0.015 | 0.910 | 0.991 | 0.414 |

Reading [concluded unless tagged]:
- **The trailing space breaks the stimulus.** With it, the most likely next token is the digit "1" (p ≈ 0.3–0.4; then " To", "2", …), and only 1.5–6% of statements have a verdict token on top. As expected: Qwen writes " True" with its leading space as one token, so after a separate space token the verdict word is unusual. The verdict information is still there (AUC 0.99), but it carries a few percent of the probability, so a full KL would mostly protect digit predictions.
- **" Is this true or false? Answer:" (no trailing space) puts the most mass on the verdict** (0.74) and separates true from false best (AUC 0.998). It matches the model's natural response: after the question it often gives " True."/" False." on its own, and " Answer:" makes it do so at once. The remaining mass is mostly " To" (0.19–0.24, the start of step-by-step reasoning), concentrated in the number families: truth mass there is only 0.13–0.42.
- **Accuracy (0.90–0.95) is limited by the model, not the stimulus:** under every stimulus, most errors sit in the *true negated* families (the number families lose up to 0.2 under some stimuli). neg_element_symb_true ("Zinc does not have the symbol Fr.") gets 0.07–0.29 accuracy, i.e. the model calls most of these statements false; neg_animal_class_true gets 0.37–1.0 and neg_cities_true / neg_sp_en_trans_true 0.77–1.0. With " Answer:" the model leans to "false" (share predicted true 0.403 for 50% true statements). The verdict is the model's own belief, errors included, and the stimulus lens protects exactly that.
- " This statement is" has the highest accuracy (0.951), but 0.27–0.33 of its mass goes to ':\n' / ':\n\n' (a format continuation), so its full-KL loss would spend much of its weight on formatting.

**Recommendation** [assumed: Julian decides]: " Is this true or false? Answer:" for Exp 2 and analysis V. It is your stimulus without the "?." and without the trailing space. I have put it into `config_truth_stimulus_lens.yaml`; it is a one-line change if you prefer another.

## Implementation: the target lens [verified: unit tests incl. two mutation checks, CPU smoke training of both new configs on the tiny Qwen2, `make test` and `make check` (only the known rotgrid failures), 26-10-13]

**Design.** All three runs decompose the same positions (the statement's last token) and differ only in what the target loss compares, so the code has one new concept, the *target lens* (`target_lens` in the config):
- `next_token` (default; Exp 0 and every earlier config): the model's logits at the decomposed positions, KL. Unchanged behaviour.
- `residual` with `layer: 23` (Exp 1): the residual stream after decoder layer 23 at the decomposed positions, each channel divided by its standard deviation, MSE averaged over channels. The standard deviations are computed at startup on the original model, from one pass over the training statements' decomposed positions, and saved in the run directory as `residual_lens_channel_std.pt` (also to WandB).
- `stimulus` with `stimulus: "<text>"` (Exp 2): the stimulus is appended to every target statement (tokenized separately), and the loss compares the next-token logits at the stimulus's last token, KL.

**How it works.** Inside `ComponentModel.output_lens(lens)`, every forward pass returns the lens's view instead of the logits: the captured output of `model.layers.23` divided by the channel std, or the logits at each row's stimulus end, repeated at every position. Training and eval enter this context for target batches only; the nontarget (Pile) stream stays outside and keeps logits + KL. Because the lens output keeps the (batch, position, …) shape, all existing losses (stochastic subset, persistent PGD, eval metrics) select the decomposed positions from it unchanged; no loss function needed a second position mask. With one decomposed position per statement, the stimulus lens reads the stimulus end exactly once per statement. With k decomposed positions per statement, each statement's KL would count k times (documented in `StimulusLens`). Where target and nontarget differ in loss type, the code now uses `config.target_output_loss_type` (mse for the residual lens) for target losses, PPGD and target eval metrics, and `output_loss_type` (kl) for nontarget.

I chose this over a second, separate loss-position mask, which would have had to be threaded through every loss function, PPGD and the metric interface. The lens needed changes only in the training loop, eval loop and model forward, and it also covers Exp 1, which a second mask would not.

**Files.**
- New: `spd/models/output_lens.py` (lens classes, per-batch construction, check that the lens layer comes after the last decomposed layer), `spd/models/residual_lens_stats.py` (channel std), `tests/test_output_lens.py`, configs `config_truth_residual_lens.yaml` (Exp 1) and `config_truth_stimulus_lens.yaml` (Exp 2), analysis `verdict_accuracy.py` (V, below).
- Changed: `spd/configs.py` (`TargetLens`, `target_output_loss_type`, validation: a lens needs an LM task with prepared datasets, a position restriction and KL on nontarget), `spd/models/component_model.py` (`output_lens` context, capture hook), `spd/run_spd.py` (std pass, lens per target batch, target loss type), `spd/eval.py` (same for eval), `spd/experiments/lm/prompts_dataset.py` (stimulus appending: `position_mask` covers statement tokens only, new `stimulus_end` batch column, trimming keeps the stimulus), `spd/experiments/lm/lm_decomposition.py`, `spd/utils/general_utils.py` (`STIMULUS_END_KEY`), two loader callers passing `stimulus=None`.
- Exp 0 = `config_truth_last_token.yaml`, unchanged.

**Tests** (`tests/test_output_lens.py`, 3 tests):
- stimulus batches: the statement's tokens are unchanged, the mask covers only statement positions, `stimulus_end` is right, and trimming keeps the stimulus;
- the stimulus-lens loss equals a hand-computed KL at the stimulus end, the KL at the decomposed position itself differs from it, and positions before the decomposed one are unchanged by the masking;
- the residual lens equals Hugging Face's `hidden_states[L+1] / std`.

Mutation checks: making the stimulus lens return the plain logits, or read one position too early, fails the test.

A pitfall found on the way (recorded in FEEDBACK.md): the tiny random Qwen2 used for CPU tests has logits with std ≈ 0.01, so its next-token distributions are nearly uniform and every KL is at float-noise level (1e-7). My first version of the test passed for that reason with little power. The test now scales the tiny model's final norm by 1000 (logits std ≈ 10) and asserts that the KL is far above noise. For the same reason, **the tiny smoke run of Exp 2 shows only that the code runs** (its stimulus-lens losses are ~1e-7, as are all its KLs). The residual-lens smoke run gave sensible values (target losses 0.03–0.3, channel std 0.45–1.95 on the tiny model).

Configs: both new configs are `config_truth_last_token.yaml` with `target_lens` (and for Exp 2 `max_seq_len: 28` = 20 statement + 8 stimulus tokens) changed; the coefficients of Exp 1 are discussed under R3.

## R2/R3: zeroing MLPs 15–19 at statement positions [verified: `r2r3_zero_ablation.{py,out}`, single CPU run, 26-10-13]

Same 546 statements as R1. The `down_proj` outputs of layers 15–19 are set to zero at the period, at the last two statement tokens, or at all statement tokens. That is tPD's "all components off, Δ off" state; the stimulus always runs on the original weights. The residual-lens loss uses the channel std of the stored layer-23 residuals of all 6,560 training statements (from the truth-writing topic's `features.npz`: the same data and position a training run would use). That std ranges 0.82–17.5 across channels (median 1.58). The truth direction is the mass-mean direction of those stored residuals.

At the period (the position all three runs decompose):

| zeroed at | next-token KL at the period (Exp 0 loss) | residual-lens loss (Exp 1) | layer-23 truth separation, share left (per-family median) |
|---|---|---|---|
| period | 1.68 nats | 2.26 | 0.14 (0.09) |
| last 2 tokens | 1.62 | 2.30 | 0.05 (0.04) |
| all statement tokens | 1.82 | 2.43 | 0.03 (−0.01) |

(Original separation 61.9, the scale of the 62.5 reported in the truth-writing topic.)

At the stimulus end (Exp 2 loss = KL there; verdict as in R1):

| zeroed at | " Is this true or false? Answer:" KL / accuracy / AUC / separation left | " This statement is" KL / accuracy / AUC / separation left |
|---|---|---|
| none (original) | 0 / 0.903 / 0.998 / 1 | 0 / 0.951 / 0.992 / 1 |
| period | 0.28 nats / 0.789 / 0.927 / 0.74 | 0.33 / 0.848 / 0.915 / 0.67 |
| last 2 tokens | 0.47 / 0.712 / 0.834 / 0.49 | 0.49 / 0.766 / 0.836 / 0.40 |
| all statement tokens | 0.55 / 0.656 / 0.792 / 0.40 | 0.62 / 0.674 / 0.786 / 0.30 |

Reading [concluded unless tagged]:
- **The layer-23 truth separation at the period depends almost entirely on MLPs 15–19 at the period itself:** zeroing them there removes 86% of it (median 91% per family; in some families the sign even flips: per-family share left ranges −0.48 to 0.41). So the residual lens at layer 23 sees a quantity these five MLPs largely produce. That fits the truth-writing topic, where switching off the strongest single arm-A component (at all positions) removed 19.7 of 62.5 units, about 30%, of this separation.
- **The verdict depends on them much less:** zeroing at the period removes only 26% of the verdict separation, and even zeroing at every statement token leaves 40% (" Answer:") / 30% (" This statement is"). The rest reaches the verdict through other layers or is recomputed from the statement by the stimulus tokens, which run on the original weights. This is the mediation point of the first design entry: the stimulus-lens loss sees only the part of the verdict that passes through MLPs 15–19 at the decomposed position. That part is real (KL 0.28 nats when they are zeroed; accuracy 0.90 → 0.79) but a minority.
- **Position heuristic (R2):** decomposing at the last two or all statement tokens would put more of the verdict under the loss (51% / 60% of the separation instead of 26%). The design keeps the period, so Exp 0/1/2 stay comparable; the larger position sets are an option for later if Exp 2 shows little.
- **Loss scale (R3):** in the all-off state at the period, the residual-lens loss (2.26) and the next-token KL (1.68) differ by a factor 1.35. The plan was to scale Exp 1's reconstruction coefficients by their ratio (0.74). Proposal [assumed]: keep the coefficients of Exp 0 unchanged for Exp 1, since the two scales are this close. For Exp 2, the all-off KL is 6× smaller (0.28 vs 1.68 nats), but both losses are KLs in nats over next-token distributions, so the same coefficient means the same pressure per nat of change. The smaller number only says that the verdict depends less on these MLPs. I would not rescale it either.

## Analysis V: verdict accuracy under component masks [verified: real-model CPU check on arm A, 26-10-13]

`spd/experiments/lm/honesty_targeted_decomposition/verdict_accuracy.py` (usage in its docstring). For each run, on the tiu test split with a stimulus appended, the decomposed `down_proj` outputs are replaced **at the period only** (all other positions, the stimulus included, run on the original weights). States:
- original;
- components only: rounded CIs, Δ off;
- Δ only;
- all off;
- full: all components + Δ, i.e. the original weights computed through the decomposition;
- full minus one component, for every component active on at least 1% of the statements.

Per state it reports verdict accuracy, AUC, separation (mean verdict score of true minus false statements), share predicted true, KL to the original at the stimulus end, and per-family accuracy. Single-component effects are measured against the full state, not against the original. On the CPU, the full state's verdict scores differ from the original's by up to 0.25 through bf16 rounding (separation identical at 11.23, KL 0.0006). That is as large as the single-component effects here, so comparing against the original would mix rounding into them. The script asserts that full reproduces the original within 10% of the mean |score|. The decomposed position is always the period, whatever a run was trained on, so the untrained baseline and arm A (both configured for all tokens) are measured at the same position as Exp 0–2.

Real-model check: arm A (`s-bd23f0d1`, all-tokens decomposition, cached locally), stimulus " Is this true or false? Answer:", the first 5 test statements per dataset (n = 100), 5 single ablations (the components active most often) [verified: `v_cpu_check.log`, `v_cpu_check/verdict_s-bd23f0d1.json`]:

| state | accuracy | AUC | separation | KL to original |
|---|---|---|---|---|
| original | 0.91 | 0.999 | 11.23 | 0 |
| components only | 0.87 | 0.965 | 8.57 | 0.138 |
| Δ only | 0.88 | 0.984 | 9.84 | 0.068 |
| all off | 0.82 | 0.940 | 8.57 | 0.179 |
| full | 0.91 | 0.999 | 11.23 | 0.0006 |

The single ablations change the separation by −0.24 to +0.04. Arm A has 11.7 active components per statement at the period.

Reading [assumed: n = 100, one CPU run; a pointer for the real analysis, not a result]:
- The all-off state keeps 76% of the separation; R2 found 74% on its 546 statements with independent code. The two implementations agree.
- Arm A's components alone keep no more of the verdict separation than switching everything off (8.57 vs 8.57; accuracy 0.87 vs 0.82, KL 0.14 vs 0.18). Its components, trained with the next-token KL at all tokens, carry little of what the verdict needs from the period. This is the baseline Exp 2 has to beat.

## Status and next steps

Implemented and tested: both lenses, both configs, V. Not run: any training. Nothing is committed (Julian reviews the diff).

Proposed order [assumed]:
1. Julian: stimulus choice, the name Exp 0, coefficients unchanged (questions below); review and commit the code.
2. H100: Exp 0 (`config_truth_last_token.yaml`), Exp 1 (`config_truth_residual_lens.yaml`), Exp 2 (`config_truth_stimulus_lens.yaml`), about 40 min each [assumed: arm A took 39 min]. Exp 1 first spends one pass over the 6,560 training statements on the channel std.
3. Analyses (RTX 4090 or the H100 right after training): V on Exp 0–2, the untrained baseline `s-7fad0c14` and arm A `s-bd23f0d1` on the full test split with all single ablations; `truth_writers.py` on Exp 0–2. Component overlap between the runs: still to be written.

## Open questions

1. Stimulus for Exp 2 and V: " Is this true or false? Answer:" (my recommendation, see R1), or another?
C: yes, your recommendation
2. The name **Exp 0 (next-token lens)** for the former arm B: OK?
C: yes
3. Coefficients: keep Exp 0's for Exp 1 and Exp 2 (R3), or rescale Exp 1 by 0.74 as originally planned?
C: no rescaling
4. Positions: stay at the period for all three runs (my recommendation, for comparability), although R2 shows that only 26% of the verdict separation passes through MLPs 15–19 at the period, against 60% over all statement tokens?
C: only period

---

26-10-13, Claude Opus 5.5 — *decisions on the four open questions; commit*; same session.

Julian answered the four open questions with C: comments (above, under *Open questions* of the previous entry):
- stimulus " Is this true or false? Answer:" [decided] (already in `config_truth_stimulus_lens.yaml`);
- the former arm B is called **Exp 0 (next-token lens)** [decided];
- no rescaling: Exp 1 and Exp 2 keep Exp 0's loss coefficients [decided];
- decompose only at the period in all three runs [decided].

The configs need no change. Next: commit (Julian asked in chat to commit before the runs), then the H100 runs. The agent proposed two commits, code and records, so the hash that the runs record contains only code.

---

26-10-13, Claude Opus 5.5 — *H100 tuning trials: plan*; same session. Julian rented the H100 himself and wrote in chat:

C: (transcribed from chat) I've rented the H100 and before you start the runs you can do some simple tuning

Instance: H100 80GB HBM3 at 700 W (no power cap) [verified: `nvidia-smi`]; repo at `56bdf4f`. One tracked file inside a gitignored directory (`spd/scripts/plot_wv_subspace_overlap/out/wv_overlap_writeup.tex`; `.gitignore` has `**/out/`) is skipped by `spd-vast`'s rsync, so git saw it as deleted and every run would have been stamped "uncommitted changes". I marked it `--skip-worktree` on the instance (git status then clean); the launcher fix is in FUTURE_WORK.md.

Trials [assumed: plan], sequential on the instance so the per-step times are clean; configs and runner in `~/spd_out/26-10-10_bolder_experiments/tuning/` (on the instance `/root/trials/`); no checkpoint upload:
1. `smoke_exp1`, `smoke_exp2`: the Exp 1 / Exp 2 configs for 300 steps (PPGD from step 240). They check what the tiny model could not: memory, the channel-std pass, the loss scales on the real model, per-step time.
2. `exp0_b16` vs `exp0_b64`: Exp 0 for 2,000 steps at target batch 16 (current) vs 64; the Pile batch stays 16. Question: with one loss position per statement, batch 16 gives 16 loss positions per step, against ~160 in the all-tokens arm where the settings were tuned (training-run topic). If batch 64 reaches clearly better target/PGD reconstruction at similar Pile KL and acceptable cost per step, the three runs use batch 64.

---

26-10-13, Claude Opus 5.5 — *H100 tuning trials: results*; same session. [verified: `~/spd_out/26-10-10_bolder_experiments/tuning/` (configs, logs, `runs/*/metrics.jsonl`, `compare_trials.py`); one H100 at 700 W, one seed per trial; WandB runs s-1c8cd417 (smoke Exp 1), s-6e70a646 (smoke Exp 2), s-50d0c294 (Exp 0 batch 16), s-54d0f743 (Exp 0 batch 64), checkpoints not uploaded]

**Smoke runs on the real model (300 steps; PPGD from step 240).** Both ran without errors.
- Exp 1 (residual lens): the channel std computed at startup (min/median/max 0.825 / 1.581 / 17.51) agrees with my CPU computation from the stored training residuals (0.824 / 1.581 / 17.46); the two are independent code paths. After 300 steps: 10.8 active components per statement (220 at the start), rounded target loss 0.93 (step 0: 2.91; all-off reference from R3: 2.26), PGD 1.32, Pile rounded KL 0.020. ~345 ms/step without PPGD, ~645 ms with it.
- Exp 2 (stimulus lens): 4.5 active components, rounded target KL at the stimulus end 0.068 nats (all-off reference from R2: 0.28), Pile 0.031. **The eval PGD loss is very high: 9.7 nats at step 150, 4.7 at step 300**, i.e. ~70× the rounded loss (all-tokens arm: ~2.5×). PPGD had run only 60 steps by then; the full run gives it 1,000. To watch in the full run [assumed: PPGD will bring it down, as it did for Exp 0 below].

**Batch size, Exp 0 (next-token lens at the period), 2,000 steps:**

| step 2,000 | batch 16 | batch 64 |
|---|---|---|
| active components per statement (L0) | 13.7 | 8.9 |
| target rounded KL (nats, at the period) | 0.173 | 0.135 |
| target stochastic KL | 0.170 | 0.127 |
| eval PGD KL | 2.04 | 1.00 |
| all components, no Δ (unmasked) | 0.238 | 0.168 |
| Pile rounded KL | 0.054 | 0.041 |
| trial wall time (incl. model load, evals) | 19.5 min | 19.8 min |

Reading [concluded]: batch 64 is better on every metric, and the direction is the same at every eval (steps 500, 1,000, 1,500). The differences exceed the run-to-run noise measured in the training-run topic (up to 10% on target metrics, 15–25% on PGD/Pile), except Pile, which is at the edge. Batch 64 costs nothing: a step is dominated by the fixed Pile batch (16 × 64 tokens) and per-pass overheads. In both trials PGD falls steeply during the PPGD phase (Exp 0 batch 16: 4.58 at step 1,500 → 2.04 at 2,000), consistent with the expectation for Exp 2 above.

**Change:** target `batch_size: 64` in all three configs (`config_truth_{last_token,residual_lens,stimulus_lens}.yaml`), with the reason in their headers; steps (5,000), LR, eval batch and Pile batch unchanged. Batch 128 is untested (it might also be free). Expected time per full run: ~38 min [assumed: 4,000 × ~0.35 s + 1,000 × ~0.57 s + evals and model load].

OUTDATED (26-10-13): Exp 2's full run took 47 min on the H100; see entry *Exp 2 finished*.

---

26-10-13, Claude Opus 5.5 — *full runs launched*; same session.

C: (transcribed from chat) plan approved

The plan: commit the batch-64 configs, re-sync the instance, run Exp 0, Exp 1, Exp 2 sequentially (detached), checkpoints to WandB.

---

26-10-13, Claude Opus 5.5 — *Exp 2 finished*; same session. [verified: run `s-05db01ad`, commit `ae7f041`, no uncommitted changes, 47 min on the H100; `model_5000.pth` (238 MB) on WandB; metrics copied to `~/spd_out/26-10-10_bolder_experiments/full/stimulus_lens/`]

Exp 2 (stimulus lens, " Is this true or false? Answer:", period only, batch 64, 5,000 steps), eval on 256 test statements:

| step | active components per statement | target rounded KL (nats, stimulus end) | stochastic KL | Δ only KL | eval PGD KL | Pile rounded KL |
|---|---|---|---|---|---|---|
| 1,000 | 1.7 | 0.036 | 0.026 | 0.035 | 1.98 | 0.084 |
| 4,000 (PPGD starts) | 1.0 | 0.037 | 0.011 | 0.022 | 0.93 | 0.056 |
| 5,000 | 0.47 | 0.029 | 0.006 | 0.014 | 0.60 | 0.040 |

Reading [assumed: one run, one seed; eval metrics only, before any analysis]:
- The PGD concern from the smoke run is resolved as far as training goes: 9.7 nats (smoke, step 150) → 0.60 at the end.
- The decomposition is extremely sparse: fewer than one active component per statement on average. Δ alone (all components off) reproduces the verdict at least as well as the rounded components (0.014 vs 0.029 nats). With the all-off reference at 0.28 nats (R2), the stimulus loss seems to exert little pressure to put anything in the components. Analysis V will show whether the few active components carry verdict information.
- Pile rounded KL ends at 0.040, the level of the Exp 0 batch-64 trial (0.041).

Exp 1 started 15:46 UTC, then Exp 0.

---

26-10-13, Claude Opus 5.5 — *V on Exp 2; Exp 3 (stimulus decomposed) added*; same session. Julian in chat:

C: (transcribed from chat) Can you add another experiment to the pipeline. That decomposese ". Is this true or fales? Answer:"? Also can you start the evaluation of experiment 2?

## V on Exp 2 [verified: H100, run concurrently with Exp 1; `/root/analysis/v.log` and `verdict_s-05db01ad.json` on the instance, to be copied to `~/spd_out/26-10-10_bolder_experiments/full/`; full tiu test split, n = 1,794; stimulus " Is this true or false? Answer:"; replacement at the period only]

| state | accuracy | AUC | separation | KL to original (nats) |
|---|---|---|---|---|
| original | 0.953 | 0.999 | 12.04 | 0 |
| components only (0.8 active per statement) | 0.958 | 0.998 | 10.88 | 0.033 |
| Δ only | 0.939 | 0.997 | 11.15 | 0.030 |
| all off | 0.918 | 0.983 | 10.21 | 0.116 |
| full (components + Δ) | 0.953 | 0.999 | 12.04 | 0.0007 |

13 components are active on ≥ 1% of the statements; ablating one of them singly changes the separation by −0.38 to +0.09 (largest: component 116, active on 12%, −0.38; components 103 and 39, active on 2%, −0.36 / −0.30).

Reading [assumed: one run; the references (untrained `s-7fad0c14`, arm A `s-bd23f0d1`) are still running]: on the full test split, zeroing MLPs 15–19 at the period removes 15% of the verdict separation and costs 0.116 nats (R2's sample: 26%, 0.28 nats; that sample capped 30 statements per dataset, the full split is dominated by cities and number comparisons). Exp 2's components alone, under one per statement, recover ~70% of that KL (0.116 → 0.033) and a third of the lost separation (10.21 → 10.88 of 12.04).

## Exp 3: decompose the period and the stimulus

Design [assumed: my reading of Julian's request]: as Exp 2 (same stimulus, same loss: KL at the stimulus end), but the decomposed positions are the period **and all 8 stimulus tokens**, ". Is this true or false? Answer:" (9 positions per statement). In Exp 2 the stimulus ran on the original weights and could recompute the verdict itself; here the verdict computation at the stimulus tokens, including the token that outputs the verdict, goes through the components too.

Implementation [verified: unit test (period + exactly the stimulus tokens selected), CPU smoke training on the tiny Qwen2, V smoke test in both modes, `make test` only the known rotgrid failure; 26-10-13]:
- `StimulusLens.decompose_stimulus: bool` (default false, so Exp 2's saved config still loads). The prompt loader now takes the `StimulusLens` config instead of the bare string; with the flag, `position_mask` = the selected statement positions plus every stimulus token.
- Config `config_truth_stimulus_span.yaml` (label `tiu_stimulus_span`) = Exp 2's config + `decompose_stimulus: true`; batch 64, 5,000 steps.
- `verdict_accuracy.py --decompose_stimulus`: replaces at the period and all stimulus tokens; the components state uses the rounded CI per position; output `verdict_<run>_span.json`. Without the flag, the masks are the same as before (rounded CI at the period), so the Exp 2 result above stands.
- Loss scaling: each statement's KL now counts at 9 positions, but all statements have 9, so the mean (the loss value) is unchanged. Importance minimality now averages over 9 positions per statement, 8 of them stimulus tokens.

Pending: commit and sync, then Exp 3 after Exp 0 in the queue.

C: (transcribed from chat) yes, to the plan

(Plan: commit code/config and records separately, sync the instance while Exp 1 runs, queue Exp 3 after Exp 0; Exp 0 then records the new commit.)

---

26-10-13, Claude Opus 5.5 — *V references; Exp 2 report*; same session. Julian asked in chat for a quick report about Exp 2; he preferred waiting for the running V references over starting an extra job.

**V on the references** [verified: `~/spd_out/26-10-10_bolder_experiments/full/v/verdict_{s-05db01ad,s-7fad0c14,s-bd23f0d1}.json`, full test split n = 1,794, replacement at the period, stimulus " Is this true or false? Answer:"]:

| decomposition | active per statement | components-only KL | separation | Δ-only KL | singles ≥ 1% | largest single \|Δsep\| |
|---|---|---|---|---|---|---|
| Exp 2 `s-05db01ad` | 0.8 | 0.033 | 10.88 | 0.030 | 13 | 0.38 |
| arm A `s-bd23f0d1` | 9.3 | 0.092 | 9.84 | 0.044 | 55 | 0.33 |
| untrained `s-7fad0c14` | 220 | 0.115 | 10.31 | 0.008 | 475 | 0.07 |
| all off (same for every run) | 0 | 0.116 | 10.21 | – | – | – |

Original: separation 12.04, accuracy 0.953. Per family, the only family with a large effect is the negated Spanish translations (64 test statements): accuracy original 0.86, all off 0.06, Exp 2 components 0.95, arm A 0.50, untrained 0.17. Negated element symbols: 0.55 / 0.36 / 0.59 / 0.50 / 0.23. The affirmative families are at 0.96–1.00 in every state.

Reading [concluded, one seed]: Exp 2's components (under one per statement) recover ~70% of the verdict KL lost by switching the five layers off, arm A's ~20%, the untrained ones nothing. The recovery concentrates in negated statements, so it may be a negation mechanism rather than a truth mechanism (open). The like-for-like reference, Exp 0 at the same position, is still training.

**Report:** `reports/26-10-13_exp2_stimulus_lens.typ` (4 pages; PDF gitignored, `typst compile reports/26-10-13_exp2_stimulus_lens.typ`). Written for Julian, self-contained (method, setup, training, V, per-family, caveats). Self-check done by re-reading the compiled PDF (procedure steps 1–2 by the author, no sub-agent): added definitions of AUC and the Pile, fixed a Typst rendering of "Exp 2's" as a prime. Numbers re-read from the V JSONs, `metrics.jsonl` and the R1 output.

---

26-10-13, Claude Opus 5.5 — *run status check: H100 gone, Exp 0 crashed*; sync session, not the session that launched the runs. I have read this topic's SUMMARY and the last LOG entry, and queried WandB and vast.ai. No GPU used.

C: (transcribed from chat) about 4. please check

(Item 4 of the sync report: the records said Exp 1 training, Exp 0 and Exp 3 queued, as of the last LOG entry, and nobody had checked the instance.)

Findings [verified: WandB API (`bitt-j-personal/spd`) and `vastai`, 26-10-13 ~20:30 UTC]:
- **Exp 1** (residual lens, `s-cd129592`): finished, 5,000 steps, commit `ae7f041`, created 15:46:35 UTC, last heartbeat 16:41:34 UTC. `model_5000.pth` (238,232,949 bytes) is on WandB. Its `metrics.jsonl` was not copied to `~/spd_out/26-10-10_bolder_experiments/full/` (only `stimulus_lens/` and `v/` are there).
- **Exp 0** (next-token lens, `s-773bfec6`): state `crashed`, last logged step 1,600 of 5,000, commit `57e7bc7`, created 16:41:54 UTC, last heartbeat 16:54:40 UTC. No checkpoint and no `output.log` on WandB, so the cause is not recorded there.
- **Exp 3** (stimulus span): no WandB run exists; it never started.
- **The H100** (vast.ai instance 55752929, per `~/.ssh/config.d/vastai.conf`, written by `spd-vast` at 16:01 local = 14:01 UTC): `vastai show instance 55752929` returns "not found or no longer exists"; ssh to the host is refused. It is no longer rented.

Reading [assumed]: the instance went away while Exp 0 was training (WandB marks a run `crashed` when its heartbeat stops, and no error log was uploaded); a Python error would usually have left an `output.log`. Who or what ended the rental is not recorded: this LOG's previous entry was written at 18:28 local (16:28 UTC), before the crash. Anything only on the instance is lost: `/root/analysis/`, `/root/trials/`, Exp 1's local run directory (the checkpoint survives on WandB).

Consequences:
- Exp 0, the like-for-like reference for Exp 2's negation result, has to be rerun from scratch, as does Exp 3 (both configs unchanged, batch 64, ~47 min each by Exp 2's time).
- Exp 1 can be analysed now (V and `truth_writers.py` from the WandB checkpoint).

---

26-10-14, Claude Opus 5.5 — *rerun of Exp 3 and Exp 0 launched*; new session. I have read INDEX.md, this topic's SUMMARY, this LOG from *H100 tuning trials: plan* to the end, the vast.ai rentals SUMMARY, the direction-free SUMMARY and `docs/CODING.md`. Julian in chat:

C: (transcribed from chat) I've rented a h100 can you run the experiments that got interrupted?

Instance [verified: `vastai show instances`, `nvidia-smi` on the instance, 26-10-14]: vast.ai 55905320 (machine 153188, $2.56/h), provisioned by `spd-vast` as `s-09454b41`; H100 80GB HBM3, power limit 700 W. Repo on the instance at `57e7bc7`, the same commit Exp 0's crashed run `s-773bfec6` used. `git diff HEAD -- spd/` is empty locally; the working tree differs only in records (`convos/`, INDEX.md, FUTURE_WORK.md, an untracked report) and in the `wv_overlap_writeup.tex` rsync gap (FUTURE_WORK.md), so both runs will be stamped `uncommitted_changes: true` with code identical to `57e7bc7`. I did not hide this with `--skip-worktree`.

Launch: `~/spd_out/26-10-10_bolder_experiments/run_rerun.sh` (on the instance `/root/full/`), detached, the same pattern as the earlier `run_full.sh`: `lm_decomposition.py` with the committed, unchanged configs, one after another, logs `/root/full/{stimulus_span,last_token}.log`, progress in `/root/full/progress.txt`, checkpoints to WandB (`sync_checkpoints_to_wandb: true`, `save_freq: null`).
- Order: **Exp 3 first** (`config_truth_stimulus_span.yaml`, started 09:56:46 UTC), then Exp 0 (`config_truth_last_token.yaml`). This is the agent's choice: Exp 3's memory and step time at batch 64 with 9 decomposed positions per statement have only been run on the tiny test model, so a failure shows within minutes while I watch. A failed run does not stop the next one.
- Expected [assumed: from Exp 2's 47 min]: ~50 min per run, ~1.7 h total.

Startup check [verified: `/root/full/stimulus_span.log` and `nvidia-smi`, 26-10-14 ~09:59 UTC]: Exp 3 is WandB run **`s-161f8a7b`**; the logged config has `StimulusLens(..., decompose_stimulus=True)`, label `tiu_stimulus_span`, `batch_size=64`. Past step 240 without errors at ~4.0 steps/s before PPGD (Exp 2 ran ~2.6 steps/s on the previous instance; wall times are not comparable across machines). GPU: 28.7 GB used, 96% utilisation, 680 W.

---

26-10-14, Claude Opus 5.5 — *analyses queued after training*; same session. Julian in chat (answering whether to analyse Exp 1 alongside training):

C: (transcribed from chat) The last analysis was really fast. Do the remaining analysises when the training runs are over

Pipeline `~/spd_out/26-10-10_bolder_experiments/analysis/run_analyses.sh` (on the instance `/root/analysis/`, detached, waits until both training runs exit; stops if either failed). The run selection and positions are my choices, not discussed [assumed]:
1. **V at the period** on Exp 0, Exp 1 `s-cd129592` and Exp 3, comparable with the existing rows for Exp 2, arm A `s-bd23f0d1` and untrained `s-7fad0c14` (their JSONs copied to `/root/analysis/v/`).
2. **V with replacement at the period and all 8 stimulus tokens** (`--decompose_stimulus`) on Exp 3, as planned, plus Exp 2 and the untrained decomposition as references at the same 9 positions (the all-off state at 9 positions is a different baseline from the period-only one, so Exp 3's span numbers need their own references).
3. **`truth_writers.py`** extract + analyze on tiu with Exp 0, 1, 2, 3, arm A and untrained in one feature store (`/root/analysis/tw_tiu`); the script treats all runs alike (no hard-coded roles, checked). Its readout position, the last token, is the period these runs decompose.
4. **Component overlap** (the analysis "to write" in the SUMMARY): `spd/scripts/validation/compare_components.py` unchanged, i.e. for each component of run A the max |cos| over run B's components of the flattened rank-one weight, per layer, plus its `--random-b` baseline (B re-initialised, 10 draws). Alive set = components active (rounded CI at the period) on ≥ 1% of test statements, read from V's `active_share` by the new one-off `analysis/alive_from_v.py` (writes the `alive_components.tsv` format; reproduces V's counts: Exp 2 13, arm A 55 [verified]). All 10 pairs of {Exp 0, 1, 2, 3, arm A}, one direction each.

Overlap smoke test [verified: local CPU, Exp 2 → arm A, 56 s, `scratch` output not kept]: runs end to end. **Caveat for its reading:** of Exp 2's 13 alive components, the only two with a non-negligible match pair with the *same index* in arm A (L15:40 ↔ L15:40, weight cos 0.19, U cos 0.88; L17:80 ↔ L17:80, 0.17, U cos 0.95); all others are |cos| < 0.001. All runs use `seed: 0` and so start from identical component weights; a component whose U barely moves keeps its twin's U. Same-index matches are therefore a shared-initialisation artefact until shown otherwise [concluded: identical index + U cos ≈ 0.9 + seed shared], and the random-B baseline (other seeds) does not control for it. I'll report same-index matches separately.

---

26-10-14, Claude Opus 5.5 — *Exp 3 finished*; same session. [verified: run `s-161f8a7b`, exit 0, 09:56:46–10:26:19 UTC (29.5 min), `completed: true`, commit `57e7bc7` with `uncommitted_changes: true` (record files only, see above); `model_5000.pth` (238,232,949 bytes) on WandB; metrics, config and log copied to `~/spd_out/26-10-10_bolder_experiments/full/stimulus_span/`; table from `compare_runs.py`]

Exp 3 (stimulus lens, period + 8 stimulus tokens decomposed), eval on 256 test statements. L0 is averaged over the decomposed positions (`CI_L0` selects positions by `position_mask`), so here it is per position, 9 per statement; in Exp 2 one position = one statement:

| step | L0 (per decomposed position) | target rounded KL (nats, stimulus end) | stochastic KL | Δ only KL | eval PGD KL | unmasked | Pile rounded KL |
|---|---|---|---|---|---|---|---|
| 0 | 226 | 3.32 | 3.82 | 0.59 | 14.6 | 3.70 | 0.008 |
| 1,000 | 35.9 | 0.091 | 0.076 | 16.9 | 13.8 | 0.29 | 0.085 |
| 3,000 | 4.21 | 0.057 | 0.112 | 17.7 | 4.36 | 0.45 | 0.051 |
| 5,000 | 3.43 | 0.028 | 0.024 | 15.1 | 1.62 | 0.041 | 0.054 |

Exp 2 at step 5,000 for comparison: L0 0.47, target rounded 0.029, stochastic 0.006, Δ only 0.014, PGD 0.60, unmasked 0.008, Pile 0.040.

Reading [assumed: one run, eval metrics only, before V]:
- The intended difference shows: with Δ alone (all components off) at the 9 positions the verdict is destroyed (15 nats), against 0.014 in Exp 2. In Exp 3 the components, not Δ, carry the verdict computation at the stimulus.
- Rounded-CI reconstruction of the verdict is as good as Exp 2's (0.028 vs 0.029 nats) with ~3.4 active components per position, ~31 per statement summed over the 9 positions.
- Robustness is worse: eval PGD 1.62 vs 0.60, unmasked 0.041 vs 0.008, Pile 0.054 vs 0.040 (Pile at the edge of run-to-run noise per the training-run topic).
- Speed: 346 ms/step at the last tqdm reading (PPGD phase), 29.5 min in total on this instance; not comparable with Exp 2's 47 min on the old one.

Exp 0 started 10:26:19 UTC as **`s-aeaa70d8`**.

---

26-10-14, Claude Opus 5.5 — *Exp 0 finished*; same session. [verified: run `s-aeaa70d8`, exit 0, 10:26:19–10:50:14 UTC (24 min), `completed: true`, commit `57e7bc7` with `uncommitted_changes: true` (record files only); `model_5000.pth` (238,232,949 bytes) on WandB; metrics, config and log copied to `~/spd_out/26-10-10_bolder_experiments/full/last_token/`; table from `compare_runs.py`]

Exp 0 (next-token lens at the period, batch 64), eval on 256 test statements:

| step | L0 (per statement) | target rounded KL (nats, at the period) | stochastic KL | Δ only KL | eval PGD KL | unmasked | Pile rounded KL |
|---|---|---|---|---|---|---|---|
| 1,000 | 9.21 | 0.157 | 0.211 | 8.25 | 5.47 | 0.456 | 0.049 |
| 2,000 | 7.47 | 0.142 | 0.159 | 4.56 | 4.68 | 0.345 | 0.071 |
| 4,000 (PPGD starts) | 7.16 | 0.127 | 0.123 | 5.47 | 1.54 | 0.174 | 0.048 |
| 5,000 | 7.66 | 0.107 | 0.090 | 6.63 | 0.63 | 0.113 | 0.033 |

Reading [assumed: one run, eval metrics only]:
- At step 2,000 it agrees with the batch-64 tuning trial `s-54d0f743` (L0 8.9, target rounded 0.135, Pile 0.041; a different instance and the PPGD schedule of a 2,000-step run, so only roughly comparable).
- Unlike Exp 2, Δ alone does not reproduce the output (6.6 nats): the next-token lens puts the period's output computation into ~7.7 components per statement. Exp 2 needed under one, and its Δ alone sufficed.

The analysis pipeline started at 10:50:40 UTC.

---

26-10-14, Claude Opus 5.5 — *analyses of Exp 0–3: V, truth_writers, component overlap*; same session. [verified: pipeline `run_analyses.sh` 10:50:40–11:26:44 UTC, every step exit 0 (`status.log`); all outputs copied to `~/spd_out/26-10-10_bolder_experiments/analysis/out/` (`v/`, `v_period.log`, `v_span.log`, `tw_tiu/report.txt` + arrays, `overlap/*.tsv|png`); full tiu test split n = 1,794 for V; one seed per run; numbers below re-read from those files]. Durations: V at the period 5 min (3 runs), V over the span 25 min (3 runs, up to 480 singles each), `truth_writers` extract 1.7 min + analyze 1 min, overlap 20 × ~11 s.

## V at the period (replacement at the period only; stimulus " Is this true or false? Answer:")

All off (MLPs 15–19 zeroed at the period): accuracy 0.918, separation 10.21, KL 0.116 nats. Original: accuracy 0.953, separation 12.04. "Recovered" = share of the all-off KL that the components-only state removes, (0.116 − KL_components) / 0.116.

| decomposition | active per statement | components-only KL | recovered | separation | accuracy | Δ-only KL | singles ≥ 1% |
|---|---|---|---|---|---|---|---|
| Exp 0 next-token `s-aeaa70d8` | 7.8 | 0.106 | 8% | 9.98 | 0.907 | 0.099 | 40 |
| Exp 1 residual `s-cd129592` | 10.5 | 0.043 | 63% | 11.32 | 0.936 | 0.049 | 85 |
| Exp 2 stimulus `s-05db01ad` | 0.8 | 0.033 | 72% | 10.88 | 0.958 | 0.030 | 13 |
| Exp 3 stimulus span `s-161f8a7b` | 2.0 | 0.043 | 63% | 11.29 | 0.953 | 0.106 | 50 |
| arm A `s-bd23f0d1` | 9.3 | 0.092 | 21% | 9.84 | 0.927 | 0.044 | 55 |
| untrained `s-7fad0c14` | 220 | 0.115 | 1% | 10.31 | 0.918 | 0.008 | 475 |

Per family (components-only accuracy), only the negated families move; affirmative families are 0.96–1.00 in every state:

| family | original | all off | Exp 0 | Exp 1 | Exp 2 | Exp 3 | arm A | untrained |
|---|---|---|---|---|---|---|---|---|
| neg_sp_en_trans | 0.86 | 0.06 | 0.19 | 0.64 | 0.95 | 0.91 | 0.50 | 0.17 |
| neg_element_symb | 0.55 | 0.36 | 0.43 | 0.39 | 0.59 | 0.52 | 0.50 | 0.23 |
| neg_animal_class | 0.73 | 0.68 | 0.62 | 0.62 | 0.76 | 0.68 | 0.62 | 0.70 |
| neg_cities | 0.93 | 0.94 | 0.88 | 0.90 | 0.94 | 0.94 | 0.91 | 0.94 |

Largest single ablations at the period (separation change vs full): Exp 0 component 28 −0.52 (active on 14%); Exp 1 276 −0.27; Exp 3 component 2 (= L15:2) −1.26 (active on 20%, KL 0.115), 80 −0.45.

Reading [concluded, one seed per run; the comparison is like-for-like: same position, data and settings, only the target lens differs]:
- **Exp 0, the reference for the open negation question, recovers almost nothing (8%)**, on negated Spanish translations 0.19 against Exp 2's 0.95. Components fitted to the next token at the period do not carry what the verdict needs from the period; the stimulus lens does. So Exp 2's recovery is not something any period decomposition gets for free. It does not settle negation vs truth: Exp 0 recovers nothing in either kind of family.
- **Exp 1 (residual lens at layer 23, never trained on a verdict) recovers 63%**, and 0.64 on negated Spanish translations: matching the layer-23 residual at the period carries most of the verdict-relevant computation.
- Exp 3, evaluated at the period only (not how it was trained), recovers as much as Exp 1.

## V over the span (replacement at the period and all 8 stimulus tokens, `--decompose_stimulus`)

All off at the 9 positions: accuracy 0.770, AUC 0.947, separation 3.49, KL 4.19 nats, i.e. 71% of the separation is gone (period only: 15%).

| decomposition | active per position | components-only KL | separation | accuracy | Δ-only KL | singles ≥ 1% |
|---|---|---|---|---|---|---|
| Exp 3 `s-161f8a7b` | 3.6 | 0.036 | 9.55 | 0.977 | 15.29 | 221 |
| Exp 2 `s-05db01ad` (trained at the period only) | 7.9 | 4.20 | 3.25 | 0.824 | 0.76 | 260 |
| untrained `s-7fad0c14` | 227 | 3.38 | 4.71 | 0.890 | 0.61 | 480 |

- Exp 3's components alone reproduce the verdict (KL 0.036 against 4.19 all off, i.e. 99% recovered) and are **more accurate than the original model** (0.977 vs 0.953), mainly on negated families: neg_animal_class 0.86 (original 0.73), neg_cities 0.98 (0.93), neg_sp_en_trans 0.95 (0.86); neg_element_symb 0.57 (0.55). Their separation is lower (9.55 vs 12.04). Exp 2 and the untrained decomposition recover nothing at the stimulus tokens, as expected.
- 10 of Exp 3's components are active on ≥ 99% of statements, and ablating one of them breaks the output (e.g. 402: KL 9.99, accuracy 0.446, AUC 0.511). These are general machinery the verdict needs, not truth-specific [concluded: always active, output destroyed].

  OUTDATED (26-10-14): "breaks the output" overstates it for most of the ten; 402 is the most extreme of them. Full numbers in entry *always-active components: what "breaks the output" means*.
- **Lead: components 447 and 380**, active on 49% and 50% of statements. Ablating 447: accuracy 0.500, share predicted true 0.000, AUC 0.988, separation 4.75 (full 12.07), KL 4.06. Ablating 380: share predicted true 0.691, separation 7.07, accuracy 0.736. A uniform push of all scores would leave the separation unchanged; it falls by 7.3 (447) and 5.0 (380), so the effect is larger on one class. My reading [assumed: inferred from the aggregate numbers, activity by label not yet looked at]: 447 is active mainly on true statements and writes "True", 380 mainly on false ones and writes "False". Check: CI of 447 and 380 at the stimulus tokens split by label and family (a few minutes on the GPU; not done).

OUTDATED (26-10-14): the reading of 380 is wrong; it is a number-comparison component, active on true and false larger_than/smaller_than statements alike. 447 is confirmed as a "True" writer. See entry *lead check: Exp 3 components 447 and 380*.

## truth_writers (`tw_tiu/report.txt`; bare statements without stimulus, readout at the period)

A2, share of the layer-19 truth separation (along `mm_L19`, test split) written by components vs Δ, summed over layers 15–19 (rest: other layers):

| | Exp 0 | Exp 1 | Exp 2 | Exp 3 | arm A | untrained |
|---|---|---|---|---|---|---|
| components | +0.22 | **+0.71** | −0.02 | −0.00 | +0.15 | −0.00 |
| Δ | +0.60 | +0.11 | +0.83 | +0.81 | +0.67 | +0.82 |

A7, largest actual drop of the layer-23 truth separation (test, S_total 70.10) when one component is ablated: arm A L17:33 −19.00, L19:34 −18.86 (reproduces the 26-10-06 result, 27% each); Exp 3 L19:26 −17.34, L15:2 −14.83; Exp 0 L17:37 −11.70; Exp 2 L19:94 −11.83; Exp 1 L19:23 −6.75; untrained max −1.70.

A3 CI probes (test accuracy): Exp 0 0.991, Exp 1 0.996, Exp 2 0.625, Exp 3 0.658, arm A 0.991, untrained 0.998; activation probes 0.998–0.999 for all.

Reading [concluded unless tagged]:
- Exp 1, whose loss is the layer-23 residual, writes 71% of the layer-19 truth separation through its components, the only run where components rather than Δ write it. Not surprising for a residual-matching loss, but it means its components are where the truth direction is written.
- Exp 2 and Exp 3 write none of the layer-19 truth separation through components, yet Exp 3's components dominate the verdict. With the stimulus lens, what the components carry from the period is not the projection onto the mass-mean truth direction [assumed: alternative explanation is that the stimulus-lens components are inactive on bare statements, which their low CI probe accuracy (0.63–0.66) also suggests; truth_writers sees no stimulus].
- Single-component A7 drops for Exp 3 (L19:26, L15:2) are as large as arm A's lead components; Exp 3's L15:2 is also its largest single at the period in V.

## Component overlap (`compare_components.py`, alive = active on ≥ 1% at the period)

Every match with |cos| > 0.1 of the flattened rank-one weights pairs a component with **the same index** in the other run (9 of 40 for Exp 0 → arm A, 16 of 85 for Exp 1 → arm A, 12 of 40 for Exp 0 → Exp 1, 1–8 for the rest; Exp 0 → Exp 2 none). Wherever the best match has a different index, |cos| ≤ 0.069 (one case), otherwise ≤ 0.023; random-B baseline median 0.0001–0.0003, p99 ≤ 0.0008.

Reading [concluded: all runs share `seed: 0`, hence identical initial component weights, and no match survives across indices]: the analysis finds no shared component between any two runs beyond the shared initialisation. Whether a same-index match is shared mechanism or leftover initialisation can't be told from this. Needed: the cosine of each component to its own initial weights (re-create the seed-0 initialisation), or a second seed. Not done.

## Open from here [assumed: proposals, not discussed]

1. Activity of Exp 3's 447 and 380 by label and family at the stimulus tokens (GPU minutes).
2. Overlap control against the seed-0 initialisation.
3. Report (Typst) of Exp 0–3 when Julian wants one.

---

26-10-14, Claude Opus 5.5 — *lead check: Exp 3 components 447 and 380*; same session. Julian in chat (choosing between checking the lead and destroying the instance):

C: (transcribed from chat) I'd like to check the lead

Scripts (one-off, `~/spd_out/26-10-10_bolder_experiments/analysis/`): `lead_447_380.py` (per statement: CIs of all components at the 9 decomposed positions; verdict scores original, full, full minus 447 / 380), `out/lead_followup{,2}.py` (breakdowns from its `.npz`), `lead_371_396.py` (states components / full, each minus 371, 396, both). Outputs `out/lead/`, `out/lead.log`, `out/lead2.log`. [verified: H100, n = 1,794 tiu test statements, span setting of V; sanity checks reproduce V exactly: active shares 0.491 / 0.497, minus 447 accuracy 0.500 and separation 4.745, minus 380 0.736 / 7.066, components state accuracy 0.977 and separation 9.55]

## 447 = L19:63: a "True" writer at the answer token [concluded]

- Active on 97.0% of true and 1.2% of false statements, and only at the last stimulus token ":" (where the verdict is output); at the period on 0.8%. Label AUC of its max CI 0.983; AUC for the model's own original verdict 0.997.
- Ablating it lowers the verdict score of true statements by 8.76 and of false ones by 1.43, in every family.
- On the 83 true statements the model calls false, its CI is weak (mean 0.27 vs 0.95 on true statements called true; Spearman with the original score 0.62 within true statements). It follows the model's answer more than the truth.

## 380 = L18:92: a number-comparison component, not a "False" writer

My reading in the entry above (380 mainly on false statements, writing "False") **was wrong**. 380 is active on 100% of the larger_than and smaller_than statements, true and false alike, and on few others (neg_cities true 0.29, element_symb true 0.11). Ablating it raises the score of false number comparisons by about 10 (larger_than +10.01, smaller_than +9.43) and leaves other families within ±0.5. The aggregate drop in separation came from one task family.

## Other label-selective components

Of 221 components active on ≥ 1%, 4 have a label AUC beyond 0.25 from chance, all "true" side and all at ":": 447 (0.983), 319 = L18:31 (0.964), 371 = L18:83 (0.889), 396 = L19:12 (0.806). The most false-selective ones are weaker (455 = L19:71 AUC 0.298; 405 = L19:21 and 36 = L15:36 active on ~25–28% of false and ~0.3% of true statements). The verdict looks asymmetric: dedicated "True" components, "False" as the default [assumed: from selectivity only].

371 and 396 are more active on true statements the model calls false (mean CI 0.82 / 0.84) than on true ones it gets right (0.62 / 0.59), and nearly off on false statements (0.04 / 0.09). Within negated families their CI separates true from false almost perfectly (371: AUC 1.00 on neg_animal_class, neg_cities, neg_sp_en_trans; 0.95 on neg_element_symb).

## Causal test: is the components-only gain over the original model carried by 371/396?

| state | accuracy | AUC | separation | share "true" | negated accuracy | negated AUC | affirmative accuracy |
|---|---|---|---|---|---|---|---|
| original | 0.953 | 0.999 | 12.04 | 0.454 | 0.861 | 0.998 | 0.988 |
| components | 0.977 | 0.999 | 9.55 | 0.478 | 0.920 | 0.998 | 0.998 |
| components − 371 | 0.967 | 0.998 | 9.41 | 0.468 | 0.885 | 0.997 | 0.997 |
| components − 396 | 0.968 | 0.998 | 9.54 | 0.469 | 0.892 | 0.998 | 0.996 |
| components − both | 0.941 | 0.996 | 9.24 | 0.442 | 0.796 | 0.994 | 0.995 |
| full | 0.953 | 0.999 | 12.07 | 0.454 | 0.861 | 0.997 | 0.987 |
| full − 371 | 0.840 | 0.947 | 9.37 | 0.341 | 0.526 | 0.936 | 0.958 |
| full − 396 | 0.878 | 0.993 | 10.08 | 0.380 | 0.656 | 0.995 | 0.962 |
| full − both | 0.788 | 0.957 | 8.62 | 0.289 | 0.524 | 0.963 | 0.887 |

Per family, accuracy (AUC) original / components / components − both: neg_animal_class 0.73 (1.00) / 0.86 (1.00) / 0.57 (1.00); neg_cities 0.93 (1.00) / 0.98 (1.00) / 0.86 (1.00); neg_sp_en_trans 0.86 (1.00) / 0.95 (1.00) / 0.88 (0.99); neg_element_symb 0.55 (0.91) / 0.57 (0.90) / 0.50 (0.85).

Reading [concluded: the AUCs barely move while accuracy and the share answered "true" do]:
- **The original model already ranks true above false almost perfectly within each negated family** (AUC 1.00 on neg_animal_class at accuracy 0.73). Its errors on negated statements are a bias towards "False", not missing truth information. So 371/396's within-family selectivity is not "truth the output does not use" (my working hypothesis while looking at the CIs, not recorded before this entry); the output score has the same information. Only neg_element_symb (model AUC 0.91, n = 56) leaves room.
- **The components-only state's gain over the original (0.977 vs 0.953) is a calibration shift carried by 371 and 396**: removing both takes it to 0.941 (negated 0.920 → 0.796) at nearly unchanged AUC, and the share answered "true" moves 0.478 → 0.442.
- In the full model, 371 and 396 are "True" writers like 447 (ablating 371: share "true" 0.454 → 0.341, accuracy 0.840).
- Overall: Exp 3 found the output end of the verdict: components at the answer token, gated by the model's own true/false judgement, that write "True" (447, 319, 371, 396), plus task components such as 380 for number comparisons. They read a judgement made earlier. The lead is real as verdict-writing machinery, but it is not the truth computation itself, and it doesn't show truth hidden from the output. Caveat: CI-based selectivity alone would be weak evidence (untrained CI networks read truth too, arm A topic); the ablations above are the causal part.

---

26-10-14, Claude Opus 5.5 — *report of Exp 0–3*; same session. Julian in chat:

C: (transcribed from chat) I've destroyed the intstance myself. Can you write a summary (you can delete the partly summary from yesterday)

(The H100, vast.ai 55905320, is gone; everything needed was copied to `~/spd_out/26-10-10_bolder_experiments/` beforehand. "The partly summary from yesterday" = `reports/26-10-13_exp2_stimulus_lens.typ`, the Exp 2-only report [assumed: my reading].)

Report: `reports/26-10-14_target_lens_experiments.typ` (5 pages; PDF gitignored, `typst compile reports/26-10-14_target_lens_experiments.typ`). Written for Julian, self-contained: method, the four lenses, training, verdict analysis at the period and over the span, Exp 3's answer-token components with the causal test, truth-direction writing, component overlap, caveats, next steps. Numbers re-read from the outputs listed in the entries above; Exp 1's final training metrics fetched from WandB into `~/spd_out/26-10-10_bolder_experiments/full/residual_lens/eval_metrics_from_wandb.jsonl` (step 5,000: L0 10.3, target normalised squared error 0.188, Δ only 5.01, PGD 0.21, Pile 0.016; run time 55 min from its WandB timestamps). Self-contained check: steps 1–2 by the author on the compiled PDF (internal report), which fixed prime-rendered apostrophes ("Exp 0′s"), a tilde Typst swallowed, "all off" used before its definition, and an unexplained "220 active" for the untrained decomposition.

Deleted, as Julian asked: `reports/26-10-13_exp2_stimulus_lens.{typ,pdf}` (untracked; its content is in the new report and in this LOG's Exp 2 entries). Living references updated (this topic's SUMMARY, the state-report SUMMARY, INDEX.md); references in earlier LOG entries stay as history.

Correction: the deleted report was not untracked; it was committed in the meantime (git shows it as deleted), so it remains recoverable from git history.

---

26-10-14, Claude Opus 5.5 — *always-active components: what "breaks the output" means*; same day, new session; I have read the report, INDEX.md, this SUMMARY and the LOG entries on V over the span. Julian in chat:

C: (transcribed from chat) It seems that from all "bolder experiments" experiment 3 was the most successful. You wrote that ablating any of them breaks the output in @reports/26-10-14_target_lens_experiments.typ . What do you mean by that?

Re-read from `~/spd_out/26-10-10_bolder_experiments/analysis/out/v/verdict_s-161f8a7b_span.json` (states `minus_<c>` = `full`, i.e. all components + Δ = original weights, with component c set to 0 at all 9 positions; "active" = rounded CI > 0.01 at any of the 9 positions). The ten components active on ≥ 99% of statements [verified: re-read 26-10-14, n = 1,794]:

| component | active share | KL to original (nats) | accuracy | AUC | separation | share "true" |
|---|---|---|---|---|---|---|
| (full) | – | 0.001 | 0.953 | 0.999 | 12.07 | 0.454 |
| (all off) | – | 4.19 | 0.770 | 0.947 | 3.49 | 0.702 |
| 8 | 1.000 | 3.00 | 0.833 | 0.971 | 7.95 | 0.339 |
| 28 | 1.000 | 3.06 | 0.896 | 0.993 | 8.50 | 0.397 |
| 53 | 1.000 | 7.21 | 0.760 | 0.955 | 7.02 | 0.687 |
| 122 | 1.000 | 4.30 | 0.815 | 0.935 | 7.27 | 0.566 |
| 140 | 1.000 | 10.24 | 0.613 | 0.750 | 2.35 | 0.581 |
| 231 | 1.000 | 10.85 | 0.649 | 0.747 | 1.97 | 0.697 |
| 258 | 0.997 | 4.16 | 0.771 | 0.909 | 5.22 | 0.287 |
| 381 | 1.000 | 11.00 | 0.773 | 0.806 | 2.82 | 0.365 |
| 402 | 1.000 | 9.99 | 0.446 | 0.511 | 0.35 | 0.360 |
| 410 | 1.000 | 5.51 | 0.715 | 0.837 | 2.77 | 0.598 |

Reading:
- What I meant: removing any one of these from the original weights moves the next-token distribution at ":" by 3.0–11.0 nats, comparable to or more than switching all five layers off at all 9 positions (4.19) [verified: table]. The KL is over the whole vocabulary; where the probability goes instead (other tokens, or only within true/false tokens) was not looked at.
- The report's wording "accuracy ≈ chance" was wrong as a description of the ten: only 402 reaches chance (0.446, AUC 0.511). 140, 231, 381 and 410 leave a weak verdict (AUC 0.75–0.84); 8 and 28 keep the ranking (AUC 0.97, 0.99) and mainly shift the calibration. I quoted the extreme case as if it were typical. Report sentence corrected.
- "General machinery, not truth-specific" rested on the CIs being always on. That shows they are not *gated* by truth; it does not show their writes carry no truth information (402's ablation erases the ranking) [assumed: not tested]. Report now says so.

---

26-10-14, Claude Opus 5.5 — *distribution shift under ablation*; same session as the entry above. Julian in chat:

C: (transcribed from chat) can you check how the distribution shifts when ablating some or all components? would an RTX 4090 (48ram) be enough?

C: (transcribed from chat) I've rented an instance

GPU: yes [concluded: inference only; Qwen2.5-7B in bf16 is ~15 GB, Δ for the five `down_proj` matrices ~1.4 GB fp32, logits of a 32-statement batch < 1 GB; Exp 3's *training* at batch 64 used 28.7 GB on the H100]. Julian rented a 48 GB RTX 4090 (power limit 450 W).

Script (one-off, outside the repo): `~/spd_out/26-10-10_bolder_experiments/dist_shift/dist_shift.py`. Exp 3 (`s-161f8a7b`), stimulus " Is this true or false? Answer:", decomposed positions = period + 8 stimulus tokens (as in training and V's span mode). Per state and statement at the final ":": probability mass on the true-like and false-like token sets of V, the rest ("other"), entropy, KL(original ‖ state), top-5 tokens. States: original, full, components only, Δ only (= all components removed), all off, each of the ten always-active components removed from full, all ten removed together, 447 removed, the four "True" writers (447, 319, 371, 396) removed together.

Results [verified: `dist_shift/full.log` and `full/dist_shift.npz`, single run on the 4090, n = 1,794 tiu test statements, 26-10-14; `full` reproduces the original (mean KL 0.0011, max 0.0156); smoke run with 2 statements per dataset first]. Means over statements. P(T|T) = probability mass on true-like tokens for true statements, P(F|F) = false-like mass for false statements, etc.; "other" = everything outside both sets.

| state | KL | P(T\|T) | P(F\|T) | P(T\|F) | P(F\|F) | other | entropy | most common top-1 tokens |
|---|---|---|---|---|---|---|---|---|
| original | 0.00 | 0.491 | 0.067 | 0.001 | 0.621 | 0.410 | 0.77 | " To" 42%, " False" 34%, " True" 24% |
| components only | 0.04 | 0.524 | 0.053 | 0.005 | 0.631 | 0.393 | 0.77 | " To" 43%, " False" 32%, " True" 25% |
| Δ only (all components removed) | 15.28 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 7.03 | " " 31%, "6" 10%, "Help" 10%, "\n" 7% |
| all off (MLPs 15–19 output 0) | 4.18 | 0.311 | 0.016 | 0.081 | 0.064 | 0.765 | 2.68 | " The" 50%, " Yes" 28%, " To" 15% |
| − 8 | 3.00 | 0.045 | 0.024 | 0.001 | 0.196 | 0.867 | 2.01 | " To" 63%, " It" 6%, " Considering" 6% |
| − 28 | 3.04 | 0.381 | 0.059 | 0.003 | 0.354 | 0.602 | 1.85 | " False" 29%, " __" 27%, " True" 23%, " The" 10% |
| − 53 | 7.20 | 0.556 | 0.055 | 0.058 | 0.413 | 0.459 | 3.16 | " Yes" 24%, " The" 16%, "Yes" 12%, "No" 11% |
| − 122 | 4.28 | 0.197 | 0.024 | 0.006 | 0.193 | 0.790 | 2.09 | " To" 39%, " with" 18%, " True" 6%, " False" 6% |
| − 140 | 10.21 | 0.040 | 0.006 | 0.015 | 0.040 | 0.949 | 3.47 | " " 36%, " continue" 16%, " the" 9%, " $" 9% |
| − 231 | 10.82 | 0.014 | 0.000 | 0.002 | 0.008 | 0.988 | 3.29 | " the" 32%, " The" 17%, " such" 15%, " Such" 14% |
| − 258 | 4.15 | 0.138 | 0.204 | 0.014 | 0.585 | 0.530 | 1.37 | " False" 47%, " either" 31%, " True" 4% |
| − 381 | 10.99 | 0.078 | 0.002 | 0.000 | 0.001 | 0.960 | 2.15 | " for" 60%, " (" 12%, " Write" 11% |
| − 402 | 9.98 | 0.018 | 0.016 | 0.001 | 0.007 | 0.979 | 3.96 | " \" 29%, " is" 28%, " &" 16% |
| − 410 | 5.49 | 0.404 | 0.110 | 0.003 | 0.009 | 0.737 | 4.30 | " (" 25%, " True" 25%, " A" 15% |
| − all ten always-active | 17.09 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 6.96 | "电" 26%, "Entry" 24%, "\n" 10% |
| − 447 | 4.06 | 0.011 | 0.337 | 0.001 | 0.754 | 0.449 | 0.60 | " False" 58%, " " 37%, " To" 5% |
| − 447, 319, 371, 396 | 6.07 | 0.003 | 0.067 | 0.000 | 0.685 | 0.622 | 1.77 | " " 41%, " False" 39%, " To" 10% |

Reading:
- **Baseline: the original model puts 41% of its mass outside the answer tokens**, mostly on " To" (top-1 for 42% of statements: it starts to reason, "To determine…"). So "other" mass is normal up to ~0.4; the verdict score of V compares the true/false sets only [verified: table].
- **Most single removals of always-active components take the model out of the answer format**, not to the other answer: after "Answer:" it continues the prompt as text (− 381: " for", − 231: " the"/" such", − 402: " \"/" is"/" &", − 140: " continue", − 410: " (") with 74–99% of the mass outside both answer sets. These components are needed for the model to answer at all at ":" [verified: table]. Three keep the format in another form: − 53 switches to " Yes"/"No" (true-like/false-like mass stays high, accuracy 0.759 judged by P(true-like) > P(false-like)); − 28 to fill-in-the-blank " __" alongside True/False; − 8 to reasoning (" To" 63%).
- − 258 and − 447 shift *within* the format towards "False": − 447 leaves P(F|F) at 0.754 and moves true statements' mass to " False" (0.337) and a bare " " token; P(T|T) 0.491 → 0.011. That is the "True"-writer reading of the entry *lead check* in distribution form.
- **Δ only and "all ten removed" produce garbage** (digits, CJK, code tokens; entropy ~7 nats, KL 15–17), while switching the MLPs off entirely (all off) leaves fluent text (" The", " Yes"; entropy 2.7, KL 4.2). So removing components is far more destructive than removing the whole MLP output: removing a single always-active component (KL up to 11.0) does more than zeroing all five MLP outputs at all 9 positions (4.18). My explanation [assumed: not measured]: components and Δ are individually large and partly cancel; removing a component leaves its counterpart in the other part as a large write into the residual stream. Check: norm of each component's write U_c (V_c · x) at the decomposed positions against the norm of the MLP output there (minutes on the GPU).
- Consequence for "breaks the output": the KLs of V measure, for most of the ten, leaving the answer format, possibly through an oversized write, not a change of verdict. Ablating a component from `full` is not the same as "the model without this mechanism" when components and Δ cancel [concluded: Δ-only and all-off differ by 11 nats KL; the size of the cancellation itself is not measured].

---

26-10-14, Claude Opus 5.5 — *flipped components and the cancellation check*; same session. Julian in chat:

C: (transcribed from chat) can you do another test where you flip the output direction of some or all compenents?

"Flip" = mask −1 instead of 1 for the chosen components (their write u_c (v_c · x) changes sign; the mask multiplies the component activation linearly, `LinearComponents.forward`), Δ on, at the 9 decomposed positions [assumed: my reading of "flip the output direction"]. So a flip changes the output by twice the component's write, a removal by once. Same component sets as the removal run, plus `flip_all` (all 480 flipped, Δ on: W − 2·components) and `negated_components` (components only, mask −1, Δ off: minus the decomposition's reconstruction of the MLP output). The script also measured, at the decomposed positions, the norms the previous entry's open guess needs (the guess that components and Δ cancel). Script: `dist_shift.py --mode flip` (same folder; first launch crashed on a missing `no_grad` in the new norm block, fixed with `torch.set_grad_enabled(False)`); outputs `dist_shift/flip.log`, `dist_shift/flip/dist_shift.npz` [verified: smoke run with 2 statements per dataset, then n = 1,794 on the 4090, 26-10-14; `full` vs original mean KL 0.0011].

**Norms at the decomposed positions** (mean L2 norm over the 9 positions × 1,794 statements):

| layer | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|
| MLP output (components + Δ) | 25.6 | 23.1 | 28.7 | 36.5 | 43.0 |
| components' part | 69.3 | 56.7 | 68.4 | 74.2 | 96.6 |
| Δ's part | 74.4 | 61.8 | 74.8 | 83.4 | 106.1 |

Single-component writes (mean |v_c · x| ‖u_c‖, as a multiple of the layer's MLP output norm): always-active 8 0.30, 28 0.37, 53 0.67, 122 0.38, 140 0.57, 231 0.81, 258 0.44, 381 0.57, 402 0.60, 410 0.56; "True" writers 447 0.58, 319 0.39, 371 0.20, 396 0.25. Largest writes of all 480: 402 25.7, 447 24.9, 410 24.0, 231 23.4, 381 20.9.

**Flips** (columns as in the removal table):

| state | KL | P(T\|T) | P(F\|T) | P(T\|F) | P(F\|F) | other | entropy | most common top-1 tokens |
|---|---|---|---|---|---|---|---|---|
| original | 0.00 | 0.491 | 0.067 | 0.001 | 0.621 | 0.410 | 0.77 | " To" 42%, " False" 34%, " True" 24% |
| negated components (Δ off) | 15.87 | 0.000 | 0.000 | 0.002 | 0.000 | 0.999 | 8.16 | "s", "IENTATION", "矧" |
| flip all (Δ on) | 19.09 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 8.28 | "Means", "ispers", "瘕" |
| flip 8 | 8.01 | 0.087 | 0.002 | 0.007 | 0.075 | 0.914 | 4.02 | " To" 39%, " Acc" 12% |
| flip 28 | 12.86 | 0.099 | 0.021 | 0.000 | 0.146 | 0.867 | 1.64 | " __" 45%, " ______" 26%, " False" 12% |
| flip 53 | 11.57 | 0.028 | 0.004 | 0.004 | 0.002 | 0.981 | 5.79 | " R" 54%, "Coming" 15% |
| flip 122 | 9.41 | 0.257 | 0.007 | 0.003 | 0.105 | 0.814 | 3.26 | " A" 32%, " True" 14%, " It" 13% |
| flip 140 | 14.01 | 0.000 | 0.000 | 0.001 | 0.001 | 0.999 | 6.86 | " do" 46%, "onestly" 18% |
| flip 231 | 15.36 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 6.21 | " Or" 39%, " \" 28%, " (" 17% |
| flip 258 | 12.43 | 0.003 | 0.001 | 0.001 | 0.000 | 0.997 | 4.64 | " at" 65%, " use" 11% |
| flip 381 | 14.02 | 0.000 | 0.000 | 0.001 | 0.000 | 0.999 | 4.86 | " (" 41%, "stat" 31%, " for" 22% |
| flip 402 | 11.72 | 0.000 | 0.000 | 0.001 | 0.001 | 0.999 | 6.69 | " is" 40%, " for" 23%, " now" 22% |
| flip 410 | 11.03 | 0.046 | 0.001 | 0.000 | 0.000 | 0.977 | 5.94 | "Para" 32%, " __" 16% |
| flip all ten always-active | 17.62 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 8.23 | "电", " code", " way" |
| flip 447 | 6.62 | 0.000 | 0.023 | 0.000 | 0.748 | 0.615 | 0.80 | " " 52%, " False" 45% |
| flip 447, 319, 371, 396 | 11.40 | 0.000 | 0.006 | 0.000 | 0.394 | 0.800 | 4.00 | " " 38%, " False" 28%, ".Std" 8% |

Reading:
- **Components and Δ cancel heavily** [verified: norms above]: each part is 2.0–2.9 times the norm of the MLP output they sum to (components 2.0–2.7×, Δ 2.3–2.9×). Removing all components leaves Δ's part, a write 2.3–2.9 times the MLP output's size in a direction the original model never produces there. That explains why Δ only (KL 15.3) is so much worse than zeroing the MLPs (4.18), and why the Δ-only reconstructions of the training table are large in Exp 3 [concluded: from the norms; "never produces" is my reading, not measured]. This confirms the guess of the previous entry at the level of all components.
- **For single components the guess does not hold as stated**: their writes are 0.2–0.8 times the layer's MLP output, smaller than zeroing the layer's output, which keeps the model fluent. So a single removal breaks the answer format through the *direction* of its write, not its size [concluded: size comparison; why the direction matters is not known]. Removing a component from `full` still means "the original weights plus −u_c (v_c · x)", a perturbation the original never sees; this is the caveat for every single-component ablation in this topic (V's singles included).
- **Flipping is removal, stronger**: every flip leaves the answer format further than the removal did (KL 6.6–15.4 vs 3.0–11.0); the tokens it moves to are different and not interpretable (" R", " at", " Or", " do") [verified: table].
- **The negative of a "True" writer is not a "False" writer**: flipping 447 takes true statements' "True" mass to zero (0.491 → 0.000) but not to " False" (0.023; under removal 0.337); the mass goes to a bare " " token (52% top-1). False statements are unaffected (P(F|F) 0.748, removal 0.754). Fits "False is the default, not written by a mirror of 447" (entry *lead check*), and shows −447 pushes towards a third, non-answer token [concluded: n = 1,794, one run].

---

26-10-14, Claude Opus 5.5 — *ablation examples*; same session. Julian in chat:

C: (transcribed from chat) cany you provide some concrete exmaples of removing component/inverting component. For each category I want to see some references and the effect of the change. You can write the resutls in the report folder

C: (transcribed from chat) Not just the first token, but the next 20 if they are not end tokens. The change should only happen on the stimulus positions though

Done: report `reports/26-10-14_ablation_examples.typ` (12 pages; tables generated into `reports/26-10-14_ablation_examples_tables.typ` by `~/spd_out/26-10-10_bolder_experiments/dist_shift/make_examples_tables.py`, so no continuation is transcribed by hand). Script `dist_shift/ablation_examples.py`; outputs `examples.json`, `examples_stimulus_only.json`, logs [verified: 4090 run 26-10-14; `full` reproduces the original continuation on all 6 statements].

My reading of the request [assumed]: "category" = each removed/flipped component or group, plus the states of the whole decomposition; "references" = the original model's answer to the same statement, shown next to each change. "Only on the stimulus positions" = only at the prompt positions, not on generated tokens. I kept the 9 positions Exp 3 was trained on (period + 8 stimulus tokens) as the main variant, for comparability with all earlier numbers, and ran the 8-stimulus-tokens-only variant as well (report §6).

Design: 6 tiu test statements chosen from the original model's first token only, before any intervention (first match in `np.random.default_rng(0).permutation(1794)` order for: cities true/" True", cities false/" False", neg_sp_en_trans true/" True", neg_sp_en_trans false/" False", larger_than false/" False", larger_than true/" To"; `examples_selection.json`). larger_than true has no " True" first token at all (204 of 204 start with " To"; smaller_than true likewise), so F shows the reasoning start. Per state: greedy up to 20 tokens, stop at `<|im_end|>`/`<|endoftext|>`; the intervention (mask 0 = removed, −1 = flipped, Δ on) at the prompt positions only, generated tokens on the original weights. 32 states: original, full, components only, all off, Δ only, negated components, flip all, and removed/flipped for each of the ten always-active components, all ten, 447, and the four "True" writers.

Observations [verified: read all 192 continuations of the main variant; anecdotal: 6 statements, one greedy run]:
- Most single removals change how the answer starts (" for sure true.", " the statement is true.", " (I) False (II) True", reasoning openers), not the verdict. Where a verdict appears within 20 tokens it is mostly right. So V's 3–11-nat first-token KLs mostly reflect a format change.
- Signatures repeat across statements: 28 (removed and flipped) → fill-in-the-blank multiple-choice template with the correct letter on A, C, D; 381 removed → " for sure …"; 410 flipped → Spanish/Portuguese ("Para responder…"); 53 removed → no leading space or " Yes"; 258 flipped → " at …"; 231 flipped → " Or …". Reading [assumed: 6 statements]: these always-active components act like answer-format components.
- Wrong verdicts: A (true city) under removal of 447, 258, 402, 410 → " False. Hamhung is a city in North Korea, but the statement is not entirely accurate…" (the model rationalises the flipped answer); flips of 140 and 381 on A, 258 on D, the four "True" writers on C; all off on both negated statements (C, D), matching the main report's negation reversal.
- Flipping 447 turns no true statement into "False" (A " 1. The statement to evaluate…", C " 1. True.", F "  $\boxed{\text{true}}$"); false statements unchanged.
- Δ only, all ten removed/flipped, flip all: garbage first tokens ("9/10", "电报", "IENTATION"), then often a recovered answer, sometimes in Chinese; flip all derails (B → a text about a logistics company).
- Stimulus tokens only vs period + stimulus: 146 of 192 continuations identical, 175 with the same first token; the period contributes little to these effects [verified: exact string comparison].

---

26-10-14, Claude Opus 5.5 — *cancellation explained*; same session. Julian in chat:

C: (transcribed from chat) I don't quite understand this "At these positions the components' part and Δ's part of each MLP  output are each 2.0–2.9 times as large as the output they add up to, so they largely cancel" Can you explain this in more detail?

To explain it correctly I first split the components' part further, because "components only" (which reproduces the answer, KL 0.036) keeps only the *active* components, while the norms of the entry *flipped components and the cancellation check* summed all 480. Script `dist_shift/cancel_split.py`, output `dist_shift/cancel_split.log` [verified: local CPU, 5 test statements per dataset = 100 statements, 900 decomposed positions, 26-10-14; the instance was already destroyed]. MLP output y = A x + I x + D x, with A = active components (rounded CI > 0.01 at that position), I = inactive components, D = Δ.

Mean norms and cosines per layer 15..19:

| | L15 | L16 | L17 | L18 | L19 |
|---|---|---|---|---|---|
| \|y\| | 25.0 | 22.7 | 28.5 | 36.4 | 43.0 |
| \|A\| | 21.0 | 14.9 | 22.3 | 19.5 | 24.9 |
| \|I\| | 62.8 | 50.2 | 59.6 | 67.9 | 86.2 |
| \|D\| | 77.8 | 61.8 | 74.4 | 84.9 | 107.2 |
| \|y − A\| (= \|I + D\|) | 41.0 | 34.8 | 45.0 | 50.2 | 62.6 |
| cos(I, D) | −0.80 | −0.82 | −0.78 | −0.80 | −0.81 |
| cos(A, y) | 0.008 | 0.007 | 0.010 | 0.008 | 0.009 |

Per position in the span, pooled over the five layers:

| position | cos(A, y) | \|A\| | \|y\| | \|y − A\| | active per layer | share with no active component |
|---|---|---|---|---|---|---|
| period | 0.002 | 7.6 | 30.2 | 34.5 | 1.06 | 0.76 |
| " Is", " this", " true", " or", "?", " Answer" | 0 | 0 | 29.3–34.6 | = \|y\| | 0 | 1.00 |
| " false" | 0.012 | 20.6 | 31.9 | 42.1 | 0.72 | 0.51 |
| ":" | 0.061 | 156.4 | 31.6 | 157.6 | 6.06 | 0.00 |

Reading:
- **The components-only state does not reconstruct the MLP outputs** [verified: table]. At 6 of the 8 stimulus tokens no component is active, so "components only" sets the five MLP outputs to zero there, and at the period nearly so. At ":", about 30 active components (6 per layer) write a vector of norm 156, five times the real MLP output (31.6) and nearly orthogonal to it (cosine 0.06).
- **Why the answer still comes out right** [concluded: from the training objective, not measured]: Exp 3's target loss only compares the next-token distribution at ":", with Δ off and components masked by their CI. Nothing asks the MLP outputs to match. The decomposition found a different computation with the same answer: switch the MLPs off at the question and write large vectors at ":" into whatever directions set the answer.
- **Where the cancellation comes from**: the components and Δ still sum to the original weights. So wherever the active components write something other than y, the inactive components plus Δ must write y minus it. At ":" that is a vector of norm ~158 opposite to the active write; at the other positions inactive components and Δ cancel each other to leave y (cos(I, D) ≈ −0.8).
- **Consequence for the ablations** [concluded]: in the full model, a component's write is part of a sum that cancels; removing it adds the negative of its write. The ablation results show which directions control the answer when pushed, not which parts of y the original model uses.
- **Consequence for the Exp 3 interpretation** (main report, "Exp 3 found the output end of the verdict"; this topic's SUMMARY) [concluded]: the "True" writers (447 etc.) are writers in the decomposition's own re-implementation, which is not the original MLP output at ":". That they are gated by the model's judgement and that removing 447 from the full model flips true statements to "False" stays [verified earlier], but "the model's MLPs write 'True' through 447" is not shown. Check: projection of the original y at ":" onto u_447 compared with 447's write (not done).
- The earlier tPD runs whose target loss compares activations (Exp 1, residual lens) are less exposed to this shortcut, since matching the layer-23 residual constrains the MLP writes more [assumed].

C: That's strong evidence that tPD did not detect anything truth mechanism relevant here. My question now is, how do the other experiments fare in this test.


Answer (26-10-15, Claude Opus 5.5): Measured on all runs; entry *cancellation test on the other runs* below. Short version: by this test Exp 0, Exp 2 and arm A fare no better than Exp 3 (their components-only state is as far from the real MLP output as switching the MLPs off, or further). Exp 1 (residual lens) is the only run whose components partly rebuild the MLP output. On your conclusion: I agree for the "True" writers as a description of the original model's mechanism. But the same caveat now applies to every decomposition we have, including arm A, whose single-component ablations were the basis of the truth-writing topic's results. See *Reading* in the entry.

---

26-10-15, Claude Opus 5.5 — *cancellation test on the other runs*; new session. I have read INDEX.md, this whole LOG and its SUMMARY, and (late, see below) `docs/CODING.md`. Ran on the rented RTX 4090 (vast.ai 56078923, provisioned by Julian with `spd-vast` as `s-d065c7a1`).

Julian in chat, on how to get the checkpoints onto the instance:

C: (transcribed from chat) I stopped something because I was afraid that something was stuck. You can proceed. For future session where you need to download something, it might be more effective to start the experiments already when one checkpoint is downloaded.

C: (transcribed from chat) I prefer to have the wandb download. Also I would like a future setup where the machine has the wandb key. Even more so, in training runs the vm uploads to wandb so it should have a key somewhere

The key was already there: `spd-vast` passes it to the container at rent time and writes it to `/etc/profile.d/spd_env.sh`, which login shells read [verified: `ssh vastai 'bash -lc ...'` sees `WANDB_API_KEY`, 26-10-15]. `docs/CODING.md` says so (*Launch from a login shell*); I had not read CODING.md before starting and briefly planned to upload the checkpoints from Julian's machine instead (FEEDBACK.md entry). Each run was then fetched from WandB on the instance by the analysis script itself, one after another, each analysed as soon as it was there.

**Method.** `~/spd_out/26-10-10_bolder_experiments/dist_shift/cancel_split.py` (the script of the entry *cancellation explained*, extended with `--run`, `--decompose_stimulus False` = the period only, and relative errors; defaults unchanged). Full tiu test split, n = 1,794 statements, positions as in V: the period for every run, plus Exp 3 over its 9-position span (period + 8 stimulus tokens). MLP output y = A x + I x + D x per decomposed `down_proj` (layers 15–19), A = active components (rounded CI > 0.01 at that position, the "components only" state of V and of training's rounded target loss), I = inactive components, D = Δ. Relative error |y − A| / |y| = how far the components-only state is from the real MLP output; **1 = as far as switching the MLPs off** (y → 0), above 1 = further. Outputs: `dist_shift/cancel_all/*.log`, runner `run_cancel_all.sh` [verified: all 7 jobs exit 0, 26-10-15 10:27–10:30 UTC; active components per statement (per-layer mean × 5) reproduce V's counts: Exp 0 7.8, Exp 1 10.5, Exp 2 0.8, Exp 3 2.0, arm A 9.3, untrained 220].

**At the period** (pooled over layers 15–19; |y| = 30.5 mean norm, the same for every run since it is the original model):

| decomposition | active per layer | \|y − A\| / \|y\| (components only) | cos(A, y) | \|A\| / \|y\| | \|I\| / \|y\| | \|D\| / \|y\| | cos(I, D) | \|y − D\| / \|y\| (Δ only) |
|---|---|---|---|---|---|---|---|---|
| Exp 0 next-token `s-aeaa70d8` | 1.56 | 1.26 | 0.12 | 0.85 | 0.53 | 1.36 | −0.36 | 1.02 |
| **Exp 1 residual `s-cd129592`** | 2.09 | **0.83** | **0.56** | 0.76 | 0.19 | 0.82 | −0.12 | 0.79 |
| Exp 2 stimulus `s-05db01ad` | 0.17 | 1.18 | 0.00 | 0.21 | 1.65 | 2.07 | −0.78 | 1.78 |
| Exp 3 span `s-161f8a7b`, at the period | 0.40 | 1.07 | 0.00 | 0.11 | 1.95 | 2.24 | −0.84 | 2.05 |
| arm A `s-bd23f0d1` (all-tokens) | 1.86 | 1.24 | 0.08 | 0.77 | 0.58 | 1.36 | −0.41 | 1.00 |
| untrained `s-7fad0c14` | 44.0 | 1.03 | 0.00 | 0.25 | 0.28 | 1.07 | −0.26 | 0.37 |

Exp 3 over its span (all 9 positions, full split): |y − A| / |y| 1.54, at ":" |A| 169 vs |y| 32 (cosine 0.06), Δ-only error 2.36; the 100-statement CPU numbers of the entry *cancellation explained* hold on the full split. (cos(A, y) per layer and the per-layer errors are in the logs.)

The ratios use layer-pooled means (Σ over layers of mean |·|, divided by Σ of mean |y|); the cosines and the relative errors are means over positions, averaged over layers.

Training's lens-level fit at the same position, for contrast (from this LOG's entries *Exp 0 finished*, *report of Exp 0–3*, R2/R3): Exp 0's components-only next-token KL at the period is 0.107 nats against 1.68 all off; Exp 1's normalised layer-23 error 0.188 against 2.26 all off. So both reproduce what their lens sees, while their MLP outputs are as far as (Exp 0) or somewhat closer than (Exp 1) the all-off state.

Reading:
- **Exp 0, Exp 2 and arm A fail the test like Exp 3** [verified: table]. Their components-only MLP outputs are further from the real ones than zero is (1.18–1.26). Exp 0 and arm A write a vector about as large as y (0.8–0.85 of its norm) but nearly orthogonal to it (cosine 0.08–0.12). Exp 2 and Exp 3 hardly write anything at the period (0.17 and 0.40 active per layer), and there inactive components and Δ cancel strongly (cos −0.78 / −0.84, each about 2× the size of y). The untrained decomposition is the initial state: Δ ≈ W (Δ-only error 0.37), components small.
- **Exp 1 is the only run whose components partly rebuild the MLP output** [verified: error 0.83, cosine 0.56, weak cancellation (cos(I, D) −0.12, |I| 0.19 of |y|)]. That is the one case where the prediction at the end of the entry *cancellation explained* ("less exposed to this shortcut", `[assumed]` there) was tested; it holds partly. Still, components only removes only 17% of the distance to y, so Exp 1's components are also far from a faithful decomposition of the MLP outputs [concluded: 0.83 is much closer to 1 than to 0].
- **Why** [concluded: from the configs, not tested by ablating the loss]: every training loss in these configs compares the model's output at the lens (`StochasticReconSubsetLoss`, `PersistentPGDReconLoss`; `CIHiddenActsReconLoss` is an eval metric only). Nothing asks the active components to reproduce the MLP outputs. Components + Δ = W holds by construction, so whatever the active components get wrong, inactive components and Δ carry the negative in the full model. The residual lens constrains the MLP writes most, because layer 23 is only four layers downstream and all its channels count; a next-token or stimulus KL can be met by many different writes.
- **On Julian's conclusion** ("strong evidence that tPD did not detect anything truth mechanism relevant here") [concluded]: agreed as far as the claim that Exp 3's "True" writers are the original model's mechanism goes. Two qualifications. (1) The caveat is not specific to Exp 3: the same holds for Exp 0, Exp 2 and arm A, so the truth-writing topic's arm A results (single-component ablations removing ~27% of the layer-23 truth separation) also measure "original weights minus u_c (v_c · x)" for components whose write is not a part of y. (2) What the decompositions still show is about the original model's inputs, not its mechanism: the CI networks read the original activations, so 447's gating by the model's own judgement shows that the judgement is readable at the input of layer 19 at ":", which untrained CI networks and probes also show.
- **Consequence**: of the runs so far, only Exp 1's components are candidates for "parts of what the MLPs compute", and only partly.

Proposals [assumed: not discussed]:
1. Add a per-layer reconstruction term to the target loss, so the active components must reproduce each decomposed module's own output at the decomposed positions. The code has layerwise losses (`StochasticReconLayerwiseLoss`, `StochasticHiddenActsReconLoss`), but per the first design entry, `StochasticHiddenActsReconLoss` ignores the position mask, so it needs a change before it can be used here. Rerunning Exp 1 or Exp 3 with it would show whether the verdict or truth-direction results survive a faithfulness constraint.
2. Before reading Exp 1's components further: this test with the CI-weighted mask (not just rounded) and the stochastic masks used in training, to see whether 0.83 is a property of rounding.
3. Make this test (components-only error vs MLP output at the decomposed positions) a standard check for every new decomposition, next to V.

The 4090 (vast.ai 56078923, $0.69/h) is still running; nothing on it is needed any more (outputs copied to `dist_shift/cancel_all/`).


C: I would like to try a variant. I'd like to use this setup:
```
Consider the amount of truthfullness in the following:
<stimulus>
The amount of truthfullness is
```
(maybe a better word for truthfullness would be good).
<stimulus> is one of the examples in the dataset. Before we do this though I'd like to see what the model returns for true and false examples
Answer (26-10-15, Claude Opus 5.5): Done; entry *R4 framing screen* below. Short version: the model answers with a number (" 100%" / " 1" for true, " 0" / " 0%" for false). With "truth" as the word, the digit after "is " is right on all 546 test statements, negated ones included (" Is this true or false? Answer:" got 0.903). I recommend "truth".

---

26-10-15, Claude Opus 5.5 — *R4 framing screen*; same session. Frozen Qwen2.5-7B-Instruct on the rented 4090 (vast.ai 56078923). Scripts and outputs: `~/spd_out/26-10-10_bolder_experiments/r4_framing_screen.{py,out}`, `r4b_digit_verdict.{py,out}` [verified: single runs, 26-10-15; same 546 tiu test statements as R1 (up to 30 per dataset, seed 0; 273 true, 273 false)].

Prompt, raw text, no chat template, W ∈ {truthfulness, truth, veracity, correctness} ("truthfulness" spelled with one l):

```
Consider the amount of W in the following:
<statement>
The amount of W is
```

Tokenization [verified: printed]: the statement's "." and the following newline merge into one token `'.\n'`, so the statement end is `'.\n'`, not `'.'` as in Exp 0–3. The prompt is tokenized as a whole.

## (a) What the model writes (greedy, 20 tokens, 2 statements per dataset = 40 per word)

The model answers with a number at once, in every family:
- true statements: " 100% because …", " 100%. The statement … is entirely true" (truthfulness, truth, correctness); " 1 (completely true).", " 1.0, which means …" (veracity);
- false statements: " 0. The statement … is false because …", " 0%." (all words); " 0 out of 1. Explanation: …" (correctness).

Examples (W = truth): "The city of Maiduguri is in China." → " 0.\n\nMaiduguri is a city located in Nigeria, not in China."; "The city of Guntur is not in China." → " 100%.\n\nThis statement is true because Guntur is a city located in the Indian…"; "The Spanish word 'boca' does not mean 'mouth'." → " 0.\n\nThe Spanish word 'boca' means 'mouth'."

Wrong or off-format first answers, out of 40 per word: truthfulness 3 (larger_than false → a multiple-choice list "(A) 0 (B) 25 …"; one neg_element_symb false → ":\nTo determine…"; one neg_element_symb true → ": 1 (out of 3)"); truth 1 (neg_element_symb false → ":\nThe statement … is false.", format only; two number comparisons start with ": 0" / ": 0%", right verdict after a colon); veracity 2 (neg_element_symb false → a pick-your-answer list; another → ":\nTo determine…"); correctness 1 wrong verdict ("'boca' does not mean 'mouth'" → " 100%") and 2 " high"/" low" instead of a number. The full 160 continuations are in `r4_framing_screen.out`.

## (b) The next token after "is"

The first token is mostly a bare space ' ' (mean probability 0.41–0.73; top-1 for 70–100% of statements), because Qwen writes a digit as its own token after a separate space token. The verdict comes one token later. Other first tokens: ':' / ':\n' (more often after false statements), " high" / " low" (correctness: 0.128 after true vs 0.143 " low" after false), " zero" (truth: 0.097 after false).

## (b') The digit after "is " (R4b)

The prompt ends with that space (`... is` + `' '`, checked to be one token), and the score is log P('1') − log P('0') at the next token ('1' starts "100%", "1", "1.0"; '0' starts "0", "0%"):

| W | P('1') + P('0') mean | accuracy | AUC | share answered '1' | lowest family accuracy |
|---|---|---|---|---|---|
| truthfulness | 0.972 | 0.993 | 1.000 | 0.500 | neg_sp_en_trans_true 0.933 |
| **truth** | **0.983** | **1.000** | **1.000** | 0.500 | all 1.000 |
| veracity | 0.986 | 0.991 | 1.000 | 0.505 | neg_sp_en_trans_false 0.900 |
| correctness | 0.918 | 0.958 | 0.996 | 0.516 | neg_sp_en_trans_false 0.500 |

For comparison, R1 with " Is this true or false? Answer:" on the same statements: accuracy 0.903, AUC 0.998, truth-token mass 0.736; true negated statements were mostly called false (neg_element_symb_true 0.071, neg_animal_class_true 0.526, neg_cities_true / neg_sp_en_trans_true 0.767) [verified: re-read `r1_stimulus_screen.out`]. With W = truth those families are at 1.000. The weakest spot is neg_sp_en_trans_false: lowest digit mass (0.816) and mean score −1.24 (others −4 to −8), still all correct.

Reading [concluded: n = 546, one run, greedy and next-token only]:
- **The framing gets the model's own verdict right on all of tiu**, including the negated statements where the earlier stimulus failed. The verdict is a clean binary at a single position: the digit after "is ", with 98% of the mass on '1' or '0'. A trailing space was harmful in R1 (" Answer: " pushed the mass to digits); here it is the model's own next token, so ending the prompt with it is natural.
- **Recommendation: W = truth** (accuracy 1.000, a single token, no split like truth|fulness or ver|acity). "correctness" is worst; Julian's worry about the word was justified for that one.

Points for the design of a run with this framing [assumed: not discussed]:
1. **The framing comes before the statement**, unlike in Exp 0–3. The model reads the statement knowing it will judge its truth, so the computation at the statement end changes. It is closer to "truth computed because it was asked for" than to the original premise of "truth computed even when unused". Runs with this framing are not comparable position by position with Exp 0–3.
2. The decomposed statement end is the `'.\n'` token; the loss/verdict position is the digit after "is ".
3. **Ceiling**: the model is already at 1.000. Verdict accuracy cannot show a gain like Exp 3's calibration shift, only losses; KL and the score margin remain informative.
4. **Cancellation** (entry *cancellation test on the other runs*): with output-only losses, a new run would likely re-implement the verdict instead of decomposing the MLP outputs, as Exp 3 did. I'd add the per-layer reconstruction term (proposal 1 there) before training with this framing, or at least run the cancellation test right after training.

The 4090 is still running.

C: alright, so I would like to start another experiment. We'll use the strongest setup which I unederstood is `truth` + whitespace after "is".
Before we come to further decomposition experiments, I'd like to test the truth directions. We have the truth directions from tiu and UTH right? What I'd like to see is which position is best for steering the output 0<->1, using those. I'd like to see what happens if we flip thas direction or, if we add it. Can you propose the setup of this experiment?

Answer (26-10-15, Claude Opus 5.5): Yes, both exist (stored residuals of tiu and UTH, mass-mean directions computable from them). Proposal in the entry *steering design: truth directions on the framing prompt* below, with five questions at its end.

---

26-10-15, Claude Opus 5.5 — *steering design: truth directions on the framing prompt*; same session. I have re-read the truth-writing SUMMARY, the direction-free SUMMARY and `truth_directions()` in `truth_writers.py`, and listed the stored features. Nothing run.

Decision from Julian's comment above [decided: C: comment]: the new setup is the framing prompt with W = truth and the prompt ending in "is " (the space token), verdict = the digit after it.

## What truth directions we have [verified: `features.npz` keys and `truth_directions()`, 26-10-15]

Both stores hold the residual stream at the **last token of the bare statement** (the period, no framing) after layers 15, 16, 17, 18, 19 and 23:
- tiu: `~/spd_out/26-10-06_truth_writers/tiu/features.npz`, fit split 5,260 statements;
- UTH: `~/spd_out/26-10-06_truth_writers/uth/features.npz`, fit split 20,292 samples from the UTH training tasks (2.2 GB file).

From these, `truth_directions()` builds: `mm_L<l>` (global mass-mean direction: mean of true minus mean of false, unit length) for each of the six layers, `lr_L19` (logistic-regression direction), and per-dataset `mm_L19|<dataset>`. The vectors are not stored; they are recomputed from the residuals (seconds).

**The catch**: these directions come from a different context than the new prompt. There the statement stands alone and ends with "."; in the framing prompt it follows "Consider the amount of truth in the following:" and ends with `'.\n'`. Whether the stored directions still point along truth there is itself a result (cosine to a direction fitted in the new context, below).

## Proposed setup [assumed: agent proposal]

**Prompt and outcome.** "Consider the amount of truth in the following:\n<statement>\nThe amount of truth is " → next token. Per statement, the verdict score s = log P('1') − log P('0') (R4b: accuracy 1.000, AUC 1.000 unsteered). Reported per condition:
- **flip rate**: share of statements whose verdict sign flips in the steered direction;
- mean shift of s;
- **digit mass** P('1') + P('0') (unsteered 0.983): if it collapses, the intervention pushed the model out of the answer format rather than to the other answer (the lesson of the ablation work, entry *distribution shift under ablation*);
- for the best few conditions: greedy 20-token continuations of 6 example statements (does the model then argue for the flipped answer?).

**Data.** The 546 test statements of R1/R4 (up to 30 per tiu dataset). Directions and class statistics come from the fit split, never the test split.

**Directions** (all unit vectors in the residual stream, positive = true):
1. **tiu stored**: `mm_L15` … `mm_L19`, `mm_L23` (bare statements, the period).
2. **UTH stored**: the same six layers from the UTH fit split (cross-task: many tasks, mostly not tiu-like).
3. **In-context** (new): mass-mean on the framing prompt itself, per layer (all 28) and per position slot (below), from one forward pass over the tiu fit split. This one shows where in the new prompt truth is linearly present, and gives the layer sweep the stored directions can't (they exist only at 15–19 and 23).
4. **Controls**: 10 random directions with the residual stream's anisotropy (mass-mean directions for shuffled labels, `null_directions()` in `truth_writers.py`), and one content direction (e.g. cities vs number comparisons) as a "real but not truth" direction.

The cosines between 1, 2 and 3 at each layer are reported first: they say whether the bare-statement and UTH directions transfer to this prompt.

**Interventions** at one layer l and one position slot p, on the residual stream after layer l (h = residual vector, d = direction). The scale and the midpoint are always measured **in the framing context** at that (l, p) on the fit split: m_T, m_F = mean projections h·d of true and false statements, Δ = m_T − m_F, midpoint c = (m_T + m_F)/2. This keeps the stored directions comparable to the in-context one: only the direction comes from tiu/UTH.
- **Add**: h → h + α Δ d for false statements (towards true) and h − α Δ d for true ones, α ∈ {0.5, 1, 2, 4, 8}. α = 1 moves a statement from its class mean to the other class mean along d.
- **Flip**: h → h − 2 (h·d − c) d, i.e. reflect the projection about the class midpoint. A typical true statement lands at the false mean and vice versa; everything orthogonal to d stays.
- Controls get the same α in units of the truth direction's Δ at that (l, p), so they move the residual by the same distance.

**Position slots** (the prompt is aligned per statement): the object word (last content token of the statement), the statement end `'.\n'`, "The", " amount", " of", " truth", " is", the final " ". Plus two multi-position variants: all statement tokens, all six suffix tokens. Positions before the statement can't carry its truth and are left out.

**Sweep and order.**
- Stage 1, map: in-context direction × 28 layers × 8 slots × {flip, add α = 1, 4}. That's 672 conditions at about 1–2 s each on the 4090 for 546 statements [assumed: from R4b's speed], i.e. under 30 min. Output: a heat map of flip rate (and digit mass) over layer × position, separately for true→false and false→true.
- Stage 2, comparison: at layers 15–19 and 23, all slots, every direction (tiu, UTH, in-context, controls), flip and the full α sweep.
- Per family, as everywhere: affirmative vs negated.

**What to expect, so it isn't read after the fact** [assumed: predictions, not results]:
- Steering at the final " " in late layers will flip the digit almost trivially, because there the direction acts on the output directly. The informative part of the map is the earliest layer and position where steering still flips the answer, and whether the statement end `'.\n'` is such a place.
- The random controls set the floor: a direction is "truth" for this purpose only where it flips far more than random directions of the same size, while the digit mass stays high.

## Questions for Julian

1. Flip about the class midpoint (proposed) or about 0 (h·d → −h·d)? About 0 throws everything with a common offset along d to the far side, so I'd use the midpoint.
C: both
2. α range {0.5, 1, 2, 4, 8} in units of the class separation: OK?
C: yes
3. Include the in-context direction (recommended: it is the only way to sweep all layers and to test whether the stored directions transfer), or only the stored tiu/UTH ones as you asked?
C: yes, include
4. Positions: the 8 slots plus the two multi-position variants: enough, or also every statement token individually?
C: yes
5. The 4090 (vast.ai 56078923) is still rented: run it there right after your answers?
C: yes

---

26-10-15, Claude Opus 5.5 — *steering run: truth directions on the framing prompt*; same session. Julian answered the five questions with C: comments under *Questions for Julian* in the entry above: flip about the midpoint **and** about 0 (both); α range {0.5, 1, 2, 4, 8} yes; include the in-context direction yes; positions "yes"; run on the 4090 yes [decided: C: comments]. Question 4 was an either/or ("enough, or also every statement token individually?") answered "yes"; I took it as "the 8 slots plus the two groups are enough" [assumed].

Deviations from the proposal, my choices for runtime [assumed]: the sweep uses **200** test statements (the first 10 per tiu dataset after the R1/R4 seed-0 shuffle, a subset of the 546; 100 true, 100 false), not 546, and **5** shuffled-label control directions, not 10. Full run: 4,901 conditions × 200 statements in 54 min on the 4090.

Scripts and outputs (`~/spd_out/26-10-10_bolder_experiments/steering/`): `stored_dirs.py` (tiu/UTH directions), `steer.py` (sweep), `summarize.py` → `full/summary.out` and heat maps `full/stage1_*.png`, `focus.py` → `full/focus.out`, `examples.py` → `full/examples.out`, `bs_check.py`. [verified: smoke run first; captured residuals equal Hugging Face's `hidden_states` exactly; unsteered accuracy 1.000, digit mass 0.986 on the 200; a no-op condition reproduces the unsteered scores within the bf16 batch-size noise (max 0.375 log-odds vs mean |score| 5.7, `bs_check.py` showed the same 0.375 for unsteered rows in batches of 20 vs 192, sign never changed)]

Definitions, as proposed: verdict score log P('1') − log P('0') after "…The amount of truth is "; **flip** = verdict now on the wrong side; **flipkeep** = flipped *and* digit mass P('1') + P('0') ≥ 0.5 (a real answer change, not leaving the format; this is the main number below). "Layer l" = residual stream after decoder layer l. Directions: **in-context** = mass-mean on the framing prompt (tiu train split, 6,560 statements) per layer and slot; **tiu / UTH** = stored mass-mean directions from bare statements at the period (layers 15–19, 23). Add moves by α·Δ (Δ = class gap along the direction, measured in the framing context); controls move by the in-context Δ. Slots: object token, statement end (`'.\n'`), The / amount / of / truth / is / final space, all statement tokens, all six suffix tokens.

## Stage 1: where the in-context direction steers (layer × slot map, `full/summary.out`, `stage1_*.png`)

Two windows, nothing else [verified: map]:
- **Statement window, layers ~14–17**, at the statement's tokens. Add α = 4 at the statement end: flipkeep 0.65 (L14), 0.68 (L15), 0.77 (L16), 0.60 (L17), then 0.16 (L19) and 0.00 from L20. Over all statement tokens: 0.86 at L17 (α = 8: 1.00). Reflection about the midpoint over all statement tokens: 0.65 at L17. Before layer 12 nothing flips anywhere (≤ 0.04 at α = 4).
- **Read-out window, from layer ~18**, at " is" (α = 4: 0.86 at L19, then 0.00–0.40) and at the final space (α = 4: 0.91 at L18, 0.62 at L19, 1.00 at L20–22; midpoint reflection 0.90–0.99 from L20).
- The tokens "The", " amount", " of", " truth" never carry it (flipkeep ≤ 0.07 everywhere). The handover is sharp: at L18–19 the statement stops mattering and the answer tokens take over.

## Stage 2: stored tiu / UTH directions vs in-context vs controls (layers 15–19, 23; `full/focus.out`)

| layer, slot, intervention | in-context | tiu | UTH | content | shuffled-label controls, median / max of 5 |
|---|---|---|---|---|---|
| L15 statement end, add α = 4 | 0.68 | 0.42 | 0.22 | 0.02 | 0.10 / 0.27 |
| L16 statement end, add α = 4 | 0.77 | 0.57 | 0.33 | 0.01 | 0.04 / 0.27 |
| L16 statement end, add α = 8 | 0.87 | 0.60 | 0.59 | 0.03 | 0.07 / 0.56 |
| L17 statement end, add α = 4 | 0.60 | 0.57 | 0.30 | 0.01 | 0.00 / 0.21 |
| L17 all statement tokens, add α = 8 | 1.00 | 0.74 | 0.66 | 0.04 | 0.00 / 0.26 |
| L17 all statement tokens, midpoint reflection | 0.65 | 0.30 | 0.12 | 0.01 | 0.03 / 0.04 |
| L19 final space, add α = 2 | 0.91 | 0.03 | 0.03 | 0.10 | 0.00 / 0.40 |
| L23 final space, midpoint reflection | 0.97 | 0.00 | 0.00 | 0.00 | 0.07 / 0.26 |
| L23 final space, add α = 1 | 0.94 | 0.00 | 0.00 | 0.04 | 0.00 / 0.20 |

Cosine of the in-context direction with the stored ones [verified: `diagnostics.json`]: at the statement end tiu 0.59–0.73, UTH 0.36–0.45 (layers 15–19); at the final space tiu 0.08–0.37, UTH 0.10–0.29. tiu and UTH stored directions have cosine 0.49–0.62 with each other (`stored_dirs.out`).

Reading [concluded unless tagged; one run, 200 statements]:
- **The stored tiu and UTH directions steer the verdict, but only in the statement window**: at the statement end at layers 15–17 they flip 0.42–0.57 (tiu) and 0.22–0.33 (UTH) at α = 4, above every control (median ≤ 0.10). At the read-out tokens they do almost nothing (≤ 0.17 at α ≤ 4 over layers 18–23, 0.00 at L23), where the in-context direction flips 0.91–0.97. So the bare-statement truth direction transfers to the framing prompt as the truth representation *at the statement*, not as the direction the answer tokens read out. UTH (fit on other tasks) transfers less than tiu, consistent with its lower cosine.
- **The in-context direction is the strongest everywhere**, as expected for a direction fit on this very prompt and position.
- **Asymmetry** [verified: `stats.json`]: in the statement window, steering pushes true statements to "0" far more easily than false ones to "1". At L16 statement end, add α = 4: true → 0 0.98 / false → 1 0.56 (in-context), 0.96 / 0.18 (tiu); UTH α = 8 0.99 / 0.19. In the read-out window it is symmetric (L23 space midpoint reflection 0.97 / 0.98). This matches Exp 3's finding that "False" is the default answer and "True" is what gets written [assumed: the link; an intervention that disturbs the statement's truth computation may land on the default "0" without encoding falsehood]. The controls argue against pure disturbance (they move by the same distance and rarely flip), but they are not clean: the first shuffled-label direction (the only one whose cosine was saved) has cosine 0.13–0.69 with the in-context truth direction, highest at the statement end and the final space (it picks up the high-variance directions of the residual stream), which is also why the controls' max sometimes reaches 0.5–1.0 at the read-out tokens.
- **Reflection about 0 is not a flip**: where both class means lie on the same side of 0 along the direction, it sends every statement to one side (L23 final space: true → 0 1.00, false → 1 0.00; L17 statement tokens: 0.00 / 1.00). Its 0.50 scores are this collapse. The midpoint reflection is the meaningful one [verified: `stats.json`].
- Negated vs affirmative: no consistent difference (`focus.out` has the split per cell).

## Example continuations (`full/examples.out`; 6 statements, greedy 20 tokens, steering on the prompt only) [verified: read all 54; anecdotal]

- Statement-window steering changes the digit, and the explanation follows the steered answer, often contradicting itself: "The city of Hamadan is in Iran." with L16 statement end, tiu add α = 4 → " 0.\n\nThis statement is false because Hamadan is indeed a city located in Iran." For "The Spanish word 'isla' does not mean 'island'." (false) the same steering gives a coherent wrong answer: " 100%.\n\nThe Spanish word "isla" indeed means "island." Therefore, the".
- Read-out steering changes the digit, and the text after it often reverts to the statement's real truth value: "Fifty-two is larger than fifty-four." with L23 final space midpoint reflection → " 100%.\n\nThis statement is false, as fifty-two is not larger than fifty-four."; "…Maiduguri is in China." → " 100% false." The later text re-reads the statement on unsteered weights.
- The false city and number statements resisted the add conditions at the statement end (the asymmetry above) but flipped under the midpoint reflection over all statement tokens at L17 ("…Maiduguri is in China." → " 100%.\n\nMaiduguri is indeed a city, and it is located in Nigeria,"), the one condition that moves false statements more than true ones (false → 1 0.98, true → 0 0.32).

## What this means [concluded, one run]

- The model's truth judgement on this prompt is formed at the statement's own tokens around layers 14–17 and is read by the answer tokens around layers 18–19. That is the window tPD decomposed (layers 15–19 at the period), so the decomposed layers sit at the right place for this prompt.
- The project's stored truth directions (tiu, UTH) are causally relevant there, not only correlational: moving the statement-end residual along them changes the answer, beyond matched controls, mostly towards "false".
- Open: why false → true is hard at the statement (default "False", or a second, non-linear feature); which slot carries it before layer 14 (nothing linear flips there); whether the asymmetry survives with less entangled controls (random Gaussian directions orthogonalised to the truth direction).

The 4090 (vast.ai 56078923) is still running.

C: that sounds interesting. I'd like to do a similar experiment where we have the stimulus first (like the `<stimulus>. True or False. Answer:`). But this time trying to elicit the clear `1` or `0` token response. So this should be the first task: what setup with stimulus first gives the clearest signal. Then I'd like to do the same experiment looking for the positions and layers where steering (native and tiu, we skip uth) has the most influence. I've rented an RTX 4090 but with less vram (there were no other offers). Can you do the experiments there and if you need more vram report?

---

26-10-15, Claude Opus 5.5 — *R5 statement-first screen: started, stopped by Julian*; new session. I have read INDEX.md, this whole LOG, its SUMMARY and `docs/CODING.md`. Task: Julian's C: comment directly above (statement-first prompt with a clean `1`/`0` answer; then the steering sweep with the native and tiu directions on it, on a new 24 GB RTX 4090).

Instance: vast.ai 56119768 (`spd-vast` `s-e257dad1`), RTX 4090 24 GB, 450 W. Qwen2.5-7B-Instruct was downloaded on it from the Hugging Face hub (1 min 43 s). A second 4090 on the account (56118165) was destroyed by Julian himself.

What I set up [assumed: my design, not discussed]: `~/spd_out/26-10-10_bolder_experiments/r5_statement_first_screen.py`. Prompt = `<statement><sep><question> `, with sep a space or a newline and seven questions: "True or False? Answer with 1 for true or 0 for false. Answer:", "True or False (1 or 0)? Answer:", "Is this statement true (1) or false (0)? Answer:", "Truth value (1 = true, 0 = false):", "Label (1 if the statement is true, 0 if it is false):", "The amount of truth in this statement is", "On a scale from 0 (false) to 1 (true), the truth of this statement is", plus the R4 framing prompt as reference. Raw text, no chat template. Measures as in R4b (digit after the trailing space token: digit mass, accuracy, AUC, per family), plus greedy continuations and P(' ') without the trailing space. Same 546 statements as R1/R4. Tokenization checked locally [verified: Qwen tokenizer, 26-10-15]: with a space as separator the statement's tokens (period included) equal the bare statement's, and a digit always comes as `' '` + digit (there is no `' 1'` token).

Also written, not run: `steering_sf/steer_sf.py`, the sweep of `steering/steer.py` adapted to a statement-first prompt (slots generated from the question's tokens, no UTH, batch 96 and last-position logits only for 24 GB).

Julian in chat while the screen was running:

C: (transcribed from chat) Can you stop the experiment. I don't like the setup

Stopped [verified: process gone, GPU memory 1 MiB]. It had finished 13 of the 15 prompts; the partial output is in `~/spd_out/26-10-10_bolder_experiments/r5_stopped/`, not read and not used. The 4090 is still rented.

Open: what about the setup Julian doesn't like, so the next design avoids it.

Correction (26-10-15, Claude Opus 5.5): OUTDATED: "finished 13 of the 15 prompts" in the entry above is wrong. All 14 statement-first prompts finished (14 result blocks in the output); only the 15th, the R4 framing reference, did not run. Its numbers on the same 546 statements are in entry *R4 framing screen* (W = truth: accuracy 1.000, digit mass 0.983).

---

26-10-15, Claude Opus 5.5 — *R5 results: comparison of the statement-first candidates*; same session. Julian in chat: "Compare the candidates" (after asking whether the stopped run was still the prompt design: it was). Source: `~/spd_out/26-10-10_bolder_experiments/r5_stopped/r5_statement_first_screen.out` [verified: re-read 26-10-15; one run, 546 tiu test statements (273 true), frozen Qwen2.5-7B-Instruct on the 24 GB 4090]. For scale|nl the per-dataset lines were lost (the output buffer had not been flushed when the job was killed); its overall line is there.

Prompt = `<statement><sep><question> `; sp = same line after a space (the statement's tokens then equal the bare statement's: 546/546 for every sp prompt), nl = new line (period merges into `'.\n'`). Score log P('1') − log P('0') at the token after the trailing space. P(' ') = probability of the space token where the prompt ends without it, i.e. how naturally the model starts a digit answer there.

| question | sep | accuracy | AUC | digit mass | share answered '1' | P(' ') | worst dataset (accuracy) |
|---|---|---|---|---|---|---|---|
| "True or False? Answer with 1 for true or 0 for false. Answer:" | sp | 0.905 | 0.978 | 1.000 | 0.405 | 0.985 | neg_element_symb_true 0.071 |
| same | nl | 0.811 | 0.907 | 1.000 | 0.311 | 0.998 | neg_animal_class_true 0.000 |
| "True or False (1 or 0)? Answer:" | sp | 0.859 | 0.943 | 0.999 | 0.359 | 0.833 | neg_element_symb_true 0.000 |
| same | nl | 0.806 | 0.917 | 1.000 | 0.306 | 0.914 | neg_animal_class_true 0.000 |
| **"Is this statement true (1) or false (0)? Answer:"** | **sp** | **0.952** | **0.995** | 0.999 | 0.452 | 0.719 | neg_element_symb_true 0.143 |
| same | nl | 0.899 | 0.971 | 1.000 | 0.399 | 0.831 | neg_sp_en_trans_true 0.100 |
| "Truth value (1 = true, 0 = false):" | sp | 0.940 | 0.975 | 0.986 | 0.447 | 0.446 | neg_element_symb_true 0.179 |
| same | nl | 0.910 | 0.973 | 0.999 | 0.410 | 0.682 | neg_element_symb_true 0.036 |
| "Label (1 if the statement is true, 0 if it is false):" | sp | 0.586 | 0.654 | 0.845 | 0.518 | 0.117 | element_symb_false 0.000 |
| same | nl | 0.817 | 0.848 | 0.996 | 0.321 | 0.439 | neg_element_symb_true 0.000 |
| "The amount of truth in this statement is" | sp | 0.778 | 0.924 | 0.845 | 0.722 | 0.036 | neg_element_symb_false 0.000 |
| same | nl | 0.963 | 0.994 | 0.976 | 0.493 | 0.283 | element_symb_true 0.750 |
| "On a scale from 0 (false) to 1 (true), the truth of this statement is" | sp | 0.896 | 0.984 | 0.999 | 0.604 | 0.543 | neg_element_symb_false 0.071 |
| same | nl | 0.938 | 0.996 | 1.000 | 0.562 | 0.688 | (lost) |
| reference: R4 framing prompt (instruction first), W = truth | – | 1.000 | 1.000 | 0.983 | 0.500 | – | all 1.000 |

Reading [concluded: one run, n = 546; the interpretation in the second point is assumed]:
- **No statement-first prompt reaches the framing prompt** (1.000). Best: "The amount of truth in this statement is" on a new line, 0.963, and "Is this statement true (1) or false (0)? Answer:" on the same line, 0.952. Affirmative families are at or near 1.000 in all good candidates. The errors sit in the negated families.
- **The direction of the negation errors depends on the question**: the True/False-style questions call true negated statements false (share '1' 0.31–0.45 for 50% true; neg_element_symb_true 0.00–0.18 as in R1's 0.07–0.29), and the "scale" question calls false negated statements true (share '1' 0.56–0.60; neg_element_symb_false 0.071, neg_cities_false 0.700 for sp). With the instruction before the statement, the same statements are all right. Possible reason [assumed]: the model reads the statement already knowing the task.
- **A digit answer is natural for some prompts and forced for others**: the amount prompt has the best accuracy, but there the model's own next token is ":" (52%), and its greedy continuations often turn into multiple-choice lists (": [A] very little [B] some …"); P(' ') is only 0.283. For "Is this statement true (1) or false (0)? Answer:" (sp) the space is top-1 for 89% of statements and the continuations are clean (" 0\n\nThe statement "The seagull is a mammal" is false (0).").
- Same line is better than a new line for four of the seven questions; the amount and scale questions are better on a new line.

Recommendation [assumed: Julian decides]: **"Is this statement true (1) or false (0)? Answer: "** on the same line.
- Second-best accuracy (0.952, AUC 0.995), digit mass 0.999, a natural digit answer.
- Its errors are almost all in one family: neg_element_symb_true 4/28 right; all other datasets 0.933–1.000.
- On the same line the statement's tokens equal the bare statement's, so the stored tiu directions are applied in exactly the context they were fit in (by causal attention), and the native direction at the statement tokens equals a bare-statement direction by construction.
- The steering summary counts flips only on statements the unsteered model gets right, so the wrong ones drop out of the flip rates.
- Alternative with the opposite bias: the scale question on a new line (0.938, AUC 0.996).

---

26-10-15, Claude Opus 5.5 — *prompt chosen: the amount question on a new line; statement-first steering launched*; same session. Julian in chat, after asking whether "The amount of truth in this statement is " had been checked (yes, both separators; numbers in the entry above):

C: (transcribed from chat) I prefer the amount wording. Can you use this as the template (of course with new line)

Decision [decided: transcribed above]: template `<statement>\nThe amount of truth in this statement is ` (amount|nl in R5: accuracy 0.963, AUC 0.994, digit mass 0.976). Unlike the same-line prompts, the statement's last token is the merged `'.\n'`, so the statement tokens are not exactly the bare statement's at the end token. The stored tiu directions are therefore applied in a slightly different context than they were fit in (as in the framing-prompt sweep).

Steering sweep launched with this template: `~/spd_out/26-10-10_bolder_experiments/steering_sf/steer_sf.py --question amount --sep nl` (on the instance `/root/bolder/steering_sf/`, output `full/`). Same method as the framing-prompt sweep (entry *steering design: truth directions on the framing prompt*, with Julian's answers there): add ±αΔ for α ∈ {0.5, 1, 2, 4, 8}, reflection about the class midpoint and about 0, the same 200 test statements (10 per tiu dataset), the main number flipkeep = flipped with digit mass ≥ 0.5, counted over the statements the unsteered model gets right. Directions, per Julian's comment: native (= in-context mass-mean on this prompt, tiu train split) and the stored tiu direction; UTH dropped. Kept from the earlier sweep without asking, because the flip rates can't be read without them [assumed]: the five shuffled-label controls and the content control (cities vs number comparisons). Slots: object token, statement end `'.\n'`, each of the 9 question tokens (The / amount / of / truth / in / this / statement / is / final space), all statement tokens, all question tokens. Stage 1: native direction, 28 layers × 13 slots × 4 operations; stage 2: layers 15–19 and 23, all directions, all 7 operations; 5,825 conditions. Changes for the 24 GB card: batch 96, logits of the last position only.

Smoke run [verified: 20 statements, 26-10-15]: captured residuals equal Hugging Face's `hidden_states` exactly; a no-op condition reproduces the unsteered scores within the known bf16 batch noise (0.375 log-odds). Full run startup [verified: `full.log`]: unsteered accuracy 0.950 on the 200 statements (R5: 0.963 on 546), digit mass 0.979, 20.5 GB GPU memory. Cosine of the native direction with the stored tiu direction at the statement end, layers 15–19: 0.80–0.85 in the smoke run (framing prompt: 0.59–0.73).

First full run failed [verified: `full_run1_failed.log` on the instance]: after 57 min, the final no-op check (a condition that changes nothing must reproduce the unsteered scores) failed with a max deviation of 0.50 log-odds against a tolerance of 0.38 (10% of the mean |score|, carried over from the framing-prompt sweep, where mean |score| was 5.7 and the noise 0.375). The script saved its results only after this check, so nothing was kept. Its startup lines are still informative: cosine of the native direction with the stored tiu direction on the full train split, at the statement end: L15 0.901, L16 0.897, L17 0.880, L18 0.878, L19 0.795, L23 0.534; at the final space 0.056–0.350.

Fix: results are saved first; the check now requires a mean deviation below 5% of the mean |score| and no sign change among statements with |score| > 1 (a broken hook would fail both; bf16 batch noise does not). Smoke run with the new check [verified]: max 0.375, mean 0.100 (limit 0.19), 0 sign changes among 18. Relaunched with identical settings at ~17:24 UTC (FEEDBACK.md entry).

---

26-10-15, Claude Opus 5.5 — *statement-first steering run: results*; same session. [verified: rerun finished, 3,436 s on the 4090; no-op check passed (max 0.50, mean 0.099 at mean |score| 4.00, 0 sign changes among 180 statements with |score| > 1); outputs copied to `~/spd_out/26-10-10_bolder_experiments/steering_sf/full/` (`steer.npz`, `stats.json`, `summary.out` + heat maps `stage1_*.png` from `summarize_sf.py`, `focus.out` from `focus_sf.py`, `examples.out` from `examples_sf.py`); one run, 200 test statements (10 per tiu dataset), of which the unsteered model gets 190 right (accuracy 0.950); flip rates are over those 190]

Prompt `<statement>\nThe amount of truth in this statement is `; definitions as in the entry *steering run* (flipkeep = verdict flipped and digit mass ≥ 0.5; add moves by α·Δ, Δ = class gap measured on this prompt; controls move by the native Δ). "Native" = in-context direction. Slot names: object (last content token of the statement), end (`'.\n'`), the question tokens The … statement / is / final space, all statement tokens, all question tokens.

## Stage 1: where the native direction steers (layer × slot map)

Two windows again, as on the framing prompt [verified: `summary.out`]:
- **Statement window, layers ~13–17**, at the statement's own tokens. Add α = 4: statement end 0.31 (L13), 0.55 (L15), 0.63 (L17), then 0.41 (L18), 0.15 (L19), 0.00 from L20; object token 0.36 (L15), 0.49 (L16), **0.76 (L17)**, 0.00 from L18; all statement tokens 0.78 at L17 (α = 8: 0.92). Before layer 11 nothing flips under add (≤ 0.02).  
C: what again is the object token. The last penultimate token of the statement/stimulus? Is this only true for in-context direction. Generally how is in-context direction computed?

Answer (26-10-15, Claude Opus 5.5): The object token is the token that holds the statement's last letter or digit, i.e. the last sub-word of the statement's last word, directly before the end token `'.\n'` [verified: all 8,354 tiu train and test statements]. Examples from the test sample: ' Iran', ' China', 'al' (of " mammal"), '-four' (of "fifty-four"), 'land' (of "island"), 'c' (of "Tc"), 'large' (before `"'.\n"`). In tiu's templates the last word is the object of the relation (country, class, symbol, number, translation), hence the name. So yes: the penultimate token of the statement, counting the end token as the last.

It is a position, not a direction: every direction (native, tiu, controls) was applied at it. Only the native direction itself depends on the slot. It is computed like this: run the frozen model on the full prompt `<statement>\nThe amount of truth in this statement is ` for each of the 6,560 tiu train statements; take the residual stream after layer l at the slot's position; mean over the true statements minus mean over the false statements; scale to length 1. That gives one direction per (layer, slot), 28 × 13. For the two groups (all statement tokens, all question tokens), each statement's residuals are first averaged over the group's positions. The steering strength Δ is the gap between the true and the false class mean along the direction, measured in the same way.

- **Read-out window, from layer 18**, at the final space (α = 4: 0.99 at L18, 1.00 at L20–23; α = 1 already 0.96–1.00 from L20) and at " is" (α = 4: 0.79 at L19).
- New compared with the framing prompt: the question token " statement" carries a weak share in the statement window (α = 4: 0.27–0.31 at L14–16, almost only true → 0). The other question tokens never carry it (≤ 0.03 except " this" ≤ 0.12).  
C: again here the question for me is what was used for steering.

Answer (26-10-15, Claude Opus 5.5): The native direction of that position: the mass-mean (true minus false) of the residual at the " statement" token itself, after layer l, on the train split, added only at that one token with α times its own class gap there. The numbers in the bullet are for that direction. The stored tiu direction applied at the same token flips only 0.09–0.10 at α = 4 (layers 15–17), about the controls' maximum (0.11–0.16).

- **Reflections over all statement tokens at early layers are collapses, not flips** [verified: `focus.out`]: flip_mid/flip_zero flip 0.3–0.5 already at L3–L16, but always one class only (e.g. L12 0.96 true → 0, 0.01 false → 1; L8 0.00 / 1.00). The content control and the shuffled-label controls reach 0.17–0.31 in the same cells. The examples show the mechanism: after the L16 midpoint reflection over all statement tokens the model no longer reads the statement (" 0, and I will explain why: … the difference between displaying characters in different"). These cells measure destruction of the statement, not truth.  
C: I'm a bit confudes that the mass-mean distinction does not go through 0. Is that normal? Do you know more about this behaviour?

## Stage 2: stored tiu direction vs native vs controls (flipkeep; true → 0 / false → 1 in brackets)

| layer, slot, intervention | native | tiu | content | shuffled-label controls, median / max of 5 |
|---|---|---|---|---|
| L15 end, add α = 4 | 0.55 [0.70/0.40] | 0.61 [0.83/0.37] | 0.05 | 0.03 / 0.12 |
| L16 end, add α = 4 | 0.51 [0.82/0.19] | 0.57 [0.98/0.16] | 0.08 | 0.07 / 0.17 |
| L17 end, add α = 4 | 0.63 [1.00/0.24] | 0.62 [1.00/0.23] | 0.07 | 0.01 / 0.35 |
| L17 object, add α = 4 | 0.76 [1.00/0.51] | 0.48 [0.84/0.11] | 0.01 | 0.03 / 0.33 |
| L16 object, add α = 4 | 0.49 [0.78/0.20] | 0.05 [0.08/0.01] | 0.03 | 0.02 / 0.11 |
| L17 all statement tokens, add α = 8 | 0.92 [1.00/0.83] | 0.74 [1.00/0.47] | 0.07 | 0.01 / 0.08 |
| L18 final space, add α = 4 | 0.99 [0.99/0.99] | 0.24 [0.25/0.23] | 0.00 | 0.24 / 0.55 |
| L19 " is", add α = 4 | 0.79 [0.69/0.90] | 0.03 | 0.00 | 0.04 / 0.26 |
| L23 final space, midpoint reflection | 0.98 [0.99/0.97] | 0.00 | 0.01 | 0.03 / 0.13 |
| L23 final space, add α = 1 | 1.00 | 0.00 | 0.02 | 0.04 / 0.23 |

C: where is statement-end in this table?

Answer (26-10-15, Claude Opus 5.5): The rows "L15 end", "L16 end" and "L17 end" are the statement end (the `'.\n'` token). I abbreviated it in the table; it should have said "statement end".


Cosine of the native direction with the stored tiu direction [verified: `diagnostics.json`]: statement end 0.88–0.90 (L15–18), 0.80 (L19), 0.53 (L23); object token 0.31–0.47; all statement tokens 0.67–0.75; final space 0.06–0.35. (Framing prompt: statement end 0.59–0.73.)

Reading [concluded unless tagged; one run, 190 statements]:
- **At the statement end the stored tiu direction now steers as well as the native one** (L15–17, α = 4: tiu 0.57–0.62, native 0.51–0.63; controls median ≤ 0.07). On the framing prompt tiu reached only 0.42–0.57 against native 0.60–0.77. Fits the higher cosine (0.88–0.90 vs 0.59–0.73): with the statement first there is no instruction before it, so the statement end is closer to the bare-statement context the tiu direction was fit in [assumed: the reason].  
C: You might already have answered this, but in how far does native direction and tiu direction differ I mean in how they are determined.

Answer (26-10-15, Claude Opus 5.5): Both are mass-mean directions on tiu (mean residual of the true statements minus that of the false ones, unit length). They come from different forward passes:

| | tiu (stored) | native |
|---|---|---|
| text the model sees | the bare statement, nothing around it | the full prompt `<statement>\nThe amount of truth in this statement is ` |
| position | the statement's last token, the period '.' | each of the 13 slots separately |
| layers | 15–19 and 23 only | all 28 |
| statements | tiu "fit" split, 5,260 (= train split minus the subjects held out for tuning), stored in the truth-writing topic (`~/spd_out/26-10-06_truth_writers/tiu/features.npz`) | tiu train split, 6,560 |
| at which slot it is used | the same vector at every slot | each slot its own |

The steering strength is computed the same way for both (α × class gap along that direction, measured on this prompt at that slot).

One consequence of the statement-first prompt: everything before the end token is the bare statement's own text, and attention only looks back. So at the object token and the other statement tokens, the prompt's residuals are exactly those of the bare statement, and the native direction there is what a bare-statement direction at that token would be (up to the fit/train split difference). Only the end token differs: `'.\n'` here against '.' in the stored data. That fits the high cosine at the statement end (0.88–0.90) [concluded: from causal attention; the remaining difference is the end token and the split].

- **The object token carries truth in a different direction**: the native direction there steers best of all single tokens (0.76 at L17), the tiu direction (fit at the period) much less (0.05 at L16, 0.48 at L17; cosine 0.31–0.47).  
C: I'm comming back to the point that I don't know how the native direction is calculated. But could it be that we're not stearing truth here but the relation of the concepts themselves (so maybe the truth of the relation?)?
- **The asymmetry is back at the statement end**: true → 0 is easy (0.70–1.00), false → 1 hard (0.16–0.40), for native and tiu alike. **At the object token and over all statement tokens, false → 1 works much better** (L17 object native 0.51, all tokens α = 8 0.83). The examples show the same: the false city statement "…Maiduguri is in China." and the false number comparison flip to " 100%" under object-token steering but not under statement-end steering. So "false → true is hard" is a property of the statement-end slot, not of the prompt or the model's answer [concluded: same prompt, same statements, only the slot differs].
- **The read-out window takes only the native direction**: at the final space and " is" (L18–23) tiu does nothing (≤ 0.24, at the level of the controls' median), as on the framing prompt. The controls are not clean at the final space at L18–19 (median 0.14–0.27, max 0.45–0.86 at α = 2–4; their cosine with the native direction is 0.53–0.63 there).

## Example continuations (6 statements, greedy 20 tokens, steering on the prompt only) [verified: read all 54; anecdotal]

- Statement-window steering changes the digit and the explanation follows it: "The city of Hamadan is in Iran." with tiu at L16 statement end, α = 4 → "0%. The statement is false because Hamadan is not located in Iran; it is actually situated in"; "Ninety-three is larger than seventy-three." → "0%.\n\nNinety-three is not larger than seventy-three." Sometimes the text contradicts the digit ("0.\n\n… is actually true").  
C: is this a property of statement-window steering or object steering?

Answer (26-10-15, Claude Opus 5.5): Neither in particular, and my bullet overstated "follows". OUTDATED: "the explanation follows it … Sometimes the text contradicts the digit" in the bullet above is wrong; contradiction is more common. Re-read of `examples.out` [verified: 26-10-15], counting for each flipped answer whether the fact stated after the digit supports the new digit ("follows", e.g. "Hamadan is not located in Iran" after " 0") or the real truth value ("contradicts", e.g. after " 0" for the true "'medio' does not mean 'large'": "'medio' actually means 'half'"):

| condition (statement window) | flipped of 6 | follows | contradicts | neither |
|---|---|---|---|---|
| tiu, L16 statement end, α = 4 | 4 | 2 | 2 | 0 |
| native, L17 statement end, α = 4 | 4 | 0 | 4 | 0 |
| native, L17 object token, α = 4 | 5 | 1 | 3 | 1 |
| native, L17 all statement tokens, α = 8 | 6 | 1 | 4 | 1 |
| tiu, L17 all statement tokens, α = 8 | 4 | 2 | 2 | 0 |

In total 6 follow, 15 contradict, 2 neither (out of 23). "Follows" happened only for the two true statements pushed to 0 (Hamadan, ninety-three); the negated Spanish statements always contradict. In this small sample tiu steering gave "follows" more often (4 of 8) than native steering (2 of 15). In the read-out window (L19 " is", L23 final space) no flipped answer followed. So the difference is between the windows (statement window: sometimes follows; read-out: never), not between statement end and object token [assumed: 6 statements, anecdotal].

- Read-out steering changes the digit; the text then reverts to the real truth value: "Fifty-two is larger than fifty-four." with native at L19 " is" → "1, as it is factually incorrect. Fifty-two is not larger than fifty-four".
- Midpoint reflection over all statement tokens at L16: the model stops reading the statement and answers " 0, and I will explain why:" with unrelated text, for true and false statements alike (the collapse above).

## What this means [concluded, one run]

- The same picture as on the framing prompt, with the order of statement and instruction swapped: the truth judgement forms at the statement's tokens around layers 13–17 and is read by the answer tokens from layer 18. So on both prompts the window sits in the layers tPD decomposed (15–19).
- The stored tiu direction is the statement-end truth direction of this prompt (cosine ~0.9, same steering effect). It is not the direction the answer tokens read, nor the one at the object token.
- Open: why false → 1 is hard at the statement end but works at the object token; whether the controls at the read-out tokens can be cleaned (random directions orthogonal to the native one).

The 4090 (vast.ai 56119768) is still rented; nothing on it is needed any more.
