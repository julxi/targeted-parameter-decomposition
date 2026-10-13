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

## Experiment 2: stimulus lens

**Design** [assumed: agent proposal]
- **Input**: statement + stimulus, tokenized separately and concatenated (see above). Example: `The city of Krasnodar is in Russia.` + ` This statement is`.

    C: I gave the stimulus "This statement is". I'm a bit afraid that for other examples the model focuses on a different attribute of the statement than truthfullness, e.g. the model selects "This statement is _funny_." That's what I meant that this simple stimulus is a stimulus for categorization. For the right dataset it might not be an issue, but maybe there's a better stimulus.

    Answer (26-10-10, Claude Opus 5.5): Resolved by your next comment; the stimulus screen R1 (answer there) measures exactly this rate of non-truth answers such as "funny".

- **Decomposed positions D**: the statement end; the stimulus runs on the original weights.
- **Loss position**: the last stimulus token only; full next-token KL there.
- This needs **two position masks** instead of one: D (where components replace the weights; also where importance minimality and the CI statistics are taken) and the loss position (where the KL is read). Today one mask does both. This is the main code change; it touches the batch format, the target losses, PPGD and the target eval metrics.
- **Stimuli**: chosen at the gate (below) from a candidate list, e.g. ` This statement is`, ` That is`, ` True or false? The statement is`, ` Is this true? Answer:`, ` In fact,` (the last one invites a correction rather than a verdict). Proposal: mix the 2–3 best; each training sample draws one at random. Since D cannot see the stimulus, mixing does not change what the components can compute; it widens what the loss protects and averages out quirks of a single phrasing.

    C: Oh, good that we are talking about the stimulus. So my comment above is resolved. I don' think that mixing the stimulus is a good idea. I'd rather have a little experiment to see which stimulus elicits truthfulness related responses at at a high rate (once we have the dataset)

    Answer (26-10-10, Claude Opus 5.5): Agreed: a single stimulus, chosen by a small screen once the dataset is fixed (R1 in the entry *answers to Julian's comments*). The screen measures how often the model answers with a truth-related token ("true", "false", "correct", "wrong", …), so the "funny" case from your comment above shows up directly as a low rate.

- **Your worry that the stimulus taps categorisation rather than truth**: the gate measures it directly, by comparing the true-vs-false output difference with the difference between two true statements of the same family (gate G's content baseline). And it is acceptable if some categorisation components appear: the per-dataset analysis (A5 of `truth_writers.py`) separates them afterwards.
- **Variant 2b (bolder)**: a binary verdict loss. Renormalise the output to two token sets (true-like vs false-like answers) and take the KL over that 2-way distribution only. The decomposition must then preserve exactly the model's verdict, nothing else. Risk: the token sets are a hand choice, and a narrow loss may be satisfied by very few components that encode the verdict for these 20 families only. I'd run the full-KL version first.

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
