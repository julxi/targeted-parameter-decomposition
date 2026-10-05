I want to do another round of testing tPD of truthfulness but this time using the dataset from "On the Universal Truthfulness Hyperplane Inside LLMs" as basis.

The old result was kind of negative / inconclusive. The trained decompositions showed behaviour that could mean that they have learned some truthfullness behaviour. This is surprising at first, as they are trained on all tokens, which could potentially require very complex behaviours. However, a non-trained decomposition scored equally high on the metric.

On the other hand we have seen that if we train on something else like snippets of code, we see that tPD seems to not be able to learn (by the component count and ablations).

The data in "On the Universal Truthfulness Hyperplane Inside LLMs" is compiled from a couple of resources and generally less clean. I want to work with it anyway just because of the big diversity. I would like to recreate their experiment. There are a lot of tasks and domains so maybe that's a bit too much and some examples are also just a bit too I'm afraid. I don't want to spend hughe amounts of compute just for another negative result, so we should start with experiments that are about the size of what we have done so far.

---

26-10-05, Claude Opus 5.5 — *first look at the UTH data and a staged proposal*

I have read INDEX.md, the SUMMARYs of all earlier topics, `scratch/arm_A_experiment_summary.md`, `scratch/uth_data.py` (your loader), and the paper's experimental setup via its arXiv HTML page (2407.08582v3). I have not read `docs/CODING.md` yet (no code written in this entry). Abbreviations: UTH = "On the Universal Truthfulness Hyperplane Inside LLMs" (Liu, Chen, Cheng, He; EMNLP 2024; data repo `hkust-nlp/Universal_Truthfulness_Hyperplane`); tiu = Truth-is-Universal, the data of the arm A experiment; CI = causal importance.

I read your request as: run tPD on Qwen2.5-7B-Instruct with a diverse target made from the UTH datasets, then test the CI values with UTH's **cross-task** protocol, at the compute scale of the arm A runs. Your sentence "some examples are also just a bit too [...]" is cut off; I guessed "too long" or "too noisy", and both show up below.

## What the paper did [verified: arXiv HTML, read 26-10-05; numbers quoted from the page]

- Models: LLaMA2-7b-chat (main), Mistral-7b-v0.1, LLaMA2-13b-chat. Not Qwen.
- 49 datasets in 17 task categories. Each training split holds up to 800 samples; test sets are capped at 5,000 per dataset.
- False samples: random wrong options for multiple-choice data; for generation tasks, GPT-3.5/GPT-4 wrote "convincing but false answers" (E2E NLG: attributes swapped at random).
- Probe input: the last token's representation, mainly attention-head outputs (residual activations were reported 3 points worse), with per-head selection on validation splits. Probes: logistic regression and mass-mean.
- Evaluation: 8 test datasets from 3 held-out task categories (sentence completion, short-answer QA, summarization). *Cross-task* = the probe was trained on the 14 other categories only.
- Cross-task average accuracy: Probe-MM 70.47 and Probe-LR 69.24 on LLaMA2-7b; Mistral-7b Probe-MM 77.11; baselines Self-Eval 64.76, probability 60.92; finetuning upper bound 77.67.
- Diversity beats quantity: "as few as 10 data points per dataset" were comparable to 800.

**Why this matters for us** [concluded]: arm A failed to discriminate because every readout was at the ceiling (residual stream 0.998, untrained CIs 0.998). Cross-task accuracy is around 0.70 even for the residual stream, so there is room in which trained CIs could beat untrained CIs, or the residual stream. A new dataset alone does not create that room; the held-out-task protocol does. So I'd make the cross-task protocol the core of the series, not the diversity of the data.

## The local data [verified: `scratch/uth_data/`, all 49 domains, train + vali splits, Qwen tokenizer, 26-10-05]

- 37,400 train samples in total (800 per dataset; the 6 SAPLMA statement datasets have 500). The `vali` files hold up to 5,000 samples, i.e. they are the paper's test sets. Labels are balanced (0.44–0.54 true in every train split).
- Train-split lengths in Qwen tokens range from a median of 9 (capitals) to 873 (cnn_dailymail_re); hellaswag 258, race 428, xsum_re 415. tiu ran at `max_seq_len` 24.
- Fraction of each train split within a length cap (selected rows; the full table is reproducible with the script in the session scratchpad, not kept):

| cap (tokens) | datasets with ≥ 90% of samples within the cap |
|---|---|
| 64 | 21: the 6 SAPLMA (animals, capitals, companies, elements, facts, inventions), counterfact, creak, arithmetic, copa, definite_pronoun_resolution, hotpot_qa_re, nq_re, nq_re_long, openbookqa, qqp, sciq, strategy_qa, tqa, triva_qa_re, winogrande; plus commonsense_qa and wsc.fixed at 0.83 |
| 128 | additionally ag_news, arc, easy_arc, e2e_nlg_cleaned, mrpc, paws, qnli, story_cloze, triva_qa_re_long; dbpedia_14 0.86, rte 0.83 |

- At cap 64, 10 of the 17 categories remain: statement fact checking, short-answer QA (all 3 datasets), long-answer QA (nq_re_long only), closed-book multiple choice (commonsense_qa, openbookqa), coreference (all 3), multi-step QA (both), paraphrase (qqp), sentence completion (copa only), other (tqa, arithmetic). Lost: NLI, summarization, sentiment, topic classification, both reading-comprehension categories, structure-to-text, reading + common sense, i.e. everything with a long context.
- **The paper's test categories don't fit**: summarization is ~400–900 tokens and two of the three sentence-completion sets are > 64. So we would pick our own held-out categories (proposal below), and our numbers would not be comparable to the paper's 0.70 anyway (other model, other layers, residual instead of heads).

**Label artefacts seen** (examples printed from the train splits):
- arithmetic: 34% of the false answers are not numbers ("What is 486 plus 493? house.") versus 0% of the true ones [verified: count]. A probe can solve those by answer format. Drop the non-numeric false answers, or drop the dataset.
- capitals: "Kiev is a name of a city." (true) / "South Ossetia is a name of a city." (false). These are about whether something is a city, not about capitals.
- counterfact false: "BBC News Online, from Chrysler.", ungrammatical.
- hotpot_qa_re true: "Which opera … did Franz Xaver Gerl sang the role of Sarastro in the premiere of The Magic Flute.", a question without an answer, labelled true.

I have not measured how frequent the last three kinds are; they are single examples [assumed: noise, not systematic]. A general warning: "convincing but false" answers written by GPT-4 can differ from true ones in style. A probe that generalises across tasks could partly read such artefacts. That holds for the residual baseline too, so the comparison of baselines stays fair, but "truth" is the label's meaning, not a guarantee.

## Proposal: two stages, with a cheap gate first

**Stage 0: baselines only, no tPD training** (cheap; answers "is there room?" before any GPU training).
1. Build `data/uth/<dataset>/v1/` with a generator under `spd/experiments/lm/honesty_targeted_decomposition/` (same convention as tiu and `pile_code`: generator committed before the data is built). Length cap 64 Qwen tokens; drop longer samples; datasets with < 80% coverage are dropped whole. Test sets subsampled to ≤ 500 per dataset to keep feature extraction small.
2. Fix the held-out task categories now, before any result. Proposal: **short-answer QA** (nq_re, triva_qa_re, sciq; it is one of the paper's three test categories), **coreference** (definite_pronoun_resolution, winogrande, wsc.fixed) and **sentence completion** (copa). That leaves 7 training categories with about 14–15 datasets.
   OUTDATED (26-10-05): this held-out set is superseded by the paper's own test set (8 datasets in 3 categories); the length cap applies to tPD training data only. See the answer to the C: comment on held-out categories below, and the entry *revised plan after Julian's comments*.
3. Probe the residual stream (layer 19 as before, plus a coarse layer sweep) and the existing bf16 untrained CI network `s-7fad0c14` (random initialisation, independent of the target data, so it is reusable). Probes trained on the training categories, scored on the held-out ones, last-token readout, logistic regression and mass-mean: the `probe_ci.py` setup with a different split.
4. Gate: if the residual stream already generalises at ≥ 0.95 to the held-out categories, the protocol has no room either, and we rethink before training. If it is around 0.7–0.85, go to stage 1. Also record whether the untrained CIs match the residual stream cross-task as they did in-distribution on tiu.

Cost: a forward pass over ~20k sequences of ≤ 64 tokens through layers 0–19. Minutes on an H100; on the laptop a few hours in bf16 [assumed: extrapolated from the 26-10-03 benchmark, 2.7 s per batch of 16 × 24 tokens].

**Stage 1: one tPD run on the training categories only.**
- Same settings as arm A (layers 15–19 `down_proj`, C = 96, batch 16, 5k steps, LR 5e-4, bf16, Pile non-target), `max_seq_len` 64. The held-out categories never enter the decomposition's training, otherwise its CIs are fit to the test distribution.
- Cost: 16 × 64 = 1,024 tokens per step against 384 for tiu. Extrapolating linearly from the 26-10-02 throughput table (batch 16: 235 ms, batch 32: 348 ms per step at 24 tokens), that is ~420 ms per step without PPGD, about 1.8× an arm A run, roughly 1 h on an uncapped H100 [assumed: linear extrapolation; a 150-step trial measures it]. It stays at the arm A scale. A 128-token cap would roughly double it again.
  OUTDATED (26-10-05): this cost estimate is not sound. It ignored the Pile batch (16 × 64 tokens) that every step already contains, and the padding to `max_seq_len`. See the answer to the C: comment "why do you recommend 128?" below.
- Probes as in stage 0, plus trained-vs-untrained at matched sparsity (top-k CIs per sample).
- **Success criterion, fixed in advance:** trained CIs beat untrained CIs on the held-out categories at matched sparsity by clearly more than the run-to-run spread, and come near or above the residual stream. On tiu the trained-vs-untrained gap (1–2 points) was within the spread between runs (0.970–0.991). If the gap is again that small, the series ends there with a clean negative answer for about one GPU hour.

## Points to decide

1. **Training positions.** Most QA samples are a prompt plus a 2–5-token answer. With "all tokens" (as arm A), most of the training signal goes to reading the question. Alternatives: the last few tokens (`last_k`, already in the code), or exactly the answer span (needs a new `loss_positions` mode; the SAPLMA statements have no prompt/answer split, so they would use all tokens). My recommendation: all tokens for the first run, for continuity with arm A; answer-span as a second arm only if stage 1 shows something. I'm not certain about this: the answer-span arm targets more precisely what UTH probes.
C: yes, all tokens

Answer (26-10-05, Claude Opus 5.5): Noted: stage 1 trains on all real tokens (`loss_positions: tokens`), as arm A did. [decided: C: comment above]

2. **Length cap 64 or 128.** 64 keeps 10 categories at ~1.8× arm A's cost; 128 adds NLI, topic classification, paraphrase and structure-to-text at about twice that again. Recommendation: 64.
C: why do you recommend 128?

Answer (26-10-05, Claude Opus 5.5): I recommended 64, not 128. But your next comment changes the picture, and I now lean towards 128, depending on a short timing trial. The reasons:

- **The cap only constrains the tPD training data.** Test tasks are used only in forward passes for the probes, and those can be long (see my answer to the next comment). So the cap no longer decides the test set; it only decides how diverse the training set is.
- **Diversity is what UTH found to matter**, more than the number of samples. With the paper's own category assignment and test categories held out [verified: token counts, train splits, 26-10-05; a dataset counts if ≥ 80% of its samples fit]:
  - Cap 64 gives 7 training categories, 19 datasets: statement fact checking (8 datasets), paraphrase (1), long-answer QA (1), closed-book multiple choice (2), coreference (3), multi-step QA (2), other (2).
  - Cap 128 gives 10 categories, 29 datasets: additionally NLI (2), topic classification (2) and structure-to-text (1), and more datasets in paraphrase, long-answer QA and closed-book multiple choice.
- **Cost: my earlier estimate was not sound.** I checked the loader (`load_prompts_dataset` in `spd/experiments/lm/prompts_dataset.py`) [verified: code read]:
  - Every target sequence is right-padded to `max_seq_len`, so the cost follows the cap, not the actual lengths.
  - Every step also contains a Pile batch of 16 × 64 tokens (`nontarget_task_config.max_seq_len: 64` in the arm A config).
  - So arm A processed 384 target + 1,024 Pile tokens per step. A cap of 64 makes that 1,024 + 1,024, a cap of 128 2,048 + 1,024.
  - My "~1.8×" for cap 64 extrapolated as if the whole step were target tokens, which it isn't. The target stream runs more forward passes per step than the Pile stream (stochastic and adversarial masks, PPGD), so per token it costs more.
  - I can't give a sound factor from the existing measurements. My guess is roughly 1.3–2× arm A for cap 64 and 2–3.5× for cap 128 [assumed].
- **Proposal:** before deciding, run two 150-step timing trials (cap 64 and 128) on the rented H100, about 10 min in total. If cap 128 costs ≤ ~2.5× an arm A run (i.e. ≲ 1.5 h), take 128.

A possible later saving: most samples are much shorter than the cap (many have a median of 40–90 tokens), so padding each batch only to its longest sample, or batching by length, would recover much of the cost. That is a loader change; not proposed for now.
3. **The held-out categories** (proposal above).
C: Let's talk about the data some more. For me it's not clear yet, what you propose as test tasks. I couldn't match what you said above to the test tasks described in the paper:
- Sentence Completion:CoPA,HellaSwag,Story Cloze
- Short Answer Close Book QA: NQ, Trivia QA, SciQ
- Summarization: CNN,DM, XSum
I'm suppose that Summarization is just too long. Maybe we can move some training tasks to test. I'm hoping on the fact that tPD as described in their paper doesn't need many examples for learning so that we might be able to get results with less training tasks.

Answer (26-10-05, Claude Opus 5.5): You're right, my proposal didn't match the paper. I had replaced its test set with categories that fit a 64-token cap (short-answer QA, coreference, copa only), and I should have said that plainly. But the cap doesn't need to apply to the test tasks at all:

- **tPD never trains on the test tasks.** They are only used to compute residual activations and CI values in a forward pass, for the probes. A forward pass of long sequences is cheap on an H100. The CI network computes a value per position from that position's `down_proj` input, so it also runs on sequences longer than those it was trained on.
- **So I now propose the paper's test set exactly**, all 8 datasets in 3 categories, with ≤ 500 samples per dataset drawn from the `vali` files (lengths measured on such a draw) [verified 26-10-05]:

| held-out category | datasets (median / max Qwen tokens) |
|---|---|
| sentence completion | copa (42 / 58; only 200 samples exist), hellaswag (262 / 386), story_cloze (85 / 117) |
| short-answer closed-book QA | nq_re (15 / 29), triva_qa_re (20 / 77), sciq (17 / 49) |
| summarization | cnn_dailymail_re (849 / 2,341), xsum_re (440 / 3,063) |

- This keeps the paper's protocol, and leaves all 14 training categories available as training data (as far as the cap allows, see the previous answer). Moving training categories into the test set would take away the diversity the paper found most important.
- **On fewer examples:** I agree that the number of samples per dataset is probably not the bottleneck. UTH found that "as few as 10 data points per dataset" were comparable to 800 for probes. tPD's numpy/pandas result was trained on just 2 prompts, but that was one narrow mechanism, which says little about a diverse truth target [concluded]. What the counts suggest is to keep many categories and, if anything, cut samples per dataset, not the other way round.
- **Costs and caveats of long test sequences:**
  - The summarization samples are 10–40× longer than anything tPD was trained on. All three readouts (residual stream, untrained CIs, trained CIs) face the same shift, so the comparison between them stays fair. Still, read the summarization numbers with that in mind.
  - On the laptop CPU, the 1,000 summarization samples alone take roughly 1–1.5 h [assumed: extrapolated from the 26-10-03 benchmark]; on an H100, minutes.
  - `probe_ci.py` currently tokenises with the run's `max_seq_len` and raises on longer inputs. It needs a separate length limit for probing (small change).
4. **Arithmetic:** drop the non-numeric false answers or the whole dataset? Recommendation: drop the dataset (its truth is computation, unlike the other domains, and it has a format shortcut).
C: no, keep it

Answer (26-10-05, Claude Opus 5.5): Kept, unchanged. [decided: C: comment above] Arithmetic is a training dataset (category "other"), so its format shortcut can only affect what the probes learn on the training side, not the held-out scores directly. I'll report probe accuracy per training dataset too, so a probe that leans on it would be visible.
5. **Your cut-off sentence** ("some examples are also just a bit too …").
C: don't know. not important I suppose.

One more critical point, so it is said: UTH shows that *residual-stream* probes generalise better with diverse data. For tPD the analogous hypothesis is that a diverse target pushes the decomposition towards mechanisms shared across tasks. The opposite is just as plausible, though: importance minimality tends to favour task-specific components. The paper's appendix also notes that mechanisms which co-activate on the target get merged. Stage 1 tests exactly this; I wouldn't count on either outcome.
C: maybe. We will see.

C: another question. how many sequences do we see in training and how many are in the training data (I know this depends on which tasks/domains we take. I'm asking for ballpark numbers.)

Answer (26-10-05, Claude Opus 5.5): With arm A's schedule (5,000 steps × batch 16), training sees **80,000 target sequences** (and as many Pile sequences, each 64 tokens). Available training data, the paper's training categories only, samples within the cap [verified: token counts, 26-10-05]:

| | train sequences available | passes over the data in 80k |
|---|---|---|
| arm A (tiu, for reference) | 6,560 | ~12 |
| UTH, cap 64 (19 datasets) | 12,999 | ~6 |
| UTH, cap 128 (29 datasets) | 21,041 | ~4 |

The 8 statement fact-checking datasets account for 4,600 of these sequences at either cap (the 6 SAPLMA sets have 500 each, counterfact and creak 800 each, all within 64 tokens), i.e. 35% of the cap-64 data and 22% at cap 128. If that imbalance matters, the obvious fix is to cap each dataset at e.g. 400 samples; UTH's quantity-vs-diversity result says that costs little.

---

26-10-05, Claude Opus 5.5 — *revised plan after Julian's comments*

I answered Julian's C: comments in place above. In short:
- **All tokens** for the first run [decided: C: comment].
- **Arithmetic stays** [decided: C: comment].
- **Test set = the paper's**, 8 datasets in 3 held-out categories: sentence completion (copa, hellaswag, story_cloze), short-answer QA (nq_re, triva_qa_re, sciq), summarization (cnn_dailymail_re, xsum_re); ≤ 500 samples each from the `vali` files [concluded; awaiting Julian]. The length cap applies to tPD training data only; test tasks only need forward passes.
- **Length cap: 64 or 128, decided by a timing trial** [concluded; awaiting Julian]. 64 gives 7 training categories / 19 datasets / 12,999 sequences; 128 gives 10 / 29 / 21,041. Training sees 80,000 target sequences either way (~6 or ~4 passes). My first cost estimate was wrong (it ignored the Pile stream and padding). Two 150-step timing trials (~10 min) settle it; take 128 if it costs ≤ ~2.5× an arm A run.

Revised order of work:
1. Generator for `data/uth/` (train: the paper's training categories within the cap; test: the 8 test datasets, uncapped, ≤ 500 each), committed before building, per the `data/` convention.
2. Stage 0 (no training): residual-stream and untrained-CI (`s-7fad0c14`) probes, trained on the training categories, scored on the 8 test datasets. `probe_ci.py` needs a probing length limit separate from the run's `max_seq_len`. Gate as before: go on if the residual stream is well below 0.95 on the held-out categories.
3. Timing trials for caps 64 and 128, then the stage 1 run.

Stage 0 needs no GPU in principle, but the summarization test samples make the laptop slow (roughly 1–1.5 h for those alone [assumed]); on the H100 it is minutes. Since stage 3 rents an H100 anyway, stage 0 and the timing trials could share one rental.

C: ok, that sounds good. Don't know if I have overlooked any more decisions. Also, I like the idea of the loader padding only to the longest sequence length. Do you think this can speed up training?

Answer (26-10-05, Claude Opus 5.5): Recorded: the test set is the paper's 8 datasets, and the cap question goes to a timing trial [decided: C: comment above]. On padding: **yes, but only together with length bucketing**; padding each random batch to its longest sample saves little.

**Simulation** [verified: Qwen token lengths of the training-category samples within the cap, 4,000 simulated batches of 16, 26-10-05; script in the session scratchpad, not kept]. Target tokens per batch, as a fraction of fixed padding to the cap:

| | fixed padding (now) | pad to batch max, random batches | pool-sorted: draw 8 batches' worth (128 samples), sort by length, split into 8 batches | real tokens (lower bound) |
|---|---|---|---|---|
| cap 64 (mean length 25) | 1,024 (1.00) | 875 (0.85) | 461 (0.45) | 405 (0.40) |
| cap 128 (mean length 44) | 2,048 (1.00) | 1,605 (0.78) | 824 (0.40) | 703 (0.34) |

- Random batches: one long sample in 16 is enough to set the batch length, so the saving is only 15–22%.
- Pool-sorted batches come close to the real token count. **Cap 128 with pool sorting has fewer padded target tokens per step (824) than cap 64 with fixed padding (1,024).** For comparison, arm A used 384.

**How much wall time this saves:** less than the token ratio, by an unknown amount [assumed].
- The Pile batch (16 × 64 tokens) stays as it is.
- At small token counts a step is overhead-bound: on tiu, batch 4 and batch 16 took the same time per pass.
- My guess: 1.5–2× faster per step at cap 128. The trial measures it directly: a fixed-padding run at `max_seq_len` 64 approximates bucketed cap 128 in token count, so the trial needs no new code to tell us whether bucketing makes 128 affordable.

**What it takes** [verified: code read]:
1. **Loader** (`spd/experiments/lm/prompts_dataset.py`): store unpadded sequences, and a sampler that draws a pool, sorts it by length and yields batches in random order; collate pads to the batch maximum and builds the position mask per batch.
2. **Persistent PGD** (`spd/persistent_pgd.py`, lines ~159–177): with `scope: per_batch_per_position` (our config), the persistent adversarial masks have the fixed shape (batch, seq_len, C), set from the first batch. Shorter batches need the masks sliced to the batch's length. Without the change it would crash, not fail silently [concluded: shape mismatch]. These masks belong to batch slots, not to samples, so slicing keeps their meaning.
3. Anything else that assumes a fixed sequence length (eval, figures). To be checked when implementing.

Plus a CPU smoke test on the tiny Qwen2. Moderate work, maybe half a day.
C: sounds good. I think pad to max saves already quite a lot. I'm a bit afraid of pooling as thes sorts the training data.

Answer (26-10-05, Claude Opus 5.5): Agreed: no pooling, only padding each (random) batch to its longest sample [decided: C: comment above]. Your worry about sorting is the one I flagged too, and it is a property of the data order that we couldn't fully check.

One correction on the numbers, so the expectation is right: in the simulation above, padding random batches to their longest sample saves 15% at cap 64 (875 instead of 1,024 target tokens per batch) and 22% at cap 128 (1,605 instead of 2,048). That is useful, but much less than sorting would give (0.40–0.45×). So at cap 128 without sorting, a step has about 1.6× the target tokens of cap 64 with today's padding. The timing trial (fixed padding at 64 and 128) brackets this: cap 128 padded per batch averages ~100 tokens per sequence.

An advantage of your choice: padding per batch leaves the training computation unchanged, apart from speed [concluded: code read, not tested]:
- the batches are the same random batches as now;
- with right padding and causal attention, padding positions can't influence the real positions;
- all losses and metrics in our config take the position mask (`importance_minimality_loss.py`, `stochastic_recon_subset_loss.py`, `targeted_recon_loss.py`, `pgd_masked_recon_loss.py`, `persistent_pgd.py`, `unmasked_recon_loss.py` and others), so padding positions never enter a loss.

That makes the correctness check simple and exact: a unit test that the same batch, padded to 24 and to 32 tokens, gives the same losses (up to floating-point error). It replaces the with/without comparison run I proposed above. The code changes stay as listed, minus the sampler: unpadded storage, a collate that pads to the batch maximum, persistent PGD masks sliced to the batch length.

**Costs beyond the code:**
- Sorted batches are more homogeneous in length, and length correlates with the dataset. The shortest batches of a pool are mostly the one-sentence fact statements, so each batch mixes fewer tasks than a random one. With a pool of 8 batches this is mild, and the order of batches stays random. I'd expect no harm, but haven't tested it [assumed].
- A correctness check: one short run with bucketing against one without, on the same data. Losses should agree within noise.

**Recommendation:** implement it before stage 1, not before stage 0 (which trains nothing). If the trial confirms it, take cap 128 with bucketing: 10 training categories at roughly the cost of cap 64 without bucketing.

**Decisions you may have overlooked** (my default in bold; all minor, and I'll go with the defaults unless you object):
1. **Imbalance between datasets:** keep all samples, as the paper did (statement fact checking is then 22% of the cap-128 data), rather than capping each dataset.
C: keep all samples
2. **Long samples in datasets that mostly fit:** drop the samples longer than the cap (affects datasets with 80–99% coverage, e.g. commonsense_qa and wsc.fixed at cap 64). Their remaining samples are slightly shorter than the dataset's average; this only affects training.
C: yes, keep the domains but filter out the long ones
3. **Test size:** **1,000 samples per test dataset** instead of 500 (copa has only 200). That puts the standard error of the 8-dataset average at about 0.6 points instead of about 0.8, against a run-to-run spread of ~2 points on tiu. On an H100 the extra forward passes are cheap.
C: I don't fully understand, but I think this is fine

Answer (26-10-05, Claude Opus 5.5): In plain terms: a probe's accuracy on a test set is measured on a random sample of that set, so it wobbles a little depending on which samples were drawn. With 500 samples per test dataset, the average over the 8 test datasets wobbles by roughly ±0.8 points (one standard error); with 1,000, by roughly ±0.6. We will compare readouts whose accuracies may differ by only a few points, so less wobble helps. The gain from 500 to 1,000 is modest, though. The comparison between readouts is also partly protected, because all readouts are scored on the same samples. I'll take 1,000; it costs only some extra forward passes on the H100.
4. **Probe protocol:**
   - last-token readout as the main metric (as the paper), mean over tokens as secondary;
   - residual stream at layer 19 plus a coarse layer sweep;
   - probe regularisation tuned only on the training categories' held-out split, never on the test categories (the paper's rule).
C: Which probe do you mean hear?

Answer (26-10-05, Claude Opus 5.5): All of the probes in the series, i.e. every classifier that predicts true/false from a feature vector. There are two kinds, each run on several feature sets:
- **logistic regression**: a linear classifier fit by gradient descent;
- **mass-mean**: the direction from the mean of the false samples to the mean of the true samples, plus a threshold. It has no tuning knob.

The feature sets are:
- the residual stream (the model's hidden state after a layer; the "probe the activations" baseline);
- the untrained network's CI values;
- the trained decomposition's CI values.

In each case the probe is trained on the training categories and scored on the 8 test datasets. All of this is what `probe_ci.py` already does on tiu, only with a different split.

The bullet was about two choices that could quietly leak information from the test tasks:
1. **Regularisation of the logistic regression**: how strongly its weights are pulled towards zero. It affects how well a probe transfers to new tasks. On tiu we used sklearn's default, untuned. If we tune it, it must be tuned on held-out samples of the *training* categories, never on the test datasets. Otherwise the cross-task score is inflated. The paper follows the same rule.
2. **Which layer of the residual stream**: we used layer 19 on tiu (the last decomposed layer). A coarse sweep over layers shows whether 19 is a poor choice for the baseline. The same rule applies: if we pick a layer, pick it on training-category data.

The CI readouts have no layer choice (the 480 CI values span layers 15–19).
5. **Built-in eval during tPD training:** uses the training categories' own held-out samples (from the `vali` files, within the cap), not the test categories.
C: yes, use extra validation files

Answer (26-10-05, Claude Opus 5.5): Done in the plan [decided: C: comment above]. Each training-category dataset gets a `test` split drawn from its UTH `vali` file, within the length cap, kept apart from the 8 test datasets. The training eval only uses a few hundred of these per eval (arm A: 16 batches × 16 = 256); the rest serves the probes' tuning split (previous answer).
6. **Seeds:** one for stage 1. If the result lands near the success threshold, a second seed is the first thing to add, because the run-to-run spread on this data is unknown.
C: ok
7. **Prompt format:** bare text, no chat template, as in arm A.
C: yes

---

26-10-05, Claude Opus 5.5 — *design consolidated after the second round of comments*

All of Julian's C: comments so far are answered in place. The design as it now stands:

**Decided** (each by a C: comment in this LOG):
- Model, training settings and positions as in arm A: Qwen2.5-7B-Instruct, `down_proj` of layers 15–19, C = 96, batch 16, 5k steps, LR 5e-4, bf16 frozen model, Pile non-target stream, loss on all real tokens, no chat template, one seed.
- **Test set:** the paper's 8 test datasets in 3 categories (copa, hellaswag, story_cloze; nq_re, triva_qa_re, sciq; cnn_dailymail_re, xsum_re), up to 1,000 samples each from the `vali` files, uncapped in length; never used in tPD training.
- **Training data:** the paper's 14 training categories. Datasets enter if ≥ 80% of their samples fit the cap, and their longer samples are dropped. All samples kept (no per-dataset balancing). arithmetic stays.
- **Training eval and probe tuning:** a `test` split per training-category dataset, drawn from its `vali` file within the cap.
- **Cap 64 vs 128:** decided by a timing trial.
- **Padding:** each batch padded only to its longest sample; no length sorting or pooling. Expected saving 15–22% of target tokens [verified: simulation]. Expected to leave the training computation unchanged apart from speed [concluded: code read]. Checked by an exact unit test (the same batch padded to two lengths gives the same losses).

**Agreed** ("I think this is fine" / "ok", but on points the agent proposed, so [concluded] rather than [decided] where noted):
- Probe protocol: last-token readout primary, mean over tokens secondary; logistic regression and mass-mean; feature sets residual stream (layer 19 plus a coarse sweep), untrained CIs (`s-7fad0c14`), trained CIs. Any tuning (regularisation, layer choice) only on the training categories' held-out split [concluded; Julian asked what "probe" meant, answered above].

**Order of work:**
1. Generator for `data/uth/` (committed before building).
2. Stage 0: baseline probes cross-task (gate: go on only if the residual stream is well below 0.95 on the test datasets). `probe_ci.py` needs a probing length limit separate from the run's `max_seq_len`.
3. Per-batch padding (loader collate, slicing of the persistent PGD masks) plus the exact unit test.
4. Rent an H100: stage 0 feature extraction, timing trial (fixed padding at 64 and 128), then the stage 1 run at the chosen cap.
5. Stage 1 probes. Success criterion, fixed in advance: trained CIs beat untrained CIs cross-task at matched sparsity by clearly more than the run-to-run spread (tiu: 0.970–0.991); otherwise the series ends.

Steps 1–3 are code on the laptop; step 2's feature extraction can run on the laptop or wait for the H100.

---

C: (transcribed from chat) ok, let's go. One change however, Stage 0: baseline probe should be on a vm. Tell me and I'll rent one

---

26-10-05, Claude Opus 5.5 — *implementation started: generator first*

Stage 0's feature extraction and probes run on a rented VM, not on the laptop [decided: transcribed chat above]. I tell Julian when the code is ready, and he rents the machine. I'm starting with the `data/uth/` generator.

---

26-10-05, Claude Opus 5.5 — *generator build_uth.py written; dry run; two upstream data bugs*

I have now read `docs/CODING.md`, `data/README.md`, `build_tiu.py`, `build_pile_code.py` and the loader `spd/experiments/lm/prepared_datasets.py`.

**New file** `spd/experiments/lm/honesty_targeted_decomposition/build_uth.py`, `--version vN`, defaults `max_tokens 128, min_coverage 0.8, n_heldout 200, n_test 1000, seed 0`. It downloads the upstream files itself, from the pinned upstream commit `f509513` (2024-10-03); it does not use `scratch/uth_data/`. It writes `data/uth/<dataset>/<version>/`:
- **train tasks** (the paper's 14 training categories): `train` = upstream train file, `test` = up to 200 samples of the upstream `vali` file; both only samples ≤ 128 Qwen tokens; a dataset is skipped if < 80% of its train samples fit.
- **test tasks** (the paper's 3 held-out categories): `test` only, up to 1,000 samples of `vali`, uncapped. There is no `train` split, so a config listing a test task for training crashes in the loader [verified: `KeyError('train')` on the dry-run output].
- Records: `{"text", "label", "dataset", "category", "n_tokens", "upstream_file", "upstream_index"}`. Manifests carry `role`, `category`, upstream file hashes and cleaning counts.
- The category table is asserted against the upstream repo's dataset list (all 49, each once). `easy_arc` isn't in the paper's list; I put it with `arc` in closed-book multiple-choice QA [assumed].
- `data/README.md`: the families list now includes `uth/` (and `pile_code/`, which was missing), plus the test-task exception to "every dataset has train and test".

**Findings while writing it:**
1. **QA format** [verified: upstream `src/utils.py`, `general_qa_prompt`]: the paper renders `{question, answer}` records as `"Question: {q}\nAnswer: {a}"`. Julian's scratch loader (and my token counts in earlier entries) joined them with a space. The generator uses the paper's template, which adds a few tokens per sample. The counts below are with the template.
2. **Upstream construction bug in story_cloze and definite_pronoun_resolution** [verified: grouping records by the text before the last `\nAnswer: `]:
   - Upstream builds these as pairs: the same story or sentence once with the right answer (label 1), once with the wrong one (label 0).
   - In about half of the pairs **both records carry the same answer text**, so one of the two labels is wrong. story_cloze `vali`: 960 of 1,871 pairs; definite_pronoun_resolution `vali`: 282 of 564.
   - In the train files most prompts appear only once, so a record from a broken pair can't be recognised. Judging from the visible pairs, roughly a quarter of those single records are mislabelled [concluded: about half come from broken pairs, and a record of a broken pair is wrong half the time].
   - Fix: for these two datasets, only intact pairs are kept (two records, different answers, labels 0 and 1). story_cloze is a **test** task: its 1,000 test samples now come from the 911 intact `vali` pairs only. definite_pronoun_resolution (training): 134 train records instead of 654.
   - Without this fix, story_cloze would have tested the probes against ~25% wrong labels [concluded]. This may also have affected the paper's own story_cloze numbers [assumed; not checked how the paper evaluated].
3. **Other cleaning**, applied to all datasets:
   - texts carrying both labels are dropped (besides the two above: capitals 12, web_nlg_re 13, wsc.fixed 18, tqa 2, sciq 1, facts 1);
   - exact duplicates are kept once (largest: triva_qa_re `vali` 261);
   - held-out samples of training tasks that also occur in the dataset's train file are dropped (triva_qa_re_long 21, facts 3, capitals 1).
   - No test-task `vali` text occurs in any training-category train file [verified].
4. **No license**: the upstream repo has no license file (GitHub API: none), so redistributing its data in our repo is unclear. The manifests say so. This needs Julian's call before the data is committed (see below).

**Dry run** [verified: full run into the gitignored `scratch/dry_uth/`, log read; loader `load_prepared_split` reads every split and verifies hashes]:
- 29 training datasets in 10 categories, **20,292 train sequences** (mean 46.6 tokens, max 128) and 5,694 held-out samples. Earlier estimate without the template and cleaning: 21,041.
- Per category: statement fact checking 4,578, closed-book multiple choice 3,143, paraphrase 2,393, coreference 1,683, multi-step 1,600, other 1,598, long-answer QA 1,586, topic 1,464, NLI 1,448, structure-to-text 799.
- Skipped for length (< 80% within 128 tokens): anli, imdb, yelp_polarity, multirc, squad, boolq, race, dream, piqa (0.75), web_nlg_re (0.74), record, cosmos_qa. So sentiment, both reading-comprehension categories and reading + commonsense are absent.
- 8 test datasets, 7,200 test samples (copa 200, the others 1,000).
- About 20 MB on disk, mostly cnn_dailymail_re (4.5 MB) and xsum_re (2.6 MB); tiu is 2 MB.
- Lint, format and basedpyright clean on the new file.

**Next, per the convention:** Julian commits the generator (and the README change), then I build `v1` into `data/uth/` so that every manifest's `repo_commit` contains the generator. Open question for Julian: commit the built data to git despite the missing license, or keep `data/uth/` local / gitignored?

---

C: (transcribed from chat) done, and uth stays in git

---

26-10-05, Claude Opus 5.5 — *v1 built into data/uth*

Julian committed the generator and the README change (`93ff44b`). The built data goes into git despite the missing upstream license [decided: transcribed chat above]. Building v1 now.

---

26-10-05, Claude Opus 5.5 — *uth v1 built; stage 0 probe script probe_uth.py written and smoke-tested*

**Data** [verified]: `data/uth/*/v1` built from the committed generator. All 37 manifests carry `repo_commit 93ff44b` and the committed script's sha256. Loader counts: 20,292 fit (train-task train), 5,694 tune (train-task held-out), 7,200 test (test tasks), identical to the dry run. 20 MB on disk. Not committed yet.

**New script** `spd/experiments/lm/honesty_targeted_decomposition/probe_uth.py <out_dir> <data_version> <run_id>...`:
- Probe splits come from the manifests' roles:
  - fit = train-task `train`;
  - tune = train-task `test`;
  - test = test-task `test`.
  - It asserts that no category appears in both fit and test.
- Feature sets:
  - residual stream after layers 7, 11, 15, 19, 23, 27 (the coarse sweep);
  - the 480 `lower_leaky` CI values per run id;
  - mean log-probability.
  - Readouts: last token and mean over tokens.
- Probes:
  - mass-mean;
  - logistic regression for C ∈ {0.01, 0.1, 1}. The reported C is the one with the best tune accuracy; it is never chosen on test.
  - Scores pooled, as the unweighted mean over datasets (the paper's average), per dataset and per category.
- Forward passes run on length-sorted batches of ≤ 4,096 padded tokens (right padding, no attention mask, as in training), so the ~3,000-token summarization samples fit. One forward pass serves all runs.
- Outputs `results.json`, `features.npz` (float16), `examples.txt`.
- Reuses `load_ci_fns`, `readouts` and `mass_mean_acc` from `probe_ci.py`. `readouts` now builds its index tensors on the input's device (needed on GPU; unchanged on CPU).
- Lint, format and basedpyright clean.

**Job size:** 3.06 M tokens (fit 0.95 M, tune 0.26 M, test 1.86 M; longest sample 2,899 tokens). That's minutes of forward passes on an H100 [assumed]; probe fitting takes seconds per fit [verified: sklearn on random 20k × 3,584 data, 1–2 s; real data may need more iterations].

**Smoke test** [verified: laptop CPU, real Qwen2.5-7B-Instruct, untrained run `s-7fad0c14`, 4 samples per dataset (116 fit / 116 tune / 32 test, test lengths up to 1,610 tokens); script and log in `~/spd_out/26-10-05_uth_experiments/`]:
- Runs end to end, exit 0. 15 feature keys, results and examples written. Accuracies are meaningless at this size and were not read.
- `examples.txt` read by eye: QA records show the paper's template, and the last tokens are the answer and the final period (e.g. `[' Science', ' and', ' Technology', '.']`).

**Padding check** (`padding_check.py`, same folder): 87 train-task samples (lengths 7–108), once all in one batch padded to 108, once each alone.
- 66 samples are bit-identical. The 21 that differ are exactly those of ≤ 15 tokens, up to 3.1% relative error in the layer-19 residual and 5.1% in CI values. All samples of ≥ 16 tokens are identical, including heavily padded ones (e.g. length 16 padded by 92).
- So the difference comes from a different CPU kernel path for very short sequences run alone, not from padding leaking into real positions [concluded: leakage would also hit the long padded samples]. The earlier correlation with padding amount (0.6) only reflects that short samples get the most padding.
- Within one probe run every sample goes through the same batched path, and all runs share each forward pass, so this doesn't affect comparisons between feature sets.

**Next:** Julian rents the VM. Command on the VM, via a login shell and detached (rules in `docs/CODING.md`):
`python spd/experiments/lm/honesty_targeted_decomposition/probe_uth.py ~/spd_out/26-10-05_uth_experiments/stage0 v1 s-7fad0c14`

---

26-10-05, Claude Opus 5.5 — *stage 0 on an RTX 4090: memory fix, launch*

Julian committed (`692bf6d` "add uth data") and rented an RTX 4090 (24 GB, power limit 300 W of 450 W max; 80 CPUs, 251 GB RAM). The 15 GB bf16 model plus fp32 copies of the full logits (4,096 positions × ~152k vocabulary, 2.5 GB each) was too close to 24 GB. Fix in `probe_uth.py`:
- `BATCH_TOKENS` 4,096 → 2,048;
- log-softmax in chunks of 512 positions (`next_token_logprobs`).

The chunked version is identical to the unchunked one [verified: random bf16 logits, max difference 0.0]. Estimated peak ~19–21 GB [assumed]. Julian first proposed renting something else and adding a memory restriction to the vast YAML. I answered that `vast_config.yaml` already has `min_gpu_ram: 20` (with `gpu_name` fixed, raising it alone matches nothing) and suggested trying the 4090 first.

C: (transcribed from chat) yes

(in reply to "Shall I start it on the 4090, or would you rather destroy it and have me set up a config for a larger GPU?")

---

26-10-05, Claude Opus 5.5 — *stage 0 running; probe fitting parallelised mid-run*

- **First launch failed**: `ModuleNotFoundError: sklearn`. scikit-learn is declared only in the `dev` dependency group of `pyproject.toml`, and `spd-vast` syncs without it. Fixed on the VM with `uv sync --group dev` (installs from `uv.lock`; scikit-learn 1.8.0, as on the laptop). Not fixed in the repo; see the open point below.
- **Feature extraction** on the RTX 4090 [verified: VM log and file times]: 18:00 → 18:08 UTC, about 7.5 min for 3.06 M tokens. GPU memory ~19–20.5 GB of 24 GB, so the memory fix was enough. `features.npz` is 2.9 GB.
- **Probe fitting was the bottleneck**: about 4.5 min per 3,584-dim feature set when run sequentially (2 of 15 done by 18:20). My earlier timing on random data (1–2 s per fit) was far off, because real activations need many more lbfgs iterations.
- Julian asked whether stopping and fitting in parallel would be faster. It would, so I restructured `probe_uth.py` into two subcommands: `run <out_dir> <data_version> <n_workers> <run_id>...` (extract features, then probe) and `probe <out_dir> <n_workers>` (fit from a saved `features.npz`; each feature set in its own process, with CPU threads split evenly via `threadpoolctl`).
  - Probes now always fit on the float16 values stored in the npz, so `probe` reproduces `run` exactly. The killed sequential version fit on the fp32 in-memory values.
  - Tested on the smoke-test features: identical test accuracies to the sequential fp32 version for all 15 sets (max difference 0.0) [verified].
- The sequential job was killed. Its log is renamed `extract.log` on the VM. `meta.json` was written by hand for this run, since the old version wrote none; it says so in a `note` field. The parallel probe step started 18:21 UTC with 15 workers.

Open point for Julian: move `scikit-learn` from the `dev` group to the main dependencies, since the probe scripts need it on rented machines. Alternatively, `spd-vast` could sync with `--group dev`.

---

26-10-05, Claude Opus 5.5 — *stage 0 results: the gate is passed; untrained CIs 9 points below the residual stream*

Run: RTX 4090 VM, `data/uth/*/v1`, untrained CI network `s-7fad0c14` (bf16 frozen model). Probe step with 15 parallel workers, 18:21–~18:31 UTC; refit with the extended C grid 18:37–18:49. Results, logs and `meta.json` copied to `~/spd_out/26-10-05_uth_experiments/stage0/`. `results_Cgrid3.json` is the first fit with C ∈ {0.01, 0.1, 1}. `features.npz` (2.9 GB) stays on the VM only.

**C grid extended** [verified: `results.json` vs `results_Cgrid3.json`]: in the first fit every residual layer selected C = 0.01, the edge of the grid, and accuracies were still rising towards it. So I added C = 1e-3 and 1e-4 (`LOGREG_CS` in `probe_uth.py`). With the wider grid, tune accuracy now peaks inside the grid for layers 15, 19 and 23 (C = 0.01), so their selected scores are unchanged. Layers 11 and 27 and the layer-19 mean readout move to C = 1e-3.

**Results** [verified: `results.json`; logistic regression, C selected on the train-task held-out split (tune); one fit each; test = mean over the 8 test datasets (the paper's average), pooled in brackets; standard error of the mean roughly ±0.006]:

| features (last-token readout unless noted) | tune | test |
|---|---|---|
| residual L7 | 0.672 | 0.581 |
| residual L11 | 0.726 | 0.654 |
| residual L15 | 0.803 | 0.739 |
| **residual L19** | **0.848** | **0.807** (0.790) |
| residual L23 | 0.829 | 0.769 |
| residual L27 | 0.819 | 0.742 |
| residual L19, mean over tokens | 0.756 | 0.705 |
| **untrained CIs `s-7fad0c14`** | **0.791** | **0.715** (0.688) |
| untrained CIs, mean over tokens | 0.664 | 0.594 |
| mean log-probability | 0.570 | 0.559 |

Per test dataset, test accuracy:

| | cnn_dm | copa | hellaswag | nq_re | sciq | story_cloze | triva_qa | xsum |
|---|---|---|---|---|---|---|---|---|
| residual L19 | 0.739 | 0.965 | 0.789 | 0.633 | 0.843 | 0.961 | 0.722 | 0.805 |
| untrained CIs | 0.542 | 0.955 | 0.574 | 0.575 | 0.843 | 0.945 | 0.660 | 0.624 |

Per test category, residual L19 / untrained CIs: sentence completion 0.883 / 0.777, short-answer QA 0.733 / 0.693, summarization 0.772 / 0.583.

Mass-mean probes score below logistic regression throughout (residual L19 last: test 0.699; untrained CIs last: 0.707). The full table is in `results.json`.

**Reading** [concluded]:
1. **The gate is passed.** The best residual readout transfers to the held-out task categories at 0.807, far below the 0.95 threshold. So unlike on tiu, there is room above the baselines. Layer 19, the last decomposed layer, is best on both tune and test, so choosing it on tune and reading the test result is consistent.
2. **Untrained CIs are not at the ceiling here.** They score 0.715, 9 points below residual L19. On tiu the two were equal (0.998). The loss is concentrated in the long-context test tasks: summarization 0.583 vs 0.772, hellaswag 0.574 vs 0.789. On the short ones they match (copa, sciq, story_cloze within 1–2 points). So a random CI readout of layers 15–19 keeps less transferable truth information than a linear probe on the residual stream. That leaves room for a trained decomposition to do better than untrained, which is what stage 1 tests.
3. **Tuning on in-distribution data costs some cross-task accuracy.** For residual L19, C = 1e-3 would give 0.834 on test (tune 0.846 vs 0.848 at the selected 0.01). Choosing C on test would be cheating, so the reported number stays 0.807. Read it as about 0.81, with 0.83 attainable under stronger regularisation. Either way the gate conclusion holds.
4. **Not comparable to the paper's numbers.** Their LLaMA2-7b-chat attention-head probes scored 69.24 (LR) and 70.47 (mass-mean). Here the model is different, the readout is the residual stream, and story_cloze uses our cleaned intact-pair subset.
5. **Easy and hard test datasets:** copa and story_cloze are near 0.95 even for untrained CIs; nq_re is hardest (0.63). The 8-dataset average weights them equally.
6. Mean-over-tokens readouts are clearly worse than last-token readouts everywhere, as expected from the paper's choice.

**Stage 1 targets** (to fix before training): trained CIs on test, compared with untrained CIs at matched sparsity (top-k per sample); 0.715 is the dense untrained value, and the residual L19 value of 0.807 is the reference ceiling. For the record: the 4090 cannot run the training (`convos/julian/26-10-05_vast_rentals_SUMMARY.md`); stage 1 needs an H100.

The VM holds nothing else needed: results and logs are copied. `features.npz` isn't needed for stage 1, which re-extracts all features in one pass with the trained run. So the VM can be destroyed unless Julian wants that file.

---

C: (transcribed from chat) about scikit learn. Just make the vastai script install dev. I've still got the rental. You can do 3. there if the vram is enough.

---

26-10-05, Claude Opus 5.5 — *spd-vast installs the dev group; the 4090 is too small for item 3*

- `spd/scripts/run_vast.py`, `_build_sync_script`: `uv sync --no-dev --link-mode copy` → `uv sync --link-mode copy`, so the dev group (scikit-learn among others) gets installed on rented machines [decided: transcribed chat above]. A docstring line says why. Lint and basedpyright clean. `spd/utils/slurm.py` keeps its own `--no-dev`, which is unrelated to vast.ai.
- **Item 3 (stage 1 code and timing trials) can't run on the 4090** [concluded]. A tPD training step at batch 16 needed ~27 GB of GPU memory with the bf16 frozen model even at tiu's 24-token statements (`convos/julian/26-10-05_no_truth_baseline_SUMMARY.md`, measured on an H100), and 64- or 128-token targets need more; the 4090 has 24 GB. Timing measured on a power-capped 4090 would also not predict H100 step times, which is what the cap decision needs. So the code part (per-batch padding, unit test, uth config) is done on the laptop, and the timing trial and stage 1 run go to an H100. The 4090 can be destroyed.

---

C: (transcribed from chat) I can also rent a H100 now

---

26-10-05, Claude Opus 5.5 — *stage 1 code done: per-batch padding, unit test, WeightMagnitude fix, uth config, trial scripts*

I answered in chat to wait until the code was ready (an H100 rented now would mostly sit idle while I code), and offered to write the trial configs first if Julian wanted timing numbers earlier.

**Per-batch padding** [verified as below]:
- `spd/experiments/lm/prompts_dataset.py`: new `trim_to_loss_positions(batch)` cuts a batch after its last loss position. `create_prompts_data_loader` applies it to every batch through a `collate_fn` (and to the static single-batch loader) whenever a position mask exists, i.e. `loss_positions` `tokens` or `last_k`. With `all` (the padded arm) batches keep the full width, because padding is trained on there. Batches are still random, with no sorting, as Julian asked.
- `spd/run_spd.py`: for `LMTaskConfig` tasks, the persistent PGD sources are created with shape (batch, `max_seq_len`) instead of the first batch's shape.
- `spd/persistent_pgd.py`: new `_slice_to_batch`, called in `get_ppgd_mask_infos`, uses the first `width` positions of each source for a narrower batch. Positions beyond the batch get no gradient in that step.
- **`spd/metrics/weight_magnitude.py` (bug found by the smoke run)**:
  - The WeightMagnitude eval figure concatenated CI tensors of all eval batches along dim 0, which crashes when batch widths differ.
  - It also ignored the position mask, so its per-component mean and max CI included padding positions. **That also affected the existing tiu runs' weight-magnitude figures**: their mean CI was diluted by padding (~58% of positions at `max_seq_len` 24).
  - Now it selects the masked positions and flattens each batch to (positions, C), like `CIMeanPerComponent`. The other metrics in the config accumulate sums, and `CIHistograms` is not allowed with position masks at all.
- Not adapted: the multibatch PGD eval (`evaluate_multibatch_pgd`) uses one fixed `batch_dims`. With trimmed batches of different widths it would fail at the mask expansion (loudly, not silently). No config of ours uses it.

**Tests:**
- `tests/test_prompts_padding.py` (new). (a) Trimming cuts exactly after the last loss position, for `tokens` and `last_k`. (b) On a tiny Qwen2 decomposition (2 layers, C = 8, global shared MLP CI), a batch padded to 20 and the same batch trimmed to 12 give identical losses for ImportanceMinimality and PersistentPGDRecon (both with arm A's settings, read from its config) and UnmaskedRecon.
  - The stochastic losses can't be compared exactly (their random masks are drawn with the batch's shape); they share the routing and masked-loss code with the compared ones.
  - CIMaskedReconLoss isn't allowed with position masks.
  - Mutation check: with `position_mask=None` (padding counted) the test fails at ImportanceMinimality [verified], so it can detect padding entering a loss.
- `make test`: 173 passed; the only failure is the known rotgrid test (`FUTURE_WORK.md`). Explicit run of `tests/metrics`, the prepared-dataset tests and the new test: all pass [verified].
- **Loader on the real uth data** [verified: 401 batches of 16 from `config_uth_all_tokens.yaml`]: mean width 101.6 tokens (min 63, max 128), i.e. **0.79×** the target tokens of fixed padding to 128. The simulation had predicted 0.78×. Every batch's width equals its longest sample.
- **CPU smoke training** [verified: `~/spd_out/26-10-05_uth_experiments/smoke_train/`, exit 0]: the uth config on the tiny Qwen2, 10 steps, batch 4, PPGD from step 5, evals at steps 0, 5 and 10. All losses, metrics and figures were produced (including targeted CI heatmap and weight magnitude). Before the WeightMagnitude fix it crashed there.

**New config** `spd/experiments/lm/honesty_targeted_decomposition/config_uth_all_tokens.yaml`: identical to `config_truth_all_tokens.yaml` except `label: uth_all_tokens`, `max_seq_len: 128` and the 29 train-task `data/uth/*/v1` datasets [verified: line diff; validates as `Config`]. Its header says what it is and that the arm A settings are untested on this data.

**Timing trial scripts** (outside the repo, `~/spd_out/26-10-05_uth_experiments/trials/`): `make_uth_trials.py` and `run_uth_trials.sh`, the same design as the 26-10-05 dtype trials. 150 steps each, no WandB, no periodic eval; arm A vs uth, each with PPGD on from step 0 and without; nvidia-smi samples every 2 s. Decision rule (agreed earlier): keep cap 128 if a uth run costs ≤ ~2.5× an arm A run on the same machine.

**H100 plan:**
1. Trials, about 20–25 min including model loads.
2. Cap decision. If cap 128 is too expensive, build `data/uth/*/v2` at `--max_tokens 64` (the generator is unchanged) and rerun the stage 0 probes on its categories.
3. Stage 1 run with `config_uth_all_tokens.yaml` (WandB, final checkpoint).
4. Stage 1 probes: `probe_uth.py run <out> v1 15 <trained run> s-7fad0c14` on the same machine. Still to write while training runs: the matched-sparsity comparison (top-k CIs per sample, trained vs untrained).
