I want to work on creating an experiment for running tPD on Qwen. For that I guess that you and I need an overview because nothing about this is in the epistemic memory yet. I also want to prevent to have a bunch of old stuff creeping into the memory I don't know of. So let me describe what I hope has been done so far.

First of all, there is some legacy stuff, that has to be deleted or reviewed. Most of the stuff in `honesty_targeted_decomposition` falls undet this expcept `build_tiu.py` and `config_truth_statements`. The other stuff needs reviewing because I don't trust the old data generation and it's also not really clear from the repo where `prompts_liu` come from. Then also the old training used all token positions for tPD which includes a lot of right padding tokens. And finally the analysation scripts are just something I don't know what they are doing.

Let's come to what I actually want. The current goal is to see to which extend tPD can recover mechanisms regarding truthfullness/honesty and find evidence for it. The main goal is not to find a concrete decomposition. At this point I think that the epistemic memory might not have much about tPD so we should fill this blank first, then I can continue explaining the goal and the motivation.

---

26-10-02, Claude Opus 5.5 — *tPD background and legacy inventory*

I have read: INDEX.md, both existing SUMMARYs (prepared datasets, memory setup), `docs/CODING.md`, the full tPD paper (`papers/Targeted_Recovery_of_Weight_Space_Mechanisms/tpd_paper.md`), the targeted parts of `spd/run_spd.py` and `spd/configs.py`, the delta-mask code in `spd/utils/component_utils.py`, `spd/metrics/targeted_recon_loss.py` (header only), `config_truth_statements.yaml`, the diff of the legacy `config_liu_true.yaml` against it, and the heads of every file in `spd/experiments/lm/honesty_targeted_decomposition/`. I have not read the SPD/APD papers, the full metric implementations, or the validation scripts beyond their names. Nothing was run.

This entry has two parts: (A) what tPD is and how this code implements it, so the memory has a reference to build on; (B) an inventory of the legacy honesty folder, checking your three concerns against the files.

## A. tPD — the method

### A1. Background: parameter decomposition (PD)

PD (the family APD → SPD → VPD, all from Goodfire / Sharkey's group) splits each weight matrix `W` of a trained network into rank-1 **subcomponents** `U_c V_c^T`. `V_c` is the input direction the subcomponent reads, `U_c` the output direction it writes. A small auxiliary **causal-importance (CI) network** predicts, per input and per position, a CI value `μ_c ∈ [0,1]` for each subcomponent: how much the output depends on it. Training enforces (paper §Introduction): (1) subcomponents sum to `W`; (2) subcomponents with low CI can be ablated in any combination without changing the output; (3) each subcomponent is simple; (4) each input activates few subcomponents (**importance-minimality** loss, a p-norm penalty on CI). Requirement (2) is enforced by **reconstruction losses**: the model is run with each subcomponent scaled by a mask `m_c ∈ [μ_c, 1]` and its output must match the original (KL divergence for LMs). The masks are either sampled at random (**stochastic** recon loss) or chosen to maximise the loss (**adversarial / PGD** recon loss, "persistent PGD" keeps the adversarial masks across steps). Full-data PD has so far been scaled to a 4-block, 28M-parameter transformer [paper §Introduction].

### A2. What tPD changes

tPD (Vigouroux & Sharkey 2026, arXiv:2607.13047) only decomposes the mechanisms used on a chosen **target** dataset. Per matrix:

`W' = Σ_c m_c U_c V_c^T + m_Δ (W − Σ_c U_c V_c^T)`

`Δ` is a full-rank **catch-all component**: whatever of `W` the subcomponents don't explain. It is never decomposed and does not count towards importance-minimality.

Training uses two data streams [paper §Method, appendix *Method details*]:

- **Target batches** (the inputs you want explained): `m_Δ` is ablated (adversarially and stochastically, anywhere in [0,1]), so the target outputs must be reconstructed from the subcomponents alone.
- **Non-target batches** (sampled from "the model's original training distribution"): `m_Δ = 1` always. Subcomponents can still be used here.
- Importance-minimality runs on both streams (coefficient doubled on non-target). Consequence: anything that fires on non-target but not on target data gets pushed into `Δ`; subcomponents that do fire on target are shaped to interfere minimally with non-target data.

Why the non-target stream matters: PD on target data alone ("naive", Christensen & Riggs 2025) reconstructs target outputs, but subcomponents are unconstrained outside the subspace the target activations span, so ablating them has arbitrary effects off-target. In the paper's 12-block numpy/pandas experiment, naive-PD subcomponents all have heavy side effects on general Pile data when ablated; tPD ones do not [paper Fig. 4A].

### A3. Paper results (as reported, not reproduced here)

- **Toy model (TMCC):** targeting 3 of 100 input features recovers exactly those 3 mechanisms; converges in ~500 vs ~2500 steps for full PD; works with as few as 5 subcomponent slots [paper §Results, Fig. 1A–C].
- **4-block Pile transformer, prompts `import numpy as` / `import pandas as`:** two seeds give similar subcomponents (V vectors especially; Q/K U vectors less so). Nested-target consistency: of 77 subcomponents in the joint {np,pd} run, 69 match a subcomponent in one or both single-prompt runs [Fig. 1D–E].
- **CSS-only submodel (same 4-block model):** about 7% of the FLOPs of the published full decomposition; 1,638 subcomponents vs 9,972. Keeping only those 1,638 preserves CSS behaviour (KL ≈ 0.6) and destroys other languages (KL ≥ 2). Shortlisted CSS-specific subcomponents can be ablated to hurt CSS only [Figs. 2, 3].
- **12-block Pile transformer (85M non-embedding params):** 248 subcomponents for numpy/pandas. Ablating prompt-specific ones erases the `np`/`pd` completion with KL < 10⁻² elsewhere; swapping the U vectors of two pairs of prompt-specific subcomponents makes the model complete `numpy`→`pd` and `pandas`→`np` [Fig. 4].

### A4. Caveats the paper itself raises — relevant to a truthfulness target

1. **tPD subcomponents only cover the activation slice the target spans.** They are not the full-data subcomponents. A higher-rank mechanism is only recovered in the directions the target data uses (TMS-5-2-id example, appendix).
2. **Subcomponent merging on narrow targets.** Two mechanisms that always co-activate on the target data, with constant inner activations, can be merged into one subcomponent. The paper's np/pd run found 77 alive subcomponents where the full decomposition has 1,708 active on the same inputs. "The target data must be chosen so that it splits the components of interest at the desired granularity" [appendix *When do full-data PD and targeted PD find different subcomponents?*]. For us: a mechanism that distinguishes true from false can only show up as distinct subcomponents if the target data contains both, varied enough that they don't co-activate in lockstep. [concluded: follows directly from the paper's merging argument; not tested]
3. **Redundant mechanisms get lost into Δ.** Minimality keeps only enough mechanisms to reproduce the output; redundant copies vanish into Δ. Preliminary evidence of this in the CSS run [appendix *Does parameter-decomposition produce incomplete circuits?*]. Relevant if we want to claim "tPD found *the* truth mechanism".
4. **Scale.** The largest model in the paper is about GPT-2-small size. Qwen2.5-7B is roughly 80× larger by parameter count [concluded: 7B vs 85M non-embedding; order-of-magnitude only].
5. **tPD is unsupervised.** You give it inputs, not a behavioural metric. The researcher then has to "read the task off" the components. So "truthfulness" enters only through the choice of target data.

### A5. How this code implements tPD

[verified: read the code 26-10-02; not exercised by a run]

- **Switching it on:** `Config.nontarget_task_config` (`spd/configs.py`). If set, targeted mode is on; it requires `nontarget_batch_size`, `nontarget_eval_batch_size`, the same `task_name` as `task_config`, and forbids `FaithfulnessLoss` (which would force subcomponents to sum to `W`). `use_delta_component: true` (default) enables Δ. `nontarget_impmin_coeff_ratio` multiplies importance-minimality on non-target batches (paper value 2).
- **Training step** (`spd/run_spd.py`, the `for step in ...` loop): one target batch forward/backward, then one non-target batch forward/backward; gradients accumulate, then one optimizer step. This matches the paper.
- **Target batch:** all configured losses. Δ mask: uniform random in [0,1] in stochastic losses (`spd/utils/component_utils.py`, `calc_stochastic_component_mask_info`); an extra adversarial source in persistent PGD (`spd/persistent_pgd.py`).
- **Non-target batch:** same loss list minus `UnmaskedReconLoss` and both persistent-PGD losses, with `force_delta=1.0`. Note: the paper's appendix describes an *unmasked* non-target loss for the CSS run ("all m_i = 1 and m_Δ = 0"). Here `UnmaskedReconLoss` is excluded from non-target and runs on target batches only. I have not checked which matches what the authors actually ran. [assumed: minor; flag only]
C: the code for masked and unmasked came from me in an earlier session. For me it's important to know how the paper ran the training (I suspect no masks).
Answer (26-10-02, Claude Opus 5.5): I checked the git history, and it splits in two. The **position masks** (`loss_positions`) are yours: commit `c774a4f` (26-10-01). The exclusion of `UnmaskedReconLoss` and the PPGD losses from non-target batches is **Antovigo's** (paper author), commit `cfcbd8a` (26-02-25). So that part is the paper's code. [verified: `git log -S`, `git blame`]

How the paper ran training, as far as the published configs and code show [verified: configs in `spd/experiments/lm/targeted_decomposition/`, the pre-`c774a4f` `prompts_dataset.py`, and the GPT-NeoX tokenizer, 26-10-02]:
- **No position masks.** That code didn't exist, so every position counted. They also didn't need masks:
  - numpy/pandas: `max_seq_len: 3`, and `import numpy as` / `import pandas as` are exactly 3 GPT-NeoX tokens each (`[2948, 36950, 347]` / `[2948, 48340, 347]`, no BOS), so there are zero padding positions.
  - CSS: target `Antovigo/pile-css-chunks` and non-target `danbraunai/pile-uncopyrighted-tok-shuffled` are pre-tokenised packed chunks of length 512. Packed means no padding. [assumed: from the dataset names and `is_tokenized`; I didn't open the datasets]
- **Unmasked recon loss:** only the CSS run used it (coeff 0.2). In the code it runs on **target** batches only, with all `m_i = 1` and **no Δ** (`make_mask_infos` without deltas). The paper appendix says it was a *non-target* loss. I think the appendix is wrong, not the code [concluded]: an unmasked loss with Δ=0 on Pile data would force the subcomponents alone to reproduce Pile outputs, which is exactly what tPD avoids. On target batches it does what the appendix says it's for ("prevent dead components from interfering with the reconstruction").
- So for our Qwen runs, using `loss_positions: tokens` on the target (as `config_truth_statements.yaml` does) is a deviation from the paper. It is forced by our variable-length statements, and it restores what the paper got for free: no padding in any loss.
- **Positions:** `LMTaskConfig.loss_positions` ∈ `{all, tokens, last_k}` builds a position mask that drops positions from every loss and eval metric. Added in `c774a4f` (26-10-01). Default is `all`, padding included.
- **Target data sources:** exactly one of `dataset_name` (HF), `prompts_file` (text, one prompt per line, right-padded to `max_seq_len`), `prepared_datasets` (versioned dirs under `data/`).
- **Targeted eval metrics:** `TargetReconLoss`, `NontargetReconLoss`, `TargetedCIHeatmap` (`spd/metrics/`).
- **Paper reproduction configs:** `spd/experiments/lm/targeted_decomposition/{numpy_and_pandas_4L, numpy_and_pandas_12L, css}/`, including `*_naive.yaml` (no non-target stream) and `*_alt_seed.yaml` variants.
- **Paper analysis scripts:** `spd/scripts/validation/` (e.g. `swap_test.py`, `effect_of_ablation.py`, `compare_to_larger.py`, `multilang_ablation.py`) and `spd/scripts/completeness_tests/`. These are the authors' tools for the ablation, rewiring, and completeness analyses above. They are probably the right starting point for *our* analyses, rather than the legacy honesty scripts. [assumed: from file names and docstrings only]

### A6. Repository lineage (corrects an imprecision in INDEX.md)

The paper's own code is `github.com/Antovigo/targeted-parameter-decomposition` [paper §Code availability]. This repo's `upstream` remote is `bbzmdn/targeted-parameter-decomposition`. All honesty-experiment files were added there in one commit, `5b49341` (26-09-01, author Madina Babazhanova, "capital honesty experiments"). So the chain is: Antovigo (paper code) → bbzmdn (Madina's fork, adds the honesty experiments) → julxi (this fork). [verified: `git log`, `git remote -v`, paper text] INDEX.md calls bbzmdn the upstream of "the code accompanying the paper". That is true of the remote, but it hides that bbzmdn is itself a fork. I'll add the lineage there.

## B. Legacy inventory: `spd/experiments/lm/honesty_targeted_decomposition/`

| File(s) | What it is | Your concern / my assessment |
|---|---|---|
| `build_tiu.py`, `config_truth_statements.yaml` | Current: tiu generator (prepared-datasets topic), and the Qwen tiu config | Keep (your call). |
| `contrast_pairs.jsonl` (71 lines) | Capital-city Q&A pairs with honest/dishonest system prompts and *model-generated* completions plus `honest_pass`/`dishonest_pass` flags | The capitals prompts were built from this. How it was generated (which model, what filter) is not in the repo. Docstrings also mention a `contrast_pairs_verified.jsonl` that doesn't exist here. |
| `build_prompts_capitals.py` → `prompts_capitals/` | 35 "The capital of X is Y." true and 35 false statements | Generator exists, but the source jsonl's provenance is unknown. Docstring names a different script and output dir (`build_bare_prompts_capitals.py`, `prompts_bare_capitals`). |
| `prompts_liu/` (181 true / 179 false / 360 combined) | Multi-domain true/false statements ("Mount Everest is the highest mountain…") | **Confirmed: no generator in the repo.** The README attributes it to "Liu et al." with no citation or link. The configs reference `build_prompts_files.py` and `memory_dryrun.py`, neither of which is in the repo. Never run through a decomposition (README). |
| `config_capitals_{true,false}.yaml`, `config_liu_{true,false}.yaml` | Legacy Qwen2.5-7B-Instruct tPD configs: layers 15–19 `mlp.down_proj`, C=96, non-target = `monology/pile-uncopyrighted` | **Padding concern confirmed:** they use `prompts_file` with no `loss_positions`, so the default `all` applies. The capitals statements are a single short sentence padded to `max_seq_len: 64` (liu: 128), so most positions in every target loss were padding. That includes importance-minimality, which therefore partly optimised for padding tokens. [concluded: from the config and the `AllPositions` default; exact padding fraction not measured] Their comments contain the useful OOM history, summarised below. |
| `find_top_components.py`, `visualize_token_firing{,_multilayer}.py` | Post-hoc analysis: load a checkpoint, rank components by \|mean CI on true − mean CI on false\| at the answer token, render per-token CI heatmaps as HTML | See below. |
| `analysis_capitals/` (SUMMARY.md, top_components.json, HTML) | Write-up of two runs (`s-d30f8be0` capitals_true, `s-047bca0c` capitals_false). Headline: the false-only run has components separating false from true; the true-only run has almost none alive. | Results are from padding-dominated training (above). The checkpoints live in `~/spd_out/spd/` on whatever machine ran them, not here. |

**What the analysis scripts do, and why I would not trust the headline result** [concluded, reasoning stated]: `find_top_components.py` compares CI at the *answer token* between "The capital of France is Paris." and "The capital of France is Lyon." But the two examples differ in that very token. A component that fires on "Lyon"-type tokens, or on low-probability continuations, separates the two classes without encoding anything about truth. So the separation is confounded with token identity and surprisal. Also, a decomposition targeted only on false statements will naturally have alive components on false inputs and dead ones elsewhere. The true/false asymmetry may just reflect *which data was the target*, plus the padding issue. Together with your distrust of the data, I'd treat `analysis_capitals/SUMMARY.md` as having no evidential value.

**Useful knowledge to salvage from the legacy config comments** before deleting [assumed: from the comments' own account; I did not reproduce these OOMs]:
- Decomposing all 28 layers × 7 projections of Qwen2.5-7B ran out of memory before step 0. `calc_weight_deltas()` materialises one dense `d_out × d_in` Δ tensor per decomposed module, all at once. Hence the narrowing to 5 `down_proj` matrices.
- The KL reconstruction loss with Qwen's ~152k vocabulary makes logits (batch × seq × vocab) the per-step memory bottleneck, hence `batch_size: 4`. `TargetedCIHeatmap` ignores the non-target batch size and OOM'd at `n_nontarget_examples: 100`.
- The non-target corpus must be tokenised with Qwen's tokenizer. The paper configs' `danbraunai/pile-uncopyrighted-tok-shuffled` is pre-tokenised for GPT-NeoX, so it can't be reused.
- The layer choice 15–19 is described as "informed by the Mahalanobis probe paper", with no citation. It's a guess, not validated.
- GPU used: the comments imply a 24 GB card.

`config_truth_statements.yaml` already inherits these settings.

### Proposed handling of the legacy folder (for you to decide)

- **Delete:** `contrast_pairs.jsonl`, `build_prompts_capitals.py`, `prompts_capitals/`, `prompts_liu/`, the four legacy configs, the three analysis scripts, `analysis_capitals/`, and the honesty section of `README.md`. The memory-relevant knowledge is preserved in this LOG (the bullet list above). Git history keeps the files if anything is needed later.
C: ok, let's delete them
Answer (26-10-02, Claude Opus 5.5): Done; see the entry *legacy deletions and answers to comments* below.
- **Optionally keep `prompts_liu` as a candidate source** only if you can find where it came from. Without provenance it shouldn't be used as target data.
C: You can delete it. I know for a fact though that the data came from the repo for the "On the Universal Truthfulness Hyperplane Inside LLMs" paper.
Answer (26-10-02, Claude Opus 5.5): Deleted. Recorded the source in the SUMMARY as your statement. (That paper is, I believe, Liu et al., EMNLP 2024, which explains the "Liu et al." in the old README; I haven't checked the citation.)
- **Rename the folder?** Once only `build_tiu.py` and `config_truth_statements.yaml` remain, "honesty_targeted_decomposition" overstates things. The tiu data is about factual truth of statements, not honesty. Not urgent.
C: let's leave it.
Answer (26-10-02, Claude Opus 5.5): OK, the folder name stays.

## C. Points to raise before the Qwen experiment design (not decisions)

1. **Truthfulness vs honesty.** The tiu data (true/false *statements*, no speaker intent) can only target the model's representation and processing of *factual truth*. Honesty (saying what you believe vs lying) needs data where the model's belief and its output can diverge, e.g. the instructed-lying setup the legacy `contrast_pairs.jsonl` was heading towards. These are different targets. Which one is the goal?
C: Both, we start with honesty and see how far we get.
Answer (26-10-02, Claude Opus 5.5): Noted. One consequence: the current target data (tiu: bare true/false statements) is a truthfulness dataset, not an honesty one. Starting with honesty needs data where the model's output departs from what it knows (e.g. instructed lying). I'll ask about this once you've explained the goal.
2. **Target composition.** Per caveat A4.2, a target of only true statements (as in the current `config_truth_statements.yaml`) can't separate a "true" mechanism from a "false" one. Merging and absorption into Δ make that worse. A mixed true+false target seems necessary to find components whose CI differs by truth value. A label-matched contrast also needs controls for token identity (A–B above).
C: This is based on general arguments. I want to do experiments for this. So there is no "this can't do that". We can suspect it won't work but we might also be wrong.
Answer (26-10-02, Claude Opus 5.5): Fair. "Can't" was too strong: it is a prediction from the paper's merging argument, not an established fact. Restated as a testable hypothesis: *with a true-only target, tPD will not yield components whose CI differs systematically between true and false statements on held-out data; with a mixed target it will.* A true-only vs mixed comparison on the same statements tests it directly.
3. **Non-target distribution.** The paper samples non-target data from the model's *own training distribution*. For Qwen2.5-7B-Instruct, the Pile is only a proxy, and it contains no chat-formatted text. Mechanisms used on chat-like inputs but not on the Pile would be pushed into subcomponents rather than Δ, inflating the decomposition. Probably acceptable for bare statements, but worth stating.
C: yes, the whole chat template issue is still open. For now I want a simple experiment to see how the decomposition works.
Answer (26-10-02, Claude Opus 5.5): Agreed; recorded as open, not blocking.
4. **Instruct model, no chat template.** The tiu statements are fed as raw text to an instruct model. That's fine for "how does the model process a true vs false statement". It is less natural for honesty-type questions.
C: yes, but as I said let's keep it simple for now. We can expand our experiments step by step.
Answer (26-10-02, Claude Opus 5.5): Agreed.
5. **Compute.** The paper's np/pd runs used 30k steps with batch 128–256; the current config has 10k steps at batch 4. Whether tPD converges in that regime on a 7B model is unknown. [assumed]
C: true, this is an issue. Training takes very long so that checking this might not be possible.
Answer (26-10-02, Claude Opus 5.5): Two cheap partial checks, as options rather than a plan:
- **Watch the curves.** Watch the eval curves already logged (`TargetReconLoss`, `NontargetReconLoss`, `CI_L0`) and see whether they plateau before the end.
- **Iterate on a smaller Qwen first.** Qwen2.5-0.5B or -1.5B-Instruct share the tokenizer and architecture family, so configs and data carry over. Runs there would be fast enough for seed and convergence comparisons before committing 7B compute.

The paper's own pitch for tPD is fast iteration, which argues for the small-model route. The risk is that mechanisms differ with scale. [assumed: not tried]

**Next:** I'll write the SUMMARY and add the INDEX row. I'm waiting for your go/no-go on the legacy deletions and for the rest of your explanation of the goal and motivation.

C: Before I give the explanation of the goal I have a question about the paper. How do they train the decomposition for numpy/pandas (this is an extension of what I remarked above). How does this exatcly work? The target data is the pile. How is the loss computed for it? I suppose that when simplified they ran the sequence through the decomposition with m_Δ = 1 and the other m_c sampled (or adveserially selected) for each position independently. This inparticular means that errors in the residual stream compound over later sequence positions, right?
What was their target data? Just the imports? And did they train the decomposition on the whole import?

Answer (26-10-02, Claude Opus 5.5): All numbers below were re-read from the configs and code; the tokenisation was checked with the GPT-NeoX tokenizer [verified 26-10-02].

**The target data is not the Pile; the Pile is the *non-target* data.** Target = only the two prompts `import numpy as` and `import pandas as` (`prompts/numpy_and_pandas.txt`, `max_seq_len: 3`). Each is exactly 3 tokens, so the whole import is the sequence, with no padding. The batch of 256 (128 in the 12-block run) is just these 2 prompts repeated 128× (`StaticBatchLoader`). Every copy gets its own random masks, so the repeats only sample more mask configurations. The KL loss is computed at all 3 output positions:
- after `import`: the next-token distribution over module names; identical for both prompts;
- after `import numpy`: predicts ` as`;
- after `import numpy as`: predicts ` np`.

So yes: they trained on the whole import, and the decomposition has to reproduce the model's full output distribution at all three positions, not only the final ` np`.

**Per training step** (`spd/run_spd.py`), with W' = Σ_c m_c U_c V_c^T + m_Δ·Δ:
1. **Target batch** (the 2 prompts):
   - `StochasticReconSubsetLoss` (coeff 1). Masks m_c = μ_c + (1−μ_c)·u with u ~ U(0,1), drawn **independently for every (sequence, position, component)**. m_Δ ~ U(0,1) independently per (sequence, position). Plus *routing* (`uniform_k_subset`): at each position a random number k of the decomposed matrices use W', and the others use the original W (`spd/routing.py`).
   - `PersistentPGDReconLoss` (coeff 0.5, from 80% of training on, or 20% in the 12L run): adversarial masks over the components **and Δ**, one persistent adversarial source per (sequence, position) (`scope: per_batch_per_position`).
   - Importance-minimality (on CI over all positions).
2. **Non-target batch** (Pile, 64 tokens, packed):
   - The same stochastic subset loss, but with **m_Δ = 1** fixed. Then W' = W − Σ_c (1−m_c) U_c V_c^T, so the only perturbation is removing part of subcomponents whose CI is low at that position.
   - Importance-minimality ×2.
   - **No PGD and no unmasked loss.**
3. Gradients from both batches accumulate; one optimizer step.

**Your compounding question: yes, with a qualification.** Each loss is one forward pass with *all* masks applied at once, at every decomposed matrix (in the paper all of q/k/v/o/up/down in every layer) and every position. A perturbation at layer l, position t changes the residual stream there. Later layers at t read that change, and through attention so do later positions t' > t. The KL at t' therefore measures the joint effect of all upstream masks, across layers and earlier positions. With 3 tokens this is shallow; with 64 Pile tokens it is the full depth of the context.

The qualification: this is deliberate, not an approximation. Requirement (2) of PD is that low-CI components can be ablated *in any combination* without changing the output, and joint, compounding ablation is that requirement. Two mechanisms limit the compounding:
- Subset routing leaves a random fraction of matrices at their original weights in each pass. This was introduced in SPD/VPD to make the problem easier [assumed: from the name and docstring; I haven't read the VPD paper].
- On non-target data, the only error source is the low-CI part of the subcomponents (Δ is on). Components that are dead on the Pile cost nothing there.

**For our Qwen setup:** only 5 `down_proj` matrices (layers 15–19) are decomposed, and no attention weights. Errors still propagate across positions, though. Layer 15's down-proj output at position t enters the residual stream, and attention in layers 16–27 carries it to later positions. So the loss at the statement's last tokens includes the effect of masks applied at earlier tokens.

---

26-10-02, Claude Opus 5.5 — *legacy deletions and answers to comments*

Context: same session as the entry above. I answered all of Julian's C: comments in place. The ones that change the record:
- How the paper trained numpy/pandas: answered beneath the last comment.
- Who wrote the masking code, and the paper's use of masks: answered beneath the comment in A5.
- The paper's appendix places the unmasked recon loss on non-target batches; the code places it on target batches. [concluded: the appendix is likely wrong]

**Deleted** from `spd/experiments/lm/honesty_targeted_decomposition/`, per Julian's approval in the C: comments under "Proposed handling of the legacy folder" [decided]:
- `contrast_pairs.jsonl`, `build_prompts_capitals.py`, `prompts_capitals/`, `prompts_liu/`
- `config_capitals_{true,false}.yaml`, `config_liu_{true,false}.yaml`
- `find_top_components.py`, `visualize_token_firing.py`, `visualize_token_firing_multilayer.py`, `analysis_capitals/`
- the "Honesty Targeted Decomposition Example" section of `README.md`

Plain `rm`, nothing staged. The folder now holds only `build_tiu.py` and `config_truth_statements.yaml`; its name stays [decided].

Follow-up edits:
- **`config_truth_statements.yaml` header:** it said settings were "carried over from config_capitals_true.yaml", a now-dangling reference. It now states the substance in place: settings adapted from the paper's numpy 4L config, not tuned for Qwen, narrowed for memory, layer choice an unvalidated guess.
- **`convos/julian/26-10-02_prepared_datasets_SUMMARY.md`:** dropped its mention of the deleted `build_prompts_capitals.py`.

**Provenance of the deleted `prompts_liu`:** per Julian, it came from the repo of the paper "On the Universal Truthfulness Hyperplane Inside LLMs" [Julian's statement in a C: comment; citation not checked].

**Status:** waiting for Julian's explanation of the goal and motivation. Goal: both honesty and truthfulness, starting with honesty [decided]. Keep the first experiment simple: no chat template yet, expand step by step [decided].

C: Ok, so here is the motivation. Another result which is not in this repo but claims (I believe the author, I'm just not sure if the setup was free of errors, that's why I'm saying claims here) that training tPD for the middle layers (as in the config) on target data selected from some domains of the "On the Universal Truthfulness Hyperplane Inside LLMs" paper and non-target data from the pile does uncover something interesting. They also used padding which means that a lot of the target tokens the decomposition was trained on where padding tokens.
They claim that they can use the learned ci functions to test for factuality using probing methods that like logistic regression or mass mean probes but not on the activations of the network but the intensities of the ci functions.
This is a bit wild for me. We have these components that are trained on whole statements why should they pick up truthfullness. Maybe there is something to discover here. For now this result is very counterintuitive and also kind of wild to me. That is why I want to start my own investigation.
So the goal for now is: Just have simple factual statements for target data and see what happens when we use all tokens or just the last token for traning the decomposition on the target data.
The longterm goal is to see if tPD can be used for truthfulness/honesty detection and if so how it compares to other methods.

---

26-10-02, Claude Opus 5.5 — *response to the motivation: CI probing claim*

Context: same session. In addition to the above, I checked how position masks enter the losses (`spd/metrics/stochastic_recon_subset_loss.py`, `importance_minimality_loss.py`), the CI network (`GlobalSharedMLPCiFn` in `spd/models/components.py`), and Qwen2.5-7B's config and tokenizer config.

## Restating the motivation (for the record)

**The claim** is reported by Julian, from someone else's work that is not in this repo. Julian believes the author but is unsure the setup was error-free [assumed: unreplicated]. tPD was run on the middle layers of Qwen (as in the config). The target was statements from some domains of the "Universal Truthfulness Hyperplane" dataset; the non-target was the Pile; padding was included in training. A linear probe (logistic regression or mass-mean) fitted to the **CI values**, not to the activations, then detects whether a statement is true.

**Immediate goal** [decided: Julian's comment above]: simple factual statements as target. Compare training the decomposition on all tokens vs only the last token.

**Long-term goal:** can tPD be used for truthfulness/honesty detection, and how does it compare with other methods?

## Why the claim is less wild than it looks, and what that implies for the test

[concluded, reasoning stated]

1. **The CI network is itself a learned readout of the activations.** In our config (`ci_config: global_shared_mlp`, `hidden_dims: [512]`), the CI network takes the concatenated inputs of all decomposed matrices at a position. Those are the MLP hidden activations of layers 15–19, 5 × 18,944 = 94,720 dims for Qwen2.5-7B (`intermediate_size: 18944`, read from the model config). It maps them through one 512-unit hidden layer to 5 × 96 = 480 CI values. So "probing the CIs" means probing a 480-dim nonlinear projection of the mid-layer activations at that position. The truth-probe papers (Bürger et al.'s Truth-is-Universal, Liu et al.'s hyperplane paper) claim that truth is linearly decodable from exactly such mid-layer activations [assumed: from the literature, not re-checked here]. So the **null hypothesis** is mundane: *CIs carry truth information because nearly any reasonably rich projection of these activations does.* The claim becomes interesting only if the trained tPD CIs do something a generic projection doesn't. Examples: comparable accuracy from a few individual components, better generalisation, or interpretable components.
2. **The reconstruction objective can also force truth-sensitivity directly.** After a false statement, the model's next-token distribution plausibly differs from after a true one (surprise, likelihood of a correction). If so, reconstructing the output at the final positions needs mechanisms that respond to truth. This is a reason to expect a signal at the end of the statement specifically. A competing explanation is a **surprisal** confound: false statements contain low-probability tokens.
3. **Padding may have helped rather than hurt.** Padding is on the right and the model is called without an attention mask. Causal attention therefore means real tokens never see the pads; the pads only add extra positions. Each pad position (Qwen's pad token is `<|endoftext|>`, per its tokenizer config) attends to the whole statement. Pads act like extra end-of-statement positions, which is where truth probes usually work best. So padding may be part of *why* the reported result appeared, not just noise. [assumed: plausible mechanism, untested]

## What the "last token only" arm actually trains

[verified: code read 26-10-02]

With `loss_positions: {type: last_k, k: 1}`:
- The stochastic and adversarial masks (including Δ) are still applied at **every** position in the forward pass.
C: I don't think that I want this. This would mean that the decomposition is indirectly trained to do more than just predicting the internal activations for the last token (where I hope the truthfulness signal is a salient part). Maybe later in the investigation we will come back to this, but now I masked positions being computed by the full model.
Answer (26-10-02, Claude Opus 5.5): Implemented. Positions outside `loss_positions` now run on the original weights in every masked forward pass (training losses and eval metrics); only the selected positions use the masked decomposition. Mechanism: the existing per-position *routing mask*, which already switches a position between component output and original module output, is ANDed with the position mask (`route_only_selected_positions` in `spd/models/components.py`). I made this the behaviour for all `loss_positions`, not an option: for `tokens`, the unselected positions are the right-padding, which real tokens never attend to, so nothing changes there. Details in the entry *arm configs and position routing implemented* below.
- Only the reconstruction KL and importance-minimality are restricted to the last position.

Consequences:
- At earlier positions, Δ is still ablated, so the subcomponents must carry everything those positions pass forward (via attention) to the last token.
- CI at earlier positions has **no sparsity pressure**, so it is unconstrained and should not be probed. In this arm, only CI at the last position is meaningful.
- For tiu statements the last token is the period (`.` in "The city of Niigata is in Japan."). The truth-probe literature often finds that the end-of-sentence token carries the strongest truth signal. [assumed: from the literature]
C: don't know if this can be confirmed. This is simply a hypothesis that we use for our setup.
Answer (26-10-02, Claude Opus 5.5): Agreed: a working hypothesis of the setup [assumed]. One fact from the data, though [verified: Qwen tokenizer over all 8,354 tiu statements, 26-10-02]: the last token is `.` for 7,652 statements and `'.` (one token) for 702, namely the Spanish-translation ones ("… means 'y'."). So the probed token's identity differs between domains, which matters for cross-domain probe generalisation.

## Proposed design for the first experiment (for Julian to decide)

**Arms**, identical except for the target `loss_positions`:
- (A) `tokens`: all real tokens, no padding.
- (B) `last_k, k=1`.
- (C, optional) `all` with padding to a fixed length. This replicates the reported setup and isolates whether padding matters (point 3).

C: yes that is reasonable. For the padding experiment we should make sure that at least 50% of the tokens being trained on are padding tokens.
Answer (26-10-02, Claude Opus 5.5): Satisfied with `max_seq_len: 24`. Train statements average 10.03 Qwen tokens (max 20), so padding is 1 − 10.03/24 ≈ 58% of trained positions [verified: tokenizer count over the train splits, 26-10-02]. All three arms use 24, so they differ only in which positions are trained.
C: Also as a todo, after we have seen the results of these experiments: If (A) gives good signals for truthfulness, we can test your hypothesis above by doning an experiment on similar sentences which don't cary a trurth signal (don't know yet, how to desing the data) and see if the accuracy compares.
Answer (26-10-02, Claude Opus 5.5): Recorded as a follow-up in the SUMMARY. A thought for the data design, when we get there: the control must keep the *labels* but remove the truth relation. One example is a probe trained to predict a random label assignment over the same statements (the standard "control task" for probes). Another is pairs differing in a truth-irrelevant token. The untrained-CI baseline (step-0 checkpoint) already covers part of point 1 for free.

**Probe protocol:**
- Extract CI at the last real token of each statement on the held-out **test** split. The tiu splits are already by subject, so no subject leaks from train to test.
- Fit logistic-regression and mass-mean probes; report accuracy and AUROC.
- Open detail to fix before running: which CI value to probe (pre-sigmoid, `lower_leaky`, or `upper_leaky`). Pre-sigmoid avoids saturation at 0/1.

**Baselines** (without them, a positive result can't be interpreted):
1. **CI from the step-0 checkpoint** (untrained CI net = a random nonlinear projection, same pipeline). This is the cheapest and most important control for point 1. It needs a save at step 0 or a fresh init.
2. **Probe on the raw activations** (the same MLP inputs, or the residual stream) at the same position. This is the standard method that tPD would be compared to in the long run.
3. **Statement log-probability under the model**, as a one-feature probe. This controls for surprisal.

**Generalisation tests** (the tiu data was prepared for these):
- Train the probe on some domains and test on others.
- Train on affirmative statements and test on negated ones (`neg_*`). The Truth-is-Universal paper's central point is that naive probes fail under negation. Generalising across negation is the strongest evidence of a truth representation rather than a surface feature. [assumed: from that paper's framing]

C: I want the training of the decompositions and the analysis of them to be different scripts because analysis is cheap and training is expensive
Answer (26-10-02, Claude Opus 5.5): Agreed. Training is the existing `spd/experiments/lm/lm_decomposition.py` with the arm configs. The probe analysis will be a separate script that only loads saved checkpoints. To make baseline 1 possible without retraining, checkpoints are now also saved at **step 0** (before the first optimizer step) whenever `save_freq` is set. The arm configs save at 0, 2000, …, 10000.

## Questions for Julian

1. **Target composition.** True statements only, or true + false mixed? And which tiu domains? The current config uses the four affirmative `*_true` sets. Do you know which the reported setup used? Probing needs false statements only at *eval* time, so a true-only target is a legitimate arm. Training on both is the other natural choice.
C: let's take all domains in data to see our ceiling for now
Answer (26-10-02, Claude Opus 5.5): Done: all 20 tiu datasets (true and false; affirmative and negated; cities, sp_en_trans, element_symb, animal_class, larger_than, smaller_than) in all three arm configs. I read "all domains" as including both labels. Tell me if you meant true-only.
C: follow up questions. According to the current config `config_truth_statements.yaml` we have 4 * 10000 traning sequences. We don't have more data than that right?
Answer (26-10-02, Claude Opus 5.5): Not quite: 10,000 is the number of **steps**, at batch size 4. So training *draws* 40,000 sequences, but from a much smaller pool of distinct statements [verified: line counts of the split files, 26-10-02]:
- The old config's four `*_true` sets had **851** distinct train statements (cities 582, sp_en_trans 142, element_symb 65, animal_class 62). That is about 47 passes over the data.
- All 20 tiu datasets: **6,560** train and **1,794** test statements, about 6 passes over the data in 10k × 4 draws.

Each step also draws a non-target batch of 4 Pile sequences. tiu has no more data than this. More target data would have to come from elsewhere (e.g. the Liu et al. hyperplane datasets).
2. **Can you get details of the reported setup?** Specifically: which positions they probed, which CI values, and whether they compared against an activation probe or an untrained-CI baseline. If they had no baseline like 1 above, that alone could explain the result.
C: I don't know the details. They didn't probe one particular position though but used maybe the mean or something. And no there was no baseline
Answer (26-10-02, Claude Opus 5.5): Noted. The analysis will therefore report both the last-token CI and the mean CI over real tokens (for arm C, also the mean including pads, to match the reported setup). In arm B only the last-token readout is meaningful: CIs at the other positions get no training signal. Without a baseline, the reported result is consistent with the null hypothesis above. Our step-0 baseline settles that.
3. **Model size.** Run the first arms on Qwen2.5-7B-Instruct (comparable to the claim), or on a smaller Qwen first, to get several arms and seeds through quickly?
C: Run the first arem on the 7B
Answer (26-10-02, Claude Opus 5.5): OK: all three arm configs use Qwen2.5-7B-Instruct.
4. **Arm C (padding replication):** include it or not?
C: yes
Answer (26-10-02, Claude Opus 5.5): Included as `config_truth_padded.yaml`.

Small flag: earlier you said "start with honesty". This first experiment (bare factual statements) is a truthfulness experiment. That fits "keep it simple", but I'm noting it so the record doesn't drift.
C: Good for noticing. I meant truthfulness. 
Answer (26-10-02, Claude Opus 5.5): Recorded: the goal is truthfulness (statement truth) [decided]. The earlier "start with honesty" is superseded.

---

26-10-02, Claude Opus 5.5 — *arm configs and position routing implemented*

Context: same session. I implemented the training side of the first experiment per Julian's comments in the entry *response to the motivation: CI probing claim*. Not committed. **Not run on the 7B model**; that is Julian's (expensive) call.

**Code changes:**
1. **Positions outside `loss_positions` run on the original weights** [decided: Julian's comment].
   - New `route_only_selected_positions(mask_infos, position_mask)` in `spd/models/components.py`: it ANDs the position mask into each module's routing mask. The forward hook (`spd/models/component_model.py`) already uses the original module output wherever the routing mask is False.
   - Applied at all six masked forward passes that take a position mask: `stochastic_recon_subset_loss.py`, `persistent_pgd.py`, `pgd_utils.py` (PGD eval), `unmasked_recon_loss.py`, `targeted_recon_loss.py`, `hidden_acts_recon_loss.py`. Other metrics are rejected by config validation for non-`all` positions anyway.
   - The `loss_positions` field description in `spd/configs.py` and the legacy `docs/PROJECT_REFERENCE.md` were updated to match.
   - New unit test: `tests/metrics/test_recon_losses.py::test_route_only_selected_positions_uses_original_weights_elsewhere`.
2. **Step-0 checkpoint** (`spd/run_spd.py`): saved whenever `save_freq` is set, before the first optimizer step. This is for the untrained-CI baseline. It affects every experiment with `save_freq` (one extra small file).
C: I don't want Step-0 checkpoints. If it's for untrainde-CI baseline only I don't think it should be included in each run. Rather we can just do a 0-step train or whatever is fair if there are any warmup steps.
Answer (26-10-02, Claude Opus 5.5): Done: the step-0 save is reverted, and `steps: 0` is now a supported "untrained baseline" run.
- **Why it needed a change:** `steps: 0` used to crash with a division by zero (`frac = step / config.steps` in the loop).
- **What a 0-step run does now:** `optimize` saves the freshly initialised decomposition as `model_0.pth` and returns before the loop. No training, no evals.
- **Fairness:**
  - The only pre-loop warmup is `faithfulness_warmup_steps`, which is 0 in our configs.
  - PPGD's warmup happens inside the loop and only shapes the adversary, not the weights.
  - [verified, tiny-model smoke tests 26-10-02] The 0-step `model_0.pth` is bit-identical to the step-0 weights of a normal run with the same config.
  - The step-0 weights are also identical between the last-token and padded arms. So **one** untrained run (any arm config with `steps: 0`) serves as the baseline for all three arms. This holds as long as seed, model, `module_info` and `ci_config` stay shared.

**Configs** (`spd/experiments/lm/honesty_targeted_decomposition/`):
- `config_truth_all_tokens.yaml` (arm A, `tokens`), `config_truth_last_token.yaml` (arm B, `last_k k=1`), `config_truth_padded.yaml` (arm C, `all`).
- All three use the 20 tiu datasets with `max_seq_len: 24`. Model, layers, losses and steps are unchanged from the old config.
- They differ only in `loss_positions` and `label` (checked with `diff`).
- `config_truth_statements.yaml` was replaced by these (arm A is its successor). References in INDEX.md, `docs/PROJECT_REFERENCE.md` and the prepared-datasets SUMMARY were updated.

**Verification:**
- `make check`: ruff clean. basedpyright reports 6 errors, all in `spd/experiments/rotgrid/analysis/rank_rotgrid_decompositions.py`, a file this session did not touch [verified: not in `git status`]. Changed files: 0 errors.
C: is there a todo that can be used to make sure this will be fixed and not forgotton
Answer (26-10-02, Claude Opus 5.5): Yes: added as an entry in `FUTURE_WORK.md` (the project's list for deferred items that belong to no single topic).
- Full test suite: 510 passed, 15 skipped. One failure, `tests/test_rotgrid_decomposition.py::test_the_checked_in_sweep_expands_to_distinct_runs` (expects 4 sweep runs, gets 8), is in rotgrid code and config this session did not touch. Probably the checked-in rotgrid sweep config was extended without updating the test [assumed]. Flagging, not fixing.
C: this one as well
Answer (26-10-02, Claude Opus 5.5): Added to `FUTURE_WORK.md` as well.
- **Smoke test** [verified 26-10-02]: arms B and C ran 4 steps each on CPU with the tiny test model `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`. Configs were derived from the real arm configs with the model, `module_info`, step counts and frequencies overridden; PPGD from step 0; scratchpad output dir. Both exited 0 and saved `model_0/2/4.pth`.
- **Position selection by eye** [verified 26-10-02]: decoded a train batch per arm (`[x]` = trained position, `·` = pad):
  - B: `Fifty-three is larger than fifty-eight[.]···`
  - A: every real token bracketed, pads not.
  - C: no mask, all 24 positions trained.

**Found, not fixed:** the non-target eval reads the same Pile split with the same `dataset_seed` as training. So its eval batches (10 × 4 documents) are documents also seen in the first training steps [concluded: from `spd/data.py`, which uses `dataset_config.seed` for both]. `monology/pile-uncopyrighted` exposes only `train` to `datasets` (`val.jsonl.zst` and `test.jsonl.zst` exist as raw files), so a clean fix needs loader changes. The effect is likely negligible (40 documents, each trained on once at the very start) [assumed]. Noted in the config header.
C: I want to start the experiment soon, but I also want to have clean results. So what do we have to do to fix this?
Answer (26-10-02, Claude Opus 5.5): Fixed. First, the scope: this leak only affected a *monitoring* metric, the non-target reconstruction KL during training. The experiment's results (probes on the tiu test split) were never affected. Still, it's now clean:
- **New option:** `LMTaskConfig.eval_data_files` (passed to `datasets.load_dataset` as `data_files={eval_data_split: eval_data_files}`), plumbed through all 14 places that build an eval loader from `eval_data_split`.
- **The arm configs** now read the non-target eval from the Pile's validation file: `eval_data_split: validation`, `eval_data_files: val.jsonl.zst`.
- **Disjoint by construction** [verified 26-10-02]: the default `train` split resolves to exactly the 30 `train/*.jsonl.zst` files and no val/test file.
- **By eye** [verified 26-10-02]: a snippet of the first eval batch was found at document 850 of `val.jsonl.zst`. The first training batch's snippet was not among the first 3,000 validation documents. A 2-step tiny-model run with these settings completed, including evals.

**To launch** (one GPU per run; three runs):
`python spd/experiments/lm/lm_decomposition.py spd/experiments/lm/honesty_targeted_decomposition/config_truth_{all_tokens,last_token,padded}.yaml`

**Next:** the separate probe-analysis script (CI extraction from checkpoints; logistic-regression and mass-mean probes; baselines: step-0 CI, raw activations, statement log-prob; cross-domain and affirmative→negated generalisation). It can be written while the runs train.

C: true, but let's do this once I gave my go that the runs have been started. Actually I kind of expect that we will have to have some iterations to get the GPU usage high. So let's park the next-steps until we have a good training setup
Answer (26-10-02, Claude Opus 5.5): Parked [decided]: no analysis script until you say the runs are going. Next focus: getting the 7B training setup efficient (GPU utilisation), when you're ready.

---

26-10-02, Claude Opus 5.5 — *0-step baseline, clean non-target eval, rotgrid TODOs*

Context: same session; responds to Julian's C: comments on the previous entry (answered in place).

Changes, all uncommitted:
- **`spd/run_spd.py`:** the step-0 checkpoint save is reverted [decided: Julian]. `steps: 0` now saves the initial decomposition and returns, for the untrained-CI baseline. `model_0.pth` is bit-identical to a normal run's starting weights, and identical across arms [verified: tiny-model smoke tests]. One 0-step run serves all three arms.
- **`spd/data.py` (`DatasetConfig.data_files`), `spd/configs.py` (`LMTaskConfig.eval_data_files`), 14 eval-loader call sites:** the eval split can be read from specific repo files. The three arm configs now evaluate the non-target stream on the Pile's `val.jsonl.zst`, disjoint from the training files [verified: resolved data files; snippet lookup]. `spd/scripts/validation/common.py`'s `--split` override resets `eval_data_files`, so the two can't mix.
- **`FUTURE_WORK.md`:** two entries for the pre-existing rotgrid problems (6 basedpyright errors; the sweep-count test failure).

**Verification:**
- ruff clean.
- basedpyright: only the 6 known rotgrid errors.
- Tests: 510 passed, 15 skipped, plus the 1 known rotgrid failure.
- Tiny-model smoke runs: `steps: 0` (last-token arm), and 2 steps with evals (all-tokens arm), both exit 0.

**Status:** the training setup is ready for a first 7B launch. The analysis script is parked until Julian confirms the runs are going [decided]. Julian expects iterations on GPU utilisation first.

**Launch:**
- The three arms: `python spd/experiments/lm/lm_decomposition.py spd/experiments/lm/honesty_targeted_decomposition/config_truth_{all_tokens,last_token,padded}.yaml`
- The untrained baseline: the same with one arm config and `steps: 0`. There's no separate config file yet; say whether you want one.

---

26-10-02, Claude Opus 5.5 — *narrowed eval_data_files plumbing*

Context: same session. Julian asked in chat which files belonged to which change, then approved narrowing the diff.

**Reverted** the one-line `data_files=task_config.eval_data_files` addition in the 11 scripts unrelated to this experiment: `spd/scripts/alpha_sweep/alpha_sweep.py`, the 5 `attention_ablation_experiment/` scripts, `compare_models.py`, `detect_prev_token_heads.py`, `plot_attention_offset_profiles.py`, `plot_qk_c_datapoint.py`, `plot_wv_subspace_overlap.py`. Each diff was exactly that one line [verified: `git diff -U0`]; after removal, `git status` shows none of them modified. `eval_data_files` is now honoured in three places only: `spd/experiments/lm/lm_decomposition.py` (training-time eval), `spd/metrics/targeted_ci_heatmap.py`, and `spd/scripts/validation/common.py`.

**Julian's question: is there a mechanism that makes other experiments crash if an extra field is set?** Partly.
- **Unknown fields crash:** `BaseConfig` (`spd/base_config.py`) uses pydantic `extra="forbid"`, so an *unknown* key fails at config load. [verified: `LMTaskConfig(dataset_name='x', eval_data_filez=...)` → `extra_forbidden`]
- **Ignored known fields don't:** `eval_data_files` is a *declared* field, so a config setting it validates everywhere. A script that never reads it won't crash for that reason alone.
- **But for our configs, ignoring it does crash.** The data files are keyed by split name, and our configs say `eval_data_split: validation`. A script that drops `eval_data_files` asks the default layout for a `validation` split, which `monology/pile-uncopyrighted` doesn't have. [verified: `ValueError: Bad split: validation. Available splits: ['train']`] That holds for any dataset whose default layout lacks the named split, which is exactly when `eval_data_files` is needed.
- **The remaining gap:** a config that names a split the default layout *does* have (e.g. `eval_data_split: train` + `eval_data_files: ...`). There, an ignoring script would silently read the default files. A general guarantee would need one shared helper that builds eval `DatasetConfig`s from an `LMTaskConfig`, used by every script. Not worth it now [concluded: the gap needs an unnatural config].

**Checks after the revert:** ruff clean; basedpyright 0 errors on `spd/scripts`, `spd/data.py`, `spd/configs.py`, `spd/experiments/lm`, `spd/metrics`.

---

26-10-02, Claude Opus 5.5 — *launching on vast.ai*

Context: same session. Julian asked in chat for the command to run training on vast.ai with the H100 config. Julian also said in chat not to mark the `eval_data_files` crash verdict (previous entry) as decided yet; it stays `[concluded]`.

What I found, from reading `spd/scripts/run_vast.py` (the `spd-vast` CLI) [verified: code read 26-10-02, not run]:
- **Registry names only:** `spd-vast <experiment>` accepts only experiment names from `EXPERIMENT_REGISTRY` (`spd/registry.py`), not config paths. The three arm configs aren't registered, so `spd-vast` can't launch them directly.
- **Checkpoint upload:** for registered experiments, `_build_config` forces `sync_checkpoints_to_wandb: true`, because the instance's disk vanishes when the instance is destroyed. Our arm configs have `sync_checkpoints_to_wandb: false`. A manual run on the instance would keep checkpoints only on its disk (`SPD_OUT_DIR`).
- **H100 config:** `--config h100` → `spd/scripts/vast_h100_config.yaml` (H100_SXM, ≥79 GB, ≤ $8/h, 100 GB disk).
- **One ssh alias:** the runner writes a single ssh alias `vastai` (`~/.ssh/config.d/vastai.conf`). A second rented instance overwrites it.

Two ways to launch:
1. **Now, without code changes:** `spd-vast --mode provision --config h100`, then on the instance run `lm_decomposition.py` with an arm config. Checkpoints must be copied back before destroying the instance, or set `sync_checkpoints_to_wandb: true` in the arm configs first.
2. **Register the arms** in `spd/registry.py`, then `spd-vast <name> --config h100 --mode detached`. That gets the forced WandB checkpoint upload for free. It needs an `expected_runtime` (minutes) per entry, which we don't know yet.

---

26-10-02, Claude Opus 5.5 — *checkpoints synced to WandB*

Context: same session. Julian asked in chat to set `sync_checkpoints_to_wandb: true` in the arm configs, so a manual run on a vast.ai instance doesn't lose its checkpoints when the instance is destroyed.

- **Configs:** done in all three arm configs [verified: all parse with `sync_checkpoints_to_wandb=True`].
- **Gap this exposed in my 0-step baseline path** (`spd/run_spd.py`): it saved `model_0.pth` locally but never uploaded it. It now calls `wandb.save` under the same condition as the training loop's checkpoint saves (`wandb_project` set and `sync_checkpoints_to_wandb`).
- **Checks:** ruff and basedpyright are clean on the file. The upload itself is untested: I didn't create a WandB run from here.

---

26-10-02, Claude Opus 5.5 — *untrained baseline config*

Context: same session. Julian asked in chat for a dedicated 0-step baseline config, without a consistency test. Julian (chat, paraphrased): don't be strict that the baseline only counts if all values match the arms; its purpose is to check whether the CI functions leak truth information even without training.

- Created `spd/experiments/lm/honesty_targeted_decomposition/config_truth_untrained.yaml`. It's a copy of `config_truth_all_tokens.yaml` with only `steps: 0` and `label: "tiu_untrained"` changed [verified: `diff` below the header; the config parses].
- Its header states the purpose. It notes that the shared seed, model, `module_info` and `ci_config` make it exactly the arms' starting point, but that the purpose doesn't depend on that.
- Launch: `python spd/experiments/lm/lm_decomposition.py spd/experiments/lm/honesty_targeted_decomposition/config_truth_untrained.yaml`. It saves `model_0.pth` (uploaded to WandB) and stops. It needs the 7B model loaded, but no training.

---

26-10-02, Claude Opus 5.5 — *chat transcript since the entry "0-step baseline, clean non-target eval, rotgrid TODOs"*

Context: same session. The entries since *0-step baseline, clean non-target eval, rotgrid TODOs* were triggered by chat messages, which I only paraphrased there. Below are Julian's chat messages verbatim (I am the scribe), in order, each with a one-line note on my reply. Trivial messages ("Comments written") are omitted.

C: (transcribed from chat) You have touched quite a lot of files. Is that for saving step 0 or for the pile evaluation leak?

My reply, file by file:
- **0-step baseline:** only `spd/run_spd.py`. The step-0 save was reverted.
- **Pile eval leak:** `spd/data.py`, `spd/configs.py`, `lm_decomposition.py`, `targeted_ci_heatmap.py`, `spd/scripts/validation/common.py`, plus 11 unrelated `spd/scripts/*` files with a one-line addition each.
- **Position routing** (earlier round): `spd/models/components.py`, the six masked-forward sites, the test, the field description in `spd/configs.py`, `docs/PROJECT_REFERENCE.md`.
- **Arm configs:** 3 new yaml files.
- **Legacy cleanup:** `README.md` and the deleted files.
- **Memory files.**

I proposed reverting the 11 unrelated scripts.

C: (transcribed from chat) Yes, please. Also isn't there a mechanisms that would make the other experiments crash if an extra field is set?

Reverted. The answer is in the entry *narrowed eval_data_files plumbing*.

C: (transcribed from chat) Alright. I'll believe you in your verdict, but I haven't fully understood it so doen't mark it [decided] yet. Can you tell me the command for running the training script on vastai with h100 config?

The crash-mechanism verdict stays `[concluded]`. The answer is in the entry *launching on vast.ai*.

C: (transcribed from chat) Can you change the configs to  have sync_checkpoints_to_wandb: true

Done, see the entry *checkpoints synced to WandB*.

C: (transcribed from chat) Where does the 0-step baseline config live?

It didn't exist. I had only suggested running an arm config with `steps: 0`, and asked whether he wanted a dedicated file, with or without a consistency test.

C: (transcribed from chat) Yes, please create it without a test. Also don't be too strict in that it only counts if all values are the same. It's basically for checking that the CI-functions leak any truth information even without training.

Done, see the entry *untrained baseline config*.

C: (transcribed from chat) Have you written our conversation to the log so far?

Not fully; this entry closes the gap.

Now traceable [decided], via the transcriptions above:
- the 11 scripts are reverted;
- `sync_checkpoints_to_wandb: true` in the arm configs;
- the untrained-baseline config exists, has no consistency test, and is not required to match the arms exactly; its purpose is to detect truth leakage through untrained CI functions.

The crash-mechanism verdict remains `[concluded]`, explicitly not `[decided]`, per Julian.

---

26-10-02, Claude Opus 5.5 — *first 7B launch: instance check*

Context: same session. Julian has a vast.ai H100 instance running (`spd-vast` alias `vastai`, run id `s-eb527257`, instance 53868897) and wants to start a decomposition.

C: (transcribed from chat) I'm not runnig config_truth_untrained on a h100

Read-only checks on the instance [verified 26-10-02, 16:10]:
- **GPU:** idle H100 80GB HBM3; no training process running.
- **Code and configs:** the three arm configs, `spd/run_spd.py`, `spd/data.py` and `spd/configs.py` are md5-identical to the local working tree.
- **Missing:** `config_truth_untrained.yaml` (it was created after the sync). Not needed there [decided: transcribed above].
- **Credentials:** `WANDB_API_KEY` and `WANDB_ENTITY` are set in login shells (`/etc/profile.d/spd_env.sh`). No HF token, which is fine: Qwen2.5-7B-Instruct is not gated [assumed].

Launch command given to Julian (all-tokens arm first); see chat.

---

26-10-02, Claude Opus 5.5 — *topic closed*

C: (transcribed from chat) Ok, so this can be closed. The next discussion is about the training run adn I don't want to commit the code before the training run has been discussed. So close the old one and start a new discussion. First point of the new discussion is: The run is going and I want to know how the system utilization is?

This topic is closed [decided: transcribed above]. Continued in `convos/julian/26-10-02_training_run_LOG.md`. Open items, and where they went:
- **To the training-run topic:**
  - GPU utilisation and batch size.
  - Untuned hyperparameters (loss coefficients, PPGD `start_frac: 0.8`, learning rate, steps).
  - Running the untrained baseline (not on an H100).
  - Committing the code: deferred until the training run has been discussed [decided: transcribed above].
- **To a future analysis topic** (parked until training works):
  - the probe-analysis script;
  - which CI value to probe;
  - Julian's TODO of a no-truth-signal control dataset.
- **Remaining open here:** the `eval_data_files` crash-mechanism verdict stays `[concluded]`. Adopting the FEEDBACK.md suggestion is Julian's call.
