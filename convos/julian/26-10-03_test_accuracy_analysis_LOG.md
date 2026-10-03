I would like to check the test accuracies of the decompositins that have been run so far (that's why I didn't want the hyperparameter experiment checkpoints to be deleted yet). Do you think this analysis could be done on my laptop (CPU only, but 55GB ram free)

---

26-10-03, Claude Opus 5.5 — *laptop feasibility check*

I have read INDEX.md and the SUMMARYs of the overview, training-run and WandB-storage topics, the CI code path in `spd/run_spd.py`, and ran a CPU benchmark on this laptop. No analysis code written yet.

**What I take "test accuracies" to mean** [assumed; please correct]: accuracy of truth probes (logistic regression, mass-mean) on the CI (causal-importance) values of each finished decomposition, on the held-out tiu test split (1,794 statements), as designed in `convos/julian/26-10-02_overview_of_goal_SUMMARY.md` (*Analysis*). The recon metrics on the test split already exist in each run's `metrics.jsonl`.
C: yes, that's what I meant

**Answer: yes, the laptop is enough** [concluded, from the benchmark below]. Why:

1. The CI functions read only the frozen model's own activations. `run_spd.py` computes CIs from `wrapped_model(batch, cache_type="input")`, the inputs to the decomposed `down_proj` layers in an unmasked forward pass [verified: code read]. So the analysis needs **one forward pass** of Qwen through layers 0–19 (layers 20–27 and the LM head can be dropped). All checkpoints' CI functions then run on the same cached inputs. The CI function costs ~49M multiply-adds per token (its 94,720 × 512 first layer), under 1% of the 20-layer forward.
2. Benchmark [verified: `bench_cpu.py` in this session's scratchpad, 26-10-03; real Qwen2.5-7B-Instruct weights, truncated to 20 layers, batch 16 × 24 tokens = `max_seq_len`, random token ids; 3 repeats after load]:

   | dtype | s per batch of 16 statements | peak RAM |
   |---|---|---|
   | bf16 | 2.71–2.84 (first call 12.2, warm-up) | 10.0 GB |
   | fp32 | 7.28–7.83 | 24.3 GB |

   CPU: AMD Ryzen AI 9 365 (10 cores, 20 threads, has `avx512_bf16`); torch uses 10 threads. 51 GB RAM available at the time.
3. Extrapolated (≈ 2.8 s per batch in bf16): test split only, 113 batches ≈ 5 min; train + test (8,354 statements, if probes should be fit on the train split) 523 batches ≈ 25 min. fp32: ×2.7. That is for all checkpoints at once.
4. bf16 is the right setting: the runs trained with `autocast_bf16: true` (fp32 weights, bf16 matmuls), so `torch.autocast("cpu", torch.bfloat16)` reproduces training conditions more closely than pure fp32 [concluded].
5. Baselines from the design: raw-activation probes come from the same pass for free. Statement log-probability needs all 28 layers and the 152k-vocab LM head, ≈ 1.4× the cost [assumed: from parameter count], still well under an hour.

**Things to be aware of:**

- **What exists to analyse.** All five runs with checkpoints are the **all-tokens arm** (arm A), differing in batch, steps and LR: run 1 `s-c2562082`, e2b `s-ba5f9dd5`, e2c `s-b3401a03`, e3a `s-a77ae04e`, e3b `s-83033a96`. Each keeps only its **final** checkpoint on WandB (238 MB each, ~1.2 GB to download); e2a's checkpoints were never kept. So this compares hyperparameter settings within one arm at their end point. Probe accuracy *over training* (intermediate steps) is no longer possible for these runs: the intermediates were deleted under option B of the WandB-storage topic. The last-token and padded arms have not been run.
- **The untrained baseline** (`config_truth_untrained.yaml`, step-0 checkpoint) has not been run. It is the most important baseline (do untrained CI functions already carry truth?). It might be generated on the laptop too, but I have not checked whether the initialisation needs anything beyond a CPU, nor whether CPU and GPU give the same initial weights for the same seed (they may not; a different draw from the same distribution would still serve as a baseline, but would not be the exact starting point of the trained runs) [assumed: unchecked].
- **The end-of-statement concern** from the training-run topic applies: if the strongest last-token CIs mark "statement end", the last-token readout may carry little truth. The raw-activation and untrained-CI baselines are what make any accuracy number interpretable.

**Proposed next step:** write the analysis script (one 20-layer bf16 pass, CI readouts at the last token and mean over real tokens, probes, the three baselines) and run it locally on the five checkpoints. Open choices for Julian: probes fit on train split / evaluated on test split, or cross-validated on test only? Generate the untrained baseline locally first?

C: I need some clarifications. You're saying per checkpoint two probes? One that reads CI of last token and one that reads mean over CI? If yes, perfects.
I'd say fit on train and evalute on test
And yes let's generate the untrainde baseline first. It doesn't matter that the weights are the same.

Answer (26-10-03, Claude Opus 5.5): Yes, two readouts per checkpoint: (1) the 480 CI values at the last real token; (2) the 480 CI values averaged over the statement's real tokens (padding excluded). Each readout gets two probe types, logistic regression and mass-mean (difference of class means as the direction, threshold fit on train). So four test accuracies per checkpoint. The same readouts on the untrained checkpoint and on the raw activations give the baseline rows; the log-probability baseline is one number per statement, no readout choice. Fit on train (6,560 statements), evaluate on test (1,794): noted. Generating the untrained baseline locally next.

---

26-10-03, Claude Opus 5.5 — *untrained baseline and probe script*

**Untrained baseline generated locally** [verified: run log `~/spd_out/26-10-03_test_accuracy_analysis/untrained_run.log`]: `python spd/experiments/lm/lm_decomposition.py spd/experiments/lm/honesty_targeted_decomposition/config_truth_untrained.yaml` on the laptop CPU, 1 min 57 s wall time. Run id `s-286aa6a9`; `model_0.pth` 238 MB, locally in `~/spd_out/spd/s-286aa6a9/` and uploaded to WandB (`bitt-j-personal/spd/s-286aa6a9`; adds 0.24 GB to the storage count). As agreed, it need not equal the GPU runs' starting weights.

**Probe script:** `spd/experiments/lm/honesty_targeted_decomposition/probe_ci.py` (in the repo rather than `~/spd_out/`, because it will be reused for the last-token and padded arms). `python probe_ci.py <out_dir> <run_id> ...`. What it does:
- Loads each run's config and latest checkpoint from WandB; asserts that all runs share model, `module_info`, `ci_config`, sigmoid type, datasets and `max_seq_len`.
- Reads train/test statements with their labels from the tiu jsonl files (hash-checked via the existing loader; asserts the label matches the `_true`/`_false` directory name). Tokenises exactly as the training loader does: no BOS, right-padded to 24, no attention mask.
- Loads the full Qwen once (fp32 weights, bf16 autocast, as in training) and runs one forward pass per batch. All six CI functions read the cached `down_proj` inputs.
- Features per statement, each with a last-real-token and a mean-over-real-tokens readout:
  - each run's 480 CI values, the `lower_leaky` variant (the one the recon losses mask with) [assumed: choice of variant; `upper_leaky` and pre-sigmoid values were not probed];
  - baseline: residual stream after layer 19, the last decomposed layer [assumed: layer choice];
  - baseline: mean next-token log-probability of the statement (one scalar).
- Probes: logistic regression (standardised features, sklearn default L2 strength C=1, untuned) and mass-mean (direction = difference of the train class means, threshold = midpoint of the projected class means). Fit on train, scored on test; per-dataset test accuracy is also recorded.
- Outputs in `~/spd_out/26-10-03_test_accuracy_analysis/`: `features.npz`, `results.json`, `examples.txt` (tokenisation check), `probe_run.log`.

Runs probed: untrained `s-286aa6a9`, run 1 `s-c2562082`, e2b `s-ba5f9dd5`, e2c `s-b3401a03`, e3a `s-a77ae04e`, e3b `s-83033a96`. Running now.

---

26-10-03, Claude Opus 5.5 — *frozen model switched to bf16*

C: (transcribed from chat) It's 1hour for 1 probe?

My chat answer: no, one hour for everything. A single forward pass over all 8,354 statements yields the features of all six checkpoints and both baselines; fitting the probes takes seconds to minutes. It is slower than my ~25 min estimate because the run uses all 28 layers plus the LM head (needed for the log-probability baseline) and fp32 weights under bf16 autocast, at ~7 s per batch of 16 instead of the benchmark's 2.8 s [verified: a 320-statement interval took over 115 s].

C: (transcribed from chat) Haven't we changed the target-model to fp16?

My chat answer: no. Loading the frozen model in bf16 (not fp16) was proposed in `convos/julian/26-10-02_training_run_LOG.md` and still open; the code passed no dtype to `from_pretrained`, so transformers 4.57.3 loaded fp32 [verified: grep of the loaders]. All five trained runs and the current probe run use fp32 weights with bf16 autocast. Autocast caches its bf16 copy only for weights that require grad, so the frozen weights are re-cast on every pass.

C: (transcribed from chat) I suppose I want to switch to bf16 in general. I don't think the extra precision is very important here anyway and there is no realy advantage with the back and forth casting.

**Implemented** [decided: Julian, transcribed above; implementation verified as below]:
- `spd/configs.py`: new field `pretrained_model_dtype: Literal["float32", "bfloat16"]`, default `float32`, so configs written before the field (all older runs, including the five trained tiu runs and `s-286aa6a9`) load as they trained. The validator requires `autocast_bf16` and a `transformers.*` model class for `bfloat16`.
- `spd/experiments/lm/lm_decomposition.py` and `ComponentModel.from_run_info` (`spd/models/component_model.py`) pass `dtype=` to `from_pretrained`. Other loaders (spd.pretrain models, `LlamaSimpleMLP` scripts) are untouched.
- `config_truth_{all_tokens,last_token,padded,untrained}.yaml`: `pretrained_model_dtype: bfloat16`. Components, CI functions and optimiser state stay fp32 (`torch.empty` defaults); the weight delta `W − UV` promotes to fp32.
- `probe_ci.py` asserts that all probed runs share `pretrained_model_dtype`, since one forward pass serves them all. A probe of new bf16 runs together with the old fp32 runs therefore needs two invocations.

**Verification:**
- Smoke test [verified: 26-10-03, CPU]: the all-tokens config with the tiny test model `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` (2 layers, `down_proj` C=8), 4 steps, PPGD from step 0, evals every 2 steps, no WandB, run once with `bfloat16` and once with `float32`. Both exit 0; every train and eval loss agrees to 2–3 significant digits between the two (e.g. eval `StochasticReconSubsetLoss` −0.01869 vs −0.01869, `ImportanceMinimalityLoss` 1.4628 vs 1.4635). Configs and logs: this session's scratchpad (`smoke_bf16.yaml`, `smoke_fp32.yaml`), not kept.
- Tests: 510 passed, 15 skipped; the known unrelated failure `tests/test_rotgrid_decomposition.py::test_the_checked_in_sweep_expands_to_distinct_runs` deselected (FUTURE_WORK.md). ruff and basedpyright clean on the changed files.
- Not verified: speed and memory on the H100. The 26-10-02 benchmark measured 10–20% faster per pass and half the memory for bf16 weights [verified there, `convos/julian/26-10-02_training_run_SUMMARY.md`].

**Side finding (pre-existing, unrelated to the dtype):** the KL-based recon losses in the smoke test are *negative* (about −0.019 nats) in both dtypes. Cause [concluded from a direct test]: under **CPU** autocast, `softmax`/`log_softmax` on bf16 logits stay in bf16, whereas CUDA autocast runs them in fp32; with bf16 probabilities the KL of near-identical distributions is wrong by up to ~10⁻³–10⁻² (test: KL of two random 151,936-vocab logit sets 5.0e-5 in fp32, 5.8e-4 with bf16 logits under CPU autocast). The H100 runs are unaffected [assumed: CUDA autocast op list]; KL metrics from CPU runs are unreliable at this scale. `probe_ci.py` casts logits to fp32 before `log_softmax`.

The running probe job (started before this change) loaded fp32 weights, matching how the five trained runs and the untrained baseline were produced. It continues unchanged.

---

26-10-03, Claude Opus 5.5 — *probe results, six checkpoints*

Run finished, exit 0 [verified: `~/spd_out/26-10-03_test_accuracy_analysis/probe_run.log`, `results.json`]. Julian hibernated the laptop afterwards; outputs were already complete and `features.npz` loads. All numbers below are re-read from `results.json` and the output of `followup.py` (same folder).

**Test accuracy** (1,794 held-out statements, 50% true, so chance = 0.50; standard error ≈ 0.004 at 0.97). All trained runs are the all-tokens arm (A); checkpoints are each run's final step.

| features | last/logreg | last/mass-mean | mean/logreg | mean/mass-mean |
|---|---|---|---|---|
| residual stream after L19 (baseline) | 0.998 | 0.981 | 0.996 | 0.522 |
| CI, untrained `s-286aa6a9` (baseline) | 0.998 | 0.980 | 0.992 | 0.824 |
| CI, run 1 `s-c2562082` (b4, 10k) | 0.972 | 0.911 | 0.992 | 0.892 |
| CI, e2b `s-ba5f9dd5` (b16, 5k; current default) | 0.970 | 0.821 | 0.996 | 0.805 |
| CI, e2c `s-b3401a03` (b4, 10k) | 0.979 | 0.802 | 0.979 | 0.827 |
| CI, e3a `s-a77ae04e` (b16, 2.5k, LR 1e-3) | 0.972 | 0.893 | 0.992 | 0.896 |
| CI, e3b `s-83033a96` (b16, 5k, LR 1e-3) | 0.984 | 0.924 | 0.991 | 0.896 |
| mean log-probability (baseline) | 0.574 (logreg on one scalar) | | | |

**Follow-up on the last-token CI readout** (`followup.py`; "active" = CI > 0.01, the configs' threshold):

| run | active components per statement (of 480) | logreg on on/off pattern only | best single component | logreg, top-5 components* | top-20* | all 480 |
|---|---|---|---|---|---|---|
| untrained | 219.9 | 0.997 | 0.829 | 0.938 | 0.995 | 0.998 |
| run 1 | 11.4 | 0.943 | 0.818 | 0.919 | 0.945 | 0.972 |
| e2b | 9.4 | 0.932 | 0.784 | 0.930 | 0.959 | 0.970 |
| e2c | 11.2 | 0.916 | 0.834 | 0.877 | 0.927 | 0.979 |
| e3a | 8.5 | 0.952 | 0.863 | 0.901 | 0.906 | 0.972 |
| e3b | 9.4 | 0.964 | 0.751 | 0.953 | 0.972 | 0.984 |

\* components ranked on the train split by the standardised difference of class means.

**Interpretation:**

1. **The reported finding is explained by the null hypothesis** [concluded]. Truth is linearly decodable at ceiling from the residual stream (0.998), and the CIs of a never-trained CI network do exactly as well (0.998). So "a probe on CI values detects truth" is no evidence of truth mechanisms: any dense nonlinear readout of these mid-layer activations does it. The motivating result ran no baseline; this one says such a result is expected regardless of training.
2. **Training makes the CIs sparse and costs a little accuracy** [verified: numbers above]. About 8.5–11.4 of 480 components are active per statement at the last token (98% of CI values exactly 0), against ~220 untrained. Last-token logreg accuracy falls from 0.998 to 0.970–0.984, i.e. 3.5–7 standard errors below the untrained baseline.
3. **The sparse code still carries most of the truth signal** [verified]: the on/off pattern of the ~10 active components alone gives 0.916–0.964. Domain identity cannot explain this, since every domain is balanced true/false. This also answers the end-of-statement worry from the training-run topic: the last-token components are not the same for every statement regardless of truth.
4. **No evidence yet that the trained components are more truth-specific than random readouts** [concluded, weak]. Picked the same way, the top-5 components of the untrained network give 0.938, within the trained runs' range (0.877–0.953). The single best component is also similar (0.829 untrained vs 0.751–0.863). A per-component comparison between a dense random code and a sparse trained one is not like for like, though; a sparsity-matched baseline is missing.
5. **Hyperparameter settings cannot be ranked from this** [concluded]. Trained runs span 0.970–0.984 (last/logreg) with one seed each; e3b is highest on most columns, but the spread is 1.4 points at ≈0.4-point standard error on a shared test set, with no seed replicates.
6. **Readout and probe details:** logreg ≥ mass-mean everywhere. Mass-mean on the mean-pooled residual stream fails (0.522): the class-mean difference is dominated by high-variance directions unrelated to truth there [assumed]. The log-probability baseline is weak (0.574), and per dataset it inverts on negated statements (e.g. `neg_sp_en_trans_false` 0.031), because one threshold cannot fit affirmative and negated statements.
7. **Per-dataset** (last/logreg; `results.json`): the residual and untrained-CI probes are ≥0.990 on every dataset. Trained-CI errors concentrate in a few datasets per run, e.g. e3a `element_symb_true` 0.607, e2b `smaller_than_false` 0.770, e3b `neg_sp_en_trans_false` 0.844. Test sets per dataset are small (tens to ~200 statements).

**What would be informative next** (proposals, not done):
- **Generalisation tests from the design** (train on affirmative, test on negated; leave-one-domain-out). Ceiling effects should shrink there, so trained vs untrained may separate. Runs on `features.npz` in minutes, no forward pass.
- **A sparsity-matched baseline**, e.g. the untrained CIs keeping only their top ~10 per statement, to test point 4 properly.
C: I have a question. What does top-5 above mean? How is it different from what you are proposing?

Answer (26-10-03, Claude Opus 5.5): They select components in two different ways.

- **Top-5 (done): the same 5 components for every statement.** On the train split I rank all 480 components by how differently they respond to true and false statements (difference of the class means divided by the component's standard deviation), keep the 5 best, and fit a logistic regression on just those 5 columns. The probe sees 5 numbers per statement, and they are always the same 5 components. It asks: how much truth do the few most truth-related components carry?
- **Sparsity-matched baseline (proposed): a different ~10 components for each statement.** For each statement I would keep the untrained network's ~10 *largest* CI values, set the other ~470 to zero, and probe all 480 columns. Which 10 stay on varies from statement to statement, chosen by activation size, not by any truth label. That gives the untrained code the same shape as a trained one (each statement lights up ~10 of 480).

The second is the fair comparison for the trained runs' "on/off pattern" (0.916–0.964) and "all 480" (0.970–0.984) columns. It asks whether a *random* readout, forced to be as sparse as tPD's, keeps as much truth information. If it scores much lower, the trained network has picked out its ~10 active components in a truth-relevant way. If it scores about the same, sparsity alone explains it. The top-5 comparison can't answer that: in the untrained network, the 5 best components are 5 of ~220 dense, always-varying features, a different kind of object from the trained network's mostly-off components.
- The other arms (last-token, padded), in bf16, probed with the same script.
