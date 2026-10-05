# SUMMARY: Test accuracies of the decompositions (probe analysis)

**Last updated:** 26-10-04 (sync: added the answered top-5 vs sparsity-matched question; removed stale "untrained baseline not generated" caveat; corrected the CPU-autocast KL error size; softened unrecorded claims)

Julian wants the test accuracies of the decompositions run so far: accuracy of truth probes on CI (causal-importance) values on the held-out tiu test split, as designed in `convos/julian/26-10-02_overview_of_goal_SUMMARY.md` (*Analysis*) [decided: Julian, C: comment in the LOG].

## Feasibility on the laptop (CPU only)

**Yes** [concluded, from a benchmark]:
- CIs are computed from the frozen model's own `down_proj` inputs [verified: `spd/run_spd.py`], so one forward pass through Qwen layers 0–19 serves all checkpoints; the CI functions themselves are under 1% of the cost.
- Benchmark [verified: 26-10-03, real Qwen2.5-7B-Instruct weights truncated to 20 layers, batch 16 × 24 tokens, AMD Ryzen AI 9 365]: bf16 2.7–2.8 s per batch at 10.0 GB peak RAM; fp32 7.3–7.8 s at 24.3 GB.
- Extrapolated: test split (1,794 statements) ≈ 5 min, train + test ≈ 25 min, in bf16. bf16 autocast matches how the runs were trained (`autocast_bf16: true`).
- The log-probability baseline needs the full model, ≈ 1.4× the cost [assumed].

## Caveats

- Only the all-tokens arm has checkpoints: final step of run 1, e2b, e2c, e3a, e3b (238 MB each on WandB). Intermediate checkpoints are deleted, so no accuracy-over-training curve for these runs. e2a has no checkpoint.
- The untrained baseline was generated on the laptop CPU (see *Done*); it is a fresh draw from the initialisation distribution, not the GPU runs' exact starting weights.

## Setup [decided: Julian, C: comment in the LOG]

- Per run, two readouts of the 480 CI values: at the last real token, and the mean over real tokens. Each probed with logistic regression and mass-mean: 4 test accuracies per run.
- Probes fit on the train split (6,560 statements, 50% true), scored on the test split (1,794, 50% true).
- Baselines: the untrained decomposition's CIs; the residual stream after layer 19 [assumed: layer choice]; mean statement log-probability.
- CI variant probed: `lower_leaky` [assumed]. Logistic regression uses sklearn's default L2 strength, untuned.

## Done

- **Untrained baseline** `s-286aa6a9` (`model_0.pth`) generated on the laptop CPU in 2 min, uploaded to WandB [verified]. Not the GPU runs' exact starting weights; Julian said that doesn't matter [decided].
- **Probe script** `spd/experiments/lm/honesty_targeted_decomposition/probe_ci.py <out_dir> <run_id>...` (in the repo, reused for later arms). One forward pass of the frozen model serves all runs. Outputs in `~/spd_out/26-10-03_test_accuracy_analysis/` (`features.npz`, `results.json`, `examples.txt`, `probe_run.log`). `examples.txt` is written for a tokenisation check (the LOG does not record its inspection result).
- **Probe run done** on untrained, run 1, e2b, e2c, e3a, e3b (fp32 weights, as they were trained), exit 0 [verified: `probe_run.log`]: about 55 min on the laptop (from output file timestamps, `examples.txt` 20:30 to `features.npz` 21:24; slower than the 25 min estimate because it used all 28 layers plus the LM head and fp32 weights). Follow-up script `followup.py` in the same folder.

## Frozen model in bf16 [decided: Julian, transcribed chat in the LOG]

- New config field `pretrained_model_dtype` (`float32` default, so old configs load as they trained). The tiu arm configs and the untrained config now set `bfloat16`. Components and CI functions stay fp32.
- Verified [26-10-03]: tiny-Qwen2 CPU smoke test, bf16 and fp32 losses agree to 2–3 significant digits; test suite passes (except the known rotgrid failure).
- Consequence: new bf16 runs and the old fp32 runs must be probed in separate invocations (the script asserts a shared dtype).
- Side finding: on **CPU**, autocast keeps softmax of bf16 logits in bf16, so the KL of near-identical distributions can be wrong by up to ~10⁻³–10⁻² nats; the smoke test's KL recon losses came out negative (about −0.019) in both dtypes [concluded: direct test]. KL metrics from CPU runs are unreliable at this scale. H100 runs unaffected [assumed: CUDA autocast runs softmax in fp32]. `probe_ci.py` casts logits to fp32 before `log_softmax`.

## Results [verified: `results.json`, `followup.py`, 26-10-03; single seed per run, all arm A]

Test accuracy, last-token readout, logistic regression (chance 0.50; standard error ≈ 0.004):
- residual stream after L19: **0.998**; untrained CIs: **0.998**; trained CIs: **0.970–0.984** (run 1 0.972, e2b 0.970, e2c 0.979, e3a 0.972, e3b 0.984); mean log-probability: 0.574.
- Mean readout: trained CIs 0.979–0.996, untrained 0.992. Mass-mean probes are lower throughout (full table in the LOG, *probe results, six checkpoints*).

Conclusions [concluded]:
1. CI probes detecting truth is **expected without any training**: untrained CIs match the residual-stream ceiling. The motivating finding (someone else's unreplicated result: probes on CI values detect statement truth, with no baseline) is fully explained by this null hypothesis.
2. Training makes the CIs sparse (8.5–11.4 of 480 active per statement vs ~220 untrained) and costs 1.4–2.8 points of accuracy.
3. The on/off pattern of the ~10 active components alone predicts truth at 0.916–0.964. Not domain identity (domains are balanced), and not a constant end-of-statement code.
4. No evidence yet that trained components are more truth-specific than random ones: the untrained top-5 components (0.938) fall within the trained range (0.877–0.953). Not a like-for-like comparison; a sparsity-matched baseline is needed.
5. The hyperparameter settings can't be ranked from these numbers (1.4-point spread, one seed each).

## Top-5 components vs the proposed sparsity-matched baseline

Julian asked (C: comment in the LOG, answered there) how the "top-5" result differs from the proposed sparsity-matched baseline:
- **Top-5 (done):** the same 5 components for every statement, ranked on the train split by standardised class-mean difference, then a logistic regression on those 5 columns. Asks how much truth the most truth-related components carry.
- **Sparsity-matched (proposed):** per statement, keep the untrained network's ~10 largest CI values, zero the rest, probe all 480 columns; which components stay on varies per statement and is chosen by size, not by label. This is the fair comparison for the trained runs' on/off-pattern (0.916–0.964) and all-480 (0.970–0.984) accuracies: much lower → tPD picked its active components in a truth-relevant way; about equal → sparsity alone explains it [concluded].

## Open

- All C: comments in the LOG are answered.
- Proposed next: generalisation tests (affirmative → negated, leave-one-domain-out) on `features.npz`; a sparsity-matched untrained baseline; probing the last-token and padded arms once trained (bf16).
- Faster re-runs for later arms: drop the log-prob baseline and the layers above 19 (proposed in chat, not recorded in the LOG, not done) [assumed].

- See also: [convos/julian/26-10-02_overview_of_goal_SUMMARY.md] — the analysis design (readouts, probes, baselines) this topic implements.
- See also: [convos/julian/26-10-02_training_run_SUMMARY.md] — the runs whose checkpoints are analysed.
- See also: [convos/julian/26-10-03_wandb_storage_SUMMARY.md] — which checkpoints survived the cleanup.
