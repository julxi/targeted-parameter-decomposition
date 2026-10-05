# SUMMARY: Training runs (tPD on Qwen2.5-7B, Truth-is-Universal arms)

**Last updated:** 26-10-04 (sync: config state corrected (final checkpoint only, bf16); untrained baseline, probe script and commit marked done; open list made self-contained; legacy-reference and CODING.md outcomes added; decision traces added)

Continues `convos/julian/26-10-02_overview_of_goal_SUMMARY.md`, which designed the experiment:
- tPD (targeted parameter decomposition) on Qwen2.5-7B-Instruct, target = Truth-is-Universal (tiu) statements;
- three arms differing only in the trained target positions (all tokens / last token / padded);
- the decomposition's CI (causal-importance) values to be probed for truth later.

This topic is about getting the training runs done well. Its focus is **training efficacy**: lower loss in less wall time [decided: transcribed chat in the LOG, after the entry *correction: focus IS training, not the truth check*]. After run 1 I proposed a quick truth check of its CIs first; Julian put that aside in favour of training (an agent first misread his message as the opposite; corrected in the LOG). The truth check was later done in `convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md`. Julian relies on my judgement and prefers small experiments over planning everything ahead [decided: transcribed chat in the LOG, before *micro-benchmark: fp32 weights and batch size*].

## Verdict (26-10-03)

[decided: Julian ratified it and asked for it to be transferred to all three arms, noting that the last-token arm especially could use its own fine-tuning (C: comment in the LOG, *clean-up options*); evidence: single-seed runs on the all-tokens arm]

**Applied** to `spd/experiments/lm/honesty_targeted_decomposition/config_truth_{all_tokens,last_token,padded,untrained}.yaml`: steps 5,000 (untrained: 0 by design); target and Pile batch 16; eval 16 × 16 = 256 statements every 500 steps; slow eval every 2,500; LR 5e-4. The shared config header records that this was tuned on the all-tokens arm only and that the last-token arm may need its own tuning [verified 26-10-04: read all four configs; the three arms differ only in `label` and `loss_positions`]. Two later changes from other topics: the arms save the final checkpoint only (`save_freq: null`, from `convos/julian/26-10-03_wandb_storage_SUMMARY.md`; the untrained config kept `save_freq: 1000`, which has no effect at 0 steps), and all four load the frozen model in bf16 (`pretrained_model_dtype: bfloat16`). So the all-tokens config equals e2b's apart from label, WandB project, checkpoint frequency and the frozen-model dtype [verified 26-10-04: key-by-key diff against `~/spd_out/26-10-02_training_run/e2/e2b_b16_s5000.yaml`].

1. **Use batch 16 instead of batch 4.** Batch 4 is overhead-bound: a training step at batch 16 costs ~1.3× one at batch 4 for 4× the data.
2. **Recommended default: batch 16, 5,000 steps, LR 5e-4 (run e2b).** The only tested setting at least as good as the old default (batch 4, 10k steps, run e2c) on every metric, in about two thirds of the wall time.
3. **Fast option: batch 16, 2,500 steps (e2a)**, at 40% of the old default's time, with worse target reconstruction.
4. **LR 1e-3 is not adopted:** better target reconstruction, worse Pile side effect (see E3). A warmup might remove the cost (untested).

**Limits:** one seed per setting; only the all-tokens arm (the last-token arm has one target token per statement, so much less signal per step); wall times for e2b/e2c are projections (below); loss coefficients and the PPGD start (80% of steps) untuned.

| final value (KL in nats/token; L0 = active components per token, of 480) | e2c: b4, 10k (old default) | e2a: b16, 2.5k | **e2b: b16, 5k** | e3b: b16, 5k, LR 1e-3 |
|---|---|---|---|---|
| target L0 | 8.2 | 8.7 | 6.2 | 6.8 |
| target rounded KL | 0.142 | 0.184 | 0.144 | 0.116 |
| PGD recon KL | 0.666 | 0.646 | 0.357 | 0.339 |
| Pile rounded KL | 0.066 | 0.038 | 0.038 | 0.070 |
| wall time, uncapped H100 | ~41 min | 16.5 min | ~28 min | ~28 min |

[verified: `metrics.jsonl`. Wall times: e2a measured; the others assumed = steps × 26-10-02 per-step times + 5.3 min eval overhead, because they ran on a power-capped GPU.]

**Noise level** (e2c vs run 1, same settings, different eval size): up to 10% on target metrics, 15–25% on PGD/Pile KL. Differences smaller than that between single runs are not meaningful [concluded].

## How we got there

**Run 1** (`s-c2562082`, 26-10-02): all-tokens arm, batch 4, 10k steps, ~38 min.
- Sparse and target-selective [concluded]: 8.6 active components per target token vs 0.9 per Pile token; removing all components costs 4.45 nats/token on target vs 0.09 on Pile.
- Its eval (32 statements) was too small to compare runs; e2c repeats it with an 8× larger eval.

**Why batch 4 was slow** [verified: micro-benchmark 26-10-02, single run per setting]: at 24 tokens, a forward+backward pass of the frozen 7B model takes the same time at batch 16 as at batch 4, so batch 4 pays fixed per-pass costs (kernel launches, weight reads). The frozen model was also loaded in fp32 [verified: transformers 4.57.3 default], and autocast re-casts frozen weights to bf16 on every pass [assumed: PyTorch caching rule]; bf16 loading saves 10–20% per pass and halves the memory [verified: same benchmark].

**Throughput** [verified: 150-step trials on the 26-10-02 instance]:

| batch | ms/step without PPGD | ms/step with PPGD | samples/s vs batch 4 (no PPGD) |
|---|---|---|---|
| 4 | 183 | 330 | 1.0× |
| 16 | 235 | 409 | 3.0× |
| 32 | 348 | 548 | 4.1× |

PPGD (the persistent adversarial phase, last 20% of steps) adds 150–200 ms per step. Peak memory ≤ 60 GB in all cases (fp32 weights).

**E2: batch 16 vs batch 4.** e2a `s-fe0f421a` (b16, 2.5k), e2b `s-ba5f9dd5` (b16, 5k), e2c `s-b3401a03` (b4, 10k). Configs identical apart from batch size and steps; eval 256 held-out statements and 10,240 Pile tokens.
- e2b beats e2c on everything except target rounded KL (a tie) [concluded: margins beyond noise].
- PGD KL seems to follow the samples seen in the PPGD phase, not the number of updates [assumed: three runs].
- At batch 4 the PGD phase does converge (flat in e2c), to a worse value than batch 16 reaches. Run 1 only looked unconverged because of its noisy eval.

**E3: 2× learning rate at batch 16.** e3a `s-a77ae04e` (= e2a with LR 1e-3), e3b `s-83033a96` (= e2b with LR 1e-3).
- 2× LR trades Pile faithfulness for target faithfulness, at both step counts [concluded: beyond noise].
- The Pile damage arises early (e3b Pile KL 0.17–0.21 at steps 500–2000) and only partly recovers. The LR schedule has no warmup [assumed: plausible cause].

## Done since the verdict (in other topics)

- **bf16 loading of the frozen model** [decided: Julian, transcribed chat in `convos/julian/26-10-03_test_accuracy_analysis_LOG.md`]: new config field `pretrained_model_dtype`, set to `bfloat16` in all tiu configs. All runs of this topic (run 1 to e3b) loaded fp32. Measured in training since (26-10-05, one H100 instance, batch 16, 150-step trials): bf16 is 8–10% faster per step than fp32 and needs ~27 GB instead of 49–54 GB; that instance was ~37% slower than the 26-10-02 one in both dtypes, so wall times differ more between hosts than between dtypes (`convos/julian/26-10-05_no_truth_baseline_SUMMARY.md`).
- **Untrained baseline** generated on the laptop CPU (`s-286aa6a9`; Julian did not want it run on an H100); see the test-accuracy topic.
- **Probe script** `probe_ci.py` written and run on run 1, e2b, e2c, e3a, e3b and the untrained baseline (test-accuracy topic).
- **Commit:** Julian committed this topic's records and code himself (`2725132`, "hyperparameter tuning for truthfulness") [verified: `git log`, 26-10-04].

## Open follow-ups

- **E5: tune the last-token arm, plus a padded-arm sanity run** (proposed, awaiting Julian). The arms differ only in `loss_positions` (all-tokens ~10 trained positions per statement, last-token 1, padded 24 with >50% padding). Per-step cost, the bf16 speedup and the eval settings transfer [concluded: same model, batch, sequence length]. The batch/LR verdicts may not, because the last-token arm's loss averages over 16 positions per batch-16 step instead of ~160 [assumed]. Proposed tests (~1.5–2 h of an uncapped H100):
  - last-token arm: batch 16 / 5k vs batch 4 / 10k;
  - optionally last-token batch 32 / 2.5k;
  - padded arm: one batch-16 / 5k sanity run (healthy curves only; no baseline).
- **Run the arms at the final settings.** None has run with the current configs (bf16, final checkpoint only). e2b's final checkpoint matches the all-tokens arm's training settings and could serve as that arm's decomposition (WandB-storage topic).
- Optional: E4a (LR 1e-3 with LR warmup at batch 16 / 5k: keep the target gain without the Pile cost?); E4b (batch 32, 2,500 steps: further speedup).
- Untuned: loss coefficients, PPGD start at 80% of steps, the decomposed layers (15–19 `down_proj`), C = 96.
- No reference point yet for the target KL (e.g. zero-ablating all five `down_proj` layers), which would tell how good ~0.14 nats/token is [assumed].
- Rent a 700 W GPU: `spd-vast` does not filter on the power limit yet (FUTURE_WORK.md).

**Early warning for probing** [assumed: visual impression of run 1 at step 8000]: the strongest CIs sit at the final token (the period), with the same few components for every statement, so they might encode statement end rather than truth. The built-in heatmap shows only the first 16 test prompts, all from `animal_class_false`, so this could not be judged from it. The test-accuracy topic later found that the on/off pattern of the active components predicts truth (0.916–0.964 test accuracy), i.e. not a constant end-of-statement code; see there.

## Practicalities

- **Metric semantics** [verified: metric code]: all recon losses are KL in nats per evaluated token. On target data, `rounded`/`CImasked`/`stochastic` run with the delta component **off**; on Pile data the delta is **on**. `UnmaskedReconLoss` = all 480 components, no delta, on target data; it is not expected to be ~0.
- **Rented-GPU rules** (from this topic's avoidable restarts and lost time) are in `docs/CODING.md`, section *Running jobs on a rented vast.ai GPU*: check the power limit, launch from a login shell (`ssh vastai 'bash -lc "<cmd>"'`, because `spd-vast` puts the WandB key in `/etc/profile.d/spd_env.sh`) [verified 26-10-02], detach jobs, `pkill` with a bracket pattern, no `HF_HUB_OFFLINE`, compare wall times only on the same machine. Julian permits writing WandB keys to vast.ai machines [decided: transcribed chat in the LOG], but the login shell makes it unnecessary. The key is also stored a second time in vast.ai's instance metadata (FUTURE_WORK.md item, filed at Julian's request).
- **Rule:** a finished run with `sync_checkpoints_to_wandb: true` has its checkpoints on WandB; no need to re-verify before destroying an instance [decided: transcribed chat in the LOG].
- **GPU power cap:** the 26-10-03 instance's H100 was capped at 500 W and throttled to ~1 GHz; batch 16 ran 40% slower [verified: `nvidia-smi`]. Filter offers on `gpu_max_power` (FUTURE_WORK.md).
- **Files:** scripts, metrics, configs, run metadata and progress logs of all runs in `~/spd_out/26-10-02_training_run/` (local only, not in git; the only copy of e2a's metrics). Rule: no artifacts in `convos/` [decided: C: comment in the LOG, *clean-up options*; recorded in `docs/CODING.md`]. Figures were deleted; the WandB runs still have them; `scripts/compare_e2.py` prints the comparison of all runs. On WandB (`bitt-j-personal/spd/<run id>`) each run keeps only its final checkpoint (intermediate ones deleted, see the WandB-storage topic); e2a has none.
- **Legacy references frozen** [decided: transcribed chat in the LOG, before *legacy references frozen; rented-GPU rules moved to CODING.md*]: edits a 26-10-02 session had made to `docs/PROJECT_REFERENCE.md` were reverted and replaced by an `OUTDATED (26-10-03)` marker; CLAUDE.md and `docs/CODING.md` now say never to add new information to the legacy references.

## Open questions to Julian

1. E5: tune the last-token arm (batch 16 / 5k vs batch 4 / 10k, optionally batch 32 / 2.5k) and a padded-arm sanity run, ~1.5–2 h of an uncapped H100.

All C: comments in the LOG are answered.

- See also: [convos/julian/26-10-03_wandb_storage_SUMMARY.md] — WandB 5 GB cap: these runs' checkpoints fill it; arm configs now save the final checkpoint only, tuning runs don't sync checkpoints.
- See also: [convos/julian/26-10-02_overview_of_goal_SUMMARY.md] — experiment design, the code changes the runs depend on, and tPD background.
- See also: [convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md] — probe test accuracies of the finished decompositions; runs on the laptop CPU; bf16 loading of the frozen model was implemented there.
- See also: [convos/julian/26-10-02_prepared_datasets_SUMMARY.md] — the tiu v1 datasets these runs train on; the OUTDATED marker placed in `docs/PROJECT_REFERENCE.md` by this topic points there.
- See also: [convos/julian/26-10-02_epistemic_memory_setup_SUMMARY.md] — set up the legacy-reference rules (`docs/PROJECT_REFERENCE.md`) that this topic changed on 26-10-03 (no new information, mark wrong passages `OUTDATED`).
- See also: [convos/julian/26-10-05_no_truth_baseline_SUMMARY.md] — first bf16 training runs at the final arm-A settings (arm A `s-bd23f0d1` matches e2b within noise) and the fp32-vs-bf16 throughput trial.
