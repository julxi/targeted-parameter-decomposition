# SUMMARY: Training runs (tPD on Qwen2.5-7B, Truth-is-Universal arms)

**Last updated:** 26-10-03 (stopped after E3; verdict ratified and applied to all arm configs; artifacts moved to `~/spd_out/`; no instance running)

Continues `convos/julian/26-10-02_overview_of_goal_SUMMARY.md`, which designed the experiment:
- tPD (targeted parameter decomposition) on Qwen2.5-7B-Instruct, target = Truth-is-Universal statements;
- three arms differing only in the trained target positions (all tokens / last token / padded);
- the decomposition's CI (causal-importance) values to be probed for truth later.

This topic is about getting the training runs done well. Its focus became **training efficacy** [decided]: lower loss in less wall time. Julian relies on my judgement and prefers small experiments over planning everything ahead [decided].

## Verdict (26-10-03)

[decided: Julian ratified it and asked for it to be transferred to all three arms, noting that the last-token arm especially could use its own fine-tuning; evidence: single-seed runs on the all-tokens arm]

**Applied** to `config_truth_{all_tokens,last_token,padded,untrained}.yaml`: steps 5,000; target and Pile batch 16; eval 16 × 16 = 256 statements every 500 steps; slow eval every 2,500; save every 1,000; LR 5e-4. The shared config header records that this was tuned on the all-tokens arm only [verified: configs load; the all-tokens config equals e2b's apart from label and WandB project].

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

**Noise level** (e2c vs run 1, same settings, different eval size): up to 10% on target metrics, 15–25% on PGD/Pile KL. Differences smaller than that between single runs are not meaningful.

## How we got there

**Run 1** (`s-c2562082`, 26-10-02): all-tokens arm, batch 4, 10k steps, ~38 min.
- Sparse and target-selective [concluded]: 8.6 active components per target token vs 0.9 per Pile token; removing all components costs 4.45 nats/token on target vs 0.09 on Pile.
- Its eval (32 statements) was too small to compare runs; e2c repeats it with an 8× larger eval.

**Throughput** [verified: 150-step trials on the 26-10-02 instance]:

| batch | ms/step without PPGD | ms/step with PPGD | samples/s vs batch 4 (no PPGD) |
|---|---|---|---|
| 4 | 183 | 330 | 1.0× |
| 16 | 235 | 409 | 3.0× |
| 32 | 348 | 548 | 4.1× |

PPGD (the persistent adversarial phase, last 20% of steps) adds 150–200 ms per step. Peak memory ≤ 60 GB in all cases.

**E2: batch 16 vs batch 4.** e2a `s-fe0f421a` (b16, 2.5k), e2b `s-ba5f9dd5` (b16, 5k), e2c `s-b3401a03` (b4, 10k). Configs identical apart from batch size and steps; eval 256 held-out statements and 10,240 Pile tokens.
- e2b beats e2c on everything except target rounded KL (a tie) [concluded: margins beyond noise].
- PGD KL seems to follow the samples seen in the PPGD phase, not the number of updates [assumed: three runs].
- At batch 4 the PGD phase does converge (flat in e2c), to a worse value than batch 16 reaches. Run 1 only looked unconverged because of its noisy eval.

**E3: 2× learning rate at batch 16.** e3a `s-a77ae04e` (= e2a with LR 1e-3), e3b `s-83033a96` (= e2b with LR 1e-3).
- 2× LR trades Pile faithfulness for target faithfulness, at both step counts [concluded: beyond noise].
- The Pile damage arises early (e3b Pile KL 0.17–0.21 at steps 500–2000) and only partly recovers. The LR schedule has no warmup [assumed: plausible cause].

## Not done (open follow-ups)

- E4a: LR 1e-3 with LR warmup at batch 16 / 5k: keep the target gain without the Pile cost?
- E4b: batch 32, 2,500 steps: further speedup.
- **bf16 loading of the frozen model** (code change, not made). The model is loaded in fp32 (transformers 4.57.3 default; checkpoint is bf16). Benchmark: bf16 saves 10–20% per pass and halves the memory [verified: benchmark 26-10-02]. Components, deltas and optimizer would stay fp32.
- **Transfer to the other arms (E5, proposed):** the arms differ only in `loss_positions` (all-tokens ~10 trained positions per statement, last-token 1, padded 24 with >50% padding). Per-step cost, the bf16 speedup and the eval settings transfer [concluded: same model, batch, sequence length]. The batch/LR verdicts may not, because the last-token arm's loss averages over 16 positions per batch-16 step instead of ~160 [assumed]. Proposed tests (~1.5–2 h of an uncapped H100):
  - last-token arm: batch 16 / 5k vs batch 4 / 10k;
  - optionally last-token batch 32 / 2.5k;
  - padded arm: one batch-16 / 5k sanity run.
- The untrained baseline (`config_truth_untrained.yaml`) still has to run, on a cheaper machine than an H100 [decided].
- No reference point yet for the target KL (e.g. zero-ablating all five `down_proj` layers) [assumed].

**Early warning for probing** [assumed: visual impression of run 1 at step 8000]: the strongest CIs sit at the final token (the period), with the same few components for every statement. They may encode statement end rather than truth. The built-in heatmap shows only the first 16 test prompts, all from `animal_class_false`, so this cannot be judged from it.

## Practicalities

- **Metric semantics** [verified: metric code]: all recon losses are KL in nats per evaluated token. On target data, `rounded`/`CImasked`/`stochastic` run with the delta component **off**; on Pile data the delta is **on**. `UnmaskedReconLoss` = all 480 components, no delta, on target data; it is not expected to be ~0.
- **Launching on vast.ai:** `spd-vast` installs the WandB key into `/etc/profile.d/spd_env.sh`; only login shells read it, so launch with `ssh vastai 'bash -lc "<cmd>"'` [verified 26-10-02]. Julian permits writing WandB keys to vast.ai machines [decided], but this makes it unnecessary.
- **Rule:** a finished run with `sync_checkpoints_to_wandb: true` has its checkpoints on WandB; no need to re-verify before destroying an instance [decided].
- **GPU power cap:** the 26-10-03 instance's H100 was capped at 500 W and throttled to ~1 GHz; batch 16 ran 40% slower [verified: `nvidia-smi`]. Filter offers on `gpu_max_power` (FUTURE_WORK.md).
- **Files:** scripts, metrics, configs, run metadata and progress logs of all runs in `~/spd_out/26-10-02_training_run/` (local only, not in git; the only copy of e2a's metrics). Rule: no artifacts in `convos/` [decided; recorded in `docs/CODING.md`]. Figures were deleted; the WandB runs still have them; `scripts/compare_e2.py` prints the comparison of all runs. Checkpoints on WandB (`bitt-j-personal/spd/<run id>`), except e2a's, which were not kept.

## Open questions to Julian

1. E5: tune the last-token arm (batch 16 / 5k vs batch 4 / 10k, optionally batch 32 / 2.5k) and a padded-arm sanity run, ~1.5–2 h of an uncapped H100.
2. bf16 loading of the frozen model (code change): proposal pending.
3. Commit: Julian commits personally [decided: `C:` comment in the LOG, *clean-up options*].

The full list of open issues (including the analysis script, the untrained baseline run and the end-of-statement confound) is in the LOG, answer to "Are there any other open issues?".

- See also: [convos/julian/26-10-02_overview_of_goal_SUMMARY.md] — experiment design, the code changes the runs depend on, and tPD background.
