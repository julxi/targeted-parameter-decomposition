# Targeted Parameter Decomposition — Project Index

This is the agents' hub of the project memory: team, current status, and the index of all discussion topics. Agents read it at the start of every session (see [CLAUDE.md](CLAUDE.md)). The human-facing description of the code is [README.md](README.md).

This repository is Julian Bitterlich's fork (`julxi/targeted-parameter-decomposition`) of Madina Babazhanova's fork (`bbzmdn/targeted-parameter-decomposition`, the `upstream` remote; it added the honesty experiments) of the code accompanying "Targeted Recovery of Weight-Space Mechanisms From Neural Networks" (Vigouroux and Sharkey, 2026; arXiv:2607.13047; original code `Antovigo/targeted-parameter-decomposition`). That paper introduces targeted parameter decomposition (tPD): decomposing only the mechanisms that process a chosen subset of inputs, while a high-rank catch-all `delta` component absorbs everything else. The code is itself a fork of SPD (Stochastic Parameter Decomposition). Coding rules are in [docs/CODING.md](docs/CODING.md). [docs/PROJECT_REFERENCE.md](docs/PROJECT_REFERENCE.md) (the former `CLAUDE.md`: environment, experiments, architecture, commands, cluster rules) is a legacy lookup being phased out; its claims are `[assumed]`.

This project uses the [Epistemic Memory](https://github.com/FlorianDietz/EpistemicMemory) conventions: all project memory lives in LOG/SUMMARY file pairs under `convos/`, governed by [CLAUDE.md](CLAUDE.md). Human guide: [docs/HOW_TO_USE.md](docs/HOW_TO_USE.md). Open ideas and deferred items: [FUTURE_WORK.md](FUTURE_WORK.md). Workflow problems and lessons: [FEEDBACK.md](FEEDBACK.md).

## Team

- Julian Bitterlich — `convos/julian/`

Currently a team of one. If more people join, add a bullet per member here; agents identify the current user by matching `git config user.name` against this list, and ask before proceeding when the match is ambiguous or missing.

## Current status and goals

Current goal: test whether tPD (targeted parameter decomposition) on Qwen2.5-7B-Instruct recovers truthfulness mechanisms, starting with whether probes on its CI (causal-importance) values detect statement truth beyond baselines [decided: `convos/julian/26-10-02_overview_of_goal_LOG.md`]. See that topic's SUMMARY.

Active branch: `main` (all feature branches are merged into it as of 26-10-02). Current work: targeted decomposition on Truth-is-Universal true/false statements (prepared datasets under `data/tiu/`, three arm configs `spd/experiments/lm/honesty_targeted_decomposition/config_truth_{all_tokens,last_token,padded}.yaml`, which differ only in the training positions). First probe result (26-10-03, arm A only): CI probes reach 0.97–0.98 test accuracy, but an untrained CI network reaches 0.998, so plain probe accuracy cannot show tPD-specific truth structure; see `convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md`.

## Maintenance

Recurring tasks:

| Task | Last run | Due when | Notes |
|------|----------|----------|-------|
| Sync (LOG/SUMMARY drift) | 26-10-04 (all 6 pairs) | After many LOG changes, or ~monthly | `!sync`; procedure: `docs/procedures/sync.md` |

## Conversation Index

All discussions live in `convos/{user}/`. **When starting a new topic, add a row here.** Read the SUMMARY first; open the LOG only if the SUMMARY is insufficient. Status cells describe current state only — rewrite them on update, don't append.

| Date | User | Topic | Status | SUMMARY | LOG |
|------|------|-------|--------|---------|-----|
| 26-10-03 | julian | Test accuracies of the decompositions (truth probes on CIs) | Results in: trained CIs 0.970–0.984 test acc, but untrained CIs and residual stream both 0.998, so CI probing alone shows nothing tPD-specific; next: generalisation tests, sparsity-matched untrained baseline (each statement keeps its own top ~10 CIs). Frozen model now bf16 (`pretrained_model_dtype`) | [SUMMARY](convos/julian/26-10-03_test_accuracy_analysis_SUMMARY.md) | [LOG](convos/julian/26-10-03_test_accuracy_analysis_LOG.md) |
| 26-10-03 | julian | WandB storage: 5 GB cap exceeded; checkpoint policy | Intermediate checkpoints deleted (1.50 GB real left; counter showed 9.87 GB, lags deletions, re-check pending); arm configs save final checkpoint only; rule in `docs/CODING.md` | [SUMMARY](convos/julian/26-10-03_wandb_storage_SUMMARY.md) | [LOG](convos/julian/26-10-03_wandb_storage_LOG.md) |
| 26-10-02 | julian | Training runs: tPD arms on Qwen2.5-7B (vast.ai H100), utilisation, batch size | Stopped after E3 (26-10-03). Verdict ratified and applied to all arm configs: batch 16, 5k steps, LR 5e-4; committed. Next: tune the last-token arm (E5), then run all arms at final settings. Artifacts in `~/spd_out/26-10-02_training_run/` | [SUMMARY](convos/julian/26-10-02_training_run_SUMMARY.md) | [LOG](convos/julian/26-10-02_training_run_LOG.md) |
| 26-10-02 | julian | Overview of goal: tPD on Qwen for truthfulness; tPD method reference; legacy honesty-folder inventory | Closed: experiment designed (3 tPD arms on Qwen-7B + untrained baseline) and training code/configs done (committed); continued in the training-run and test-accuracy topics | [SUMMARY](convos/julian/26-10-02_overview_of_goal_SUMMARY.md) | [LOG](convos/julian/26-10-02_overview_of_goal_LOG.md) |
| 26-10-02 | julian | Prepared datasets (tiu data, generators under `spd/`) | Done: tiu generator moved into `spd/`, v1 rebuilt after committing it (data unchanged, provenance updated) | [SUMMARY](convos/julian/26-10-02_prepared_datasets_SUMMARY.md) | [LOG](convos/julian/26-10-02_prepared_datasets_LOG.md) |
| 26-10-02 | julian | Epistemic Memory setup | Setup complete and committed | [SUMMARY](convos/julian/26-10-02_epistemic_memory_setup_SUMMARY.md) | [LOG](convos/julian/26-10-02_epistemic_memory_setup_LOG.md) |
