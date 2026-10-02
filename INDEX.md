# Targeted Parameter Decomposition — Project Index

This is the agents' hub of the project memory: team, current status, and the index of all discussion topics. Agents read it at the start of every session (see [CLAUDE.md](CLAUDE.md)). The human-facing description of the code is [README.md](README.md).

This repository is Julian Bitterlich's fork (`julxi/targeted-parameter-decomposition`, upstream `bbzmdn/targeted-parameter-decomposition`) of the code accompanying "Targeted Recovery of Weight-Space Mechanisms From Neural Networks" (Vigouroux and Sharkey, 2026; arXiv:2607.13047). That paper introduces targeted parameter decomposition (tPD): decomposing only the mechanisms that process a chosen subset of inputs, while a high-rank catch-all `delta` component absorbs everything else. The code is itself a fork of SPD (Stochastic Parameter Decomposition). Coding rules are in [docs/CODING.md](docs/CODING.md). [docs/PROJECT_REFERENCE.md](docs/PROJECT_REFERENCE.md) (the former `CLAUDE.md`: environment, experiments, architecture, commands, cluster rules) is a legacy lookup being phased out; its claims are `[assumed]`.

This project uses the [Epistemic Memory](https://github.com/FlorianDietz/EpistemicMemory) conventions: all project memory lives in LOG/SUMMARY file pairs under `convos/`, governed by [CLAUDE.md](CLAUDE.md). Human guide: [docs/HOW_TO_USE.md](docs/HOW_TO_USE.md). Open ideas and deferred items: [FUTURE_WORK.md](FUTURE_WORK.md). Workflow problems and lessons: [FEEDBACK.md](FEEDBACK.md).

## Team

- Julian Bitterlich — `convos/julian/`

Currently a team of one. If more people join, add a bullet per member here; agents identify the current user by matching `git config user.name` against this list, and ask before proceeding when the match is ambiguous or missing.

## Current status and goals

Current goal: train a tPD (targeted parameter decomposition) of a Qwen model [decided: `convos/julian/26-10-02_epistemic_memory_setup_LOG.md`, entry *setup implemented*]. Details will be recorded in upcoming topics.

Active branch: `main` (all feature branches are merged into it as of 26-10-02). Current work: targeted decomposition on Truth-is-Universal true/false statements (prepared datasets under `data/tiu/`, config `spd/experiments/lm/honesty_targeted_decomposition/config_truth_statements.yaml`).

## Maintenance

Recurring tasks:

| Task | Last run | Due when | Notes |
|------|----------|----------|-------|
| Sync (LOG/SUMMARY drift) | — | After many LOG changes, or ~monthly | `!sync`; procedure: `docs/procedures/sync.md` |

## Conversation Index

All discussions live in `convos/{user}/`. **When starting a new topic, add a row here.** Read the SUMMARY first; open the LOG only if the SUMMARY is insufficient. Status cells describe current state only — rewrite them on update, don't append.

| Date | User | Topic | Status | SUMMARY | LOG |
|------|------|-------|--------|---------|-----|
| 26-10-02 | julian | Epistemic Memory setup | Setup complete; awaiting diff review and commit | [SUMMARY](convos/julian/26-10-02_epistemic_memory_setup_SUMMARY.md) | [LOG](convos/julian/26-10-02_epistemic_memory_setup_LOG.md) |
