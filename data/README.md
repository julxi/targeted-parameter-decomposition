# Prepared datasets

Git-tracked datasets for LM decompositions, referenced from a config by path:

```yaml
task_config:
  task_name: lm
  prepared_datasets:
    - data/tiu/cities_true/v1
    - data/tiu/sp_en_trans_true/v1
```

The train splits of all listed datasets are concatenated for training, the test splits for eval.

## Layout

```
data/<family>/<dataset>/<version>/
    manifest.yaml                      # provenance, build params, split hashes
    train.jsonl
    test.jsonl
```

This folder holds data only. Generators are code and live in the `spd` package, next to the
experiment that uses the data (all code in this project lives under `spd/`). Each manifest's
`build.script` gives the generator's path.

- **Versions are immutable.** `v1`, `v2`, ... are never edited after being committed; changing the
  data means building a new version. Loading verifies every split file against the `sha256` in its
  manifest, so a run's config (which names the version) pins the exact data it trained on.
- **Pre-split.** Every dataset ships its own `train` and `test` split (exception: the `uth/` test
  tasks, below, ship only a `test` split, so they cannot be trained on by accident). Generators that build several
  related datasets should split them consistently (e.g. by subject), so mixing datasets in one run
  never leaks a test subject into training.

## manifest.yaml

```yaml
name: tiu/cities_true
version: v1
format: text                # record format, see below
description: ...
source:                     # where the data comes from
  reference: <citation>
  url: <upstream repo / page>
  commit: <upstream commit or revision>
  file: <upstream file>
  file_sha256: <hash of the upstream file as downloaded>
  license: <license>
build:
  script: <generator path under spd/>
  script_sha256: <hash of the generator at build time>
  repo_commit: <this repo's HEAD at build time>
  built_at: <UTC timestamp>
  params: {...}             # generator arguments
splits:
  train: {file: train.jsonl, n_records: <int>, sha256: <hash>}
  test: {file: test.jsonl, n_records: <int>, sha256: <hash>}
```

## Record formats

- `text`: `{"text": "<the sequence>", ...metadata}`. Metadata fields (label, subject, upstream row,
  ...) are free-form and ignored by training.
- `chat` (planned, not yet loadable): `{"messages": [{"role": ..., "content": ...}, ...], ...metadata}`,
  rendered with the tokenizer's chat template at load time.

## Families

- `tiu/`: true/false statements from Truth-is-Universal (Bürger et al., NeurIPS 2024), one dataset
  per (upstream file, label), split by subject. Build with
  `python spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py --version vN`.
- `pile_code/github_lines`: single lines of source code from the Pile's GitHub subset, length-matched
  to tiu v1; a low-truth control target. Build with
  `python spd/experiments/lm/honesty_targeted_decomposition/build_pile_code.py --version vN`.
- `uth/`: the Universal Truthfulness Hyperplane datasets (Liu et al., EMNLP 2024), one dataset per
  upstream dataset, true and false mixed (record field `label`). Each manifest has a `role`:
  `train_task` (train + held-out test split, length-capped) or `test_task` (the paper's held-out
  categories: test split only, not length-capped). Cleaning and roles are described in the
  generator's docstring. Build with
  `python spd/experiments/lm/honesty_targeted_decomposition/build_uth.py --version vN`.
