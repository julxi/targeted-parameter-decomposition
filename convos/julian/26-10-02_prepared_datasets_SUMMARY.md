# SUMMARY: Prepared datasets (Truth-is-Universal data under data/tiu/)

**Last updated:** 26-10-02 (generator moved into `spd/`; tiu data deleted, Julian rebuilds v1 after committing the move)

**What exists.** `data/` holds git-tracked, versioned, pre-split datasets for LM decompositions. A config lists them under `task_config.prepared_datasets`; training concatenates their train splits, and eval uses their test splits. Loader: `spd/experiments/lm/prepared_datasets.py` (checks every split file against the sha256 in its manifest). Layout and manifest fields: `data/README.md`. The infrastructure and the tiu v1 data predate this topic (commits `c774a4f`, `a68389e`, made without a LOG).

The one family so far is `data/tiu/`: 20 datasets from the Truth-is-Universal statements (Bürger et al., NeurIPS 2024), one per (upstream CSV, label), e.g. `data/tiu/cities_true/v1/`. All of one version share a train/test split by subject (city, Spanish word, element, animal, number pair). Used by `spd/experiments/lm/honesty_targeted_decomposition/config_truth_statements.yaml`. Record format (`text`), one JSON object per line:

```json
{"text": "The city of Krasnodar is in Russia.", "label": true, "subject": "Krasnodar", "upstream_file": "datasets/cities.csv", "upstream_row": 0, "city": "Krasnodar", "country": "Russia", "correct_country": "Russia"}
```

(First line of `data/tiu/cities_true/v1/train.jsonl`. The fields after `upstream_row` are the extra upstream columns and vary by CSV.)

**Convention: generators live under `spd/`, data under `data/`** [decided: Julian, chat request transcribed in the LOG]. The tiu generator moved from `data/tiu/build_tiu.py` to `spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py`, next to its only consumer config and the similar `build_prompts_capitals.py` [concluded: `spd/data/` would clash with `spd/data.py`]. Each manifest's `build.script` records the generator path, derived automatically.

Build: `python spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py --version vN` (refuses to overwrite an existing version).

**tiu v1 is currently deleted, pending a rebuild** [decided: Julian, safe because no training had used it yet]. A trial rebuild from the moved script produced split files byte-identical to the previously committed v1; only the manifests' `build` fields changed [verified: sha256 of all 40 split files compared, 26-10-02]. That rebuild was deleted again so Julian can rebuild after committing the moved script; the manifests' `repo_commit` will then point to a commit that contains it [decided]. Until then, `config_truth_statements.yaml` cannot load its data. After the rebuild, `git diff` should show changes only in the manifests' `build` blocks.

- See also: [convos/julian/26-10-02_epistemic_memory_setup_SUMMARY.md] — set up the memory system and the coding rules (`docs/CODING.md`: the project is a single `spd/` package) that prompted this move.
