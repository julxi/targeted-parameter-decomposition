# SUMMARY: Prepared datasets (Truth-is-Universal data under data/tiu/)

**Last updated:** 26-10-04 (sync: generator location reworded after its first consumer config was replaced; FEEDBACK-entry decision and C: status added)

**What exists.** `data/` holds git-tracked, versioned, pre-split datasets for LM decompositions. A config lists them under `task_config.prepared_datasets`; training concatenates their train splits, and eval uses their test splits. Loader: `spd/experiments/lm/prepared_datasets.py` (checks every split file against the sha256 in its manifest). Layout and manifest fields: `data/README.md`. The infrastructure and the tiu v1 data predate this topic (commits `c774a4f`, `a68389e`, made without a LOG).

The one family so far is `data/tiu/`: 20 datasets from the Truth-is-Universal statements (Bürger et al., NeurIPS 2024), one per (upstream CSV, label), e.g. `data/tiu/cities_true/v1/`. All of one version share a train/test split by subject (city, Spanish word, element, animal, number pair). Used (all 20) by the three arm configs `spd/experiments/lm/honesty_targeted_decomposition/config_truth_{all_tokens,last_token,padded}.yaml`. Record format (`text`), one JSON object per line:

```json
{"text": "The city of Krasnodar is in Russia.", "label": true, "subject": "Krasnodar", "upstream_file": "datasets/cities.csv", "upstream_row": 0, "city": "Krasnodar", "country": "Russia", "correct_country": "Russia"}
```

(First line of `data/tiu/cities_true/v1/train.jsonl`. The fields after `upstream_row` are the extra upstream columns and vary by CSV.)

**Convention: generators live under `spd/`, data under `data/`** [decided: Julian, chat request transcribed in the LOG]. The tiu generator moved from `data/tiu/build_tiu.py` to `spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py`, in the folder of the configs that use it [concluded: `spd/data/` would clash with `spd/data.py`; a `spd/experiments/lm/prepared_datasets/` package would clash with the loader module]. Its consumer at the time, `config_truth_statements.yaml`, was later replaced by the arm configs (`convos/julian/26-10-02_overview_of_goal_SUMMARY.md`). Each manifest's `build.script` records the generator path, derived automatically.

Build: `python spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py --version vN` (refuses to overwrite an existing version).

**tiu v1 was rebuilt from the moved script** [decided: Julian, safe because no training had used it yet]. Julian committed the script first (`85aa0bd`), then built and committed the data (`8ab6fda`), so every manifest's `repo_commit` points to a commit that contains the generator, and `script_sha256` matches it. Compared with the old v1, the split files are byte-identical; only the manifests' `script`, `script_sha256`, `repo_commit` and `built_at` changed [verified: sha256 of all 40 split files and `git diff 4c8f58c HEAD`, 26-10-02]. Lesson for future rebuilds: commit the generator before building, so `repo_commit` contains it.

**No FEEDBACK.md entry for this lesson** [decided: Julian, "Delete it", transcribed in the LOG]. An agent had written one; Julian chose deletion after the agent argued that the lesson is too specific for FEEDBACK.md, which no agent reads by default (only a feedback review Julian starts reads it) [concluded]. The lesson lives in this SUMMARY only.

All C: comments in the LOG are answered.

- See also: [convos/julian/26-10-02_epistemic_memory_setup_SUMMARY.md] — set up the memory system and the coding rules (`docs/CODING.md`: the project is a single `spd/` package) that prompted this move.
- See also: [convos/julian/26-10-02_overview_of_goal_SUMMARY.md] — the tPD-on-Qwen experiment that uses these datasets; it targets all 20 tiu datasets (true and false statements mixed). Whether a true-only target would yield truth-separating components is an untested hypothesis there, not a fact.
- See also: [convos/julian/26-10-02_training_run_SUMMARY.md] — the 7B training runs on these tiu v1 datasets (batch/steps tuning); its LOG entry *legacy references frozen; rented-GPU rules moved to CODING.md* placed the `OUTDATED (26-10-03)` marker in `docs/PROJECT_REFERENCE.md` that points here for the current LM target-data facts.
