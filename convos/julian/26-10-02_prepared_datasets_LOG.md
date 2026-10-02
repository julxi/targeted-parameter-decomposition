C: (transcribed from chat) I read that the rule is thet code lives under `spd`, I think at least the important ones should live there. In a previous session I created the `data` folder, but I didn't know of the rule yet, so the code for creating the data lives in that folder.

I want to move the generator script to `spd`. Remove the data in the data folder and rerun the data generation. This is safe to do because the data has not been used in any training yet.

---

26-10-02, Claude Opus 5.5 — *generator moved into spd, tiu v1 rebuilt*

I have read INDEX.md, `docs/CODING.md`, `data/README.md`, the generator `data/tiu/build_tiu.py`, the loader `spd/experiments/lm/prepared_datasets.py` and the config `spd/experiments/lm/honesty_targeted_decomposition/config_truth_statements.yaml`. The datasets were built and committed in an earlier session that has no LOG (commits `c774a4f` "Add prepared datasets under data/ ..." and `a68389e` "Add Truth-is-Universal prepared datasets v1").

**Background (what the data is).** `data/tiu/` holds 20 prepared datasets built from the Truth-is-Universal true/false statements (Bürger et al., NeurIPS 2024): one per (upstream CSV, label), e.g. `data/tiu/cities_true/v1/`. Each version dir has `train.jsonl`, `test.jsonl` and a `manifest.yaml` (provenance, build params, sha256 of each split). The loader checks every split file against its manifest hash. All datasets in a version share one train/test split by subject (city, Spanish word, element, animal, number pair), so mixing them never leaks a test subject into training.

**What I did.**

1. Moved the generator from `data/tiu/build_tiu.py` to `spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py` (plain `mv`, not `git mv`, so no git state change) [decided: code lives under `spd/`, chat request above].
   - Location: [concluded] next to its only consumer `config_truth_statements.yaml` and the existing builder `build_prompts_capitals.py` in the same folder. I rejected two alternatives. `spd/data/` would clash with the existing module `spd/data.py`. A `spd/experiments/lm/prepared_datasets/` package would clash with the loader `prepared_datasets.py` and force an import rename. Moving the file again later is cheap if Julian prefers somewhere else.
   - Code changes: output dir is now `REPO_ROOT / "data" / "tiu"` (`REPO_ROOT` from `spd.settings`) instead of the script's own folder. The manifest's `build.script` is now derived from the script's location instead of a hardcoded string, so it cannot go stale on a future move. The docstring states why the script lives apart from its outputs.
   - basedpyright flagged `df[df["label"] == ...]` (pandas stubs type a boolean-mask index as `Series | DataFrame`). That was invisible before because type checking covers only `spd` and `tests`, not `data/`. I changed it to `df.loc[...]`, which has the same runtime behaviour and is typed `DataFrame`. The file now passes basedpyright, ruff check and ruff format [verified: run 26-10-02 on this file only; I did not run the full `make check`].
2. Deleted `data/tiu/` entirely and rebuilt with `python spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py --version v1` (same defaults as before: `test_frac 0.2`, `split_seed 0`; recorded in each manifest). I rebuilt twice, because the type fix changed the script's hash. The committed manifests come from the final script.
3. Verification [verified: 26-10-02]:
   - Before deleting, I saved the sha256 of all 40 split files. After the rebuild, all 40 match byte-for-byte. Expected, because the upstream CSVs are fetched at a pinned commit (`605ef00`) and the split is a deterministic hash of subject + seed. So the data content of v1 is unchanged; only provenance moved.
   - Manifest diffs touch only the `build` block: `script` (new path), `script_sha256` (`84b48d…` → `9bea8d…`, which matches `sha256sum` of the moved script), `repo_commit` (`c774a4f` → `4c8f58c`) and `built_at`.
   - `load_prepared_datasets` over all 20 `data/tiu/*/v1` dirs loads 6560 train and 1794 test texts with all hash checks passing.
4. Docs: `data/README.md` now says the folder holds data only and generators live under `spd/` (layout block, manifest example, build command). In the legacy `docs/PROJECT_REFERENCE.md`, the directory-tree line for `data/` is corrected in place.

**Things to know.**

- [concluded] Rebuilding `v1` in place breaks the "versions are immutable" rule in `data/README.md`. That is acceptable here because nothing has trained on it (Julian's statement above) and the split contents are byte-identical. Only the manifests' `build` fields differ from the committed version.
- Provenance caveat: `repo_commit` records `4c8f58c`, the HEAD at build time, but the moved script is not committed yet, so that commit does not contain the script at its new path. `script_sha256` identifies the exact script, so the build is still traceable once the move is committed. The old build had the same kind of gap: it recorded `c774a4f`, which predates the commit that added v1. If this matters, a later improvement would be to have the generator refuse to run on a dirty tree, or to record the dirty state. I have not done that (YAGNI) [assumed].
- Unchanged: the loader, the config and the dataset paths (`data/tiu/<name>/v1`), so `config_truth_statements.yaml` works as before.

---

C: (transcribed from chat) ah, yes. I forgot. Delete the data files again. I will generate them after the commit

---

26-10-02, Claude Opus 5.5 — *tiu data deleted, to be rebuilt after commit*

I deleted `data/tiu/` again, which removes all 60 generated files (20 datasets × `train.jsonl`, `test.jsonl`, `manifest.yaml`). `data/README.md` stays. This settles the provenance caveat in the entry *generator moved into spd, tiu v1 rebuilt*. Once the moved script is committed, a rebuild records a `repo_commit` that contains the script at its new path. That means the split-file hash comparison in that entry no longer describes anything in the working tree. Its conclusion should still hold for the next rebuild, because the upstream CSVs are pinned and the split is deterministic. A check after the rebuild: `git diff` of the new v1 should touch only the manifests' `build` blocks.

Until the data is rebuilt, `config_truth_statements.yaml` fails at load time, because its `data/tiu/*/v1` dirs are missing. The generator also refuses to write into an existing version dir, so the deletion is needed for the rebuild anyway.

Rebuild after committing: `python spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py --version v1`
