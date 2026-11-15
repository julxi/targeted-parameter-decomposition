# SUMMARY: Open-ended research on the targeted decomposition methods (agent-led)

**Last updated:** 26-11-15 (GPU session done and the instance destroyed; all results in; report `reports/26-11-15_method_flaws.typ` written; nothing ratified by Julian yet)

Julian asked the agent to do its own research, starting from the paper draft (`reports/26-11-04_tpd_truth_paper_draft.typ`): open-ended, focused on evaluating the three targeted decomposition methods (plain tPD, recon, ortho) and their flaws, with the test techniques as a second result, seeds and underspecification kept in mind; the rented H100 could be used until the vast.ai credit ran out and was to be destroyed afterwards [decided: Julian's opening text in the LOG]. Everything below the brief is the agent's own choice and reading; no result is ratified [concluded unless tagged].

**Deliverable.** A self-contained report, `reports/26-11-15_method_flaws.typ` (14 pages; `typst compile reports/26-11-15_method_flaws.typ`; its tables are generated into `reports/26-11-15_method_flaws_tables.typ` by `~/spd_out/26-11-15_open_research/analysis/report_tables.py`). Its section 4 lists proposals for the paper draft.

## What was run

One session on the rented H100, 26-11-15 18:11–21:03 UTC (credit $17.81 → $4.40; instance destroyed) [verified]. All on the test split, 1,794 statements, 906 with a twin of equal length and opposite verdict; one run per cell. Decompositions: plain s1/s2, recon s1/s2, ortho s1–s3 and a fourth ortho seed trained in the session (`s-22600dd8`, the finals' config with seed 4; checkpoint only local, `~/spd_out/runs/spd-s-22600dd8/`).

- **The draft's P/R tests** for every decomposition: P = active set at mask 1, R = everything else; interventions zero, mean, fixed shift, twin average, swap; ten readouts.
- **Clamped swap** (new): the swapped part from the twin, the other part held at its clean value, so the two parts' effects are separable.
- **Model-anchored parts** (new): the original matrix restricted to an active set's write span, read span, or both; controls of the same size (random subspace, input PCA = the data-aligned basis, PCA of the output's difference between train twins).
- **Inspection pass** (new, `paper_inspect.py`): for all decompositions at once, Gram matrices of the twin differences and raw writes; gives the cancellation ratio κ, the *present share* ⟨δp,δy⟩/|δp|² (how much of P's twin difference is in the model's), and the agreement between decompositions cos(δp_i, δp_j).
- **Pile removal** (new, `paper_pile.py`): P removed at every position of 256 Pile validation windows, against the site removal.

Predictions were written into the LOG before the outputs of the new tests were read (entry *plan: finish the P/R grid…*); plain and recon came out as predicted, ortho's active set did not.

## Results [verified numbers: generated tables in `~/spd_out/26-11-15_open_research/analysis/`; readings concluded]

Swap fraction ν under the training probe (site swap 0.93):

| | \|P\| | P-swap | R-swap | P-swap, R held | R-swap, P held | write-span swap |
|---|---|---|---|---|---|---|
| plain s1 / s2 | 27 / 44 | 0.92 / 0.85 | 0.54 / 0.51 | 0.65 / 0.70 | 0.32 / 0.26 | 0.00 / 0.00 |
| recon s1 / s2 | 387 / 432 | 0.79 / 0.81 | 0.41 / 0.33 | 0.75 / 0.81 | 0.19 / 0.14 | 0.80 / 0.83 |
| ortho s1 / s2 / s3 / s4 | 29 / 31 / 58 / 38 | 0.31 / 0.67 / 0.40 / 0.43 | 0.68 / 0.47 / 0.65 / 0.62 | 0.12 / 0.34 / 0.21 / 0.23 | 0.80 / 0.60 / 0.72 / 0.72 | 0.03 / 0.11 / 0.10 / 0.04 |

1. **No decomposition isolates the truth value.** Swapping the rest R moves the verdict in all eight (0.33–0.68; ideal 0). With the other part held, the two effects add up to the site swap within 0.04, and P's own share is 0.65–0.81 for plain and recon and 0.12–0.34 for ortho. Removal tests had suggested the opposite (with P removed little separation is left); the losses only ask that P suffices, never that Δ is uninformative.
2. **Plain re-implements.** Its components' write is orthogonal to the layers' output (present share 0.004; |δp|² is 2.7–2.9 times |δy|²; κ 6.3–6.8); the model's own output in their write directions carries none of the swap (0.00, the random-subspace level); two seeds are unrelated (cos 0.05). The P-swap read by a truth probe rates it best of the three.
3. **Recon is faithful on the write side and broad.** Present share 1.03, write-span swap = P-swap, seeds agree (0.72; raw writes 0.96). But its active set is 70–79% of the output's energy at every statement token and its removal costs 77–80% of the site removal before the object; on the Pile it is quiet (6% of the energy). It is targeted at the statements as an input distribution, not at the verdict. Its read side is undetermined: read-span swap 0.13 / 0.16, and the two seeds' read subspaces overlap by 0.06 (write subspaces 0.59).
4. **Ortho is orthogonal only where it is trained.** All components together are a part of the output (their swap 0.26–0.58, the swap of their write span 0.30–0.40, random 0.01). The active set is not: 60–87% of its raw write is cancelled by the *other components*, and its write-span swap is 0.03–0.11 against a P-swap of 0.31–0.67. Its components carry the verdict for comparisons and negations (a different subset per seed) and 0.02–0.08 for affirmative facts. Seeds: raw writes group as (s1, s4) 0.48 and (s2, s3) 0.42, 0.02–0.12 across; Julian's "two modes", confirmed by the fourth seed.
5. **A baseline does better.** The twin-difference PCA part of the size of ortho s1's active set (29 directions) gives P-swap 0.80 (Yes/No 1: 0.77), R-swap 0.37, with side effects below the site's; 14 directions give 0.76, 387 give 0.86 with R-swap 0.05 (as broad as recon). It is a part of the model by construction and supervised by which statements are twins.
6. **About the tests.** The write-span swap separates decomposition from re-implementation where the truth-probe swap cannot; removal and swap disagree in both directions; unclamped P- and R-swaps are not additive; the read-span swap is not a criterion (the faithful method fails it); weight-space similarity to W is low even for recon (0.10–0.18), so components are parts of the weights only as far as the statements can see.

## Limits and open points [assumed where proposals]

- All tests use components at mask 1, not weighted by causal importance. For ortho this matters most (the unimportant components cancel the active set at mask 1). A CI-weighted P is the first thing to test before the ortho claims go into the paper.
- One run per cell; in-distribution twins of equal length only (no translation families, no OOD).
- Dropped for GPU time: complements of the anchored parts, all-components runs for plain s2 / recon s2 / ortho s4.
- Seed 4 was trained on an H100 next to other jobs (seeds 1–2 on an RTX PRO 6000).

## Code and data

- Code (uncommitted, on top of Julian's staged versions; `make check` clean, tests pass): `paper_interventions.py` (`span_split`, `SwapClamped`), `paper_tests.py` (targets `readspan | writespan | bothspan | randspan | pcaspan | diffspan`, `--span_base all`, `--span_k`, `--span_seed`; intervention `swap_clamped`), new `paper_inspect.py`, `paper_pile.py`, `tests/test_paper_span_splits.py`.
- Outputs: `~/spd_out/26-11-15_open_research/vm/` (sha256-checked copy of the instance's output directory: `test/` per-statement arrays and summaries, `logs/`, `scripts/` as run, `progress`), `analysis/` (scripts and generated tables), `scripts/` (job lists, configs `ortho_s4.yaml`).
- Run notes worth keeping: an 80 GB card does not hold a training (31 GB) plus two test workers plus a `pcaspan` job (32 GB); see FEEDBACK.md.

## See also

- See also: [convos/julian/26-11-04_paper_draft_SUMMARY.md] — the paper draft whose tests (section 2) and results skeleton (section 3) this topic fills for all decompositions and extends; the seeded-defect (scaling dial) runs there are the earlier seed-1 version of the P tests.
- See also: [convos/julian/26-11-05_evaluation_categories_SUMMARY.md] — the definition and failure modes (second route, re-implementation, κ) that the R-swap, the write-span swap and the present share measure.
- See also: [convos/julian/26-10-29_paper_writing_SUMMARY.md] — phase D tables (T0, A8/T3, "full − inactive") that the swap results correct or explain.
- See also: [convos/julian/26-10-03_wandb_storage_SUMMARY.md] — run register row for `s-22600dd8` (checkpoint not on WandB).
