I want to test your abbilities to do your own research. As a starting point you should take this research about mostly captured by `reports/26-11-04_tpd_truth_paper_draft.typ`.

Here are some points about the desired results:
- first and foremost it's open ended
- it's primary focus is on evaluating the different targeted decomposition methods and I think the main results will be their flaws. Examples of what I think could be some of the results:
    - plain tPD suffers from reimplementation (carving out it's own mechanism within the weight matrices that does not reflect at all how the model does the computation)
    - recon tPD can not really be called targeted. It uses more paramaters
    - ortho is more targeted looks better than plain tPD but still shows clear weaknesses
- an auxiliary result is our techniques to actually show the methods deficenies
- this is one of the challenges of this project: we are analysing a technique and also the methods to analyse the techniques themselves.
- we want to compare the methods at their best, for that we have run different seeds. At least ortho seems to have at least two modes it converges to. This behaviour can also be seen in full parameter decomposition on toy models (my expierence). This also has to be kept in mind.
- most of this projects test battle with underspecification: : multiple, distinct explanations can equally well account for the same input–output patterns
    - that is why we also consider additional activation based tests
    - nontheless maybe there can be enough behavioural tests to capture at least enough failure modes to cover the proposed methods
- The results so far are a result of iterative explorative work and consolidation your approach is open to you. If you see a thread that you want to follow go for it. If you think only some final tests have to be run and the results only have to put into perspective go for it too.
- you have full access to the rented h100. You have as long as you want to come up with your result. You can use up all of the credits for vast.ai on the h100. If you don't need all the time, then destroy the machine after your work.
---

26-11-15, 18:29 UTC, Claude Fable 5.2 — *plan: finish the P/R grid, add model-anchored tests, compare seeds functionally*

Context I have: CLAUDE.md, INDEX.md, `docs/CODING.md`, the paper draft (`reports/26-11-04_tpd_truth_paper_draft.typ`, all of it), the SUMMARYs of the paper-draft, paper-writing, tPD-truth-paper, evaluation-categories and vast-rentals topics, the dial analysis tables (`~/spd_out/26-11-04_paper_draft/dial/analysis/tables.md`), and the code of `paper_tests.py` / `paper_interventions.py`. I did not read the long LOGs. (This file was `convos/julian/new.md`; renamed, Julian's text above is unchanged.)

**Budget** [verified: `vastai show instances`, `vastai show user`, 18:14 UTC]. One H100 SXM is running (instance 60820277, $4.57/h, power limit 700 W); the account credit is $17.81, so about 3.9 h of GPU in all, until about 22:00 UTC. Everything below is sized to that: inference only at first, no new training unless time is left (one final run takes about 41 min on an H100 alone [assumed: 1.01 h on the RTX PRO 6000 from the run metadata, times the 0.68 speed ratio of the vast-rentals SUMMARY]).

**Where the paper stands, as I read it.** Section 2 of the draft defines the behavioural tests (interventions removed / twin average / swap / fixed shift on a part P of the seven `down_proj` outputs or on the rest R, read by truth probes and by other-quality probes). What exists: site versions, and the scaling dial on seed 1 of each method. Missing: the P and R tests for the other four decompositions, every R test, and anything that says whether a part P that moves the verdict is a part of what the model itself writes. The last point is the one Julian's first expected result needs (plain tPD "carving out its own mechanism").

**Plan.** Three threads, all on the test split (no thresholds are tuned, as in the dial run).

1. *Finish the grid* (batch A): P = active set and R = everything else, for all seven final decompositions (plain s1/s2, recon s1/s2, ortho s1–s3), five interventions, ten readouts, plus the site version again on this GPU so every reference is same-machine. Existing code, no changes.

2. *Model-anchored versions of each P* (batch B; new targets in `paper_tests.py`, `paper_interventions.span_split`). For a component set S with read vectors $v_c$ and write vectors $u_c$ in one layer, let $Q_V$, $Q_U$ be orthonormal bases of their spans. Three parts of the ORIGINAL matrix W with S's dimensions:
   - `readspan`: $W Q_V Q_V^\top$, what W itself writes in response to the input directions S reads;
   - `writespan`: $Q_U Q_U^\top W$, the part of W's output in the directions S writes to;
   - `bothspan`: $Q_U Q_U^\top W Q_V Q_V^\top$.
   Swapping such a part between twins is a subspace interchange intervention on the unsplit model; nothing of the decomposition's own gains enters. Reason for the test: if S is a part of the model's computation, then S's twin difference $\delta p$ is contained in the model's twin difference inside S's write span, $Q_U Q_U^\top \delta y = \delta p + Q_U Q_U^\top \delta r$, so the `writespan` swap should move the verdict at least about as far as the P-swap, unless the rest of the model writes against P in P's own directions. A P-swap that moves the verdict much further than its `writespan` swap is therefore the behavioural signature of a re-implementation: the verdict effect comes from writes the model does not have. This is the swap version, over all seven layers at once, of the frozen plan's A8/T3 removal (project the write span out of the original output, one layer at a time). Controls with the same per-layer dimensions: `randspan` (random output subspace) and `pcaspan` (top input principal directions; this is the data-aligned basis cut to size). The `pcaspan` control also answers Julian's second expected result for recon: if a top-PCA part of the same size does what recon's active set does, the selection of recon's components carries no targeting.
   Caveat, stated before the data: a faithful part need not be reproduced by either anchored version (two mechanisms can share read and write directions; separating them is what parameter decomposition is for). So agreement supports faithfulness; a deficit shows cancellation in P's own directions, which is the specific failure we call re-implementation.

3. *One inspection pass over all seven decompositions* (batch C; new script `paper_inspect.py`): one clean forward pass per twin, then Gram matrices between the twin differences of every decomposition's P-write, its anchored versions and the site output, per layer, family and position class. From them: the cancellation ratio κ between twins (proposed in the draft, never computed); the share of each P's twin difference that is present in the model's ($\langle\delta p, \delta y\rangle / |\delta p|^2$); and, new, the agreement between seeds and methods as functions, $\cos(\delta p_i, \delta p_j)$. The last is my handle on Julian's seeds/modes point and on underspecification: if two seeds of a method found the same mechanism, their P-writes agree on the same inputs; component-by-component matching is not needed.

   Then batch D: the R mirrors (Δ-swap = R of all components; complement of the anchored spans).

**Predictions, written before any output of B–D is read** [assumed]:
- plain: P-swap ν ≈ 0.9 (known for s1), `writespan` swap near the random-subspace level, κ ≫ 1, present share ≪ 1; seeds disagree as functions.
- recon: `writespan` and `readspan` swaps about equal to its P-swap; `pcaspan` of the same size about equal too; κ ≈ 1; seeds agree.
- ortho: `writespan` swap ≥ its P-swap (0.31 for s1); κ near 1; s1 differs from s2/s3.
- `randspan` at these sizes: ν near 0.

**Code changes** [verified: unit check of the span algebra on small tensors, ruff, basedpyright; not yet run on the model]: `paper_interventions.py` + `span_split` (returns an ordinary `ComponentSplit`, so the hooks are unchanged); `paper_tests.py run` + targets `readspan | writespan | bothspan | randspan | pcaspan` and `--span_seed`; new `paper_inspect.py`. Driver scripts and job lists: `~/spd_out/26-11-15_open_research/scripts/` (`worker.sh`, `jobsA–D.txt`); outputs on the instance under `/root/spd_out/open/`.

---

26-11-15, 19:05 UTC, Claude Fable 5.2 — *batch A in: the rest R carries the verdict in every method; run notes*

Context: as above, plus batch A's summaries (15 JSON files, `~/spd_out/26-11-15_open_research/vm/test/`; tables `~/spd_out/26-11-15_open_research/analysis/tables.md`, written by `analysis/extract.py` and `analysis/tables.py`).

**Batch A done** [verified: 15 runs exited 0 on the H100, test split, 1,794 statements, 906 twin rows; one run per cell]. P = active set at mask 1, R = everything else of the seven sites. Swap fraction ν under the training probe (truth rating 1) and under Yes/No 1 (0 = the statement's own score, 1 = its twin's; site swap 0.93 / 0.95):

| decomposition | size of P | P-swap ν (TR1 / YN1) | R-swap ν (TR1 / YN1) | P-swap: word-count KL (site 0.079) | P removed: prefix next-token KL (site 2.28) |
|---|---|---|---|---|---|
| plain s1 | 27 | 0.92 / 1.00 | 0.54 / 0.62 | 0.113 | 0.13 |
| plain s2 | 44 | 0.85 / 0.90 | 0.51 / 0.65 | 0.076 | 0.28 |
| recon s1 | 387 | 0.79 / 0.74 | 0.41 / 0.50 | 0.033 | 1.75 |
| recon s2 | 432 | 0.81 / 0.74 | 0.33 / 0.43 | 0.049 | 1.82 |
| ortho s1 | 29 | 0.31 / 0.27 | 0.68 / 0.78 | 0.011 | 0.27 |
| ortho s2 | 31 | 0.67 / 0.36 | 0.47 / 0.68 | 0.030 | 0.04 |
| ortho s3 | 58 | 0.40 / 0.25 | 0.65 / 0.74 | 0.025 | 0.07 |

First reading [concluded: from the table; single run per cell, intervals in the JSON files not yet tabulated]:
- **No method passes the R-swap.** The draft expects ν near 0 when the rest's writes are swapped ("the verdict stays with x"). It is 0.33–0.68 under the training probe and 0.43–0.78 under Yes/No 1, in all seven decompositions. So in every decomposition the part outside the active set carries a copy of the truth value that the verdict follows. This was invisible in the removal tests: with P removed, the rest alone leaves little of the verdict separation (0.02–0.31 here; T0 in the phase D tables), which reads as "P is necessary". The swap shows the rest does carry the value; removing P breaks the verdict for another reason (the state is far from any natural one). The training losses explain why nothing prevents this: the verdict must survive with Δ masked out (P sufficient), but no term asks that Δ's writes be uninformative about the verdict.
- **ν_P + ν_R exceeds the site's ν for plain and recon** (plain s1 1.46, recon s1 1.20; ortho s1 0.99). Each part is clamped to the twin's values at all seven layers while the other is recomputed, so the recomputed part at later layers follows what the clamped part wrote earlier. Parts are not independent routes; the sum is not a decomposition of the site effect.
- plain: P-swap as strong as the site swap, side effects at or above the site's (word count 0.113 vs 0.079 for s1). "Only P" (R removed) damages the statement's next-token predictions more than removing the whole sites (prefix KL 2.60 / 2.58 vs 2.28; object + end 3.65 / 3.74 vs 2.54): P alone writes things the model does not.
- recon: P removed costs 77–80% of the site removal's prefix KL (1.75 / 1.82 of 2.28): its active set is most of the layer's generic computation, as expected for this method.
- ortho: smallest side effects, but its P carries the minority of the swap effect in s1 and s3 (0.31, 0.40), and s2's P-swap does not transfer to Yes/No (0.67 → 0.36). The seeds differ in kind, not only in degree.

**Run notes.**
- Three test workers plus anything else do not fit: with the seed-4 training (31 GB) and two test workers the card was at 77.7 of 80 GB, and a `pcaspan` job needs about 32 GB by itself (seven 18,944² second-moment matrices and their eigendecomposition). Four jobs died with CUDA OOM or a cusolver handle error (`smokeB_rand`, `B_{ortho1,plain1,recon1}_readspan`) [verified: tracebacks in the instance logs]. Changes: QR for the span bases and the eigendecomposition of the twin-difference moment now run on the CPU; one test worker at a time while the training runs (`scripts/single.sh`); the `pcaspan` jobs moved to a last batch E. To make the workers skip the three `pcaspan` jobs of batch B, I appended lines `B_*_pcaspan exit 0 NOT-RUN-moved-to-jobsE-(memory)` to the instance's progress file; they are not results.
- Decided to train **one more ortho seed** (seed 4, config = `ortho_s3.yaml` with the seed and name changed and `sync_checkpoints_to_wandb: false`, because WandB is at about 4.75 of 5 GB; `scripts/ortho_s4.yaml`), started 18:59 UTC next to the tests [concluded: agent's choice within Julian's open brief; reason: his point on ortho's modes rests on three seeds]. A fifth seed only if time is left.
- Added after the plan: target `diffspan` (the top principal directions of the site output's difference between train twins, same per-layer size as an active set; the simplest truth-specific subspace, fitted without a decomposition), and `paper_pile.py` (P removed at every position of Pile validation windows, against the site removal).
- Inspection smoke on val (plain s1, ortho s1; 510 twin rows) [verified: one run]: plain's active-set twin difference has 2.9 times the energy of the model's whole twin difference at these sites and almost none of it is in the model's difference (⟨δp, δy⟩/|δp|² = 0.004; κ = 6.7). Ortho s1's is small (0.09 of the energy) but also mostly absent from the model's difference (0.106; κ = 1.16). The test-split run with all seven follows.

---

26-11-15, 19:35 UTC, Claude Fable 5.2 — *write-span swaps, inspection and Pile in: plain is parasitic, recon is faithful but broad, ortho is cancelled on differences*

Context: as above, plus the outputs named below. All on the test split, 906 twin rows, one run per cell; tables generated by `~/spd_out/26-11-15_open_research/analysis/{tables,score,inspect_tables}.py` (`tables.md`, `score.md`, `inspect.md` there).

**Write-span swap** (batch B, first half) [verified: 7 runs]: swap, between twins, only the part of the ORIGINAL `down_proj` output that lies in the span of the active set's write vectors (same directions and dimensions as P; nothing of the decomposition's gains). ν under the training probe, next to the P-swap:

| | P-swap ν | write-span swap ν | write span removed: separation left |
|---|---|---|---|
| plain s1 / s2 | 0.92 / 0.85 | 0.00 / 0.00 | 0.99 / 0.99 |
| recon s1 / s2 | 0.79 / 0.81 | 0.80 / 0.83 | 0.30 / 0.25 |
| ortho s1 / s2 / s3 | 0.31 / 0.67 / 0.40 | 0.03 / 0.11 / 0.10 | 0.50 / 0.26 / 0.19 |
| random subspace, sizes of plain s1 | | 0.00 | 1.01 |

(All cells read from `analysis/tables.md`, rows `<run> | writespan | P`, columns *swap nu* and *zero sep*, training probe.)

Predictions check: plain as predicted (random-subspace level); recon as predicted (equal to its P-swap); **ortho not as predicted**: I expected its write-span swap to be at least its P-swap; it is 0.03–0.11.

**Inspection** (batch C, `paper_inspect.py`; file `vm/test/inspect_test_plain1+…+ortho3.npz`) [verified: one run; self-checks in the script passed]. With p = the active set's write at mask 1, r = y − p, δ = difference between twins at the object and end tokens, pooled over the seven layers:

| | \|δp\|²/\|δy\|² | share of δp present in δy, ⟨δp,δy⟩/\|δp\|² | κ | the same share for all components | agreement with the other seed(s), cos(δp_i, δp_j) |
|---|---|---|---|---|---|
| plain s1 / s2 | 2.68 / 2.93 | 0.004 / 0.004 | 6.3 / 6.8 | 0.003 / 0.002 | 0.05 |
| recon s1 / s2 | 0.25 / 0.27 | 1.03 / 1.03 | 0.98 / 0.98 | 1.05 / 1.05 | 0.72 |
| ortho s1 / s2 / s3 | 0.09 / 0.10 / 0.07 | 0.10 / 0.13 / 0.11 | 1.17 / 1.17 / 1.13 | 0.29 / 0.29 / 0.27 | s1–s2 0.05, s1–s3 0.04, s2–s3 0.20 |

Raw writes at the statement tokens (mean of per-position ratios): plain's components at full strength write 1.9 (before the object) to 16 (end token) times |y|², with ⟨C, D⟩/|y|² equal to minus that, so ⟨C, y⟩ ≈ 0: the component write is orthogonal to the layer's output and Δ contains its negative. Recon's active set is 0.70–0.79 of |y|² at every position class with ⟨p, r⟩/|y|² between 0.005 and 0.03. Ortho: all components together have ⟨C, D⟩/|y|² between −0.010 and 0.001, as its loss asks, but the active set alone has ⟨p, r⟩/|y|² of −0.13 to −0.30 (s1) against |p|²/|y|² of 0.15 to 0.34: the components outside the active set write against it (split of the cross term, s1: ⟨p, inactive components⟩/|y|² = −0.13 to −0.32, ⟨p, Δ's write⟩/|y|² = −0.001 to 0.02).

Reading [concluded: from these tables; no intervals for the inspection numbers]:
- **plain is not a decomposition of the layer at all.** Its components write vectors that the layer's output does not contain (present share 0.004, write-span swap 0.00, removal of the write span changes nothing), larger than the output itself, and Δ cancels them. Its P-swap moves the verdict because the un-cancelled write is 2.7–2.9 times the size of the model's whole twin difference. Two seeds give unrelated functions (cos 0.05). This is Julian's first expected result, now with a behavioural test (write-span swap) and three internal numbers behind it.
- **recon is a faithful part**: its twin difference is an orthogonal piece of the model's (present 1.03, κ 0.98), the write-span swap equals the P-swap, and its seeds agree (0.72; raw writes 0.96). The price is breadth (next entry's Pile numbers; prefix KL 77–80% of the site's).
- **ortho's orthogonality holds only where it is trained.** On raw writes with all components it is exact. On the active set it fails (60–87% of its raw write is cancelled, by the inactive components), and on the twin differences, which are what carries the truth value, even the full component sum is 71–73% absent from the model's difference (for the active set alone: 62% of |δp|² cancelled by the inactive components and 25–28% by Δ in s1 and s2, 46% and 43% in s3). So the verdict effect of ortho's P-swap comes mostly from writes the model does not have in those directions, the same failure as plain's in a milder form.
  OUTDATED (26-11-15): holds for the active set only. The write span of all components together moves the verdict about as far as the components' own swap; see the entry *GPU session closed: all results, seed 4, baselines, the report*, result 4. The seeds fall into (s2, s3), weakly related, and s1, unrelated to both: Julian's "at least two modes".

**Pile** (batch C, `paper_pile.py`, 256 validation windows of 64 tokens; P removed at every position) [verified: one run]: site removal KL 0.742. Share of that for the active set / all components: plain 0.010–0.014 / 0.11; recon 0.187 / 0.19; ortho 0.006–0.009 / 0.025–0.028. Mean |p|²/|y|² on the Pile: plain 0.05–0.07 / 0.72–0.73, recon 0.06 / 0.08, ortho 0.025–0.036 / 0.12. So recon's components are quiet on the Pile (6–8% of the output energy, against 70–79% on the statements) while their removal still costs 19% of the site removal: recon is targeted at the statements as an input distribution, not at the verdict. Plain's components write 72–73% of |y|² on the Pile too (cancelled by Δ).

**Weight space, no data** (from the same npz): ⟨P, W⟩_F/|P|_F² per layer is 0.000–0.002 for plain, 0.001–0.053 for ortho, 0.10–0.18 for recon; plain's components have |P|_F²/|W|_F² = 0.95–0.97 (all components). Even recon's components are mostly not in W as matrices: off the statement distribution the component matrices are unconstrained. I report this as a caveat on calling the components "parts of the weights", not as a test (recon passes every data-based test with 0.10–0.18 here).

Still running: read-span and both-span swaps, random and PCA controls, the clamped swaps (new intervention `swap_clamped`: the edited part from the twin and the other part held at its clean value, to separate a part's own effect from the other part following it downstream), Δ-swap, twin-difference PCA baseline, write span of all components, ortho seed 4.

---

26-11-15, 21:40 UTC, Claude Fable 5.2 — *GPU session closed: all results, seed 4, baselines, the report*

Context: everything above, all outputs of the session (`~/spd_out/26-11-15_open_research/vm/`, 353 files), and the tables generated from them (`~/spd_out/26-11-15_open_research/analysis/`: `score.md`, `tables.md`, `inspect.md`, `family.md`, `family_yn.md`). Every number below was read from those files after they were generated.

**Session facts** [verified]. H100 instance 60820277, 18:11–21:03 UTC, destroyed (`vastai destroy` printed "destroying instance 60820277."; `vastai show instances` empty). Credit $17.81 → $4.40. Outputs copied and compared by sha256 (353 files identical). In `vm/progress`: 100 lines `exit 0 <time>` (jobs that ran; 6 of them smoke tests, 2 the training and the recovery of seed 4), four early failures with exit 1 (out of memory; three rerun successfully, the fourth a smoke test, also rerun), and 16 lines marked `NOT-RUN`, which are jobs I moved or dropped, not results. Dropped for time: the complements of the write/read spans, all components as P for plain s2 / recon s2 / ortho s4, the 8-per-layer size of the twin-difference baseline, `pcaspan` at plain s1's sizes.

**Ortho seed 4** [verified: one run]. Run `s-22600dd8` (`scripts/ortho_s4.yaml`: the finals' config, seed 4, checkpoint not synced to WandB), trained 18:59–20:14 UTC next to inference jobs (1.25 h). Checkpoint copied to `~/spd_out/runs/spd-s-22600dd8/` (sha256 equal to the instance's); row added to the run register. Active set 38 components (`vm/recovery/recovery_s-22600dd8.json`).

**Results** (test split, 906 twin rows, one run per cell; ν = swap fraction under the training probe unless marked; site swap 0.93). The report (below) has every table.

| | \|P\| | P-swap | R-swap | P-swap, R held | R-swap, P held | write-span swap | all comp. swap | Δ-swap |
|---|---|---|---|---|---|---|---|---|
| plain s1 | 27 | 0.92 | 0.54 | 0.65 | 0.32 | 0.00 | 0.74 | 0.54 |
| plain s2 | 44 | 0.85 | 0.51 | 0.70 | 0.26 | 0.00 | – | 0.51 |
| recon s1 | 387 | 0.79 | 0.41 | 0.75 | 0.19 | 0.80 | – | 0.25 |
| recon s2 | 432 | 0.81 | 0.33 | 0.81 | 0.14 | 0.83 | – | 0.23 |
| ortho s1 | 29 | 0.31 | 0.68 | 0.12 | 0.80 | 0.03 | 0.42 | 0.67 |
| ortho s2 | 31 | 0.67 | 0.47 | 0.34 | 0.60 | 0.11 | 0.58 | 0.40 |
| ortho s3 | 58 | 0.40 | 0.65 | 0.21 | 0.72 | 0.10 | 0.26 | 0.45 |
| ortho s4 | 38 | 0.43 | 0.62 | 0.23 | 0.72 | 0.04 | – | 0.66 |

1. **The rest carries the truth value in every decomposition** [verified: R-swaps and clamped swaps; reading concluded]. With the other part held at its clean value ("held", intervention `swap_clamped`), P's and R's swap fractions add up to the site swap within 0.04 in all eight decompositions under both probes. P's own share: plain 0.65–0.70, recon 0.75–0.81, ortho 0.12–0.34. So the R-swap result is not only the recomputed components following the rest. With all components as P, swapping Δ alone moves the verdict by 0.23–0.67.
2. **Plain re-implements** [verified numbers, reading concluded]: as in the previous entry; in addition the write span of *all* its components gives 0.02 against 0.74 for the components' own swap.
3. **Recon: faithful on the write side, broad, read side undetermined.** New here: its read-span swap is only 0.13 / 0.16, and the two seeds' read subspaces overlap by 0.06 (write subspaces 0.59) while their writes agree (cos 0.72 on twin differences, 0.96 raw). So the read-span test is not a criterion (the faithful method fails it), and the components do not identify which inputs the layers respond to [concluded]. Against baselines of its size (387 directions): input PCA P/R-swap 0.70 / 0.62, recon 0.79 / 0.41. My prediction "PCA of the same size about equal" was wrong in part: recon's selection is better than top-variance directions.
4. **Ortho** [verified numbers; reading concluded]. (a) All components together roughly pass the write-span test (their swap 0.42 / 0.58 / 0.26 for s1–s3, the swap of their write span 0.36 / 0.40 / 0.30; a random subspace of 96 dimensions per layer 0.01), so the full component sum is a part of the output, as its loss asks. (b) The active set is not: 60–87% of its raw write is cancelled, by the other components (table *Who cancels the active set?* in `inspect.md`), and its write-span swap is 0.03–0.11. This corrects the previous entry's wording "the same failure as plain's in a milder form": true for the active set, not for the full sum. (c) Its components carry the verdict for comparisons and negated statements, a different subset per seed, and 0.02–0.08 for the affirmative fact families (cities, element symbols) in all four seeds (`family.md`). (d) Seeds: raw active-set writes group as (s1, s4) cos 0.48 and (s2, s3) 0.42, across groups 0.02–0.12; on twin differences no pair exceeds 0.20. Seed 4 confirms two modes.
   OUTDATED (26-11-16): (c) holds for unclamped swaps only. Clamped (the rest held at its clean value), ortho's C carries 0.13–0.32 of the verdict on comparisons and 0.04–0.59 on negations; only s2 carries much there. See the entry *GPU session 2 results…*, test 1.
5. **Baseline** [verified]: the twin-difference PCA part (fitted on train twins; a part of W by construction) with the per-layer sizes of ortho s1's active set (29 directions): P-swap 0.80 (Yes/No 1: 0.77), R-swap 0.37, word-count KL under the swap 0.031 (site 0.079), prefix share 0.06. Two directions per layer (14): 0.76 / 0.54; one per layer: 0.12 / 0.85; 387 directions: 0.86 / 0.05 with prefix share 0.69; 96 per layer: 0.88. Weakest on the affirmative facts (cities 0.50, element symbols 0.34). No trained decomposition offers this at the size of a plain or ortho active set.

Check of my predictions (entry *plan*): plain ✓; recon ✓ except the PCA comparison; ortho ✗ for the active set (write-span swap far below the P-swap), ✓ for the grouping of seeds (s1 apart from s2/s3).

**Report** [concluded: agent's text; numbers generated]: `reports/26-11-15_method_flaws.typ` (+ generated `reports/26-11-15_method_flaws_tables.typ`; `typst compile reports/26-11-15_method_flaws.typ`), 14 pages: setting, the added tests, results per method, baseline, what the tests showed about the tests, proposals for the paper draft, limits, all tables. The tables file is written by `analysis/report_tables.py`; I checked every number in the prose against the generated tables once, by hand.

**Code** [verified: `make check` clean, 18 tests pass incl. the new `tests/test_paper_span_splits.py`; all new paths ran on the GPU]. Uncommitted, on top of Julian's staged versions:
- `paper_interventions.py`: `span_split` (readspan / writespan / bothspan / randspan / pcaspan / diffspan), intervention `SwapClamped` (`swap_clamped`), `BatchContext.own_y`; span bases by SVD on the CPU.
- `paper_tests.py`: the span targets, `--span_seed`, `--span_base all`, `--span_k`, `twin_difference_directions`.
- new `paper_inspect.py` (Gram matrices of twin differences and raw writes for several decompositions in one pass; weight-space numbers) and `paper_pile.py` (removal on Pile validation windows).

**What I would do next** [assumed: proposals]:
- Test P weighted by causal importance per token (all tests here use mask 1; for ortho the unimportant components cancel the active set at mask 1, so this is the most important open check before the ortho claims go into the paper).
- Twins outside the training families, and the families without equal-length twins.
- Put the clamped swap, the write-span swap, the present share and the agreement table into the draft's evaluation and results sections (proposals in the report's section 4).

C: Alright, there are some promossing results in here. Next steps:
- I think we should actually use S = all for the tests, for two reasons (1) the decompositions seem to still use the inactive components (mostl likely in a statistical manner). (2) the ortho loss uses all of the components
- We can remove some less expressive tests. Can you give me a minimal selection of tests that we should report in the main part of the paper?

Answer (26-11-16, Claude Opus 5.5): I have read this LOG in full, the SUMMARY, sections 4–6 of `reports/26-11-15_method_flaws.typ`, and the generated tables in `~/spd_out/26-11-15_open_research/analysis/` (`score.md`, `tables.md`, `inspect.md`). Every number below was read from those tables in this session. ν = swap fraction under the training probe (truth rating 1); site swap 0.93.

**1. S = all (P = all 96 components per layer = C, R = Δ).** I agree, and there are two more reasons for it besides yours:
- *Same size for every method.* C has 672 components in every decomposition. The active sets range from 27 (plain s1) to 432 (recon s2), so every comparison between methods on the active set mixes in a size effect.
- *It is the split the method proposes.* Our definition of a perfect decomposition (evaluation-categories topic, steps 4–5) is stated over C and Δ, not over the active set. The cut "active set at mask 1" appears in no loss: training samples every component's mask between its CI and 1.

On your reason (1): the evidence we have is that, at mask 1, ortho's inactive components cancel 60–87% of the active set's raw write. That fits "used statistically", because the random masks in training make their summed effect matter. But it is equally consistent with a mask-1 artefact: at their own CI (≈ 0) they would barely write at all. A CI-weighted P separates the two readings. It is still untested [assumed either way].  
OUTDATED (26-11-17): "still untested" is wrong for the components-only state. Phase D table 1c already compares all components at CI with the active set alone at CI (the inactive ones add 0.03–0.05 of KL recovered). Only the swap version is untested. See the answer *inactive components: what earlier topics already measured* at the bottom of this file.  
Your reason (2) holds: the ortho term is ⟨ΣC, Δx⟩, so only the full sum is constrained.

What S = all already shows (cells that exist):

| | C-swap | write span of C, swap | Δ-swap | present share of δC | C removed on the Pile, share of site removal | seed agreement cos(δC_i, δC_j) |
|---|---|---|---|---|---|---|
| plain s1 / s2 | 0.74 / – | 0.02 / – | 0.54 / 0.51 | 0.003 / 0.002 | 0.112 / 0.113 | 0.02 |
| recon s1 / s2 | – / – | – / – | 0.25 / 0.23 | 1.047 / 1.045 | 0.191 / 0.189 | 0.82 |
| ortho s1 / s2 / s3 / s4 | 0.42 / 0.58 / 0.26 / – | 0.36 / 0.40 / 0.30 / – | 0.67 / 0.40 / 0.45 / 0.66 | 0.29 / 0.29 / 0.27 / 0.29 | 0.028 / 0.025 / 0.025 / 0.027 | 0.07–0.11 |

Reading [concluded, from these cells; one run each]:
- Plain still re-implements, and more clearly than before: C-swap 0.74 against 0.02 for the model's own output in C's write span.
- Ortho's full sum passes the write-span test (0.30–0.40 against 0.26–0.58). The headline flaw then moves from "the active set is cancelled" to "Δ carries most of the truth value" (Δ-swap 0.40–0.67).
- Ortho's seeds disagree even on the full sum (0.07–0.11). The two-mode grouping seen in the active sets' raw writes is not visible in this column.
- There is a tension: ortho's C passes the write-span swap, but 71–73% of its twin difference is absent from the model's (present share 0.27–0.29). My guess is that the absent part does not drive the verdict [assumed, untested].

Costs of the switch:
- (a) The question "does the CI pick out the mechanism for this input" drops out of the main tests. That question is what the active set stands for, and it is tPD's actual output per input. I would keep one number for it (next point, test 6).
- (b) For recon, C is 76–88% of the output's energy at the statement tokens. "Targeted" then has to be shown by the off-target test (test 4); the truth tests cannot show it.
- (c) The report (`reports/26-11-15_method_flaws.typ`) uses the active set throughout and would need its result sections redone.
- (d) Cells are missing and need a GPU session (vast credit left: $4.40, about 1 h of H100 at $4.57/h):
  - the C-swap for plain s2, recon s1/s2 and ortho s4;
  - the write-span swap of C for the same four;
  - the clamped C/Δ swaps for all eight (none exist with S = all; without clamping, the C-swap and the Δ-swap do not add up, LOG entry *GPU session closed…*, result 1).
  The Pile, present-share and seed-agreement numbers already exist for S = all.

**2. Minimal test set for the main text** [concluded, a proposal]. All tests use S = all and two readouts: the training probe, plus Yes/No 1 as a transfer check.
1. *Clamped C-swap and clamped Δ-swap* (each part taken from the twin while the other part is held at its clean value). The two numbers add up to the site swap, so together they say how the truth value is shared between C and Δ. This one test covers "C sufficient" and "Δ uninformative". It replaces the removals and the unclamped swaps.
2. *Write-span swap of C* next to the C-swap, plus one random-subspace control number. This is the behavioural re-implementation test, anchored in the original model. It is the only test that separates plain from a real decomposition without a threshold.
3. *Other qualities under the C-swap* (word-count KL against the site's 0.079). This is your "flip the verdict with the other qualities unchanged" criterion. Its power is limited: on the active sets it caught plain s1 (0.113) but not plain s2 (0.076). With S = all it gives plain s1 0.135 and ortho 0.027–0.032.
4. *Off-target cost: C removed on Pile text,* as a share of the site removal. This is where recon's breadth shows (0.19 against ortho's 0.03).
5. *Seed agreement* cos(δC_i, δC_j). It measures identifiability and answers the seed/mode question without matching components.
6. *One CI number* (the active set's or the CI-weighted set's clamped share of the C-swap). Main text only if the paper says anything about what tPD's CI identifies; otherwise appendix.
Plus a *baseline row* in tables 1–2: the twin-difference PCA part with C's size (96 directions per layer). It already has a P-swap of 0.88, against 0.26–0.58 for ortho's C.
Plus *one per-family breakdown* of test 1, because the affirmative fact families behave differently from negations and comparisons.

To the appendix, with a sentence each on why:
- removal (zero / mean): misleading about the rest, which carries the value although removing P leaves little separation;
- twin average: about half the swap, adds nothing;
- fixed shift: in the 4e work an untrained data-aligned basis did it as well, so it does not discriminate;
- unclamped swaps;
- read-span and both-span swaps: the faithful method fails them, so they are not a criterion;
- the input-PCA control;
- κ and the present share: internal companions of test 2, one sentence in the main text;
- weight-space ⟨P, W⟩, raw-write Grams and subspace overlaps.

Proposed next step [assumed]: one GPU session for the missing S = all cells in (d), together with the CI-weighted P for the eight decompositions (test 6, and it settles your reason 1). After that, I would rewrite the report's result sections on S = all. Shall I prepare the job list?

C: Is there any interesting result hiding in the other quality readouts?

Answer (26-11-16, Claude Opus 5.5): Yes, three things, and one of them changes test 3 of the minimal set above. Context: this LOG, the generated tables, and a new analysis of the per-statement arrays (`~/spd_out/26-11-15_open_research/analysis/other_qualities.py` → `other_qualities.md`). Every number below was read from that output or the per-statement arrays in this session. Scope: twin rows only (906 rows = 453 pairs, test split, one run per cell, 95% bootstrap over pairs). Humor, emotion, surprise and object recall exist only for the active-set P/R and the site (batch A). The S = all runs read only the verdict, Yes/No 1, word count and subject recall.

Method. Per readout I reduce the answer to one number m: the mean digit for the digit questions, p_any for recall, the 1−0 score for the truth prompts. Three quantities per readout:
- *dependence*: how far apart m is for true and for false statements in the original model, in standard deviations of m.
- *transfer* ν_q: the regression slope of the swap's change on the original twin difference (for the verdict it reproduces the tables' ν within 0.01).
- *bias*: the mean change. Every pair runs in both directions, so a pure transfer has mean 0. A non-zero bias means the readout moves the same way whichever twin is swapped in, which is damage, not transfer.

**1. Most "other qualities" depend on truth in the original model** [verified: per-statement arrays, site file]. Dependence: emotion 1.16 sd, surprise 0.32, subject recall 0.32, object recall 0.27, humor 0.17, word count 0.16 (the verdict: 1.82).
- Emotion is in effect a second 1/0 truth readout. Its answers are only "0" or "1" (argmax over 906 rows). True statements get 0.64 on average, false ones 0.34. Its twin difference correlates 0.68 with the verdict's. My guess: the question has the training prompt's form, so the 1/0 answer leaks into it [assumed].
- Word count: in larger_than, true statements are estimated at 6.28 words and false ones at 5.28, although twins have equal token length.
- So a perfect truth swap *should* move these readouts, and the site swap does: transfer ν_q of 0.81 for emotion, 0.69 surprise, 0.71 humor, 0.56 subject recall, 0.48 word count, 0.39 object recall. The draft's criterion "other qualities unchanged" does not hold for them. The digit KL in our side-effect tables mixes legitimate transfer with damage [concluded].

**2. The bias separates plain cleanly; the KL did not** [verified numbers; reading concluded]. Mean shift under the swap, with the site swap as reference:

| | word count (digits; sd of the rating 0.91) | subject recall (p, original 0.85) | object recall (p, original 0.94) |
|---|---|---|---|
| site | +0.08 [+0.06, +0.11] | +0.01 | +0.00 |
| plain P / R (active) | −0.38 / −0.17 (s1), −0.32 / −0.39 (s2) | −0.03 to −0.08 | −0.10 / −0.39 (s1), −0.17 / −0.45 (s2) |
| plain C / Δ (S = all) | −0.69 / −0.66 (s1), Δ −0.25 (s2) | −0.10 to −0.12 | not run |
| recon, all splits | −0.03 to +0.01 | +0.01 | −0.02 to +0.01 |
| ortho P / R (active) | −0.10 to +0.04 / −0.18 to +0.01 | −0.04 to +0.01 | −0.09 to +0.01 |
| ortho C / Δ (S = all) | −0.18, −0.00, −0.11 / −0.24, −0.62, −0.33, −0.22 | −0.03 to +0.01 | not run |

- Plain's swaps lower the word-count estimate and the probability of recalling the statement's own object, whichever twin is swapped in. In my earlier answer, test 3's word-count KL caught plain s1 (0.113 against the site's 0.079) but not plain s2 (0.076). The bias catches both, with intervals far from the site's.
- Recon shows no damage.
- With S = all, ortho's Δ-swap also lowers the word count (−0.22 to −0.62), while its recall stays clean. That fits the present share of 0.27–0.29: the Δ-swap injects the part of C's twin difference that the model does not have.

**3. The components transfer the verdict more than the other truth readouts** [verified numbers; reading concluded]. Relative transfer = the part's ν_q divided by the site's ν_q (1 = as much as the whole sites), active sets:

| | verdict | Yes/No 1 | emotion | surprise | humor |
|---|---|---|---|---|---|
| plain P (s1 / s2) | 0.98 / 0.91 | 1.02 / 0.92 | 0.93 / 0.83 | 0.58 / 0.65 | 0.61 / 0.66 |
| recon P (s1 / s2) | 0.86 / 0.88 | 0.70 / 0.71 | 0.48 / 0.53 | 0.46 / 0.51 | 0.38 / 0.41 |
| ortho P (s1–s4) | 0.37–0.74 | 0.20–0.31 | 0.07–0.19 | 0.16–0.32 | 0.11–0.15 |
| ortho R (s1–s4) | 0.48–0.75 | 0.71–0.81 | 0.56–0.88 | 0.81–1.03 | 0.79–1.10 |

- For recon and ortho, the components' share falls from the training readout to Yes/No to emotion. What they carry is partly specific to the training prompt's answer, not only the truth value that the model's other answers share. Ortho's R carries the other readouts at near-site level.
- Plain's P transfers almost uniformly. So readout transfer does not detect re-implementation; it rates the re-implementer best. This is the same pattern as lesson L4 in the paper-writing topic: plain's failure shows only in tests anchored in the original model.
- Caveat: surprise and humor are only weakly truth-related (their twin difference correlates 0.33 and 0.20 with the verdict's). Their transfer also measures content (the object differs between twins), and R holds most of the layer output. Emotion (0.68) is the clean case.

OUTDATED (26-11-16): the reading of point 3 is wrong. The pooled gap between verdict and emotion is mostly a difference in which statement families each readout weights (emotion: 68% cities; verdict: 18% cities, 41% larger_than), not something "specific to the training prompt's answer". Within a family the components transfer the two readouts about equally. See the answer beneath Julian's last comment at the bottom of this file (*Mostly no…*, part B2). The table's numbers are correct as pooled numbers.

**4. Side finding: under a near-linear readout, plain passes the P/R test** [verified numbers; reading concluded]. Under emotion, the unclamped P- and R-swaps add up to 0.60–0.77 against the site's 0.81 in all eight decompositions. Under the verdict they add up to 1.06–1.50 against 0.95.
- Both readouts see the same intervened states at the statement tokens: both prompts start with "\n", and the swap acts only there. So the excess under the verdict comes at least partly from how the verdict prompt reads the state (saturation), not only from the network recomputing the other part. The clamped swaps showed that recomputation contributes too (entry *GPU session closed…*, result 1).
- Consequence for plain: under emotion its P-swap is 0.75 and its R-swap 0.02, which is the ideal pattern ("P carries it, R is uninformative"). Its R-swap is silent because R = y − p, so the swap brings in the model's twin difference minus plain's injected one, and on a linear readout the two cancel. So a silent R does not rule out re-implementation; only the write-span swap does [concluded].

OUTDATED (26-11-16): point 4's explanations are not supported. "Additive under emotion, super-additive under the verdict" is again a family effect (in larger_than both readouts are super-additive, in cities neither is), so it shows no saturation of the verdict prompt; and plain's silent R under emotion is plain's weak R in cities, not a cancellation on a linear readout. The last sentence survives as an example only. See the answer beneath Julian's last comment at the bottom of this file, part B3.

**Consequences for the minimal test set** [assumed: proposals]:
- Test 3 becomes: the *bias* of word count and subject/object recall under the clamped C- and Δ-swaps, with intervals, against the site's bias. It replaces the digit KL.
- Emotion moves from "other qualities" to the truth readouts, as a second, near-linear one. It gives the transfer profile of point 3 and an additive P/R split.
- The paper should say that humor, surprise and recall depend on truth in the original model, and by how much.
- For the S = all GPU session: add `digit:emotion` and `recall:object` to the C/Δ jobs (none of the S = all runs has them). This adds only readout passes, no new interventions.

OUTDATED (26-11-16): the second and third bullet are withdrawn (emotion as a main-text truth readout; its "transfer profile"). The first bullet stands with refinements (bias read against the site's, per family group, only on readouts that are not truth readouts). `recall:object` stays in the GPU jobs, `digit:emotion` becomes optional. See the answers beneath Julian's two comments below.

C: Let me go through this again. I liked the minimal test set above because for me it seemed that the set covered showed something about each decomposition.

Answer (26-11-16, Claude Fable 5.2): The set stands as proposed. Of everything in the other-quality answer, one thing changes it: the statistic of test 3 (a mean shift instead of the digit KL). The other changes I proposed there I withdraw (answer below). Your reading is right, each method has a test it fails. With the cells that exist (S = all = all 96 components per layer as C against the catch-all Δ; ν = share of the twin's verdict that a swap brings over, site swap 0.93; numbers re-read from `analysis/score.md` and `tables.md`, tests 4–5 from the S = all table in the first answer above):

| test | plain | recon | ortho |
|---|---|---|---|
| 1. clamped C-swap / Δ-swap | not run with S = all; active set's own share 0.65–0.70 | not run; active set 0.75–0.81 | not run; active set 0.12–0.34, **the rest carries most of the value** |
| 2. swap of the model's own output in C's write span, next to the C-swap | **0.02 against 0.74: re-implementation** | not run with S = all; active set passes (0.80–0.83 against 0.79–0.81) | passes (0.30–0.40 against 0.26–0.58) |
| 3. damage to readouts that should not shift | **flagged in both seeds** (comparisons) | clean except word count in cities (below) | Δ-swap lowers the word count |
| 4. C removed on Pile text, share of the site removal | 0.11 | **0.19: broad** | 0.03 |
| 5. agreement between seeds | **0.02: unrelated** | 0.82 | **0.07–0.11** |

So plain fails 2, 3 and 5, recon fails 4, ortho fails 1 and 5. Test 1 is the cell that still needs the GPU session for every method.

C: What you just said for me souds like another result. The idea of a clear distinction of the mechanisms for truth and other qualities is questionable. The obvious explanation is an intersection of their mechanisms. Now I didn't quite get if this gives us more insights about the behaviour of the decompositions themselves. Can you clarify this?

Answer (26-11-16, Claude Fable 5.2) — *Mostly no: one decomposition insight (damage), two withdrawn after a per-family check*. Context: this LOG in full, the SUMMARY, `analysis/other_qualities.md`, `score.md`, `family.md`, and a new analysis of the per-statement arrays made for this answer: `~/spd_out/26-11-15_open_research/analysis/transfer_profile_check.py` (→ `transfer_profile_check.md`: both readouts on one scale, transfer per family, bias per family). Every number below was read from those outputs in this session. Twin rows only (906 rows, test split, one run per cell); the per-family numbers have no intervals.

Short version. Yesterday's answer mixed three kinds of statement. Sorted:
- about the **model and our criterion**, not about the decompositions: point 1 (the other qualities depend on truth). This is your "another result".
- about the **decompositions**: point 2 (the bias as a damage measure). It stands, with refinements.
- presented as decomposition insights but **wrong as read**: point 3 (components carry something specific to the training prompt) and the explanations in point 4. I checked both today because your question hinges on them. They are effects of which statement families each readout weights. Marked `OUTDATED` above.

**A. Your "intersection" reading** (model and criterion).

What the data supports [concluded]:
- The truth-dependence of these readouts passes through the same seven sites: the site swap brings over 0.39–0.81 of their twin differences.
- New today: within a family, a part that brings over a share of the verdict brings over about the same share of emotion, in the two families where emotion depends on truth most (cities, larger_than; table in B2). So at these sites the two prompts read the same thing, whatever part of the layers we cut out. That is what an intersection predicts.

What it does not show [assumed either way]:
- That the shared thing is *truth*. Twins differ in the object ("… is in France / Germany"), so a swap moves content along with the truth value. A judgement can depend on the content directly.
- That the mechanisms intersect rather than run in series (the truth value is computed once, and a later judgement such as "surprising" takes it as an input). For our tests the two cases have the same consequence.
- For emotion there is a cheaper explanation: the prompt has the training prompt's form and its answers are only 0 or 1, so it may partly be answered as a truth question.
- The dependence is uneven across families: 68% of emotion's squared twin difference sits in cities and 19% in larger_than (verdict: 18% and 41%).

Consequence, and this is the part that concerns decompositions in general, not our three [concluded: follows from the dependence, whichever explanation holds]: if these readouts depend on the truth value in the unsplit model, then *no* decomposition can pass "flip the verdict, other qualities unchanged", a perfect one included. The criterion has to become: under a swap of the truth part, another readout moves as far as it moves between twins in the unsplit model (reference: the site swap), and not otherwise. "Not otherwise" is the bias of B1. I agree this is a result. I would put it in the paper as part of the test design (the auxiliary result on techniques), in one paragraph, and not as a claim about intersecting mechanisms, which one model with in-distribution twins cannot carry.  
C: I have a question. If we do activation swap between twins then obviously we swap any surpise. Do I have the wrong experiment in mind?

Answer (26-11-16, Claude Fable 5.2) — *right experiment; the site swap moving surprise is expected*. Context: as in the answer above, plus the twin definition in `paper_tests.py` (`twin_partner`), example statements from `data/tiu/*/v3/test.jsonl`, and a new analysis `~/spd_out/26-11-15_open_research/analysis/truth_linked_transfer.py` (→ `truth_linked_transfer.md`), read in this session.

You have the right experiment in mind, and the objection holds against my first bullet under "What the data supports" above. Twins are two different statements: same subject, another object ("The city of Lu'an is in China." / "… is in Turkey."; "Fifty-four is larger than fifty-two." and the same pair in the other order). Swapping the layer outputs between them brings over whatever these layers hold about the other statement, including whatever makes it more or less surprising. So "the site swap brings over 0.39–0.81 of the other readouts' twin differences" says nothing about truth mechanisms. I should not have listed it as support for an intersection.

What is not obvious, and what the tests can use:
- *How much comes over.* The swap replaces only the MLP outputs of seven layers at the statement tokens. The tokens themselves, attention and the other layers stay with the original statement. Pooled, the site swap brings over 0.95 of the verdict's twin difference but 0.69 of surprise's, 0.48 of the word count's and 0.39 of object recall's.
- *The question is about a part, not the site.* For the whole site nothing should stay behind. For a part that is meant to hold the truth value, the question is which portion of another readout's twin difference it should bring over. I split each twin difference into a truth-linked portion (the average gap between the true and the false twin within a family) and a pair-specific rest (this pair's content). For the verdict the truth-linked portion is 92–99% of the squared twin difference in every family. For surprise it is 41% in cities, 27% in neg_cities, 9% in larger_than and 1% in smaller_than: most of what a swap moves in surprise is simply "another statement", as you say.
- *The bias is untouched by this.* It is the mean change over both swap directions, so anything that is brought over from the twin cancels, whether truth or content. That is why it works as a damage measure.

The answer beneath your next comment uses the split.

**B. What it says about the decompositions.**

*B1. Damage (point 2) stands: plain's swaps break things that a swap should leave alone* [verified numbers; reading concluded]. Bias = mean change of a readout under the swap. Every pair runs in both directions, so bringing over the twin's value cancels out and what remains is a shift in one direction. It is read against the site swap's bias. Three refinements from today:
- *Where.* Plain's damage is in the number comparisons. Word-count estimate (original mean 6.00 words, sd 0.91) under plain's P-swap: −0.44 to −0.82 words in larger_than and smaller_than (site +0.11 and −0.07), that is 0.5–0.9 sd. Probability of recalling the statement's own object (original 0.94): −0.18 to −0.53 under the P-swap and −0.69 to −0.94 under the R-swap in the comparisons (site 0.00). In cities plain does less: word count −0.09 to −0.20 under the P-swap (but −0.42 under seed 2's R-swap), object recall within 0.07 of zero under both swaps.
- *Recon is not entirely clean.* Its P-swap lowers the word count in cities by 0.16–0.20 words (site +0.05), about as much as plain's P-swap does there. The pooled −0.03 I reported hid this, because recon follows the site's +0.11 in larger_than. Its object recall stays within 0.06 of zero in every family, as the site's does. So object recall separates plain from recon more cleanly than word count does.
- *Only on readouts that are not truth readouts.* Under the truth readouts the faithful parts are the biased ones: recon's P-swap shifts the verdict score by −0.65 / −0.75 (0.13–0.15 of its sd of 4.82) and the Yes/No score by −2.9 / −2.8 (0.33–0.35 of its sd of 8.35), the twin-difference PCA baseline by similar amounts, plain's P-swap by +0.3 / 0.0 and +0.7 / +0.6. My guess: a state mixed from two statements reads as "false" [assumed]. So a bias in a truth readout is not damage.

What this adds about plain: its failure showed so far only in the write-span swap, which needs the original weights. The bias shows it from outputs alone, and locates it.

*B2. Point 3 withdrawn: the components do not carry something specific to the training prompt* [verified numbers; reading concluded for cities and larger_than, open for the negated families]. I had read "recon's and ortho's components bring over more of the verdict than of emotion" as readout-specific content. Two checks:
- Same scale for both readouts (the logit difference between the answers "1" and "0"): the pooled gap stays (recon P 0.87–0.88 of the site's verdict transfer against 0.51–0.56 of its emotion transfer; ortho P 0.37–0.74 against 0.10–0.22). So it is not a scale artefact.
- Per family, it largely disappears. Relative transfer (part / site), verdict / emotion:

| | cities (230 rows) | larger_than (204) | smaller_than (194) | neg_cities (210) |
|---|---|---|---|---|
| share of the squared twin difference, verdict / emotion | 0.18 / 0.68 | 0.41 / 0.19 | 0.19 / 0.05 | 0.18 / 0.04 |
| site swap (absolute ν) | 0.88 / 0.81 | 1.00 / 0.95 | 0.97 / 0.95 | 0.90 / 0.55 |
| plain P, s1 | 0.97 / 0.95 | 0.99 / 0.90 | 1.16 / 0.80 | 0.79 / 0.61 |
| plain P, s2 | 0.82 / 0.72 | 0.91 / 0.97 | 1.10 / 1.79 | 0.80 / 0.46 |
| recon P, s1 | 0.42 / 0.34 | 0.97 / 0.96 | 1.06 / 1.08 | 0.86 / 0.29 |
| recon P, s2 | 0.48 / 0.42 | 0.97 / 0.93 | 1.08 / 1.06 | 0.88 / 0.44 |
| ortho P, s1 | 0.09 / −0.00 | 0.60 / 0.47 | 0.31 / 0.08 | 0.18 / −0.25 |
| ortho P, s2 | 0.06 / 0.05 | 0.82 / 0.62 | 1.08 / 0.92 | 0.88 / −0.03 |
| ortho P, s3 | 0.04 / 0.03 | 0.27 / 0.30 | 0.81 / 0.68 | 0.65 / 0.20 |
| ortho P, s4 | 0.07 / 0.03 | 0.63 / 0.63 | 0.47 / 0.27 | 0.61 / 0.18 |
| plain R, s1 | 0.25 / −0.04 | 0.77 / 0.70 | 0.30 / 0.37 | 0.88 / 0.27 |
| plain R, s2 | 0.28 / −0.15 | 0.62 / 0.55 | 0.43 / 0.28 | 0.81 / 0.71 |

  In cities and larger_than, which hold 87% of emotion's weight, the two numbers differ by at most 0.20 for every P, and by at most 0.08 for recon. The pooled gap arises because the pooled ν weights each row by that readout's own squared twin difference: emotion's weight is in cities, where recon's and ortho's components carry little (0.42–0.48 and 0.04–0.09 of the site's verdict transfer); the verdict's weight is in the comparisons, where they carry a lot.
- Left over: in neg_cities emotion is brought over much less than the verdict (recon 0.29–0.44 against 0.86–0.88). Emotion is a weak truth readout there (4% of its weight; even the site brings over only 0.55), so I cannot read this either way.

What remains as a decomposition statement is the family coverage, which does not need the other qualities: with any truth readout, the components carry the truth value for comparisons and much less for affirmative facts. The report says this for ortho. It holds for recon too, though recon's active set is 70–79% of the output's energy: its P-swap is 0.37 / 0.42 in cities against the site's 0.88, and 0.97 in larger_than (`family.md`).  
OUTDATED (26-11-16): the family coverage is a property of the unclamped swaps. With all components and the rest held at its clean value, recon's C carries 0.79 of the verdict on affirmative facts (0.82–0.83 in cities), 0.95–0.96 on negations and 0.86–0.89 on comparisons; ortho's C carries little in every group. See the entry *GPU session 2 results…*, test 1.
C: This is a continuation of my comment above. So have we established already that everything that carries truth verdict also has to carry surprise verdict?

Answer (26-11-16, Claude Fable 5.2) — *No: not "has to", and the readouts fall into two groups*. Context and source as in the answer beneath your previous comment (`truth_linked_transfer.md`; active sets P/R and the site, unclamped swap, 906 twin rows, one run per cell; 95% intervals by bootstrap over pairs). The table above did not establish it: it compared the verdict with emotion only, in two families, without separating truth from content. Checked now.

Method. Per family and readout: a = the gap between the true and the false twin in the original model (the truth-linked portion); τ = the share of that gap a swap brings over. Cells where a is not distinguishable from 0 are left out.

**1. The gaps themselves: some follow the truth value, some follow the content** [verified numbers; reading concluded for cities / neg_cities, one family pair]. The negated family separates the two, because it uses the same cities and countries: "The city of Lu'an is not in Turkey." is true and names the wrong country. Gap a in sd of the readout:

| readout | cities | neg_cities | larger_than | smaller_than | follows |
|---|---|---|---|---|---|
| verdict | +1.55 | +1.62 | +2.55 | +1.79 | truth |
| emotion | +2.69 | −0.01 (none) | +1.19 | +0.80 | truth, in affirmative statements only |
| surprise | +0.39 | +0.38 | +0.34 | none | truth (small) |
| humor | +0.41 | −0.79 | +0.56 | +0.47 | content: the sign flips with "not" |
| word count | +0.60 | −0.99 | +1.10 | −0.12 | content: flips |
| subject recall | −0.76 | +0.89 | −0.96 | −0.66 | content: flips |

A flipped sign means the readout responds to whether the named country is the city's own, not to whether the statement is true. So for word count, humor and subject recall the dependence I called "depends on truth" is a dependence on the content, at least in the city families. This weakens the intersection reading for them and leaves it open for emotion and surprise.

**2. What the parts bring over** [verified numbers; reading concluded]. τ for the verdict next to τ for the other readout, P = active set:

| | cities: verdict / emotion / surprise | larger_than: verdict / emotion | smaller_than: verdict / emotion | neg_cities: verdict / surprise / humor / word count |
|---|---|---|---|---|
| site | 0.88 / 0.81 / 1.17 | 1.00 / 0.99 | 0.97 / 1.01 | 0.89 / 1.45 / 1.01 / 0.47 |
| plain P, s1 | 0.85 / 0.79 / 1.01 | 1.00 / 1.03 | 1.14 / 0.74 | 0.71 / 1.23 / −0.51 / −0.97 |
| plain P, s2 | 0.71 / 0.58 / 0.65 | 0.91 / 1.17 | 1.08 / 1.60 | 0.73 / 0.91 / 0.19 / −0.11 |
| recon P, s1 | 0.37 / 0.28 / 0.39 | 0.97 / 1.03 | 1.03 / 1.05 | 0.77 / 1.60 / −0.26 / 0.10 |
| recon P, s2 | 0.42 / 0.34 / 0.47 | 0.97 / 0.99 | 1.05 / 1.01 | 0.79 / 1.59 / −0.12 / 0.03 |
| ortho P, s1 | 0.08 / −0.00 / −0.03 | 0.61 / 0.45 | 0.31 / 0.18 | 0.17 / 0.05 / −0.11 / −0.08 |
| ortho P, s2 | 0.06 / 0.02 / 0.01 | 0.82 / 0.71 | 1.06 / 0.95 | 0.79 / 0.88 / −0.33 / −0.74 |
| ortho P, s3 | 0.03 / 0.03 / 0.26 | 0.27 / 0.27 | 0.79 / 0.70 | 0.64 / 1.16 / −0.00 / −0.18 |
| ortho P, s4 | 0.06 / −0.01 / −0.23 | 0.63 / 0.74 | 0.47 / 0.35 | 0.53 / 0.73 / −0.23 / −0.10 |
| recon R, s1 / s2 | 0.35, 0.24 / 0.19, 0.05 / 0.58, 0.21 | 0.53, 0.50 / 0.42, 0.50 | 0.36, 0.27 / 0.69, 0.48 | 0.33, 0.22 / 0.62, 0.49 / 0.76, 0.87 / 0.49, 0.14 |

(Surprise's intervals are wide: in cities ±0.1 to ±0.2, in neg_cities ±0.3 to ±0.6. In larger_than its gap is too small to measure a share, so it is left out.)

- *Readouts that follow truth go with the verdict.* Where emotion or surprise has a gap, a part brings over about the share of it that it brings over of the verdict: recon P in cities 0.37–0.42 / 0.28–0.34 / 0.39–0.47, in the comparisons about 1.0 for both; ortho's seeds keep their order (in smaller_than 0.31, 1.06, 0.79, 0.47 for the verdict and 0.18, 0.95, 0.70, 0.35 for emotion). So for these two, in this data, what carries the verdict carries them as well. That is an observation over four methods' parts, not a necessity.
- *Readouts that follow content do not.* In neg_cities recon's active set brings over 0.77–0.79 of the verdict and −0.26 to 0.10 of the humor and word-count gaps; its rest brings over those (0.76–0.87 of humor). Ortho s2–s4 the same way. So the verdict can be carried without them: "has to" is false for these readouts.

**3. One thing this says about the decompositions** [concluded for recon; assumed for the plain observation]. In the negated cities, recon's (and ortho's) components carry the truth value and leave the content-linked gaps to the rest. This is the first place where a part separates truth from content, and affirmative twins cannot show it, because there truth and "right country named" coincide. Plain s1 does something else: its P-swap moves the word count by −0.97 of the gap, that is, against the twin and in the direction that "true" has in the affirmative cities. My guess is that its components inject a truth signal of the affirmative kind [assumed; one seed, s2 shows −0.11].

OUTDATED (26-11-16): "(and ortho's)" holds for the active sets only. With all components (C against Δ), only ortho s2 shows it; the C of s1 and s3 brings over the word-count gap and little of the verdict. For recon the C-swap with all components was never run. See the entry *S = all version of the negated-cities test* at the bottom of this file.

Limits: the split is post hoc; the truth-versus-content separation rests on one family pair (cities / neg_cities); active sets only, unclamped swaps; no S = all version.

For the tests: the bias (test 3) needs none of this. The transfer of other readouts is not a pass/fail criterion; I would report it only for the negated families, in the appendix, as "carries the verdict without the content-linked readouts".

*B3. Point 4's explanations withdrawn* [verified numbers; reading concluded]. I wrote that the unclamped P- and R-swaps add up under emotion but exceed the site swap under the verdict, and took that as saturation of the verdict prompt. Per family it is the family, not the readout: in larger_than both readouts are super-additive (recon s1, P + R relative to the site: verdict 0.97 + 0.53, emotion 0.96 + 0.45), in cities neither is (0.42 + 0.40; 0.34 + 0.34). Plain's "ideal pattern" under emotion (P 0.75, R 0.02) is its weak R in cities (rows *plain R* above); in larger_than its R-swap brings over 0.55–0.70 of the site's emotion transfer. The sentence "a silent R does not rule out re-implementation" survives as an example (plain in cities), not as a mechanism.

**So, to your question.** The other-quality readouts give one thing about the decompositions: a second, output-only sign of plain's re-implementation (damage to word count and object recall, in the comparisons), and a weaker version of it for ortho's Δ-swap. They give one thing about the tests: the "other qualities unchanged" criterion must be replaced by "moves as between twins, and no further". They do not show that the components carry something verdict-specific.

**For the minimal set** [assumed: proposals]:
- Test 3: bias of object recall and word count under the clamped C- and Δ-swaps, against the site's bias, with intervals. Object recall first.
- Tests 1 and 3 by family group (affirmative facts, negated statements, comparisons) in the main text, not only pooled. Three of yesterday's pooled readings (the transfer profile, the additivity, "recon shows no damage") came apart per family. This replaces the single "per-family breakdown" I listed.
- Emotion is not needed in the main text. GPU jobs: add `recall:object` to the C/Δ runs; `digit:emotion` optional.

Not done here: intervals for the per-family numbers; S = all versions of any of this (the S = all runs have neither emotion nor object recall).

C: can you write a summary of the new minimal tests including these you just proposed?

Answer (26-11-16, Claude Fable 5.2) — *the minimal test set, consolidated*. This replaces the lists in my earlier answers; it contains the two changes proposed above (test 3 measures a shift instead of a KL; tests 1 and 3 are reported per family group) and nothing else new. All of it is a proposal [concluded: reasons given per test; not ratified]. Existing numbers were re-read from `analysis/score.md`, `tables.md` and `other_qualities.md` in this session; those of tests 4 and 5 are taken from the S = all table in my first answer above.

**Common setup.**
- *Split.* Each decomposition splits the output of the seven decomposed layers (the MLP down-projections of layers 11–17) into C, the sum of all 96 components per layer at full strength, and Δ, the remainder. C has 672 components in every method, so the methods are compared at equal size.
- *Data.* Test split, 453 twin pairs: two statements with the same subject, another object, opposite truth value and equal token length. Interventions act at the statement tokens.
- *Swap.* A part's write is replaced by the write it has on the twin. ν is the share of the twin's verdict difference that this brings over (0 = verdict unchanged, 1 = the twin's verdict). Reference: swapping the whole layer outputs ("site swap") gives 0.93.
- *Clamped.* While one part is swapped, the other is held at the value it has on the original statement, so it cannot follow the swapped part downstream. Only then do the two parts' effects add up to the site swap (within 0.04 on the active sets).
- *Readouts.* The verdict under the training prompt ("The amount of truth in this statement is" → 1 or 0), and a Yes/No question as a check that the result is not specific to that prompt.

**The tests.**

| # | test | what is measured | a perfect decomposition | what we have | status |
|---|---|---|---|---|---|
| 1 | Where is the truth value? Clamped C-swap and clamped Δ-swap | ν of each | C near the site swap, Δ near 0 | active sets: own share 0.65–0.70 plain, 0.75–0.81 recon, 0.12–0.34 ortho. Unclamped Δ-swap 0.23–0.67 in all eight | to run for all eight |
| 2 | Is C a part of the model's output? Swap only the original layer output inside the span of C's write directions, next to the C-swap; a random subspace of the same size as control | ν of each | about equal; random near 0 | plain s1 0.02 against 0.74 (fails); ortho s1–s3 0.30–0.40 against 0.26–0.58 (passes); random 0.01 | to run: plain s2, recon s1/s2, ortho s4 |
| 3 | Does the swap damage anything? Mean change of object recall and of the word-count estimate under the clamped C- and Δ-swaps | shift, with interval, next to the site swap's shift | as the site swap (near 0) | active sets, unclamped: plain lowers object recall by 0.10–0.45 from 0.94 and the word count by 0.17–0.39 words; recon within 0.03 on both; site 0.00 and +0.08 | to run for all eight (object recall is missing in every S = all run) |
| 4 | Off-target cost: C removed at every position of ordinary text (Pile) | loss in next-token prediction as a share of removing the whole layers | near 0 | plain 0.11, recon 0.19, ortho 0.03 | exists |
| 5 | Do two seeds find the same thing? Cosine between two seeds' C-writes, on the difference between twins | cosine | near 1 | plain 0.02, recon 0.82, ortho 0.07–0.11 | exists |
| 6 | (optional) What does the causal importance pick out? The same as test 1 for the components weighted by their causal importance per token | share of the C-swap | – | nothing | to run; main text only if the paper makes a claim about causal importance |

Why each test is in the set:
- Test 1 replaces the removal tests. Removing a part suggested "the components are necessary"; the swaps show that the rest carries the value too.
- Test 2 is the only test that separates a re-implementation from a decomposition without a threshold. A truth readout alone rates plain best.
- Test 3 measures a shift, not a KL. Each pair is swapped in both directions, so whatever is brought over from the twin cancels and only one-directional damage remains. It is read on readouts that are not truth readouts. Object recall is the cleaner one of the two.
- Test 4 is where "targeted" is tested: recon's C is most of the layer's output on the statements, and the truth tests cannot show that.
- Test 5 needs no matching of components between seeds.

**Reported with them.**
- *Baseline row* in tests 1–4: the top 96 principal directions per layer of the layer output's difference between train twins. It is a part of the model by construction and already has an unclamped swap of 0.88.
- *Per family group* for tests 1 and 3: affirmative facts (cities, element symbols, animal classes), negated statements, number comparisons. The components carry the truth value mostly for comparisons and negations, and plain's damage sits in the comparisons; pooled numbers hid both.

**Which method fails what** (as far as measured): plain 2, 3, 5; recon 4; ortho 1, 5.

**Moved to the appendix,** each with one sentence on why: removals; twin average; fixed shift; unclamped swaps; read-span and both-span swaps; the input-PCA control; the internal numbers behind test 2 (cancellation ratio, present share); weight-space similarity; the transfer of emotion, surprise, humor and word count, including the negated-cities observation from the answer above.

**To run** (one GPU session; job list not written yet): the clamped C- and Δ-swaps with object recall and word count added as readouts, for all eight decompositions and the baseline; the C-swap and its write-span swap for plain s2, recon s1/s2 and ortho s4; optionally test 6.

---

26-11-16, Claude Fable 5.2 — *S = all version of the negated-cities test: what exists, and the tests added to the GPU list*

Context: this LOG in full, the outputs named below. No GPU was used; the existing per-statement arrays were re-analysed.

C: (transcribed from chat) Can you also add tests for S = all about the tests that support "In the negated cities, recon's (and ortho's) components carry the truth value and leave the content-linked gaps to the rest"?

**What the claim rested on.** Active sets only (P = active set, R = the rest), unclamped swaps, readouts humor and word count (answer *No: not "has to"…* above). S = all means C = all 96 components per layer at full strength against Δ, the remainder.

**What exists for S = all** [verified: `~/spd_out/26-11-15_open_research/analysis/truth_linked_transfer_all.py` → `truth_linked_transfer_all.md`, read in this session; unclamped swaps, 210 twin rows of neg_cities, 95% intervals by bootstrap over pairs]. The S = all runs have only two of the content-linked readouts, word count and subject recall; humor is missing. The C-swap exists for plain s1 and ortho s1–s3, the Δ-swap for all eight. τ = share of the gap between the true and the false twin that the swap brings over; the gap itself is +1.62 sd for the verdict, −0.99 sd for word count and +0.89 sd for subject recall (both flipped against the affirmative cities, so both follow the content).

| neg_cities | verdict | Yes/No | word count | subject recall |
|---|---|---|---|---|
| site swap | 0.89 | 1.13 | 0.47 | 0.41 |
| baseline (top 96 directions per layer of the output's twin difference) | 0.87 | 1.23 | −0.06 | 0.25 |
| recon Δ, s1 / s2 | −0.00 / −0.00 | 0.35 / 0.29 | 0.68 / 0.31 | 0.07 / −0.08 |
| ortho C, s1 / s2 / s3 | 0.10 / 0.63 / 0.35 | 0.32 / 0.40 / 0.38 | 0.43 / −0.26 / 0.60 | 0.22 / 0.19 / −0.32 |
| ortho Δ, s1 / s2 / s3 / s4 | 0.80 / 0.10 / 0.29 / 0.54 | 1.12 / 0.61 / 0.97 / 0.88 | 0.21 / −0.28 / −0.54 / 0.07 | 0.77 / 0.36 / 0.22 / 0.62 |
| model's own output in the write span of ortho's C, s1 / s2 / s3 | 0.13 / 0.29 / 0.24 | 0.45 / 0.31 / 0.57 | 0.02 / −0.14 / −0.09 | −0.10 / −0.14 / −0.49 |
| plain C, s1 | 0.78 | 0.70 | −0.25 | 0.77 |
| plain Δ, s1 / s2 | 0.56 / 0.55 | 0.76 / 0.80 | 0.26 / −0.58 | 0.26 / 0.08 |

Reading [concluded: from this table; one run per cell]:
- **Recon: half supported, half untested.** Its Δ brings over none of the verdict (0.00 in both seeds, interval ±0.02) and 0.68 / 0.31 of the word-count gap, against 0.47 for the whole site. That is the Δ side of the claim. The C side cannot be checked: recon's C-swap was not run. Two qualifications. Under the Yes/No question recon's Δ does bring over 0.29–0.35 of the truth gap, so "Δ carries no truth value" holds for the training prompt only. And Δ brings over none of the subject-recall gap (0.07 / −0.08), so that content-linked gap is not "left to the rest".
- **Ortho: the claim does not carry over to all components.** Only s2's C brings over the verdict (0.63) without the word-count gap (−0.26). The C of s1 and s3 does the opposite: 0.43 and 0.60 of the word-count gap, at or above the site's 0.47, with 0.10 and 0.35 of the verdict. On the active sets these two seeds had −0.08 and −0.18 for word count, so the word-count gap is carried by their inactive components. The model's own output in the same write directions brings over none of it (0.02, −0.09): this part of C's write is not in the layer output, the same sign as plain's re-implementation in test 2, here for another readout [concluded from these two cells; no interval on the difference].
- **Plain s1** brings over the verdict (0.78) and nearly twice the site's share of the subject-recall gap (0.77 against 0.41). No separation.
- **The baseline shows the separation exists in the model.** A 96-direction part of the layer output carries the verdict (0.87) without the word-count gap (−0.06 [−0.15, 0.00]). So a decomposition could find such a part; of the cells we have, only ortho s2's C does.
  OUTDATED (26-11-16): withdrawn. Under the clamped swap the baseline's C brings over the content-linked gaps as the site does (humor 1.01, word count 0.44, subject recall 0.39 in neg_cities), so its unclamped separation came from the rest at later layers. Ortho s2's clamped C fails too (humor 0.22, subject recall 0.26). Of all parts, only recon's clamped C separates, for humor and word count. See the entry *GPU session 2 results…*, the negated-cities criterion.
- Limit of the word-count column: even the whole site brings over only 0.47 of that gap, so half of it does not pass through these layers at all.

**Tests added to the GPU list** [assumed: proposal; they extend the runs already planned for tests 1 and 3 of the minimal set, no new interventions].
- *Runs.* C-swap and Δ-swap, each unclamped and clamped, for all eight decompositions and for the baseline.
- *Readouts.* Verdict, Yes/No, and four other readouts: humor, word count, subject recall, object recall. Surprise and emotion only if time is left (they follow truth, so they do not test this claim).
- *Analysis.* `truth_linked_transfer_all.py` on the new files: per family the share τ of each readout's truth-linked gap, with intervals. The claim is read in neg_cities, with cities as the reference for which gaps flip. The other negated families are too small (30 and 4 rows).
- *Criterion, fixed before the data.* The claim holds for a decomposition if, in neg_cities under the clamped swaps, C brings over at least half of the verdict gap, and of the humor, word-count and subject-recall gaps a share whose interval includes 0 or lies below it, while Δ brings over the larger share of each of those three. Reference rows: the site and the baseline.
- *Predictions* [assumed]: recon's C passes for humor and word count and fails for subject recall (its Δ carries none of it); ortho passes in s2 only; plain fails.
- *Cost* [assumed: from the job times in `vm/progress` of the 26-11-15 session, about 12 s per intervention × readout pass]. 18 jobs (eight decompositions and the baseline, C and Δ) of 12 passes each come to roughly 45 min of H100, before instance setup and the four missing write-span jobs. The remaining credit ($4.40, about 1 h at $4.57/h) is therefore tight; dropping object recall from the Δ jobs or topping up are the two ways out.

Where this goes in the paper: appendix, as proposed, unless the new runs show a clean pass for recon; then it is a positive result about recon worth a sentence in the main text.

---

26-11-16, ~15:00 UTC, Claude Opus 5.5 — *GPU session 2: the S = all job list launched on a 48 GB RTX 4090*

Context: CLAUDE.md, INDEX.md, `docs/CODING.md`, the vast-rentals SUMMARY, this LOG in full, the 26-11-15 job scripts (`~/spd_out/26-11-15_open_research/scripts/`) and `paper_tests.py`'s `run`. Not read: the report, the analysis scripts' code.

C: (transcribed from chat) Can you run the tests given at the end of the research log? Vast.ai is up

**Machine and budget** [verified: `vastai show instance 60932116`, `vastai show user`, `nvidia-smi`, 14:50 UTC]. Instance 60932116, RTX 4090 48 GB, power limit 450 W, $0.69/h, repo at `c793a07` (the provision sync; only the convos files differ from HEAD there). Credit $13.48, so the budget worry of the previous entry (H100 at $4.57/h, $4.40 left) no longer applies.

**What runs** (scripts and job lists `~/spd_out/26-11-15_open_research/scripts2/`; outputs on the instance under `/root/spd_out/open2/`). All on the test split; S = all (C = all 96 components per layer, Δ = the rest):
- *Tests 1 and 3 of the minimal set, and the negated-cities criterion*: target `all`, parts P (= C) and R (= Δ), interventions `swap,swap_clamped`, readouts verdict, Yes/No (`yes_no:question`), humor, word count, subject recall, object recall; for all eight decompositions (16 jobs).
- *Baseline row*: the twin-difference PCA part with C's size (`diffspan --span_base all`, 96 directions per layer, fitted on train twins), P and R, same interventions and readouts (2 jobs).
- *Site reference* with the same six readouts on this machine (1 job, swap).
- *Test 2*: swap of the model's own output in the write span of all components for the four cells missing (plain s2, recon s1/s2, ortho s4); readouts verdict, Yes/No, object recall (4 jobs). The random-subspace control exists (0.01, ortho s1's sizes).
- Not included [concluded: agent's choice]: surprise and emotion (they follow truth, so they do not test the negated-cities claim; the previous entry made them optional), and test 6 (CI-weighted P), for which no code exists yet.

Ortho seed 4's checkpoint is only local; uploaded to `/root/spd_out/spd/s-22600dd8/` (sha256 equal to the local copy) [verified]. Smoke test (val, 2 statements per dataset, Δ of ortho s1, all six readouts, both interventions) exited 0 [verified: one run; NaN warnings from its single twin pair are expected]. Main list: two workers in parallel (37 GB of 48 GB at the start), seed 4's three jobs after them.

*Addendum, ~15:25 UTC (same session): stall and restart.* The two baseline jobs (`B_diff_all_P/R`) sat at 0% GPU for over 20 min in the CPU step that fits the twin-difference directions (`eigh` of seven 3584×3584 matrices), against ~3.5 min per job on the H100. Cause [verified: `/sys/fs/cgroup/cpu.max` = 23 CPUs, `nproc` 192, 251 threads per job, `cpu.stat` 123,000 s throttled]: torch started 96 threads per job on a container with a 23-CPU quota. Julian noticed the idle GPU and asked whether the CPU work could run on the laptop; it does not need to, the step takes seconds once the threads are capped, and splitting it off would mean shipping the matrices between two halves of one job. Fix: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1` in `scripts2/worker.sh`; workers and jobs killed by PID and the run restarted (one intermediate restart went out without the cap because the edit had failed, and was killed too). The killed jobs left no progress lines, so they rerun; the site job's result (exit 0) is kept. After the restart both jobs passed the step in under 2.5 min, GPU at 100% [verified: `nvidia-smi`, logs]. FEEDBACK.md entry of 26-11-16.

*Addendum, ~15:30 UTC: one job failed on a partial checkpoint.* `C_plain1_all_R` exited 1 at load (`PytorchStreamReader failed reading zip archive`) [verified: its log]. The two workers had started plain s1's C and Δ jobs at the same moment; both fetched the checkpoint from WandB into the same file, and the helper (`spd/utils/wandb_utils.py`, `download(exist_ok=True, replace=False)`) reuses a file that exists, even while another process is still writing it. A partial file cannot pass silently: `torch.load` refuses it. Fix: the remaining four checkpoints (recon s1/s2, ortho s2/s3) downloaded one after another before the workers reached them (`scripts2/predownload.py`; all seven files 333,524,341 bytes) and the failed job appended to `jobs_late.txt`.

---

26-11-16, ~16:20 UTC, Claude Opus 5.5 — *GPU session 2 results: clamped C/Δ swaps, all write-span cells, damage, the negated-cities criterion*

Context: as in the entry *GPU session 2*, plus all of its outputs and the tables generated from them. Every number below was read from `~/spd_out/26-11-15_open_research/analysis/session2.md` (written by `analysis/session2.py` from `vm2/test/`), test split, one run per cell, 95% intervals by bootstrap over pairs (1,000 draws).

**Session facts** [verified]. 23 jobs exited 0 (`vm2/progress`), the one failure (`C_plain1_all_R`, partial checkpoint, addendum above) included after its rerun. Outputs copied to `~/spd_out/26-11-15_open_research/vm2/`: 78 files, sha256 identical to the instance's (compared after sorting both lists the same way). Instance 60932116 destroyed after the copy check, before the analysis ("destroying instance 60932116."; `vastai show instances` empty). Credit $13.48 → $12.66.

**Twin rows differ by machine** [verified: row counts; cause concluded from the selection rule]. This session has 898 twin rows (449 pairs), the H100 session 906. Twins are kept only where the model's own verdicts on the two statements differ, and 4 near-tie pairs come out differently under the 4090's bf16 arithmetic. The numbers below are comparable among themselves; next to the 26-11-15 numbers they differ by this row set as well. Site swap here 0.94 (H100 0.93).

**Test 1: where is the truth value?** ν = share of the twin's verdict difference a swap brings over; "clamped" = the other part held at its value on the original statement.

| | C clamped | Δ clamped | sum (site 0.94) | Yes/No: C / Δ clamped (site 0.91) | C clamped by group: affirmative / negated / comparisons | unclamped C / Δ |
|---|---|---|---|---|---|---|
| baseline (96 dir./layer) | 0.94 | 0.01 | 0.96 | 0.90 / 0.01 | 0.85 / 0.88 / 1.00 | 0.89 / 0.03 |
| plain s1 / s2 | 0.66 / 0.63 | 0.32 / 0.35 | 0.99 / 0.98 | 0.72, 0.70 / 0.25, 0.26 | 0.72, 0.71 / 0.58, 0.64 / 0.67, 0.60 | 0.73, 0.73 / 0.56, 0.52 |
| recon s1 / s2 | 0.86 / 0.89 | 0.10 / 0.09 | 0.96 / 0.97 | 0.70, 0.73 / 0.19, 0.17 | 0.79, 0.79 / 0.95, 0.96 / 0.86, 0.89 | 0.88, 0.88 / 0.29, 0.26 |
| ortho s1 / s2 / s3 / s4 | 0.11 / 0.33 / 0.14 / 0.19 | 0.84 / 0.61 / 0.80 / 0.75 | 0.94–0.95 | 0.16, 0.21, 0.13, 0.20 / 0.70, 0.64, 0.73, 0.68 | affirmative 0.11–0.15; negated 0.04, 0.59, 0.21, 0.27; comparisons 0.13, 0.32, 0.13, 0.18 | 0.46, 0.61, 0.28, 0.39 / 0.69, 0.43, 0.47, 0.66 |

Reading [concluded: from the table]:
- Clamped, the two parts add up to the site swap: 0.94–0.99 against 0.94 under the verdict, 0.85–0.97 against 0.91 under Yes/No. So test 1's two numbers do split the site effect, as intended.
- **Recon is the best of the three methods:** C 0.86–0.89, Δ 0.09–0.10 (Yes/No 0.70–0.73 / 0.17–0.19). The baseline does better (0.94 / 0.01).
- **Ortho's C carries little of the truth value itself:** 0.11–0.33 clamped against 0.28–0.61 unclamped. Most of its unclamped C-swap effect is Δ at later layers following the swapped C writes. Δ carries 0.61–0.84.
- Plain: C 0.63–0.66, Δ 0.32–0.35.
- **Clamping changes the family picture.** Recon's C on affirmative facts: 0.50–0.52 unclamped, 0.79 clamped. Ortho's C on comparisons: 0.35–0.77 unclamped, 0.13–0.32 clamped. The earlier statements "the components carry the truth value for comparisons and negations, much less for affirmative facts" (for ortho, and in the answer *Mostly no…* B2 also for recon) rest on unclamped swaps, where the recomputed rest at later layers adds to or works against the swapped part. Clamped, recon carries the value in every group (0.79–0.96), and ortho's C carries little in every group except s2 on negations (0.59). Marked `OUTDATED` at both places.

**Test 2: is C a part of the model's output?** All eight cells now exist. ν of the C-swap against ν of the swap of the original layer output inside C's write span (unclamped; each pair from one session and its rows):

| | plain s1 | plain s2 | recon s1 | recon s2 | ortho s1 | ortho s2 | ortho s3 | ortho s4 |
|---|---|---|---|---|---|---|---|---|
| verdict | 0.73 / 0.02 | 0.73 / 0.01 | 0.88 / 0.88 | 0.88 / 0.88 | 0.47 / 0.39 | 0.61 / 0.43 | 0.28 / 0.32 | 0.39 / 0.35 |
| Yes/No | 0.77 / 0.02 | 0.80 / 0.02 | 0.72 / 0.74 | 0.70 / 0.72 | 0.35 / 0.31 | 0.24 / 0.25 | 0.14 / 0.26 | 0.29 / 0.30 |
| session | H100 | 4090 | 4090 | 4090 | H100 | H100 | H100 | 4090 |

Random-subspace control: 0.01 (H100, ortho s1's sizes; not rerun). Reading [concluded]: plain fails in both seeds; recon and ortho pass in every seed.

**Test 3: damage.** Bias = mean change of a readout under the clamped swap; what is brought over from the twin cancels because every pair runs in both directions. Pooled, with the worst group where it matters:

| | object recall (original 0.937) | word count (original 6.01 words, sd 0.90) | subject recall (original 0.853) |
|---|---|---|---|
| site (unclamped) | +0.00 | +0.09 | +0.01 |
| baseline C / Δ | +0.00 / −0.00 | +0.09 / +0.02 | +0.01 / −0.00 |
| plain C, s1 / s2 | −0.68 / −0.63 (comparisons −0.97 / −0.98) | −0.49 / −0.53 | −0.05 / −0.06 |
| plain Δ, s1 / s2 | −0.65 / −0.65 (comparisons −0.97 / −0.96) | −0.53 / −0.65 | −0.04 / −0.06 |
| recon C / Δ, both seeds | −0.00 to +0.01 / +0.00 | +0.04 to +0.05 / −0.02 to −0.01 | +0.00 to +0.01 |
| ortho C / Δ, s1–s4 | −0.00 to +0.01 / −0.01 (negated: Δ −0.04 to −0.06, site −0.02) | −0.07 to −0.02 / −0.05 to +0.03 | −0.01 to +0.01 |

Reading [concluded]:
- Under the clamped swaps plain is the only method with damage, and it is gross. In the number comparisons the recall of the statement's own object goes from 0.94 to about 0 under both swaps. This is what the cancellation predicts: C from one statement plus Δ from the other no longer cancel, and plain's raw component write is 1.9–16 times |y|² (entry *write-span swaps, inspection and Pile in*).
- Recon and ortho are clean. Ortho's damage under the unclamped Δ-swap (word count −0.20 to −0.62 here as on 26-11-15) vanishes when C is held (−0.05 to +0.03). So it comes from ortho's C at later layers responding to a swapped Δ, not from Δ's own write.

**The negated-cities criterion** (fixed before the data in the entry *S = all version of the negated-cities test*): in neg_cities under the clamped swaps, C brings over at least half of the verdict gap; for humor, word count and subject recall C's share τ has an interval that includes 0 or lies below it; and Δ's share is the larger one for each of the three. Reference, site swap in neg_cities: verdict 0.89, humor 1.01, word count 0.51, subject recall 0.41.

| neg_cities, clamped | verdict: C | humor: C [lower bound] / Δ | word count: C [lower bound] / Δ | subject recall: C [lower bound] / Δ | holds |
|---|---|---|---|---|---|
| baseline | 0.87 | 1.01 [0.87] / 0.03 | 0.44 [0.38] / 0.08 | 0.39 [0.29] / 0.00 | no |
| plain s1 / s2 | 0.58 / 0.63 | −0.23 / −0.04, Δ 1.10 / 1.07 | −0.34 / 1.77, Δ 0.67 / −1.40 | 0.53 / 0.81, Δ 0.30 / −0.02 | no / no |
| recon s1 / s2 | 0.96 / 0.98 | 0.17 [0.09] / 0.13 [0.04], Δ 0.90 / 0.94 | −0.40 / −0.00, Δ 0.83 / 0.42 | 0.34 [0.26] / 0.42 [0.32], Δ 0.02 / −0.14 | no / no |
| ortho s1 / s2 / s3 / s4 | 0.04 / 0.57 / 0.22 / 0.28 | 0.19 / 0.22 / 0.12 / 0.14, all lower bounds > 0 | 0.05 / −0.02 / 0.58 / 0.03 | 0.33 / 0.26 / 0.13 / 0.32 | no (all four) |

Predictions check:
- Plain fails: ✓.
- Recon "passes for humor and word count, fails for subject recall": word count ✓ and subject recall ✓, humor ✗, narrowly (C 0.13–0.17, lower bounds 0.04 / 0.09; Δ 0.90–0.94).
- Ortho "passes in s2 only": ✗. s2 fails humor (0.22) and subject recall (0.26); s1, s3 and s4 already fail the verdict condition.

Reading [concluded: from these tables; one family, one run per cell]:
- **The claim fails as stated for every decomposition.** By the rule fixed beforehand it stays in the appendix.
- **Recon's C comes closest, and it is the only part that separates at all.** In neg_cities it brings over the verdict (0.96–0.98) with 0.13–0.17 of the humor gap and none of the word-count gap; Δ carries those two. In cities, where the word-count gap follows the truth value, recon's C does bring it over (0.92 / 1.02). The subject-recall gap goes with C (0.34–0.42, the site's 0.41). So recon's C follows the truth-linked part for humor and word count, but not for subject recall.
- **Withdrawn: the baseline's separation.** Clamped, the baseline's C brings over the content gaps as the site does (humor 1.01, word count 0.44, subject recall 0.39). The unclamped separation (word count −0.09 here, −0.06 on the H100) came from the rest at later layers. This is expected of a part fitted to twin differences, which contain the content as well. The sentence "such a part exists in the model" is marked `OUTDATED` in the entry *S = all version of the negated-cities test*. Recon's C does better than the baseline on this point.
- Object recall's τ in neg_cities is unusable (gap 0.35 sd, intervals up to ±10; plain's values near 11 are its damage, not a transfer).

**Which method fails what, now measured** (S = all; tests 4–5 from the 26-11-15 session):

| test | plain | recon | ortho | baseline |
|---|---|---|---|---|
| 1. clamped C / Δ | 0.63–0.66 / 0.32–0.35 | 0.86–0.89 / 0.09–0.10 | **0.11–0.33 / 0.61–0.84** | 0.94 / 0.01 |
| 2. write span against C-swap | **0.01–0.02 against 0.73** | 0.88 against 0.88 | 0.28–0.43 against 0.28–0.61 | (a part of W by construction) |
| 3. damage, clamped | **object recall −0.63 to −0.68** | clean | clean | clean |
| 4. C removed on the Pile, share of the site removal | 0.11 | **0.19** | 0.03 | – |
| 5. seed agreement | **0.02** | 0.82 | **0.07–0.11** | – |

Same failure pattern as proposed: plain 2, 3, 5 (and a split C/Δ in 1); recon 4; ortho 1, 5. The new numbers sharpen ortho's failure on test 1 and make plain's on test 3 the largest effect in the set.

**Not run:** test 6 (CI-weighted components; no code yet), emotion and surprise.
**Proposals** [assumed]: family-coverage statements in the report and the draft are restated from the clamped swaps. The negated-cities result goes into the appendix as "recon's C separates the truth value from two of three content-linked readouts".

C: Can you look through other logs to see if you can find evidence  that we already gather that the inactive components play a crucial role in the overall functioning?

Answer (26-11-17, Claude Opus 5.5) — *inactive components: what earlier topics already measured*. Context: this LOG in full, its SUMMARY, INDEX.md; a grep for "inactive" over all of `convos/julian/`, then the passages it found, read in full: the 26-10-22 tpd-truth-paper LOG (your comment "we have seen that inactive components can play a major role…" with its answer, and section 6 of *first pass*), the 26-10-29 paper-writing LOG (bullet *Extra state* and your comment beneath it), the 26-10-10 bolder-experiments LOG (*cancellation explained*, *cos(A + I, y) added*, the 4c cancellation test), the 26-10-16 faithful-targeting LOG (4d and 4e), and the 26-11-04 paper-draft LOG (scaling dial, *All against active*). Also read: the phase D tables `scratch/26-10-29_paper_writing/tables/tables.md` (sections 1b, 1c, 1d, 5a) and the state definitions in `paper_recovery.py`. Every number below was read from these files in this session. The truth-direction shares in point 4 I computed from table 5a (1 − share of A − share of Δ).

**Short version.** Yes. You asked on 26-10-27 for exactly these numbers "as an additional finding" (26-10-22 tpd-truth-paper LOG), and they were measured in phase D on the final runs (table 1c). They were never written up as a finding. In short:
- At full strength (mask 1), the components outside the active set are not inert in any method. Removing them from the full model costs 17–60% of the verdict separation.
- At their own CI they contribute little: 0.03–0.05 of KL recovered.
- So they matter as part of the full sum at mask 1 (where they cancel or complement other writes), not as a small extra contribution at their CI.
- The "used statistically" reading is weakly supported for ortho only.

**Definitions in phase D.** S = the *functional active set*: the smallest top-m set of components, ranked by mean CI on val, whose CI-weighted components-only state recovers ≥ 95% of what all components recover. It is one fixed set per run: plain 27 / 44, recon 387 / 432, ortho 29 / 31 / 58 components. "Inactive" = the components outside S. That is not "CI ≈ 0 on this token": a component outside S can have CI > 0 on some tokens. This matters most for recon (5.8% of its CI values are non-zero, table 1d).

**1. Table 1c** [verified: `tables.md` 1c; test split, 1,794 statements, training prompt, one run per cell]. The left three columns are KL recovered with Δ off (1 = original verdict distribution, 0 = the decomposed MLPs off). The right four are the full model (Δ kept) with every component outside S removed (mask 1 → 0). Separation share: 1 = the original true–false gap of the verdict score. Flips are macro over families.

| run | all at CI | only S at CI | mean mask CI + (1 − CI)/2 | outside S removed: agreement | separation share | 1→0 | 0→1 |
|---|---|---|---|---|---|---|---|
| plain s1 | 0.961 | 0.909 | 0.964 | 0.795 | 0.446 | 0.045 | 0.327 |
| plain s2 | 0.958 | 0.915 | 0.964 | 0.822 | 0.405 | 0.113 | 0.310 |
| recon s1 | 0.974 | 0.928 | 0.974 | 0.958 | 0.675 | 0.193 | 0.017 |
| recon s2 | 0.971 | 0.934 | 0.973 | 0.974 | 0.830 | 0.055 | 0.036 |
| ortho s1 | 0.906 | 0.868 | 0.959 | 0.907 | 0.602 | 0.122 | 0.031 |
| ortho s2 | 0.893 | 0.861 | 0.960 | 0.917 | 0.682 | 0.014 | 0.216 |
| ortho s3 | 0.875 | 0.832 | 0.948 | 0.870 | 0.735 | 0.160 | 0.160 |
| untrained s1 / s2 | −0.002 / −0.003 | – | 0.006 / 0.049 | – | – | – | – |

Reading [concluded: from the table]:
- **Removal from the full model: every method fails "honest CI".** A perfect decomposition loses nothing when the inactive components are removed (E1/D1 in the evaluation-categories definition, test 0 in the draft). Here the separation share left is 0.41–0.45 for plain, 0.68–0.83 for recon and 0.60–0.74 for ortho. Plain is worst: its removal pushes false statements to "1" (0→1 flips 0.31–0.33). This is the most direct evidence we have.
- **At their CI they add little.** Components only, Δ off: all components at CI against S alone at CI differ by 0.03–0.05 of KL recovered in every run. So their effect in the full model comes from the part of their write that the CI weighting leaves out.
- Per method, the two points fit what we know about the writes:
  - *Plain:* the components write vectors orthogonal to y, and Δ contains their negative (entry *write-span swaps, inspection and Pile in*). Removing the outside-S components leaves Δ's negative of their writes uncancelled.
  - *Recon:* the components are a real part of y (present share 1.03–1.05), and S covers only some of them. Removing the rest removes real parts of the output. Some of these components are also active on some tokens (definition above).
  - *Ortho:* the inactive components cancel 60–87% of the active set's raw write (26-11-15 inspection). Removing them un-cancels it.

**2. Mean mask: the "statistical use" test, weak support for ortho only** [verified numbers; reading concluded]. The mean-mask state was added to separate "inactive components matter" from "they matter through the training's random masks". That is your low-CI averaging hypothesis (26-10-22 tpd-truth-paper LOG, *first pass* section 6), where you decided "report this as a theory if we keep seeing low-CI component clusters that have big influence".
- Ortho gains 0.05–0.07 of KL recovered under the mean mask (0.875–0.906 → 0.948–0.960). Plain gains 0.003–0.006, recon ≤ 0.002. So ortho's components alone reproduce the verdict better in the state that training sees on average than at their CI. That is the pattern the hypothesis predicts.
- Two limits:
  - The untrained s2 control gains 0.049 too (from −0.003). Half of every component's write beats nothing even without any learned reliance, so a gain of this size needs no learned reliance.
  - The mean mask also raises active components whose CI lies strictly between 0 and 1. In ortho they are at least as common as CI = 1 (table 1d: 0.33–0.48% of all CI values in (0, 1) against 0.17–0.31% at exactly 1). So the gain cannot be assigned to the CI ≈ 0 components alone.

**3. In this topic** (the entries above):
- Ortho: the inactive components cancel 60–87% of the active set's raw write at mask 1, and swapping all components moves the verdict more than swapping the active set (s1 0.42 against 0.31).
- Plain: swapping all components moves the verdict less than swapping the active set (0.74 against 0.92; also in the 26-11-04 scaling dial). Plain's inactive components counteract its active ones.

**4. Along the truth direction they carry little** [computed from `tables.md` 5a: mass-mean direction after each layer, statement's last token; here "inactive" = the (1 − CI)-weighted remainder of all components' writes, a slightly different cut]. Their share of the true–false gap per layer: plain −0.16 to +0.16, recon −0.01 to +0.22, ortho −0.04 to +0.15 (untrained within ±0.02). Δ carries most of the gap for plain and ortho (0.49–1.07 and 0.80–0.99 per layer). So their role in the verdict does not run through writing the truth direction directly.

**5. Earlier exploratory runs** (before the paper's three methods; one seed each):
- Exp 3 (verdict lens over the appended question, MLPs 15–19): |I| is 1.9–2.5 times |y| per layer (100 test statements), and cos(I, Δ) is about −0.8. In Exp 0–3, all components are no closer to y than the active ones (cos(A + I, y) − cos(A, y) ≤ 0.02). There the inactive components are half of a cancelling pair with Δ, not a part of the output.
- Exp 4a (verdict lens only): the active layer-11 write is 12.9 times |y|, undone by the inactive components and Δ (3–14 times |y|, cosine −0.92 to −0.97).
- 4d: the cancellation moved into I and Δ (each up to 11 times |y|), but it is invisible to the verdict (Δ alone changes the verdict KL by 0.027).
- 4e: a small active-against-inactive opposition (cos(A, I) −0.23 at L13, −0.12 at L17). The integrated-gradients split gave the inactive components −1.16 of a total 8.23 (path-dependent, not causal, as noted there).
- 4c (with per-layer reconstruction): the opposite pattern. cos(A + I, y) is 0.60 against cos(A, y) 0.46 at ordinary tokens, so the inactive components carry further parts of y. That is your y = t + r picture, and the recon-like case.

**What this means for the S = all argument** [concluded]:
- Your reason (1) holds in this form: at mask 1, the components outside the active set are a needed part of the full sum in every method (1c removal). In plain and ortho they are partly a cancelling counterweight, in recon partly real output.
- Not established: that the methods *use* them at their own CI. That gain is 0.03–0.05 of KL recovered. The random-mask reading has only the weak ortho evidence of point 2.
- Correction to my answer of 26-11-16 (beneath your first comment): I wrote that the mask-1-artefact reading was "still untested". For the components-only state, 1c's "all at CI" against "only S at CI" already tests it: at CI, the inactive components add little. Untested is only the swap version (test 6). Marked `OUTDATED` there.
- Proposal [assumed]: put table 1c into the paper as test 0 (honest CI), with the mean-mask column and its untrained control. It fails for every method, and it is the finding you asked for on 26-10-27. The 26-10-29 paper-writing LOG describes the 1c state as "all components with CI ≈ 0 removed". The code removes the components outside the fixed set S (`paper_recovery.py`, state `full_without_inactive`). Marked `OUTDATED` there.

C: Can you write a completely new tpd_truth_paper_draft? It should include:
- why we take S = all (a simple argument suffices)
- the new minimal test in the main part
- further minor findings can go in the appendix

Answer (26-11-17, Claude Opus 5.5) — *new paper draft written from scratch*. Context: this LOG and its SUMMARY; the 26-11-04 draft (`reports/26-11-04_tpd_truth_paper_draft.typ`, all of it) and its topic SUMMARY; the 26-11-15 report `reports/26-11-15_method_flaws.typ`; the self-contained-check procedure. Numbers re-read in this session from the generated tables: `~/spd_out/26-11-15_open_research/analysis/` (`session2.md`, `score.md`, `inspect.md`, `tables.md`, `truth_linked_transfer.md`), the phase D tables (`scratch/26-10-29_paper_writing/tables/tables.md`, 1c) and ortho seed 4's recovery output (`vm/recovery/recovery_s-22600dd8.json`). The method description, the model and layer-scan tables, the site versions and the dial numbers are taken from the 26-11-04 draft.

**File:** `reports/26-11-17_tpd_truth_paper_draft.typ` (15 pages; `typst compile reports/26-11-17_tpd_truth_paper_draft.typ`, compiles without warnings). The 26-11-04 draft is left unchanged.

**Structure.**
1. *Introduction* (half a page): question, the three methods, the result, the secondary result about the tests.
2. *Setting*: model, task and score, families (table of the model's accuracy); what is decomposed and why (layer-scan table, limits); the three methods (losses, the two figures from the old draft, settings table with a "components alone recover" row for all eight runs).
3. *Evaluation*:
   - 3.1 *The split: all components against the remainder.* Your request, three bullets: it is the split the method defines (the losses compare components with Δ; "active" needs a cutoff that is in no loss; ortho constrains only the sum); the inactive components are not inert (17–60%, from phase D table 1c); equal size in every method. Plus one sentence on what this leaves out (CI-weighted tests not run).
   - 3.2 Twins, swaps, clamped swaps, the site swap, readouts (probe table).
   - 3.3 The five tests: a table of what each measures, what a perfect decomposition gives and what it detects, then one paragraph per test on why it is in the set.
   - 3.4 Baseline.
4. *Results*: an overview table (each method against the five tests, the failing cell in bold) and a findings box, then one subsection per test with its full table and a short reading.
5. *Discussion*: what each method's failure means, what we learned about the tests, limits.
- *Appendix* (the minor findings):
  - A1: active set against all components (table 1c, with ortho seed 4 added from its recovery output, and the active-set swap table);
  - A2: clamped against unclamped swaps, and removals;
  - A3–A5: inside plain, recon and ortho;
  - A6: the other-quality readouts depend on truth and on content;
  - A7: the negated-cities criterion;
  - A8: smaller baselines;
  - A9: validating the readouts (site versions, scaling dial);
  - A10: run facts.

**Choices you may want to change** [concluded: agent choices, not ratified]:
- Test 6 (CI-weighted components) appears only as a limit, since it was not run.
- The old draft's evaluation theory is not carried over: the causal model with assumptions A1–A3, the definition of "P captures T", the failure-mode list and the glossary of phase D analyses. The new section 3 motivates each test directly. If you want the definition back, it would go before 3.3.
- The overview marks failures in bold without a threshold. The caption says that no pass threshold was fixed in advance.
- In the discussion, the baseline is called "a reference, not a competing method", because it is supervised by which statements are twins.
- Every number in the main text is a measured cell. The appendix uses the H100 session (906 twin rows) and phase D where no 4090 numbers exist; the run-facts section says which test ran where.

**Self-contained check:** done as a self-check while re-reading. I grepped the draft for history references and internal names (T0, G2, Exp 4a, phase D, convos): none in the text. The fresh-reader sub-agent check of `docs/procedures/self_contained_check.md` was not run. I can run it if the draft is going beyond you.

---

C: (transcribed from chat) What do the non-clamped C and Δ numbers say in this test?

(Asked with the cursor on "Test 1: where is the truth value?" in `reports/26-11-17_tpd_truth_paper_draft.typ`.)

---

26-11-17, Claude Opus 5.5 — *reading of the unclamped C/Δ swaps in test 1*; I have read INDEX.md, the draft's test-1 section and appendix table `tab-app-unclamped`, and the LOG entry *GPU session 2 results…* (test 1 table, source of all numbers below; 898 twin rows, RTX 4090, one run per cell) [verified numbers: re-read from that table]

Unclamped = the swapped part taken from the twin, the other part recomputed live, so at later layers it reads what the swapped part wrote earlier. Each unclamped number is therefore "the part's own effect + the other part's downstream response". The clamped − unclamped gap measures that response.

| | C unclamped / clamped | Δ unclamped / clamped | unclamped sum (site 0.94) |
|---|---|---|---|
| baseline | 0.89 / 0.94 | 0.03 / 0.01 | 0.92 |
| plain s1, s2 | 0.73, 0.73 / 0.66, 0.63 | 0.56, 0.52 / 0.32, 0.35 | 1.29, 1.25 |
| recon s1, s2 | 0.88, 0.88 / 0.86, 0.89 | 0.29, 0.26 / 0.10, 0.09 | 1.17, 1.14 |
| ortho s1–s4 | 0.46, 0.61, 0.28, 0.39 / 0.11, 0.33, 0.14, 0.19 | 0.69, 0.43, 0.47, 0.66 / 0.84, 0.61, 0.80, 0.75 | 1.15, 1.04, 0.75, 1.05 |

Reading [concluded from the table unless tagged]:
1. *The baseline barely changes* (gaps ≤ 0.05; sum 0.92). That is what a part with little cross-part mediation looks like, and it is the reference for the other rows.
2. *Recon's C is robust; its Δ is not silent.* C 0.88 either way. But Δ goes from 0.09–0.10 clamped to 0.26–0.29 unclamped: when Δ's writes come from the twin, recon's C at later layers recomputes toward the twin's verdict. So C writes the truth value, but partly computes it from what Δ wrote at earlier layers. Two readings, not separated [assumed, either]: (a) benign: Δ carries the statement content (object identity etc.) that a truth mechanism must read; (b) a defect: part of the truth computation runs through Δ upstream. Against (a): the baseline's Δ, which also carries content, gives only 0.03 unclamped. The draft's text ("recon carries most of the truth value") reports clamped numbers only and does not mention this.
   OUTDATED (26-11-17): the "Against (a)" sentence does not hold. The baseline's C is built from the top principal directions of the twin differences, so it takes the twin-varying content too (its clamped C brings over humor 1.01 and word count 0.44 in neg_cities). Its Δ therefore carries little content that differs between twins, and its 0.03 says nothing about (a). See the entry *clamped or unclamped: which question each answers*.
3. *Ortho: the two parts work in opposite directions downstream.* C goes up unclamped (0.28–0.61 against 0.11–0.33; Δ at later layers follows C's swapped writes, already in the draft). Δ goes down unclamped (0.43–0.69 against 0.61–0.84): when C is recomputed after a Δ swap, it moves the verdict back toward the original statement. The second half is not in the draft or the SUMMARY; its mechanism is not checked [assumed: compensation by C].
4. *Plain is super-additive* (sums 1.25–1.29): either part swapped alone, with the other recomputed, brings over more than half the verdict difference. Unclamped Δ 0.52–0.56 against clamped 0.32–0.35, so plain's C follows Δ as well. Consistent with the cancellation picture (C and Δ are large, opposite writes), but not a test of it.
5. *Per group* (from the draft's `tab-app-unclamped`): on affirmative facts the unclamped C-swap drops for the baseline too (0.64 against 0.85 clamped), not only for recon (0.50–0.52 against 0.79). So the "affirmative facts are missing from C" reading of the unclamped swaps was largely a property of that family (something downstream restores the original verdict there), not of recon.

Ranking of the methods is the same as clamped: C recon > plain > ortho; Δ baseline < recon < plain ≈ ortho.

---

C: (transcribed from chat) I don't quite understand why the clamped swaps are supposed to be better than the non-clamped ones. I actually think the non-clamped are more realistic and it caputres more the indirect effects of the interventions.

---

26-11-17, Claude Opus 5.5 — *clamped or unclamped: which question each answers*; I have read INDEX.md, the SUMMARY, this LOG's entries on the clamped swaps and the reading of the unclamped ones, the draft's sections on swaps, test 1 and appendix `app-clamp`, and the swap code in `paper_interventions.py` (`Swap`, `SwapClamped`, `edit_fn`)

**What is actually held fixed** [verified: code read]. Both swaps act only on the decomposed down_proj outputs of layers 11–17, at the statement tokens. Attention, layer norms, every layer after 17 and all probe tokens are recomputed in both. So the clamped swap does not freeze the network. It removes one route: the swapped part's earlier writes → the other part of the same matrices at later sites → verdict. The routes through attention and through layers 18+ count in both versions.

**"More realistic": I don't think so** [concluded]. In the unclamped C-swap, C is pinned to the twin's value at all seven sites, including the later ones, and only Δ responds to the edited stream. In the real network the whole matrix W = C + Δ would respond to an earlier change. So the unclamped run lets half of each later matrix react and freezes the other half. That is no closer to a state the model reaches than the clamped run, where both halves at the sites are values from real runs (q(x') + rest(x)). Neither one is realistic. A version that would really capture indirect effects swaps C only at early layers and lets C and Δ at later layers both recompute. That run doesn't exist yet.

**"Captures the indirect effects": yes, and that is both its value and its problem for test 1** [concluded]. Unclamped is a total effect (the part's own writes plus how the other part answers them). Clamped is the direct effect at the sites. Test 1 asks how the truth value is shared between C and Δ. For a share, the two numbers have to divide one effect:
- Clamped, they add up to the site swap: 0.94–0.99 against 0.94 in all eight decompositions. Unclamped they don't: 0.75 to 1.29. In a super-additive case like plain (1.25–1.29) each part looks as if it carries most of the value, so the mediated effect is counted twice.
- A pass/fail criterion needs a known value for a perfect decomposition. Clamped, that value is clear: C at the site swap, Δ at 0. Unclamped, the ideal Δ-swap is not 0 even for a perfect decomposition. Δ legitimately carries content that differs between twins (the object), and a correct truth mechanism in C has to read it. So when Δ comes from the twin and C is recomputed, C would correctly compute the twin's truth value from the twin's content. The unclamped Δ-swap therefore can't separate "Δ holds the truth value" from "Δ holds the inputs to the truth value". This is readings (a) and (b) in the previous entry. The argument I gave there against (a) is wrong; it is now marked OUTDATED.

**Where Julian is right** [concluded]. The indirect part is real information that the clamped number hides, and the draft puts it only in the appendix:
- Recon: Δ 0.09–0.10 clamped against 0.26–0.29 unclamped. Recon's C partly computes the truth value from Δ's earlier writes. Benign (content) or a defect is open.
- Ortho: C 0.11–0.33 clamped against 0.28–0.61 unclamped. Ortho's components move the verdict mainly by steering Δ downstream.
- In both, "C is upstream of the truth value" (total effect) and "C's writes are the truth value" (direct effect) come apart. Which one a decomposition should be judged on depends on what we want C to be. It should be a property we state, not one hidden in the choice of intervention.

**Proposal** [assumed; for Julian to decide]: in test 1 of the main text, report three numbers per part: the clamped share (sums to the site, has an ideal), the unclamped number, and the gap between them, labelled as mediated through the other part. Then the claims "recon's Δ feeds C" and "ortho's C acts through Δ" are visible where the methods are compared. Optional run, if indirect effects should be measured properly: swap C only at layers 11–13 and let all of layers 14–17 recompute. This is the version that matches "more realistic".

---

C: (transcribed from chat) Can you create a better figure for this? It should be more conceptual so it's easier to understand the training setup

---

26-11-17, Claude Opus 5.5 — *conceptual setup figure for the 26-11-17 draft*; I have read INDEX.md, the draft and its old diagram file

Julian's selection was `fig-setup` in `reports/26-11-17_tpd_truth_paper_draft.typ` (the token-by-layer grid from `reports/26-11-04_tpd_truth_paper_draft_diagram.typ`). Replaced it with a new figure in `reports/26-11-17_tpd_truth_paper_draft_diagram.typ` (`setup_figure`); the old diagram file is unchanged, still used by the 26-11-04 draft and for `restriction_panels`.
- (a) one decomposed matrix: $W' = m_1 U_1V_1^top + … + m_96 U_96V_96^top + m_Delta Delta$ drawn as column-times-row icons plus a full-rank Δ block, and the shared CI network giving a sparse CI bar chart; the importance-minimality penalty is shown as a note under the chart.
- (b) the target stream: statement (W' in the MLPs 11–17) and probe (original W), attention arrow from the statement to the probe, verdict bars (original vs W') with KL; the dashed orange outline for recon/ortho kept; mask rule and the pressure it creates, in one line each.
- (c) the non-target stream: Pile text, W' at every token, KL at every token; mask rule ($m_Delta = 1$, $m_c$ in [CI, 1]) and the pressure that pushes off-target computation into Δ.
Colours match `fig-restrictions` (components blue, Δ brick red). Left out on purpose, for readability: the random number of masked layers per position, the adversarial masks and the weight decay (all in the methods text). Bars are illustrative, and the caption says so. Rendered and checked by eye [verified: typst compile, page 2 inspected].

---

C: (transcribed from chat) Can you split (a) and (b) into two figures. (a) can go into the introduction because it's typical tPD. (b) is explicity about this paper

---

26-11-17, Claude Opus 5.5 — *setup figure split into intro and setting figures*; continuing from the entry above

Split as asked. `tpd_figure` (the former panel (a)) is now Figure 1 (`fig-tpd`) in the introduction, referenced from the tPD paragraph. It is made generic: $N$ components instead of 96, and the CI network is "shared by all decomposed matrices" instead of "one for all 7 layers". Its caption describes tPD in general terms: components, Δ, masks, CI network, importance penalty, and the target/non-target pressure. `setup_figure` (`fig-setup`, Setting) keeps both data streams, now labelled (a) target and (b) Pile. I read Julian's "(b)" as both streams, since the Pile stream is also this paper's setup. Its caption now points to `fig-tpd` for the decomposition and states the paper-specific part (`down_proj` of layers 11–17, 96 components per layer) [verified: typst compile, pages 1–2 inspected].
