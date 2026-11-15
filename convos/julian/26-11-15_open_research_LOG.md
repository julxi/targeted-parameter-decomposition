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
