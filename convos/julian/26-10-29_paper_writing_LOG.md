I need a new conversation for the tpd truth paper. So far we have roughly fine-tuned and trained the decompositions and we did quite a bit of analysis. Now we have to write the paper:
- analyse the data
- discuss the explanations of the findings
- ideally find a lesson
- actually write it

(Julian's opening request, from chat; the first message was cut off and resent, both merged here verbatim.)

---

26-10-29, Claude Opus 5.5 — *topic opened: state and four-stage plan*; new session. I have read CLAUDE.md, INDEX.md, the SUMMARY of `convos/julian/26-10-22_tpd_truth_paper_SUMMARY.md` (the experiments topic this paper rests on), its LOG entries *D3 proposal* to *D3 frozen; A5 started* (the frozen analysis plan) and the list of its later entries. I have listed the phase D outputs but not opened their contents. Not read yet: `docs/CODING.md`, the six `paper_*.py` scripts.

## Where we start

- **Experiments** (see the experiments topic's SUMMARY for every number): tPD (targeted parameter decomposition) of `down_proj` in layers 11–17 of Qwen2.5-7B-Instruct at the statement tokens. The target loss is the KL of the verdict digit under the verdict prompt `<statement>\nThe amount of truth in this statement is `. Three methods: plain tPD, + per-layer reconstruction ("recon"), + global orthogonality term ("ortho"). Each was tuned on val and trained in two final seeds (1, 2). Controls: untrained decomposition (seeds 1, 2) and the data-aligned basis (top principal directions of each MLP's inputs). [verified there]
- **Analysis plan frozen** before any test/OOD look [decided: experiments topic, entry *D3 frozen; A5 started*]: recovery, termination T0–T3, correlation K0–K3, manipulation (core A + exploratory val tables + paired-twin test), truth directions, active vs inactive components, CI valley check. Reporting conventions: every number with both seeds and its controls, no method ranking where the seeds overlap, everything listed gets reported whatever the outcome.
- **Raw outputs exist, no summary tables yet** [verified: directory listing 26-10-29]: `~/spd_out/26-10-22_tpd_truth_paper/phase_d/h200_paper_d/out/` holds 18 files each for correlation, directions, manipulation, termination (json + npz for 9 objects: 6 finals, 2 untrained, 1 data-aligned basis; 5.2 GB); `…/recovery/` holds the recoveries and the CI dumps for the valley check (130 MB). All 457 invariant checks passed on the GPU host.
- **Analysis code committed** (`d06399b`, `8ca8bad`); git is clean.
- **Open from the run**: the aggregation question. Macro-averaged KL recovered is dominated by families with a small all-off KL (ortho s2, test, second prompt: pooled 0.735 vs macro 0.162, cities alone) [verified there]. The frozen plan says "macro-average over families", so changing it is a plan change and needs Julian.
- **Not done from the old plan**: A1 (living report) was skipped; phase E (registry entries with public canonical WandB runs) is still to do.

## Plan for the four stages [assumed: proposal, not ratified]

**Stage 1, analysis (offline, CPU, no GPU needed).**
- 1a. One summary script (`paper_tables.py` or similar, in the repo next to the `paper_*.py` scripts) that reads every output and writes the frozen plan's tables: recovery (test, each OOD file, second prompt), side effects (Δ-only Pile KL), T0–T3 with k-curves, K0–K3, manipulation core + paired-twin test, directions, active-vs-inactive, valley histograms. Both seeds and the controls in every row; bootstrap intervals from the npz. Each table is checked against at least one number from the run logs (Observe & Verify rule). Output: tables + figures as files, not chat.  
C: OK, that's what we do first. Provide the tables and plots and then I'll come back to you. We do one step after another, so ignore the rest for now.
Answer (26-10-29, Claude Opus 5.5): Noted [decided: C: comment]: 1a only, then I stop and hand over. Since the aggregation question is open, every table carries pooled, macro and per-family numbers side by side, so nothing has to be decided before you look.
- 1b. The exploratory part: browse the val tables (manipulation wide tables); whatever looks interesting becomes a hypothesis tested on test/OOD before it counts (rule from the frozen plan). Anything that would need new GPU runs gets listed, not run, until Julian decides.
- 1c. A results digest in this LOG: per category what came out, with numbers and controls, and which pre-run expectations held (e.g. "components write little along the truth direction" [assumed there]).

**Stage 2, explanations.** For each main finding: the candidate explanations, and which existing number tells them apart. Explanations already on the table from earlier topics: the verdict is a truth *tag* read by verdict prompts, not the model's knowledge (B3, steering); components re-implement the lens rather than decomposing the MLP output (cancellation test, Exp 0–4a); whatever a verdict-target decomposition finds, a data-aligned basis may find too (4e). Where the data cannot separate two explanations, the paper says so.

**Stage 3, the lesson.** I'd keep this open until stage 1 is done: deciding the lesson first invites reading the tables toward it. What I'd do now is write down candidate lessons with what would refute each, so that stage 1 tests them instead of decorating them [assumed]. Candidates from the records:
- (L1) "Without the data-aligned control, tPD results on a behavioural target overstate what the decomposition found": refuted if the trained components clearly beat the data-aligned basis on termination/manipulation in both seeds.
- (L2) "The target loss decides what gets decomposed; a verdict lens decomposes the readout of truth, not truth itself": refuted if components also move recall / second-prompt behaviour in a belief-like way.
- (L3) "Methodological: CI probes and single recovery numbers are not evidence of a mechanism; the four-category protocol (correlation, manipulation, termination, recovery) with controls is the minimum": a methods lesson, holds largely regardless of outcome.
L3 is the safest but also the least interesting; a paper that only says L3 is a negative-results note.

**Stage 4, writing.** Proposal: a Typst draft in `reports/` (as the earlier reports; `reports/template.typ`) as `{date}_tpd_truth_paper_DRAFT`-style working text, edited in place, discussed here. Order: results section first (straight from stage 1 tables), then methods, then discussion/lesson, intro and abstract last. Earlier negative results (arm A, UTH, Exp 0–3) in an appendix [decided there]. Self-contained check (`docs/procedures/self_contained_check.md`) before anything leaves the project.

## Questions for Julian

1. **Venue, audience, format**: arXiv preprint, workshop paper (page limit?), or LessWrong/Alignment Forum post? That decides length, LaTeX vs Typst, and how much tPD background goes in. Co-authors (e.g. Madina Babazhanova, whose fork added the honesty experiments)?
2. **Aggregation** (plan change): keep macro over families as primary, or switch to pooled with per-family tables? My proposal: per-family tables as the primary record, pooled as the headline number, macro reported next to it, because macro over families with near-zero all-off KL measures noise.
3. **Start**: shall I begin with 1a (read `docs/CODING.md` and the scripts' output formats, write the summary script)?
4. **Phase E** (registry entries, public WandB runs): here, or stays in the experiments topic?

---

26-10-29, Claude Opus 5.5 — *stage 1a done: tables and figures*; same session. Julian (C: comment under 1a): "Provide the tables and plots and then I'll come back to you. We do one step after another, so ignore the rest for now." I have now also read `docs/CODING.md` and the six `paper_*.py` scripts in full.

**What exists** [verified: run 26-10-29, `make check` clean]:
- Script `spd/experiments/lm/honesty_targeted_decomposition/paper_tables.py` (new, CPU only, ~20 s). It reads the phase D outputs (`~/spd_out/26-10-22_tpd_truth_paper/phase_d/h200_paper_d/{out,recovery}/`) and the finals' val readout of phase C (`…/phase_c/instance_paper_c/rounds/final_*/summary.json`, for the side-effects table).
OUTDATED (26-10-29): the output folder moved to `scratch/26-10-29_paper_writing/tables/` (entry *outputs moved to scratch*).  
- Output in `~/spd_out/26-10-29_paper_writing/tables/`:
  - `tables.md`: every table of the frozen plan, all 9 bases side by side (6 finals, 2 untrained decompositions, the data-aligned basis): 0 side effects, 1 recovery (+ active set, active vs inactive, valley check), 2 termination T0–T3, 3 correlation K0–K3, 4 manipulation (groups, singles vs random, exploratory val wide table), 5 truth directions.
  - `details.md`: per-family recovery with each family's all-off KL, write norms per layer, CI bins per layer, the full T2 grid incl. the second prompt and full OOD, T1 per active component (val vs test), every single manipulation condition, top gap-share components.
  - `wide_val_reverse_<run>.csv`: the exploratory val tables, per family.
  - `figs/`: active-set curve, CI histograms, sorted mean CI, T1 drops, T2 top-k curves, K1 sparse probes.
- **Consistency checks**, run by the script on every call [verified]: pooled KL recovered of two recovery states recomputed from the per-statement arrays (8 runs × 4 sets), macro separation share of T2 at the smallest and largest k (9 bases), pooled flips of one manipulation group (9 bases): all equal to the stored summaries. The all-off KL is identical across the 8 runs (it does not depend on the basis), as it must be.

**Reading conventions written into the tables** (choices I made, open to change):
- KL recovered is shown both pooled (with a new pooled bootstrap interval: same resampling within family as the stored macro intervals) and macro, because the aggregation question is open. Separation share and flips stay macro as stored; agreement pooled.
- OOD comparisons of top-k against random components use the common control subset (≤ 50 per dataset, 950 statements); full-OOD numbers for the main conditions are in `details.md`.
- The main T2 tables show k = 1, 5, 10, 20, the trained runs' active-set sizes and all candidates; the full grid is in `details.md`.

**Gaps against the frozen plan** [verified: outputs]: the side effects (Pile Δ-only KL) exist on val only, from the finals' readout; no test-set Pile run was made. The manipulation plan's "activity profile per family × label" for the wide tables is not tabulated (the correlation npz holds the per-statement values; I can add it). Nothing else in the plan is missing.

No interpretation in this entry, by design (Julian's step-by-step instruction). Next: Julian reads the tables.

---

26-10-29, Claude Opus 5.5 — *outputs moved to scratch*; same session. Julian in chat:

C: (transcribed from chat) Can you move the plots and all into this projects scratch folder?

Moved `~/spd_out/26-10-29_paper_writing/tables/` (tables, details, CSVs, figures) to `scratch/26-10-29_paper_writing/tables/` in the repository [verified: listing; the old folder is removed]. `scratch/` is gitignored, so the files stay local, like the earlier `scratch/*_summary.md` files. This departs from `docs/CODING.md` ("run outputs … go outside the repository, into … `~/spd_out/`"), at Julian's request. The script still takes the folder as `--out`; rerun with `--out scratch/26-10-29_paper_writing/tables` to regenerate in place. INDEX.md and the SUMMARY now name the new path.

---

26-10-29, Claude Opus 5.5 — *figure changes after Julian's first read*; same session. Julian's questions in chat (what "finals", bracket intervals, macro, separation share, the active-set search and the A8 projection mean) were answered in chat only; they are definitions already in the script's table texts. Changes to `paper_tables.py` and the outputs in `scratch/26-10-29_paper_writing/tables/` [verified: regenerated, `make check` clean]:
- Legend handles lengthened in all figures: seed 2's dashes were hidden behind the marker (Julian: the active-set plot "only shows solid lines").
- Mean CI per component: linear y-axis only. Julian, after seeing both versions:

C: (transcribed from chat) The linear one

- New figure `figs/t3_a8.png` (Julian: "Can you provide a plot that shows the effect in relation to the controls? You can leave out some data if that would make the plot noisy."): per method (columns) and span kind (rows: all 96 write vectors, active components only), separation drop (1 − separation share) per layer for both seeds, against the top-PCA control of the same dimension (gray, per seed) and the random subspaces as one gray min–max band over both seeds' draws. Test, training prompt; flips and OOD stay in the table only.

---

26-10-29, Claude Opus 5.5 — *K1 grid and K2 figures*; same session. Julian in chat: "In K1 can you show the plot … as 2 x 2", then, after my K2 proposal (a val-vs-OOD AUC scatter per basis plus a dot plot of the shares, with the affirmative transfer split into plain affirmative vs conj/disj): "Can you add the plots to `tables.md`?" [verified: regenerated, `make check` clean]:
- K1 figure: 2 × 2 (rows label/verdict, columns test/OOD), one legend outside.
- K2, coef features: `figs/k2_shares.png` (transfer share per basis in three panels: negated → negated, affirmative → plain affirmative, affirmative → conj/disj; label and verdict; number of val-selective components) and `figs/k2_scatter_{label,verdict}.png` (one point per selective component × OOD family, val AUC vs OOD AUC, 3 × 3 grid of bases, transfer region shaded). The split shares are recomputed from the stored per-family AUCs with the K2 rule; the script asserts that the parts recombine to the stored shares and pair counts for all 9 bases (passes).

---

26-10-29, Claude Opus 5.5 — *manipulation figures (4a)*; same session. Julian in chat: "I also need a plot for 4a. Manipulation". Added to `paper_tables.py`, linked in `tables.md` section 4a [verified: regenerated, `make check` clean]:
- `figs/m_groups.png`: 2 × 2 panels (reverse/amplify × top10/selective group), one row per basis, the macro flip shares 1→0 (▼) and 0→1 (▲), test filled and OOD (13 files) hollow, training prompt; group sizes printed. The second prompt stays in the table only.
- `figs/m_twin.png`: paired-twin test on test (share of negated statements whose edited verdict equals the affirmative twin's original verdict), the four group edits per basis, the original model as a dashed line (asserted identical across bases).
Also earlier in this stretch (chat only, no change): Julian asked what "share of pairs that transfer" means in the K2 figure, about the 0–1 AUC axes of the K2 scatter, and whether perfect transfer would be the diagonal (yes for identical behaviour; selection makes points regress toward 0.5 even with real transfer).

---

26-10-29, Claude Opus 5.5 — *directions figures (5a)*; same session. Julian in chat: "I need some graphs for 5a." Added to `paper_tables.py`, linked at the top of section 5a in `tables.md` [verified: regenerated, `make check` clean]:
- `figs/d_gap_shares.png`: 2 × 2 (rows mm/lr direction, columns gap share of A = components' CI-weighted writes, D = Δ's write), per layer, all 9 bases.
- `figs/d_alignment.png`: max |cos(write vector, d)| over each layer's 96 write vectors per basis, mm and lr, against the expected maximum of 96 random vectors.
- `figs/d_removal.png`: verdict separation drop when the direction itself is projected out of the original MLP output, per layer and all layers at once, mm and lr (from the data-aligned run, which ran `--remove_directions`; the directions do not depend on the basis).
The Spearman T1-vs-gap-share rows stay in the table only (few candidates per layer, some undefined).
Also in this stretch (chat only, no change): what the selective group is (union of the K2 val-selective components, coef readout, not size-matched across bases: 13 to 485), and how the paired-twin test works (edited negated statement vs the twin's unedited verdict; it cannot separate "the not is ignored" from "negated verdicts inverted").

C: OK, I've gone through most of the data. You said at one point that you report 0 -> 1, 1 -> 0 to get the shifting of the separation. Wouldn't it be easier to report separation shift alongside separation share?
Answer (26-10-30, Claude Opus 5.5): Yes, add it. I read "separation shift" as the signed move of the midpoint between the true-class and false-class mean verdict scores, in units of the original gap: shift = ((m_T′ + m_F′)/2 − (m_T + m_F)/2) / (m_T − m_F), with the score s = log P('1') − log P('0'). Positive means the edit moves the scores toward "1". With separation share, it describes both class means completely: share says how far apart they still are, shift says where they went. The flips mix the two, so a 0→1 rate alone can't tell "everything moved toward 1" from "the gap closed and statements near the threshold crossed". I checked this on the stored per-statement scores [verified: `scratch/26-10-29_paper_writing/shift_probe.py`, test, training prompt, macro over families; it reproduces the stored 1→0 flips of plain s1's reverse-top10 condition (0.35) and its T2 top-1 share (0.81)]:

| basis, condition | sep. share | shift | flips 1→0 / 0→1 |
|---|---|---|---|
| plain s1, reverse top10 group | +0.15 | −0.02 | 0.35 / 0.48 |
| plain s2, reverse top10 group | +0.41 | +0.02 | 0.15 / 0.39 |
| recon s1, reverse top10 group | +0.52 | +0.15 | 0.02 / 0.43 |
| data-aligned, reverse top10 group | +0.31 | −0.04 | 0.30 / 0.25 |
| recon s1, T2 top10 removed | +0.72 | +0.10 | 0.00 / 0.26 |
| ortho s1, T2 top10 removed | +0.38 | +0.01 | 0.12 / 0.34 |

What this shows, from these rows only (the other conditions are not computed yet):
- Mostly the shift is near zero while the flips are large in both directions: the flips come from the gap closing, not from a shift. The scores are spread wide relative to the gap (plain s1's original test scores: true mean 4.85, false mean −3.80, overall SD 4.82), so once the gap shrinks, statements on both sides of 0 cross. The shift shows this at a glance; the flip pair doesn't.
- recon s1 is the exception: shift +0.10 to +0.15 and flips almost only 0→1. Here the shift is the finding, and it's one signed number instead of an asymmetry you have to read off two rates.

Three caveats, so I'd add the shift rather than replace the flips:
1. **The macro shift can cancel across families.** Per family the shift ranges much more widely (plain s1 reverse: −0.32 to +0.10; ortho s1 T2 top10: −0.68 to +0.35). A macro value near 0 can hide one family going to "0" and another to "1". So also report the macro of |shift| (or per family in `details.md`).
2. **Shift and share are about class means; flips are about the decision.** The paper's manipulation claims ("reversing these components changes the verdict") are claims about the decision, and how many statements a given gap change flips depends on the spread and on where the threshold sits. The flips are also what the frozen plan lists, so dropping them would be a plan change; adding the shift isn't. Since the shift is added after looking at the data, label it descriptive, not pre-registered.
3. **Means of log-odds are sensitive to outliers** (an edit that removes digit mass can produce large |s|). Share has the same weakness. If that becomes a problem, medians are the fallback.

Proposal [assumed, not ratified]: a "shift" column next to every separation share in `tables.md` (termination T0–T3, A8, manipulation), macro signed plus macro |shift|, per family in `details.md`; flips stay. Figures: the manipulation figure shows share and shift per basis, with flips as the secondary panel. Shall I implement it?

---

26-10-30, Claude Opus 5.5 — *separation shift implemented*; same session as the answer above. Julian in chat, replying to that proposal:

C: (transcribed from chat) Yes, implement it

So the shift is added next to the flips, not in place of them [decided: the C: line above ratifies the proposal at the end of my answer]. Changes to `paper_tables.py`, outputs regenerated in `scratch/26-10-29_paper_writing/tables/` [verified: run 26-10-30, ~18 s, `make check` clean]:
- **Computation**: `sep_shift` (definition as in the answer above) and `add_shift`. When a json output is loaded, `load_with_shift` adds `shift` (pooled, per family, macro) and `abs_shift` (macro of the per-family |shift|) to every stored summary that has per-statement scores in its npz: the recovery states, the termination conditions (T1 test, T2, A8), the manipulation conditions and the val wide table. The OOD controls are computed against their own capped-subset reference pass, and the main conditions restricted to that subset against the full pass's scores on those rows, exactly as the stored summaries were made. Before adding a shift, the function recomputes the stored separation share (pooled and every family) from the same arrays and asserts equality. That checks the arrays are aligned with the summary [verified: equal for 208 recovery, 3,968 termination and 2,726 manipulation summaries; the count is printed in the head of `tables.md`].
- **Not covered** (only summaries were stored, no per-statement scores): T1's val ranking and the direction removal in 5a. Both tables say so.
- **Tables**: shift and |shift| next to the separation share in 1c (full model minus inactive components), T0 (inside the cells), T2 (all variants in `tables.md` and `details.md`) and the 4a groups. Shift only (no |shift|): the T1 test table in `details.md`, the A8 cells (signed, after the share; random = mean over the 3 draws), the 4b single conditions in `details.md` and the 4c top 5. `wide_val_reverse_<run>.csv` gets a pooled `shift` and a `<family>:shift` per family.
- **Figure**: new `figs/m_share_shift.png` (4a; same 2 × 2 layout and rows as `m_groups.png`): separation share ●, shift ◆, |shift| as a tick, test filled and OOD hollow, training prompt. It comes first in section 4a; the flip figure follows as the secondary one.
- The `tables.md` head defines the shift and marks it as added after the outputs were read (descriptive, not in the frozen plan).

Spot checks: recon s1's reverse-top10 shift in the new table is +0.149, the same as in the earlier probe (+0.15) [verified]. No interpretation of the new numbers in this entry (step by step, as before).

C: Does `tables.md` also have data about the giant cancelations activations of plain tPD?
Answer (26-10-30, Claude Opus 5.5): Not in `tables.md`; the data is in `details.md`, section *Write norms per layer (test, training prompt)*. It is computed in `paper_recovery.py` (`ci_counts_and_norms`) at the statement tokens on test. Per run and layer it gives |C|/|y| (C = the writes of all components at mask 1), |D|/|y| (D = y − C, Δ's write) and ⟨C, D⟩/|y|², the cancellation diagnostic. What it shows [verified: `details.md`, generated 26-10-30]:

| run | \|C\|/\|y\| (L11–L17) | \|D\|/\|y\| | ⟨C,D⟩/\|y\|² |
|---|---|---|---|
| plain s1 | 1.20–1.97 | 1.79–2.29 | −14.3 (L11), −3.6 to −6.1 (L12–L17) |
| plain s2 | 1.17–1.86 | 1.77–2.18 | −12.8 (L11), −3.6 to −5.9 |
| recon s1/s2 | 0.88–0.96 | 0.21–0.38 | −0.02 to +0.01 |
| ortho s1/s2 | 0.16–0.43 | 0.90–0.97 | −0.01 to 0.00 |
| untrained s1/s2 | 0.23–0.37 | 1.04–1.07 | −0.14 to −0.11 |

So plain tPD cancels and the other methods don't. In plain tPD the components and Δ each write more than the MLP output itself, in opposite directions. The active components' CI-weighted write is only 0.04–0.30 |y| (the |A|/|y| column in the same table), so the large C comes from the components' full writes at mask 1, which Δ cancels.

**The cancellation is much smaller than 4a's.** 4a (plain tPD at the statement end, before the paper's protocol) had |C|/|y| of 3.3–13.4 and ⟨C, Δ⟩/|y|² of −183 to −11 (faithful-targeting topic, entry with the table "How the existing runs would score"). That number was estimated from per-layer mean norms as (1 − |C|² − |D|²)/2. The same estimate on the paper's plain runs gives about −2.3 (L11) to −3.7 (L17) [concluded: arithmetic on the norms above, e.g. plain s1 L11: (1 − 1.318² − 1.984²)/2 = −2.34]. That is one to two orders of magnitude less than 4a. The phrase "giant cancellation" fits 4a, not the paper's plain runs; the paper should describe them as moderate cancellation.

**Two caveats for reading the numbers:**
- The stored ⟨C,D⟩/|y|² is a mean over positions of the per-position ratio, while the norm columns are ratios of means. The two don't match: −14.3 vs −2.3 at L11. So the ratio must be much larger at some positions than at others. Positions with small |y| inflate it [assumed: likely, but not checkable offline; only the means were stored].
- 4a and the paper runs differ in positions, LR and training length, so the comparison is between two plain-tPD set-ups, not a controlled test of what reduced the cancellation.

**The tables also show the effect in two places:**
- **A8 (2d):** removing the components' write span from the original MLP output barely changes the verdict, while removing the components themselves (T2) destroys the verdict separation. That is the functional sign of cancellation.
- **T0 (2a):** Δ alone.

Proposal [assumed]: move the write-norm table into `tables.md` as section 1d "Cancellation". Show both estimators, one row per run with the layer range (per layer stays in `details.md`). Quote 4a's values in the text as the earlier reference, and add a small figure of ⟨C,D⟩/|y|² per layer and basis. Shall I?

C: No, it's fine. Do you have an idea why plain and recon have these spikes in "functional active set: recovery of the top-m components". I woould have expected curves like for ortho as the most important components come first. I know this is a simplistic picture and the x-axis is logairthmic, but it still strikes me as a bit of strange that there are suddes spikes in the middle. 

Answer (26-10-30, Claude Opus 5.5): Short answer: the curve orders components by **mean CI** (as the frozen plan defines the active set), not by importance. Mean CI equals importance only in ortho. In plain the verdict needs a set of about 20 components together, and in recon the high-mean-CI components rebuild generic MLP output, not the verdict. I checked the ranking against T1, the single-removal drop of the verdict separation on val, from the stored outputs [verified: `recovery_<run>.json` (`mean_ci_val`, `active_set.curve`) and `termination_<run>.json` (`t1_val`, `ranked`), 26-10-30]:

**First, what the plot draws.** The curve is evaluated only at m = 1, 2, 4, 8, 16, 32, … up to the first grid point that reaches the target, plus the bisection points near it (`paper_recovery.py`). For plain s1 that is 1, 2, 4, 8, 16, 24, 26, 27, 28, 32. So the "sudden" rise is two straight segments over a doubling with no measurement in between (8 → 16 → 24). The rise could be gradual there; the plot can't tell. The negative values at small m are real measurements, though: the top 8 alone recover −0.82 (plain s1) and −0.55 (plain s2). Those components alone push the verdict further from the original than switching the decomposed MLP outputs off.

**ortho (monotone): one component dominates, and it is the most important.** Mean CI of the top component: ortho s1 L13:19 0.61 (the next 0.05), and it also has by far the largest T1 drop (0.47; next 0.07). ortho s2: L15:9 0.41 and L14:15 0.33 at the top; its T1 top 5 sit at mean-CI ranks 1–5 except L11:44 (rank 15). Here "most active first" really is "most important first", and each added component helps on its own. That is your expected picture.

**plain (negative, then a jump between 8 and 24): there is no dominant component, and the useful set works only together.** Mean CIs are flat and small: top 0.08–0.10, rank 16 about 0.03, rank 24 about 0.02 (both seeds). The T1-important components are all inside the top 29 by mean CI, but spread over it (plain s1: T1 top 10 at mean-CI ranks 1–25; plain s2: 1–29). So by m ≈ 24–32 the curve has collected all of them, and recovery reaches 0.89–0.92. Before that it is negative, which needs more than "important ones missing": a strict subset of the set is worse than no write at all. My candidate explanation [assumed, untested]: the plain components come in opposing pairs or groups (e.g. a "true"-pushing and a "false"-pushing side, as 4e's analysis found a "false" signal against a "true" default). A subset that contains one side but not its counterpart biases every verdict in one direction, which gives a large KL. The cancellation table (answer above) is consistent with this, though it measures cancellation against Δ, not within the components. The new shift metric would test it: one-sided subsets should show a large |shift|. But the curve stored only KL recovered, no per-statement scores, so the test needs the top-m states rerun on a GPU (val, about 10 forward passes per run). I'm listing it, not running it.

**recon (dips to −0.26/−0.38 at m = 32, recovers only at 128–256): the mean-CI ranking has almost nothing to do with the verdict.** Recon components rebuild the MLP output, so mean CIs are high and broad (rank 1: 0.52–0.64; rank 256 still 0.047). The T1-important components sit deep in that order: recon s1's T1 top 10 at mean-CI ranks 1, 36–251; recon s2's at 34–248. Their individual T1 drops are also small (0.03–0.09). So the top-m curve measures how much of the generic MLP output has been rebuilt. Part of y without the rest moves the residual stream off its usual distribution [assumed], hence the dip; recovery comes only when most of y is back. This fits recon's large active sets (387 / 432 components).

**For the paper** [concluded from the above]: the active-set curve is not an importance curve. Its shape says how mean CI relates to function in each method, and that is worth one sentence: monotone only where activity and function coincide (ortho). If you want an importance-ordered curve, the same recovery evaluated along the T1 ranking would be one (GPU, val; also listed, not run). One caveat: T1 is measured in the full model with Δ, the curve without Δ, so the two orderings answer different questions.

C: ok, I keep that in mind for the paper. Another question:
Shouldn't we take for T0 also KL recovered so we have the same measurement?

Answer (26-10-30, Claude Opus 5.5): Yes. Δ-only KL recovered is also the measure your tuning axis (c) used ("Δ only's KL recovered, ideal 0"), so leaving it out of T0 was a gap. It was already stored (the recovery state `delta_only`; the script's recomputation check already covered it), so I added it without a new run. Section 2a of `tables.md` now starts with Δ-only KL recovered, pooled and macro, with the same 95% bootstrap intervals and the same denominator as the components-only recovery in section 1. The old verdict-measure cells follow below [verified: regenerated 26-10-30, `make check` clean; values equal to a direct read of `recovery_<run>.json`].

Pooled, test / training prompt (components only from section 1 for comparison):

| run | components only | Δ only |
|---|---|---|
| plain s1 / s2 | 0.961 / 0.958 | 0.081 / −0.050 |
| recon s1 / s2 | 0.974 / 0.971 | −0.238 / −0.440 |
| ortho s1 / s2 | 0.906 / 0.893 | 0.170 / 0.011 |
| untrained s1 / s2 | −0.002 / −0.003 | 0.888 / 0.984 |

Reading aids (no interpretation beyond the definitions):
- The two recoveries need not add to 1: KL is not additive over parts of the MLP output.
- Negative = Δ alone is further from the original verdict distribution than switching off all decomposed MLP outputs (recon on test).
- KL recovered counts every change of the verdict distribution, including confidence and digit mass. Separation share counts only the true/false discrimination, so the two can disagree. The OOD columns differ a lot from test (e.g. recon Δ only +0.34 to +0.48 on OOD vs negative on test); the full table has all four sets.

Question: T2 (top-k removal) also has KL recovered stored for every condition. Should T2 show it too, so that T0 → T2 → recovery use one measure throughout? It is one more column; the data is there.

C: Nah, that can wait. Another question: why does the label / ood curve not go up in "K1: probes on the k most selective single component"?

Answer (26-10-30, Claude Opus 5.5): Mostly because the threshold doesn't transfer, not the information. The probe's direction keeps working on OOD and gets better with k, as expected. But its threshold, fit on tiu train, is off on OOD, so accuracy stays flat. ortho s1 is the exception: it has no OOD truth information at all.

I checked this offline by refitting the K1 probes (coef features, label target) from the correlation npz with the same pipeline as `paper_correlation.probe_scores` [verified: `scratch/26-10-29_paper_writing/k1_ood_probe_check.py`, 26-10-30; it reproduces the stored OOD macro accuracy of all 36 probes exactly (9 bases × k = 1, 2, 5, 10), then adds three measures on the 13 ground-truth OOD files (4,292 statements, true share 0.509)]:
- *direction* = within-family AUC of the probe score, averaged over families (threshold-free)
- *bias* = share of OOD statements the probe calls true
- *oracle* = macro accuracy with each family's best threshold. This is an upper bound (the threshold is fit on the OOD labels), so it shows only what a recalibrated probe could reach.

| basis | k = 1 → 10: stored acc | direction (AUC) | predicted true (true 0.509) | oracle acc |
|---|---|---|---|---|
| plain s1 | 0.571, 0.562, 0.578, 0.686 | 0.801, 0.866, 0.942, 0.964 | 0.987 … 0.937 | 0.795 → 0.926 |
| plain s2 | 0.751, 0.770, 0.857, 0.617 | 0.915, 0.896, 0.934, 0.874 | 0.853, 0.821, 0.579, 0.042 | 0.899 → 0.844 |
| recon s1 | 0.590, 0.672, 0.668, 0.892 | 0.571, 0.826, 0.940, 0.960 | 0.984 … 0.589 | 0.692 → 0.943 |
| recon s2 | 0.854, 0.835, 0.788, 0.767 | 0.914, 0.895, 0.939, 0.933 | 0.692 … 0.829 | 0.900 → 0.902 |
| ortho s1 | 0.545, 0.519, 0.532, 0.532 | 0.546, 0.500, 0.495, 0.507 | 0.022 … 0.028 | 0.613 → 0.624 |
| ortho s2 | 0.512, 0.609, 0.642, 0.705 | 0.800, 0.806, 0.811, 0.840 | 0.907 … 0.657 | 0.781 → 0.805 |
| untrained s1 | 0.540, 0.675, 0.647, 0.658 | 0.550, 0.759, 0.773, 0.826 | 0.490 … 0.224 | 0.616 → 0.785 |
| untrained s2 | 0.609, 0.584, 0.634, 0.618 | 0.648, 0.652, 0.685, 0.719 | 0.337 … 0.195 | 0.677 → 0.712 |
| data-aligned | 0.813, 0.827, 0.830, 0.861 | 0.956, 0.983, 0.982, 0.975 | 0.378 … 0.586 | 0.920 → 0.941 |

What this shows:
- **The direction transfers and improves with k** for most bases (plain s1 0.80 → 0.96, recon s1 0.57 → 0.96, untrained s1 0.55 → 0.83). This is the rise you expected. It just doesn't reach the accuracy.
- **The threshold is off.** plain s1 calls 94–99% of the OOD statements true at every k, so its accuracy stays near 0.57–0.69 even with a within-family AUC of 0.96. plain s2's drop at k = 10 (0.857 → 0.617) is the same thing flipped: the 10-component probe calls only 4% true, while its direction barely changes (0.93 → 0.87). With per-family thresholds, plain s1 would reach 0.80 → 0.93.
- **ortho s1 is a real failure**: within-family AUC ≈ 0.50 at every k, so its train-selective components carry no truth information on OOD. Recalibration can't fix that (oracle 0.61–0.62, probably mostly the oracle's own upward bias: picking the best threshold per family lifts even an uninformative score above 0.5 [assumed, not measured]).
- **Why the offset** [assumed]: the OOD files differ in form (conjunctions, German, other facts), which shifts the components' activation levels. An intercept and scale fit on tiu train then put most OOD statements on one side. It doesn't happen in-distribution because test shares the train families.
- **Side note:** the stored "(pooled AUC)" in brackets mixes families, so it is neither the threshold-free per-family measure nor the accuracy. It is not a clean measure of the direction.

**For the paper** [concluded]: K1's OOD accuracy mixes two things, whether the components carry truth information on OOD and whether a tiu-calibrated readout fits OOD. Reporting the within-family AUC next to the accuracy separates them. That applies to K0 too: the residual-stream probe's OOD accuracy (0.856) is also below its pooled AUC (0.966).

Proposal [assumed]: add the within-family AUC (macro) and the predicted-true share as columns to K0 and K1 in `tables.md`, and switch the K1 figure to two rows (accuracy, within-family AUC). `paper_tables.py` would refit the probes from the npz (CPU, seconds) and assert the stored accuracies, as the check script does. Shall I?

C: Nah, let's go to my next question. I was expecting plain tPD to fail in some performance metric. I can't really see one (I see two or three hints). I would have thought that we could find a failure mode for plain's presumed self/re-construction.

Answer (26-10-30, Claude Opus 5.5): I think the failure is in the tables, but not in the metrics we called "performance". Recovery, termination and manipulation are all measured inside the decomposition, where components + Δ = W holds exactly. A decomposition that re-implements the verdict and cancels the rest with Δ passes them by construction. Recovery is its training objective. And removing or reversing components that Δ was cancelling injects a large uncancelled write, so termination and manipulation even look *strong*. The failure shows up only where plain is compared with (a) a random control inside the same decomposition or (b) the original model's own activations. All numbers below are from `tables.md` / `details.md` as generated 26-10-30 [verified: read from the files], test, training prompt, both seeds.

**1. Plain's top components are no more necessary than random ones (T2, section 2c).** Separation share after removing the top-k (by val T1 drop) vs the mean of 5 random k-subsets of the active set:

| k | plain s1 | plain s2 | recon s1 | recon s2 | ortho s1 | ortho s2 |
|---|---|---|---|---|---|---|
| 5 | 0.387 vs 0.455 | 0.821 vs 0.672 | 0.762 vs 0.994 | 0.892 vs 0.984 | 0.405 vs 0.769 | 0.527 vs 0.856 |
| 10 | 0.248 vs 0.242 | 0.393 vs 0.310 | 0.717 vs 0.974 | 0.848 vs 0.995 | 0.380 vs 0.924 | 0.303 vs 0.853 |

For recon and ortho the top components matter far more than random ones. For plain, removing any 10 of its active components destroys the separation about as much as removing the "most important" 10 (in plain s2 the random sets are even worse). The T1 ranking picks out nothing specific. Caveat: plain's active sets are small (27 and 44), so at k = 10 a random subset already covers a quarter to a third of S. That's why I stop at k = 10.

**2. The components' write span plays no role in the original model (A8, section 2d).** This is the most direct test of re-implementation. Project the span of a layer's 96 component write vectors out of the *original* MLP output, one layer at a time, and read the verdict separation:
- plain: 0.968–1.008 in every layer and both seeds, the same as the random 96-dimensional subspaces (0.98–1.01).
- recon: 0.68–0.93 and ortho: 0.80–1.05 (10 of 14 layer values 0.80–0.90). That is about the level of the top-96 principal directions of y (0.73–0.91).

So the subspace plain's components write into carries no more of the original verdict computation than a random subspace. Yet removing the components themselves destroys the separation (T2: 0.04 / 0.13 at k = 20). The decomposed model relies on a subspace the original model doesn't use. That's re-implementation, measured.

**3. The write vectors don't point along the truth direction (5a).** Max |cos(write vector, mass-mean truth direction)| over a layer's 96 write vectors: plain 0.044–0.084, at the level expected for 96 random vectors (0.042–0.046) and like the untrained decompositions. recon 0.17–0.41; ortho 0.06–0.15.

**4. Cancellation (details, write norms):** components at mask 1 and Δ each write 1.2–2.3 |y| against each other (answer above).

**Counter-hints, for honesty:**
- plain recovers the *best* on OOD (components only: 0.32–0.37 vs recon 0.18–0.23 and ortho 0.10–0.29). A re-implemented verdict readout may generalise well precisely because it targets the lens output.
- plain's Pile Δ-only KL (0.10) sits between ortho (0.02) and recon (0.14), so it is no side-effect outlier.

**Why the expected failure didn't appear in the verdict metrics** [concluded: follows from W = components + Δ and the cancellation norms]: every performance metric we planned is invariant to how the work is split between components and Δ, as long as the split reproduces W and the lens. Re-implementation changes the split, not W. So it can only be caught by tests anchored outside the decomposition: random controls of the same decomposition (1), the original model's activations (2, 3), or the norms (4).

**For the paper**, this may be the lesson candidate we were missing [assumed, a proposal]: "*L4: within-decomposition metrics (recovery, ablation, steering) cannot distinguish decomposing a mechanism from re-implementing it; tests anchored in the original model (span projection, alignment with model-derived directions) and random-component controls can.*" plain tPD is the worked example, with recon and ortho as the cases that pass. It fits L3 (the protocol lesson) but is more specific.

**A test that would make (1) sharper** [assumed; needs a GPU, listed, not run]: if plain's removal effects come from the size of the uncancelled write and not from function, a component's T1 drop should track the norm of its mask-1 write. Neither T1 per component nor per-component norms were stored in a way that allows this offline.

C: what surpsises me is that we don't see any great performance drop of Δ alone. If delta writes huge (or big) activations in new directions, this should interfere with the model more than the arguably more targeted decompositions.

Answer (26-10-30, Claude Opus 5.5): The training makes Δ neutral for the verdict, whatever its size. So Δ alone acts on the verdict like writing nothing, which is the all-off baseline (KL recovered ≈ 0). A big Δ could only show its damage in readouts that neither the training nor our metrics look at. Sources: training code read 26-10-30, numbers from `tables.md` / `details.md` as generated 26-10-30.

**1. What the training enforces** [verified: code]. On target data, Δ's mask is drawn uniformly from [0, 1] per position and layer (`spd/utils/component_utils.py`: `torch.rand(...)` unless `force_delta` is set, and the target losses don't set it). The components' masks are drawn from [CI, 1]. So the verdict must come out the same with Δ fully on, fully off or anywhere in between. To first order (R = the verdict's sensitivity to a write at the statement tokens): R(D) ≈ 0, and because D = y − C, also R(C) ≈ R(y). Δ alone writes D, so its effect on the verdict ≈ 0, the same as writing nothing (all-off). KL recovered ≈ 0 is what the training predicts, independent of |D| [concluded: first-order argument; the exact Δ-only state is never sampled, because the active components' masks stay ≥ CI ≈ 1, so the measured deviations from 0 are the non-linear part]. The big part of D lies in directions the verdict path doesn't read. Any reading would make the verdict depend on Δ's mask, and the loss penalises that. These are your "new directions". A8 (section 2d) shows the same thing from the original model's side.

**2. The data fit this, and Δ's norm doesn't predict the damage** [verified: section 2a, test, training prompt, pooled]. Δ only: plain 0.081 / −0.050 (all-off level), ortho 0.170 / 0.011, recon −0.238 / −0.440. If a large Δ hurt, plain would be clearly below 0. The only clearly negative method is recon, which has the *smallest* Δ (|D|/|y| 0.21–0.38, vs plain 1.77–2.29). Weak extra pressure from tuning: axis (c), "Δ only's KL recovered (ideal 0)", was one of the three axes by which the settings were picked by eye (experiments topic).

**3. On Pile, Δ alone is (nearly) a trained state** [verified: code]. In the non-target block of `spd/run_spd.py`, Δ's mask is forced to 1 (`force_delta=1.0`), the importance-minimality coefficient is doubled (`nontarget_impmin_coeff_ratio: 2`), and the components' masks are random in [CI, 1]. With CI ≈ 0 on Pile, the next-token output must hold for any component masks from 0 to 1, which includes Δ alone. So the components must write little that matters on Pile. The large cancelling writes exist where the components fire, at the statement tokens; the write norms were measured only there. On Pile, C is probably small, so there Δ ≈ y [assumed: C's norm on Pile not measured]. Pile Δ-only KL (val): plain 0.10, ortho 0.02, recon 0.14. Plain is about 4× ortho, in the direction of your intuition. But recon, which doesn't cancel, is worse, so Pile KL doesn't single out cancellation. The Pile KL with all MLPs off was not measured, so I can't say how large 0.10 is.

**4. Smaller points.**
- *Where the perturbation sits*: only the statement tokens are decomposed; per the config header, "the question tokens run on the original weights" [assumed: config comment]. The verdict at the last token sees the statement only through attention, and that is exactly the path the training made insensitive to D.
- *"Huge" relative to what*: compared with the original model, Δ alone changes each layer's write by −C (|C| = 1.2–2.0 |y|), all-off by −y (|y|). So Δ alone perturbs 1.2–2× as much as the baseline it is measured against, not orders of magnitude more. Compared with the residual stream it is probably smaller still [assumed: residual norms not measured].
- *OOD*: Δ only stays at the all-off level for plain (−0.044 / 0.072), but recon and ortho keep 0.34–0.45 (section 2a, ood/train). So plain's verdict-neutral Δ also holds on OOD, while recon's and ortho's Δ carries part of the OOD verdict. This fits plain having the best OOD recovery (0.32–0.37).

**Where your intuition should show, and a test** [assumed: prediction, not run; needs a GPU]. Look at readouts nobody trained on. At the statement tokens the original model also predicts the next statement token, and other prompts read the same positions. The part of D the verdict ignores need not be ignored there. If you're right, then on those readouts plain's Δ alone does clearly *more* damage than all-off, while recon's and ortho's don't. Test: next-token KL within the statement (Δ only vs all-off vs original, per method), plus a different question about the same statement (e.g. the direct-question recall prompt from the steering topic). The second prompt doesn't test this: it differs from the training prompt only from the statement's last token on. If the prediction holds, it is another test anchored in the original model for L4 (the lesson candidate that within-decomposition metrics can't tell decomposing from re-implementing).

Listed, not run. Shall I put it on the GPU list with the two from earlier (top-m states with per-statement scores; T1 drop vs write norm)?
OUTDATED (26-10-30): the reason given above for why the second prompt doesn't test this is wrong. The second prompt is a different question (" Is this statement true (1) or false (0)? Answer: "), but it is still a truth verdict, the readout the training targeted. So it tests transfer, not side effects; see the answer to the next comment.

C: Let's not put anything on a GPU list yet. I think yes, we'd need a test that shows that the components strange writes have some side-effects, that the other's don't have.
- KL on statements is a good idea
- Also a different OOD stimulus. The stimulus could also just be tailored for a single domain. Any idea?
So let's discuss these and then we can put them on a list.

Answer (26-10-30, Claude Opus 5.5): Agreed, and here is a concrete design to discuss. Nothing goes on a list yet.

**What the test has to show.** The hypothesis: plain's components have large writes that Δ cancels, in directions the verdict path doesn't read. Every intervention that breaks the cancellation (Δ alone, removing or reversing components) injects these writes, and they should disturb readouts other than the verdict. For recon and ortho, an intervention removes or changes a real part of the MLP output. That also disturbs other readouts, but in proportion to what it removes. So the main difficulty is **the comparison**, not the readout: every intervention on layers 11–17 damages something.
- **Δ alone vs all-off** is confounded: Δ carries a different share of y in each method (ortho's Δ ≈ 0.9–0.97 |y|, so Δ alone is close to the original on any readout; recon's Δ is small, so Δ alone is close to all-off). Δ alone shows little about *strange* writes.  
C: I think Δ alone is reasonable. We should measure: how much does this tPD technique come close to our Platonistic ideal of a truth mechanism decomposition. On the statement tokens. In theory there should be little truth computation except on the last tokens, so the components shouldn't matter. If we see that plain behaves differently from the other two that supports that reading.
Answer (26-10-30, Claude Opus 5.5): Agreed: Δ alone, measured as next-token KL at the statement positions (A0). Report it against both references: KL(original ‖ Δ alone) (ideal 0), and as a share of the all-off KL. Expectations, written before any run [assumed: predictions]:
- ortho: close to the ideal. Its Δ is 0.90–0.97 |y| and its components are small.
- recon: near the all-off level *by design*. Its components rebuild most of the MLP output (|C| ≈ 0.9 |y|), so Δ alone is close to removing the MLPs. That misses the ideal, but expectedly so; it says nothing about cancellation.
- plain: your reading predicts clearly *worse than all-off*, because Δ alone writes y − C with |C| = 1.2–2.0 |y|.

So "plain behaves differently from the other two" has to take this specific form. recon will differ from ortho as well, so if plain lands at the all-off level like recon, the test doesn't separate them.

On "little truth computation except on the last tokens": the causal layer scan supports this if "last tokens" includes the object word (`convos/julian/26-10-10_bolder_experiments_SUMMARY.md`, *Causal layer scan*). Ablating MLPs 11–17 at the end token alone kept 82% of the verdict separation, and the object token carries the rest. The tiu data help here. True and false statements of a family share the prefix and differ only from the object on ("The city of Lu'an is in" + "Turkey" / "China"). Before the object, the statement's truth is not yet determined, so an ideal truth decomposition should change nothing there. So I'd report A0 separately for the prefix positions (before the object) and for object + end; the prefix is the cleanest test of your ideal. One exception: at "is in" the model already recalls the country (China), which the later judgement uses. So that one position gets its own column.
- **Fair comparison: side effect at matched verdict effect.** For each basis, T2's top-k removals (k = 1, 5, 10, 20) and random k-sets, plus the reverse-top10 group. Plot the off-target damage against the verdict separation drop. Prediction [assumed: written before any run]: plain's points lie clearly above recon's, ortho's and the data-aligned basis's (more side effect per unit of verdict effect), and plain's top-k lie on its random-k points. The data-aligned basis is the natural reference: a basis of the real activations, with nothing to cancel.  
C: I don't fully understand, but it also sounds complicated.
Answer (26-10-30, Claude Opus 5.5): Dropped. The idea in one sentence: every intervention damages something, so I wanted to compare methods at equal verdict effect. With Δ alone read against the all-off baseline, we don't need that.
- **Size or direction?** Also log the norm of the injected write (|Σ removed c_i| / |y|) and plot the damage against it. If plain lies on the others' curve, its side effects come only from the size of its writes. If it lies above, the directions are unnatural too. This answers your "strange writes" question in its strong form.  
C: Also here I don't quite follow. We don't need any more investigation in the internals of the activations.
Answer (26-10-30, Claude Opus 5.5): Dropped.
- **Extra state:** 1c's "full model minus inactive components" (all components with CI ≈ 0 removed, Δ kept). For plain it already costs the verdict more than for the others (separation share 0.45 / 0.41 vs recon 0.68 / 0.83, ortho 0.60 / 0.68; `tables.md` 1c). So plain's "inactive" components are not inert.  
C: This goes in a different direction. This was evidence for "inactive components still cary function".
Answer (26-10-30, Claude Opus 5.5): Agreed, that's a separate question; it stays out of this test.

**Readouts (stimuli), in three classes. Only the first two test side effects.**

*A. Truth-independent readouts* (the answer is the same for a true and a false statement; reference = the original model's distribution, measured by KL):
- **A0, next-token KL within the statement** (your first point). It needs no new prompt: the logits at the statement positions are computed in every verdict forward pass anyway, they just weren't stored. Caveat: some positions are knowledge positions (e.g. "The city of Lu'an is in" → the next token is the model's recall of the country). A truth mechanism may legitimately change those. So I'd report knowledge positions (the token before the object) separately from the rest.  
C: yes, sounds good
- **A1, copy**: "\nRepeat the sentence word for word: " + teacher-forced KL over the copied tokens. Works for every family and OOD file. Caveat: copying may run mostly on early-layer token identity, so it may be insensitive to layers 11–17 [assumed].  
C: yes, let's skip this
- **A2, content questions tailored per domain** (your second point). The tiu data carry the metadata needed (`subject`, `city`, `country`, `n1`, `n2`):

| family | stimulus | answer |
|---|---|---|
| cities | "\nThe country named in this sentence is" | `country` (the stated one, not the true one) |
| animal_class | "\nThe animal named in this sentence is the" | `subject` |
| element_symb | "\nThe chemical symbol in this sentence is" | the stated symbol |
| sp_en_trans | "\nThe Spanish word in this sentence is '" | `subject` |
| larger/smaller_than | "\nThe first number in this sentence, in digits, is" | `n1` |
| neg_* | "\nDoes this sentence contain the word 'not'? Answer yes or no:" | yes |

Two choices still open: first answer token only, or teacher-forced over the whole answer; and checking that the original model answers these near-perfectly before we use them.  
C: Isn't that also most likely determined by early layers?
Answer (26-10-30, Claude Opus 5.5): Probably yes [assumed]. Naming the country or the animal mostly needs the token's identity, which the residual stream at that position carries from the embedding on, so writes in layers 11–17 likely leave it intact. That's the same reason as for copying. Dropping A2.
C: I was thinking more of the line. "<statement> The humor in the statement is: ". The idea is that plain might write into some other conceptual space (even though it still has to be truth related in some way as it's steering the verdict). The problem is that I don't know which space it could be.
Answer (26-10-30, Claude Opus 5.5): Good idea. It reads the statement through a question the training never saw, about a property other than truth. Since we don't know which space plain might write into, I'd use a small set of questions rather than one. All of them use the verdict's answer format (a digit after "is "), so the comparison with the verdict stays direct:
- "\nThe amount of humor in this statement is " (your line, in the training prompt's form)
- "\nThe amount of emotion in this statement is "
- "\nThe amount of surprise in this statement is " (truth-adjacent: false statements should be surprising)
- "\nThe number of words in this statement is " (a digit question with a checkable answer, unrelated to truth)

Why the digit format: one candidate for the "other space" is the digit itself. Plain may have learned "push '1' vs '0'" rather than "true vs false" [assumed: hypothesis]. If so, its interventions also move the digits in the humor and word-count questions, and a word answer ("funny" / "serious") would miss that. Readout: KL of the digit distribution (0–9, renormalised) vs the original model, plus the shift of the mean digit.

Caveats:
1. Check the original model first: the digit must be a plausible next token. For the verdict prompt the model's own next token was often ":" (bolder-experiments topic), so the digit distribution is read after the space anyway. Humor ratings of factual statements may be almost constant, which is fine for KL, since any shift counts.
2. All-off shifts these readouts too, so Δ alone is compared with all-off, as in A0.

States: original, all-off, Δ alone, and the reverse-top10 group. The last one because "plain steers the verdict and may write into some other space while doing so" is a claim about the steering edit itself.

*B. Knowledge readouts of the same statement (single domain, exploratory).* The fact the verdict needs, asked after the statement: cities "\nIn fact, the city of {city} is in" → `correct_country`; larger_than "\nThe larger of the two numbers is"; sp_en_trans "\nIn English, '{subject}' actually means '". This is not a clean side-effect test, because a real truth mechanism might use the recall, so the prediction runs in both directions. But it is the most interesting readout for L2 (the target loss decides what gets decomposed; a verdict lens may decompose the readout of truth, not truth itself). I'd do it on **cities only**: richest metadata, and the most knowledge-driven verdict.  
C: I don't think this will work. Even if had substantially altered the models perception (which we most likely have not), I'm pretty certain it would recover in this test.
Answer (26-10-30, Claude Opus 5.5): Agreed, dropped.

*C. Other truth questions (transfer, not side effects).* A truth question in a different form, e.g. "\nIs this correct? Answer Yes or No:" (Yes/No instead of 1/0). Correction to my last answer: the second prompt is already a different question (" Is this statement true (1) or false (0)? Answer: "), but with the same 1/0 truth verdict, so it is a transfer test like C. I marked the wrong reason OUTDATED above. Here the prediction runs the other way: a truth mechanism should also lose its separation under a top-k removal in C; a re-implemented verdict readout might not. Recovery on the second prompt is already high for plain (0.94), so C's 1/0 version adds little; the Yes/No version would be new.  
C: that could be worth a shot.
Answer (26-10-30, Claude Opus 5.5): Then it's the transfer test, e.g. "\nIs this statement true? Answer Yes or No: ", scored as log P('Yes') − log P('No'). The current score is fixed to the '1'/'0' tokens, I believe [assumed: not checked in the code], so this needs a small change. States as for the second prompt: the recovery states and T2. Prediction [assumed]: if plain has re-implemented a 1/0 readout rather than truth, its components-only recovery and T2 effects fall more under Yes/No than recon's and ortho's.

**My recommendation** [assumed]: A0 + A2 on test as the side-effect test, with the matched-verdict-effect comparison and the write norms logged. Add B on cities. A1 and C only if they are cheap, since they use the same machinery. A0 costs almost nothing, because it rides on forward passes we run anyway. The rest needs a new stimulus per prompt. `paper_common.py` already handles two stimuli (`verdict_prompts`), but I haven't checked how much code a per-family stimulus with an answer needs. Everything here is exploratory (decided after reading the data), so the predictions above should be fixed before the run, as for the frozen plan.

Questions for you:
1. Which classes: A only (side effects), or A + B (+ C)?
2. Content questions (A2) as the single-domain version, or all families?
3. Which states: T2 top-k and random-k, reverse-top10, Δ alone, full minus inactive — all of them, or a subset?
C: answers above

---

26-10-30, Claude Opus 5.5 — *side-effect test: revised design after Julian's comments*; same session. Julian answered the design in place (C: comments in the answer above, my replies beneath each). What remains [assumed: my consolidation of those comments; not yet ratified as a list]:

- **Question** (Julian's framing): how close does each method come to the ideal of a truth-mechanism decomposition? At the statement tokens, the components should matter only where truth is computed, so Δ alone should leave everything else as the original has it.
- **State**: Δ alone (plus the original model and all-off as references). For the digit questions, also the reverse-top10 group.
- **Readout 1, next-token KL within the statement** (A0): KL(original ‖ Δ alone), as a share of the all-off KL. Reported separately for the prefix (before the object; truth not yet determined), the recall position before the object, and object + end.
- **Readout 2, digit questions about other properties** (Julian's humor line): humor, emotion, surprise, word count, each as "\nThe amount of … in this statement is " with a digit answer; KL of the digit distribution and the shift of the mean digit.
- **Readout 3, Yes/No truth question** (transfer, not side effects): recovery states and T2 under "\nIs this statement true? Answer Yes or No: ".
- **Dropped**: matched-verdict-effect comparison, write-norm analysis, full-minus-inactive, copy (A1), content questions (A2), knowledge readouts (B).
- **Predictions, fixed before any run**: ortho near the ideal; recon near the all-off level by design; plain clearly worse than all-off if its cancelling writes have side effects. Only that last outcome separates plain from recon.
- Data: test split. All of it exploratory (designed after reading the outputs), labelled as such in the paper.

Open: is this the list to put on the GPU list? Also: the original-model check for the digit questions (is a digit a plausible answer?) can run first, as a short job.

C: For the Yes/No, we should iterate on some prompts first that shift most mass into "Yes"/"No".
The rest is fine and should be put on a GPU list. One caveat. I don't want to spent lots of compute on something without signal. I want on the GPU list some probes first and then when they are promising further evaluations.
Answer (26-10-30, Claude Opus 5.5): Done: the GPU list is now a section in this topic's SUMMARY (*GPU list*), in stages [decided: the C: comment above approves the design and asks for probes first]. Each stage runs only if the one before shows signal. The signal criteria below are my proposal [assumed], written before any run so they can't be fitted to the outcome; please confirm or change them.

**Stage G0: original model only (short job, no decompositions).**
- Yes/No prompt iteration (your point): 6–8 candidate prompts on a val subset (about 20 statements per family). Measure the probability mass on the Yes/No tokens (with and without a leading space), accuracy and AUC. Pick the prompt with the most Yes/No mass among those with accuracy close to the 1/0 prompt's.
- Digit questions: the same check for humor, emotion, surprise and word count. Is a digit a likely next token, and does the answer vary across statements? A question whose digit mass is low gets rephrased or dropped.
- Go to G1 with the prompts that pass.

**Stage G1: signal probes, one seed per method (s1 of plain, recon, ortho, plus untrained s1), val subset.**
- Next-token KL within the statement: original, all-off, Δ alone; split into prefix, recall position, object + end.
- The digit questions under Δ alone and the reverse-top10 edit.
- Yes/No: components-only recovery, compared with the 1/0 prompt on the same statements.
- Signal criteria [assumed: proposal]:
  - (a) Next-token KL: plain's Δ-alone KL is clearly above the all-off KL (share > 1, bootstrap interval excluding 1) while recon's is not, in at least the prefix or the object + end positions.
  - (b) Digit questions: plain's KL or digit shift under Δ alone or reversal clearly exceeds recon's and ortho's on at least one question.
  - (c) Yes/No: plain's recovery falls clearly more from 1/0 to Yes/No than recon's and ortho's.
- Each readout that meets its criterion goes to G2. One that doesn't is reported as a null probe and stops.

**Stage G2: full evaluation, only for readouts with signal.** All 9 bases (both seeds of each method, both untrained, data-aligned), test split, bootstrap intervals as in the paper tables; for Yes/No also T2.

**Code needed** (not written yet): the edit hook in `paper_common.py` covers the states. New: storing the logits at the statement positions, the prefix/object position split per family, digit scoring for the new questions, and a Yes/No token pair in place of '1'/'0'. Cost per stage to be measured in a smoke test.

Not on the list: the two GPU tests from earlier answers (top-m states with per-statement scores; T1 drop vs write norm per component). The write-norm one falls under your "no more investigation of the internals". Say if you want either back.

C: I don't want to put the others on a GPU list. I don't see a need for them (yet). So put them on a shelf. I think what we also need is a ortho s3. I've got the feeling that s2 got screwed.
Answer (26-10-30, Claude Opus 5.5): The two other tests (top-m states with per-statement scores; T1 drop vs write norm per component) are shelved: a *Shelf* line in the SUMMARY's GPU list section, no GPU.

**Ortho s3: on the list [decided: C: comment above].** A third seed is worth it: ortho is the method that passes the original-model tests, and its two seeds disagree. Before it runs, three points.

**1. Is s2 the broken one? By the training's own numbers, no** [verified: `tables.md` sections 0, 1, 2a, and the K1 answer above]:
- Val readout s1 / s2: components-only KL recovered 0.907 / 0.906, Pile Δ-only KL 0.024 / 0.023, PGD 2.75 / 2.54, L0 2.8 / 3.2. These are equal within the seed spread of the other methods.
- s2's outlier is the second prompt: val 0.808 vs 0.934; test pooled 0.735 vs 0.925, macro 0.162 vs 0.833; Δ alone −0.545 vs 0.148.
- But s1 has its own outlier. Its train-selective components carry no truth information on OOD (K1 within-family AUC 0.50–0.55 at every k, vs s2's 0.80–0.84).  
C: this could also be interpreted as another point against s2. The story would go like this: s1 is a decently targeted decomposition of the truth mechanism on the target data. On OOD data it fails because it's missing important pieces of their relevant mechanisms.
C: anyway, we'll see what s3 says
Answer (26-10-30, Claude Opus 5.5): Agreed, s3 decides. One check on that story: it predicts that s1 recovers less of the verdict on OOD than s2 does, because pieces are missing. The data show the opposite. Components-only KL recovered on OOD, pooled, s1 / s2: 0.198 / 0.168 (training prompt), 0.291 / 0.101 (second prompt) [verified: `tables.md` section 1]. So s1's K1 failure is about its train-selective *single* components carrying no OOD truth information; the decomposition as a whole still recovers at least as much as s2. Your story doesn't fit these numbers well, though it isn't ruled out [concluded].

So each seed fails one generalisation test and passes the other. "s2 got screwed" holds for the second prompt; on OOD it is s1. What makes you think s2? If it is the second prompt, a third seed tells which pattern is typical. That's a good reason to run it, but it doesn't make s2 a failed run.

**2. Conditions, so that a third seed can't become seed selection** [assumed: proposal; please confirm]:
- s3 is added, not a replacement. All three seeds are reported in every table, s2 included, whatever s3 shows. Decided now, before s3 exists.  
C: s3 will replace s1 or s2 if they agree well with the other. Of course the other seed will be reported but not shown everywhere. If s3 does not agree with s1 and s2 across the board then we'll deal with that somehow (which we won't plan now)
Answer (26-10-30, Claude Opus 5.5): Understood; it's your call [decided: this C: comment]. Two things make it hold up under review:
1. **Fix "agree well" now, before s3 exists.** Otherwise the choice of which seed to show is made after seeing all three. Proposal [assumed]: a short list of headline numbers, chosen now:
   - components-only KL recovered (test, both prompts)
   - T2 top-10 separation share
   - A8 drop (all-96 span)
   - K1 OOD within-family AUC at k = 10

   s3 "agrees with" s1 (or s2) if it lies closer to that seed on most of them; the other seed moves to the appendix. If s3 is closer to neither seed on most of them, it is your "deal with it later" case.
2. **Say it in the paper.** The main figures then show the two agreeing of three seeds, which understates the seed spread. One sentence ("seed 3 was added because seeds 1 and 2 disagreed; the figures show the two that agree; all three are in the appendix tables") is enough. The appendix tables carry all three.

Should I write the headline list into the plan as above, or do you want different numbers on it?
- Same config as the finals (`config_paper_ortho.yaml`, seed 3, 5,000 steps, PGD as in the finals), and the full phase D analysis with the same scripts, so s3 appears in every table and figure.
- Analysis on the same GPU type as phase D (H200 NVL): the active set turned out GPU-sensitive (ortho s1: 29 vs 32 components on two GPU types).
- The paper states why the seed was added (seeds 1 and 2 disagreed on second-prompt and OOD transfer). Adding seeds only where results disagree is a form of optional stopping, and saying so keeps it honest.

**3. Ortho only, or s3 for all three methods?** With 3 seeds for ortho and 2 for the others, the seed spreads aren't comparable. The rule "no method ranking where the seeds overlap" then holds ortho to a stricter standard than the others. s3 for plain and recon costs two more trainings (the estimate before the finals was ~1 h each on the RTX PRO 6000 [assumed: estimate in the experiments LOG, not a measurement]) plus their phase D analyses. My recommendation: all three if the budget allows, ortho only otherwise, with the asymmetry stated.
C: we have to compute all numbers for s3 as well, yes.
Answer (26-10-30, Claude Opus 5.5): Noted: the full phase D analysis for s3 [decided]. I read this as **ortho s3 only**, no s3 for plain and recon. Correct me if you meant all three.

Logistics: training needs a host outside China (the training code calls the Hugging Face Hub API), and the WandB cap needs room for one more 238 MB checkpoint per run (run register). The same rental can run G0/G1 of the side-effect test.
OUTDATED (26-10-30): the checkpoint is 333.5 MB, not 238 MB (run register); see the answer to the WandB comment below.
C: true, we might have to make some space on wandb. Add this to the plan.
Answer (26-10-30, Claude Opus 5.5): Added to the GPU list in the SUMMARY [decided: this C: comment]. Correction: the paper checkpoints are **333.5 MB**, not 238 MB as I wrote above. 238 MB is the figure in the paper configs' comment, but the run register lists 333.5 MB for all six paper finals [verified: `convos/julian/26-10-03_wandb_storage_RUNS.md`]. The register's 13 checkpoints add up to about 4.0 GB [concluded: sum of the register's size column]; the storage topic projected ≈ 4.36 GB of total usage after the finals [assumed: projection there, not re-measured]. One more checkpoint would bring it to roughly 4.3–4.7 GB, under the 5 GB cap but tight. Plan:
1. Before the rental, read the actual usage via the WandB API (read-only).
2. If it doesn't leave a clear margin, you pick a checkpoint to delete. The candidates with byte-identical local copies are the tiu untrained baseline `s-7fad0c14`, arm A `s-bd23f0d1` and Exp 1 `s-cd129592` (238–241 MB each). Agents never delete without asking.
3. Alternative: train s3 with `sync_checkpoints_to_wandb: false`, keep the local copy, and upload it in phase E only if it becomes a canonical run.

Add the row to the run register when s3 is launched.
C: I don't want to choose any headline numbers in advance. It's overcomplicating things. Just compute it and report. I'll do the picking then.
Yes ortho only.
Your plan is fine, except that in 2. you can delete whatever you want as long as it's not the papers runs (plain/recon/ortho x s1/s2)Answer (26-10-30, Claude Opus 5.5): Both noted [decided: the C: comment and the line "Yes ortho only." above]: no headline numbers fixed in advance; s3 gets the full analysis, all numbers reported, and you pick the seeds to show once you've seen them. s3 is for ortho only. The one thing I'd still keep is the sentence in the paper that the seeds shown were chosen after the results were seen, with all three in the appendix. It costs nothing and answers the obvious reviewer question. The SUMMARY's GPU list is updated; the headline-list proposal is dropped.

The configs' comment "Final checkpoint only (238 MB each)" in `config_paper_{plain,recon,ortho,untrained}.yaml` was wrong (run register: 333.5 MB) and caused the error marked OUTDATED above; corrected to 333.5 MB in all four (comment only, no setting changed).

---

26-10-31, Claude Opus 5.5 — *GPU list prepared: side-effect probe script and ortho s3 bundle*; new session. I have read CLAUDE.md, INDEX.md, this topic's SUMMARY and the LOG from *Where plain tPD fails* on, the experiments topic's SUMMARY, `docs/CODING.md`, `paper_common.py`, `paper_recovery.py`, `paper_manipulation.py` and the phase D driver scripts. Julian in chat: "I want to do the extra GPU work of the GPU list now", then, after asking why I write `.sh` driver scripts (answer: detached multi-hour chains with a progress file, as in phase D): "Ok, keep going. I'll rent the machine meanwhile".

**Code** [verified: `make check` clean, `tests/test_paper_common.py` 13 passed; not yet run on a model]:
- New `spd/experiments/lm/honesty_targeted_decomposition/paper_side_effects.py`, two modes. `g0`: original model, 20 val statements per dataset; 8 Yes/No prompt candidates (mass on 'Yes'/'No' and ' Yes'/' No', accuracy, AUC, next to the 1/0 prompt) and the 4 digit questions (humor, emotion, surprise, word count: digit mass, spread and label means of the mean digit); prints decoded prompts with top next tokens. `g1`: one decomposition on val, states full (sanity), all_off, delta_only, components_ci, reverse_top10 (= manipulation's group, from the T1 ranking in the termination JSON). Readouts: next-token KL at the statement positions per class (prefix / recall / object_end) as a share of all-off with bootstrap intervals; the digit questions (renormalised 0–9 KL, mean-digit shift); Yes/No vs 1/0 KL recovered. All per-statement log-probs stored in the npz.
- Object position per family = the token holding the first character after the last anchor (" in ", " a(n) ", " symbol ", " mean(s) '", " than "). CPU check over all val (986) and test (1,794) statements [verified]: every statement has positions in all three classes; token ids from the offset tokenization equal the batch ids; one tagged example per family read by eye, e.g. cities `' is'/P ' in'/R ' Nigeria'/O '.\n'/O`, sp_en_trans `" '"/R 'dog'/O`, element_symb `' symbol'/R ' N'/O 'p'/O`.
- `paper_common.Prompt.name` widened from `Literal["train", "second"]` to `str` (only printed), so probe prompts carry their own names.

**Ortho s3 bundle** in `~/spd_out/26-10-29_paper_writing/gpu/`: `ortho_s3.yaml` = phase C's `ortho_s2.yaml` with `seed: 3` and name `paper_final_ortho_s3` (diff checked: only those lines; validated by `Config`), `s3.sh` (training, then the phase D chain with the same arguments as `phase_d/h200_paper_d/rest.sh`, then `paper_checks.py`), `g0.sh`, the B2 scores and check script, the termination JSONs of plain s1, recon s1, ortho s1, untrained s1, and the untrained s1 checkpoint (G1 runs).

**WandB** [verified: API storage counter 26-10-31]: 4.42 GB used; s3's checkpoint (333.5 MB) brings it to ≈ 4.75 GB, under the 5 GB cap, so no deletion. (Julian's last word on deletions, under *side-effect test: revised design*: I may delete anything except the six paper runs; the SUMMARY said "Julian picks", corrected there.)

Plan for the rental (H200 NVL, the phase D GPU type, so the active set is comparable): `s3.sh` and `g0.sh` at once; G0 results go here before G1, whose prompts I pick by the rule in the GPU list (most Yes/No mass among prompts about as accurate as 1/0; digit questions with low digit mass dropped).

---

26-10-31, Claude Opus 5.5 — *G0 results and G1 prompt choice*; same session. Rental: vast.ai instance 58527420, H100 SXM 80 GB (power limit 700 W), $2.98/h, rented by Julian (an earlier instance, 58526819, disappeared during setup). **Not the H200 NVL of phase D**: the active set was GPU-sensitive between the RTX 4090 and the H200 (ortho s1 32 vs 29), so ortho s1's recovery is rerun here (`rec_check.sh`) and compared with the H200 output before s3's numbers count as comparable [assumed: Hopper ≈ Hopper, unverified]. Ortho s3 training started 11:43 UTC (run `s-6acf4d80`, 3.26 it/s, ~26 min).

**G0** [verified: one run, original model, 316 val statements = 20 per dataset (fewer where a dataset is smaller); outputs `/root/gpu/side/g0.{json,npz}`, to be copied to `~/spd_out/26-10-29_paper_writing/gpu/`]:
- Reference, 1/0 training prompt: digit mass 0.977, accuracy 0.937, AUC 0.987.
- Yes/No: every candidate answers with the leading-space tokens; the bare 'Yes'/'No' get < 3e-5 of the mass even after a prompt ending in ': ' (the model writes ' No' after the space token). With ' Yes'/' No' (mass, accuracy, AUC): is_true_sp 0.826 / 0.649 / 0.976; is_true 0.910 / 0.687 / 0.963; is_true_nl 0.998 / 0.861 / 0.946; true_or_false 0.989 / 0.797 / 0.924; correct 0.992 / 0.756 / 0.923; **question** 0.924 / 0.908 / 0.990; **same_line** 0.991 / 0.896 / 0.995; same_line_sp 0.886 / 0.851 / 0.990. Every prompt leans towards 'No' (share answered Yes 0.15–0.41 at threshold 0, vs 0.50 for 1/0), which is why no accuracy reaches 1/0's while the AUCs do. The KL readout uses the whole distribution, so the threshold doesn't matter for it.
- Digit questions, all usable (digit mass; mean digit ± std across statements; true / false mean): humor 0.995; 0.94 ± 0.60; 0.91 / 0.97. emotion 0.984; 0.60 ± 0.31; 0.69 / 0.50. surprise 0.998; 0.73 ± 0.36; 0.74 / 0.72. words 1.000; 5.99 ± 0.84; 6.05 / 5.94 (argmax '1' for 94 statements, else 5–9). Surprise does not separate true from false in the model, so it is not the truth-adjacent readout I expected.

**G1 prompt choice** [assumed: my call under the rule in the GPU list]: the rule (most Yes/No mass among prompts about as accurate as 1/0) picks same_line. But same_line puts the question on the statement's line, so the statement ends with '.' instead of the merged '.\n': its last decomposed token differs from training as well as the answer format (as for the second prompt). question keeps the statement tokens identical to training, so it isolates the format change. G1 is cheap, so both run. All four digit questions run. `paper_side_effects.py g1` now takes several Yes/No keys and always scores ' Yes'/' No' (flag dropped, see the G0 finding).

G1 started 11:50 UTC (`g1.sh`: plain s1, recon s1, ortho s1, untrained s1, one after the other, next to the s3 training and the recovery check).

---

26-10-31, Claude Opus 5.5 — *G1 results: next-token damage yes, Yes/No no, digits mixed*; same session. G1 crashed once at once on all four runs (fire passes `question,same_line` as a tuple; fixed, nothing had been measured), then ran 11:52–11:59 UTC [verified: one run per basis, val, 986 statements; outputs copied to `~/spd_out/26-10-29_paper_writing/gpu/results/side/g1_<run>.{json,npz}`]. Sanity: the full state reproduces the original (statement KL 0.0000 everywhere). Bases: plain s1 `s-7152548c`, recon s1 `s-eeac663f`, ortho s1 `s-0c2fa0d7`, untrained s1 `s-d5c56fb0`.

**1. Next-token KL at the statement positions, Δ alone, as a share of the all-off KL** (pooled; 95% bootstrap within family; all-off KL 2.28 / 2.46 / 2.47 nats per position for prefix / recall / object_end, the same for every basis because it doesn't depend on the basis):

| basis | prefix | recall | object_end |
|---|---|---|---|
| plain s1 | 0.763 [0.752, 0.774] | **2.239** [2.177, 2.300] | **1.665** [1.628, 1.704] |
| recon s1 | 0.828 [0.820, 0.835] | 0.561 [0.535, 0.587] | 0.844 [0.822, 0.864] |
| ortho s1 | 0.131 [0.126, 0.135] | 0.609 [0.589, 0.629] | 0.801 [0.784, 0.818] |
| untrained s1 | 0.050 | 0.150 | 0.157 |

- Criterion (a) is **met**: plain's Δ alone is clearly worse than switching the MLPs off at the recall position and at object + end, recon's (and ortho's) are not. Per family at object_end, plain is above 1 in 8 of 10 families (0.88 cities, 0.91 animal_class are the exceptions; up to 2.49 smaller_than); recon 0.45–0.92, ortho 0.29–0.95.
- Where: the excess damage sits where plain's components are active (the recall token and from the object on), not in the prefix (0.76 < 1). That fits the earlier reading that the large cancelling writes exist where the components fire [concluded: matches the position pattern; C's norm per position class not measured].
- Reading against Julian's ideal [concluded]: at the prefix, ortho comes closest (0.13: its components barely matter where truth isn't determined yet), recon and plain are near all-off there. From the recall token on, every trained method changes the next-token prediction substantially (0.56–0.84 of all-off), so none is "only truth"; plain alone goes beyond the all-off damage.
- Untrained: Δ alone ≈ original (0.05–0.16), as expected for components that write almost nothing; it is the "everything in Δ" reference.
- Side observation: ortho's components alone (mask = CI, Δ off) are worse than all-off at every position class (1.23–1.30), plain's only at object_end (1.43); recon's components alone stay close to the original (0.07–0.15), as its reconstruction term demands.

**2. Digit questions** (digit KL of Δ alone and of reverse-top10, as a share of the all-off digit KL, [95% interval], signed mean-digit shift):

| question | state | plain | recon | ortho |
|---|---|---|---|---|
| humor | Δ alone | 0.94 [0.87, 1.02] | 1.21 [1.07, 1.35] | 1.27 [1.16, 1.39] |
| emotion | Δ alone | 0.44 [0.42, 0.47] | 0.77 [0.73, 0.81] | 0.57 [0.54, 0.60] |
| surprise | Δ alone | 0.45 [0.42, 0.48] | 1.23 [1.17, 1.29] | 1.48 [1.43, 1.53] |
| words | Δ alone | **1.14** [1.05, 1.22], shift −0.66 | 0.83 [0.76, 0.89], +0.30 | 0.86 [0.80, 0.92], −0.02 |
| humor | reverse-top10 | 0.91 | 0.71 | 1.08 |
| emotion | reverse-top10 | 0.43 | 0.26 | 0.47 |
| surprise | reverse-top10 | 0.60 | 0.32 | 1.09 |
| words | reverse-top10 | **1.12** [1.04, 1.21] | 0.22 [0.21, 0.24] | 0.62 [0.57, 0.67] |

(untrained ≤ 0.19 everywhere.) Criterion (b) as I wrote it ("plain clearly above recon and ortho on at least one question") is met by the word count only, under both edits. On humor and surprise plain does the *least* damage under Δ alone; ortho's Δ alone is worse than all-off on humor and surprise (1.27, 1.48). So the digit questions show no general "plain writes into another conceptual space" [concluded]; the word count is a single exception, and found among 4 questions × 2 states, so it could be chance selection. reverse-top10 is not comparable across methods anyway, because the groups differ in how much they move the verdict.

**3. Yes/No transfer** (components-only KL recovered, 1/0 → Yes/No):

| basis | 1/0 | question | same_line |
|---|---|---|---|
| plain s1 | 0.964 | 0.654 (−0.31) | 0.829 (−0.14) |
| recon s1 | 0.977 | 0.818 (−0.16) | 0.868 (−0.11) |
| ortho s1 | 0.906 | 0.557 (−0.35) | 0.783 (−0.12) |

Criterion (c) **not met**: ortho drops as much as plain. Under the staged rule this is a null probe and stops. All methods transfer only partly to a Yes/No answer when the statement tokens are the training ones (question), better when the question shares the statement's line (same_line).

**Decision needed (G2)** [assumed: my reading of the staged rule; the criteria were my proposal, not confirmed by Julian]: (a) goes to G2 (all 9 bases, test, intervals). (b): only the word count passed, and the pattern across questions runs against the hypothesis, so I'd stop it or carry only the word count along at no extra cost (it rides on the same script). (c) stops. G2 needs one addition: the data-aligned basis (the script only handles decompositions so far). The current rental (busy with s3 for ~3 more hours) can run G2 at no extra rental time.

Also [verified: rec_check on the H100, val part]: ortho s1's active set on the H100 is 29 components with KL recovered 0.8622, identical to the H200 (29, 0.8622). Per-statement comparison pending.

---

26-10-31, Claude Opus 5.5 — *G2 for all readouts*; same session. Julian in chat, after the G1 results and the Yes/No prompt questions:

C: (transcribed from chat) These probes are cheap, right? So let's run them all. I find it interesting that yes/no performance drops for all of them I also find it interesting that plain is beter for other digit questions which is surprising for me. I don't know if we use the numbers for the paper, but let's collect the data. (It is cheap right?)

So G2 runs for every readout, regardless of the G1 criteria [decided: the comment above]. Cost [assumed: estimate from G1's 1.5–3 min per basis on 986 val statements]: test has 1,794 statements; with T2 under both Yes/No prompts ≈ 10–15 min per basis, ≈ 1.5–2 h for the 9 bases + s3, on the rental already running for s3.

Code [verified: `ruff`, `basedpyright` clean; smoke test on the instance pending]: `paper_side_effects.py g1` now takes `--basis decomposition|data_aligned` (the data-aligned basis via `paper_common.make_basis`, as in phase D; no CI network, so no components_ci state), `--split val|test` and `--t2`. T2 = phase D's removal sets under both Yes/No prompts: top-k of the T1 ranking for every k in the termination JSON's `ks`, and for an active set 5 random size-k subsets, redrawn with paper_termination.py's generator (`default_rng(0)`; they are its first draws, so the sets are identical). Two sets (top10, rand10 #0) also run under the 1/0 prompt, to compare per statement with phase D's stored scores: this checks the redraw and the GPU. Outputs renamed to `side_<run>_<basis>_<split>.{json,npz}` (G1's files renamed to `…_decomposition_val`, local and on the instance). Driver `~/spd_out/26-10-29_paper_writing/gpu/g2.sh`: smoke test (2 statements per dataset, plain s1 and data-aligned), then the 9 bases on test one at a time, then ortho s3 once its termination JSON exists.

---

26-10-31, Claude Opus 5.5 — *token identification checks*; same session. Julian in chat, before G2 started: "before you start the script. Is there any reasonable test you can run to make sure that the token identifications for the runs are actually correct?" Three checks of `object_token` / `position_classes` (scripts `~/spd_out/26-10-29_paper_writing/gpu/{token_check,recall_check}.py`):

1. **String check** [verified: CPU, all 4,742 tiu statements of train, val and test]: the tokens before the object index decode exactly to the text before the object, and the tokens from it to the statement's last token decode to the object plus the final punctuation and the merged '\n'. 0 failures.
2. **Pair check** [verified: CPU, same statements]: grouped by family and the text before the object, the 1,186 groups that hold both true and false statements all have identical token ids before the object and the same object index, so the prefix positions really are the same input for true and false. sp_en_trans and neg_sp_en_trans have only 5 such groups each (true and false translations rarely share the Spanish word); there the object is the English word by construction.
3. **Model check** [verified: H100, original model, 1,794 test statements]: P(object's first token) at the recall position (obj − 1), and as a control one position earlier. Where the object follows from knowledge in a fixed phrase, the model predicts the correct object at exactly that position: element_symb true 0.77 (top-1 1.00), sp_en_trans true 0.86 (0.97), neg_element_symb false (stated symbol correct) 0.36 (0.89), neg_sp_en_trans false 0.50 (0.88); the same families with a wrong object ≤ 0.09. One position earlier: ≤ 0.011 in every family. Cities are weaker than I assumed: true 0.10 (top-1 0.09); after "… is in" the model often continues differently, so for cities "recall position" overstates what happens there (the position itself is right, by checks 1 and 2). Comparisons and animal classes are low by nature (many continuations fit after "than" / "is a").

Also [verified: rec_check finished]: ortho s1's recovery rerun on the H100 completed; per-statement comparison with the H200 outputs pending (files to be copied).

G2 started after these checks.

---

26-10-31, Claude Opus 5.5 — *H100 reproduces the H200; untrained s2 recreated byte-identically*; same session.

**GPU type** [verified: ortho s1's recovery rerun on the H100, outputs `~/spd_out/26-10-29_paper_writing/gpu/results/rec_check/`, compared array by array with `phase_d/h200_paper_d/recovery/`]: same active set (29 components, same order), the active-set curve equal to 4 decimals at every point, per-statement differences with median 0 in every array and at most one bf16 step (0.375–0.5 log-odds, a few OOD statements), verdict signs equal for 100% of test and 99.98% of OOD statements, pooled summaries equal to 4 decimals (components-only KL recovered test 0.9056 / 0.9250, OOD 0.1979 / 0.2914, training / second prompt). So s3's analysis on this H100 is comparable with the H200 runs [concluded]. (The 4090-vs-H200 difference in phase D came from a different GPU architecture; Hopper H100 and H200 agree.)

**Untrained s2**: uploading phase D's checkpoint ran at ~18 KB/s, so I stopped it and recreated both untrained decompositions on the instance exactly as phase D's `resume12.sh` did (`config_paper_untrained.yaml`, seed patched to 1 / 2, `steps: 0`; script `gpu/untrained.sh`). sha256 equal to phase D's checkpoints for both seeds (s1 `a4bde0cf…`, s2 `ee9055b3…`) [verified]. The recreated seed-2 checkpoint was copied to `/root/spd_out/spd/s-25d2cca7/`, the path phase D's termination JSON names, so G2 uses that name. The recreation made two WandB runs without checkpoints (`s-f48d3fc2`, `s-cad69866`, `sync_checkpoints_to_wandb: false`).

G2 smoke tests passed (plain s1 and data-aligned; T2 sets as in phase D: 31 and 15). G2 on the 9 bases is running.

---

26-10-31, Claude Opus 5.5 — *destroy instruction*; same session. Julian in chat:

C: (transcribed from chat) also when you have all the data kill the instance

Plan [decided: the comment above]: when G2 (9 bases + s3) and the s3 chain (through the invariant checks) are done, copy `/root/gpu/` (outputs, logs, progress files) and s3's run directory to `~/spd_out/26-10-29_paper_writing/gpu/results/`, verify with sha256 on both sides, then destroy instance 58527420 and confirm the destroy output. (Status so far: G2 plain s1 on test done; its T2 sets reproduce phase D's stored scores per statement, max |diff| 0.25 log-odds, median 0.)

---

26-10-31, Claude Opus 5.5 — *G2 results, 9 bases on test*; same session. G2 ran 12:33–13:12 UTC without failures [verified: one run per basis, test, 1,794 statements; tables generated from the stored summaries by `~/spd_out/26-10-29_paper_writing/gpu/g2_tables.py` into `…/gpu/results/g2_tables_test.md`; raw outputs `…/results/side/side_<run>_<basis>_test.{json,npz}`]. Check: plain s1's redrawn T2 sets reproduce phase D's stored 1/0 termination scores per statement (top10 and rand10 #0: max |diff| 0.25 log-odds, median 0, verdict sign equal for 99.6% / 99.7%; original scores max 0.125) [verified]. Exploratory (designed after the phase D outputs were read). Ortho s3 follows after its termination.

**1. Next-token KL at the statement positions, Δ alone, share of all-off [95%]** (all-off 2.28 / 2.31 / 2.54 nats per position):

| basis | prefix | recall | object_end |
|---|---|---|---|
| plain s1 / s2 | 0.69 / 0.46 | **2.16** [2.12, 2.20] / **1.34** [1.31, 1.37] | **1.55** [1.52, 1.58] / **1.94** [1.90, 1.98] |
| recon s1 / s2 | 0.82 / 0.84 | 0.54 / 0.67 | 0.85 / 0.87 |
| ortho s1 / s2 | 0.12 / 0.10 | 0.62 / 0.52 | 0.78 / 1.04 [1.02, 1.06] |
| untrained s1 / s2 | 0.05 / 0.04 | 0.14 / 0.12 | 0.15 / 0.13 |
| data-aligned | 0.80 | 0.59 | 0.78 |

The val result replicates on test in both plain seeds: plain's Δ alone damages the next-token predictions clearly beyond all-off at the recall position and from the object on, every other basis stays below all-off there (ortho s2 at object_end only marginally above, 1.04) [concluded: intervals don't overlap 1 for plain, both seeds]. At the prefix ortho is near the ideal (0.10–0.12); plain, recon and data-aligned are at 0.46–0.84. reverse-top10 (same table in the file): plain 2.20 / 1.74 at object_end, ortho 1.23 / 1.44, recon 0.24 / 0.18, data-aligned 0.72.
OUTDATED (26-10-31): ortho s3 (no cancellation: |C|/|y| 0.16–0.41, <C,D> ≈ 0) also exceeds all-off at recall (1.87) and object_end (1.37), so this test does not single out plain or cancelling writes; see the entry *ortho s3 results; G2 reading corrected*.

**2. Digit questions, Δ alone** (digit KL share of all-off): no consistent method pattern. humor: plain 0.84 / 1.18, recon 1.20 / 1.22, ortho 1.20 / 1.75, data-aligned 1.12. emotion: plain 0.45 / 0.78, recon 0.79 / 0.86, ortho 0.53 / 0.55. surprise: plain 0.48 / 0.39, recon 1.23 / 1.07, ortho 1.45 / 1.05, data-aligned 0.90. words: plain 0.98 / 0.88, recon 0.73 / 0.64, ortho 0.73 / 0.91. **The val word-count signal (plain 1.14) does not replicate on test** (plain s1 0.98 [0.92, 1.04]); what replicates is that plain's Δ alone disturbs surprise least (both seeds 0.39–0.48 vs ≥ 1.05 for recon and ortho). Under reverse-top10, ortho s2 is an outlier: humor 8.81, surprise 3.90, emotion 1.31, words 1.32 (ortho s1: 1.06 / 1.07 / 0.44 / 0.55); data-aligned humor 2.44.

**3. Yes/No transfer, components-only KL recovered (1/0 → question / same_line):**
- plain s1 / s2: 0.961 → 0.662 / 0.781; 0.958 → 0.680 / 0.773
- recon s1 / s2: 0.974 → 0.814 / 0.840; 0.971 → 0.848 / 0.846
- ortho s1 / s2: 0.906 → 0.553 / 0.716; 0.894 → **0.244 / 0.326**

Every method loses recovery under Yes/No; recon loses least, ortho s2 collapses, as it did on the second 1/0 prompt (val 0.808, test pooled 0.735). Δ alone under same_line is strongly negative for every trained basis (−0.34 to −1.32), so the verdict-neutrality training gave Δ holds for '1'/'0' only.

**4. T2 under Yes/No** (separation share left, top-k / mean of 5 random k, question prompt): plain s1 k = 10 0.31 / 0.32, k = 20 0.11 / 0.08; plain s2 k = 10 0.48 / 0.43, k = 20 0.17 / 0.32 (random worse than top); recon s1 k = 10 0.74 / 0.98, recon s2 0.92 / 0.97 (k = 20: 0.66 / 0.99); ortho s1 0.39 / 0.76, ortho s2 0.32 / 0.83. So phase D's finding transfers to the Yes/No readout: plain's T1 ranking picks out nothing specific (top ≈ random), recon's and ortho's do [concluded]. same_line gives the same picture.

**Reading for the paper** [assumed: proposal, not discussed with Julian]: (1) is a further original-model-anchored test for L4 (within-decomposition metrics can't tell decomposing from re-implementing): plain's Δ alone, the state the training made verdict-neutral, is the only one that damages ordinary next-token prediction beyond removing the MLPs. (2) is null for the method question. (3) shows limited transfer for all methods and singles out ortho s2. (4) repeats T2's L4 evidence on an untrained readout.
OUTDATED (26-10-31): ortho s3 (no cancellation: |C|/|y| 0.16–0.41, <C,D> ≈ 0) also exceeds all-off at recall (1.87) and object_end (1.37), so this test does not single out plain or cancelling writes; see the entry *ortho s3 results; G2 reading corrected*.

---

26-10-31, Claude Opus 5.5 — *Julian away: autonomous to the end*; same session. Julian in chat:

C: (transcribed from chat) I'm away so if there's any problem in the later runs, you have to make your own judgment calls. So don't want for any confiramition from me. Tasks ends when you have destroyed the instance and gathered the data

Status 13:13 UTC: s3 termination in its last part (OOD, second prompt); s3's active set has 58 components (s1 29, s2 31) [verified: s3_term.log, T1 over 58 candidates].

---

26-10-31, Claude Opus 5.5 — *ortho s3 results; G2 reading corrected; instance destroyed*; same session, Julian away (autonomous, see *Julian away*).

**Runs finished** [verified: progress files]: s3 chain 12:31–13:37 UTC (recovery, termination with A8, correlation, manipulation, directions), invariant checks **53 PASS, 0 FAIL** (the same 53 as every other trained decomposition in phase D); s3's G2 13:25. Everything copied to `~/spd_out/26-10-29_paper_writing/gpu/results/` (`instance_gpu/` = the instance's `/root/gpu/`: s3's phase D outputs in `out/` and `recovery/`, G0/G1/G2 in `side/`, logs, progress files, scripts; `run_s-6acf4d80/` = s3's run directory with `model_5000.pth`, also on WandB), sha256 identical on both sides for all 134 files [verified]. Instance 58527420 destroyed 13:4x UTC (`vastai`: "destroying instance 58527420", none listed afterwards) [verified].

**Ortho s3 next to s1 / s2** (test unless noted; pooled) [verified: read from the stored JSONs]:

| | s1 | s2 | s3 |
|---|---|---|---|
| active set | 29 | 31 | **58** |
| components-only KL rec., training / second prompt | 0.906 / 0.925 | 0.893 / 0.735 | 0.875 / 0.797 |
| Δ only KL rec., training / second | 0.170 / 0.148 | 0.011 / −0.545 | −0.107 / −0.078 |
| OOD components-only, training / second | 0.198 / 0.291 | 0.168 / 0.101 | 0.171 / 0.093 |
| T2 top10 vs random-10 separation share (macro) | 0.380 / 0.924 | 0.303 / 0.853 | 0.554 / 0.852 |
| A8 all-96 span, separation share per layer | 0.84–1.05 | 0.80–1.00 | 0.80–0.99 |
| G2 Δ alone, statement KL share of all-off (prefix / recall / object_end) | 0.12 / 0.62 / 0.78 | 0.10 / 0.52 / 1.04 | 0.14 / **1.87** / **1.37** |
| G2 Yes/No components-only (question / same_line) | 0.553 / 0.716 | 0.244 / 0.326 | **0.017** / 0.436 |
| write norms |C|/|y| per layer, <C,D>/|y|² | 0.16–0.43, ≈ 0 | 0.16–0.42, ≈ 0 | 0.16–0.41, ≈ 0 |

On transfer (second prompt, OOD, Yes/No) s3 sits with s2, not s1; its active set is twice as large; on T2 its top-10 matter less than s1's and s2's [concluded: from the table; no seed choice made, that is Julian's]. The paper tables (`paper_tables.py`) do not include s3 yet: the script reads phase D's directory and a fixed run list.

**Correction of the G2 reading.** s3 has no cancellation (norms above, like s1 and s2; plain: |C| 1.2–2.0 |y|, <C,D> −3.6 to −14), yet its Δ alone damages the statement's next-token predictions beyond all-off (recall 1.87 [1.84, 1.92], object_end 1.37 [1.35, 1.40]), as plain's does. So next-token damage beyond all-off does not identify cancelling writes or re-implementation, and it is not specific to plain [concluded: one counterexample with no cancellation]. The passages in *G2 results* that said so are marked OUTDATED. What remains: plain exceeds all-off in both seeds, ortho in one of three seeds (s3; s2 marginally at object_end), recon in none. Possible reading [assumed]: removing even a small C (0.3 |y|) can hurt more than removing y entirely when the remaining D is an off-distribution write; the norm of C doesn't decide it. Also s3 resembles plain on the surprise digit question (Δ alone 0.42 vs plain 0.39–0.48, recon 1.07–1.23, ortho s1/s2 1.45 / 1.05). The T2-under-Yes/No result (plain top ≈ random; recon and ortho, s3 included, top < random: s3 0.51 / 0.90 at k = 10) is unaffected.

Tables with all 10 bases: `…/gpu/results/g2_tables_test.md`.

---

26-11-04, Claude Opus 5.5 — *ortho s3 added to paper_tables.py*; new session. I have read CLAUDE.md, INDEX.md, this topic's SUMMARY, the LOG entries from *Julian away* on, `docs/CODING.md` and `paper_tables.py` in full. Julian in chat:

C: (transcribed from chat) Can you update `tables.md` with the ortho s3 results?

**Changes to `paper_tables.py`:**
- `ortho s3` (`s-6acf4d80`) added to `FINALS`, so every table and figure now shows 10 bases.
- Each basis carries its output directory (`Obj.src`, class `Roots`), because s3's outputs live in `~/spd_out/26-10-29_paper_writing/gpu/results/instance_gpu/{out,recovery}/` and not in phase D's directory. New argument `--s3`.
- Seed 3 is drawn dotted in the line plots and with a gray outline in the K2 dot plot.
- The two fixed grids (CI histograms 2×4, K2 scatter 3×3) now use one row per method (plain, recon, ortho; seeds side by side), then a row for the untrained and data-aligned bases.
- Counts in the check lines are computed, not hard-coded. "Both seeds" is now "every seed".
- The `tables.md` header says s3 was trained and analysed on an H100 SXM, not phase D's H200 NVL. It also says ortho s1's active set reproduced there (29 components, 0.8622; LOG entry *G1 results…*) and that s3 has no phase C readout (section 0 table).

Command (CPU, ~30 s):
`python spd/experiments/lm/honesty_targeted_decomposition/paper_tables.py --phase_d ~/spd_out/26-10-22_tpd_truth_paper/phase_d/h200_paper_d --s3 ~/spd_out/26-10-29_paper_writing/gpu/results/instance_gpu --phase_c ~/spd_out/26-10-22_tpd_truth_paper/phase_c/instance_paper_c/rounds --out scratch/26-10-29_paper_writing/tables`

**Checks** [verified: run 26-11-04, `make check` clean]:
- All recomputation checks pass for the 10 bases, including the shift checks: 236 / 4,496 / 3,008 recovery / termination / manipulation summaries.
- No number of the 9 earlier bases changed. Against a copy of the previous output, every removed line of `tables.md` is header or count text. All 76 removed lines of `details.md` are per-family rows that reappear unchanged once the new s3 column is dropped. The 6 earlier `wide_val_reverse_*.csv` files are byte-identical.
- s3's rows match the s3 table in *ortho s3 results…* (test unless noted; pooled):

| measure | s3 |
|---|---|
| components-only KL recovered, training / second prompt | 0.875 / 0.797 |
| same on OOD | 0.171 / 0.093 |
| Δ-only KL recovered, training / second prompt | −0.107 / −0.078 |
| active set | 58 |
| T2 top-10 separation share vs random 10 of the active set | 0.554 vs 0.852 |

New observation [verified: recomputation check line]: the all-off KL (the denominator of KL recovered, with the decomposed MLP outputs set to 0, so independent of the basis) is identical across the 8 phase D runs (max |diff| / mean = 0). Between s3's H100 run and the H200 runs it differs by at most 1.33e-02 of the mean, on the worst statement. This is GPU numerics. It shifts s3's KL recovered a little relative to the others but is much smaller than the seed differences [concluded: size comparison only, not propagated through to the ratios].

Still open, unchanged: Julian picks which ortho seeds go into the main figures.
