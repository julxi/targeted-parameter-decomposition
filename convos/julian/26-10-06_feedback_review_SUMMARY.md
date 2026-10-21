# SUMMARY: Review of FEEDBACK.md entries

**Last updated:** 26-10-21 (round 2 in progress: (1), narrow F5a, append-only wording and single `OUTDATED` definition adopted; (2) and (3) awaiting Julian)

Julian asked to go through FEEDBACK.md (26-10-06, right after a sync). The agent triaged all 10 entries, numbered F1–F10 in file order (table in the LOG, entry *triage of all FEEDBACK.md entries*).

- **Closed before this review (5):** F1, F3, F4 and F9 were adopted as rules earlier; F10's bug is fixed in code.
- **Adopted in this review** [decided: Julian, `C: (transcribed from chat)` in the LOG]:
  - F2: chat decisions get transcribed verbatim in that turn's LOG entry (CLAUDE.md, `[decided]` bullet).
  - F6: predictions about experiments not yet run are `[assumed]` (CLAUDE.md, `[concluded]` bullet).
  - F8 (reframed after Julian pointed out that code and config comments sit outside the memory system and may be stale): a claim resting on such a comment is `[assumed]`, and the comment's decisive words are quoted, not paraphrased (CLAUDE.md, *Quantitative Claims*).
  - F10: before wrapping an external CLI with captured output, check `--help` for prompts and assert on the success output (`docs/CODING.md`, *Assertions and error handling*).
  - F5b: the sync inserts `OUTDATED` markers next to reversed LOG passages that lack one; that is its only change to existing LOG text (`docs/procedures/sync.md`). It went adopted → withdrawn → adopted. Julian first withdrew it to keep LOGs append-only. Then the agent pointed out that CLAUDE.md's `OUTDATED` bullet already sanctions such markers ("how you warn readers inside an append-only LOG"; 23 exist in the LOGs), and Julian re-adopted it. In-LOG markers therefore stay allowed.
- **F5a (whoever reverses a decision adds the `OUTDATED` marker in the same turn):** set aside on 26-10-06 because of the cognitive load on the agent making the change. On 26-10-21, after that day's sync had to add 30 markers, a narrow version was adopted [decided: Julian, transcribed chat in the LOG, entry *narrow F5a adopted*]: when an entry says it corrects or contradicts an earlier passage, the agent marks that passage in the same turn (CLAUDE.md, append-only paragraph). It applies only where the agent has already located the passage; reversals nobody calls out are left to the sync (F5b).
- **Rejected** [decided: Julian]: F7 (describe a method's objectives as aims), because it is too specific; and a proposed reader-side sentence for CLAUDE.md ("check the SUMMARY before relying on an old LOG passage"), because every line in the always-loaded CLAUDE.md costs tokens in every session.
- **Open:** only whether to report the generic fixes (F2, F5b, F6) upstream to the Epistemic Memory template; Julian hasn't raised it.
- **Round 2 (26-10-21)** (FEEDBACK entry *living-document drift after state changes*; LOG entries from *round 2: proposals for the living-document-drift entry* on):
  - (1) adopted [decided: Julian, transcribed chat]: the `Last updated` line is the date plus the current state, replaced on each update, not a changelog (`docs/procedures/sync.md` step 2 and the CLAUDE.md SUMMARY sentence); the remaining chain in the truth-writing SUMMARY was fixed.
  - (2) stale tags after an answer, open: the sync's C: check would also retag the items an answer settles; for within a session, the agent proposes one clause in CLAUDE.md *Keeping files current* ("when an answer settles an item, retag that item there too") or nothing, since a sample of three syncs showed only one clear case.
  - (3) cross-topic status changes, open: a `docs/CODING.md` rule that names the owning topics (rentals → vast-rentals LOG; WandB runs and checkpoints → the run register and the WandB-storage LOG) and asks for their SUMMARY and INDEX row to be updated in the same turn; alternative: an object → topic list in INDEX.md. No rule for commit status.
  - Append-only wording and `OUTDATED` format adopted [decided: Julian, transcribed chat; LOG entry *append-only paragraph and OUTDATED definition applied*]: Julian noted that `C:` answers and `OUTDATED` markers are inserted in place. The name "append-only" stays; the CLAUDE.md paragraph now says existing text is never edited or deleted and lists the in-place exceptions (`C:` comments, answers beneath them, `OUTDATED` markers). The `OUTDATED` format, previously written four inconsistent ways, is defined once in the CLAUDE.md marker bullet: `OUTDATED (<date>): <reason, pointer to the replacement>`. Tags inside LOGs are never updated in place; current status lives in later entries and the SUMMARY.
  - Of the other three 26-10-21 FEEDBACK entries, the F5a one is adopted (above); the detach-pattern and conventions-recorded-only-where-they-happened entries are open.
- Placement principle [concluded]: CLAUDE.md is loaded in every session, so only the short, general rules go there.

- See also: [convos/julian/26-10-06_truth_writing_components_SUMMARY.md] — source of F7, F8, F9 (the pkill rule was adopted there).
- See also: [convos/julian/26-10-05_no_truth_baseline_SUMMARY.md] — source of F6 (the disproved control-probe prediction).
- See also: [convos/julian/26-10-05_vast_rentals_SUMMARY.md] — source of F10 (`vastai destroy` without `-y`).
