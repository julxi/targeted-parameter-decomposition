# SUMMARY: Review of FEEDBACK.md entries

**Last updated:** 26-10-06 (review complete: F2, F5b, F6, F8, F10 adopted; F5a not for now; F7 and the reader-side sentence rejected)

Julian asked to go through FEEDBACK.md (26-10-06, right after a sync). The agent triaged all 10 entries, numbered F1–F10 in file order (table in the LOG, entry *triage of all FEEDBACK.md entries*).

- **Closed before this review (5):** F1, F3, F4 and F9 were adopted as rules earlier; F10's bug is fixed in code.
- **Adopted in this review** [decided: Julian, `C: (transcribed from chat)` in the LOG]:
  - F2: chat decisions get transcribed verbatim in that turn's LOG entry (CLAUDE.md, `[decided]` bullet).
  - F6: predictions about experiments not yet run are `[assumed]` (CLAUDE.md, `[concluded]` bullet).
  - F8 (reframed after Julian pointed out that code and config comments sit outside the memory system and may be stale): a claim resting on such a comment is `[assumed]`, and the comment's decisive words are quoted, not paraphrased (CLAUDE.md, *Quantitative Claims*).
  - F10: before wrapping an external CLI with captured output, check `--help` for prompts and assert on the success output (`docs/CODING.md`, *Assertions and error handling*).
  - F5b: the sync inserts `OUTDATED` markers next to reversed LOG passages that lack one; that is its only change to existing LOG text (`docs/procedures/sync.md`). It went adopted → withdrawn → adopted. Julian first withdrew it to keep LOGs append-only. Then the agent pointed out that CLAUDE.md's `OUTDATED` bullet already sanctions such markers ("how you warn readers inside an append-only LOG"; 23 exist in the LOGs), and Julian re-adopted it. In-LOG markers therefore stay allowed.
- **Not adopted for now** [decided: Julian, "on the fence"]: F5a (whoever reverses a decision adds the marker in the same turn), because of the cognitive load on the agent making the change. With F5b, missed markers are added at the next sync.
- **Rejected** [decided: Julian]: F7 (describe a method's objectives as aims), because it is too specific; and a proposed reader-side sentence for CLAUDE.md ("check the SUMMARY before relying on an old LOG passage"), because every line in the always-loaded CLAUDE.md costs tokens in every session.
- **Open:** only whether to report the generic fixes (F2, F5b, F6) upstream to the Epistemic Memory template; Julian hasn't raised it.
- Placement principle [concluded]: CLAUDE.md is loaded in every session, so only the short, general rules go there.

- See also: [convos/julian/26-10-06_truth_writing_components_SUMMARY.md] — source of F7, F8, F9 (the pkill rule was adopted there).
- See also: [convos/julian/26-10-05_no_truth_baseline_SUMMARY.md] — source of F6 (the disproved control-probe prediction).
- See also: [convos/julian/26-10-05_vast_rentals_SUMMARY.md] — source of F10 (`vastai destroy` without `-y`).
