C: (transcribed from chat) Now I would like to go through feedback

---

26-10-06, Claude Opus 5.5 — *triage of all FEEDBACK.md entries*; same session as the 26-10-06 sync; I have read FEEDBACK.md, FUTURE_WORK.md, CLAUDE.md, `docs/CODING.md`, `docs/procedures/sync.md` and `docs/procedures/self_contained_check.md`

FEEDBACK.md has 10 entries (26-10-02 to 26-10-06). I numbered them F1–F10 in file order. For each one I checked whether its suggestion already appears in the rule files [verified: grep of CLAUDE.md, `docs/CODING.md` and `docs/procedures/` for the suggested wording, 26-10-06].

**Already closed (5):**
- F1 (26-10-02, pointers to deprecated docs): adopted 26-10-02; the rule is in CLAUDE.md.
- F3 (26-10-02, ssh job launching: `HF_HUB_OFFLINE`, pkill, login shell): adopted 26-10-03; in `docs/CODING.md` *Running jobs on a rented vast.ai GPU*.
- F4 (26-10-03, legacy refs gaining new info): adopted 26-10-03; the OUTDATED-marker rule is in CLAUDE.md.
- F9 (26-10-06, pkill killed the agent's own shell): adopted today; the sentence is in `docs/CODING.md`.
- F10 (26-10-06, `vastai destroy` without `-y`): the bug is fixed in `run_vast.py`. The general lesson (read `--help` before wrapping a CLI, check its output rather than its exit code) is in no rule file. See the proposal below.

**Open, with a concrete proposal each (6):**

| # | Problem | Proposed rule text | Where | Evidence it recurs | My recommendation |
|---|---------|--------------------|-------|--------------------|-------------------|
| F2 | Chat decisions paraphrased, not transcribed | "Chat messages that make a decision or trigger a change get transcribed verbatim in that turn's LOG entry; purely informational questions don't." | CLAUDE.md, `[decided]` bullet | The `[decided]` bullet already requires transcription *for a `[decided]` tag*. The gap: transcription gets skipped when no one tags anything. No new case since 26-10-02 that I know of. | Adopt (one sentence; it closes the gap that leaves later tags unratifiable) |
| F5a | Reversed LOG passages left without an OUTDATED marker | "When a new entry reverses or invalidates an earlier passage, insert an `OUTDATED (<date>): <reason, pointer>` line next to that passage in the same turn." | CLAUDE.md, append-only rule | **Recurred:** the 26-10-04 sync had to add 4 markers, and today's sync added 9 in four LOGs. | Adopt (strongest case on the list) |
| F5b | Sync agents unsure whether they may add markers | "Sync agents may insert `OUTDATED` markers next to reversed LOG passages; they don't edit LOG text otherwise." | `docs/procedures/sync.md`, step 2 | Today I had to state it in every sub-agent prompt by hand. | Adopt |
| F6 | A prediction tagged `[concluded]`, then disproved by the run (0.654 vs predicted ≥ 0.97) | "Predictions about experiments not yet run are `[assumed]`." | CLAUDE.md, `[concluded]` bullet | Today's sync found the same disproved prediction still in the no-truth-baseline SUMMARY, still tagged `[concluded]`. | Adopt (cheap, unambiguous) |
| F7 | A method's design intent written as its outcome ("tPD decomposes only what the target uses") | "When summarising a method from its paper, state objectives as aims and name the competing loss terms; check the description against our own runs before reusing it in a self-contained summary." | `docs/procedures/self_contained_check.md` (new step), not CLAUDE.md | One case, but it spread into two outward-facing summaries. | Adopt in the procedure file only. It is too specific for the always-loaded CLAUDE.md. |
| F8 | A code comment paraphrased into a hard constraint ("OOM'd beyond 5 matrices" vs the comment's "unvalidated guess") | "When a claim rests on a code or config comment, quote the decisive words instead of paraphrasing, especially for limits." | CLAUDE.md, *Quantitative Claims and Status Claims* | One case; same family as "never write a number from memory". | Adopt (one sentence in an existing section) |
| F10 | (general lesson, see above) | "Before wrapping an external CLI whose output is captured, check its `--help` for interactive prompts and assert on its success output, not its exit code." | `docs/CODING.md`, *Fail Loudly* | One case, but it cost money (an instance kept billing). | Adopt |

**Cost consideration** [concluded]: CLAUDE.md is loaded into every session, so each sentence added there costs reading time in every future session. That's why F7 and F10 go to files read on demand. F2, F5a, F6 and F8 are one sentence each, extending an existing bullet. Together they would add about four lines.

**Upstream note** [assumed: CLAUDE.md's memory-system part comes from the Epistemic Memory template, `FlorianDietz/EpistemicMemory`]: F2, F5a, F5b and F6 are generic workflow fixes, not specific to this project. If you want, they could also be reported to the template's author. That's your call.

**Bookkeeping after the decisions:** append each adopted entry's status to FEEDBACK.md in place, the way F1/F3/F4/F9 carry "Adopted <date>".

---

C: (transcribed from chat) F2 is fine. I don't know what are the consequences of F5a, F5b is fine but surpises me why this needs updating. I thought the whole idea of sync is to update the memory structure to be consistent. F6 is definitely the right thing. F7, I don't quite understand. For me that sounds like something very specific. About F8, code and config is not part of the epistemic system. So I guess everything that's written there could be outdated.

---

26-10-06, Claude Opus 5.5 — *F2, F5b, F6 adopted; F5a/F7/F8 discussed*; continuing the triage

**Adopted** [decided: Julian, transcribed above]:
- F2: CLAUDE.md, `[decided]` bullet, now ends "Chat messages that make a decision or trigger a change get transcribed verbatim in that turn's LOG entry; purely informational questions don't."
- F6: CLAUDE.md, `[concluded]` bullet, now ends "Predictions about experiments not yet run are `[assumed]`, however strong the argument."
- F5b: `docs/procedures/sync.md`, step 2, new sub-bullet *Marking reversed LOG passages*: sync agents insert `OUTDATED (<date>): <reason, pointer>` next to reversed passages that lack one, and that is the only change they make to existing LOG text.

**Why F5b was needed at all** (Julian's surprise): you're right that keeping things consistent is the whole point of a sync. But the sync.md task list only said "update the SUMMARY", and the LOGs are append-only. The 26-10-04 sync orchestrator read that as "don't touch LOGs at all", so the markers had to be added afterwards. CLAUDE.md already allowed the markers; the new sub-bullet only removes the ambiguity for a sync agent that reads sync.md literally. It's a clarification, not a new idea [concluded].

**F5a, its consequences:** the rule would make whoever reverses a decision also insert the one-line marker at the old passage, in the same turn, instead of leaving it for the next sync.
- Gain: a reader who lands on the old passage between the reversal and the next sync (via grep, or because you point an agent at the LOG) sees the warning. Today's sync had to add 9 such markers, so that window is real.
- Cost: a few seconds per reversal, and more in-place insertions into append-only LOGs. These are already sanctioned, and the marker is dated, so provenance is preserved.
- Risk: an agent marks the wrong passage. Nothing is deleted, and the next sync can fix it.
- Weight [concluded]: new agents enter through the SUMMARY, which the reversing agent already updates, so the harm in that window is limited to readers who go straight to the LOG. A modest gain; adopting or skipping are both defensible. With F5b in place, skipping F5a just means the sync catches them.

**F7: you're right, it's too specific.** The general principle ("don't state a design goal as an achieved outcome") is true but too vague to act on, and the specific version covers one kind of document. Recommendation: no rule. The FEEDBACK entry stays as the record of the lesson.

**F8: your point holds, but it targets a different failure.** Code and config comments are indeed outside the epistemic system and can be stale. The failure in F8 was different: the agent misreported what the comment said. The comment hedged ("an unvalidated guess"), the paraphrase dropped the hedge and turned it into a hard limit ("OOM'd beyond 5 matrices"). Quoting keeps the source's own uncertainty visible, and that matters even more when the source is unreliable. Your point suggests a reframed rule: "A claim that rests on a code or config comment is `[assumed]`, like the legacy references; quote the comment's decisive words rather than paraphrasing." Options: adopt the reframed version, or drop F8.

**F10** (the general CLI-wrapping lesson for `docs/CODING.md`) got no answer yet.

---

C: (transcribed from chat) I think there's a good reason why the log is append only and I don't think we should break that. For example F5a would require that every change would need to recheck the log for any places which need to get updated. At least tha's how it reads to me. So I would like to take back F5b and don't adpot F5a. Maybe there's another way to circumvent any perceived problems.

new F8 and F10 are fine

---

26-10-06, Claude Opus 5.5 — *F5b withdrawn, F5a rejected, F8 and F10 adopted; in-LOG markers already sanctioned*; continuing the triage

Done [decided: Julian, transcribed above]:
- **F5b withdrawn:** the *Marking reversed LOG passages* sub-bullet is removed again; `docs/procedures/sync.md` is back to its committed text [verified: `git diff` empty].
- **F5a not adopted.**
- **F8 adopted, reframed:** a new paragraph in CLAUDE.md *Quantitative Claims and Status Claims*. A claim that rests on a code or config comment is `[assumed]`, and the agent quotes the comment's decisive words rather than paraphrasing them.
- **F10 adopted:** new bullet in `docs/CODING.md` *Assertions and error handling*. Before wrapping an external CLI with captured output, check its `--help` for prompts and assert on the success output, not the exit code. (The proposal had said *Fail Loudly*, but that section lives in CLAUDE.md, and this lesson is too narrow for the always-loaded file.)

**Flag: the current rules already insert markers into LOGs.** CLAUDE.md, *Epistemic Status Markers*, `OUTDATED` bullet: "This is how you warn readers inside an append-only LOG without editing history: place it directly before or after the outdated content." So inserting a dated `OUTDATED` line next to an old LOG passage is the convention as written; CLAUDE.md treats it as compatible with append-only, because nothing is rewritten. It has been used a lot [verified: `git grep` 26-10-06]: the tracked LOGs plus today's new one contain 23 such lines. Today's sync added 9 of them, I added 1 (the vast publickey flag), and the 26-10-04 sync added more. Julian's reason for rejecting F5a/F5b ("don't break append-only") argues against that existing bullet too. The two possible readings:
- (a) Keep the bullet: markers are allowed but nobody is obliged to add them. This is the state after today's decision, and it is consistent with it. The sync may still add them when it notices one, as it did today.
- (b) LOGs become strictly append-only: no markers inserted into LOGs at all. The `OUTDATED` marker would stay for other files (e.g. the legacy `docs/PROJECT_REFERENCE.md`). Then decide whether to remove the 23 existing markers or leave them as history.

**The other way around the problem** (Julian asked for one): fix it on the reader side, not in the LOG. The problem F5a addressed is a reader who lands on an old LOG passage (via grep) and takes it as current. CLAUDE.md already says that later LOG parts may contradict earlier ones. One added sentence would make the reader responsible: "A LOG passage records what was true when it was written; before relying on one, check the topic's SUMMARY and the later entries of that LOG." This needs no edits to old text, works with either (a) or (b), and costs the reader one look at the SUMMARY, which is the current-state document anyway [concluded].

---

C: (transcribed from chat) Ok, I do a double switcheroo. F5b is fine, I'm still on the fence about F5a just for cognitive load reasons.

---

26-10-06, Claude Opus 5.5 — *F5b re-adopted, F5a on the fence*; continuing the triage

- **F5b re-adopted** [decided: Julian, transcribed above]: the *Marking reversed LOG passages* sub-bullet is back in `docs/procedures/sync.md`, step 2, with the same wording as before. This also settles the flagged conflict in favour of option (a): `OUTDATED` markers inside LOGs stay allowed (CLAUDE.md's bullet is unchanged), and the sync adds missing ones.
- **F5a on the fence** [decided: not adopted for now]. Julian's concern is the cognitive load on the agent making a change. With F5b in place, a marker missed at reversal time gets added at the next sync, so leaving F5a out costs only the gap until then.
- Still unanswered: the reader-side sentence (proposed in the previous entry) and F7 (agent recommends no rule).

---

C: (transcribed from chat) The reader-side sentence does also increase the token stuff. No.
yes, F7 dropped

---

26-10-06, Claude Opus 5.5 — *reader-side sentence rejected, F7 dropped; review complete*; continuing the triage

[decided: Julian, transcribed above]
- The reader-side sentence is not added: it would cost reading time in every session, because CLAUDE.md is always loaded.
- F7 is dropped: no rule. The FEEDBACK entry stays as the record of the lesson.

Final state of the review: F2, F5b, F6, F8 (reframed) and F10 were adopted. F5a is not adopted for now. The reader-side rule and F7 were rejected. F1, F3, F4 and F9 had been closed before the review. The only thing left open is reporting the generic fixes upstream to the Epistemic Memory template; Julian hasn't raised it.
