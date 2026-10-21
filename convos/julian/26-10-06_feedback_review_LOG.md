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

---

26-10-21, Claude Opus 5.5 — *round 2: proposals for the living-document-drift entry*; I have read CLAUDE.md, INDEX.md, FEEDBACK.md, `docs/procedures/sync.md` and this topic's SUMMARY and LOG end; Julian asked in chat what to do about the FEEDBACK entry *26-10-21 (sync): living-document drift after state changes*

The entry names three failure kinds. My proposal for each:

**(1) `Last updated` lines that grow into chains.** CLAUDE.md already forbids "Previously: ..." chains in living documents, so the rule exists and was not followed. Likely cause [concluded: from reading all 16 SUMMARY headers on 26-10-21]: agents write the parenthetical as a changelog of what this update changed ("sync: X corrected, Y added"). Most current headers read that way, and `sync.md` step 2 invites it ("with the sync date in its `Last updated` line"). A changelog naturally accumulates; a state description does not. One more chain exists that the sync did not reach: `convos/julian/26-10-06_truth_writing_components_SUMMARY.md` ("…; earlier, 26-10-13: …"). Proposal:
- `sync.md` step 2, last bullet → "Rewrite the `Last updated` line: the sync date and the topic's current state in one clause. Replace the whole line; it is not a changelog."
- CLAUDE.md, the SUMMARY bullet: "a `**Last updated:**` date near the top" → "a `**Last updated:**` line near the top (date and current state; replaced on each update, never appended to)". This edits an existing sentence and adds about ten words. The chains are created in normal sessions, not by the sync, so a sync-only rule would only clean up afterwards.
- Fix the truth-writing header now.

**(2) Tags not updated when an answer ratifies an item.** One instance, and the sync's C: check is where it surfaces. Proposal: extend the C: bullet in `sync.md` step 2 with "and that the items an answer ratifies or rejects carry the updated tag". No CLAUDE.md change.

**(3) A state change does not reach the other topics' SUMMARYs.** The entry's prevention ("grep the SUMMARYs and INDEX.md in the same turn") is a same-turn obligation of the same kind as F5a, which Julian set aside for cognitive load; the sync did catch all three cases. But the stakes are higher than for an unmarked LOG passage: SUMMARYs and INDEX rows are what agents trust, and "RTX PRO 6000 not yet rented" could lead to a second paid rental, "checkpoint on WandB" to a download plan that fails. All instances found involve something that has its own home topic (WandB storage, vast rentals). A narrow rule covers them: "When you change the state of something another topic tracks (a WandB run or checkpoint, a rented instance), append a short entry to that topic's LOG and update its SUMMARY and INDEX row in the same turn." The agent then knows exactly which files to touch; no project-wide grep. It merges with the next FEEDBACK entry (*conventions and picks recorded only where they happened*, part 1: the run register created in another topic's session). Placement [assumed: Julian's call]: `docs/CODING.md`, *Running jobs on a rented vast.ai GPU*, next to the proposed run-register bullet; every instance so far is a WandB or rental action, and agents read CODING.md before experiments. A general sentence in CLAUDE.md *Cross-Referencing Between Topics* would cover other kinds too, at the cost of always-loaded tokens.

The "uncommitted" case: no rule. Commits are Julian's action, so the agent that writes the SUMMARY cannot know about them, and `git log` answers the question at once; the sync fixes such words when it meets them.

Suggestion: the other three 26-10-21 FEEDBACK entries (detach pattern `-f`; F5a evidence; conventions recorded only where they happened) could be decided in the same round, since (3) overlaps with the last two.

Open, awaiting Julian: adopt (1) both parts / sync.md only; (2); (3) and its placement.

---

26-10-21, Claude Opus 5.5 — *round 2: concrete instances behind the drift entry*; same session; Julian asked in chat for an example of each problem and how the proposed rules would prevent it. Explained in chat; the evidence, for the record [verified: `git show` of the commits named]:

1. Chain: `convos/julian/26-10-10_bolder_experiments_SUMMARY.md` at `86284b2` (before the sync) had `**Last updated:** 26-10-16 (component analysis of 4b/4c: …; Exp 4c … done: …; Exp 4a/4b done: …; causal layer scan: …; report section … reframed …; earlier: where truth is written, statement-first prompt and steering, cancellation test on all runs, Exp 0–3 analysed)`: each session put its news in front of the old parenthetical. Still present: `convos/julian/26-10-06_truth_writing_components_SUMMARY.md` ("…; earlier, 26-10-13: …; sync: …").
2. Stale tag: the same SUMMARY at `86284b2` had "**Exp 4a/4b** [assumed: agent's design, not ratified]" although Julian's C: comments under *Questions for Julian* had answered (changed the positions, no objection to the rest); the sync (`4afa65d`) changed it to `[decided: C: comments …]`.
3. Cross-topic status: the INDEX row of the vast-rentals topic at `86284b2` ended "new config `--config rtxpro6000` (RTX PRO 6000, 96 GB, from ~$1.4/h; not yet rented)", while `convos/julian/26-10-16_faithful_targeting_LOG.md` already recorded Julian's message that he had rented one and the 4e runs on it.

Calibration [concluded]: (1) costs reading time and blurs the current state, but misleads nobody badly; (2) can make an agent re-ask or re-open a settled question; (3) is the one with a real cost, since an agent trusting the INDEX row could suggest renting a machine that is already running. None of the proposals guarantees prevention: (1)'s rule already existed and was ignored, so its rewording only makes the right form more obvious where the line is written; the sync remains the backstop for all three.

---

C: (transcribed from chat) 1. looks like a good change.
2. Maybe there is also something that we can add that reduces the amount of wrong tags at least in one session. Any idea?
3. How does the agent know which topic owns rentals?

---

26-10-21, Claude Opus 5.5 — *round 2: (1) adopted; ideas for (2), owner lookup for (3)*; same session

**(1) adopted** [decided: Julian, transcribed above; I read "looks like a good change" as both parts of the proposal, Julian may narrow it]:
- `docs/procedures/sync.md`, step 2, last bullet: "Updating the SUMMARY if needed, and rewriting its `Last updated` line: the sync date and the topic's current state in one clause. Replace the whole line; it is not a changelog."
- CLAUDE.md, SUMMARY bullet: "Every SUMMARY carries a `**Last updated:**` line near the top (date and current state; replaced on each update, never appended to) so readers can judge staleness."
- `convos/julian/26-10-06_truth_writing_components_SUMMARY.md`: the chained line replaced by the current state (analyses done, follow-ups in the bolder-experiments and direction-free topics, A1 open in FUTURE_WORK.md).

**(2) Fewer wrong tags within a session.** How often it happens [verified: tag lines changed by the syncs `7bfb785`, `abc9c0b`, `4afa65d`, read by eye]: most tag changes in syncs are rewordings or added sources (`[decided]` → `[decided: transcribed chat in the LOG]`); the only clear "answered but still tagged as pending" case is the Exp 4a/4b one. So this is rare. The mechanism of that case: the agent answered Julian's C: comments in the LOG and added the new facts to the SUMMARY, but the SUMMARY item that said "not ratified" sat elsewhere and was not revisited. Ideas, cheapest first:
- (a) One clause in CLAUDE.md *Keeping files current*: "update the SUMMARY whenever an important change lands in the LOG; when an answer settles an item, retag that item there too." Costs about a dozen words in the always-loaded file, and it hooks onto a step the agent already performs.
- (b) Pending tags name the question they wait on, e.g. `[assumed: awaiting Julian, LOG entry *Questions for Julian*]`. The answering agent knows that entry title, and a grep for it finds every item waiting on it. No CLAUDE.md change, but it only works if agents write such tags and think to grep.
- (c) A script that lists pending tags, as a Stop hook: mechanical, but noisy (many `[assumed]` tags are correct and permanent), so not recommended.
Recommendation [concluded: one clear instance in three syncs]: (a) if Julian wants a within-session nudge; otherwise rely on the sync (the C: check extension proposed as (2) earlier) and do nothing more.

**(3) How the agent knows which topic owns rentals.** At present nothing tells it: topics are named after discussions, not objects, and the only route is recognising the INDEX row "vast.ai rentals: …" while recording something else. That route failed in the observed case. Proposal: the rule itself names the owners, in `docs/CODING.md` *Running jobs on a rented vast.ai GPU* (it already points into `convos/julian/`):
> **Status changes reach the owning topic.** When you rent or destroy an instance, change the launcher or a vast config, record it in `convos/julian/26-10-05_vast_rentals_LOG.md` too; when you launch a WandB run, change or delete a checkpoint, add or update its row in `convos/julian/26-10-03_wandb_storage_RUNS.md` and record deletions in `convos/julian/26-10-03_wandb_storage_LOG.md`. Update that topic's SUMMARY and INDEX row in the same turn.
If other objects later show the same drift, they are added to this list. Alternative: a short "standing topics" list (object → topic) in INDEX.md, which every session reads; more general, but always-loaded tokens. Recommendation [assumed: Julian's call]: the CODING.md version.

---

C: (transcribed from chat) that's sounds reasonable can you update the rule

(Context: Julian had asked what the concrete suggestion of the FEEDBACK entry *reversed passages without markers, again — evidence for revisiting F5a* is; the agent explained the narrow same-turn rule in chat.)

---

26-10-21, Claude Opus 5.5 — *narrow F5a adopted*; new session; I have read CLAUDE.md, INDEX.md, the FEEDBACK entries for F5a, and this LOG's round-2 entries

Adopted [decided: Julian, transcribed above] the narrow version of F5a. CLAUDE.md, append-only paragraph, after "If a decision changes, append a new section that says so.":
> When your entry says it corrects or contradicts an earlier passage, put an `OUTDATED (<date>): <reason, pointer to your entry>` line next to that passage in the same turn.

Difference from the F5a that Julian set aside on 26-10-06 for cognitive load: F5a applied to any reversal, so the agent had to notice on its own that its change invalidated some passage elsewhere. This rule applies only when the agent's entry already names the passage it corrects, so the passage is already located and the cost is one edit. Reversals that no entry calls out still rely on the sync (F5b). FEEDBACK.md: resolution notes added to the 26-10-04 entry and the 26-10-21 entry.

Still open from round 2: (2) stale tags, (3) cross-topic status changes and placement; the 26-10-21 FEEDBACK entries on the detach pattern and on conventions recorded only where they happened.

---

C: (transcribed from chat) Is it true that LOGs are append-only? I mean sometimes I write comments and claude answers right under them. And tags are also excluded from append-only. Maybe there's a better formulation?

---

26-10-21, Claude Opus 5.5 — *append-only wording: proposal "never rewritten"*; same session

Julian is right that "append-only" is inaccurate. What actually happens in LOGs:
- In-place insertions, all sanctioned by CLAUDE.md: `C:` comments (anywhere), answers directly beneath them (*Where to write*), `OUTDATED` markers next to superseded passages (the `OUTDATED` bullet, sync.md step 2, and since today the same-turn rule).
- Deletions or rewrites of existing text: none by the three syncs [verified: `git show 7bfb785 abc9c0b 4afa65d -- 'convos/*_LOG.md'` contains no removed lines]. That includes tags: I found no rule or practice that changes a tag inside a LOG; tags are updated in SUMMARYs, and a superseded LOG passage gets an `OUTDATED` marker or a later entry. (If Julian had a case of retagging inside a LOG in mind, that would be a deviation worth flagging.)

So the real invariant is "existing text is never edited or deleted", not "text is only added at the end". The paragraph's closing sentence ("reading top-to-bottom always gives the true sequence of events") is also only true of the entries, since in-place answers sit earlier in the file than their date; they are dated for that reason.

Proposed replacement for the CLAUDE.md paragraph [assumed: awaiting Julian]:
> **LOGs are never rewritten.** Do not edit or delete existing text in a LOG; this guarantees that the record cannot be quietly changed. New entries go at the bottom, so the entries read chronologically. Three kinds of text are inserted in place instead: humans' `C:` comments, answers directly beneath them, and `OUTDATED` markers next to passages that are no longer true; the answers and markers carry their date. If a decision changes, append a new entry that says so; when your entry corrects or contradicts an earlier passage, put an `OUTDATED (<date>): <reason, pointer to your entry>` line next to that passage in the same turn.

Follow-up wording changes so the term stays consistent: CLAUDE.md DRAFT paragraph ("keeps the LOG's append-only guarantee intact" → "no-rewrite guarantee"), the `OUTDATED` bullet ("inside an append-only LOG" → "inside a LOG"), `docs/procedures/multi_agent_debate.md` line 28 ("append-only, answers inline with `C:`, nothing rewritten" → "never rewritten, answers inline beneath `C:` comments"). Historical mentions in LOGs and FEEDBACK.md stay as they are.

---

C: (transcribed from chat) Your new proposal. can it be shortened?

---

26-10-21, Claude Opus 5.5 — *append-only wording: shortened proposal*; same session

Shortened version of the proposal in *append-only wording: proposal "never rewritten"* (that one is superseded by this) [assumed: awaiting Julian]:
> **LOGs are never rewritten:** don't edit or delete existing text, so the record can't be quietly changed. Besides new entries at the bottom, only `C:` comments, answers beneath them, and dated `OUTDATED` markers go in place. When your entry corrects or contradicts an earlier passage, mark that passage `OUTDATED (<date>): <reason, pointer to your entry>` in the same turn.

Dropped, because other paragraphs already say it: "If a decision changes, append a new entry that says so" (follows from never-rewrite plus entries at the bottom); "the answers carry their date" (the `C:` paragraph prescribes `Answer (date, author)`); "entries read chronologically" (*Where to write*). Follow-up wording changes as in the previous entry.

---

C: (transcribed from chat) The part about format of OUTDATED does it also live somewhere else?

---

26-10-21, Claude Opus 5.5 — *OUTDATED format: one definition*; same session

Yes, in four places, and they disagree [verified: `grep -rn OUTDATED CLAUDE.md docs/`, 26-10-21]:
- CLAUDE.md, legacy-references paragraph: `OUTDATED (<date>): <reason>` plus "points to where the current fact is recorded in `convos/`".
- CLAUDE.md, append-only paragraph (added today): `OUTDATED (<date>): <reason, pointer to your entry>`.
- CLAUDE.md, *Epistemic Status Markers*, `OUTDATED` bullet: `OUTDATED: <reason>`, no date, no pointer. This is the definition, and it is the incomplete one.
- `docs/procedures/sync.md` step 2: `OUTDATED (<date>): <reason, pointer to the newer entry>`.

Proposal [assumed: awaiting Julian]: the bullet becomes the only place with the format, `OUTDATED (<date>): <reason, pointer to the replacement>`; the others just say "mark it `OUTDATED`". The paragraph from *append-only wording: shortened proposal* then ends: "When your entry corrects or contradicts an earlier passage, mark that passage `OUTDATED` in the same turn." Legacy-references paragraph: "mark the passage `OUTDATED`, pointing to where the current fact is recorded in `convos/` (record it there first)". sync.md: "insert an `OUTDATED` marker next to it". CLAUDE.md is always loaded, so sync.md can rely on the bullet.

---

C: (transcribed from chat) Let's keep LOGs are append-only and then write the exceptions as you did. Also Yes there should be one definition

---

26-10-21, Claude Opus 5.5 — *append-only paragraph and OUTDATED definition applied*; same session

Applied [decided: Julian, transcribed above]. The name "append-only" stays, so the follow-up wording changes to the DRAFT paragraph, the `OUTDATED` bullet and `multi_agent_debate.md` from *append-only wording: proposal "never rewritten"* are dropped.
- CLAUDE.md, append-only paragraph (replaces the version from *narrow F5a adopted*; the rule's content is unchanged): "**LOGs are append-only:** don't edit or delete existing text, so the record can't be quietly changed. New entries go at the bottom; the only exceptions, inserted in place, are `C:` comments, answers beneath them, and `OUTDATED` markers. When your entry corrects or contradicts an earlier passage, mark that passage `OUTDATED` in the same turn."
- CLAUDE.md, `OUTDATED` bullet, now the single definition: `OUTDATED (<date>): <reason, pointer to the replacement>`.
- CLAUDE.md, legacy-references paragraph: "mark the passage `OUTDATED`, pointing to where the current fact is recorded in `convos/` (record it there first)".
- `docs/procedures/sync.md` step 2: "insert an `OUTDATED` marker next to it".
