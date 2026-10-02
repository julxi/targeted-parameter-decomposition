C: (transcribed from chat) Can you setup the Epistemic memory for this repo. This repo has already a CLAUDE.md that somehow has to be included into it. Can you ask me questions how to set it up?

---

26-10-02, Claude Opus 5.5 — *setup interview and integration plan*

I have read the template (`EpistemicMemory-main/`: CLAUDE.md, README.md with its setup checklist, docs/, FEEDBACK.md, FUTURE_WORK.md), the repository's former root CLAUDE.md, README.md, CONTRIBUTING.md, .gitignore and the git remotes.

Ways this repository differs from the fresh repo the template assumes:

- `README.md` already has a job: it is the public, human-facing page of the paper code on the GitHub fork (`julxi/targeted-parameter-decomposition`, upstream `bbzmdn/targeted-parameter-decomposition`). The template wants README.md to be the agents' hub (team list, current status, conversation index).
- The root `CLAUDE.md` was a large project reference (752 lines: environment, experiments, architecture, commands, cluster rules, coding guidelines).
- Both rule sets agree on fail-fast coding. They conflict on comments: the repo said "comments hide sloppy code"; the template asks for why-comments, comments that guard deliberate deviations, and notes on what was tried. The template's `docs/CODING.md` "Project Structure" section (`projects/<name>/src/utils.py`) is an example layout that does not fit this repo.

Interview answers (questions were asked as multiple choice; option labels are quoted where the user picked one):

- Where the memory files live and whether git tracks them:
  C: (transcribed from chat) I want the template layout at root. But I don't understnad what you need the README for?
- How to combine with the existing CLAUDE.md:
  C: (transcribed from chat) I would rather have a fully epistemic CLAUDE.md and have the old one a reference that can be checked and overwritten
- Which rules win when the repo's coding rules and the template's conflict:
  C: (transcribed from chat) Template rules win
- Team and folder slug:
  C: (transcribed from chat) For now just me, but other's might join in later
- Whether the LOGs may be committed to the public fork:
  C: (transcribed from chat) Track them, public is fine
- Git workflow (agents never commit vs. agents may commit):
  C: (transcribed from chat) Agents never commit (Recommended)
- Whether critical project facts (activate .venv, max 8 GPUs, never --no-verify, PR template) stay always-loaded in CLAUDE.md:
  C: (transcribed from chat) Reference file only

I explained in chat that in the template, README.md is the agents' hub, read at every session start, and also the human front page; here the human front page already exists, so only the filename of the hub was open. Follow-up question, which file should be the agents' hub:
  C: (transcribed from chat) New INDEX.md (Recommended)

---

26-10-02, Claude Opus 5.5 — *setup implemented*

What I did (all uncommitted, for review in the diff):

- `CLAUDE.md` (root) is now the template's core file, adapted:
  - every reference to README.md as the hub now points to `INDEX.md`; a sentence states that README.md is the public paper-code page and not part of the memory system;
  - the template note is removed;
  - `docs/PROJECT_REFERENCE.md` is added to the on-demand reading list and to Reading Priority item 5, with the instruction to keep it current by fixing it in place;
  C: you wrote in `CLAUDE.md` "Read the relevant sections before any project work — coding, running experiments, or answering questions about the codebase". My goal is to only use the epistemic memory system in the future. So the old `CLAUDE.md` should eventually be phased out, especially it shouldn't be read before any project work.
  Answer (26-10-02, Claude Opus 5.5): Agreed and changed. In CLAUDE.md, `docs/PROJECT_REFERENCE.md` is now described as a legacy lookup being phased out. Agents do not read it by default and consult a section only when a task needs a project fact the `convos/` files don't cover. When they rely on a fact, they check it against the code and record it with its status tag in the topic where it was used, so the knowledge gradually moves into the memory system. I removed it from Reading Priority; INDEX.md and the file's own header say the same.
  Two things to decide for the phase-out:
  (1) Some of its content is rules, not facts. Venv activation and "never --no-verify" are also in `docs/CODING.md`, but the cluster limit (never more than 8 GPUs at once) now exists only in the reference. Since nobody reads it by default, an agent launching SLURM jobs could miss that limit. I suggest moving it into `docs/CODING.md` §Additional instructions.
  (2) Nine subpackage `CLAUDE.md` files (e.g. `spd/harvest/CLAUDE.md`, `spd/app/CLAUDE.md`) belong to the old system too. Claude Code auto-loads them whenever it works in those directories, so they are still read before project work there. Should they be phased out as well, or stay as documentation that lives next to the code?
  C: (transcribed from chat) 1. No, that is not important right now.
  C: (transcribed from chat) 2. Yes, they should also be phased out
  Answer (26-10-02, Claude Opus 5.5):
  (1) The 8-GPU limit stays only in `docs/PROJECT_REFERENCE.md` §Cluster Usage Guidelines. Nothing changed.
  (2) Done. I renamed the nine files from `CLAUDE.md` to `REFERENCE.md` in their own directories (`spd/{app,autointerp,clustering,dataset_attributions,experiments/completeness,graph_interp,harvest,investigate,pretrain}/REFERENCE.md`). Claude Code only auto-loads files named `CLAUDE.md`, so the rename stops the automatic loading, and each subpackage's documentation stays next to its code. Each file got the same header as `docs/PROJECT_REFERENCE.md`: legacy, not read by default, every claim `[assumed]`, check against the code before relying on it.
  References updated to match: the bullet in root CLAUDE.md now covers these files too; `docs/PROJECT_REFERENCE.md` links point to the new names; so do `docs/CODING.md` §Project Structure and two internal references (investigate's file tree, autointerp → harvest). One code file changed: the docstring of `spd/dataset_attributions/harvest.py` ("See REFERENCE.md in this directory"). OUTDATED: that docstring pointer was removed instead; see entry *legacy pointers removed from code and CODING.md* at the bottom. The `docs/CODING.md` rule "Update CLAUDE.md files when changing code structure" now reads: record structural changes in the topic LOG/SUMMARY, and correct a legacy reference in place if the change makes it wrong.
  Checked before renaming: the investigation agents (`spd-investigate`) don't depend on these files. `spd/investigate/scripts/run_agent.py` starts Claude Code with its own working directory (`cwd=inv_dir`) and its own prompt.
  - the private index is named `convos/{user}/private/INDEX.md` (template: `private/README.md`), for consistency with the hub's name.
- `docs/PROJECT_REFERENCE.md` is the former root CLAUDE.md, moved (plain `mv`, not `git mv`, per the no-git-writes rule; git detects the rename at commit time). The header is replaced with one stating what the file is and where it came from. Its "Coding Guidelines & Software Engineering Principles" section moved out of it (see next bullet); everything else is unchanged.
C: maybe you can add that everything written in the old CLAUDE.md should has default [assumed]
Answer (26-10-02, Claude Opus 5.5): Done. The header of `docs/PROJECT_REFERENCE.md` now says every claim in it is `[assumed]` and must be checked against the code before an agent relies on it; the pointer in CLAUDE.md repeats this. One boundary: the coding guidelines that moved from the old CLAUDE.md into `docs/CODING.md` are rules, not claims, so I did not tag them; they stay in force as rules. Tell me if you meant those too.
C: Can you give me an overview what in `docs/CODING.md` comes from the old repo guides and what comes from the EpistemicMemory?
Answer (26-10-02, Claude Opus 5.5): By section, top to bottom:

| Section in `docs/CODING.md` | Origin | Changes |
|---|---|---|
| Title + intro paragraph | Template | none |
| §Project Structure | Written by me during setup | Replaces the template's example layout (`projects/<name>/src/utils.py`), which doesn't describe this repo. Now: one paragraph pointing to `docs/PROJECT_REFERENCE.md` and the subpackage CLAUDE.md files, plus the template's `.env` rule (ask the user, never hardcode keys), with "expected keys in `.env.example`" added |
| §User Interaction for Coding Projects | Template | none |
| §Code Comments | Template | none. This section won over the old repo's "Comments" section, which was dropped |
| §Logging & Observability | Template | none |
| §Additional instructions: heading | Template (placeholder slot meant for project rules) | placeholder text replaced |
| §Additional instructions: first two paragraphs | Written by me | a precedence note (template sections win on conflict, comments follow §Code Comments), venv activation and `make check` / `make test` (both taken from the old CLAUDE.md's Environment Setup and Development Commands) |
| ### Research-code principles, with subsections Tests, Assertions and error handling, Control Flow, Types/Arguments/Defaults, Tensor Operations, Other Important Practices, GitHub | Old repo CLAUDE.md, §"Coding Guidelines & Software Engineering Principles" | verbatim except: (a) its "Comments" subsection removed; (b) "Update CLAUDE.md files…" now names the subpackage CLAUDE.md files or `docs/PROJECT_REFERENCE.md`; (c) the commit bullet starts with "Agents commit only when explicitly asked"; (d) subsection headings demoted one level |

Short version: the four general sections are the template's. Everything under §Additional instructions is the old repo's coding rules, apart from the two short paragraphs I wrote. §Project Structure is the only template section I rewrote.

Overlap to be aware of: the old rules' "fail fast" principles and assertion rules repeat what CLAUDE.md §"Fail Loudly, Never Silently" already says. They don't contradict each other, so I left both; the old version adds repo-specific examples (narrow types, `match`, jaxtyping, einops).
- `docs/CODING.md` is the template's file, with two changes:
  - "Project Structure" is replaced with a pointer to `docs/PROJECT_REFERENCE.md` and `.env.example`, because the template's example layout does not describe this repo;
  - "Additional instructions" now holds the repo's former coding guidelines (research-code principles, tests, assertions, control flow, types, tensor ops, practices, GitHub), with these changes. The repo's "Comments" section is dropped, because §Code Comments of the template wins per the interview. "Update CLAUDE.md files" now names the subpackage CLAUDE.md files or `docs/PROJECT_REFERENCE.md`. The commit bullet is prefixed with "agents commit only when explicitly asked". A preamble states that the template sections win on conflict, plus the venv activation and `make check` / `make test`.
- `docs/HOW_TO_USE.md`, `docs/procedures/*.md`: copied; their README-index references point to `INDEX.md`.
- `INDEX.md`: created from the template's `docs/README_SKELETON.md`. It holds the project description, team, current status, the maintenance table (sync row) and the conversation index with this topic. The one-time-deadlines table is omitted because there are none yet.
- `FEEDBACK.md`, `FUTURE_WORK.md`: headers only; template entries and setup paragraphs removed.
C: I think `FEEDBACK.md` and `FUTURE_WOKR.md` can be deleted, right? I would like to only keep files necessary for the EpistemicMemory to work
Answer (26-10-02, Claude Opus 5.5): Not quite. The LOG/SUMMARY mechanism works without them, but other rules point to both files:
- FEEDBACK.md is the target of CLAUDE.md §Feedback. It is the loop by which the conventions get corrected: agents note there where the instructions failed them. I recommend keeping it, especially now, because this setup is adapted from the template and untested, so it will need corrections.
- FUTURE_WORK.md is where `docs/CODING.md` §Code Comments sends TODOs that aren't tied to one line of code, and INDEX.md links it. It is easier to drop: deferred items could live in the relevant topic's SUMMARY instead.
Deleting either file also means rewriting the rules that point to it; otherwise agents will recreate the file or have nowhere to put the item. I have not deleted anything yet. Tell me which to remove and I'll adjust the references.
By the same "necessary for it to work" test, two more files are candidates: `docs/HOW_TO_USE.md` (a guide for humans; agents only read it when onboarding a new user) and `docs/procedures/multi_agent_debate.md` (only needed if you run multi-agent debates).
C: OK, they are important for EpistemicMemory, so keep them.
Answer (26-10-02, Claude Opus 5.5): Kept, unchanged: FEEDBACK.md and FUTURE_WORK.md, and also `docs/HOW_TO_USE.md` and `docs/procedures/multi_agent_debate.md`. I read "them" as covering all four; if you meant only the first two, say so and I'll remove the other two.
- `.gitignore`: added `convos/*/private/` and `convos/*/private` so the private-records feature is available. No private folder was created, because the user chose to track memory files publicly and did not ask for one.
- Not copied: the template's `LICENSE` (the repo has its own), `README.md` and `docs/README_SKELETON.md` (consumed by setup), `convos/jamie/` (demo content).

Choices I made without explicit ratification:
- Folder slug `julian` [assumed]: the team question was answered with "just me", not with a slug.
C: ok
Answer (26-10-02, Claude Opus 5.5): Noted; the slug `julian` is now `[decided]` in the SUMMARY.
- Status section in INDEX.md lists only what is observable from the branch and recent commits [concluded: I do not know the goals, so I did not invent any].
C: The current goal is to train tPD for Qwen. Details will follow in future converstations
Answer (26-10-02, Claude Opus 5.5): Recorded in INDEX.md §Current status and goals as the current goal, `[decided]`, citing this comment.

Open points for the user:
- The template download `EpistemicMemory-main/` is still at the repo root, untracked. It contains a second CLAUDE.md and the demo user `convos/jamie/`; agents browsing it could mistake it for live project memory. Suggest deleting it, or keeping it outside the repo if you want to diff future template updates.
C: yes it should be deleted
Answer (26-10-02, Claude Opus 5.5): Deleted (`rm -rf EpistemicMemory-main/`; it was untracked, so it does not appear in the diff). The template's origin is still linked from INDEX.md.
- Claude Code also has its own auto-memory (outside the repo, under `~/.claude/projects/…/memory/`). It overlaps with this system; under CLAUDE.md's rule "nothing important should exist only in chat", project knowledge belongs in `convos/` files, not there.
C: good to know. what will you do about it?
Answer (26-10-02, Claude Opus 5.5): The auto-memory directory for this project was empty, so nothing needed migrating. I added one entry, `use-epistemic-memory-not-auto-memory.md` (plus its line in that directory's `MEMORY.md`). It tells future sessions to write project knowledge, decisions and preferences into the repo's memory files instead of auto-memory, and cites this topic as the reason. I can't switch auto-memory off from inside a session. You can turn it off yourself in Claude Code's settings (the `/memory` command); the pointer entry then becomes moot but does no harm. I'd keep auto-memory on with the pointer: it costs nothing and catches the case where a future session would otherwise save something there.
- Nothing is committed. Review the diff and commit when satisfied.

---

C: (transcribed from chat) You wrote something like `See REFERENCE.md in this directory for usage instructions.` This is not right. These are references that are supposed to phased out.

---

26-10-02, Claude Opus 5.5 — *legacy pointers removed from code and CODING.md*

Agreed: when I phased out the subpackage files, I repointed the docstring to the renamed file. That sent readers of the code to a document we are phasing out. The correct move was to drop the pointer.

- `spd/dataset_attributions/harvest.py`: the sentence "See REFERENCE.md in this directory for usage instructions." is removed from the module docstring. The rest of the docstring is unchanged. [verified: grep finds no `REFERENCE.md` mention in any .py/.ts/.svelte/.yaml file under `spd/`, `scripts/`, `tests/`, or in the Makefile]
- `docs/CODING.md` had the same problem in two places, both fixed:
  - §Additional instructions: the venv rule's "(details in `docs/PROJECT_REFERENCE.md` §Environment Setup)" is removed; the rule itself stays.
  - §Project Structure: no longer says layout etc. "are described in" the legacy references. It now says such facts belong in topic LOG/SUMMARY files, and the legacy references may be consulted under CLAUDE.md's rules (claims `[assumed]`) until a fact has been recorded.
- Remaining mentions of the legacy references, all describing the phase-out rather than sending readers there as an authority: root CLAUDE.md (the rule for how to treat them), INDEX.md (one sentence), `docs/CODING.md` (the two places above and the rule to correct a legacy reference when a change makes it wrong).
- Lesson recorded in FEEDBACK.md.

---

C: (transcribed from chat) yes

(In reply to the agent's chat question whether to add the FEEDBACK.md suggestion "Do not add or keep pointers to the legacy references in code or active docs." to the root CLAUDE.md.)

---

26-10-02, Claude Opus 5.5 — *no-pointers rule added to CLAUDE.md*

Added to the legacy-reference bullet in root CLAUDE.md (§ reading list at the top): "Do not add or keep pointers to the legacy references in code or active docs: when you find one that sends readers there as the authority, remove it (don't repoint it) and record the fact where it belongs." The FEEDBACK.md entry is marked as adopted.
