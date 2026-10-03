# Coding Guidelines — Reference

Read this in full before writing code or running an experiment in this project. The critical rules that are easy to violate (Fail Loudly; Observe & Verify; never hardcode keys; no git writes unless asked) live in CLAUDE.md; this file holds the how-to.

## Project Structure

This repository is a single Python package (`spd/`) installed in editable mode into the project-local `.venv` (`make install-dev`). Facts about layout, entry points, experiments and data flow belong in the topic LOG/SUMMARY files. Until a fact has been recorded there, the legacy references (`docs/PROJECT_REFERENCE.md`, subpackage `REFERENCE.md` files) may be consulted under the rules in CLAUDE.md: their claims are `[assumed]`.

**No artifacts in `convos/`** [decided: Julian, `convos/julian/26-10-02_training_run_LOG.md`, *clean-up options*]: `convos/` holds only the Markdown records (LOG, SUMMARY, DRAFT, …), which may state results and numbers. Run outputs (metrics, configs, progress logs, figures) and one-off experiment scripts go outside the repository, into a topic folder under the local output directory `~/spd_out/` (`SPD_OUT_DIR`), e.g. `~/spd_out/26-10-02_training_run/`; figures stay on WandB. The records name that path, so a reader on this machine can find the source of a number.

**If `.env` is missing or keys don't work**: ask the user to provide it. Do not proceed with hardcoded keys. Expected keys are listed in `.env.example`.

## User Interaction for Coding Projects

The workflow:
1. The user describes the project in a LOG file; most interaction happens there
2. Design decisions: the agent writes options and questions into the LOG, the user responds with `C:` comments
3. Once decisions are made: implement, optionally run, document what you did in the LOG

(The git rule — no git write operations unless explicitly asked — is in CLAUDE.md.)

**Coding project SUMMARYs** should additionally include (beyond the general SUMMARY guidelines in CLAUDE.md):
- What the project does and what its data formats look like (with JSON examples)
- What is implemented vs. planned
- Any refactorings or convention changes that explain why older code may look different

If a project grows large, split the SUMMARY into focused sub-documents.

## Code Comments

Code should be self-documenting about *what* it does; comments document what is NOT in the code — for a future agent with zero history:

- **Why, not what.** Reasons, constraints, invariants ("must run before X because Y"). Never narrate the code ("loop over users") and never narrate the *change* ("now handles the edge case") — that addresses today's reviewer, not the next reader.
- **Guard deliberate deviations.** Anything intentionally non-canonical — a missing retry, an odd ordering, a "redundant" check — WILL look like a bug to a future agent and get "fixed". Mark it: "intentionally X, not Y, because Z."
- **Record what was skipped or tried.** "Deliberately no caching — tried, reverted (see the topic LOG)" closes dead ends a fresh agent would re-enter.
- **Document contracts at boundaries.** Docstrings with units, shapes, invariants, side effects; one top-of-file line on what the module does and who reads/writes its data. Agents edit from partial reads — the contract line is what makes that safe.
- **Comments are part of the edit.** When you change code, updating the adjacent comments is part of the change — a stale comment misleads the next agent worse than no comment.
- **Placement follows P(seen when needed):** what must be known at the point of edit goes in the code; narrative, alternatives, and results go to LOG/SUMMARY. Comments must stand alone (Self-Contained Artifacts rule in CLAUDE.md): state what was chosen and why in place; add a LOG reference only for decisions important enough to warrant the full discussion.
- TODOs in code only when tied to that exact spot and short-lived; anything else goes to FUTURE_WORK.md — in-code TODOs rot invisibly.

## Logging & Observability

This is the how-to for the Observe & Verify rule in CLAUDE.md — the rule for operations that fail *silently* by producing plausible-but-wrong output of the right shape instead of crashing. "Ran without errors" proves nothing for these; you only catch them by logging the actual input/output and reading it on real data.

Where this matters most: regexes, fuzzy matching, truncation, normalization, encoding round-trips, parsing model output, and **prompt composition** — a subtly malformed prompt (a dropped section, a misplaced join) silently degrades every result in a batch. For prompts, log the exact final string with visible delimiters around each composed section, then read it.

Structure hints:
- Log *both sides* of a decision boundary — the cases that matched **and** the cases that didn't. False positives hide in "matched," false negatives in "didn't."
- For high volume, aggregate with counts, and keep a few raw examples per bucket so correctness can still be checked by eye.
- Log the score next to each fuzzy match and sort by it — errors cluster near the threshold.
- Log a stable id linking each output back to its input, so a suspicious aggregate is traceable to a concrete case.

## Running jobs on a rented vast.ai GPU

Lessons from avoidable restarts and lost time (26-10-02/03; details in `convos/julian/26-10-02_training_run_LOG.md`). Rent with `spd-vast --mode provision --config h100`; it writes the ssh alias `vastai`.

- **Check the GPU's power limit after renting:** `ssh vastai nvidia-smi --query-gpu=power.limit --format=csv`. H100 offers come with 500–700 W limits. At 500 W the GPU throttled to ~1 GHz and batch-16 training ran 40% slower than at ~600 W. `spd-vast` doesn't filter on this yet (FUTURE_WORK.md).
- **Launch from a login shell:** `ssh vastai 'bash -lc "<cmd>"'`. `spd-vast` puts the WandB key in `/etc/profile.d/spd_env.sh`, which plain `ssh vastai '<cmd>'` does not read, so WandB runs fail with "No API key configured".
- **Detach long jobs:** `setsid nohup <script> > <log> 2>&1 < /dev/null &`, and call ssh with `-n` and its output redirected. Otherwise the local ssh call can hang until the remote job ends.
- **Kill by a bracket pattern or PID:** `pkill -f "run_trials[.]sh"`. A plain `pkill -f run_trials.sh` inside `ssh vastai '...'` matches its own command line and kills the ssh session, so the rest of the command never runs.
- **Don't set `HF_HUB_OFFLINE=1`:** the LM decomposition still calls the Hugging Face API after loading cached weights and crashes in offline mode.
- **Wall times are only comparable on the same machine.** Compare speed with per-step times measured on one instance, not with clock times from different rentals.
- **Before the instance is destroyed:** checkpoints of runs with `sync_checkpoints_to_wandb: true` are on WandB; copy runs without WandB and any metrics/logs you need to `~/spd_out/<topic>/` (not `convos/`, see above).

## Additional instructions

Project-specific rules for this repository (SPD / targeted parameter decomposition). Where they conflict with the sections above, the sections above win; in particular, comments follow §Code Comments, not a minimal-comment style.

Before running Python or git: `source .venv/bin/activate`. Check code with `make check` (basedpyright, ruff lint, ruff format) and `make test`.

### Research-code principles
**This is research code, not production. Prioritize simplicity and fail-fast over defensive programming.**

Core principles:

- **Fail fast** - assert assumptions, crash on violations, don't silently recover
- **No legacy support** - delete unused code, don't add fallbacks for old formats or migration shims
- **Narrow types** - avoid `| None` unless null is semantically meaningful; use discriminated unions over bags of optional fields
- **No try/except for control flow** - check preconditions explicitly, then trust them
- **YAGNI** - don't add abstractions, config options, or flexibility for hypothetical futures

```python
# BAD - defensive, recovers silently, wide types
def get_config(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

config = get_config(path)
if config is not None:
    value = config.get("key", "default")

# GOOD - fail fast, narrow types, trust preconditions
def get_config(path: Path) -> Config:
    assert path.exists(), f"config not found: {path}"
    with open(path) as f:
        data = json.load(f)
    return Config(**data)  # pydantic validates

config = get_config(path)
value = config.key
```

#### Tests

- The point of tests in this codebase is to ensure that the code is working as expected, not to prevent production outages - there's no deployment here. Therefore, don't worry about lots of larger integration/end-to-end tests. These often require too much overhead for what it's worth in our case, and this codebase is interactively run so often that issues will likely be caught by the user at very little cost.

#### Assertions and error handling

- If you have an invariant in your head, assert it. Are you afraid to assert? Sounds like your program might already be broken. Assert, assert, assert. Never soft fail.
- Do not write: `if everythingIsOk: continueHappyPath()`. Instead do `assert everythingIsOk`
- You should have a VERY good reason to handle an error gracefully. If your program isn't working like it should then it shouldn't be running, you should be fixing it.
- Do not write `try-catch` blocks unless it definitely makes sense
- **Write for the golden path.** Never let edge cases bloat the code. Before handling them, just raise an exception. If an edge case becomes annoying enough, we'll handle it then — but write first and foremost for the common case.

#### Control Flow

- Keep I/O as high up as possible. Make as many functions as possible pure.
- Prefer `match` over `if/elif/else` chains when dispatching on conditions - more declarative and makes cases explicit
- If you either have (a and b) or neither, don't make them both independently optional. Instead, put them in an optional tuple

#### Types, Arguments, and Defaults

- Write your invariants into types as much as possible.
- Use jaxtyping for tensor shapes (though for now we don't do runtime checking)
- Always use the PEP 604 typing format of `|` for unions and `type | None` over `Optional`.
- Use `dict`, `list` and `tuple` not `Dict`, `List` and `Tuple`
- Don't add type annotations when they're redundant. (i.e. `my_thing: Thing = Thing()` or `name: str = "John Doe"`)
- Differentiate no data from empty collections. Often it's important to differentiate `None` from `[]`
- Don't use bare dictionaries for structures whose values aren't homogenous
  - good: {<id>: <val>}
  - bad: {"tokens": …, "loss": …}
- Default args are rarely a good idea. Avoid them unless necessary. You should have a very good reason for having a default value for an argument, especially if it's caller also defaults to the same thing
- This repo uses basedpyright (not mypy)
- Keep defaults high in the call stack.
- Don't use `from __future__ import annotations` — use string quotes for forward references instead.

#### Tensor Operations

- Try to use einops by default for clarity.
- Assert shapes liberally
- Document complex tensor manipulations

#### Other Important Software Development Practices

- Don't add legacy fallbacks or migration code - just change it and let old data be manually migrated if needed.
- Delete unused code.
- If an argument is always x, strongly consider removing as an argument and just inlining
- **Record structural changes in the topic LOG/SUMMARY** (changed code structure, added/removed files, modified key interfaces). If the change makes a legacy reference wrong (`docs/PROJECT_REFERENCE.md`, or a subpackage `REFERENCE.md` such as `spd/harvest/REFERENCE.md`), don't update it: mark the passage `OUTDATED (<date>): <reason>` with a pointer to the LOG/SUMMARY that records the current fact. The legacy references receive no new information.

#### GitHub

- To view github issues and PRs, use the github cli (e.g. `gh issue view 28` or `gh pr view 30`).
- When making PRs, use the github template defined in `.github/pull_request_template.md`.
- Agents commit only when explicitly asked (rule in CLAUDE.md §"API keys and git"). When asked: commit to the current branch (agents don't create their own branches; rule in CLAUDE.md §"API keys and git") and do not use `git add .` to add all unstaged files. Instead, add only the individual files you changed, don't commit all files.
- When the user wants a new branch, use branch names `refactor/X` or `feature/Y` or `fix/Z`.
- NEVER use `--no-verify` to skip pre-commit hooks. They are there for a good reason. If pre-commit hooks fail, you MUST fix the underlying problem.
