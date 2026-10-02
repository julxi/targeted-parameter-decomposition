# FEEDBACK

A list of problems with the documentation or workflow, noted by agents (or humans), each with constructive criticism on how it could have been prevented. This is how the conventions improve instead of rotting: when an agent makes a mistake that better instructions would have prevented, the mistake gets an entry here, and a periodic review turns recurring entries into new rules.

Entry format: date, what went wrong (or what friction occurred), and — where possible — a concrete suggestion: which file should say what, so the next agent doesn't hit the same problem.

---

- 26-10-02 — While phasing out the subpackage `CLAUDE.md` files (renamed to `REFERENCE.md`), the agent mechanically repointed every reference to the new filename, including a code docstring ("See REFERENCE.md in this directory for usage instructions") and a "details in" pointer in `docs/CODING.md`. That kept readers being sent to documents being phased out; Julian caught it. Prevention: when a document is deprecated, each pointer to it must be classified first — a pointer that describes the deprecation stays; a pointer that sends readers there as the authority (from code, or from active rules) gets removed, not renamed. Suggested addition to CLAUDE.md, near the legacy-reference rule: "Do not add or keep pointers to the legacy references in code or active docs." (Adopted 26-10-02, approved by Julian in chat.)
