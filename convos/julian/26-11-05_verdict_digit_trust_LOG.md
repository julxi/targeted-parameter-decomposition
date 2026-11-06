C: (transcribed from chat) have we checked before that the verdict is 0 or 1 and not something in between like 0.9?

---

C: (transcribed from chat) I saw that the conversation ladded in @convos/julian/26-11-05_evaluation_categories_LOG.md . Can you move it somewhere more relevant?

26-11-05, Claude Opus 5.5 — *entries moved here from the evaluation-categories topic*; same session as the entries below. They were written at the bottom of `convos/julian/26-11-05_evaluation_categories_LOG.md` (untracked, never committed) and moved here verbatim on Julian's request, because they are about the paper's verdict readout, not about evaluation categories. Their output folder stays `~/spd_out/26-11-05_evaluation_categories/` (a run was writing there during the move).

---


26-11-05, Claude Opus 5.5 — *is the verdict binary? distribution of P('1') under the verdict prompt*; same session. Julian asked in chat whether we had checked before that the verdict is 0 or 1 and not something in between like 0.9. I grepped all `convos/julian/` LOGs and SUMMARYs and computed the distribution from stored per-statement scores.

**Earlier checks: only the format and the sign.** We checked that the answer is a digit: digit mass P('1') + P('0'), 0.984 pooled in-distribution (`convos/julian/26-10-10_bolder_experiments_LOG.md`, "98% of the mass on '1' or '0'"; paper draft Table 1). We also checked accuracy and AUC of the sign of s = log P('1') − log P('0'). Nobody looked at how confident the verdict is, i.e. whether P('1') among the two digits sits near 0 or 1. The verdict "1 vs 0" is binary only because it is defined as sign(s).

**Data.** `~/spd_out/26-10-22_tpd_truth_paper/b2_full/b2_scores.npz` (plan step B2: the original model, prompt `amount|nl` = the verdict prompt of all decompositions, v3 test split; one run, bf16). Sanity check: in-distribution accuracy 0.979 and digit mass 0.984 come out as recorded. q = P('1') / (P('1') + P('0')) = sigmoid(s).

| | in-distribution (n = 1,794) | OOD (n = 9,328) |
|---|---|---|
| q < 0.01 or > 0.99 | 0.525 | 0.028 |
| q < 0.1 or > 0.9 | 0.804 | 0.424 |
| q in [0.1, 0.9] | 0.196 | 0.576 |
| median \|s\| | 4.75 | 1.88 |
| \|s\| < 0.12 (≈ one bf16 step) | 1% of statements | 5% |
| wrong verdicts: n, median \|s\|, share with q < 0.1 or > 0.9 | 38, 0.38, 0.132 | 1,855, 0.88, 0.143 |

[verified: computed 26-11-05 from the npz above, script inline in this session, not saved]

In-distribution, per file, share with q in [0.1, 0.9]: larger_than 0.000, smaller_than 0.113, neg_animal_class 0.135, sp_en_trans 0.219, cities 0.259, neg_cities 0.334, neg_sp_en_trans 0.344, animal_class 0.351, element_symb 0.429, neg_element_symb 0.554. Cities has accuracy 1.000 but 26% of its verdicts in the middle band.

**Reading** [concluded: from the table; one prompt, original model only]:
- In-distribution the verdict is mostly sharp but not binary: half the statements at q beyond 0.01/0.99, a fifth in the 0.1–0.9 band. Errors are mostly hesitant (13% of them confident), so the model's errors are mostly near-ties, not confident false beliefs.
- OOD the verdict is mostly soft: 58% in the 0.1–0.9 band and almost nothing beyond 0.99. Every OOD analysis on sign(s) puts a cut through a soft distribution.
- For this topic: τ = sign(s) is unstable for the near-ties (1% in-distribution, 5% OOD, within one bf16 step of 0). For P4 (swap the active components' write between statements of opposite τ) that suggests either (a) a continuous target: the swap should move s to s(x'), not only its sign; or (b) restricting the pairs to confident statements, e.g. |s| > 2.2 (q beyond 0.1/0.9) for both members [assumed: thresholds not tested]. (a) is stricter and uses all pairs. Julian's choice.
- For the paper: KL recovered and the separation share use s or the full distribution, so they are not affected by this. Accuracy-type numbers (verdict agreement, "inverts the verdict") count a move from q = 0.55 to 0.45 as a full flip; on OOD that is the common case.

---

C: (transcribed from chat) I don't quite understand what you answered. What I meant, when the next token is `0` it could actually be the first token of `0.9`

26-11-05, Claude Opus 5.5 — *does '0' start "0.9"? multi-token answers after the verdict digit*; same session. My entry *is the verdict binary?* above answered a different question (how confident the model is between '1' and '0'). Its numbers are right but they don't answer Julian's question, which is about what follows the digit.

**The concern applies.** The verdict prompt (`amount|nl`) is "<statement>\nThe amount of truth in this statement is " and the score reads only the first token after the trailing space: s = log P('1') − log P('0'). Qwen tokenizes numbers one digit per token, so '0' is also the first token of "0.9", "0.5", "0%", and '1' is the first token of "10%", "1.0", "100%". "0.9" (mostly true) and "10%" (mostly false) would both be counted as the opposite verdict.

**What we have checked before: only anecdotes.** I grepped all LOGs and the outputs under `~/spd_out/`. No systematic check exists. What exists:
- Unsteered greedy continuations of 6 statements under exactly this prompt (`~/spd_out/26-10-10_bolder_experiments/steering_sf/full/examples.out`) [verified: read 26-11-05]: "1.0 (completely true).", "0%.", "100%.", "0 out of 1.", "100%.", "0.\n\nTo determine…". So the model does continue the digit with multi-token numbers, all six consistent with the digit's reading in these cases.
- Under interventions, hedges appear: "0.5" (component steering, L11:70 reversed; `convos/julian/26-10-16_component_steering_LOG.md`), "0%", "100%" (belief steering).
- The R5 prompt screen's greedy continuations were run *without* the trailing space (they show ": [A] very little …"), so they don't say what follows the digit.

**Proposed check** [assumed: design, not run]: per statement, the probability of each answer format, read from 2–3 extra forward passes (prompt + '0', prompt + '0.', prompt + '1', prompt + '1' + '0'):
- after '0': '.' then a digit 1–9 ("0.x", partial truth), '%' or '.'/newline/space (plain 0), ' out' ("0 out of 1");
- after '1': '0' then '%' or '0' ("10%", "100%"), '.' then '0' ("1.0") or a digit 1–9 ("1.x", odd).
- Outcome: per family, the mass on answers whose meaning contradicts the first digit ("0.x" with x ≥ 5, "10%"), and the share of statements where it exceeds, say, 0.1. If small, s is a valid readout of the verdict; if not, the score should be computed over the parsed number.
- Feasible locally on the CPU (Qwen2.5-7B-Instruct is cached, 48 GB RAM free, no GPU here) on a stratified subset, e.g. 20 test statements per in-distribution family = 200; the interventions (where "0.5" was seen) would need a GPU.

---

C: (transcribed from chat) Can you run a very short test and then tell me how long the 200 test statements would take?

26-11-05, Claude Opus 5.5 — *digit-continuation smoke test: timing and first observations*; same session. Script `~/spd_out/26-11-05_evaluation_categories/digit_continuations.py`. Per statement, three token-id sequences (prompt + '0' + '.', prompt + '1' + '0', prompt + '1' + '.') in one forward pass, giving P(first answer token), P(second | first) and P(third | first two). Original Qwen2.5-7B-Instruct, bf16, local CPU (10 threads). Statements from the stored B2 test set. Output `smoke/` and `smoke.log` there.

**Timing** [verified: `smoke.log`, 10 statements, one run]: model load 1 s (weights already in the page cache; cold from disk will be longer), first statement 10.4 s (warm-up), then 1.2 s per statement, 27 s wall time in total, peak memory 14.9 GB. **200 statements with this design: about 4–5 min** [assumed: linear extrapolation of the 9 warm statements].

**Two design problems the test showed:**
1. *Sampling:* "first n per file" picked only false statements, because B2 stores each file's `_false` rows first. Fixed: now the first n of each of the 20 `_true`/`_false` datasets (`--n_per_dataset 10` = 200, balanced). Not rerun yet.
2. *Formats are richer than the three planned sequences:* Qwen has merged tokens like '%.' and '%.\n\n', and '.' alone is often a sentence period (followed by ' \n\n'), so after '0' the decimal reading is '.' followed by a digit, not '.' alone. Other formats seen: "0 out of …", "1 out of …", "1/…", "1, …". "1 out of 10" or "1/5" would mean mostly false, and they need more than three tokens to resolve.

**First observations** [verified: read all 10 in `smoke.log`; anecdotal, all 10 are false statements, so the case that matters most, true statements answered '0' as in "0.9", is not covered yet]:
- After '0' the model mostly writes "0%." or a plain "0." with a newline. That is a clean false verdict.
- On false negated statements, "0.5"/"0.1" hedges carry some mass: neg_element_symb ("Thallium does not have the symbol Tl."): P('0') 0.754 × P('.' | '0') 0.331 × P('5' | "0.") 0.539 ≈ 0.13 on "0.5" [verified: product of logged values]. Also neg_animal_class, neg_cities, smaller_than show digits after "0.".
- The '1' mass on false statements is partly "10%", i.e. also false: cities ("…Lu'an is in Turkey."): P('1') 0.033, of which P('0' | '1') 0.816 and then P('%'-tokens | "10") ≈ 0.99, so about 81% of the '1' mass is "10%". For the number families "10" continues to "100" (P('0' | "10") ≈ 0.99), there "1" + "0" means 100.
- Consequence if this holds up: s = log P('1') − log P('0') mixes formats with opposite meanings under the same digit. In these examples the sign of s is unaffected (the '1' mass is small), but its size is.

**Proposed instead** [assumed: design and time estimate, not tested]: a small tree search per statement. Expand every answer prefix with probability ≥ 0.005, up to about 5 tokens or the first non-number token, parse each leaf into a value ("0.5", "10%", "1 out of 10", "1/5"), and report per statement the distribution over parsed values and its expected truth value. About 10–30 sequences per statement instead of 3. Estimated 10–30 min for 200 on the CPU.

---

C: (transcribed from chat) My question is, how much can we trust the first digit of the verdict, not the continuation of the reverse of a verdict. Let's say model verict is 1, how likey is this an expression of 10% or the verdict is 0, how likely is this an expression of 0.99 or other creative ideas. My idea is just to take greedy continuation and check them manually. better idea?

26-11-05, Claude Opus 5.5 — *trusting the verdict digit: sampled continuations instead of greedy*; same session.

The question is P(the full answer means the same as the digit | the first token is the verdict digit), for the digit the model actually picks.

**Greedy can mislead here, shown by the smoke test** [verified: `~/spd_out/26-11-05_evaluation_categories/smoke.log`, statement [6] "Thallium does not have the symbol Tl.", verdict '0']: after '0' the most likely single token is '.' (0.331), ahead of '%.\n\n' (0.227), '%.' (0.201) and '.\n\n' (0.177), and after "0." it is '5' (0.539). So greedy writes "0.5". But the plain-false readings ("0%", "0." + newline, "0 out …") together carry at least 0.227 + 0.201 + 0.177 + 0.027 = 0.63 of the '0' branch, against 0.331 × 0.539 ≈ 0.18 for "0.5". Greedy picks per token, and the same meaning is split over several tokens ('%.' vs '%.\n\n'), so it can report a minority meaning as the answer. It also gives one path per statement, so it can't say "how likely".

**Proposal** [assumed: design, not run]: per statement, force the first token to the verdict digit (argmax of '0'/'1'), then
1. sample about 10 continuations of ~10 tokens at temperature 1 (the model's own distribution), plus the greedy one for reading;
2. parse each with a regex into a value ("0%", "0.9", "10%", "100%", "1 out of 10", "1/5", plain "1"/"0") and classify as agrees with the digit / contradicts it (e.g. '0' → "0.9" or "0.5+"; '1' → "10%", "1 out of 10") / unparsed;
3. read all contradicting and unparsed samples by eye and a random sample of the agreeing ones (checks the regex);
4. report per family: the mean share of contradicting continuations (the "trust" number), and the statements where it is above, say, 0.2.

Same 200 statements (10 per `_true`/`_false` dataset). Per-statement precision is coarse with 10 samples (one sample = 0.1), but per family that is 200 samples. CPU time estimated at 15–30 min [assumed: about 10 decoding steps with a KV cache for a batch of 11, not measured].

---

C: (transcribed from chat) yes, run this

26-11-05, Claude Opus 5.5 — *verdict trust: 200 statements, sampled continuations after the verdict digit*; same session. Script `~/spd_out/26-11-05_evaluation_categories/verdict_trust.py`, analysis `verdict_trust_analysis.py`; outputs `trust_full/verdict_trust.json`, `trust_full.log` (every continuation), `trust_full_analysis.out`. Original Qwen2.5-7B-Instruct, bf16, CPU. 200 in-distribution test statements (the first 10 of each of the 20 `_true`/`_false` datasets in B2's stored order). First token forced to the verdict digit (the more likely of '0'/'1'), then 10 continuations of 10 tokens each, sampled at temperature 1 by a hand-written loop (no top-k/top-p/repetition penalty; Qwen's generation_config would have sharpened them), plus greedy. Each answer is regex-parsed into a truth value and classed against the digit: agree_exact (0 or 1, incl. "0%", "100%", "0 out of 10", "1.0"), agree_partial (same side, not extreme, e.g. '0' → "0.3"), hedge ("0.5"), contradict (other side, e.g. '0' → "0.6", '1' → "1/3"), unclear (no value in [0, 1]). Run time 10.7 s per statement, 36 min in total (the smoke test's 7.4 s did not hold for the longer statements).

**Results** [verified: `trust_full_analysis.out`, one run, seed 0, 2,000 samples; verdict = label for 196/200]:

| verdict digit | statements | agree_exact | agree_partial | hedge "0.5" | contradict | unclear |
|---|---|---|---|---|---|---|
| '1' | 96 | 0.996 | 0.000 | 0.000 | 0.001 | 0.003 |
| '0' | 104 | 0.961 | 0.012 | 0.024 | 0.004 | 0.000 |
| all | 200 | 0.978 | 0.006 | 0.013 | 0.003 | 0.002 |

- Greedy: 196 agree_exact, 4 hedges, no contradiction.
- **Hedges sit at the near-ties** (q_max = the larger of P('0'), P('1') renormalised over the two): 11 statements with q_max < 0.7 have a hedge share of 0.145; the 37 with 0.7–0.9 0.014; the 152 with q_max ≥ 0.9 0.003. Most hedges come from four statements: "The city of Malacca is not in Malaysia." (P('0') = P('1') = 0.42; 5 of 10 samples "0.5"), "…Seongnam-si is not in South Korea." (4), "Hafnium has the symbol Hf." (3; a wrong '0' verdict), "Silver does not have the symbol Ag." (3). By dataset, at most neg_cities_false 0.100; the wrong verdicts (element_symb_true answered '0', 3 statements) have 0.167.
- **All contradictions read by eye** (6 of 2,000): '0' → "0.6" (3: "The whale is a fish.", "The rat is not a mammal.", and the true "Thallium has the symbol Tl.", where 0.6 agrees with the label), '0' → "0.63" ('venir' = 'to know'), '1' → "1/3" ("Silver has the symbol Ag."). Plus one regex miss, classed agree_exact: '1' → "1 (on a scale of 1-10", where 1 is the bottom of the scale. So '1' contradicts in 2 of 960 samples, '0' in 4 of 1,040.
- **Unclear** (3): "1.2.", "12." (degenerate), "110%." (emphatic true).
- **Partial** (12): '0' → "0.1"–"0.4" or "0.333…", mostly on false negated animal statements; same side as the digit.
- **No "0.7"/"0.8"/"0.9" after '0' and no "10%" after '1'** [verified: search over all 2,200 answers]. "out of 10/100" appears only as "0 out of 10" (agree).
- **Regex check:** the 40 random agree_exact samples and the 27 that mention a scale all read correctly, except the "scale of 1-10" case above. Most of the 27 are "1 (on a scale where 0 is completely false…", which agrees.

**Reading** [concluded: from the table and the reading by eye; in-distribution, original model, one prompt]:
- **The verdict digit can be trusted as the model's answer on in-distribution statements.** A '1' means "true" in ≥ 99.6% of its continuations. A '0' means "false" or false-side in 97.3%, "undecided" (0.5) in 2.4%, and the true side in 0.4%. The worry "'0' is really 0.9" did not show up: the largest value after '0' is 0.63.
- **The one real effect is hedging, and s already shows it.** "0.5" answers are the model's own "undecided" and come almost only from statements where P('0') ≈ P('1') anyway. So P('0') contains a small "undecided" share at the near-ties, which s scores as false.
- **Not covered:** (1) OOD statements, where the verdict is much softer (58% of OOD statements have q in [0.1, 0.9], entry *is the verdict binary?*), so hedges may be more common there; (2) decompositions and interventions, where "0.5" answers were seen before; (3) the minority digit's meaning (e.g. the small '1' mass on false statements being "10%", smoke test). (3) changes the size of s, not its sign. (1) and (2) need a GPU or a long CPU run.
