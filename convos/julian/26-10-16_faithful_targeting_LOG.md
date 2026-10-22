We have some experiments that try to apply tPD in order to find truth mechanisms. I think some of the best results are the experiments 4a and 4b/4c. I like 4a because it's principled but the decomposition has too much freedom. I like 4b/4c because they are faithful but they are less targeted. I had an idea of a middle approach. Use 4a but instead of reconstruction terms we take a orthogonality penalty. This is not very principled but more like a band aid solution. So I wanted to brainstorm and ask if you have any ideas?

---

26-10-16, Claude Opus 5.5 — *brainstorm: between 4a and 4b*; I read INDEX.md, the bolder-experiments SUMMARY (Exp 4 section), the component-steering SUMMARY, the 4a/4b result entries in the bolder-experiments LOG (*Exp 4a/4b*, *options for the idle H100*, the y = t + r comment), the 4a/4b configs and how the code masks Δ on target data. The file was renamed from `find_a_better.md`; the request above is verbatim.

Background, for a reader without the other topics: Exp 4a and 4b are tPD runs on the `down_proj` of Qwen2.5-7B-Instruct layers 11–17 at the statement end `'.\n'` of the prompt "<statement>\nThe amount of truth in this statement is ", trained so that the components reproduce the model's verdict digit (KL at the answer). 4b adds a per-layer reconstruction term: under the stochastic masks, each MLP's output at `'.\n'` must match the original output y. 4c is 4b at every statement token. Results (one seed each, from the bolder-experiments SUMMARY): 4a reproduces the verdict with 2.2 active components, but through a layer-11 write 12.9× the norm of y and nearly orthogonal to it (cosine 0.03), undone in the full model by inactive components and Δ. 4b's components are a part of y (relative error |y − A|/|y| 0.44, cosine 0.88) with 10.8 active components, most of them task/family detectors; in 4c the components write only 13% (end token) / 25% (object token) of the truth gap directly, Δ 73%.

### 1. Why 4a has so much freedom (and why that matters for the choice of fix)

- **Δ is already randomly masked on target data** [verified: code read; `spd/utils/component_utils.py`, `calc_stochastic_component_mask_info`: with `force_delta=None`, which is the target-data case, Δ's mask is `torch.rand`, uniform in [0, 1]; the target loss of 4a is `StochasticReconSubsetLoss`]. So "everything that is not a component must be removable without changing the verdict" is already part of 4a, and the inactive components get the same treatment. 4a's decomposition satisfies that criterion on average (its adversarial-mask loss is poor: PGD 1.20 vs 4b 0.13).
- What 4a lacks is any constraint on the decomposed position itself. The verdict is read nine tokens later, so the loss sees the residual at `'.\n'` only through what later attention layers read from it (and through MLPs 12–17 at that position). Most directions of the write are invisible to the loss [concluded: architecture; how large the invisible part is was not measured]. Whatever lives in that invisible space costs nothing, so the cancelling writes of the inactive components and Δ can live there too. The narrower the lens, the larger that free space. In ordinary SPD the output is every next-token distribution at every position, so the space is far smaller.
- So 4a is principled in the sense of tPD's own criterion: what the target sees is reproduced, and everything else can be removed. The problem is that the criterion only looks at outputs. A fix therefore needs a constraint on the internal state at the decomposed position. The question is how weak that constraint can be.

### 2. Julian's orthogonality penalty is more principled than it looks

Reading [assumed: please confirm]: "orthogonality" = the components' write A and the rest y − A = I + Δx (inactive components plus Δ) should be orthogonal at each decomposed position and layer: penalise ⟨A, y − A⟩² (relative to |y|⁴).

- Geometry [concluded: algebra]: ⟨A, y − A⟩ = 0 means A lies on the sphere with diameter y (Thales). Equivalently, **A is exactly the orthogonal projection of y onto A's own direction**: the components write precisely y's coordinate along the direction they write. Consequences: |A| ≤ |y|, cos(A, y) = |A|/|y|, error |y − A|/|y| ≤ 1. 4a's 13× write is impossible, and so is a small foreign write, because A cannot contain anything y doesn't have along A. That is a formal version of "A is a part of y" in the y = t + r picture (y = truth part t + rest r), without requiring A ≈ y.
  OUTDATED (26-10-21): "and so is a small foreign write" is wrong: on the sphere, the part of A orthogonal to y can still reach |y|/2, so the condition caps a write only along directions y has little of. Correction in the entry *4d: orthogonality penalty implemented, trial, two full runs launched*, section 2.
- It is **stronger and cleaner than the hinge the agent proposed on 26-10-16** (bolder-experiments LOG, *options for the idle H100*: penalise only negative A·(y − A)). The hinge allows A anywhere inside the Thales ball, so a small A that is almost orthogonal to y (a foreign write) passes. Prefer Julian's equality version.
- **Per-component version (my recommendation):** require y − A ⟂ each active component's write direction u_c (penalty Σ_c CI_c ⟨u_c, y − A⟩² / (|u_c|² |y|²)). That says A is the least-squares fit of y within the span of the active components' write vectors, i.e. "where you write, write exactly what the MLP writes; where you don't, no claim". With one direction it reduces to Julian's condition; with several it forbids the case where two components overshoot in opposite directions and still have a sum that satisfies the single condition. Same cost: one dot product per active component per position.
- Open details: (a) which state to apply it to: the CI-weighted state is the simplest; applying it under the stochastic masks would also constrain the inactive components and drift towards 4b; (b) the metric: Euclidean in the raw residual basis lets the massive-activation channels dominate (one channel held 22–29% of the squared norm at the period, 26-10-10). Per-channel std normalisation as in Exp 1's residual lens is the obvious choice [assumed]. (c) A = 0 satisfies it trivially, so only the verdict loss pushes components on; the coefficient trades verdict recovery against faithfulness.
  OUTDATED (26-10-21): in (b), the 22–29% was the residual stream at the period, not the MLP outputs. On the MLP outputs at the statement end the top channel holds at most about a sixth of |y|², and the plain Euclidean metric was chosen instead of per-channel std normalisation. See the entry *4d: orthogonality penalty implemented, trial, two full runs launched*, section 1.
- Remaining weakness: it constrains the geometry of the write, not its mechanism. A projection of y onto a direction the decomposition is free to pick could still be read downstream in an unusual way. I don't see a concrete cheat; the cancellation test and analysis V would show one.

### 3. Other ideas

a. **4b with a smaller reconstruction coefficient (0.1, 0.01)** — the cheapest middle ground, no new code. 4a = coefficient 0, 4b = 1, untuned [assumed in the 4b config header]. Any new penalty should be compared with this baseline. Otherwise "better than 4a and 4b" may only mean "a weaker reconstruction term".

b. **Why 4b is untargeted, and a weighted reconstruction.** Squared error weights each direction of y by its variance. Truth is a small part of the variance (10–15% of the across-statement variance of the residual stream at the period, measured 26-10-10; not measured for the MLP outputs at `'.\n'`), task and family structure a large part. So with C = 96 per layer, the components spend their capacity on the high-variance parts, and truth is the first thing left to Δ. That fits 4c's 73% of the truth gap in Δ [assumed: hypothesis, untested]. A fix: reconstruct in a metric that up-weights what the verdict depends on, (y − A)ᵀ(λI + G)(y − A), with G = mean of g gᵀ and g the gradient of the verdict score with respect to that MLP output, computed once on the original model (top eigenvectors suffice). Expanded to second order around the original model, the verdict KL weights the error by G, so this term is a linearised verdict lens plus an isotropic floor λ; λ → ∞ gives 4b. The linearisation removes the room for nonlinear re-implementation, and λ blocks foreign writes. The term involves no probe direction, only the model's own sensitivity. Cost: computing G per layer, plus a new loss.

c. **Reconstruct the truth-dependent variation, not y.** Pair each statement with one from the same family and the opposite label, and require the components to reconstruct the difference y(x) − y(x′) instead of y(x). Whatever is shared within a family (task, template, massive activations) then needs no components. This targets the complaint about 4b/4c directly: task detectors become unnecessary. Caveat: the pair also differs in content (object, or negation for the affirmative/negated twins), so the difference is not pure truth. The tiu structure decides which pairing is cleanest. A related variant: resample Δ's write from the paired statement instead of zeroing it (interchange-style), which makes the "off" state stay on-distribution.

d. **Post-hoc selection from a faithful decomposition** (no retraining). Take 4b/4c as they are, then learn a sparse mask over their components that keeps the verdict when all other parts are mean-ablated. That gives faithful parts from the reconstruction and targeting from causal selection. Limitation: it can only select what 4b/4c found, and they left most of the truth write in Δ (point b).

e. **Mean instead of zero as the "off" value** for Δ (and inactive components). Zeroing seven MLP outputs is a large off-distribution change that the components have to compensate for, and compensation is one driver of re-implementation; the layer scan preferred mean ablation for the same reason. On its own this doesn't stop foreign writes, so it is a complement to 2 or b, not an alternative.

### 4. Before choosing: what "more targeted" should be measured by

So far "less targeted" in 4b/4c is a qualitative reading ("most components are task detectors"). Proposed numbers to compare 4a, 4b, 4c and any new variant [assumed: proposal]:
- faithfulness: the cancellation test (error, cosine, |A|/|y|), plus the foreign part |A − proj_A(y)| / |A|;
- targetedness: share of active components whose removal from the components-only state changes the verdict; active components per statement; share of the direct truth write in components vs Δ;
- verdict recovery: share of the all-off KL recovered (analysis V).

A critical point on the premise: some task detectors may genuinely be part of the verdict computation. The steering topic found that L17:5 and the "false negated" group work as negation × relation components. Whether 4b/4c's task detectors are untargeted or necessary is exactly the removal test above. If most of them matter for the verdict, 4b is less untargeted than it looks, and the real deficit is point b (truth left in Δ).

### 5. Recommendation [assumed: not discussed]

1. Run 4a + per-component orthogonality (Julian's idea, section 2) at `'.\n'` ("4d"), side by side with 4b at reconstruction coefficient 0.1 (3a). Same settings as 4a/4b, so the comparison is clean. Both are about one hour of training each on the H100; the orthogonality loss needs new code and a test, roughly as much work as the hidden-activation reconstruction term was [assumed].
2. Evaluate everything with the metrics in section 4.
3. If 4d is faithful but still leaves the truth write in Δ, try the sensitivity-weighted reconstruction (3b).

All 4a/4b/4c numbers are from one seed each; a variant that differs from them by a small margin is not evidence of anything until a second seed exists.

---

C: (transcribed from chat) I'd like to start a run with orthogonality penalty (Julian's idea).
C: (transcribed from chat) Some comments about this:
- I want the all-component version: y - A is orthogonal to A. I'm not quite sure why, but the idea is that the interaction between different components should not be restricted.
-  apply it to CI-weighted state
- don't know about the metric. What is the sensible thing to do here
- Yes this pushes A down, but we always have a trade off here
C: (transcribed from chat) Talking about tradeoffs. It's not clear to me however how we should finetune this now that we have another parameter.
C: (transcribed from chat) I have rented a H100, please use that
C: (transcribed from chat) if you need to do any small experiments
C: (transcribed from chat) or big

---

26-10-20, Claude Opus 5.5 — *4d: orthogonality penalty implemented, trial, two full runs launched*; I re-read this LOG in full, INDEX.md, this topic's SUMMARY, the Exp 4 section of the bolder-experiments SUMMARY, the 4a/4b configs, `hidden_acts_recon_loss.py`, the training step in `run_spd.py`, and `exp4_analysis.py`.

Julian's comments answer the open details of section 2 of the brainstorm entry. Decided [decided: transcribed above]: one condition for all components together, ⟨A, y − A⟩ = 0 (not the per-component version I had recommended), in the CI-weighted state. Open were the metric and how to tune the new coefficient.

### 1. The metric: Euclidean, checked on the data

The worry (brainstorm, section 2b) was that a few massive-activation channels dominate |y|² and so the inner product. Measured on the H100 (`~/spd_out/26-10-16_faithful_targeting/channel_share.{py,out}`; 15 test statements per tiu dataset, 300 in total; original model, MLP output y of `down_proj` 11–17 at the statement end) [verified: one run]:

| layer | median \|y\| | top channel's share of \|y\|² (median / p90) | top 5 (median) | top channel's share of the across-statement variance |
|---|---|---|---|---|
| 11 | 19.1 | 0.032 / 0.065 | 0.072 | 0.009 |
| 12 | 18.4 | 0.020 / 0.074 | 0.040 | 0.017 |
| 13 | 22.4 | 0.147 / 0.204 | 0.272 | 0.011 |
| 14 | 19.7 | 0.014 / 0.059 | 0.034 | 0.020 |
| 15 | 26.3 | 0.174 / 0.210 | 0.315 | 0.011 |
| 16 | 20.8 | 0.016 / 0.046 | 0.035 | 0.011 |
| 17 | 26.6 | 0.122 / 0.162 | 0.234 | 0.010 |

The top channel is 2570 (the massive-activation channel of the bolder-experiments topic) in 54–100% of statements. It holds at most a sixth of |y|² at the statement end, so no channel dominates. (The 22–29% recorded on 26-10-10 was the residual stream at the period, not these MLP outputs.) The Spanish-translation statements end in the token `'.\n` (with the quote), not `.\n`; the position is the same.

**Recommendation and implementation: plain Euclidean inner product, per position divided by |y|²** [concluded]. Reasons: residual-stream addition happens in this basis; every faithfulness number so far (cancellation test, 4b's relative reconstruction term) is Euclidean, so 4d stays comparable with them; and the check shows no channel dominating. My earlier suggestion of per-channel std normalisation is [superseded]: with centring, the condition would no longer be about the decomposition y = A + (y − A) of each single output (A = 0 would stop satisfying it), and without centring it is unnecessary given the table.

### 2. What the condition does and does not do (correction to the brainstorm)

- On the sphere ⟨A, y − A⟩ = 0, A is y's projection onto A's own direction u: **a write along any direction u is limited to y's own coordinate along u**, |A| = |y|·|cos(u, y)|. 4a's layer-11 write (12.9 |y| at cosine 0.03) would be capped at 0.03 |y| along that direction. That is what should block the re-implementation [concluded: algebra].
- The brainstorm said this leaves "no foreign content". That needs a qualifier: the part of A orthogonal to y's own direction can still reach |y|/2 (at |A| = |y|/√2). This is no flaw of the condition: a genuine part t of y = t + r with t ⟂ r also has cos(t, y) = |t|/|y|, so a small genuine part is necessarily almost orthogonal to y as a whole. But it means "foreign part relative to y" is no measure of unfaithfulness here. Whether such a write is read downstream as cancelled content depends on the reader [concluded: algebra].
- It does not simply push A down. Inside the ball (e.g. a partial copy A = t·y, 0 < t < 1) the penalty pushes A outward, towards the full projection: A = t·y costs (t(1 − t))², with A = y/2 the worst at 1/16. Outside the ball it pushes inward.
- **Scale of the penalty** [concluded: algebra; seen in the trial below]: with r = |A|/|y|, (⟨A, y − A⟩/|y|²)² = (r·cos − r²)². Near A = 0 it grows like r⁴: a write of 0.3 |y| along a direction y has nothing of costs (0.09)² ≈ 0.008, while 4a's write costs ≈ 166² ≈ 3·10⁴. So the term rules out large writes at any reasonable coefficient, but small and medium ones only at larger coefficients. That is why the coefficient matters, see section 4.

### 3. Implementation [verified: tests]

- `spd/metrics/orthogonality_loss.py`: `component_orthogonality_loss` and the eval metric `ComponentOrthogonalityLoss`. Config class `ComponentOrthogonalityLossConfig` (`spd/configs.py`; supports the position mask). Per decomposed module, at the selected positions: y = W x and A = Σ_c CI_c U_c (V_c · x), both on the module's input x from the original forward (the training step's existing input cache, passed to `compute_losses` as the new argument `pre_weight_acts`). This is the same A as in the cancellation test and needs no extra forward. Note the difference to 4b's reconstruction term, which compares outputs in one masked forward through all modules. fp32, no autocast (⟨A, y⟩ − |A|² is a difference of similar numbers near the sphere). Mean of (⟨A, y − A⟩/|y|²)² over positions and modules. The eval metric additionally logs, per module, the mean signed ⟨A, y − A⟩/|y|² (`…/inner/…`) and |A|/|y| (`…/norm_ratio/…`). Excluded from the Pile losses, like the reconstruction term. WandB short name `CompOrtho` (`spd/utils/wandb_utils.py`).
- Tests `tests/test_orthogonality_loss.py`: hand computation at the selected positions (also distinguishable from a loss that ignores the mask); components set to t·W, so A = t·y, give t(1 − t) for t ∈ {0, 0.5, 1, 2, −1}, for the loss and the eval metric; gradients reach the components and the CIs only at selected positions. Mutation check: replacing ⟨A, y − A⟩ by ⟨A, y⟩ fails 5 of 7 tests, dropping the CI weighting fails 2. New generic test `tests/test_metric_short_names.py`: the first trial crashed at startup with `KeyError: 'ComponentOrthogonalityLoss'` in `flatten_metric_configs`, because every metric class needs a WandB short name. The test now catches that before a GPU run.
- `make check`: clean for all changed files. Pre-existing, not from this change: 6 basedpyright errors in `spd/experiments/rotgrid/analysis/rank_rotgrid_decompositions.py`, and `tests/test_rotgrid_decomposition.py::test_the_checked_in_sweep_expands_to_distinct_runs` fails with and without my changes (checked with `git stash`).
- Config `spd/experiments/lm/honesty_targeted_decomposition/config_truth_sf_verdict_ortho.yaml` ("Exp 4d") = 4a plus the term at coefficient 1, everything else identical to 4a.

### 4. Tuning: how to handle the extra coefficient

My answer to Julian's question [assumed: proposal]:
- Change only the new coefficient. All other settings stay at 4a's values. They were never tuned for 4a either, and retuning them now would make 4a/4b/4d incomparable.
- Treat the term as a constraint, not as a third objective to trade against. There is no single number that says which point of the KL / orthogonality / L0 front is best, so tuning means fixing a rule in advance. The proposed rule: take the smallest coefficient at which the components no longer re-implement, measured by the cancellation test (|A|/|y| ≤ 1 per layer, no I/D cancellation in the full model). Among those, prefer the one with the best verdict recovery. The config header's "RMS violation ≤ 0.05" is too weak as a criterion, because of the r⁴ scale near 0 (section 2); the cancellation numbers decide.
- Because of the r⁴ shape, coefficients a decade apart behave qualitatively differently. At 1, only large writes are blocked. At 10, the trial's small orthogonal writes (penalty ~10⁻³) already cost about a third of the verdict KL (~0.03). So the first step runs both: 1 and 10.
- If more than two or three values become necessary, or the term goes into more runs: replace the coefficient by a tolerance with an adaptively raised Lagrange multiplier (dual ascent on λ, as in GECO-style constrained training). That trades a coefficient without units for a tolerance in the units of ⟨A, y − A⟩/|y|². Not implemented (YAGNI).
- One seed per run: differences between 4a, 4b and 4d that are smaller than the run-to-run spread mean nothing. That spread is unknown for Exp 4.

### 5. Trial (300 steps) [verified: one run, `s-37397e7d`, H100; config, log and analysis smoke output in `~/spd_out/26-10-16_faithful_targeting/trials/`]

Stable, 300 steps in 3:21 including evals, no NaNs. L0 2.39 at step 300. The components write tiny vectors: CI-weighted |A|/|y| = 0.02–0.035 at initialisation and 0.014–0.030 at step 300 in every layer; the penalty stayed between 10⁻⁶ and 10⁻³. The 4a/4b trials ran at the earlier all-tokens setting, so they are no comparison.

Analysis smoke test on this checkpoint (`exp4d_analysis.py`, 10 statements per dataset, 200 in total): it runs. Already at step 300, the active components (rounded CI) write 0.07–0.17 |y| at cos(A, y) ≈ 0. Inactive components and Δ are each 2–4.6 × |y| and cancel (cos(I, D) −0.89 to −0.96), presumably the initial state of components + Δ = W [assumed]. 30% of the active CIs lie between 0.01 and 0.99, so the rounded A (the analysis's "components" state) is about twice the CI-weighted A (on which the penalty acts) and violates the condition more (RMS 0.06–0.19 vs 0.02–0.05). Julian chose the CI-weighted state; the analysis therefore now reports both, plus a verdict state "components_ci".

### 6. Full runs launched (26-10-20, 13:03 UTC; H100 rented by Julian)

`~/spd_out/26-10-16_faithful_targeting/full/run_full.sh`: Exp 4d at orthogonality coefficient 1 (`ortho1.yaml` = the committed config), then at 10 (`ortho10.yaml`, otherwise identical), 5,000 steps each, checkpoints to WandB, each followed by `exp4d_analysis.py`. `exp4d_analysis.py` (in `~/spd_out/26-10-16_faithful_targeting/`) is `exp4_analysis.py` plus the orthogonality numbers ⟨A, y − A⟩/|y|² (mean and RMS), the foreign part |A − proj_y A|/|y| for the rounded and the CI-weighted A, the share of active CIs strictly inside (0.01, 0.99), and the verdict state "components_ci" (CI-weighted components, Δ off).

Prediction [assumed]: at coefficient 1, a write several times |y| like 4a's is gone, but writes of a few tenths of |y| orthogonal to y remain possible. At coefficient 10, the components either stay a part of y with fewer of them active, or the verdict KL gets worse than 4a's. Not proposed here and possibly worth it as a reference: 4b at reconstruction coefficient 0.1 (brainstorm section 3a), to see whether "weaker than 4b" alone gives the same thing.

The GPU shows 26–31 GB in use by something outside our container even when nothing of ours runs (no process visible in our container). Our jobs fit next to it, but wall times on this instance are not comparable with other rentals.

---

26-10-20, Claude Opus 5.5 — *4d at coefficient 1: results; coefficient-10 run stopped*; continuing the entry above.

**Run** `s-5c284ebb` (Exp 4d, orthogonality coefficient 1; 34:53 on the H100; checkpoint on WandB [verified: API, `model_5000.pth`, 334 MB]). Analysis `exp4d_analysis.py` on all 1,794 test statements, replacement at `'.\n'` only. Outputs: `~/spd_out/26-10-16_faithful_targeting/full/` (`analysis_ortho1.log`, `analysis/exp4_s-5c284ebb.{json,npz}`, training log `ortho1.log`) [verified: one seed].

| run | components only: KL (share of all-off KL 0.193 recovered), accuracy | Δ only: KL, accuracy | active per statement | \|A\|/\|y\| per layer | inactive I, Δ D vs \|y\| |
|---|---|---|---|---|---|
| 4a `s-4cd6abe5` | 0.011 (94%), 0.980 | 0.091, 0.933 | 2.2 | up to 12.9 (L11) | 3–14×, cancelling |
| 4b `s-717b50c9` | 0.023 (88%), 0.968 | 0.109, 0.909 | 10.8 | 0.81–0.96 | no cancellation |
| 4d, coeff 1 | 0.147 (24%), 0.884; CI-weighted 0.187 (3%), 0.861 | **0.027 (86%), 0.960** | 2.3 | 0.04–0.17, cos(A, y) ≈ 0 | 3.1–11.2×, cancelling (cos(I, D) −0.93 to −0.99) |

(4a/4b numbers re-read from `~/spd_out/26-10-10_bolder_experiments/exp4/full/analysis/exp4_s-*.json`. Original model accuracy 0.978, all off 0.855.)

- **Final training losses:** stochastic verdict loss 0.0087 (4a 0.0082, 4b 0.0017), orthogonality penalty 0.0009, L0 2.0 (4a 2.0, 4b 10.4), PGD 2.01 (4a 1.20, 4b 0.13) [verified: eval lines in the logs].
- **Orthogonality:** CI-weighted mean ⟨A, y − A⟩/|y|² −0.000 to −0.005 per layer (RMS ≤ 0.045). For the rounded A: mean −0.015 to −0.156, RMS up to 0.80 (L11), so a few positions violate the condition strongly once the CIs are rounded. 17–30% of the active CIs lie strictly inside (0.01, 0.99).

**Reading** [concluded unless tagged]:
1. The penalty removed 4a's re-implementation: no multi-|y| component write. But it did so by taking the components out of the verdict. They write 4–17% of |y|, almost orthogonal to y, and recover a quarter of the all-off KL (rounded) or almost nothing (CI-weighted). **Δ alone carries the verdict** (KL 0.027). So the run lands in the trivial corner A ≈ 0 that the brainstorm had flagged (section 2c).
2. The training objective does not register this: its stochastic verdict loss equals 4a's. With randomly masked Δ (and each loss sample routing only a random subset of layers), Δ can carry the verdict without the components. So nothing in tPD's loss at this position forces the verdict computation into components; in 4a the components "did" it only by re-implementing [concluded: from the equal loss values; the mechanism is not tested].
3. The 4a-style cancellation moved from A into the inactive components and Δ (each up to 11× |y|, cancelling). The orthogonality term constrains only A, so that is expected. These large opposite writes are invisible to the verdict (Δ alone, 11× |y| at L11, changes the verdict KL by only 0.027), which confirms the large verdict-invisible subspace at `'.\n'` (brainstorm, section 1) [verified: this run].
4. The constraint is not binding at coefficient 1 in the final state (penalty 0.0009 against a verdict loss of 0.0087). A higher coefficient can only push further towards A = 0, so **I stopped the coefficient-10 run** after ~4 minutes (no checkpoint; its partial log `ortho10.log` is kept). The informative next step would have been the other direction (0.1, 0.01: middle ground, or does re-implementation return?). Not launched, because Julian is shutting down the VM.

C: (transcribed from chat) Can you interupt run 2. I have to go now and want to shut down the vm

**Open, for Julian** [assumed: proposals]:
- Lower coefficients (0.1, 0.01). Cheap and the natural next test. My expectation: either 4a's re-implementation returns, or a few components write a part of y with low verdict recovery as here. Probably not the hoped-for middle.
- The deeper problem from point 2: at the statement end the verdict depends only weakly on MLPs 11–17 (all-off KL 0.193, accuracy 0.855). That leaves little loss signal to pull verdict computation into components, and Δ takes it for free. Possible fixes: decompose more positions (4c's setting), make Δ's off-state stricter on target data, or use a loss that rewards the components-only state directly. Any of these needs discussion first.
- 4b at reconstruction coefficient 0.1 (the comparison offered earlier) is still open.

C: I guess one problem is the focus on inactive components. First, inactive is an abitrarcy number. Second why not have y - C (C all components)?

Answer (26-10-20, Claude Opus 5.5): My reading: the condition should not depend on which components count as active; instead use C = the write of all components at mask 1, i.e. require ⟨C, y − C⟩ = 0. Since components + Δ = W by construction, y − C = Δx, so the condition reads **⟨C, Δx⟩ = 0: Δ does not cancel the components.** Please correct me if you meant something else.

*Arbitrariness* [concluded]. The threshold enters only the analysis: "active" there means rounded CI > 0.01 (`exp4d_analysis.py`), and with 17–30% of the active CIs strictly inside (0.01, 0.99), the rounded and CI-weighted A differ a lot (RMS of ⟨A, y − A⟩/|y|² up to 0.80 vs ≤ 0.045). The penalty itself has no threshold, but it weights by CI, and the CIs are learned too. So the decomposition can satisfy any condition on A by moving the cancelling work into what the CIs call inactive. 4d did exactly that: A became small, and the cancellation moved into I (inactive components) and Δ, each 3–11 |y|. That is the stronger argument for your version.

*How the existing runs would score* (per layer, from the cancellation numbers in the analysis JSONs; |C| = |y − Δx| is the `rel_err_delta` column; ⟨C, Δ⟩/|y|² estimated as (1 − |C|² − |Δ|²)/2 from per-layer mean norms, so order of magnitude only):

| run | ⟨C, Δ⟩/\|y\|² L11 … L17 | \|C\|/\|y\| | penalty |
|---|---|---|---|
| 4a `s-4cd6abe5` | −183, −54, −28, −28, −12, −23, −11 | 3.3–13.4 | ~10²–10⁴ |
| 4b `s-717b50c9` | ≈ 0 in every layer | 0.88–0.96 | ≈ 0 |
| 4d `s-5c284ebb` | −120, −37, −20, −13, −9, −13, −10 | 3.1–10.8 | ~10²–10⁴ |

So the C-condition flags both failure modes, 4a's (at L11, A 12.9 |y| and I 6.6 |y| point the same way and Δ cancels them) and 4d's (I against Δ), and 4b already satisfies it. Unlike the A-condition, its penalty is not small near the failure: these are violations of order 10–100, not the r⁴-suppressed ~10⁻³ from section 2 of the entry *4d: orthogonality penalty implemented*.

*What it does not do* [concluded: algebra]:
- On its own it does not constrain how C splits among components. A = 12.9 |y| with I ≈ −A satisfies it. **I'd keep both terms**: ⟨A_ci, y − A_ci⟩ = 0 (yours, CI-weighted) plus ⟨C, Δx⟩ = 0. Together they give |A_ci| ≤ |y|, |C| ≤ |y|, |Δx| ≤ |y|, and so |I_ci| = |C − A_ci| ≤ 2|y|: no term can be large, so no large cancellation anywhere. (The rounded A is still not bounded by this.)
- It does not force the verdict into the components either: C = 0, Δx = y satisfies it, and point 2 of the result entry above (Δ carries the verdict for free) still applies. What changes is that Δ can no longer hide a multiple of |y| in the verdict-invisible subspace, so whatever Δ carries is a genuine part of y and gets randomly scaled by Δ's stochastic mask in the verdict loss. Whether that pushes the verdict into the components is a prediction, not a result [assumed].

*Practical points* [assumed]:
- Cost: C = (W − Δ)x is one extra matmul per module, which is cheap.
- The 4d trial's I and Δ were already 2–4.6 |y| and cancelling at step 300, so the C-term starts large and dominates early training at coefficient 1. It will first shrink all the components' writes at this position (fine for tPD: inactive components should write nothing on target data). A warm-up of the coefficient may be needed; the trial would show.
  OUTDATED (26-10-21): the prediction that the C-term starts large is wrong: in the 4e end-token trial it started at 0.022 (|C|/|y| 0.34–0.42 at initialisation), so no warm-up was needed. See the entry *4e: analysis check and end-token trial*.
- Implementation: an option in `ComponentOrthogonalityLossConfig` that selects the state (CI-weighted A, or all components C), so both terms can run in one config. About the same work as the eval additions; tests as for the existing term.

Proposed run [assumed: not discussed]: 4a + both terms at coefficient 1 ("4e"), trial first, then 5,000 steps, analysed with `exp4d_analysis.py` (which already reports |I|, |Δ| and cos(I, Δ)). One seed. Needs a GPU, since the H100 was shut down.

C: So, let's start this orthogonal experiment new. First let's delete the old run from wandb because I have to save space. I only want the global orthogonality term, not the active orthogonality one. Let's see what happens. Also right now, the goal is to find any result, comparability between the other results is less important. Meaning, please use finetuning of all hyperparameters to get good numbers (whatever that means). Then we can further analyse any promising decomposition to see if it has uncovered any mechanism. First ored of business are questions from your side. Ask a lot.

---

26-10-20, Claude Opus 5.5 — *4e restart: questions before tuning*; I re-read the end of this LOG (Julian's comment above), the 4d config, the WandB storage SUMMARY, and listed the WandB runs of 26-10-20 (read-only API).

Julian's direction (comment above) [decided]: restart the orthogonality experiment with **only the global term ⟨C, y − C⟩ = ⟨C, Δx⟩ = 0** (C = all components at mask 1), without the CI-weighted active term.  
C: Wait, I'm confused. Why are all componest at mask 1? I don't understand that.

Answer (26-10-20, Claude Opus 5.5): "Mask 1" was jargon; it means only that every component counts fully, whatever its CI. C is the plain sum of all components' writes, C = Σ_c U_c (V_c · x) over all 96 components of the layer. No CI and no mask enter. Since components + Δ = W by construction, y − C = Δx. Why unweighted: weighting by CI would give back the 4d term. The inactive components have CI ≈ 0, so the CI-weighted sum over all components is the CI-weighted A, i.e. the active term you wanted to drop. Only the unweighted sum is independent of which components are active, which is how I read your point that "inactive is an arbitrary number". If you meant a different C, the term changes, so please say.

 The goal is now any result: hyperparameters may be tuned freely, comparability with 4a/4b/4c is secondary, and promising decompositions get analysed for mechanisms afterwards. Before deleting or implementing anything, my questions follow. Each one has my default in brackets, so a short "ok" is enough where you agree.

**Deleting the old run**
1. WandB currently holds four runs from 26-10-20. Only one has a checkpoint: `s-5c284ebb` (4d, coefficient 1), `model_5000.pth`, 333.5 MB. **No local copy exists**; locally there are only its analysis outputs (`~/spd_out/26-10-16_faithful_targeting/full/analysis/exp4_s-5c284ebb.{json,npz}`), so the deletion can't be undone. The other three hold no checkpoint, so deleting them frees nothing: `s-b8c8c610` (failed start, the missing short name), `s-37397e7d` (300-step trial), `s-15cc0844` (coefficient 10, stopped). [Default: delete only the `.pth` file of `s-5c284ebb` and keep the run with its logged metrics, which take almost no space; leave the other three alone.] Or should I delete whole runs?
C: please delete the runs that are not a complete experiment. Also delet the checkpoint of the successful run today.

**What "good numbers" means** (this decides the tuning, so it matters most)
2. Which number is the tuning objective? Candidates: (a) verdict recovery of the components-only state (rounded CIs, Δ off): share of the all-off KL 0.193 recovered; (b) the same for CI-weighted components; (c) few active components per statement (L0); (d) the components' share of the direct truth write vs Δ's. [Default: maximise (a), subject to the constraint in question 3; break near-ties by lower L0.]  
C: what does roundedn CIs mean? I think (a) first. How does CI-weighted components work? Just set mask to CI?

Answer (26-10-20, Claude Opus 5.5): Both are states of the model in which each decomposed layer at the decomposed positions is replaced by a masked sum of the components, with Δ off (mask 0); everything else runs on the original weights.
- **Rounded CIs**: mask 1 for every component with CI > 0.01, mask 0 otherwise. This binarises at threshold 0.01, not at 0.5. It is the "components only" state of all Exp 4 analyses and the same rule as the eval metric `TargetReconLoss` (`rounding_threshold: 0.01`, `spd/metrics/targeted_recon_loss.py`).
- **CI-weighted**: yes, mask = CI exactly (continuous in [0, 1]).

They differ where CIs lie in between: in 4d, 17–30% of the active CIs were strictly inside (0.01, 0.99), and the CI-weighted state recovered 3% of the all-off KL vs 24% rounded. (a), the rounded state, is the objective.
OUTDATED (26-10-21): Julian changed the objective to the CI-weighted state (his comment under the draft plan in the entry *4e: follow-up questions and draft plan*: "I want the CI-weighted not rounded one").

3. What counts as "faithful enough" to pass? [Default: per layer |C|/|y| ≤ 1.2, |Δx|/|y| ≤ 1.2, and no large cancellation among the components either: |A|/|y| and |I|/|y| ≤ 1.5, cos(A, I) not strongly negative. The global term alone allows A ≈ −I (the answer to the y − C comment above), so I would watch for it explicitly but not penalise it.] Should a run that recovers the verdict through A ≈ −I count as a failure, like 4a?
C: just ask me when the numbers are in
4. When do I stop tuning? [Default: when a configuration passes question 3 and recovers ≥ 50% of the all-off KL, or when the budget in question 8 is used up. Then report the whole front, not just the winner.]
C: after each experiment you report the numbers (you can already start the follow up experiments though they can always be stopped if we change course)

**What may change**
5. Which of these is fixed and which is tunable? Model (Qwen2.5-7B-Instruct), data (tiu), prompt ("<statement>\nThe amount of truth in this statement is "), target loss (KL of the verdict digit), decomposed layers (`down_proj` 11–17), decomposed positions ('.\n' only). [Default: model, data, prompt and lens fixed; layers and positions tunable.]
C: None of these are really tunable. As you said model, data, prompt and lens is fixed. Layers have been identified in previous experiment, and only should be changed if new evidence surfaces. Positions should be checked for end and whole stimulus.
6. Positions in particular: at '.\n' alone the verdict depends only weakly on MLPs 11–17 (all-off KL 0.193, accuracy 0.855 vs 0.978). The 4a config header records that mean-ablating 11–17 at '.\n' alone keeps 82% of the verdict separation [assumed: config comment]. I suspect that is the main reason Δ could carry the verdict for free in 4d. 4c's setting (all statement tokens) gives the loss much more signal. [Default: try all statement tokens early, as one of the first two configurations, not as a late variant.] Ok?
C: I want to try both
7. Are structural changes to the loss in scope, or only config values? Candidates in the order I'd try them: (i) a warm-up for the orthogonality coefficient (the C-term starts at an estimated 10²–10⁴ in 4a/4d-like states); (ii) a training term on the components-only state (Δ off), which addresses point 2 of the 4d result entry directly; (iii) mean instead of zero as Δ's off value. [Default: (i) yes; (ii) and (iii) only after asking you, with a short reason.]
C: no, I don't set my hope in engineering the loss will surface new behaviour
OUTDATED (26-10-21): the estimate in (i) that the C-term starts at 10²–10⁴ is wrong (it started at 0.022); see the entry *4e: analysis check and end-token trial*.

**Budget and machines**
8. GPU: will you rent an H100 again (or should I, via `spd-vast`)? How many GPU hours or dollars may the tuning use? For scale: one 4d run (5,000 steps, '.\n' only) took 34:53 on the last H100, which was shared with an outside 26–31 GB process (entries *4d: orthogonality penalty implemented* and *4d at coefficient 1*); all statement tokens costs more [assumed: not measured for this prompt]. [Default: an H100, about 10–15 GPU hours, i.e. roughly 15–20 full runs at '.\n', fewer at all tokens.]
C: I think we should just make a plan and then as we execute it adapt it. So budget is not an issue for now. Yes, h100 is ready
9. Short runs as a proxy: may I screen with 1,000–2,000 steps and run only the best few to 5,000? Risk: the importance-minimality schedule (warm-up 20%, anneal to the end) and PGD (starts at 80%) are tied to the fraction of training, so a short run is a compressed run, not an early stop. [Default: yes, screen at 1,500 steps, confirm at 5,000.]  
C: Don't know actually. Maybe some wrong parameters can be quickly excluded with pure 3000 steps without PGD
10. Several runs in parallel on one H100? 7B in bf16 plus a run fitted next to the outside process last time. [Default: try two at once, measure the throughput, keep it only if the total is faster.]
C: Yes, check that.

**Method**
11. Search strategy. [Default: by hand, one or two hyperparameters at a time, in this order: orthogonality coefficient (with warm-up) → positions → importance-minimality coefficient → C per layer → LR / steps → PGD coefficient. Not a WandB sweep: the objective needs the cancellation analysis after each run, which the sweep can't compute.]  
C: by hand
12. Tuning split: so far every number came from the tiu test split (1,794 statements). Tuning many configurations on it would overfit the hyperparameters to the test statements. [Default: hold out a validation subset of the train split for tuning; touch the test split only for the final configurations.] Ok?
C: OK
13. Seeds: one per configuration during the search, then a second seed for the winner and the runner-up? [Default: yes. The run-to-run spread of Exp 4 is unknown, so without a second seed a winner may be noise.]
C: discuss this later

**Storage and code**
14. Checkpoints of tuning runs: per `docs/CODING.md`, trial and tuning runs don't sync to WandB. [Default: tuning runs keep checkpoints on the instance only, the analysis runs there before the instance is destroyed, and only the final 1–2 decompositions sync to WandB (~334 MB each at 11–17, more if the CI function's input grows) and get copied to the laptop.] Is there room on the laptop for that?
C: they don't have to get copied but might be copied in a later analysis. But not now
15. The analysis script `exp4d_analysis.py` lives outside the repo (`~/spd_out/26-10-16_faithful_targeting/`). It will become the tuning objective. [Default: move it into the repo next to the other honesty analysis scripts, add ⟨C, Δx⟩, |C|/|y| and cos(A, I), and test it on CPU against the stored 4d numbers.]
C: yes, should live in repo
16. Code for the global term: [Default: one option in `ComponentOrthogonalityLossConfig` that selects the state (`ci_weighted` = the 4d term, `all_components` = new), not a second class; the 4d config stays as it is for reference; a new config `config_truth_sf_verdict_ortho_global.yaml` ("4e").] Ok?
C: don't care

**Interpretation**
17. "Find any result": what would make a decomposition worth the mechanism analysis? My guess: faithful (question 3), most of the verdict in a small set of components, and at least some components that are not just task/family detectors (the 4b/4c complaint). If you have a specific mechanism in mind (e.g. negation handling, which 4c's L17:5 hinted at in the steering topic), it could guide which positions and layers to prefer.
C: this is an open ended experiment
18. A caveat I'd rather raise now: with free tuning, a "good number" can come from a configuration that has found a new loophole, as 4a did. I would run the cancellation test on every candidate and report loopholes as findings, not tune past them silently. Agreed?
C: yes

---

26-10-20, Claude Opus 5.5 — *4e: follow-up questions and draft plan*; I read Julian's answers in the entry *4e restart: questions before tuning* and `docs/CODING.md`, answered the two inline questions in place (what "mask 1" means; rounded vs CI-weighted), and checked the H100.

**Status**
- WandB deletion: approved by Julian (C: comment under question 1: delete the three incomplete runs `s-b8c8c610`, `s-37397e7d`, `s-15cc0844`, and the checkpoint `model_5000.pth` of `s-5c284ebb`, 333.5 MB). **Not done**: the Claude Code auto-mode permission check blocked the API call. Julian either deletes them on the website or allows the command.
- H100 reachable (`ssh vastai`): power limit 700 W, 0 MiB in use (no outside process this time), 92 GB free on `/root`, `/root/spd` present [verified: `nvidia-smi`, `df`]. I have not checked yet whether the repo there is current.
- Consequence of the deletion for testing: the moved analysis script can't be checked against 4d's stored numbers once its checkpoint is gone. I will check it against 4a `s-4cd6abe5` instead, whose checkpoint and stored numbers both exist.

**Follow-up questions** (my default in brackets)

F1. Positions, "end and whole stimulus": in the config vocabulary the *stimulus* is the appended question "\nThe amount of truth in this statement is " (`target_lens.stimulus`, decomposed in Exp 3). Do you mean (a) all statement tokens (4c's setting), (b) the question tokens, or (c) both? [Default: (a), because that is what question 6 was about. Under (b) the decomposed MLPs would read the statement only through attention at the question tokens.]
C: stimulus is the statement. Old terminology was wrong. So it's (a)

F2. What is left to tune, given that layers, positions (apart from the two settings in F1) and the loss form are fixed: the orthogonality coefficient; importance-minimality coefficient and p-norm; C per layer (96 now); LR and steps; stochastic-loss routing (random layer subset as now, or all layers); PGD on/off and its coefficient; CI-function width (512). Anything to add or exclude? [Default: tune in this order: orthogonality coefficient → importance-minimality coefficient → C → LR → routing → PGD.]
C: Yes these have to be tuned. However the order should be different. We have never fine tuned the other parameters.

F3. A warm-up of the orthogonality coefficient (linear ramp over the first x% of steps, as `ImportanceMinimalityLoss` already has with `coeff_warmup_frac`). It needs a little code but adds no new loss. Does that count as the loss engineering you excluded? Why I'd want it: at initialisation the term is likely 10²–10⁴ per position (4d's trial already had I and Δ at 2–4.6 |y| at step 300), so at a fixed coefficient it dominates the start. [Default: allowed, as a config option; tried only if the plain coefficients fail.]
C: ok, include it, if you have a good reason for it.
OUTDATED (26-10-21): the premise of F3 (term 10²–10⁴ at initialisation) is wrong: it started at 0.022, so no warm-up was used; see the entry *4e: analysis check and end-token trial*.

F4. Validation split: 6,560 train / 1,794 test statements over the 20 tiu datasets. [Default: hold out 15% of each dataset's train split (seeded), about 980 statements, as `val`. Training uses the remaining ~5,580, test stays untouched until the final configurations. Implemented as a new dataset version `v2` (train/val/test) through the tiu generator, so provenance stays recorded.] Ok, or would you rather tune on part of the test split?
C: I don't care. You can choose

F5. Screening runs: 3,000 steps with PGD off (your suggestion), objective (a), analysed on the validation split. Full runs: 5,000 steps with PGD as in 4a (coefficient 0.5 from 80% of training). [Default: yes. Turning PGD off changes the training, not the objective, so a screening run ranks configurations but does not predict the full run's numbers [assumed].]
C: yes

**Draft plan** [assumed: proposal; adapted as results come in]
1. Code (local, CPU tests): global-term option in `ComponentOrthogonalityLossConfig`; the analysis script moved into the repo with ⟨C, Δx⟩/|y|², |C|/|y| and cos(A, I) added, plus a `--split val` option; checked against 4a's stored numbers; validation split (F4).
2. On the H100: one short trial (300 steps) per position setting, to see the start-up scale of the term and the step time. Then check whether two runs in parallel give more total throughput (question 10).
3. Screen round 1: orthogonality coefficient ∈ {0.01, 0.1, 1, 10} × positions {end token, all statement tokens} = 8 runs at 3,000 steps without PGD. Report after each run: verdict recovery (a) and (b), L0, |A|, |I|, |C|, |Δx| relative to |y|, cos(I, Δ), cos(A, I), ⟨C, Δx⟩/|y|², all per layer.
   OUTDATED (26-10-21): round 1 was run over LR × positions instead (orthogonality coefficient fixed at 1), because Julian (F2) rejected tuning the orthogonality coefficient first; see the entry *4e implementation, part 2: analysis script, configs, trials launched*, *Proposed tuning order*. The orthogonality coefficient was never tuned.
4. Following rounds by hand along F2's order, from the best one or two configurations of the round before.
5. The best one or two configurations as full runs (5,000 steps, PGD, checkpoint synced to WandB), evaluated on the test split. Then seeds and mechanism analysis, to be discussed then.

Every run's numbers go into a LOG entry here, with a short chat report.
C: yes
C: change of objective. I want the CI-weighted not rounded one

---

26-10-20, Claude Opus 5.5 — *4e implementation, part 1: global term, tiu v2 with val split*; continuing from Julian's answers to F1–F5 above.

Decisions taken from the answers [decided: C: comments above]: positions = statement end and all statement tokens (F1; "stimulus" in Julian's sense is the statement, while the config field `target_lens.stimulus` still names the appended question; not renamed); objective = verdict recovery of the **CI-weighted** components-only state (last comment, replacing the rounded state); the coefficient warm-up only with a concrete reason (F3); screening at 3,000 steps without PGD (F5). Julian rejected my tuning order because the other hyperparameters were never tuned either; he gave no new order, so I will propose one with the first trials.

Done:
- **Global term**: `ComponentOrthogonalityLossConfig.state` (required): `ci_weighted` (the 4d term; the 4d config now sets it explicitly) or `all_components` (A = Σ_c U_c (V_c x) without CI, so y − A = Δx). Wired into the training loss and the eval metric. Tests [verified: `tests/test_orthogonality_loss.py`, 14 passed]: the all-components loss equals the hand-computed mean of (⟨Cx, Δx⟩/|y|²)² with y − Cx = Δx checked numerically, it is independent of the CIs and differs from the CI-weighted loss, the scaled-reconstruction cases t(1 − t) hold for both states, and gradients reach U and V but not the CIs. Mutation check: making the all-components branch weight by CI fails 2 tests.
- **Validation split**: `build_tiu.py` got `val_frac` (the subject's hash decides test / val / train, so test stays identical and train + val = the old train). Julian asked me to commit the generator before generating data, so that the manifests' `repo_commit` contains it: committed as `b1192cb` (generator only). I had already built v2 once from the uncommitted script; I deleted that build and rebuilt. `data/tiu/<dataset>/v2` (`--val_frac 0.12`): train 5,574, val 986, test 1,794 statements [verified: test byte-identical to v1 in all 20 datasets, train + val = v1 train, subjects disjoint across the three splits, every manifest records `repo_commit` b1192cb and the generator's sha256]. Small families get small val sets (animal_class 7–8 statements per dataset).
- Loader: the `Split` type accepts "val", and prepared-dataset runs evaluate on `task_config.eval_data_split` ("test" by default as before, "val" for tuning). Not committed.

Next: the analysis script into the repo (with `--split`, ⟨C, Δx⟩/|y|², |C|/|y|, cos(A, I), CI-weighted parts and the share of all-off KL recovered), checked against 4a's stored numbers on the H100; the two 4e configs; trials.

Commits (26-10-20, at Julian's request in chat: "can you write commits and then go to the next step"): `6cdf196` orthogonality penalty (4d and 4e states, tests, 4d config), `ffd6be0` val split support + tiu v2 data + `data/README.md`, `143e35b` vast blacklist host 638511, then the records. Before committing: `make check` clean except the 6 pre-existing basedpyright errors in `rank_rotgrid_decompositions.py`; `make test` 144 passed, 1 failed (the pre-existing `test_the_checked_in_sweep_expands_to_distinct_runs`) [verified].

---

26-10-20, Claude Opus 5.5 — *4e implementation, part 2: analysis script, configs, trials launched*; same session.

- **Analysis script in the repo**: `spd/experiments/lm/honesty_targeted_decomposition/verdict_lens_analysis.py` (successor of `exp4d_analysis.py`, which lived in `~/spd_out/`). `--split val|test` (required), reads data, lens and positions from the run's config, output `verdict_<run id>_<split>.{json,npz}`. Added to the earlier numbers: `kl_recovered` = 1 − KL(state)/KL(all off) per verdict state (the tuning objective is that of `components_ci`); per layer |C|/|y|, ⟨C, Δx⟩/|y|² (mean and RMS; the 4e penalty's inner product), cos(C, y), cos(A, I); the CI-weighted split A_ci, I_ci = C − A_ci with norms, cos(A_ci, I_ci), |y − A_ci|/|y|. basedpyright and ruff clean. Not yet verified: it runs on the H100 now against 4a (`s-4cd6abe5`, test split), to be compared with the stored `exp4_s-4cd6abe5.json`.
- **Configs** `config_truth_sf_verdict_ortho_global.yaml` (statement end) and `config_truth_sf_verdict_ortho_global_alltok.yaml` (all statement tokens, as 4c): 4a's settings plus the global term at coefficient 1 (untuned starting point), tiu v2, `eval_data_split: val`, checkpoints not synced. Loss metrics are evaluated automatically, so the term's per-layer numbers appear in the training evals without an extra eval entry.
- **Trials** (300 steps, PGD off as in screening; `~/spd_out/26-10-16_faithful_targeting/e4/`, on the instance `/root/e4/`): end token, then all statement tokens, each followed by the analysis on val. Launched 17:32 UTC. The new configs and the analysis script are not committed yet, so the runs are stamped "uncommitted changes" (code = working tree synced at `094db84` + these files). I will commit them once the 4a check passes.

**Proposed tuning order** [assumed: proposal] (Julian, F2: not orthogonality first, because the other hyperparameters were never tuned either):
1. LR ∈ {1e-4, 5e-4 (now), 2e-3} × positions {end, all statement tokens}, orthogonality coefficient 1: the LR decides whether any comparison afterwards means anything.
2. Importance-minimality coefficient ∈ {1e-4, 1e-3 (now), 1e-2}: it sets the trade-off between few active components and verdict recovery, which is directly the objective.
3. Orthogonality coefficient ∈ {0.1, 1, 10}, plus a warm-up if the trials show the term dominating the start.
4. C per layer ∈ {48, 96, 192}; then stochastic-loss routing (random layer subset vs all layers) and CI-function width (512 vs 1024).
5. Full runs (5,000 steps, PGD; PGD coefficient as in 4a, 0.5, varied only if the PGD loss stays high).
One parameter at a time from the best configuration so far; interactions are missed by design. Each round is about 3–6 screening runs.

---

26-10-20, Claude Opus 5.5 — *4e: analysis check and end-token trial*; same session. Julian is away for 1–2 hours and asked me to keep the H100 busy with experiments.

**Analysis script check** [verified: `~/spd_out/26-10-16_faithful_targeting/e4/analysis/check4a.log` and `verdict_s-4cd6abe5_test.json`, H100, 26-10-20]: `verdict_lens_analysis.py` on 4a (`s-4cd6abe5`, test split) reproduces every verdict-state number of the stored `exp4_s-4cd6abe5.json` exactly, and the cancellation numbers within 0.2%. Committed with the two 4e configs as `40bfe88`. The exact per-position mean ⟨C, Δx⟩/|y|² of 4a is −194 at L11, −64 to −12 at L12–17 (my estimate from mean norms in the answer to the y − C comment: −183 to −11).

**End-token trial** (`s-16271b38`, 300 steps, PGD off, coefficient 1, LR 5e-4; val split, 986 statements) [verified: one run]:
- Start-up: the term starts small, not at 10²–10⁴ as I had predicted (F3): penalty 0.022 at step 0, |C|/|y| 0.34–0.42 at initialisation. The large I/Δ writes in 4d's trial arose during training, without this term. **No warm-up needed**; F3 is moot.
- Training: penalty 0.022 → 0.003 at step 300; stochastic verdict loss 0.022 → 0.013; L0 2.5 → 1.5.
- Analysis: no cancellation (|Δx| 0.99–1.01 |y|, |C| 0.26–0.33 |y|, ⟨C, Δx⟩/|y|² −0.04 to −0.05 per layer, cos(I, Δ) −0.11 to −0.16), but the components are almost all inactive (1.6 active per statement) and write nothing (|A| ≤ 0.02 |y|). Verdict: components only (CI-weighted) KL 0.175, i.e. **14% of the all-off KL 0.204 recovered** (rounded 21%); Δ only 84%. Original accuracy on val 0.966 (test 0.978).
- Reading [assumed: 300 steps is far from converged]: the global term does what it should against cancellation, and at this point the trial sits in the trivial corner C ≈ small, Δ ≈ y. Whether training pulls the verdict into components later is what the 3,000-step round shows.

Plan while Julian is away: the all-token trial runs now; afterwards round 1 (LR ∈ {1e-4, 5e-4, 2e-3} × {end, all statement tokens}, 3,000 steps, PGD off, two runs at a time) starts automatically. Scripts: `make_round.py`, `run_round.sh` in `~/spd_out/26-10-16_faithful_targeting/e4/` (instance `/root/e4/rounds/<round>/`).

**All-token trial** (`s-294c4acf`, same settings, all statement tokens decomposed; val) [verified: one run]: 2.53 it/s (end token 2.61). All off now takes the verdict to chance (accuracy 0.511, KL 0.669; at the end token alone 0.838, KL 0.204), so the loss has much more signal here. Components only (CI-weighted) recover 24% of the all-off KL (rounded 25%), **Δ only 39%** (accuracy 0.777): unlike the end-token trial, Δ does not carry the verdict alone. No cancellation (|Δx| 1.02–1.03 |y|, |C| 0.21–0.27 |y|, ⟨C, Δx⟩/|y|² −0.06 to −0.10); 2.3 active components at the end token, 0.7 per middle token, none at the first token. 300 steps only.

Round 1 (LR) started 17:46 UTC, two runs in parallel.
Parallel throughput (question 10) [verified: step 300 of round 1's first pair]: two runs at once 1.49 + 1.48 = 2.97 it/s in total, vs 2.53–2.61 it/s for one run alone: about 15% more, at 64.5 GB and 100% GPU utilisation. Kept at two. A 3,000-step screening run then takes ~34 min plus evals; round 1 (three pairs) should end around 19:30 UTC.

---

26-10-20, Claude Opus 5.5 — *4e round 1 (LR): first pair*; same session. Numbers from `verdict_lens_analysis.py` on val (986 statements), summarised with `~/spd_out/26-10-16_faithful_targeting/e4/summarize.py`; JSONs in `e4/analysis/r1_lr/` [verified: one seed each]. Objective = KL recovered by the CI-weighted components (1 = original verdict distribution, 0 = all decomposed outputs off).

| run | positions, LR | KL recovered: CI-weighted / rounded / Δ only | accuracy (CI-weighted) | active per statement | max over layers: \|I\|, \|Δx\|, \|C\| (/\|y\|) | min ⟨C, Δx⟩/\|y\|² | max \|A\|/\|y\| |
|---|---|---|---|---|---|---|---|
| trial `s-16271b38` | end, 5e-4, 300 steps | 0.14 / 0.21 / 0.84 | 0.848 | 1.6 | 0.33, 1.01, 0.33 | −0.05 | 0.02 |
| trial `s-294c4acf` | all, 5e-4, 300 steps | 0.24 / 0.25 / 0.39 | 0.739 | 54.7 | 0.26, 1.03, 0.27 | −0.10 | 0.03 |
| `s-c706fefb` | end, 1e-4 | 0.25 / 0.35 / 0.88 | 0.863 | 1.9 | 0.35, 0.97, 0.35 | −0.02 | 0.03 |
| `s-114f2c0e` | all, 1e-4 | 0.29 / 0.48 / 0.42 | 0.674 | 60.5 | 0.19, 1.01, 0.20 | −0.03 | 0.03 |

(All off: end token accuracy 0.838, KL 0.204; all tokens accuracy 0.511, KL 0.669. Original on val 0.966.)

Reading so far [concluded unless tagged]: the global term prevents cancellation in every run (Δ ≈ |y|, all components together ≤ 0.35 |y|). The components write very little (|A| ≤ 0.03 |y|) and still recover a quarter to half of the verdict's KL, so whatever they write lies in directions the verdict reads [assumed: interpretation, not tested]. At all tokens the rounded state (48%) is well ahead of the CI-weighted one (29%): many active CIs are below 1, so the objective penalises half-on components. The accuracy of the CI-weighted state at all tokens (0.674) is lower than at the end token although its KL recovery is higher, because the all-token all-off baseline is at chance.

C: Can you explain this some more: "At all tokens the rounded state (48%) is well ahead of the CI-weighted one (29%): many active CIs are below 1, so the objective penalises half-on components"?
What are the percentages and where do you get the rounded state from? I only saw CI-weighted runs above.

Answer (26-10-20, Claude Opus 5.5): "CI-weighted" and "rounded" are not two kinds of runs. They are two ways the analysis evaluates **the same trained decomposition** afterwards. Every run is trained the same way: the CI-weighted active term no longer exists, only the global term ⟨C, Δx⟩ = 0 is trained. After training, `verdict_lens_analysis.py` takes the run's checkpoint, computes the CIs for each validation statement, and runs the model in several *states*. In each state the decomposed MLP outputs (at the decomposed positions) are replaced by a masked sum of the components, with Δ off:
- **CI-weighted** (`components_ci`): component c gets mask = its CI (a component with CI 0.3 contributes 30% of its write). This is the objective you chose.
- **Rounded** (`components`): mask = 1 if CI > 0.01, else 0, so every active component contributes its full write. This is the "components only" state of all earlier Exp 4 analyses, and the same rule as the training eval `TargetReconLoss`.
- For reference, **Δ only** (components off, Δ on) and **all off** (decomposed outputs 0).

The percentages are "KL recovered" = 1 − KL(state)/KL(all off), with KL = KL(original ∥ state) of the next-token distribution at the answer, averaged over statements. 100% = the state gives the original verdict distribution, 0% = no better than switching the decomposed MLP outputs off. For the all-token run at LR 1e-4 (`s-114f2c0e`): all off KL 0.669; CI-weighted KL 0.475 → 1 − 0.475/0.669 = 29%; rounded KL 0.348 → 48%; Δ only KL 0.390 → 42%. Verdict accuracy in the same states: 0.674, 0.818, 0.766 (all off 0.511, original 0.966).

Why they differ: the two states differ only where an active component's CI lies strictly between 0.01 and 1. In this run that is 13–18% of the active CIs per layer, but those components carry a large part of the write: the CI-weighted write |A_ci| is about half the rounded one |A| in every layer (e.g. L17 0.017 vs 0.029 |y|). So the CI-weighted state applies roughly half of what the active components write, and recovers less of the verdict. My sentence above ("many active CIs are below 1") overstated the count: it is a minority of the CIs, but a large share of the write. "Penalises half-on components" meant: under the CI-weighted objective, a component that is only half on also counts only half, whereas the rounded state counts it fully.

Relevance for tuning: training never uses exactly the CI-weighted masks (the stochastic masks are CI + (1 − CI)·uniform noise), so nothing pushes the CIs of useful components to 1. If the CI-weighted state stays well behind the rounded one, the importance-minimality settings (they shape the CIs) are where it would change [assumed].

---

26-10-20, Claude Opus 5.5 — *4e round 1 (LR): second pair (5e-4)*; same session [verified: one seed each, val].

| run | positions, LR | KL recovered: CI-weighted / rounded / Δ only | accuracy CI-weighted / rounded / Δ only | active per statement | max \|I\|, \|Δx\|, \|C\| (/\|y\|) | min ⟨C, Δx⟩/\|y\|² |
|---|---|---|---|---|---|---|
| `s-e6688f03` | end, 5e-4 | 0.08 / 0.08 / 0.79 | 0.844 / 0.841 / 0.947 | 0.7 | 0.60, 0.86, 0.60 | −0.01 |
| `s-348832ad` | all, 5e-4 | **0.37 / 0.38 / 0.12** | 0.756 / 0.778 / 0.684 | 19.7 | 0.36, 0.98, 0.36 | −0.04 |

Reading [concluded unless tagged]:
- **All tokens at LR 5e-4 is the best run so far**: the CI-weighted components recover 37% of the all-off KL and, for the first time in Exp 4d/4e, **more than Δ alone (12%)**. So the verdict-relevant part of the MLP outputs is carried by the components rather than by Δ, and without cancellation (|Δx| ≤ 0.98 |y|, |C| ≤ 0.36 |y|). The CI-weighted and rounded states now agree (5–9% of the active CIs in between), unlike at LR 1e-4. About 1 active component per layer at the statement end, 0.2 per middle token, none at the first token.
- At the end token the higher LR made things worse (8% vs 25% at 1e-4): the components are nearly all off (0.7 active per statement) and Δ carries the verdict, as in 4d.
- Caveat: one seed each, and 37% is still far from 4b's 88% (4b reconstructed y; rounded state; test split, so not directly comparable).

The LR 2e-3 pair runs now (end ~19:45 UTC).

---

26-10-20, Claude Opus 5.5 — *ablations of the best 4e run: plan*; same session. Julian asked whether there is room for ablations of `s-348832ad` (all tokens, LR 5e-4). GPU memory was 65.3 of 81.6 GB (the LR 2e-3 pair), too little for a third 7B job; the checkpoint (only on the instance, tuning runs don't sync) is now also copied to `~/spd_out/26-10-16_faithful_targeting/e4/checkpoints/s-348832ad/`. Options offered: (1) wait until the LR 2e-3 pair ends (~19:45 UTC), then ablations next to one round-2 run; (2) stop the end-token 2e-3 run now.

C: (transcribed from chat) Yes, option 1 and your measurement A1, A2 and A3

Measurements (val split, 986 statements; replacement at all statement tokens as in training):
- A1: single-component removal from the full model (mask 0 for one component at all decomposed positions, everything else original): change of verdict separation, accuracy, KL, by label and family, for every component active (rounded CI) on ≥ 1% of statements; control: size-matched random sets of rarely active components.
- A2: the same removals from the CI-weighted components-only state (Δ off).
- A3: per component, where it is active (CI by label, family, position) and how much it writes along the truth direction.

Implementation: `~/spd_out/26-10-16_faithful_targeting/e4/ablate_4e.py` (adapted from the steering topic's `flip_components.py`; on the instance `/root/e4/`). Details fixed while writing [assumed: my choices]: A1 in two variants, removal at every decomposed position ("ungated", a weight edit W − u_c v_cᵀ) and only where the component is active ("gated"); A2 removal from the CI-weighted state (mask = CI, Δ off), reported as the change of KL recovered; candidates = active (rounded CI) at some statement token in ≥ 1% of the val statements; controls = 30 single removals of rare components (active in > 0 and < 1% of statements) and 10 size-matched random rare sets vs removing all candidates. A3's "truth direction" = each layer's own mass-mean direction of the MLP output at the statement end (mean y over true minus false statements), and the component's share of that gap (CI-weighted write); Δ's share alongside. Chained on the instance to start when round 1 ends; round 2 then runs one job at a time next to it.

---

26-10-20, Claude Opus 5.5 — *4e round 1 complete: LR 2e-3*; same session [verified: one seed each, val, 3,000 steps, PGD off].

| run | positions, LR | KL recovered: CI-weighted / rounded / Δ only | accuracy CI-weighted / Δ only | active per statement | max \|I\|, \|Δx\|, \|C\| (/\|y\|) | min ⟨C, Δx⟩/\|y\|² | min cos(A, I) | max \|A\|, \|A_ci\| (/\|y\|) |
|---|---|---|---|---|---|---|---|---|
| `s-9dc19f9f` | end, 2e-3 | 0.86 / 0.72 / 0.84 | 0.947 / 0.961 | 2.1 | 0.66, 0.88, 0.66 | −0.02 | −0.28 | 0.22, 0.10 |
| **`s-eb1c20d5`** | **all, 2e-3** | **0.91 / 0.87 / −0.34** | **0.974** / 0.556 | 27.5 | 0.46, 0.97, 0.45 | −0.04 | −0.23 | 0.16, 0.08 |

(Original on val: accuracy 0.966. All off: end 0.838 / KL 0.204, all tokens 0.511 / KL 0.669.)

Reading [concluded unless tagged]:
- **`s-eb1c20d5` is the first decomposition in Exp 4 whose components carry the verdict without cancellation and without Δ.** CI-weighted components alone: KL 0.060 (91% of the all-off KL recovered), accuracy 0.974, at or above the original. Δ alone, i.e. the MLP outputs with the components' write removed: KL 0.897, *worse* than switching the MLPs off (0.669), accuracy 0.556. So removing a small part of y (the components write |A_ci| ≤ 0.08 |y|, all components together |C| 0.19–0.45 |y|) destroys the verdict, and that small part alone restores it. The global term holds (⟨C, Δx⟩/|y|² −0.015 to −0.038 per layer, RMS ≤ 0.09), so along C's direction Δ has (almost) nothing: Δ-only is close to a directional ablation of C from y. This is the y = t + r picture: components ≈ the verdict-relevant part t, Δ ≈ the rest r [assumed: interpretation; A3 and the ablations test it].
  OUTDATED (26-10-21): the reading "components ≈ the verdict-relevant part t" was contradicted by the partial A3 (entry *all runs stopped; partial A3 of `s-eb1c20d5`*: the components write 0.1–6.6% of the mass-mean truth gap at the statement end) and replaced by "the components' statement-specific activity supplies a 'false' signal against a 'true' default" (entry *what do the components carry: extended ablations*); at L11 the split is a genuine part of y (entry *A8: the L11 split exists in the original MLP output*). "Δ alone worse than all off" turned out to be a seed-0 calibration detail (entry *round 2: seed 1 of `s-eb1c20d5` replicates*).
- Weak spots: active and inactive components partly oppose at L13 and L17 (cos(A, I) −0.23 and −0.12; |A| 0.16 and 0.12 |y|, |I| 0.42 and 0.47 |y|), so a small A-vs-I cancellation exists, the loophole the global term allows; small in size. The CI-weighted state fails on element_symb_true (accuracy 0.21, 14 statements; original 1.00 there) and neg_element_symb_true (0.57).
- At the end token the components also recover 86%, but Δ alone does too (84%): the verdict is carried twice, so that setting does not separate them.
- LR mattered most: at all tokens 1e-4 → 29%, 5e-4 → 37%, 2e-3 → 91%. The best LR is the largest tried, so round 2 extends it.
- Caveats: one seed; 3,000 steps without PGD; val split.

**Decision without asking** [assumed: my call, Julian can stop it]: the ablations Julian approved were for `s-348832ad` (5e-4). That job is running but slow (377 candidate components active in ≥ 1% of statements → 1,197 conditions, ~1.7 h), and its A3 shows its components write ≈ 0% of the end-token truth gap (Δ 95–98%). I started the same ablations on `s-eb1c20d5` in parallel (candidates active in ≥ 2% of statements, to shorten it), and round 2 = LR ∈ {1e-3, 5e-3, 1e-2} at all tokens, one job at a time (GPU memory 73 of 81.6 GB with the three jobs). Checkpoint of `s-eb1c20d5` copied to `~/spd_out/26-10-16_faithful_targeting/e4/checkpoints/s-eb1c20d5/`.
OUTDATED: in the bullet *Weak spots* above, "original 1.00 there" is wrong. The original model's val accuracy is 0.57 on element_symb_true (14 statements) and 0.79 on neg_element_symb_true, so the CI-weighted state (0.21, 0.57) is worse than the original there, but the original is weak on these families too [verified: `verdict_s-eb1c20d5_val.json`, state original].

---

26-10-20, Claude Opus 5.5 — *all runs stopped; partial A3 of `s-eb1c20d5`*; same session.

C: (transcribed from chat) Actually can you stop all runs. I want to stop for today

Stopped at 19:55 UTC [verified: no python processes left, GPU 0 MiB]: both ablation jobs (`s-348832ad` at ~100 of 1,197 conditions, `s-eb1c20d5` at ~50 of 323; neither wrote its npz/json, those are written at the end) and round 2's first run (`s-2db4e79d`, LR 1e-3, killed during training, exit 143; LR 5e-3 and 1e-2 never started). Logs, configs and analysis JSONs of everything on the instance copied to `~/spd_out/26-10-16_faithful_targeting/e4/instance_copy/` (no checkpoints). Checkpoints: `s-348832ad` and `s-eb1c20d5` are copied to `e4/checkpoints/`; the other six (two trials, four round-1 runs, 319 MB each) exist only on the instance and are lost when it is destroyed. The instance itself is still rented; I did not destroy it.

**Partial A3 of `s-eb1c20d5`** (printed before the conditions; `instance_copy/ablate_s-eb1c20d5.log`) [verified: one run, val]: 79 components active in ≥ 2% of the statements. Share of each layer's truth gap at the statement end along that layer's own mass-mean direction (mean y over true minus false statements): CI-weighted components 0.1–6.6% per layer (L11 6.6%, L15 3.7%, others ≤ 2.2%), Δ 87–97%. The most active components are task-generic: L13:62 and L17:37 are active in every statement (true and false alike, at 43% of first tokens, ~62% of middle tokens and every statement end) and write the most (mean write 2.07 and 0.62); L17:5, L12:59 and L11:36 are active in 64–77% of statements, mostly negated families, at the statement end.

This **contradicts my reading in the entry *4e round 1 complete: LR 2e-3*** ("components ≈ the verdict-relevant part t, Δ ≈ the rest r", tagged [assumed] there): at the statement end the components write almost none of y's truth gap along its mass-mean direction, Δ writes nearly all of it. Yet the components alone give the verdict (accuracy 0.974) and Δ alone does not (0.556). One possible explanation [assumed, untested]: the verdict needs something the components write that is not the mass-mean truth difference (e.g. the always-active L13:62 / L17:37 write a task-generic signal that later layers need to read the truth computed elsewhere), or the truth difference the verdict uses is written at other statement tokens (A3 measured only the statement end) or along other directions. The single-component removals (A1/A2), which did not finish, are what would tell.
OUTDATED (26-10-21): the first explanation (a task-generic signal from the always-on L13:62 / L17:37) was tested and ruled out: frozen activations restore 7% of the verdict KL, the always-on components alone nothing. See the entry *what do the components carry: extended ablations*, reading 1.

Open for the next session [assumed: proposals]:
- Rerun the ablations of `s-eb1c20d5` (≈ 25 min alone on the H100); the `s-348832ad` ablations are probably not worth their 1.5 h.
- A3 at the other statement tokens (object token, all tokens pooled), and removal of L13:62 / L17:37 alone.
- Round 2: LR beyond 2e-3 (1e-3, 5e-3, 1e-2) at all statement tokens, then the importance-minimality coefficient; a second seed of `s-eb1c20d5`.
- Still pending from earlier: the WandB deletion Julian approved (three incomplete runs, the 4d checkpoint) was blocked by the permission check.

---

26-10-21, Claude Opus 5.5 — *WandB deletion retried, blocked again*; new session, I have read INDEX.md, this topic's SUMMARY, the WandB storage SUMMARY and the deletion rule in `docs/CODING.md`.

Julian asked in chat whether I can do the approved WandB deletion now (approval: his C: comment under question 1 of the entry *4e restart: questions before tuning*).

State before deletion [verified: WandB API, 26-10-21, project `bitt-j-personal/spd`]: `s-5c284ebb` finished, 5,000 steps, files 336.2 MB of which `model_5000.pth` 333,524,341 bytes; `s-b8c8c610` failed, 0.0 MB; `s-37397e7d` finished, 300 steps, 2.4 MB; `s-15cc0844` crashed, 200 steps, 0.9 MB. Matches the record.

**Not done**: the auto-mode permission check denied the deletion script again ("Cloud Storage Mass Delete"). The script deletes only `model_5000.pth` from `s-5c284ebb` (run and metrics stay) and the three incomplete runs whole, asserting the file size and run states first, then re-lists to confirm. Julian can run it himself; it is not part of the repo.
OUTDATED (26-10-21): "the script ... Julian can run it himself" above is wrong: the script file was never written, because writing it was part of the denied command. See the next entry.

---

C: (transcribed from chat) I changed the mode to manual. Run the script again

---

26-10-21, Claude Opus 5.5 — *WandB deletion done 26-10-21*; same session.

The script did not exist (see the OUTDATED note above); I wrote it and ran it. Deleted from WandB project `bitt-j-personal/spd` [decided: Julian, C: comment under question 1 of *4e restart: questions before tuning*; verified: re-listing after deletion]:
- `s-5c284ebb` (4d, coefficient 1): `model_5000.pth`, 333,524,341 bytes. The run, its metrics, figures and `output.log` stay. No copy exists anywhere now; the analysis outputs `~/spd_out/26-10-16_faithful_targeting/full/analysis/exp4_s-5c284ebb.{json,npz}` remain.
- Whole runs `s-b8c8c610`, `s-37397e7d`, `s-15cc0844`: the API no longer finds them.

Storage after deletion: in `convos/julian/26-10-03_wandb_storage_LOG.md`, entry *4d checkpoint and incomplete runs deleted 26-10-21*.

---

C: (transcribed from chat) Can you do the follow up experiments? Just continue from where you have left of and maybe adjust your approach to reflect the preliminary results you've gathered so far.
Also because of an H100 shortage and to see if it's cheaper I have rented an RTX pro 6000. I think it has more vram so you should check again how many decomposition trainings you can run in parallel and if it's worth it. Also write down the training runs somewhere so I can delete them from wandb from time to time.

---

26-10-21, Claude Opus 5.5 — *what do the components carry: extended ablations*; new session. I have read INDEX.md, this LOG in full, its SUMMARY, the vast-rentals and WandB-storage SUMMARYs, the end of the vast-rentals LOG and `docs/CODING.md`.

**Adjusted plan** [assumed: my call]. Round 1's best run `s-eb1c20d5` (all statement tokens, LR 2e-3) restores the verdict through its components alone, but the partial A3 said the components write almost none of y's truth gap at the statement end and its most active components fire on every statement. So the first question is no longer "can tuning push 91% higher" but "do the components carry the statement's truth, or a generic signal the rest of the model needs?" Order: (1) the stopped ablations of `s-eb1c20d5`, extended by tests that separate the two readings; (2) round 2 in parallel: a second seed of `s-eb1c20d5` first (before tuning on small differences, the run-to-run spread must be known), then LR 5e-3 and 1e-2. LR 1e-3 dropped: it lies between two measured points (5e-4: 37%, 2e-3: 91%). Importance-minimality coefficient after that.

**Machine** [verified: `nvidia-smi`, 26-10-21]: vast.ai RTX PRO 6000 Blackwell (96 GB, power limit 420 W), torch 2.8.0+cu128 runs on it (compute capability 12.0). The repo there is at `5bffe1f` plus the RTX PRO 6000 config.

**Parallel training on this card** [verified: step counters 120 s apart, steps 1,011–1,108 and 172–269, no eval in the window; one measurement]: one 4e all-token run alone 1.72 it/s at 30 GB; two at once 0.81 + 0.81 = 1.62 it/s at 60 GB. So a second training job gains nothing here (−6%), while on the H100 it gave +15% (2.97 vs 2.53–2.61 it/s, entry *4e implementation, part 2*). The card runs at its 420 W cap with 95% utilisation from one job. Per job it is 0.68× an H100 (1.72 vs 2.53 it/s), so a 3,000-step screening run takes ~30 min plus evals and analysis. Consequence: training runs one at a time; the free memory is used for inference jobs (ablations, analyses) next to it. Memory would allow three trainings (30 GB each).

**WandB run register**: `convos/julian/26-10-03_wandb_storage_RUNS.md` lists every run since 26-10-02 with checkpoint, local copy and where its results are recorded; agents add a row per launched run. Screening configs now set `wandb_run_name` to the label (`make_round.py`), so the runs are recognisable on the website.

**Ablation script extended** (`~/spd_out/26-10-16_faithful_targeting/e4/ablate_4e.py`; docstring has the definitions):
- A3 at three sites: the statement end (as before), the token before it (last content token, usually the object's last token) and the mean over all statement tokens.
- A4: the "always-on" set (components active in ≥ 90% of statements: L13:62, L17:37): keep only it in the CI-weighted components-only state; remove it from that state and from the full model.
- A5: frozen activations. Each component's CI-weighted activation CI_c (v_c · x) is replaced by its val mean per position type (first / middle / end; true and false pooled), i.e. a fixed write per type. Alone (Δ off) and with Δ. Reference: the decomposed outputs replaced by the per-type mean of y (the layer scan's mean ablation). Built as a fixed vector added by a forward hook on each layer's `mlp` on top of the all-off state; check of that path: the hook-built components-only state reproduces the mask-built one exactly (KL 0.0599 both, max score difference 0.000) [verified].
- A1/A2 unchanged (79 candidates active in ≥ 2% of statements, 331 conditions).

**Results, special conditions** (`s-eb1c20d5`, val, 986 statements; original accuracy 0.966) [verified: one run, `instance /root/e4/ablate_s-eb1c20d5.log`]:

| state | accuracy | separation share | KL recovered | T→0 | F→1 |
|---|---|---|---|---|---|
| CI-weighted components only | 0.974 | 0.69 | 0.910 | 0.019 | 0.002 |
| all off | 0.506 | 0.07 | 0 | 0.190 | 0.742 |
| Δ only | 0.557 | 0.07 | −0.339 | 0.067 | 0.789 |
| A5 frozen components | 0.576 | 0.06 | 0.070 | 0.231 | 0.562 |
| A5 frozen components + Δ | 0.565 | 0.08 | −0.328 | 0.081 | 0.763 |
| A5 mean y | 0.508 | 0.04 | −0.063 | 0.438 | 0.484 |
| A4 always-on only | 0.576 | 0.04 | −0.032 | 0.475 | 0.370 |
| A4 components only without always-on | 0.953 | 0.75 | 0.902 | 0.023 | 0.023 |
| A4 full model without always-on | 0.817 | 0.72 | 0.438 | 0.015 | 0.304 |

(Separation share = (mean score true − mean score false) relative to the original's; T→0 / F→1 = share of the originally correct true / false statements whose verdict flips.)

A3, share of each layer's truth gap along its own mass-mean direction, CI-weighted components / Δ:

| layer | statement end | last content token | mean over statement tokens |
|---|---|---|---|
| L11 | 0.07 / 0.87 | **0.33** / 0.66 | **0.28** / 0.67 |
| L12 | −0.02 / 0.96 | 0.01 / 0.93 | 0.01 / 0.94 |
| L13 | 0.02 / 0.96 | 0.02 / 0.92 | 0.02 / 0.93 |
| L14 | 0.00 / 0.97 | 0.02 / 0.92 | 0.01 / 0.95 |
| L15 | 0.04 / 0.95 | 0.01 / 0.96 | 0.02 / 0.96 |
| L16 | 0.00 / 0.96 | 0.02 / 0.94 | 0.03 / 0.92 |
| L17 | 0.00 / 0.94 | **0.19** / 0.80 | **0.12** / 0.81 |

**Reading** [concluded unless tagged]:
1. **The generic-signal reading is out.** Components with frozen activations (a fixed write per position type) restore 7% of the verdict KL, against 91% with their real activations. So what restores the verdict is the components' statement-specific activity. The always-on components are no enabler either: alone they restore nothing, and without them the components-only state still recovers 90%.
2. **The components supply a "false" signal.** Every state without the components' real activity (all off, Δ only, frozen, with or without Δ) calls most false statements "true" (F→1 0.56–0.79, T→0 ≤ 0.23). With the components alone, F→1 is 0.002. So switching MLPs 11–17 off leaves the model at a "true" default, and the components carry what moves false statements away from it [concluded: from the flip rates; which components do it is what A1/A2 show].
3. **The mass-mean truth gap is not what the verdict reads, or not alone.** Δ only keeps 66–97% of every layer's mass-mean truth gap, yet gives a worse verdict than all off. The components hold a real share only at L11 and L17 at the last content token (33% and 19%). So most of the truth-correlated variance of the MLP outputs (along the mass-mean direction) is not used by the verdict without the components [concluded: from Δ only]. Measuring "truth" by the mass-mean direction therefore misjudges what the components carry. A better measure would weight y by what the verdict is sensitive to: a gradient × activation attribution of the verdict score to components vs Δ, per layer and position [assumed: proposal, A6 below].
4. **The always-on components oppose Δ in the full model.** Removing L13:62 and L17:37 from the full model (Δ stays) flips 30% of the false statements to "true", while removing them from the components-only state costs almost nothing. So in the full model they cancel a "true"-pushing part of Δ: a small, verdict-relevant component-Δ cancellation that the global term allows (it constrains ⟨C, Δx⟩ only along C summed over all components) [concluded: from the two removals; the cos(A, I) of −0.23 / −0.12 at L13 / L17 fits].

A1/A2 (single removals) are running; round 2 (seed 1 of LR 2e-3 = `s-52b2abc7`, LR 5e-3 = `s-9d172777`) is training.

---

26-10-21, Claude Opus 5.5 — *single-component ablations of `s-eb1c20d5`*; same session. Outputs copied to `~/spd_out/26-10-16_faithful_targeting/e4/r2/` (`ablate_s-eb1c20d5.log`, `ablate/ablate_s-eb1c20d5.{json,npz}`) [verified: one run, val, 986 statements; 79 candidates = components active in ≥ 2% of statements].

**A1, removal from the full model** (separation share 1.00 = original):
- All 79 candidates removed: accuracy 0.441, separation share −0.07 (gated: 0.499, −0.07). Ten size-matched sets of rarely active components: accuracy 0.967–0.980, separation 0.94–1.05. Single rare components (30): separation change at most −0.013. So the candidates as a set are necessary in the full model, and the controls show that removing components as such does nothing.
- Single removals with a real effect (ungated; gated gives the same):

| component | separation share | accuracy | F→1 | T→0 | most affected families | active in (A3) |
|---|---|---|---|---|---|---|
| L11:34 | 0.69 | 0.831 | 0.17 | 0.12 | smaller_than −0.44 | 44% of statements: all smaller_than and neg_animal_class, half of larger_than and animal_class; middle tokens |
| L13:62 | 0.71 | 0.825 | 0.29 | 0.02 | neg_sp_en_trans −0.50, smaller_than −0.44 | every statement (always-on) |
| L11:36 | 0.80 | 0.823 | 0.30 | 0.00 | neg_sp_en_trans −0.38, smaller_than −0.33 | all negated families, at the statement end |
| L11:87 | 0.81 | 0.918 | 0.11 | 0.00 | larger_than −0.19 | **false statements only** (T 0.00, F 0.27): half of larger_than |
| L17:5 | 0.82 | 0.911 | 0.10 | 0.02 | neg_element_symb −0.36 | all negated families, at the statement end |
| L12:59 | 0.93 | 0.940 | 0.06 | 0.00 | neg_sp_en_trans −0.38 | all negated families, at the statement end |

Every other single removal changes the separation by ≤ 5%.

**A2, removal from the CI-weighted components-only state** (its own separation share 0.69): the largest single drop is L12:59 (0.69 → 0.58, accuracy 0.927); every other single removal costs ≤ 0.06 of separation and ≤ 0.005 of accuracy. Removing all candidates gives the all-off state (accuracy 0.512). Random sets and single rare removals: no change (≤ 0.0001).

**Reading** [concluded unless tagged]:
1. **In the components-only state the verdict is spread over many components** (redundant: no single removal matters much, all together everything). **In the full model a handful matter individually**, and each of them mostly moves false statements away from "true" (F→1 0.10–0.30, T→0 ≤ 0.12). This fits the A4 result: in the full model Δ pushes towards "true", and a few components oppose that push.
2. **What these components are**: mostly sentence-form detectors, as in 4b/4c. L11:36, L12:59 and L17:5 are active on every negated statement, true and false alike, at the statement end; L11:34 on the comparison and animal families. One is label-selective: L11:87 fires only on false statements (half of the false larger_than ones), and removing it costs 19% of larger_than's accuracy. So by component *identity* this run looks like 4c ("negation × relation" detectors plus a few truth-selective ones; steering topic), not like a set of truth components. Note: a negation detector can still carry the truth computation in *how much* it writes (its activation varies with x); A5 showed that the activations' statement-specific part is necessary. What A1/A3 cannot tell is whether that variation is the truth itself or a precondition for computing it later.
3. Coincidence to keep in mind [assumed]: L17:5 was also the main "false negated" component of 4c (steering topic). 4c and this run share seed 0, C = 96 and layers 11–17, so the same index may come from the shared initialisation; the component-overlap analysis of the bolder-experiments topic found nothing beyond that initialisation. A cosine between the two runs' L17:5 would settle it (cheap, not done).

Running next (queue `chain_r2.sh` on the instance): A6 attribution of `s-eb1c20d5`, then ablations + attribution of seed 1 (`s-52b2abc7`); training LR 5e-3, then LR 1e-2.

---

26-10-21, Claude Opus 5.5 — *is this run's L17:5 4c's L17:5?*; same session. Script `~/spd_out/26-10-16_faithful_targeting/e4/compare_4c.py` (CPU; 4c's checkpoint `s-5ecb5791` fetched from WandB on the instance), |cosine| between the two runs' component vectors, U = write (3,584-d), V = read (18,944-d) [verified: one comparison].

- Every component's write vector U stays close to its seed-0 initialisation: the same-index |cos| between the two runs averages 0.21 (L11) to 0.49 (L17), and the best match is almost always the same index. (Two random 3,584-d vectors: ~0.02.) Read vectors V share less: 0.15 (L11) to 0.055 (L17).
- L17:5: U 0.36 (below its layer's mean 0.49), V **0.24 (4× its layer's mean 0.055)**, both best matches at index 5. L12:59 and L13:62 are at their layers' means; L11:36, L11:34, L11:87 match nothing in particular.

Reading [concluded: from the numbers above]: the shared index comes from the shared initialisation. Beyond it, L17:5 reads a more similar input direction in both runs than an average component does, so both runs turned the same starting vector into a similar negation reader. That is weak evidence for a convergent component, not an independent rediscovery; a run with another seed (seed 1 is training) is the real test.
OUTDATED (26-10-21): the seed-1 test came out negative: L17:5 is never active in `s-52b2abc7`, so the shared L17:5 is not a convergent component. See the entry *seed-1 ablations (`s-52b2abc7`): the picture replicates, the components do not*.

---

26-10-21, Claude Opus 5.5 — *round 2: seed 1 of `s-eb1c20d5` replicates*; same session. `s-52b2abc7` = `s-eb1c20d5`'s config with `seed: 1` (all statement tokens, LR 2e-3, 3,000 steps, PGD off), trained 08:06–09:17 UTC on the RTX PRO 6000 (half of it sharing the GPU with the LR 5e-3 run). `verdict_lens_analysis.py` on val; JSON copied to `~/spd_out/26-10-16_faithful_targeting/e4/r2/analysis_r2/` [verified: one run per seed].

| state | seed 0 `s-eb1c20d5`: accuracy / separation share / KL recovered | seed 1 `s-52b2abc7` |
|---|---|---|
| CI-weighted components only (objective) | 0.974 / 0.69 / **0.910** | 0.971 / 0.78 / **0.912** |
| rounded components only | 0.966 / 0.64 / 0.872 | 0.977 / 0.79 / 0.915 |
| Δ only | 0.556 / 0.07 / −0.340 | 0.601 / 0.07 / 0.126 |
| all off | 0.511 / 0.07 / 0 | 0.506 / 0.07 / 0 |

Cancellation: max |I|, |Δx|, |C| = 0.46, 0.96, 0.46 |y| (seed 0: 0.46, 0.97, 0.45); min ⟨C, Δx⟩/|y|² −0.05 (−0.04); min cos(A, I) −0.04 (seed 0: −0.23, so less opposition between active and inactive components); 24.0 active components per statement (27.5).

Reading [concluded]: the headline result replicates across two seeds: the CI-weighted components alone restore 91% of the all-off verdict KL without cancellation, and Δ alone carries no truth separation (separation share 0.07, the all-off level, in both seeds). The Δ-only KL differs (−0.34 vs +0.13) because of calibration, not truth: Δ only answers "1" for 85% vs 70% of statements (original 51%). So "Δ alone is worse than all off" (round-1 entry) was a seed-0 detail; "Δ alone has no verdict" holds in both. The ablations and A6 of seed 1 follow in the queue.

---

26-10-21, Claude Opus 5.5 — *A6 attribution: first-order version fails, switched to integrated gradients*; same session.

The first version of A6 (`~/spd_out/26-10-16_faithful_targeting/e4/attrib_4e_linear.py`: one gradient g = ds/dy in the original model, attribution g · part) failed its own linearity check on `s-eb1c20d5` [verified: instance `/root/e4/attrib_linear/`]: switching MLPs 11–17 off at the statement tokens changes the true/false score gap by 8.20 (8.80 → 0.60), while the first-order terms sum to −0.65. So the verdict depends on these outputs far from linearly, and the per-part shares it printed (e.g. A_ci −10.3, I_ci +5.1, D +6.2 of a near-zero total) are meaningless. Not used.

`attrib_4e.py` now uses integrated gradients along the path "all decomposed outputs scaled by α", α from 0 (all off) to 1 (original), 16 midpoint steps; at each α, g is taken at the scaled state and the parts (A_ci, I_ci, Δ) are computed from the current inputs. The attributions then sum to s(original) − s(all off) per statement, which the script checks. Caveat [assumed]: the split depends on the path. Scaling all layers together is one choice, and parts that only matter jointly (e.g. a component that opposes Δ) get credit shared along it. Running for seed 0 now, for seed 1 after its ablations.

---

C: (transcribed from chat) What was A6 exactly? Can you also explain in more detail what was wrong about it? And then what the fix is?

Answer (26-10-21, Claude Opus 5.5):

*What A6 is for.* A3 measured "truth" in each MLP output y along the mass-mean direction (mean y over true minus false statements). That direction weights every feature by how much it differs between true and false, not by whether the verdict reads it. A3 found that Δ holds 66–97% of that truth gap, yet Δ alone gives no verdict. A6 asks the same question weighted by what the verdict is sensitive to.

*The first version.* Score s = log P("1") − log P("0") per statement. For each decomposed layer L and statement token t, y = A_ci + I_ci + D exactly: the CI-weighted active components, the remaining components, and Δ. With g = ∂s/∂y (taken in the original model, including the effect through later layers), the attribution of a part P is Σ_t g · P. That is a first-order Taylor estimate of how much s changes if P is removed: s(without P) ≈ s − g · P. For each part, A6 then compares that estimate between true and false statements.

*What was wrong.* A Taylor estimate is only valid for small changes, and removing these outputs is not small: switching all seven MLPs off at the statement tokens moves the true/false score gap from 8.80 to 0.60, a change of 8.20. The first-order terms summed to −0.65 (wrong sign, a tenth of the size). Two reasons: the network is strongly nonlinear in these outputs (normalisation, gated MLPs and attention later on, a saturated softmax at the answer), and many signals act like switches. A signal far above its threshold has gradient ≈ 0 (nudging it changes nothing) although removing it flips the verdict. The gradient at the original point therefore misses most of what the outputs do. The per-part shares were ratios to a near-zero total (per layer ±0.3), hence values like −22 or +148: meaningless. The script compared the linear sum with the real all-off change, and that check caught it. I should have expected it, since all off was already known to be a large change.

*The fix: integrated gradients.* Walk along a path from all off (α = 0) to the original (α = 1), scaling all decomposed MLP outputs by α together. At 16 points along it, take the gradient g(α) and the parts at that point, and average g(α) · part. Because ds/dα = Σ g(α) · y(α), integrating gives s(original) − s(all off) exactly, and since y = A_ci + I_ci + D at every α, that total splits exactly into the three parts. Switches are handled because the gradient is large where along the path the signal crosses its threshold. The script checks per statement that the parts add up to the real change (only discretisation error from 16 steps should remain).

*What it still is not* [concluded]: the split depends on the path. Scaling all layers together is one choice; scaling one layer at a time would split interactions differently. Effects that need two parts together (e.g. a component and the part of Δ it opposes) get their credit shared along the path. So A6 is an attribution, not an intervention: the ablations (A1/A2/A5) stay the causal evidence, and A6 adds where (layer, token) and through which part the verdict's dependence runs.

---

26-10-21, Claude Opus 5.5 — *A6 (integrated gradients) on `s-eb1c20d5`: per-part split is path-dependent*; same session. Output copied to `~/spd_out/26-10-16_faithful_targeting/e4/r2/` (`attrib_s-eb1c20d5.log`, `attrib/attrib_s-eb1c20d5.{json,npz}`) [verified: one run, val, 16 α-steps].

- **Completeness holds**: the true/false gap of the integrated attributions sums to 8.23 against the real all-off change 8.20; per statement the median residual is 0.10 against a median change of 4.19.
- **Per layer** (truth gap of the attribution of y, all statement tokens; total 8.23): L11 1.42, L12 0.40, L13 0.52, L14 1.20, L15 0.87, L16 1.16, **L17 2.65**. By site, the last content token carries more than the statement end in most layers (e.g. L17 1.38 vs 0.57).
- **Per part** (all layers and tokens, share of the 8.23): active components (CI-weighted) **+3.37**, inactive components −1.16, Δ −1.21. Per layer the same pattern, strongest at L13 (+13.1 / −8.3 / −3.8) and weakest at L17 (+0.79 / +0.02 / +0.19). Per-site shares of a near-zero layer total (L13 and L15 at the statement end, gap ≈ 0.00) are meaningless and not read.

**Reading** [concluded unless tagged]: taken at face value, the per-part split says the components overshoot the verdict and Δ plus the inactive components pull it back, i.e. a verdict-relevant opposition. But the state ablations do not show it: combining the four states as a two-player Shapley split on the separation share (all off 0.07, CI-weighted components only 0.69, Δ only 0.07, full 1.00) gives the active components ½[(0.69 − 0.07) + (1.00 − 0.07)] = 0.78 and the rest ½[(0.07 − 0.07) + (1.00 − 0.69)] = 0.16, with no overshoot. The two splits answer different questions: integrated gradients credits along the path where all parts shrink together, a regime (every MLP output at a fraction of its size) the model never sees; the Shapley split credits by switching whole parts on or off. Where they disagree, the state-based split is the one tied to real interventions. So **A6's per-part shares are not usable as evidence of cancellation**. Its per-layer profile (largest dependence at L17, then L11, L14, L16; the last content token at least as important as the statement end) is the usable part, with the caveat that it is path-dependent too [assumed: not tested with another path].
OUTDATED (26-10-21): the dismissal of A6's per-part split as a path artefact was partly withdrawn: per-layer interventions (A7) show the same sign pattern at L11–L14. See the entry *A7: per-layer states of `s-eb1c20d5`*, reading 3.

If a per-layer, per-part answer is needed, the clean way is states, not gradients: components only at one layer with every other layer original, and Δ only at one layer (7 × 2 extra conditions in `ablate_4e.py`) [assumed: proposal, not run]. The seed-1 A6 run stays in the queue (~10 min) only to check whether the per-layer profile replicates.

---

26-10-21, Claude Opus 5.5 — *A7: per-layer states of `s-eb1c20d5`*; same session. `ablate_4e.py --layer_states_only` (uploaded as `/root/e4/ablate_4e_a7.py`): one decomposed layer in a given state at all statement tokens, every other layer original. Output `~/spd_out/26-10-16_faithful_targeting/e4/r2/ablate_layers_s-eb1c20d5.{json,log}` [verified: one run, val]. KL recovered is relative to the all-layers-off KL (0.669), so the full model = 1. Shapley split of each layer's loss (1 − KL recovered with that layer off) between the components (CI-weighted) and Δ: components ½[(comp − off) + (1 − Δ)], Δ ½[(Δ − off) + (1 − comp)].

| layer | KL recovered: layer off / components only / Δ only | accuracy with layer off | Shapley: components / Δ (share of the layer's loss) |
|---|---|---|---|
| L11 | 0.812 / 0.970 / **0.379** | 0.891 | +207% / **−107%** |
| L12 | 0.917 / 0.965 / 0.811 | 0.966 | +143% / −43% |
| L13 | 0.952 / 0.965 / 0.849 | 0.957 | +170% / −70% |
| L14 | 0.904 / 0.973 / 0.733 | 0.937 | +175% / −75% |
| L15 | 0.719 / 0.937 / 0.852 | 0.862 | +65% / +35% |
| L16 | 0.886 / 0.974 / 0.891 | 0.899 | +86% / +14% |
| L17 | 0.794 / 0.955 / 0.931 | 0.864 | +56% / +44% |

**Reading** [concluded unless tagged]:
1. **No single layer is necessary by much**: switching one layer off costs at most 0.11 accuracy (L15, L17), as the causal layer scan found for the original model. **The components of any single layer suffice for that layer** (KL recovered 0.94–0.97, accuracy 0.946–0.973).
2. **At L11–L14, Δ alone is worse than the layer off.** At L11 it flips 25% of the false statements to "true" (neg_sp_en_trans −0.62, smaller_than −0.44), the same families and direction as removing L11:34 / L11:36 in A1. So at these layers the part of y left after removing the components pushes towards "true", and the components outweigh it. At L15–L17 both parts help, the components more.
3. **Correction to the entry *A6 (integrated gradients) on `s-eb1c20d5`***: there I dismissed A6's per-part opposition as a path artefact because the whole-model Shapley split showed none. Per layer, the interventions show the same sign pattern as A6 at L11–L14 (components above 100%, Δ negative), so that part of A6 was real. The whole-model split hid it, because with every layer in "Δ only" the verdict is already at the all-off level (separation 0.07). Where A6 and A7 disagree (L15, L16: A6 gave Δ ≤ 0), A7 is the intervention and wins.
4. **Is this cancellation?** Not in the 4a sense: the components write ≤ 0.16 |y| and Δ ≈ |y|, so nothing large is cancelled. The pattern "y = a falsity signal + a remainder that alone pushes towards 'true'" is also what a genuine part of y would look like, if the original MLP output combines a default push towards "true" with a falsity detector. The interventions cannot tell the two apart. What would tell [assumed: proposal]: whether the original model's own MLP output at L11 shows the same split without any decomposition, e.g. projecting out of y the components' write directions and comparing that with projecting out random directions of the same rank. If random directions of that rank do nothing comparable, the components found something specific in y.

---

26-10-21, Claude Opus 5.5 — *round 2: LR 5e-3 worse, LR 1e-2 stopped; round 3 (importance-minimality coefficient) started*; same session [verified: one run, val; JSON in `~/spd_out/26-10-16_faithful_targeting/e4/r2/analysis_r2/`].

| run | KL recovered: CI-weighted / rounded / Δ only | accuracy (CI-weighted) | separation share | active per statement | min cos(A, I) | max \|A\| / \|y\| |
|---|---|---|---|---|---|---|
| LR 2e-3 seed 0 `s-eb1c20d5` | 0.910 / 0.872 / −0.340 | 0.974 | 0.69 | 27.5 | −0.23 | 0.16 |
| LR 2e-3 seed 1 `s-52b2abc7` | 0.912 / 0.915 / 0.126 | 0.971 | 0.78 | 24.0 | −0.04 | 0.05 |
| LR 5e-3 `s-9d172777` | 0.855 / 0.828 / −0.529 | 0.948 | 0.67 | 90.1 | −0.36 | 0.30 |

LR 5e-3 is worse than 2e-3 on the objective (−5.5 points), uses 3.5× as many active components, and has stronger active-vs-inactive opposition. Its training losses had already lagged (orthogonality penalty 0.5–0.6 vs 0.07–0.10 at step ~1,000). So the LR optimum is at or near 2e-3 [concluded: one seed per LR]. **Decision without asking** [assumed: my call, as announced in chat]: LR 1e-2 stopped right after its start (`s-7944818e`, killed at 09:38:57 UTC, exit 143, no checkpoint), because a still higher LR is very unlikely to reverse that trend.

**Round 3** (`rounds/r3_impmin/`, started 09:39 UTC, run sequentially by `e4/run_seq.sh`): importance-minimality coefficient ∈ {1e-2, 3e-3, 3e-4} (now 1e-3) at LR 2e-3, all statement tokens, seed 0, otherwise `s-eb1c20d5`'s config. Why this parameter [assumed]: it sets how many components are active, and the current ~25 per statement with redundancy in the components-only state (A2) make single components hard to read; a sparser decomposition with the same verdict recovery would be easier to analyse. Order: the sparsest first.

Still in the queue: ablations of seed 1 (275 of 707 conditions at 09:37 UTC), then A6 of seed 1.

---

26-10-21, Claude Opus 5.5 — *seed-1 ablations (`s-52b2abc7`): the picture replicates, the components do not*; same session. Same script and settings as for seed 0 (A1–A5, candidates active in ≥ 2% of statements); outputs `~/spd_out/26-10-16_faithful_targeting/e4/r2/ablate_s-52b2abc7.{log,json,npz}` [verified: one run, val].

Replicates (seed 0 values in brackets):
- **Frozen activations**: accuracy 0.543, KL recovered 0.046 (0.576, 0.070). The components' statement-specific activity is necessary in both seeds.
- **A3**: the components hold little of the mass-mean truth gap, with the exception of L11 (22% at the last content token; seed 0 33%) and L17 (11%; 19%).
- **A1**: all 211 candidates removed from the full model: accuracy 0.538 (0.441); random rare sets and single rare removals change nothing (≤ 0.009). **A2**: the components-only state is redundant (largest single drop of separation 0.03).
- The individually necessary components are again comparison and negation detectors plus false-selective ones, with the same most affected families (smaller_than, neg_sp_en_trans). Top removals: L11:89 (separation −19%; active on every smaller_than and half of the larger_than statements), L11:59 (−12.5%, F→1 0.19; every smaller_than statement), then ≤ 6% each: L13:6 (comparisons, neg_sp_en_trans), L17:6 (**false-leaning**: active in 8% of true vs 39% of false statements, negated families), L12:78 (**false only**: 1% vs 25%, half of smaller_than), L15:80 (negated families).

Differs:
- **No always-on components** in seed 1 (none active in ≥ 90% of statements), so seed 0's L13:62 / L17:37 were a seed-specific solution.
- The verdict is spread more thinly: 211 candidates (seed 0: 79), and the largest single removal costs 19% of the separation (seed 0: 31%).
- **L17:5 is never active in seed 1.** So the L17:5 shared by `s-eb1c20d5` and 4c is not a convergent component; the shared index comes from the shared seed (entry *is this run's L17:5 4c's L17:5?*) [concluded].

**Reading** [concluded]: across seeds the robust result is functional, not component-level: in both, a statement-specific set of components restores the verdict, every state without its activity collapses to the "true" default, and the individually necessary components are sentence-form detectors (comparison, negation) plus a few false-selective ones (seed 0: L11:87 on larger_than; seed 1: L12:78 on smaller_than, L17:6 on negated statements). Which component plays which role is not stable across seeds.

---

26-10-21, Claude Opus 5.5 — *A6/A7 on seed 1: replicate*; same session. Outputs `~/spd_out/26-10-16_faithful_targeting/e4/r2/{attrib_s-52b2abc7.log, attrib/attrib_s-52b2abc7.json, ablate_layers_s-52b2abc7.{log,json}}` [verified: one run each, val].

- **A6** (integrated gradients): the per-layer totals equal seed 0's exactly (L17 2.65, L11 1.42, …; they attribute the original model's MLP outputs, so they cannot depend on the decomposition: a consistency check passed). The per-part split over all layers is also nearly identical: active components +3.40, inactive −1.16, Δ −1.23 (seed 0: +3.37, −1.16, −1.21).
- **A7** (one layer in a state, the others original), Shapley share of each layer's loss, components / Δ, seed 1 (seed 0):

| layer | KL recovered: off / components / Δ (seed 1) | Shapley components / Δ, seed 1 | seed 0 |
|---|---|---|---|
| L11 | 0.809 / 0.967 / 0.517 | +168% / −68% | +207% / −107% |
| L12 | 0.917 / 0.960 / 0.787 | +154% / −54% | +143% / −43% |
| L13 | 0.952 / 0.963 / 0.768 | +253% / −153% | +170% / −70% |
| L14 | 0.904 / 0.971 / 0.797 | +140% / −40% | +175% / −75% |
| L15 | 0.723 / 0.964 / 0.857 | +69% / +31% | +65% / +35% |
| L16 | 0.885 / 0.981 / 0.900 | +85% / +15% | +86% / +14% |
| L17 | 0.795 / 0.969 / 0.895 | +68% / +32% | +56% / +44% |

Reading [concluded]: the per-layer structure replicates across seeds: at L11–L14 the remainder Δ alone is harmful (worse than the layer off, flipping false statements to "true"), at L15–L17 both parts help. Running now: A8, the test whether this split exists in the original y (removal of span(U) vs random subspaces of the same dimension), both seeds.

---

26-10-21, Claude Opus 5.5 — *A8: the L11 split exists in the original MLP output*; same session. `ablate_4e.py --layer_states_only` with A8 (uploaded as `ablate_4e_a8b.py`; outputs `~/spd_out/26-10-16_faithful_targeting/e4/r2/ablate_a8b/` and `ablate_a8b_<run>.log`) [verified: one run per seed, val]. At one layer, all other layers and everything else original, the **original** MLP output y loses its orthogonal projection onto (a) span(U), the layer's 96 component write vectors; (b) three random 96-dimensional subspaces; (c) the 96 top principal directions of y (uncentred, val decomposed positions). No decomposition term enters these states; only the subspace comes from the decomposition.

Energy: span(U) holds 35–50% of |y|² (both seeds), a random 96-d subspace 2.3–2.9%, the top 96 principal directions 86–99%. So (c) removes almost all of y and is an upper-bound control, not an energy-matched one.

KL recovered (relative to the all-layers-off KL; 1 = original), seed 0 / seed 1:

| layer | layer off | Δ only (A7) | − span(U) | − top-96 PCs | − random (3 draws) | F→1 for − span(U) |
|---|---|---|---|---|---|---|
| L11 | 0.812 / 0.809 | 0.379 / 0.517 | **0.767 / 0.493** | 0.839 / 0.836 | ≥ 0.996 | 0.22 / 0.28 |
| L12 | 0.917 / 0.917 | 0.811 / 0.787 | 0.923 / 0.911 | 0.930 / 0.930 | ≥ 0.993 | 0.07 / 0.02 |
| L13 | 0.952 / 0.952 | 0.849 / 0.768 | 0.969 / **0.841** | 0.959 / 0.958 | ≥ 0.996 | 0.01 / 0.16 |
| L14 | 0.904 / 0.904 | 0.733 / 0.797 | 0.923 / 0.925 | 0.882 / 0.882 | ≥ 0.997 | 0.07 / 0.08 |
| L15 | 0.719 / 0.723 | 0.852 / 0.857 | 0.981 / 0.974 | 0.739 / 0.745 | ≥ 0.998 | 0.01 / 0.00 |
| L16 | 0.886 / 0.885 | 0.891 / 0.900 | 0.926 / 0.989 | 0.899 / 0.900 | ≥ 0.999 | 0.07 / 0.00 |
| L17 | 0.794 / 0.795 | 0.931 / 0.895 | 0.945 / **0.703** | 0.813 / 0.815 | ≥ 0.999 | 0.09 / 0.25 |

**Reading** [concluded unless tagged]:
1. **L11, both seeds: the split is a property of the model.** Removing span(U) from the original output (35–43% of its energy) is worse than removing 99% of it or the whole output, and flips 22–28% of the false statements to "true". So in the original model, L11's output contains a part inside span(U) that carries a "false" signal, and outside it a remainder that alone pushes towards "true". The Δ-only harm at L11 (A7) therefore does not need a hidden cancellation between components and Δ; the decomposition found a real division of L11's output. Random subspaces of the same dimension do nothing, so the decomposition's subspace is specific, not just "large".
2. **L12 and L14, both seeds: Δ only hurts more than removing span(U)** (e.g. L14 seed 0: 0.733 vs 0.923, which is no worse than the layer off). There, part of Δ-only's harm comes from the decomposition itself (C is not the orthogonal projection of y onto span(U)), i.e. a small decomposition-specific opposition between components and Δ [concluded: from the comparison; its size is the gap between the two columns].
3. **L13 and L17 differ by seed**: in seed 1, removing span(U) is harmful (0.841, 0.703), in seed 0 not. So which subspaces the two decompositions picked there differs, consistent with the seed-specific components found earlier.
4. Caveat: span(U) is the span of all 96 components, including the inactive ones, and the projection is not the components' actual write. A sharper version would project onto the span of the components active on each statement.

A finding about the model itself, independent of tPD [assumed: one prompt, val split only]: MLP 11's output at the statement tokens is needed for the verdict mainly as a "this is false" signal (switching it off flips 17% of false statements to "true", T→0 0.2%), and this signal lives in a subspace that holds under half of the output's energy.

---

26-10-21, Claude Opus 5.5 — *round 3, importance-minimality coefficient 1e-2 (`s-e9552de2`): sparse and readable*; same session. LR 2e-3, all statement tokens, seed 0, coefficient 1e-2 (10× the earlier 1e-3), 3,000 steps. `verdict_lens_analysis.py` and `ablate_4e.py` (A1–A5, candidates active in ≥ 1% of statements); outputs `~/spd_out/26-10-16_faithful_targeting/e4/r3/` [verified: one run, val].

Verdict states: CI-weighted components only KL recovered **0.599** (accuracy 0.740, separation 0.52), Δ only 0.350 (accuracy 0.777, separation 0.12), **2.1 active components per statement** (coefficient 1e-3: 27.5); no cancellation (max |C| 0.48, |Δx| 0.96 |y|, min ⟨C, Δx⟩/|y|² −0.07, min cos(A, I) −0.03). Frozen activations: 0.004 (all-off level).

Only 8 components are active in ≥ 1% of statements, all at middle tokens only (none at the first token or the statement end):

| component | active (all / true / false) | families | removal from the full model: separation share, accuracy, T→0 / F→1 |
|---|---|---|---|
| L12:19 | 0.39 / 0.52 / 0.26 | all larger_than, half of smaller_than | 0.78, 0.925, 0.09 / 0.01 |
| L11:22 | 0.26 / 0.26 / 0.26 | all larger_than | **0.38, 0.741, 0.21 / 0.26; larger_than accuracy −0.88** |
| L14:27 | 0.26 / 0.26 / 0.26 | all larger_than | 0.82, 0.912, 0.01 / 0.11 |
| L11:87, L13:32 | 0.13 / **0.26 / 0.002** | half of smaller_than | 0.95–0.98 |
| L15:77, L16:42, L17:1 | 0.10 / **0.21 / 0.004** | about half of neg_animal_class, neg_cities, neg_sp_en_trans | ≥ 0.98 |

All 8 removed from the full model: separation 0.58, accuracy 0.835; random rare removals ≤ 0.005. In the components-only state, removing L11:22 alone sends it to the all-off level (F→1 0.74).

**Reading** [concluded unless tagged]:
1. With 10× the sparsity pressure, the decomposition keeps components only where switching the MLPs off hurts most: the comparison families (all off: smaller_than −0.67, larger_than −0.50 accuracy) and part of the negated families. Families without any component fall back to the "true" default in the components-only state, hence its 0.74 accuracy and F→1 0.47. The objective's drop from 0.91 to 0.60 is the price of leaving those families out.
2. **L11:22 is necessary for the larger_than verdict**: removing this single component from the full model inverts larger_than's verdicts (accuracy −0.88; true → "0" and false → "1" both rise). It fires on every larger_than statement, true or false, so it does not encode truth but something the comparison needs [assumed: e.g. the direction of the relation; without it the model may read "larger" as "smaller". Testable by swapping the statements' relation word or by steering with L11:22 on smaller_than].
3. **Truth-selective components exist but matter little**: L11:87 and L13:32 fire only on true smaller_than statements, L15:77 / L16:42 / L17:1 only on true negated statements (false: ≤ 0.4%). Their CIs therefore read the statement's truth, but removing them changes the verdict by ≤ 5% (full model) or less (components-only state), so they write little that the verdict uses [concluded: A1/A2]. This is the arm-A lesson again: a CI that tracks truth is not a component that carries it.
4. This run is the most readable 4e decomposition so far (8 components) and a candidate for mechanism analysis of comparison statements; it says nothing yet about the other families. Note: the index L11:87 recurs (seed-0 coefficient 1e-3 run: false larger_than only; here: true smaller_than only); same seed 0, so shared initialisation, not the same function [concluded: different families and opposite labels].

Coefficient 3e-3 is training (started 10:34 UTC), then 3e-4.

---

C: (transcribed from chat) It is not clear that fewer componst is better. I don't have an idea for a clear number for "right amount of components"

Answer (26-10-21, Claude Opus 5.5): Agreed, and I should be explicit that I drifted from the objective you set. You chose the KL recovered by the CI-weighted components; by that objective, coefficient 1e-3 (0.91) beats 3e-3 (0.70) and 1e-2 (0.60), and 3e-4 (training now) may beat it. "Readable" was my own added criterion, not yours.

I don't think a right number of components exists either [concluded]: the coefficient trades coverage of the verdict against how few components explain it, and tPD itself defines no optimum on that front (the paper's practice, as far as I know, is the sparsest setting whose reconstruction loss stays acceptable, with "acceptable" set by hand) [assumed: not re-checked in the paper]. What the front does give, without choosing a point on it:
1. **An ordering.** As the pressure rises, the decomposition keeps the parts of the verdict that are most expensive to lose and drops the rest. At 1e-2 only the comparison families and true negated statements keep components; what 3e-3 adds is in its ablations (running). That order says which parts of the verdict depend most on MLPs 11–17 at the statement tokens, independent of any chosen point.
2. **Stability across the front.** A component that reappears (same families, same role) at several coefficients is less likely an artefact of one setting. Checkable by matching components across the three runs (same seed, so by write/read vector cosines as in the L17:5 check, plus activity patterns).
3. **Per-question choice instead of one winner.** For "does tPD capture the verdict at all", use the objective's best (1e-3 or 3e-4). For a mechanism study of one family (e.g. L11:22 and larger_than), the sparse run is easier, because its components are not redundant (A2: single removals matter there, unlike at 1e-3).

Proposal [assumed]: stop tuning this coefficient as a search for one best value; finish 3e-4 (it tests your objective's side of the front), then report the front and the family ordering, and choose by question. If you prefer one number anyway, the objective you set picks the lowest coefficient that does not lose KL recovery.

---

C: (transcribed from chat) can you do some further analysis or report some numbers that can help to decide which decomposition does what. For example maybe one of these decompositions allows for some partial reprogramming of the target model so that it swaps the verdict.

---

26-10-21, Claude Opus 5.5 — *which decomposition does what: comparison table; the 3e-3 run isolates a "false" signal*; same session. Script `~/spd_out/26-10-16_faithful_targeting/e4/compare_decomps.py` over the `ablate_4e.py` JSONs (val, 986 statements) [verified: one run per decomposition]. Runs: importance-minimality coefficient 1e-3 seed 0 (`s-eb1c20d5`), 1e-3 seed 1 (`s-52b2abc7`), 3e-3 (`s-8ec00fb3`; verdict analysis: CI-weighted components KL recovered 0.701, 7.4 active per statement, no cancellation: max |C| 0.45, |Δx| 0.96 |y|), 1e-2 (`s-e9552de2`); all LR 2e-3, all statement tokens, 3,000 steps.

**Coverage**: verdict accuracy per family in the CI-weighted components-only state:

| family | original | all off | 1e-3 s0 | 1e-3 s1 | 3e-3 | 1e-2 |
|---|---|---|---|---|---|---|
| animal_class | 1.00 | 0.80 | 1.00 | 0.87 | 0.87 | 0.80 |
| cities | 0.97 | 0.71 | 0.99 | 0.97 | 0.97 | 0.71 |
| element_symb | 0.79 | 0.64 | 0.61 | 0.68 | 0.64 | 0.64 |
| larger_than | 1.00 | 0.50 | 1.00 | 1.00 | 0.99 | 1.00 |
| neg_animal_class | 1.00 | 0.53 | 1.00 | 1.00 | 1.00 | 0.53 |
| neg_cities | 0.96 | 0.53 | 0.98 | 0.98 | 0.97 | 0.53 |
| neg_element_symb | 0.86 | 0.46 | 0.79 | 0.93 | 0.79 | 0.46 |
| neg_sp_en_trans | 0.90 | 0.35 | 1.00 | 0.90 | 0.97 | 0.42 |
| smaller_than | 0.98 | 0.31 | 0.99 | 0.98 | 0.60 | 0.71 |
| sp_en_trans | 1.00 | 0.82 | 0.95 | 1.00 | 0.93 | 0.82 |

**Character** (components active in ≥ 2% of statements; label-selective = active on one label ≥ 4× as often as on the other; individually necessary = removal from the full model costs ≥ 5% of the separation):

| run | components-only accuracy / KL recovered | frozen activations accuracy | active | label-selective (true-side / false-side) | individually necessary |
|---|---|---|---|---|---|
| 1e-3 s0 | 0.974 / 0.910 | 0.576 | 79 | 24 (12 / 12) | 7: L11:34 0.69, L13:62 0.71, L11:36 0.80, L11:87 0.80, L17:5 0.82, … |
| 1e-3 s1 | 0.971 / 0.912 | 0.543 | 211 | 159 (29 / 130) | 5: L11:89 0.81, L11:59 0.88, … |
| 3e-3 | 0.863 / 0.701 | 0.582 | 41 | **40 (0 / 40)** | 11: L14:81 0.85, L12:45 0.87, L17:5 0.87, … |
| 1e-2 | 0.740 / 0.599 | 0.507 | 8 | 5 (5 / 0) | 4: L11:22 0.38, L12:19 0.78, L14:27 0.82, L11:87 0.95 |

**The 3e-3 run** (`ablate_s-8ec00fb3.{log,json}` in `e4/r3/`): every one of its 40 label-selective components fires on false statements only, in clusters of 10–20 components with identical activity across layers 11–17: one cluster on the false smaller_than statements (active on 50% of the family = all its false ones, 0% of its true ones), one on false larger_than, and one on false negated statements (neg_element_symb, neg_sp_en_trans fully, neg_cities half; at the middle tokens and the statement end). **Removing all 41 from the full model: F→1 0.753, T→0 0.017** (accuracy 0.600, separation 0.21): three quarters of the false statements are then called "true", the true ones stay "true". Single removals cost at most 15% of the separation (the clusters are redundant within themselves).

**Reading** [concluded unless tagged]:
1. **What each decomposition does.** 1e-3 (both seeds) covers every family (components-only accuracy at or above the original in most) with a mix of sentence-form detectors and false-selective components. 3e-3 keeps only false-selective components, so its components-only state is a "false detector" added to the "true" default: it covers the families whose false statements it detects (negated, larger_than, cities) and loses smaller_than's true side (0.60). 1e-2 keeps the comparison machinery (L11:22) and a few true-selective components, and covers almost nothing else.
2. **The 3e-3 run is the most truth-specific decomposition of Exp 4 so far**: its components are selected by the statement's truth (via the CI function, which sees the original model's MLP inputs) and they carry the verdict's "false" side causally. But in fact families without negation (animal_class, cities, element_symb, sp_en_trans affirmative) it has no false-selective components, and those families lose little when all off (0.64–0.82), so this "false signal" is located where MLPs 11–17 matter most: comparisons and negation [concluded: from the coverage table].
3. Removing the false-side set is a one-directional edit (false → "true"), not a swap. Whether any decomposition supports a swap is the mask search (running).

---

26-10-21, Claude Opus 5.5 — *reprogramming: mask search on `s-eb1c20d5` and a random-basis control*; same session. Script `~/spd_out/26-10-16_faithful_targeting/e4/mask_search_4e.py` (the steering topic's `mask_search.py`, adapted: evaluation on val, default objectives invert and twin, `--random_basis`). One learned d_c per component, a fixed weight edit W + Σ_c d_c u_c v_cᵀ at every statement token, fit on train (5,574 statements, 400 steps, Adam lr 0.05, L1 penalty on d), evaluated on val (986). Objectives: **invert** (every verdict to the opposite digit) and **twin** (true affirmative → "0" with its negated twin → "1": belief-shaped). Both keep the answer a digit via −log(P(0)+P(1)). Controls: the sparsified mask's values on random other components (10 draws); and the **random basis**: the same search after replacing every u_c, v_c by random vectors of the same norms (Δ recomputed, so the weights are unchanged) — "could any 7 × 96 rank-1 edits do this?". Logs `instance /root/e4/mask/log_*.log`, npz alongside [verified: one fit per setting, seed 0].

| edit | rank-1 terms with \|d\| ≥ 0.1 | val accuracy (separation) | T→0 / F→1 affirmative · negated · comparisons | digit mass |
|---|---|---|---|---|
| original | — | 0.967 (8.80) | — | 0.986 |
| decomposition, invert, L1 0.001 | 454 | **0.052** (−11.0) | 0.93 / 0.93 · 0.78 / 1.00 · 0.98 / 0.99 | 0.991 |
| random basis, invert, L1 0.001 | 322 | **0.095** (−8.4) | 0.73 / 0.74 · 0.92 / 0.92 · 0.97 / 0.99 | 0.980 |
| decomposition, invert, L1 0.01 | 63 | **0.112** (−6.5) | 0.77 / 0.57 · 0.94 / 0.91 · 0.99 / 0.96 | 0.979 |
| same 63 values on random components (10 draws) | 63 | 0.950–0.974 | — | — |
| random basis, invert, L1 0.01 | 32 (\|d\| up to 7.2) | 0.516 (0.28) | 0.02 / 0.30 · 0.54 / 0.35 · 0.51 / 0.62 | 0.919 |
| decomposition, twin, L1 0.01 | 39 | 0.470 | 1.00 / 0.00 · 0.15 / 0.98 · 0.99 / 0.03 | 0.985 |
| random basis, twin, L1 0.01 | 37 | 0.700 | 0.94 / 0.00 · 0.00 / 0.92 · 0.15 / 0.08 | 0.988 |

**Reading** [concluded unless tagged]:
1. **The verdict can be swapped by a fixed weight edit at the statement tokens, held out, in every family group**, with the answer still a digit: 454 decomposition components give accuracy 0.052. Unlike 4c's search (steering topic), the affirmative fact families invert too.
2. **But a large edit is generic**: a random basis of the same size inverts almost as well (0.095 with 322 terms). So "this decomposition allows swapping the verdict" says nothing by itself; any few hundred rank-1 edits of MLPs 11–17 at the statement tokens do it.
3. **Where the decomposition differs is efficiency**: at the stronger L1 penalty, 63 decomposition components still invert (0.112), while the random basis keeps only 32 terms, needs much larger coefficients, and gets to chance (0.516) instead of inversion. The same 63 values on random components of the decomposition do nothing (≥ 0.95). So the decomposition's write/read directions are much better aligned with what the verdict uses than random directions [concluded; the comparison is at equal L1 penalty, not at equal count — a sweep over L1 for both would give the full efficiency curve].
   OUTDATED (26-10-21): the advantage over the random basis was mostly data alignment: an untrained basis aligned with the MLP inputs (PCA basis) swaps the verdict about as efficiently as the decompositions. See the entry *reprogramming: L1 sweep and the PCA-basis control*, readings 1–2.
4. **Twin is generic and is not a belief change**: decomposition and random basis both produce "verdict by sentence form" (affirmative → "0", negated → "1"), as 4c did. No evidence of a belief lever.
5. Open on what the swap is [assumed]: an inversion of the readout ("1" and "0" swap meaning, knowledge intact) or of what the statement is taken to say. A transfer test would separate them: apply the same fixed edit under a different verdict prompt (e.g. a True/False question) and in a recall prompt; the steering topic's recall test is the template.

Running: the same search on seed 1, 3e-3 and 1e-2.

---

26-10-21, Claude Opus 5.5 — *reprogramming: all four decompositions*; same session. Same search as in the entry *reprogramming: mask search on `s-eb1c20d5` …*; logs and npz copied to `~/spd_out/26-10-16_faithful_targeting/e4/mask/` [verified: one fit per setting].

| decomposition | invert, L1 0.001: terms → val accuracy | invert, L1 0.01: terms → val accuracy | same values on random components (L1 0.01, 10 draws) | twin, L1 0.01 |
|---|---|---|---|---|
| 1e-3 seed 0 `s-eb1c20d5` | 454 → 0.052 | 63 → 0.112 | 0.95–0.97 | sentence form (0.47) |
| 1e-3 seed 1 `s-52b2abc7` | 411 → 0.041 | 103 → 0.134 | 0.84–0.97 | sentence form (0.60) |
| 3e-3 `s-8ec00fb3` | 424 → **0.032** | 95 → **0.083** | 0.94–0.98 | sentence form (0.53) |
| 1e-2 `s-e9552de2` | 407 → **0.026** | 84 → 0.104 | 0.72–0.96 | sentence form (0.60) |
| random basis (norms of seed 0) | 322 → 0.095 | 32 → 0.516 | — | sentence form (0.70) |

**Reading** [concluded unless tagged]:
1. Every trained decomposition supports a held-out verdict swap with 60–100 of its 672 rank-1 terms, the random basis does not (at equal L1 penalty). Between the trained decompositions the differences are small (accuracy 0.08–0.13 at L1 0.01) and within what one fit per setting can resolve [assumed].
   OUTDATED (26-10-21): "the random basis does not" holds only for the data-blind random basis; the data-aligned PCA basis swaps about as well as the decompositions. See the entry *reprogramming: L1 sweep and the PCA-basis control*.
2. **The swap does not use the decomposition's active components specifically**: the 1e-2 run has only 8 components active on these statements, yet its 84-term edit is mostly made of components that are inactive there (largest: L17:15, L13:2, L15:24, none of its 8). So "swappable" measures the direction set U, V as a whole, not what the decomposition says the model uses on these inputs.
3. That raises a confound: trained V directions are aligned with where the MLP inputs have energy; random 18,944-d vectors barely touch the inputs. Control running: a data-aligned, untrained basis (v_c = top-96 principal directions of each layer's input x, u_c = W v_c; `mask_search_4e.py --pca_basis`). If it swaps as efficiently, the decompositions add nothing beyond "data-aligned directions" for this purpose.
4. Twin: every basis, random included, produces verdict by sentence form; no belief lever anywhere.

So for deciding "which decomposition does what", the reprogramming numbers do not separate the decompositions; the ablation comparison (entry *which decomposition does what*) does.

---

26-10-21, Claude Opus 5.5 — *round 3 complete: the importance-minimality front*; same session. `s-a55b67a4` = coefficient 3e-4 (LR 2e-3, all statement tokens, seed 0, 3,000 steps); analysis JSON in `~/spd_out/26-10-16_faithful_targeting/e4/r3/` [verified: one run per coefficient, val].

| coefficient | run | KL recovered: CI-weighted / rounded / Δ only | accuracy (CI-weighted) | active per statement | max \|C\|, \|Δx\| (/\|y\|) | min cos(A, I) |
|---|---|---|---|---|---|---|
| 3e-4 | `s-a55b67a4` | 0.903 / 0.833 / −0.019 | 0.946 | 105.2 | 0.45, 0.96 | −0.13 |
| 1e-3 | `s-eb1c20d5` / `s-52b2abc7` | 0.910 / 0.912 (CI-weighted) | 0.974 / 0.971 | 27.5 / 24.0 | 0.45–0.46, 0.96–0.97 | −0.23 / −0.04 |
| 3e-3 | `s-8ec00fb3` | 0.701 / 0.703 / −0.775 | 0.863 | 7.4 | 0.45, 0.96 | −0.02 |
| 1e-2 | `s-e9552de2` | 0.599 / 0.599 / 0.350 | 0.740 | 2.1 | 0.48, 0.96 | −0.03 |

Reading [concluded: one seed per coefficient except 1e-3]: by the objective Julian set (KL recovered by the CI-weighted components), **1e-3 is the best coefficient**: recovery plateaus at ~0.91 there, and 3e-4 adds 4× the active components without gain (and more half-on components: rounded 0.833 vs CI-weighted 0.903). No setting shows cancellation. With this, LR (2e-3) and the importance-minimality coefficient (1e-3) are tuned; the earlier proposed next parameters (orthogonality coefficient, C, routing, CI width) were not run. Ablations of 3e-4 are running.

---

26-10-21, Claude Opus 5.5 — *reprogramming: L1 sweep and the PCA-basis control*; same session. Invert objective only, L1 penalties 0.003 and 0.03 added to the earlier 0.001 and 0.01 (`e4/mask2/`); PCA basis: v_c = the 96 top principal directions of each layer's input x (uncentred, statement tokens of 512 train statements), u_c = W v_c, i.e. d_c scales the MLP's response to input direction v_c (`mask_search_4e.py --pca_basis`, `e4/mask3/`). Numbers: terms with |d| ≥ 0.1 → val accuracy of the sparsified edit (original 0.967; 0 = every verdict swapped) [verified: one fit per cell].

| basis | L1 0.001 | L1 0.003 | L1 0.01 | L1 0.03 |
|---|---|---|---|---|
| 1e-3 seed 0 `s-eb1c20d5` | 454 → 0.050 | 257 → 0.059 | 63 → 0.116 | 19 → 0.280 |
| 1e-3 seed 1 `s-52b2abc7` | 411 → 0.037 | 269 → 0.041 | 103 → 0.134 | 9 → 0.438 |
| 3e-3 `s-8ec00fb3` | 424 → 0.033 | 220 → 0.047 | 95 → 0.090 | 20 → 0.340 |
| 1e-2 `s-e9552de2` | 407 → 0.026 | 207 → 0.054 | 84 → 0.115 | 13 → 0.461 |
| **PCA basis (untrained, data-aligned)** | 234 → 0.045 | **129 → 0.057** | 47 → 0.158 | 11 → 0.575 |
| random basis | 322 → 0.094 | 150 → 0.124 | 32 → 0.542 | 3 → 0.693 |

Random-component controls (the edit's values on other components of the same basis): ≥ 0.90 in every PCA-basis cell, as for the decompositions. The PCA edits concentrate on L17 (largest terms L17:70, L17:85, L17:69, …).

**Reading** [concluded unless tagged; one fit per cell]:
1. **Swapping the verdict is not evidence for any decomposition.** An untrained basis aligned with the MLP inputs swaps it as efficiently as the trained decompositions: with ~130 terms better than they do with ~200–270, with ~50 about as well as they do with 60–100. Only at the smallest budgets (≤ 20 terms) are the trained bases ahead (0.28–0.46 vs 0.58), a small edge.
2. So the earlier "decomposition beats random basis" (entry *reprogramming: all four decompositions*, reading 1) was mostly data alignment: random 18,944-d read vectors barely touch the inputs.
3. **The reprogramming numbers do not separate the decompositions**; what separates them is the ablation comparison (entry *which decomposition does what*): coverage per family, label selectivity (3e-3: all false-side), and which components are individually necessary.
4. What the reprogramming result does say about the model [assumed: one prompt, val only]: a fixed rank-~50–130 edit of MLPs 11–17 at the statement tokens swaps the verdict on held-out statements of every family, with the answer still a digit. Whether that is a readout inversion or a change in what the statement is taken to say is untested (transfer to another verdict prompt and a recall prompt would tell).

---

C: (transcribed from chat) Can you give me a more detailed overview of swapping. Are there any numbers that tell untrainde basis and the decompositions apart. Also what do you mean by untrained basis or by random edits?

---

26-10-21, Claude Opus 5.5 — *swapping in detail*; same session. Script `~/spd_out/26-10-16_faithful_targeting/e4/swap_details.py` over the saved fits (`e4/mask*/mask_search_*.npz`) [verified: one fit per cell, val].

**What an edit is.** In each of MLPs 11–17, at the statement tokens only, the down-projection W becomes W + Σ_c d_c u_c v_cᵀ. Each term u_c v_cᵀ is a rank-1 matrix: it reads the MLP's hidden activation x along v_c and writes along u_c. A *basis* is the set of 672 pairs (u_c, v_c) (96 per layer). The search learns one number d_c per pair, the same for every statement, on the train split, to make the answer the opposite digit (with an L1 penalty that sets most d_c to 0, and a term that keeps the answer a digit). d_c = −1 removes the term's contribution, +1 doubles it. Evaluated on the 986 val statements.

**The three kinds of basis.**
- *Decomposition*: (u_c, v_c) = the trained components of a tPD run (4 runs).
- *Random basis* ("random edits"): every u_c, v_c replaced by a random Gaussian direction of the same length as the decomposition's. Untrained and blind to the data: a random 18,944-d read direction v_c sees almost nothing of the actual x.
- *PCA basis* (what I called "untrained basis"): v_c = the 96 directions in which x varies most over 512 train statements (principal components), u_c = W v_c, i.e. the MLP's own response to input direction v_c. So d_c = −1 removes the MLP's response to the c-th main input direction. No tPD training and no use of the verdict; only data statistics. "Untrained" meant "not trained by tPD", in contrast to the decompositions; "data-aligned but untrained" is the precise term.

**Per edit** (sparsified, |d| ≥ 0.1): terms, val accuracy (0.967 original, 0 = every verdict swapped), digit mass, share of Σ|d| on L17, and for decompositions the share of Σ|d| on components active (in ≥ 2% of statements) in that decomposition:

| basis | ≈ 10–20 terms | ≈ 50–100 terms | ≈ 130–270 terms |
|---|---|---|---|
| 1e-3 s0 | 19 → 0.280; on active 0.62 | 63 → 0.116; mass 0.980; on active 0.37 | 257 → 0.059 |
| 1e-3 s1 | 9 → 0.438; on active 0.89 | 103 → 0.134; mass 0.972; on active 0.60 | 269 → 0.041 |
| 3e-3 | 20 → 0.340; on active 0.86 | 95 → 0.090; mass 0.981; on active 0.38 | 220 → 0.047 |
| 1e-2 | 13 → 0.461; on active 0.11 | 84 → 0.115; mass 0.981; on active 0.10 | 207 → 0.054 |
| PCA | 11 → 0.575 | 47 → 0.158; mass 0.964 | 129 → 0.057; L17 share 0.47 |
| random | 3 → 0.693 | 32 → 0.542; mass 0.920 | 150 → 0.124 |

Share of statements answered opposite to the truth, per family (original: element_symb 0.21, others ≤ 0.14):

| edit | animal_class | cities | element_symb | larger_than | neg_animal | neg_cities | neg_element | neg_sp_en | smaller_than | sp_en_trans |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-3 s0, 63 terms | **0.53** | 0.85 | 0.25 | 0.96 | 0.93 | 0.94 | 0.86 | 0.88 | 0.96 | **0.38** |
| 3e-3, 95 | 0.20 | 0.89 | 0.36 | 0.96 | 0.87 | 0.97 | 1.00 | 1.00 | 0.96 | 0.55 |
| 1e-2, 84 | 0.40 | 0.81 | 0.18 | 0.99 | 0.53 | 0.93 | 0.96 | 0.97 | 0.98 | 0.42 |
| PCA, 47 | **0.07** | 0.89 | 0.36 | 0.91 | 0.87 | 0.95 | 0.79 | 0.60 | 0.94 | **0.07** |
| PCA, 129 | 0.67 | 0.99 | 0.39 | 0.98 | 1.00 | 0.97 | 0.82 | 0.88 | 1.00 | 0.68 |
| random, 150 | 0.33 | 0.95 | 0.32 | 0.97 | 0.33 | 0.94 | 0.71 | 0.90 | 0.97 | 0.05 |
| 1e-3 s0, 257 | 0.53 | 0.96 | 0.25 | 0.97 | 1.00 | 0.99 | 1.00 | 1.00 | 0.98 | 0.78 |

**Numbers that separate PCA basis and decompositions** [concluded unless tagged; one fit per cell, comparisons at unequal term counts]:
1. *Very small budgets*: with 9–20 terms the decompositions reach 0.28–0.46, the PCA basis 0.58 with 11. A modest edge.
2. *Which families swap at mid budget*: with 50–100 terms the easy families (comparisons, cities, negated) swap in every basis; the affirmative fact families separate them. 1e-3 s0 (63 terms) swaps 53% of animal_class and 38% of sp_en_trans, the PCA basis (47 terms) 7% and 7%. At ~130 terms the PCA basis catches up (0.67, 0.68). element_symb resists every edit (≤ 0.39; the original already errs on 21%).
3. *Where the edit sits*: the PCA edits put 44–47% of their weight on L17; the decompositions spread it evenly over L11–L17 (L11 0.08–0.21), the random basis avoids L11–L12. So the same behaviour is reached through different layers [concluded: from the layer shares; whether L11 edits act like A8's "false" subspace is untested].
4. *Side effects*: digit mass at mid budget 0.972–0.981 (decompositions) vs 0.964 (PCA) vs 0.920 (random): the decomposition edits disturb the answer format least, by a small margin.
5. *Consistency with the decomposition itself*: at small budgets the decompositions' edits are built mostly from their own active components (62–89% of Σ|d| for 1e-3 and 3e-3), at large budgets mostly from inactive ones (≤ 34%). The 1e-2 run is the exception (≤ 11% at every budget: its 8 active components are too few for the job).

Not separating: accuracy at ≥ 130 terms, the size of the inverted margin (separation −7 to −11 everywhere), the belief-shaped twin objective (sentence-form verdicts for every basis).

Overall reading: the decompositions are somewhat better swapping bases than a data-aligned one at small budgets and in the affirmative fact families, but the gap is small, measured with one fit per cell, and closes as the budget grows. Swapping is a weak tool for telling decompositions apart; the ablation comparison is the stronger one.

---

C: (transcribed from chat) d_c = -1 removes the terms contribution? Why is it not d_c = 0?

Answer (26-10-21, Claude Opus 5.5): clarification of the entry *swapping in detail*, whose wording "d_c = −1 removes the term's contribution" was ambiguous. The search multiplies each component by a mask 1 + d_c, and since W = Σ_c u_c v_cᵀ + Δ, every component's contribution u_c v_cᵀ is already inside W. So d_c = 0 is no edit (mask 1, weights stay W); d_c = −1 is mask 0, i.e. the added term −u_c v_cᵀ cancels the component's existing contribution and removes the component from W; d_c = +1 doubles it. "Removes the contribution" meant the component's contribution to the original weights; the edit term d_c u_c v_cᵀ itself is zero at d_c = 0. PCA basis: u_c v_cᵀ = W v_c v_cᵀ is the part of W acting on input direction v_c, so −1 removes the MLP's response to that direction. Random basis: there u_c v_cᵀ is not a meaningful part of W (Δ absorbs the difference), so −1 merely subtracts a random rank-1 matrix of that size.

---

26-10-21, Claude Opus 5.5 — *3e-4 ablations; pause*; same session. Ablations of `s-a55b67a4` (coefficient 3e-4; 1,523 conditions; `e4/r3/ablate_s-a55b67a4.{log,json}`) [verified: one run, val]: components-only accuracy 0.946 (KL recovered 0.903), frozen activations 0.559; 483 components active in ≥ 2% of statements, 248 label-selective (43 true-side); individually necessary (separation share ≤ 0.95): L12:3 0.88, L11:93 0.89, L11:22 0.90, L11:87 0.92, L14:53 0.93, L12:59 0.94, L12:30 0.95. Coverage per family like 1e-3 (neg_cities 0.87, neg_element_symb 0.68, element_symb 0.54 slightly lower). Reading [concluded]: the same function as at 1e-3, spread over 6× as many components, none individually important. The indices L11:22, L11:87 and L12:59 recur across the seed-0 runs (shared initialisation; their roles differ between runs).

C: (transcribed from chat) I'd want to pause after that. I need to understand the current results first.

Done before pausing [verified]: nothing else is queued or running on the instance; the 3e-4 checkpoint and all logs, scripts and analysis outputs of 26-10-21 are copied to `~/spd_out/26-10-16_faithful_targeting/e4/` (checkpoints of all six analysed 4e runs in `checkpoints/`, instance files in `instance_copy_26-10-21/`, 49 MB). Only the checkpoints of discarded tuning runs remain solely on the instance (`s-9d172777` LR 5e-3, round-1 runs other than `s-348832ad` / `s-eb1c20d5`, the trials), which the checkpoint policy lets die with it. The instance is still rented; stopping it is Julian's call. The SUMMARY now opens with an overview of where 4e stands.

---

C: (transcribed from chat) I need a summary of the latest experiments (the orthogonal penalty ones). It should focus on them but also include important comparisons to the other experiments (I think the experiments number 4)

---

26-10-21, Claude Opus 5.5 — *report on 4d/4e with comparisons to 4a–4c*; new session. I read INDEX.md, this topic's SUMMARY, the LOG entries from *4e: follow-up questions and draft plan* to the end, the Exp 4 section of the bolder-experiments SUMMARY, the 4c mask-search part of the component-steering SUMMARY, `reports/26-10-16_exp4_statement_first.typ` (4a–4c tables) and the 4c vs 4e configs.

Wrote `reports/26-10-21_orthogonality_experiments.typ` (7 pages, self-contained, PDF gitignored; compiled with `typst compile`). Contents: background on 4a–4c and the problem; 4d; 4e tuning (LR × positions, importance-minimality coefficient); a comparison table 4a–4e; what 4e's components carry (frozen activations, "false" against a "true" default, per-layer states, A8); the sparsity front by family; reprogramming incl. the 4c mask-search comparison; caveats and open items. All numbers re-read from this LOG, the SUMMARYs and the 26-10-16 report.

Points made in the report that are not spelled out elsewhere [concluded unless tagged]:
- **The 4c-vs-4e comparison (74% vs 91% KL recovered) is not controlled.** Besides the loss term, 4e's best run differs from 4c in LR (2e-3 vs 5e-4), steps (3,000 vs 5,000), PGD (off vs on), training split (tiu v2 train, 5,574 vs v1 train, 6,560) and evaluation split (val vs test); its rounded number (0.87) is the one comparable in kind to 4c's 74%. 4e at 4c's LR recovered 0.37 [verified: `s-348832ad`, round 1], and 4c was never run at 2e-3, so the gain cannot be attributed to the orthogonality term alone. A controlled comparison (4c at LR 2e-3 on the same split) is listed as an option [assumed: proposal].
- **Size of the components' write**: 4e's active components write at most 0.16 |y| (`s-eb1c20d5`), 4b's 0.81–0.96 |y|, 4c's 0.29–0.66 |y|: 4e is the only non-cancelling Exp 4 decomposition whose components are a small part of the output [verified: numbers from the cited analyses; comparison across splits].
- **4c's mask search vs 4e's**: 4c inverted cities, comparisons and three negated families and collapsed animal_class, element_symb, sp_en_trans; 4e's edits also swap most of animal_class and sp_en_trans, but element_symb resists every edit (≤ 0.39 answered against the truth) [verified: entry *swapping in detail*]. The SUMMARY's "swaps the verdict ... of every family" holds for family groups (affirmative / negated / comparisons), not for element_symb.
