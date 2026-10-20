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
- It is **stronger and cleaner than the hinge the agent proposed on 26-10-16** (bolder-experiments LOG, *options for the idle H100*: penalise only negative A·(y − A)). The hinge allows A anywhere inside the Thales ball, so a small A that is almost orthogonal to y (a foreign write) passes. Prefer Julian's equality version.
- **Per-component version (my recommendation):** require y − A ⟂ each active component's write direction u_c (penalty Σ_c CI_c ⟨u_c, y − A⟩² / (|u_c|² |y|²)). That says A is the least-squares fit of y within the span of the active components' write vectors, i.e. "where you write, write exactly what the MLP writes; where you don't, no claim". With one direction it reduces to Julian's condition; with several it forbids the case where two components overshoot in opposite directions and still have a sum that satisfies the single condition. Same cost: one dot product per active component per position.
- Open details: (a) which state to apply it to: the CI-weighted state is the simplest; applying it under the stochastic masks would also constrain the inactive components and drift towards 4b; (b) the metric: Euclidean in the raw residual basis lets the massive-activation channels dominate (one channel held 22–29% of the squared norm at the period, 26-10-10). Per-channel std normalisation as in Exp 1's residual lens is the obvious choice [assumed]. (c) A = 0 satisfies it trivially, so only the verdict loss pushes components on; the coefficient trades verdict recovery against faithfulness.
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

F4. Validation split: 6,560 train / 1,794 test statements over the 20 tiu datasets. [Default: hold out 15% of each dataset's train split (seeded), about 980 statements, as `val`. Training uses the remaining ~5,580, test stays untouched until the final configurations. Implemented as a new dataset version `v2` (train/val/test) through the tiu generator, so provenance stays recorded.] Ok, or would you rather tune on part of the test split?
C: I don't care. You can choose

F5. Screening runs: 3,000 steps with PGD off (your suggestion), objective (a), analysed on the validation split. Full runs: 5,000 steps with PGD as in 4a (coefficient 0.5 from 80% of training). [Default: yes. Turning PGD off changes the training, not the objective, so a screening run ranks configurations but does not predict the full run's numbers [assumed].]
C: yes

**Draft plan** [assumed: proposal; adapted as results come in]
1. Code (local, CPU tests): global-term option in `ComponentOrthogonalityLossConfig`; the analysis script moved into the repo with ⟨C, Δx⟩/|y|², |C|/|y| and cos(A, I) added, plus a `--split val` option; checked against 4a's stored numbers; validation split (F4).
2. On the H100: one short trial (300 steps) per position setting, to see the start-up scale of the term and the step time. Then check whether two runs in parallel give more total throughput (question 10).
3. Screen round 1: orthogonality coefficient ∈ {0.01, 0.1, 1, 10} × positions {end token, all statement tokens} = 8 runs at 3,000 steps without PGD. Report after each run: verdict recovery (a) and (b), L0, |A|, |I|, |C|, |Δx| relative to |y|, cos(I, Δ), cos(A, I), ⟨C, Δx⟩/|y|², all per layer.
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
