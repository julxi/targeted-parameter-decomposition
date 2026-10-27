I want to create a reasearch artifact of some of the results that we have made so far. The story is not quite clear so far and will need some extra experiments or redoing of some experiments.

I'll want to concolidate the experiments that are used for this (so they should become registered in the end, but not before they are finalised).

The working title is "Trying to decompose a truth mechanism with tPD" and that's what we do.

On a high level we should evaluate them according to these categories (adapted  to mechanistic decomposition):
1. Correlation: Experiments conducted under this category aim to pinpoint neural correlates.
Reading techniques like LAT only provide evidence of a correlation between specific neural activity
and the target concepts or functions. The strength and generalization of this observed correlation
can be assessed via prediction accuracy across both in-distribution and out-of-distribution data. In
order to conclude causal or stronger effects, the following categories of experiments should be
considered.
2. Manipulation: Experiments conducted under this category are designed to establish causal relationships. They necessitate demonstrating the effects of stimulating or suppressing the identified
neural activity compared to a baseline condition.
3. Termination: Experiments conducted under this category seek to reveal the necessity of the
identified neural activity. To do so, one would remove this neural activity and measure the
resultant performance degradation, akin to Lesion Studies commonly performed in neuroscience.
4. Recovery: Experiments conducted under this category aim to demonstrate the sufficiency of
the identified neural activity. Researchers perform a complete removal of the target concepts or
functions and then reintroduce the identified neural activity to assess the subsequent recovery in
performance, similar to the principles behind Rescue Experiments typically carried out in genetics.

For the out-of-distribution data we should also create some extra data similar to tiu:
- factual recall: we need a new category at is equally well curated as the tiu dataset. We already have animals, cities, elements. We need another class of facts for which we can generate enough unambigious data.
- arithmetic. We could do addition like eleven plus twelve is twentythree
- language (might be the same as factual recall maybe): like spanish but with german

We should also make sure that the the datasets in the domains are balanced since we sometimes do analysis on singular domains.

Now let's go to the setup: We compare different tPD methods on "[stimulus]. The amount of truth is " (the exact sentence might be a bit different). The decomposition is on the stimulus tokens while the loss it taken on the next token after the sentence. We decompose the down projections of some layers (11-17 I think). We have to report why we made these choices (they are briefly mentioned in the main text and elaborated on in the appendix):
- the stimulus-lens: because it hopefully condences the internal believe of the model into the answer (a single digit called the verdict). The surfacing of internal believe is an assumption of course.
- why we decompose on all stimulus tokens and not just on the last token of the stimulus (this was another hypothesis that this stores the verdict). We have some mlp-ablations experiments that show that ablating only the last does shrink the verdict separation only a little
- why the chosen layers?: we haves some verdict affecting ablation experiments for that as well
- why the down projections? I vaguely remember that we had another ablation experiment for that that shows that the attention also work worse.

The methods that we compare are:
- plain tPD
- tPd + layer faithfulness
- tPd + orthogonality

We try to see which ones of these do the best job of recovering a truth mechanism. I suspect that the true answer will be, none of them, but at least some parts of a truth mechanism can be uncovered. Plain tPD will be evidently worse than the others, we have seen that before with the giant orthogonal activations. We don't have a great answer what a metric for deciding this should be. So let's discuss this here a bit:
- first we need to finetune these methods. Here we wan't to avoid computational expensive analyses. I suggest some undefied pareto optimum of:
    - the components capture the computation on target-data: kl of components only an target data
    - the components don't matter on non-target data: kl of delta only on non-target data ()
    - shrink of verdict separation of delta only on target data
- the most promising candidates should then be analysed further by the four categories above. Exactly how these should look like we have to decide.

Let's talk about the results:
- What the main result is, is still open. I think it will be, tPD doesn't work well. However, I'm not use yet. Maybe we can show a plausable partial decomposition of a truth mechanism. In either case we also should find or hypothesise a plausable reason for why this doesn't work.
- We have one for pure tPD: the components have too much freedom and can do what they want. This might be a general problem for tPD which doesn't require the parameters to be faithful.
- Another finding that might affect the PD decomposition methods in general is that in our decompositions there are a lot of components with low CI, which still are important together. We often have performance degrade quite a bit if we mask all components by their CI values. I think this is due to the stochastic sampling of the mask which gives an averaging effect. What we should look into is how this also persists after the adveserial training.

Before we write the paper, I want to create a living report like document that gets extended during the writing process. Focus is getting the right results and story not the writing itself.

There's quite a lot above that neends clarification and discussions, so let's do that. And if you think I've forgotton to mention something relevant speak up.

---

26-10-22, Claude Opus 5.5 — *first pass: what exists, what each claim rests on, open questions*

Context: I have read INDEX.md, the SUMMARYs of the bolder-experiments, faithful-targeting, component-steering and prepared-datasets topics, FUTURE_WORK.md, the stochastic-mask and PGD mask code (`spd/utils/component_utils.py:36`, `spd/persistent_pgd.py:365`), `spd/registry.py`, and the dataset list of the upstream Truth-is-Universal repository at the commit our tiu generator pins. I have not read the LOGs. I renamed your file from `tying_everything_together.md` to this name; your text above is unchanged.

Shorthand used below: **tiu** = our Truth-is-Universal statement data (`data/tiu/`, 20 datasets = 10 families × true/false). **Verdict prompt** = `<statement>\nThe amount of truth in this statement is `, verdict score = log P('1') − log P('0'). **4a / 4c / 4e** = the three existing decompositions of the `down_proj` of Qwen2.5-7B-Instruct layers 11–17 with the verdict digit's KL as target loss: 4a = plain tPD (end token only), 4c = + per-layer reconstruction term, all statement tokens, 4e = + global orthogonality term ⟨C, Δx⟩ = 0, all statement tokens. **KL recovered** = share of the verdict KL lost by switching MLPs 11–17 off that the components alone (Δ off) restore. **Cancellation test** = relative error |y − A| / |y| of the components' write A against the real MLP output y (> 1: the components write something the rest of the weights must undo).

## 1. The three methods are not comparable yet

| | plain tPD (4a) | + layer faithfulness (4c) | + orthogonality (4e best) |
|---|---|---|---|
| positions | end token `'.\n'` only | all statement tokens | all statement tokens |
| LR | 5e-4 | 5e-4 | 2e-3 |
| steps / PGD | 5,000 / on | 5,000 / on | 3,000 / off |
| data | tiu v1, test split for analysis | v1, test | v2, tuned and analysed on val |
| tuned? | no | no (reconstruction coefficient 1, never varied) | LR, importance-minimality coefficient |
| KL recovered | 0.94, through cancellation (layer-11 write 12.9× \|y\| at cosine 0.03) | 0.74, no cancellation | 0.91 (two seeds), no cancellation |

So "4e beats 4c" is not established [concluded: 4e at 4c's LR 5e-4 recovered only 0.37, so LR alone could explain the gap; the 26-10-21 report already says this]. Plain tPD was never run at all statement tokens. Your plan to redo everything under one protocol is needed, not optional. Concretely, one protocol = same positions (all statement tokens), same data version and splits, same step count and PGD setting, and **the same tuning budget per method** (otherwise the method we tuned most wins).

## 2. Setup choices: what each one rests on

- **Verdict lens** (the KL of the verdict digit as target loss). Evidence: (i) Exp 0, the natural next-token lens at the statement's period, recovered 8% of the all-off verdict KL vs 63–72% for lenses that read a verdict (Exp 0–3 used layers 15–19 and an older appended-question prompt, so this is indirect); (ii) on the verdict prompt the model is accurate (0.963 accuracy, AUC 0.994 on the 546-statement screen) and steering with the truth direction at the statement tokens (layers ~13–17) moves the digit. The assumption "the verdict surfaces the model's belief" is not just an assumption we can state; we have evidence *against* a strong form of it: in the steering topic's recall test, masks that invert 4c's verdict leave direct-question recall unchanged (cities 0.771 or higher), and after an inverted digit the model mostly states the true fact [verified there, one seed]. So the verdict can be changed without the belief. I'd state the lens as "decomposes the computation of the verdict", with "belief" as the hypothesis the analyses test.  
C: You are right. I don't think that we can fundamentally change the models believe. However, we should add an experiment for checking how fundamentally we can check the models believe by activation steering. Also maybe we can find out something about why the model recovers (e.g. from where does it get that information)  
Answer (26-10-22, Claude Opus 5.5): Agreed; added to the plan (entry *plan of steps*, step B3). Design [assumed: proposal]: steer with the in-context truth direction at the statement tokens, layers 13–17 (the window where steering worked on the verdict prompt), and read four outcomes: the verdict digit, the digit under a second verdict prompt, the direct recall question ("In which country is the city of X?", as in the steering topic's recall test) and the free continuation. OUTDATED (26-10-26): B3 contradicts this hint for steering at all statement tokens. After layer-17 steering the model states the flipped digit, then restates the true fact, as after mask inversion. See entry *B3 results*. A hint that steering may reach deeper than the component masks: in the statement window, steering made the model's explanation follow the steered digit, while after a mask-inverted digit the model mostly restates the true fact [verified in the bolder-experiments and component-steering topics; a few example statements each, anecdotal]. On where the model gets the information back after a mask edit at the statement tokens of layers 11–17, two candidate routes: (i) attention at the answer tokens re-reads the statement in layers outside 11–17; (ii) layers 18+ at the statement tokens recompute it. Test: widen the edit (more layers, then the answer positions too) and see where the true fact stops coming back.
C: I have another comment about mass mean and logistic regression truth directions. We should say why we use mass-mean. My reasons are:
  - they are not very different (has to be verified)
  - they are faster to compute
Answer (26-10-22, Claude Opus 5.5): Both reasons are fine, and (1) is cheap to check from stored residuals: cosine between the two directions per layer, plus how well each steers (plan, step A4). A third reason, if it holds up: Marks & Tegmark (2023, *The Geometry of Truth*) report that mass-mean directions are more causally implicated than logistic-regression directions when used for interventions [assumed: from memory; check before citing]. In the paper the choice matters only where truth directions are used: the steering baseline above and the truth-gap shares of the ablations.
- **All statement tokens, not just the last.** Your memory is right: in the causal layer scan (mean ablation of MLPs 11–17 on the verdict prompt, full test split), ablating only at the end token keeps 82% of the verdict separation, because the object token carries it too [verified there]. 4b (end token only) vs 4c is a second, weaker argument (4b's single components barely beat random sets in the steering topic).
- **Layers 11–17.** Same scan: no single MLP layer from 5 to 24 is necessary (each keeps ≥ 0.77 of the separation); the block 11–17 removes the judgement (separation share 0.05, accuracy 0.541, digit mass still 0.977); blocks 11–14 and 15–19 each keep about a third (0.38 / 0.34), 6–10 keeps 0.18 [verified there]. Caveat for the appendix: only these few blocks were tried, so 11–17 is "sufficient to remove the verdict", not "minimal".
- **`down_proj`: I found no experiment for this.** The only attention-vs-MLP evidence in the records is a direct-write attribution on *bare* statements at the period (not the verdict prompt): attention writes 0.14–0.20 of the truth gap along the layer-15/17/19 truth directions, MLPs most of the rest [verified: bolder-experiments SUMMARY, *Where the truth direction is written*]. That is not a causal ablation, and it says nothing about `down_proj` vs `gate_proj`/`up_proj`. The `down_proj` choice was inherited from arm A, the first tiu decomposition [assumed: "carried over" in the bolder-experiments design]. If I missed an experiment, please point me to it. Otherwise two things:
  1. A cheap causal check: the layer-scan script with attention outputs instead of MLP outputs mean-ablated at the statement tokens (block 11–17 and single layers). A few minutes of GPU, no training.
  2. A limitation to state in any case: decomposing only `down_proj` decomposes how the MLP's hidden activations are *read out*; the computation of those hidden activations (gate, up, the nonlinearity) runs on the original weights. So even a perfect decomposition shows which hidden directions carry the verdict, not how the MLP computes them. That matters for the word "mechanism" in the title.  
C: You are right. We don't have such an experiment. I meant the truth direction experiment, but that is something else. Actually, most likely the attention layers are equally important. I think the idea to take down projections came from the `tpd_paper.md`. I believe they used also the down projections only. If not then we have to think about the implications of only using the down projections.
Answer (26-10-22, Claude Opus 5.5): The tPD paper does not decompose down projections only. Its numpy/pandas decompositions allocate components to `c_fc`, `down_proj`, `o_proj`, `q_proj`, `k_proj` and `v_proj`, the CSS decomposition likewise [verified: `papers/Targeted_Recovery_of_Weight_Space_Mechanisms/tpd_paper.md`, appendix *Hyperparameters and data preparation*]. Our choice goes back to arm A's config, whose comment reads: "Narrowed for memory: decomposing all 28 layers x 7 projections OOMs on the dense weight-delta tensors … Layers 15-19 down_proj are an unvalidated guess." (`spd/experiments/lm/honesty_targeted_decomposition/config_truth_all_tokens.yaml`, lines 18–20). Implications: (1) whatever we find is a result about a restricted tPD, not the paper's method. (2) If the attention scan shows that attention in 11–17 carries a comparable share of the verdict, the cheapest extension is `o_proj`: 3,584 × 3,584 = 12.8M weights per layer, about 19% of a `down_proj` (18,944 × 3,584 = 67.9M) [verified: the model's `config.json`]. That treats the attention write like the MLP write. (3) The "readout, not computation" limitation is only addressed by also decomposing q/k/v or gate/up; gate and up are each as large as `down_proj`, a much bigger memory step. Decide after the scan.
- **Terminology.** "Stimulus" has meant two things in this project: Exp 2/3's appended question ("stimulus lens"), and later the statement itself (your use in the 4e restart). For the paper I suggest: *statement*, *verdict prompt*, *verdict lens*.  
C: yes, that's good. Additionally officially it should be stimulus = statement if we ever use stimulus.
Answer (26-10-22, Claude Opus 5.5): Noted [decided: C: comment]: *statement*, *verdict prompt*, *verdict lens*; "stimulus", if used at all, means the statement.

## 3. The tuning objective: plain tPD would win it

OUTDATED (26-10-22): two claims in the next paragraph and in the last bullet of this section are wrong: Δ does carry verdict in 4a, and axis (b) does separate 4a from 4b. Re-checked numbers in the answer below the C: comment that follows the next paragraph.
Your three axes: (a) KL of the components alone on target data, (b) KL of Δ alone on non-target data (Pile), (c) shrink of the verdict separation with Δ alone on target data. Plain tPD (4a) is the best known run on (a) (KL recovered 0.94), and by construction Δ then carries no verdict, so it also does well on (c). The thing that makes plain tPD "evidently worse" is the cancellation, and none of the three axes sees it [concluded: 4a's numbers above]. So I propose a fourth axis, used as a constraint rather than traded off: cancellation-test error |y − A| / |y| < 1 per layer (or the 4e metric ⟨C, Δx⟩ / |y|² near 0 and |C| / |y| ≤ 1). A run that fails it is excluded, whatever its other scores.  
C: I'm not so sure about this. Obviously the huge cancelations of 4a show that it's doing something else. So that's why we came up with the cancelation stuff. But ideally it should be a training loss, not an evaluation metric. So the question is, is there some other natural criterion by which 4a can be excluded.
Answer (26-10-22, Claude Opus 5.5): Fair point: the cancellation test is close to what 4e is trained on (⟨C, Δx⟩), so selecting methods by it favours 4e by construction. I re-read 4a's numbers. Criteria that don't measure cancellation already separate 4a from 4b [verified: `convos/julian/26-10-10_bolder_experiments_LOG.md`, entry *Exp 4a finished*; one seed each, end token only]:
- **Adversarial (PGD) reconstruction loss** at step 5,000: 4a 1.20, 4b 0.13. PGD is part of tPD's own objective in the paper.
- **Pile (non-target) KL**: 4a 0.083, 4b 0.021. This is your axis (b), which I wrongly expected to decide little.
OUTDATED (26-10-26): the next bullet implies axis (c) separates 4a from 4b. It doesn't: 4b's Δ alone also keeps much of the verdict (0.44 of the all-off KL recovered vs 4a's 0.53). See entry *phase A without A1*, section A3.
- **Your axis (c), measured against all off**: Δ alone keeps half of 4a's verdict (KL to the original 0.091 vs 0.193 with all off; separation share 0.85 vs 0.73). The original mechanism stays in Δ while the components re-implement the verdict on top. In 4e, Δ alone sits at the all-off separation (both seeds). My earlier "by construction Δ then carries no verdict" was wrong.

Caveat: 4b differs from 4a by the reconstruction term, which may improve robustness on its own, and a tuned plain tPD at all tokens may do better. Proposal: select by tPD's own losses (target KL, Pile, PGD), your three axes and transfer to a second verdict prompt; report the cancellation test as a diagnostic that explains *why* a method fails, not as a selection criterion. Plan step A3 first computes these numbers for the existing 4a/4c/4e runs.

Further points on the axes:
- **Which component state?** 4e switched from rounded CIs (mask 1 if CI > 0.01) to CI-weighted masks. These differ by about 2× in the components' write (4d trial: 30% of active CIs lie strictly between 0.01 and 0.99). Pick one for all methods; it interacts with your low-CI point (section 6).  
C: I'd go wtih CI-weighted. Active CIs are just a random cutoff point.
Answer (26-10-22, Claude Opus 5.5): Noted [decided: C: comment]: all metrics in the CI-weighted state (mask = CI).
- **Axis (c) needs a reference**: Δ alone should look like "all off", not like 0. In 4e seed 0, Δ alone was worse than all off (a calibration effect). I'd define it as Δ alone's share of the original-minus-all-off separation (ideal 0).  
C: Ok, so your idea is that Δ shouldn't change the verdict spread. My idea was that Δ shouldn't contain any verdict relevant mechanisms. Which one is better now?
Answer (26-10-22, Claude Opus 5.5): Yours is the goal; mine measures one half of it. "Δ contains no verdict-relevant mechanism" has two measurable sides: (i) Δ alone does nothing for the verdict (Δ only ≈ all off; my version); (ii) Δ is not needed once the components are there (components only ≈ original; your axis (a)). A mechanism that needs components and Δ together, like 4e seed 0's always-on components opposing a "true"-pushing part of Δ, shows in neither and only appears in removal tests from the full model (termination). Proposal: measure (i) as Δ only's share of the all-off KL recovered (ideal 0; KL rather than separation, so a Δ-only state that breaks the output also counts), take (ii) from axis (a), and report the joint case from the termination analysis. As the answer above shows, (i) is what flags 4a.
OUTDATED (26-10-26): "(i) is what flags 4a" is wrong: (i) flags 4b at the same end-token position almost as much (0.44 vs 0.53). See entry *phase A without A1*, section A3.
- **Tuning on (a) makes in-distribution recovery circular.** Recovery is one of your four categories, and (a) is recovery on the tuning distribution. So in the paper the recovery evidence has to come from the test split and the OOD sets, not from the tuned number.
- (b) is already part of tPD's training loss; it should be cheap to report but will rarely decide anything [assumed].

## 4. The four categories

These are the evaluation categories of Zou et al. (2023), *Representation Engineering* [assumed: from memory; check the citation].  
C: yes it is  
Answer (26-10-22, Claude Opus 5.5): Noted; citation confirmed.
Mapped to what we have, and the control each needs. The main lesson of 4e is that **controls must include an untrained basis aligned with the data**: in the reprogramming test, an untrained basis built from the MLP inputs' principal directions swapped the verdict as efficiently as the trained components (129 terms → accuracy 0.057), while a data-blind random basis was clearly worse [verified there]. Random-component controls alone overstate the decompositions.

1. **Correlation**: label selectivity of components and CI probes, in- and out-of-distribution. Caveat from arm A: an untrained CI network already reads truth at 0.998, so probe accuracy alone means nothing; selectivity must be compared with the untrained decomposition and must control for sentence form (negation, comparison) and for object-word shortcuts (in larger_than and animal_class the object word alone predicts the label at 0.70–0.75). Also: correlate with the model's own verdict, not only the ground-truth label (Exp 3's "True" writer 447 followed the model's answer).  
C: So CI-probes would be Correlation, but we can't use them because an untrained CI gives equally good probes? That doesn't give us evidence. We have to find something better here.
Answer (26-10-22, Claude Opus 5.5): Right: the CI network is a function of the activations, so a probe over all CIs measures what the activations already contain; that is why the untrained one reads truth too [concluded: arm A results]. Correlation evidence must be something an untrained decomposition or a data-aligned basis can't produce just by reading the activations. Candidates [assumed: proposals, not fixed in advance]:
1. **Sparsity**: how few components carry the correlation. Single-component AUC, and probe accuracy with k = 1, 2, 5 components, against the untrained decomposition and the data-aligned basis at the same k. With arm A (next-token lens) this gave only 1–2 points, but the verdict-lens runs have clearly label-selective single components (e.g. the 40 false-only components of 4e at importance-minimality 3e-3).
2. **Transfer**: a component selective for "false" on tiu's negated cities that is also selective, with the same sign, on neg_inventors, the German statements and conj/disj. An untrained or random basis has no reason to transfer family by family.
3. **Truth vs verdict**: on the statements the model gets wrong (mostly true negated ones), does the component follow the label or the model's verdict?

All three are claims about single components, which is what a decomposition claims; a probe over all CIs is a claim about the activations.
2. **Manipulation**: reversal/amplification and the mask search. Controls: data-aligned untrained basis; and a belief readout (recall question, a second verdict prompt) to separate "changes the verdict" from "changes the belief".  
C: Ok, let's see what the numbers say again. This might give weak evidence, as the decompositions steers better with smaller numbers of components. Other verdict prompt is a very good idea. This might actually help to eliminate 4a. Let's see.
Answer (26-10-22, Claude Opus 5.5): Agreed. The few-component regime (≤ 20 terms, affirmative fact families) is the one place where the decompositions beat the data-aligned basis, so the report should show the whole curve over the number of terms, not one point.
3. **Termination**: removal from the full model, with size-matched random and data-aligned controls; plus A8-style removal of the components' write subspace from the *original* MLP output (no decomposition involved). A8 is our strongest positive result so far: at layer 11, in both 4e seeds, it flips 22–28% of false statements to "true", random subspaces of the same size do nothing [verified there, val split].
4. **Recovery**: components alone (KL recovered), on test and OOD (see section 3).

Missing before any of this: **what counts as a truth mechanism**. Without a definition, "none of them recoverone" is unfalsifiable and "a partial one" is unbounded.  
C: I don't have a definiton. A mechanism should be like an internal noisy algorithm that we uncover. So yes, we should work with heuristics and be clear about.  
A proposal [assumed]: a set of components whose activity (1) separates true from false within sentence forms and families, including OOD families, (2) changes the verdict when manipulated, in a fact-dependent way that transfers to a second verdict prompt, (3) is necessary and (4) sufficient for the verdict, each better than the data-aligned untrained basis. "Partial" = this holds for a subset of families or one sentence form (e.g. negation handling). What the records currently show is closer to "verdict readout gated by sentence form" (4e: the components supply a "false" signal against a "true" default; the individually important components are mostly negation/comparison detectors) [concluded there].  
C: I don't want a fixed heuristic definion of mechanism. What you just listed are heuristic criteria that we can use to support our claim that we have discovered _something_. We should stay honest that we don't really understand what's going on actually. Also I think what you proposed is reasonable (though for (1) we need something else as said above), but let's not fix any of that in advance.
Answer (26-10-22, Claude Opus 5.5): Understood [decided: C: comments above]: no fixed definition. The criteria are heuristics supporting the claim that we found *something*, stated as such and not fixed in advance; the report says plainly that we don't understand the mechanism.


## 5. Data

**The upstream Truth-is-Universal repository has families we never used** [verified: GitHub API listing and row counts at commit `605ef005`, 26-10-22]:

| file | statements (true) | example |
|---|---|---|
| inventors / neg_inventors | 406 (229) / 406 (177) | "Edwin Herbert Hall lived in the U.S." |
| facts / neg_facts | 561 (279) / 561 (282) | "The human body has more than 600 muscles." |
| German versions `*_de` (cities, neg_cities, sp_en_trans, inventors, …) | 50 each | "Die Stadt Krasnodar liegt in Russland." |
| conjunctions / disjunctions (e.g. cities_conj, cities_disj) | 1,498 / 500 | "It is the case both that the city of … and that …" |

Also `common_claim_true_false` and `counterfact_true_false` (larger, less curated). So:
- **Factual recall**: inventors/neg_inventors is a curated family with affirmative and negated forms, the same structure as cities. It is the natural first candidate before generating a new one. Its labels need a check that Qwen knows the facts (inventors are less famous than cities). facts/neg_facts is heterogeneous (science claims, quantities), less clean but a useful harder OOD set.  
C: fantastic let's use these
Answer (26-10-22, Claude Opus 5.5): Noted [decided: C: comment]: inventors/neg_inventors and facts/neg_facts as OOD data.
- **Language**: your "like Spanish but with German" reads to me as a German–English translation family ("The German word 'Hund' means 'dog'."), which is lexical recall like sp_en_trans. The `*_de` files are something different: the same statements written in German, an OOD test of the language of the statement. Only 50 per file, so test-only.  
C: that's actually an even better OOD distribution
Answer (26-10-22, Claude Opus 5.5): Noted [decided: C: comment]: the German `*_de` files as an OOD test (test only, 50 statements each).
- **Arithmetic** ("Eleven plus twelve is twenty-three."): close to larger_than/smaller_than, which also use number words, so a near-OOD test of the comparison machinery. First check the model's verdict accuracy on it (it may be poor at spelled-out sums), and decide how false sums are made (off by one vs random; this sets the difficulty).  
C: let's put this on the back shelf. Let's see what we can get from the already existing data.
Answer (26-10-22, Claude Opus 5.5): Noted [decided: C: comment and your answer to question 6]: arithmetic shelved, no generated families; the unused upstream data is the OOD set.
- **Every new family should come with negated forms** [concluded: the components found so far are mostly negation and comparison detectors, so only affirmative/negated pairs can separate truth from sentence form OOD].  
- **Conjunctions/disjunctions** are an OOD test of composition, possibly the strongest test of "truth" vs "sentence form" we have available for free.  
- A second OOD axis besides data: a different verdict prompt. A decomposition of "the amount of truth … is" might capture that readout specifically.



**Balance** (tiu v2 train, statements): larger_than + smaller_than 2,636 of 5,574 (47%), cities + neg_cities 2,016 (36%), the other six families 922 (17%; animal_class ≈ 56 per label). Test cells are as small as 18 statements (animal_class_true). Two separate problems: (1) the training mix shapes which families the decomposition covers (the steering topic suspected this for the families the mask search could invert [assumed there]); (2) per-family analyses on 18–28 statements per cell are noisy. The small families can't grow from upstream (animal_class has ≈ 56 per label, elements are bounded at 118). Options: cap every family at a common size (losing most cities/comparison data), or keep the data and weight the target loss per family. Both change training, which you are redoing anyway.  
C: I would just trim them all to a similar size for training. Test can stay bigger for some classes, so they are not that noisy.
Answer (26-10-22, Claude Opus 5.5): Noted; pros and cons with numbers under question 5 below, since you asked there.

## 6. Your low-CI point, made precise

The code supports your averaging explanation [concluded: from the code]: during training each component's mask is m = ci + (1 − ci)·u with u uniform on [0, 1], drawn independently per component (`spd/utils/component_utils.py:36`). A component at CI ≈ 0 therefore contributes on average half its write. Summed over hundreds of low-CI components, those random contributions concentrate around their mean (law of large numbers), so training practically never sees *all* of them off together, and the CI-weighted (or rounded) evaluation state is exactly that unseen state.

PGD's mask is m = ci + (1 − ci)·adv (`spd/persistent_pgd.py:365`), and the adversary may set all low-CI components to 0 jointly. So PGD should penalise this reliance [assumed: prediction, not tested]. The 4e screening runs had PGD off, so they can't answer it; the final runs with PGD can. Measurement proposal: per run, KL recovered under rounded, CI-weighted, mean-mask (ci + (1 − ci)/2) and stochastic masks, and the summed write of components with CI < 0.5 relative to |y|. If it holds with PGD off and disappears with PGD on, it is a general (small) PD finding; if it persists, a stronger one. Both need the PGD-on/off pair under otherwise equal settings.  
C: No extra experiment here. Just report this as a thery if we keep seing low-CI component-clusters that have big influence on our target metrics.
Answer (26-10-22, Claude Opus 5.5): Noted [decided: C: comment]: no extra experiment; reported as a hypothesis if low-CI clusters keep mattering.

## 7. What I think is missing from your list

- **Seeds.** One seed per setting except 4e at importance-minimality 1e-3. All runs so far use seed 0 and thus identical initial weights; the shared L17:5 between 4c and 4e came from that [concluded there]. Final candidates need ≥ 2 seeds, preferably different from the tuning seed. The seed policy is still pending in the faithful-targeting topic.  
C: yes, of course. Once we have created the experiments, we'll do this.
Answer (26-10-22, Claude Opus 5.5): Noted: seeds come with the final runs (plan, phase D).
- **What's in Δ.** In 4c, Δ holds 73% of the direct truth write [concluded there: direct writes]. A paper about decomposing a truth mechanism has to say what the part outside the components does.  
C: you mean more qualitative analysis? I don't know. I'd say something like "4c seems to keep most functionality within Δ and thus fails on this heuristic" is enough for me.
Answer (26-10-22, Claude Opus 5.5): No new analysis meant; that sentence in the report is enough.
- **Ground truth vs the model's judgement.** The model errs on true negated statements. The decomposition targets the model's verdict, so evaluation should use both labels.  
C: good point
Answer (26-10-22, Claude Opus 5.5): Noted: every analysis reports both the ground-truth label and the model's own verdict.
- **Fix the analysis plan before the final runs.** With many analyses and few seeds, deciding the metrics after seeing results invites forking paths. Freezing the tuning objective and the four-category analyses before the final runs would make "registered" meaningful. 
C: Yes, we should do this. But fixing that and the final runs comes at the very end. We don't put the cart before the horse
Answer (26-10-22, Claude Opus 5.5): Agreed; in the plan the freeze sits between tuning and the final runs (decision D3).
- **The earlier negative results** (arm A, the UTH series, Exp 0–3) are part of the story at least as an appendix: they are why the setup looks the way it does.  
C: yes
- **Compute.** 4c took 58 min on an H100 (5,000 steps); a 4e 3,000-step screening run ≈ 30 min on the RTX PRO 6000. Three methods × a tuning grid × seeds, plus OOD analyses, is a few days of GPU time [assumed: rough].  
C: we'll have to make due with less. That's too expensive and I also bank on the different methods being fundamentally different enought that we can clearly see their performances far appart. No tuning grid, but sequential tuning. Then the final runs are 2 seeds x 3 methods x 1h = 6h which is fine.
Answer (26-10-22, Claude Opus 5.5): Agreed: sequential tuning, finals 2 seeds × 3 methods. One caution: if two methods end up close, 2 seeds won't separate them, and the report should then say so instead of ranking them.

## 8. Candidate stories, as I see them now

- (A) **Negative, with a diagnosis**: a target-only loss lets tPD re-implement the verdict instead of decomposing the MLPs (4a); faithfulness constraints remove that (4c, 4e), and what remains is a verdict readout gated by sentence form; interventions on the components don't beat a data-aligned untrained basis; the model's knowledge survives verdict inversion. Reason for the failure: the target loss constrains only the output, so nothing ties the components to the weights' own computation (your "too much freedom" point) — plus the `down_proj`-only limitation from section 2.
- (B) **Partial mechanism**: the layer-11 "false" subspace (A8) and the negation/comparison components as a partial decomposition of how the model handles negated and compared statements. Current evidence: one method, val split only, A8 in two seeds. Promising but thin.

The data so far fit (A) better, with (B) as the part worth testing hardest [concluded: sections 4 and 7]. The OOD sets with negations and the conj/disj sets are what could move it towards (B).

## Questions for you

1. "Registered": do you mean `spd/registry.py` (the paper code's `EXPERIMENT_REGISTRY`, configs with a canonical WandB run), the run register `convos/julian/26-10-03_wandb_storage_RUNS.md`, or both?  
C: I meant the registry. This is important for reproducability for other researches. THe wandb_storage is just for me to keep track which ones I can delete to make room for others.
Answer (26-10-22, Claude Opus 5.5): Understood. What this means for the finals [verified: `spd/registry.py`, `tests/test_wandb_run_loading.py`, run register]: a registry entry is a config plus an optional canonical WandB run, which `tests/test_wandb_run_loading.py` loads. So canonical runs must be readable by others: the project `bitt-j-personal/spd` made public, or the runs moved to a public project. Their checkpoints must also stay: six final runs on layers 11–17 are 6 × 333.5 MB = 2.0 GB of the 5 GB cap (3.00 GB used on 26-10-21). And the one-off analysis scripts under `~/spd_out/` (`ablate_4e.py`, `mask_search_4e.py`, `compare_decomps.py`, …) have to move into the repo so others can reproduce the results (plan, step A5).
2. Living report: a Typst file in `reports/` like the earlier reports (my suggestion: `reports/26-10-22_tpd_truth_DRAFT.typ`, edited in place, with this LOG recording the changes), or a Markdown DRAFT next to this LOG?  
C: I think the document living in `reports` is fine. It might be some sort of uber summary of different convos. Basically it's the first internal representation of the final blog post or paper.
Answer (26-10-22, Claude Opus 5.5): Agreed. I'll start it as `reports/26-10-22_tpd_truth_paper.typ` (dated like the other reports): an overview across the relevant topics, every result tagged with its status and its source topic (plan, step A1).
3. Add the cancellation constraint to the tuning objective (section 3)?  
C: I hope we can find soemthing better
Answer (26-10-22, Claude Opus 5.5): See my answer under your comment in section 3: tPD's own PGD and Pile losses and your axis (c) measured against all off already separate 4a.
4. CI-weighted or rounded state for all metrics?
C: CI-weighted
Answer (26-10-22, Claude Opus 5.5): Noted.
5. Balance: cap families, weight the loss per family, or something else?
C: what are the pros and cons?
Answer (26-10-22, Claude Opus 5.5): Cell sizes in tiu v2 train: element_symb 51 per label, animal_class 55–57, sp_en_trans 117–130, cities 504, comparisons 659.
- **Trim to a common size** (your preference). Pros: every family weighs the same in training; no code change, only a new data version. Cons: the training set shrinks. Trimming every cell to 51 gives 1,020 statements instead of 5,574. At 5,000 steps × batch 64 each statement is then seen ~310 times, so overfitting the CI network to individual statements becomes a risk [assumed]. A cap of 130 per cell (the sp_en_trans size) keeps 1,962 statements, each seen ~160 times, with a largest imbalance of 2.5:1 instead of 13:1. Either way most cities and comparison statements go, which are the families whose verdict depends most on MLPs 11–17 (4e at 3e-3); that may change which components appear, which is not necessarily bad.
- **Weighting** (per-family loss weights or a family-balanced sampler). Pros: keeps all data, every family equal weight in expectation, and the large families keep their variety. Cons: needs a loader change; the small families' statements are repeated as often as under trimming, so they gain nothing.
- **For evaluation** neither is needed: macro-averaging (every family weighs the same in val/test metrics) solves that side without touching the data [concluded].

My recommendation: trim to the 130 cap, keep val/test unchanged, report macro-averaged metrics. Your call on 51 vs 130.
6. OOD data: start from inventors/neg_inventors, facts/neg_facts, the conj/disj files and the German `*_de` files; generate arithmetic (with negations) and German–English translations. Agreed, and is one new factual family on top of inventors still needed?
C: The non-used OOD data is good enough, I'd say.
Answer (26-10-22, Claude Opus 5.5): Noted [decided: C: comment].
7. Run the attention-ablation scan to justify `down_proj`, or treat it as a stated limitation?
C: I'd like to check this
Answer (26-10-22, Claude Opus 5.5): In the plan as step B1, before any training, because it may change the module set.
8. Is a definition of "truth mechanism" along the lines of section 4 what you want to evaluate against?
C: as I said above
Answer (26-10-22, Claude Opus 5.5): Noted.

C: can you make a plan of which steps we should do in which order?
Answer (26-10-22, Claude Opus 5.5): Yes: entry *plan of steps* below.


---

26-10-22, Claude Opus 5.5 — *plan of steps*; same session, after answering Julian's C: comments on *first pass* in place. I additionally checked the tPD paper's decomposed modules, 4a's state table (bolder-experiments LOG, *Exp 4a finished*), `spd/registry.py` and the model's dimensions.

**Settled by your comments** [decided: C: comments in *first pass*]: CI-weighted state for all metrics. OOD data = the unused upstream tiu files (inventors/neg_inventors, facts/neg_facts, German `*_de`, conj/disj), no generated families, arithmetic shelved. Training data trimmed to a similar size per family, test kept larger. No fixed definition of "mechanism": heuristic criteria, stated as such, not fixed in advance. "Registered" = `spd/registry.py`. Living report in `reports/`. Sequential tuning, no grid; finals 2 seeds × 3 methods. Attention check wanted. Low-CI effect: no extra experiment. Terms: statement, verdict prompt, verdict lens. Analysis plan frozen only after tuning. Earlier negative results go in an appendix.

**Order** — every phase depends on decisions from the one before:

**Phase A — no training, mostly CPU**
- A1. Living report skeleton `reports/26-10-22_tpd_truth_paper.typ`. Sections: setup with appendix justifications, the three methods, selection criteria, the four categories, data, results, limitations, earlier negative results. Filled with what exists now, every result tagged as screening-era (to be redone) or final.
- A2. Data v3 via `build_tiu.py`: trimmed train split (cap 51 or 130, your call), val/test unchanged. New test-only datasets from upstream: inventors, neg_inventors, facts, neg_facts, the `*_de` files, conj/disj. Commit the generator before building (lesson from the prepared-datasets topic).
- A3. Selection criteria on the existing runs 4a, 4c, 4e: Δ only's share of the all-off KL, Pile KL, PGD loss, components-only KL. Question: do 4a's failures show without a cancellation metric? (Partly yes, see my answer in section 3.) Some numbers exist in the LOGs; 4c's Δ-only state and the PGD values of 4e's PGD-off screening runs need a short GPU job.
- A4. Mass-mean vs logistic-regression direction: cosine per layer from stored residuals (CPU).
- A5. Move the one-off analysis scripts (`ablate_4e.py`, `mask_search_4e.py`, `compare_decomps.py`, `attrib_4e.py`) into the repo next to `verdict_lens_analysis.py`, with the CI-weighted state as default and macro-averaging over families. Needed for the registry anyway.

**Phase B — GPU, no training (one rental, a few hours)**
- B1. Attention scan: the layer-scan method (mean ablation at the statement tokens) applied to the attention outputs: single layers and the block 11–17, with the MLP rows rerun as reference in the same script.
- B2. The original model on the OOD sets: verdict accuracy per family (which sets does the model know well enough to use?), plus choosing the second verdict prompt from the candidates of the earlier statement-first screen (R5).
- B3. Belief steering (your first comment; independent of the decompositions, so it can also run alongside phase C): steering at the statement window, outcomes verdict / second prompt / recall / continuation, then widening the edit to find where the recovered information comes from.

**Decision D1 (you)**, from A3, B1, B2: the selection criteria; the module set (`down_proj` only or + `o_proj`); the trim cap; tuning with PGD on or off (see question 1).

**Phase C — sequential tuning** (v3 train, val split, one seed, ~3,000 steps per run, same number of runs per method)
- C1. Plain tPD at all statement tokens: LR, then importance-minimality coefficient. First run first, because it was never run at all tokens.
- C2. + reconstruction term: LR, reconstruction coefficient (never varied so far), importance-minimality.
- C3. + orthogonality: start at 4e's optimum (LR 2e-3, importance-minimality 1e-3) and check whether it holds on v3.

**Decision D3 (you)**: freeze the analysis plan in the report (metrics, controls, the four categories' analyses) before the final runs.

**Phase D — final runs and analyses**
- 2 seeds (neither is seed 0, the tuning seed) × 3 methods, 5,000 steps, PGD on, ~6 h by your estimate.
- Analyses on test, the OOD sets and the second prompt, in order of cost: recovery → termination (incl. A8, removing the components' write subspace from the original MLP output) → correlation (sparsity, transfer, truth vs verdict) → manipulation (mask search against the data-aligned basis over the number of terms, recall test). Controls everywhere: the untrained decomposition and the data-aligned basis.

**Phase E — registry and write-up**: registry entries with canonical runs (public WandB project; checkpoints kept, ~2.0 GB of the cap), then the report becomes the paper or blog draft.

Rough GPU budget [assumed: from earlier run times, not measured for v3]: B ~2–3 h; C ~12–15 runs at 0.5–1 h; D 6 h plus analyses.

**Questions on the plan**
1. Tune with PGD on or off? I recommend on. PGD is one of the selection criteria, and the low-CI effect you described depends on it. Screening without PGD (as in 4e) would tune for a different objective from the one the finals train with. Cost: slower steps, not measured on our setup.  
C: tuning with PGD on.
2. Trim cap: 51 (exact balance, 1,020 statements) or 130 (1,962 statements, at most 2.5:1)?  
C: 130
3. Should I start with phase A now (A1, A2, A4, A5 need no GPU)?
C: please start with phase A, but skip A1 for now

---

26-10-26, Claude Opus 5.5 — *phase A without A1: v3 data code, existing selection numbers, mass-mean vs logistic regression*; new session. I have read INDEX.md, this whole LOG and its SUMMARY, `docs/CODING.md`, the prepared-datasets SUMMARY, the bolder-experiments LOG entries *Exp 4b finished*, *Exp 4a finished* and *Exp 4c finished*, and the faithful-targeting SUMMARY. Julian's answers to *Questions on the plan* [decided: C: comments above]: tune with PGD on; trim cap 130; start phase A, skip A1 (the living report).

## A2: data v3 (code ready, build waits for a commit)

`spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py` now has two additions; `data/README.md` describes v3.
- **`--train_cap N`**: each in-distribution dataset keeps at most N train statements; val and test are those of the uncapped build. Statements are selected by a hash of their subject, so a kept city stays in cities_true, cities_false and both neg_cities files: the trimmed train split keeps the true/false/negated statements about the same subjects. Kept statements stay in upstream order.
OUTDATED (26-10-26): the counts in the next bullet and under **Build** are wrong: 28 files, 56 OOD datasets, 76 directories. See entry *v3 committed and checked; next: phase B*.
- **OOD datasets** (manifest `role: ood_test`, test split only, so no config can train on them): every unused upstream file except the large, less curated common_claim and counterfact. That is 26 files × 2 labels = 52 datasets: inventors, neg_inventors, facts, neg_facts; German versions of 6 families with their negations (12 files, 49–50 statements each); conj/disj of 6 families (12 files, 500 each except cities_conj 1,498). The first pass listed fewer: upstream also has German and conj/disj versions of animal_class, element_symb, sp_en_trans, inventors and facts. In-distribution records keep their exact field order, so the split files stay comparable across builds.

Checks [verified: trial builds into `scratch/` (gitignored, deleted afterwards), 26-10-26]:
- A v2 rebuild with the modified script is byte-identical to the committed v2: 60 of 60 split files.
- v3 (`--val_frac 0.12 --train_cap 130`): val and test byte-identical to v2 (40 of 40 files). Train has 1,962 statements (v2: 5,574). Each v3 train file is a subset of v2's, in upstream order. Only cities, neg_cities, larger_than and smaller_than are cut (504 or 659 → 130 each); the others were already ≤ 130. In the capped namespaces the same 130 subjects are kept in every dataset (city: 130 of 130 across the four files; number pairs: 130 of 130).
- Upstream duplicates dropped, as for the other files: animal_class_disj 119 of 500, inventors_disj 34, facts and neg_facts 13 each, the rest ≤ 6.
- OOD records, e.g. `{"text": "Emile Berliner lebte in Russland.", "label": false, "upstream_file": "datasets/inventors_de.csv", "upstream_row": 1}`. cities_conj keeps its upstream columns (statement1, city1, …).

One thing to know about the OOD sets: they are not split by subject, so their subjects can occur in training. Of the 25 cities in cities_de, 19 are in v2's train split but only 4 in v3's [verified: same check]. A German statement about a city seen in English training is a weaker language-OOD test than one about an unseen city. B2 can report both groups if it matters.

**Build** (after the commit, so that the manifests' `repo_commit` contains the generator; lesson from the prepared-datasets topic):
`python spd/experiments/lm/honesty_targeted_decomposition/build_tiu.py --version v3 --val_frac 0.12 --train_cap 130`
It writes 72 new directories under `data/tiu/` (20 in-distribution, 52 OOD); a new version never overwrites an existing one.

## A3: what the existing runs say about the selection criteria

From the records, no GPU [verified: re-read today from the bolder-experiments LOG entries *Exp 4a/4b/4c finished* (test split, 5,000 steps, PGD on in training, rounded CIs) and from 4e's training logs in `~/spd_out/26-10-16_faithful_targeting/e4/instance_copy*/rounds/` (val split, 3,000 steps, PGD off in training, CI-weighted); one seed each except 4e]. *Recovered* = share of the all-off verdict KL that the state restores, (KL_all-off − KL_state) / KL_all-off; 1 = original, 0 = all off.

| run | positions | components only, recovered | Δ only, recovered (ideal 0) | Pile KL | PGD loss at eval |
|---|---|---|---|---|---|
| 4a plain tPD | end token | 0.94 | 0.53 | 0.083 | 1.20 |
| 4b + reconstruction | end token | 0.88 | 0.44 | 0.021 | 0.13 |
| 4c + reconstruction | all statement tokens | 0.74 | −0.18 | 0.028 | 1.25 |
| 4e + orthogonality, seed 0 | all statement tokens | 0.91 | −0.34 | 0.054 | 4.43 |
| 4e seed 1 | all statement tokens | 0.91 | ≈ 0 (separation at the all-off level) | 0.065 | 5.24 |

The Pile column is `eval/nontarget/rounded_recon_loss` for 4e. For 4a–4c it is what the LOG calls "Pile rounded KL"; I assume it is the same metric [assumed].

Reading [concluded unless tagged]:
- **No criterion in this table separates plain tPD from all three others.** Δ only separates 4a from 4c and 4e, but not from 4b (0.53 vs 0.44). Both run at the end token, where all off still keeps 73% of the verdict separation, so Δ there can carry a lot without any re-implementation. Pile KL and PGD separate 4a from 4b and 4c. But 4e's screening runs are worse than 4a on PGD (4.4–5.2 vs 1.20) and between 4a and 4c on Pile, because they trained without PGD.
- This corrects part of my answer in section 3. "(i) is what flags 4a" is wrong, since 4b fails Δ only almost as much. The statement that PGD and Pile separate 4a from 4b stands. Marked OUTDATED in place.
- The runs differ in positions, PGD and split, so they can't answer the real question: do tPD's own losses and the Δ-only share exclude plain tPD once everything else is equal? Tuning step C1 (plain tPD at all statement tokens, PGD on, the first run of phase C) answers it directly. **Proposal**: drop the planned A3 GPU job (CI-weighted re-evaluation of 4a and 4c) and decide the between-method criteria when C1 is in [assumed: proposal]. Within-method tuning in phase C doesn't need them.

## A4: mass-mean vs logistic-regression truth directions

Script and output: `~/spd_out/26-10-22_tpd_truth_paper/mm_vs_lr.{py,out}` [verified: one run, CPU, 26-10-26]. Data: the stored residuals of the 26-10-06 truth-writers extraction (tiu v1, **bare statements, last token**, after layers 15–19 and 23; fit 5,260 / tune 1,300 / test 1,794). mm = unit(mean true − mean false) on fit, exactly the directions used for steering (`stored_dirs.npz`). lr = logistic regression on standardised features, C chosen on tune, mapped back to the raw basis (as `lr_L19` in `truth_writers.py`). *Split-half* = cosine between the directions fit on two random halves of fit. *Projection accuracy* = test accuracy of thresholding the projection onto the direction at the midpoint of the fit class means.

| layer | cos(mm, lr) | split-half mm | split-half lr | projection acc. mm (pooled / family macro) | projection acc. lr |
|---|---|---|---|---|---|
| 15 | 0.33 | 0.987 | 0.65 | 0.925 / 0.910 | 0.998 / 0.999 |
| 16 | 0.36 | 0.992 | 0.68 | 0.935 / 0.894 | 0.998 / 0.997 |
| 17 | 0.36 | 0.993 | 0.72 | 0.945 / 0.925 | 0.998 / 0.999 |
| 18 | 0.45 | 0.996 | 0.72 | 0.984 / 0.964 | 0.998 / 0.999 |
| 19 | 0.46 | 0.998 | 0.75 | 0.980 / 0.959 | 0.998 / 0.999 |
| 23 | 0.41 | 0.998 | 0.75 | 0.933 / 0.867 | 0.998 / 0.997 |

Random directions in 3,584 dimensions have |cos| ≈ 0.017.

Reading [concluded unless tagged]:
- **Your reason (1), "they are not very different", does not hold as stated.** The two directions are related (cos 0.33–0.46, 20–30× the random level) but clearly different, and lr separates better (0.998 vs 0.925–0.984 pooled; on the family macro mm drops to 0.867 at layer 23).
- **mm is far more stable**: two halves of the data give the same mm direction (0.99) but lr directions only 0.65–0.75 apart. lr spends weight on directions that happen to separate this sample. For an intervention direction that is an argument for mm.
- Reason (2), speed, holds: mm is two means, lr needs a C search (here 5 values × 6 layers, minutes on CPU).
- A third reason now has a source: Marks & Tegmark (2023), *The Geometry of Truth* (arXiv:2310.06824), report in the abstract that mass-mean probes are about as accurate as other probes but "more causally implicated in model outputs" [verified: abstract, via web search 26-10-26; check their intervention section before citing details]. Our own numbers match the accuracy side only for layers 18–19.
- Proposed justification for the paper [assumed: proposal]: "We use mass-mean directions because they are hyperparameter-free and stable across data halves (cos 0.99 vs 0.65–0.75 for logistic regression), and because Marks & Tegmark found them more causally implicated; logistic-regression directions classify slightly better (Appendix)." A direct check costs little in phase B: steer once with lr instead of mm in the B3 setup.
- Caveat: bare statements at the last token, not the verdict prompt (no stored verdict-prompt residuals), and no layers 13–14.

## A5: moving the analysis scripts: deferral proposed

Inventory [verified: diffs, 26-10-26]: the newest versions are the local ones in `~/spd_out/26-10-16_faithful_targeting/e4/`.
- `ablate_4e.py` (A1–A8 incl. the projPCA control) is identical to the instance variant `_a8b`. The older `ablate_4e.py`, `_a7` and `_a8` in `instance_copy_26-10-21/` are superseded.
- `mask_search_4e.py`: the local copy adds `--pca_basis` (the data-aligned untrained control). It is newer than the instance copy.
- `attrib_4e.py`: identical in both places.
- `compare_decomps.py`: CPU post-processing of the ablation JSONs.
- B3 will also reuse `~/spd_out/26-10-16_component_steering/{mask_search,recall_test,flip_components}.py`.

I did not move them. Reasons [concluded]: (1) a moved script can only be checked by reproducing a stored number on Qwen-7B, which needs a GPU; moving them without that check risks a silent change. (2) The plan's changes (CI-weighted default, macro-averaging, test split) depend on the analysis plan that is frozen only at D3, so moving now means moving twice. **Proposal**: move them in the phase B rental, each with one reproduction check against its stored output, or after D3. The registry only needs the final versions.

## Questions for Julian

1. Please commit `build_tiu.py` and `data/README.md`. Then I build v3, or you run the command above yourself, and the data gets its own commit.
2. A3: drop the GPU re-evaluation of 4a/4c and decide the between-method criteria on C1's numbers?
3. A5: move the scripts during the phase B rental (with reproduction checks), or after D3?
4. Mass-mean justification: take the proposed wording (stability, no hyperparameters, Marks & Tegmark), and add the lr-vs-mm steering check to B3?

---

26-10-26, Claude Opus 5.5 — *v3 committed and checked; next: phase B*; same session. Julian in chat:

C: (transcribed from chat) Ok, I commited the script and the data. Yes, we can delay A3. We can justify mass-mean by refering to their argumentation. What is next?

Decisions [decided: transcribed above]: A3 delayed (no GPU job now; the between-method criteria come later, see the previous entry); mass-mean directions are justified by Marks & Tegmark's argument (more causally implicated in model outputs, arXiv:2310.06824), so no lr-vs-mm steering check. Question 3 of the previous entry (when to move the scripts, A5) is still open.

**v3 as committed** (Julian built it: `001590a` generator, `129c90e` data) [verified: manifests and hashes, 26-10-26]: 76 datasets, 20 in-distribution and 56 OOD. Every manifest records `repo_commit` `001590a` and the sha256 of the committed script; every split file matches its manifest hash; in-distribution val/test are byte-identical to v2; train totals 1,962.

**Count correction**: the OOD files are 4 + 12 German + 12 conj/disj = 28, so 56 datasets and 76 directories. The previous entry says 26 / 52 / 72 (marked OUTDATED there).

## Next: phase B (GPU, no training, one rental)

B2 has to come before B3, because B3 reads its outcome under the second prompt that B2 picks. Each script gets a CPU smoke or short GPU smoke before its full run. I adapt them locally before the rental:

- **B1, attention scan.** `~/spd_out/26-10-10_bolder_experiments/layer_scan/layer_scan.py` with the attention output (`o_proj` output) mean-ablated at the statement tokens: single layers 5–24 and the block 11–17, with the MLP rows rerun in the same script as reference. Outcomes as in the MLP scan: verdict separation share, accuracy, digit mass, KL. Data: tiu test (identical in v1–v3). Decides whether `o_proj` joins the module set (decision D1).
- **B2, the original model on the OOD sets and a second prompt.** Verdict accuracy, AUC and digit mass per dataset on the 56 OOD sets, with the verdict prompt and a second prompt. My candidate for the second prompt is "Is this statement true (1) or false (0)? Answer: " on the same line, the R5 runner-up (accuracy 0.952, AUC 0.995 vs 0.963 / 0.994 for the amount prompt; one run, 546 statements). Its errors sit mainly in neg_element_symb_true, where the amount prompt errs on other statements, so the two prompts are not trivially the same readout [assumed]. `verdict_accuracy.py` is the starting point; I still have to check that it handles test-only datasets and a second prompt. This decides which OOD sets the model knows well enough to be usable.
- **B3, belief steering.** Steer with the in-context mass-mean truth direction at the statement tokens, layers 13–17 (`steering_sf/steer_sf.py`). Outcomes: the verdict digit, the digit under the second prompt, a direct recall question (`~/spd_out/26-10-16_component_steering/recall_test.py`), and the greedy continuation. Then widen the edit (more layers, then the answer positions too) to see where the true fact stops coming back.
- A5 (scripts into the repo, each checked by reproducing one stored result) fits into the same rental if Julian wants it then.

Rough cost [assumed: from earlier run times]: 2–3 GPU hours, a 48 GB card is enough (the steering run used 20.5 GB).

**Questions for Julian**
1. Shall I prepare the B scripts now, and do you rent the GPU when they are ready (or shall I, with `spd-vast --mode provision`)?
2. Second prompt: "Is this statement true (1) or false (0)? Answer: " (same line)?
3. Usable OOD set: proposal = the original model's verdict accuracy ≥ 0.8 and AUC ≥ 0.9 under the verdict prompt. Sets below that are reported but not used for transfer claims, since the decomposition can't recover a judgement the model doesn't make [assumed: thresholds are a guess].
4. Recall question in B3: the existing test covers cities only ("In which country is the city of X?"). Extend it to inventors ("In which country did X live?") and elements ("What is the chemical symbol of X?")?
5. A5: during this rental or after the analysis freeze (D3)?

---

26-10-26, Claude Opus 5.5 — *phase B started on the rented GPU*; same session. Julian in chat, answering question 1 of the entry above:

C: (transcribed from chat) do it on the remote. It's cheap enough. The GPU is rented.

So the scripts are developed and smoke-tested directly on the instance [decided: transcribed above]. Questions 2–5 are unanswered. I proceed with the proposals as working assumptions [assumed]: B2 screens the second-prompt candidate alongside the verdict prompt; the OOD threshold is applied only after the numbers are in; recall questions for inventors and elements are added if cheap; A5 is not done in this rental unless Julian says so.

**The rented host is in China, and how the model got onto it** (26-10-26; host vast.ai, RTX 4090 48 GB, 450 W, IP geolocation Hangzhou, China Mobile):
- huggingface.co is blocked there. DNS returns a poisoned address (108.160.163.106) and connections time out. raw.githubusercontent.com is blocked too. PyPI, GitHub, WandB (api.wandb.ai, wandb.ai), ModelScope and hf-mirror.com are reachable [verified: curl from the instance].
- Workarounds used, all visible and reversible:
  1. Qwen/Qwen2.5-7B-Instruct downloaded through the third-party mirror hf-mirror.com (`HF_ENDPOINT` set only for that download), pinned to revision `a09a354` (= the local cache's `refs/main`). Every file the code uses was checked against the official copy in the local Hugging Face cache: 11 of 11 sha256 equal (4 safetensors shards, config, generation config, tokenizer files, index) [verified: `sha256sum -c` against `~/spd_out/26-10-22_tpd_truth_paper/qwen_local_sha256.txt`]. LICENSE and README were not checked; the code doesn't read them. No remote code is executed (no `trust_remote_code`).
  2. `refs/main` written by hand into the instance's cache (content `a09a354…`, as locally), because a pinned download doesn't write it and offline loading needs it.
  3. The phase-B scripts run with `HF_HUB_OFFLINE=1` and load the model from the snapshot directory (`--model_path`). With a repo id, transformers queries the Hub API even offline (the crash `docs/CODING.md` warns about).
- One attempt to run a script with `HF_ENDPOINT=https://hf-mirror.com` (API metadata through the mirror) was refused by Claude Code's auto-mode permission check. I did not retry it and switched to step 3.
- Consequence for phase C: the spd training code loads the model by repo id and calls the Hub API, so it can't run on this host without further workarounds. Julian asked in chat whether to rent a non-Chinese machine instead.

C: (transcribed from chat) Can we just rent a non-chinese vm?

C: (transcribed from chat) Did you do any dirty tricks to make it work?

**B1 smoke** [verified: 40 test statements, 3 per dataset of train for the means, 26-10-26]: runs end to end; the no-op condition reproduces the original exactly (max |score diff| 0.0000). Full B1 launched on this host (results don't depend on the machine), output `~/paper/b1_full/` on the instance.

---

26-10-26, Claude Opus 5.5 — *B1 and B2 results*; same session. Julian in chat: "Ok, that is fine. Tell me when the results are in" (go-ahead to finish phase B on the Chinese host). Scripts `~/spd_out/26-10-22_tpd_truth_paper/phase_b/` (`common.py` loader, `b1_attn_scan.py`, `b2_ood_verdicts.py`, `b3_belief_steering.py`; `steer_sf.py`, `r4_framing_screen.py`, `r5_statement_first_screen.py` are unchanged copies from `~/spd_out/26-10-10_bolder_experiments/steering_sf/`). Outputs copied to `~/spd_out/26-10-22_tpd_truth_paper/{b1_full,b2_full}/`.

## B1: attention outputs vs MLP outputs at the statement tokens

[verified: one run, 1,794 tiu test statements, means from the 6,560 v1 train statements, 728 s on the 4090. The no-op condition reproduces the original exactly. The MLP block 11–17 (mean) reproduces the 26-10-16 MLP scan: accuracy 0.538 vs 0.541, separation share 0.046 vs 0.05, digit mass 0.977 vs 0.977, KL 0.680 vs 0.68.] Original: accuracy 0.977, AUC 0.996, separation 8.66. *Separation share* = mean verdict score of true minus false statements, relative to the original (1 = untouched, 0 = no judgement left).

Mean ablation at all statement tokens:

| block | MLP: separation share / accuracy / KL | attention: separation share / accuracy / KL |
|---|---|---|
| 6–10 | 0.18 / 0.707 / 0.44 | 0.63 / 0.853 / 0.23 |
| 11–14 | 0.38 / 0.881 / 0.23 | 0.44 / 0.742 / 0.63 |
| **11–17** | **0.05 / 0.538 / 0.68** | **0.21 / 0.655 / 0.66** |
| 15–19 | 0.33 / 0.663 / 0.49 | 0.75 / 0.949 / 0.04 |
| 20–27 | 0.20 / 0.785 / 4.43 (digit mass 0.02: broken, not judged) | 1.00 / 0.977 / 0.00 |
| MLP + attention 11–17 | 0.02 / 0.518 / 0.66 | |

Single attention layers keep ≥ 0.80 of the separation, except layer 0 (0.00: removing the first layer's attention at the statement tokens destroys the input).

Reading [concluded unless tagged]:
- **In 11–14, attention matters as much as the MLPs.** Separation share 0.44 vs 0.38, and attention costs more KL (0.63 vs 0.23). In 15–19 the MLPs matter more (0.33 vs 0.75). Over the decomposed block 11–17 the MLPs carry more of the separation (0.05 left vs 0.21), but the KL cost is about the same (0.68 vs 0.66).
- So `down_proj` alone misses a part of the verdict computation of the same order as the part it covers, at least in 11–14. The choice is no longer just "unjustified"; it is a measured limitation.
- Caveat [assumed]: switching off attention at the statement tokens also cuts the information flow between the statement's words (e.g. subject to object), so it removes more than one "write". The two rows are not strictly the same kind of intervention.
- Side observation: mean ablation of MLP 3, 4, 26 and 20–27 breaks the output (digit mass down to 0.02–0.73). These are the mean-ablation artifacts at massive-activation layers already seen in the MLP scan.
- **For decision D1** [assumed: proposal]: add `o_proj` of layers 11–17 to the decomposed modules. It is about 19% the size of a `down_proj` per layer, so the memory cost is small. The alternative is to keep `down_proj` only and state the limitation with these numbers.

## B2: the original model on the OOD sets and second-prompt candidates

[verified: one run, all 11,122 v3 test statements (1,794 in-distribution, 9,328 OOD), 274 s. Under the verdict prompt the in-distribution accuracy is 0.979, matching the 0.978 recorded for the same test split.] Per-file tables: `~/spd_out/26-10-22_tpd_truth_paper/b2_full/b2_full.log`, `b2.json`.

**Usable OOD sets.** With the proposed threshold (verdict prompt: accuracy ≥ 0.8 and AUC ≥ 0.9), 13 of the 28 OOD files pass: animal_class_conj, cities_conj, element_symb_conj, sp_en_trans_conj; animal_class_de, cities_de, element_symb_de, sp_en_trans_de, inventors_de, neg_cities_de, neg_inventors_de; facts, inventors.
- Fail: **every disjunction** (accuracy 0.506–0.814; the model calls most false disjunctions true), facts_conj (0.786), inventors_conj (0.761), neg_facts (0.715), neg_inventors (0.741), facts_de (0.800, AUC 0.897), and four negated German files (neg_animal_class_de 0.660, neg_element_symb_de 0.460, neg_facts_de 0.469, neg_sp_en_trans_de 0.600).
- Three files fail only on accuracy, not AUC: neg_inventors (AUC 0.921), neg_animal_class_de (0.941), element_symb_disj (0.963). Their score order is right, but the threshold is shifted (neg_inventors: only 0.590 of the false statements answered "0").
- **Consequence**: the OOD tests of negation, the property the decompositions' individually important components detect, are mostly unusable. Only neg_cities_de and neg_inventors_de pass, plus neg_inventors with AUC alone. That weakens the planned "transfer of the negation detectors" test [concluded].
- Proposal [assumed]: use AUC ≥ 0.9 as the criterion. The decompositions are judged by separation (KL, separation share), not by the threshold at 0. That gives 16 files, including neg_inventors (406 statements).

**Second prompt** (in-distribution pooled accuracy / family macro; OOD macro; verdict agreement with the verdict prompt, in-distribution / OOD; score correlation in-distribution):

| prompt | in-dist | OOD macro | agreement | correlation | worst in-dist family |
|---|---|---|---|---|---|
| amount\|nl (verdict prompt) | 0.979 / 0.968 | 0.777 | – | – | element_symb 0.875 |
| is_true\|sp "Is this statement true (1) or false (0)? Answer: " | 0.983 / 0.953 | 0.807 | 0.971 / 0.731 | 0.905 | neg_element_symb 0.571 (true ones 0.143) |
| scale\|nl "On a scale from 0 (false) to 1 (true), the truth of this statement is " | 0.976 / 0.943 | 0.846 | 0.970 / 0.858 | 0.948 | neg_element_symb 0.518 (false ones 0.036) |

C: what does correlation mean?
Answer (26-10-26, Claude Opus 5.5): Pearson correlation, over the 1,794 in-distribution test statements, between the verdict score s = log P('1') − log P('0') under the verdict prompt and s under the candidate prompt [verified: `np.corrcoef` in the B2 check]. 1 would mean the two prompts give the same scores up to scale and shift. A lower value means the candidate orders the statements' confidence more differently, even where both prompts give the same digit (in-distribution agreement 0.97 for both candidates). So 0.905 vs 0.948: is_true|sp is the somewhat more different readout.

Reading [concluded unless tagged]: both candidates are as accurate as the verdict prompt in-distribution and give a clean digit (mass 0.999–1.000 vs 0.984). scale\|nl is more accurate OOD, but closer to the verdict prompt (same "the truth … is" form on a new line; correlation 0.948, OOD agreement 0.858). is_true\|sp is a question with an answer slot, on the same line; its scores are less correlated with the verdict prompt (0.905, OOD agreement 0.731). For a transfer test the more different readout is the stronger test, so I recommend **is_true\|sp** [assumed: Julian decides]. Both fail neg_element_symb in opposite directions, so per-family results on it need care.

B3 (belief steering) is running; results in the next entry.

---

26-10-26, Claude Opus 5.5 — *B3 results: activation steering changes the verdict, not the belief*; same session. Script `~/spd_out/26-10-22_tpd_truth_paper/phase_b/b3_belief_steering.py`. Outputs `~/spd_out/26-10-22_tpd_truth_paper/b3_v2/` (final) and `b3_full/` (first run, whose recall measure was broken, see below).

**Setup** [verified: one run, 26-10-26].
- Steering direction: the in-context mass-mean direction of the verdict prompt for the group "all statement tokens", computed from the 6,560 tiu v1 train statements.
- Step: α × the class gap Δ along the direction. Δ at the statement tokens is 1.13 (L11) to 3.85 (L17), in the range of the 26-10-15 run.
- Each statement is pushed towards the opposite label. Only the statement's tokens are edited; question, answer and generated tokens run unsteered.
- Conditions: L17 α = 4 and 8; L13–17 each at α = 4 (wider edit); a shuffled-label direction at L17, moved by the same distance (control); the answer position (final space) at L18, α = 4, for the verdict prompt only ("verdict without belief" reference).
- Readouts: (a) the verdict prompt; (b) the second prompt "Is this statement true (1) or false (0)? Answer: " (1,794 test statements); (c) an in-context recall question after the statement, 30 greedy tokens: 166 true + 166 false city statements ("In which country is the city of X?") and 28 + 28 element statements ("What is the chemical symbol of X?"); (d) 30-token continuations of the verdict prompt for 8 city statements.
- Rerun check: the verdict numbers of the rerun are identical to the first run's.

**(a, b) Verdict flips** (flipped and digit mass ≥ 0.5, among statements the unsteered model gets right):

| condition | verdict prompt (n = 1,752) | second prompt (n = 1,764) |
|---|---|---|
| statement tokens, L17, α = 4 | 0.863 | 0.770 |
| statement tokens, L17, α = 8 | **0.975** | **0.994** |
| statement tokens, L13–17, α = 4 each | 0.972 | 0.977 |
| shuffled-label control, L17, α = 8 | 0.076 | 0.026 |
| answer position, L18, α = 4 | 0.985 | (not defined) |

This is a reproduction check of the 26-10-15 run, on the whole test split instead of 200 statements: 0.863 / 0.975 / 0.985 there vs 0.78 / 0.92 / 0.99.

**(c) In-context recall**: share of answers that name the correct country / symbol.

| condition | cities, true statement | cities, false statement | elements, true | elements, false |
|---|---|---|---|---|
| question alone (no statement) | 0.940 | 0.940 | 0.857 | 0.857 |
| statement, unsteered | 1.000 | 0.952 | 1.000 | 1.000 |
| L17, α = 4 | 1.000 | 0.958 | 1.000 | 1.000 |
| L17, α = 8 | 0.988 | 0.976 | 1.000 | 1.000 |
| L13–17, α = 4 | 1.000 | 0.964 | 1.000 | 0.964 |
| shuffled control | 1.000 | 0.976 | 1.000 | 1.000 |

- First run's measure (12 tokens; "the claimed answer appears") was broken. It counted denials ("Lu'an is not in Turkey. It is…") as adopting the claim, and 12 tokens often ended before the correct country. Its recall table is not used.
- I also dropped a negation-aware "claim affirmed" measure. Checked by eye, it mislabels corrections where the denial follows the claim (`"…is in Brazil" is incorrect`, `"Po" is the symbol for …`). All element "affirmed" cases were corrections. A few genuine adoptions exist for cities with and without steering, e.g. "The city of New Kingston is located in South Africa" (unsteered and steered alike).

**(d) Continuations** (8 city statements, anecdotal). After statement steering the model outputs the flipped digit and then restates the true fact:
- "0.\n\nThe city of Lu'an does not exist in China. Lu'an is actually a city located in Anhui province, China".
- For false ones: "100%.\n\nMarrakesh is a real city, and it is located in Morocco".

Answer-position steering does the same ("0%. … Jiaxing is indeed in China, so the statement is true").

Reading [concluded unless tagged; one run]:
- **Steering the statement tokens changes the verdict under both prompts.** The shuffled-label control does almost nothing (≤ 0.08). So the direction changes something that both verdict readouts take from the statement: an assessment of the statement, not the wording of one prompt. Unlike answer-position steering, this transfers.
- **It does not change what the model knows.** When asked about the city or element right after the steered statement, the model names the correct answer as often as without steering (0.95–1.00 in every condition, the wider L13–17 edit included). In its own continuations it contradicts the steered verdict.
- So the mass-mean truth direction at the statement tokens behaves like a **truth tag on the statement**: verdict prompts read it; fact retrieval ignores it. This matches the component-mask result of the steering topic (verdict inverted, recall intact). The verdict lens therefore targets that tag, not the belief. For the paper: "decomposes the computation of the verdict (the truth assessment of the statement), not the model's knowledge".
- Corrects my earlier hint that steering in the statement window makes the explanation follow the steered digit (marked OUTDATED in *first pass*). The earlier observation came from a few examples of the 26-10-15 run; for steering at all statement tokens of L17 it doesn't hold.
- Not done: the planned "widen the edit to the answer positions". Recall survives the widest statement edit (L13–17) because the recall question names the subject again and the model retrieves the fact from it. Editing the question positions would steer the question, not the statement. A sharper belief test would need a readout that has to use the statement's content but isn't a verdict; I have no clean candidate [assumed]. I propose to stop B3 here.

---

26-10-26, Claude Opus 5.5 — *D1 answers; next step: phase C configs*; same session. Julian in chat:

C: (transcribed from chat) I added one comment about the correlation.
1. I don't quite understand why to add o_proj. When ablating layer 11-17 mlp seems to be able to destroy the verdict entiely.
2. I think Accuracy is only important on the traning data. For eval we wanted to consider the model verdict. I would keep many classes, except the ones where the model is really bad
3. Yes
4. Don't know. What is the next step?

(The correlation comment is answered in place, under the B2 table.)

**1. `o_proj`: you're right, I withdraw the proposal** [concluded].
- MLP 11–17 off leaves 0.05 of the separation. So every route the verdict takes from the statement tokens passes through those MLPs, and decomposing them gives a handle on all of it. 4e's components alone already recover 0.91 of the verdict KL, with attention on original weights.
- What B1 shows is that attention 11–14 is *also* necessary (0.44 left, vs 0.38 for the MLPs). The two are in series: attention moves information between the statement's words, the MLPs transform it, and removing either breaks the chain.
- So the decomposition can show the MLP part of the computation but not the attention part (which word's information reaches which position). That is a limitation for the word "mechanism", like the gate/up limitation. It is no reason to expect a better-targeted decomposition with `o_proj`.
- For the paper: keep `down_proj`; state the limitation with the B1 numbers. Module set = `down_proj` 11–17 [decided: transcribed above, point 1, with this reading].

**2. OOD criterion** [decided: point 2]: the evaluation target is the model's own verdict, not the ground truth, so model accuracy matters only for the training families. OOD sets stay in unless the model is really bad on them.
- My proposal for "really bad" [assumed]: AUC < 0.7 under the verdict prompt, i.e. the model's scores barely order true above false. That excludes four files (B2, verdict prompt AUC): neg_element_symb_de 0.570, neg_facts_de 0.607, neg_sp_en_trans_de 0.666, inventors_disj 0.685. The other 24 stay.
- Analyses that use the ground-truth label (truth vs verdict) need sets where the model is mostly right. For those the 13 sets above the stricter bar are the natural subset.

**3. Second prompt** [decided: point 3]: "Is this statement true (1) or false (0)? Answer: " on the same line.

**4. A5 timing** [assumed: my recommendation]: after the analysis freeze (D3). The ablation, mask-search and attribution scripts are phase-D analyses. Tuning (phase C) needs only `verdict_lens_analysis.py`, which is already in the repo. Moving the scripts once, in their final form, with a reproduction check each, is cheaper than moving them now and again after the freeze.

## Next step: phase C (sequential tuning)

Settled inputs: `down_proj` 11–17, C = 96 per layer; all statement tokens; tiu v3 train (1,962 statements), evaluation on val; PGD on; CI-weighted metrics; seed 0 for tuning; one method after the other.

Proposed protocol [assumed: proposal]:
- **Three configs that differ only in the method term**, written from the current 4e config (`config_truth_sf_verdict_ortho_global_alltok.yaml`): plain tPD (no extra term), + per-layer reconstruction (`CIHiddenActsReconLoss` in the loss, coefficient 1 as in 4b/4c), + global orthogonality (coefficient 1 as in 4e). A diff of the three configs must show only that term.
- **Same budget per method**: 3,000 steps per run. LR {5e-4, 2e-3, 5e-3}, then importance-minimality coefficient {3e-4, 3e-3} at the best LR (1e-3 is the default in the LR runs). That's 5 runs per method, 15 in total.
  - The method coefficients (reconstruction, orthogonality) stay at 1: tuning them would give those two methods more runs than plain tPD.
  - Limitation to state: the method coefficients were not tuned.
- **Tuning objective within a method** (as in 4e): share of the all-off verdict KL recovered by the CI-weighted components alone, on val.
  - Logged alongside for the later between-method choice (A3, delayed): Δ only, Pile KL, PGD loss.
  - Missing: transfer to the second prompt. `verdict_lens_analysis.py` doesn't have it yet; I'd add it before C so that every tuning run gets it.
- **PGD detail to know**: all current configs start the persistent PGD term at 80% of training (`start_frac: 0.8`). With 3,000 steps it acts in the last 600. I'd keep that, since it is what 4a–4c used.
- **First on the new host**: one 300-step trial per config, to measure step time with PGD on, then the 15 runs one at a time. Rough cost [assumed: from 4c's 58 min per 5,000 steps on an H100]: 35–45 min per run on an H100, about 10 h in all.
- Host: outside China. `spd-vast` has no country filter yet. I'd add a geolocation exclusion (after checking that vast's query syntax supports it) or Julian picks the offer by hand with `--offer_id`.

**Questions for Julian**
1. Budget: 5 runs per method as above, method coefficients fixed at 1?
2. Shall I write the three configs and the second-prompt readout in `verdict_lens_analysis.py` now (CPU, with tests), and the `spd-vast` geolocation filter?
3. OOD "really bad" bar: AUC < 0.7 (four files out)?

---

26-10-26, Claude Opus 5.5 — *tuning budget revised; phase C preparation*; same session. Julian in chat:

C: (transcribed from chat) C: you mean you keep PGD for the final runs not for finetuning, right?

1. No, the method coefficients fixed at 1 is not right. Becaues 10h H100 is quite expensive I suggest:finetune tPD as you suggested. Then finetune extra parameter for the other two approaches. That can be mentioned as a limitation due to cost.
2. Yes
3. Yes

**PGD question**: no, I meant PGD on in tuning as well. That was your decision of 26-10-22 ("tuning with PGD on", C: comment under *Questions on the plan*, question 1). The "80%" is within each run: every current config starts the persistent PGD term at 80% of the training steps (`start_frac: 0.8`), so in a 3,000-step tuning run it acts in the last 600 steps, and in a 5,000-step final run in the last 1,000. I keep that unless you say otherwise. Turning PGD off in tuning would make the runs cheaper, but they would then tune a different objective from the finals (my argument of 26-10-22).

**Revised tuning plan** [decided: point 1 above; the coefficient values are my proposal, assumed]:
- Plain tPD: LR {5e-4, 2e-3, 5e-3}, then importance-minimality coefficient {3e-4, 3e-3} at the best LR (1e-3 in the LR runs): 5 runs.
- + reconstruction and + orthogonality: take plain tPD's LR and importance-minimality; tune only the method coefficient, {0.1, 1, 10} each: 3 runs per method. Total 11 runs instead of 15.
- Limitation to state: LR and importance-minimality were tuned only for plain tPD, for cost. A concrete risk: in the 4e screening the orthogonality method needed LR 2e-3, and at 5e-4 it recovered only 0.37 of the verdict KL. So an LR inherited from plain tPD can handicap a method. If plain tPD's best LR is far from 2e-3, one extra LR run for the orthogonality method would test that [assumed: proposal].
- OOD bar [decided: point 3]: sets with verdict-prompt AUC < 0.7 are excluded (neg_element_symb_de, neg_facts_de, neg_sp_en_trans_de, inventors_disj); 24 OOD files stay.
- Go for the preparation [decided: point 2]: three configs, the second-prompt readout in `verdict_lens_analysis.py`, a country exclusion in `spd-vast`. Implementation notes follow in the next entry.

---

26-10-26, Claude Opus 5.5 — *phase C preparation done: configs, second-prompt readout, country filter*; same session.

**Configs** `spd/experiments/lm/honesty_targeted_decomposition/config_paper_{plain,recon,ortho}.yaml`. They were generated from `config_truth_sf_verdict_ortho_global_alltok.yaml` (the 4e base config) by a script.
- Against the 4e config: label, `steps: 3_000` (tuning length; finals 5,000) and data v3 instead of v2.
- Among themselves they differ only in the method term, the label and one header paragraph [verified: `diff` of the three files]. plain = no extra term; recon = `StochasticHiddenActsReconLoss` (coefficient 1, relative; = 4b/4c); ortho = `ComponentOrthogonalityLoss` (coefficient 1, `all_components`; = 4e).
- Unchanged from the 4e base: LR 5e-4, importance-minimality coefficient 1e-3, PGD from 80% of training, batch 64, C = 96 per layer. The tuning runs override LR, importance-minimality and the method coefficient.
- All three load as `Config` [verified: `Config.from_file`, loss lists printed]. Not committed.

**Second-prompt readout** in `verdict_lens_analysis.py` (repo): the verdict states (full, components, components_ci, delta_only, all_off) are now also evaluated under "<statement> Is this statement true (1) or false (0)? Answer: ". The run's decomposed positions and CIs are recomputed on that prompt.
- Output: `summary["second_prompt"]` and `score2_*` / `kl2_*` arrays. Cancellation numbers stay computed on the training prompt only.
- Tokenization checked on CPU [verified: 3 statements]. Under both prompts the decomposed positions are exactly the statement's tokens (ending in `'.\n'` resp. `'.'`), and the verdict is read at the final space.
- Not yet run on a GPU. First check on the new host: rerun it on the 4e checkpoint `s-eb1c20d5` (val) and compare its training-prompt numbers with the stored ones (CI-weighted KL recovered 0.910).

**`spd-vast` country filter**: new required field `excluded_countries` in `VastConfig`, set to `[CN]` in all four machine configs.
- It adds `geolocation notin [...]` to the offer search and checks every offer, `--offer_id` included, by the country code at the end of vast's `geolocation` string ('Jiangsu, CN').
- Format checked on live listings [verified: read-only `vastai search offers`, 26-10-26]. `spd-vast --list_offers` returns 7 RTX 4090 offers, all in the US.

**Checks**: `make check` passes; the full test suite without testmon: 535 passed, 15 skipped [verified].
- One failure, deselected, not caused by these changes: `tests/test_rotgrid_decomposition.py::test_the_checked_in_sweep_expands_to_distinct_runs` expects 4 sweep configs, but `rotgrid_sweep_params.yaml` now expands to 8. That file was changed by commit `f4f51ec` ("current rotgrid configs") after the test was written. Julian's call whether the test or the sweep is the intended state.

**Hardware outside China** [verified: read-only offer search, 26-10-26, reliability > 0.97]: no H100 SXM or PCIe offers at all; 1 H100 NVL ($3.03/h, US); 13 RTX PRO 6000 WS (from $1.14/h, California); A100 SXM4 from $0.67/h (40 GB); RTX 4090 48 GB from $0.53/h.
- The RTX PRO 6000 ran a 4e training at 0.68× the H100's speed (26-10-21).
- Estimate for the 11 tuning runs on it [assumed: from 4c's 58 min per 5,000 steps on an H100, PGD included]: about 50 min each, 9–10 h, roughly $11–14.
- Proposal: `spd-vast --config rtxpro6000` (now with the country filter).

The Chinese 4090 is still rented and idle; all phase-B outputs are copied locally.

---

26-10-26, Claude Opus 5.5 — *rotgrid test fixed*; same session. Julian in chat: "can you fix the failing test if it's easy. I'm renting a machine meanwhile".

The sweep file is the intended state: Julian's commit `f4f51ec` set seeds 0–3 × importance-minimality {0.01, 0.005} = 8 runs. So the test was updated, not the sweep. `test_the_checked_in_sweep_expands_to_distinct_runs` now reads its expectations from `rotgrid_sweep_params.yaml` instead of hardcoding the old 4-point grid:
- the number of configs equals the product of all `values` list lengths;
- the set of importance-minimality coefficients in the configs equals the swept values;
- all run names are distinct.

[verified: `tests/test_rotgrid_decomposition.py` 3 passed, 1 skipped; ruff and basedpyright clean.]

---

26-10-26, Claude Opus 5.5 — *phase C started: host, reproduction check, trials*; same session. Julian in chat: "The remote is rented you can start", then:

C: (transcribed from chat) while fine tuning can you make sure that wandb doesn't get full (clean up the checkpoints after you)

**Host** [verified: `nvidia-smi`, ipinfo, curl]: RTX PRO 6000 Blackwell Workstation (96 GB, 500 W), Sevilla, Spain. Hugging Face reachable. The repo working tree is synced; the key files are identical to the local ones by sha256.

**WandB storage**: 3.32 GB of 5 GB before phase C [verified: `storageBytes`, 26-10-26].
- Every tuning config has `sync_checkpoints_to_wandb: false` and `save_freq: null`, asserted by the round generator. So tuning runs upload no checkpoints, only metrics (1–3 MB per run). WandB storage doesn't grow, and there is nothing to clean up there.
- The final checkpoints stay on the instance (333.5 MB each; 93 GB free) and are gone when it is destroyed.
- The final runs (phase D) will sync checkpoints, 6 × 333.5 MB = 2.0 GB. That fits next to 3.32 GB only after deleting some old ones; a list of candidates for Julian then.

**Bug found by the planned GPU check**: the second prompt is 5 tokens longer than the training prompt, and 9 val statements then exceeded the run's `max_seq_len` 30 (crash, not a silent error). My CPU check had used three short statements only.
- Fix in `verdict_lens_analysis.py`: the second prompt gets `max_seq_len + 5 + 1` (the extra 1 because its '.' doesn't merge with the next token).
- Checked on CPU over all 986 val and 1,794 test statements [verified]. Not committed yet.

**Reproduction check** on the 4e checkpoint `s-eb1c20d5` (val, 986 statements) [verified: one run, `~/spd_out/26-10-22_tpd_truth_paper/phase_c/check_4e/`]. The training-prompt numbers equal the stored ones: CI-weighted components alone KL recovered 0.910 (stored 0.910), Δ only −0.339 (stored −0.34), accuracy 0.974.

New, the **second prompt** on the same decomposition:

| state | KL recovered | accuracy | separation share |
|---|---|---|---|
| original | 1 | 0.975 | 1 |
| components (CI-weighted) | **0.915** | 0.980 | 0.70 |
| components (rounded) | 0.874 | 0.936 | 0.62 |
| Δ only | 0.213 | 0.567 | 0.13 |
| all off | 0 | 0.557 | 0.14 |

Reading [concluded, one run]: what 4e's components carry transfers to a verdict prompt they were never trained on, about as fully as on the training prompt (0.915 vs 0.910). This matches B3, where statement steering also transferred. The verdict lens decomposes something both prompts read from the statement, not one prompt's readout.

**Phase C tooling** (`~/spd_out/26-10-22_tpd_truth_paper/phase_c/`): `make_round.py` builds tuning configs from `config_paper_{plain,recon,ortho}.yaml`, changing only the swept keys and the run name, and asserts no checkpoint sync, val evals and PGD present. `run_one.sh` trains, then runs the analysis (both prompts) on the final checkpoint. `run_round.sh` runs a round's jobs one at a time. Round `t300`: one 300-step trial per method (PGD from step 240), to check for crashes and measure step time. Launched 26-10-26.

**Trials** (round `t300`) [verified: all three exit 0, training and analysis; outputs in `~/spd_out/26-10-22_tpd_truth_paper/phase_c/t300/`]: plain `s-72534289`, recon `s-6fccc66b`, ortho `s-4acee2cb`; rows added to the run register.
- First launch failed at config validation: `slow_eval_freq` must be a multiple of `eval_freq`. `make_round.py` now validates every config with spd's `Config` before upload.
- Step time with PGD active, read from tqdm in the last 60 steps: 1.1 s (plain, ortho), 1.35 s (recon).
- Estimate for a 3,000-step run [assumed]: about 40 min with evals and analysis, so about 2 h per round of 3. This assumes the pre-PGD steps run at the 0.58 s measured for 4e on this GPU type.

**Round `c1_lr` launched** (26-10-26, ~21:27 UTC): plain tPD at LR 5e-4, 2e-3, 5e-3, importance-minimality 1e-3, 3,000 steps, PGD from step 2,400, one run at a time.

---

26-10-26, Claude Opus 5.5 — *overnight plan*; same session. Julian in chat:

C: (transcribed from chat) I want to go to bed soon but I want you to keep the runs going and do all the fine-tuning during the night and also the final runs then. I suppose for the final runs you'll have to register the experiment, right?

Julian's answers to three questions (multiple choice in chat), transcribed:

C: (transcribed from chat) [WandB space: may I delete the 4 tier-2 checkpoints just before the finals?] "Yes, delete tier 2 (Recommended)"
C: (transcribed from chat) [Coefficient rule for recon/ortho?] "Didn't we once have a preselection for this?"
C: (transcribed from chat) [Extra ortho run at LR 2e-3 if plain tPD's best LR is not 2e-3?] "Yes, one extra run (Recommended)"

**Registry**: a registry entry (`spd/registry.py`) is a config path plus a canonical WandB run, so it can only be written after the final runs exist. Running doesn't need it. Plan: write the final configs into the repo as the exact configs used; add the registry entries in phase E, after the analysis freeze. Making the WandB project public (needed for others to load canonical runs) is still Julian's decision.

**Analysis freeze (D3) and the final runs**: training the finals before the analysis plan is frozen is fine, as long as nobody looks at their test-split results first. The driver runs only the val analysis on them; the test split stays untouched until D3.

OUTDATED (26-10-26): Julian rejected this rule minutes later (no cancellation criterion for coefficients). The driver no longer chooses coefficients; see entry *overnight plan revised*.
**Coefficient rule** [assumed: Julian points to an earlier preselection; this is my reading of it]. The preselection is the 4d tuning proposal in the faithful-targeting topic: "treat the term as a constraint: take the smallest coefficient at which the cancellation test shows no re-implementation, then the best verdict recovery" (faithful-targeting SUMMARY, *Exp 4d*, tuning; never ratified). Operationalised for the driver:
- A coefficient passes if in every decomposed layer |C|/|y| ≤ 1.5 and |Δx|/|y| ≤ 1.5, from the analysis's cancellation test, all components and Δ at the decomposed positions. For scale: 4a reached 12.9 at layer 11; 4e stays ≤ 1.03.
- Among passing coefficients the highest CI-weighted KL recovered wins; ties within 0.005 go to the smaller coefficient.
- If no coefficient passes, the driver stops and waits for Julian.

OUTDATED (26-10-26): steps 3–5 below were replaced (no automatic coefficient choice; only plain tPD's finals overnight); see entry *overnight plan revised*.
**Driver** `~/spd_out/26-10-22_tpd_truth_paper/phase_c/driver.py`, copy on the instance at `/root/paper_c/driver.py`, log `/root/paper_c/driver.log`. Started 26-10-26 21:38 UTC on the instance, so it doesn't depend on this laptop. Rules fixed before any tuning result was seen:
1. c1_lr (already running): plain tPD, LR {5e-4, 2e-3, 5e-3}. Pick by the objective; ties within 0.005 to the LR closest to 2e-3.
2. c2_imp: plain at that LR, importance-minimality {3e-4, 3e-3}. Pick among these and 1e-3; ties to 1e-3.
3. c3_coef: recon and ortho at plain's LR and importance-minimality, coefficient {0.1, 1, 10}. Pick by the rule above.
4. c4_lr, only if plain's LR ≠ 2e-3: ortho at 2e-3 with its chosen coefficient. It is used if it passes the cancellation check and beats the inherited LR by more than 0.005.
5. Final: delete the 4 tier-2 `.pth` files from WandB, with an exact name and size check and a re-listing afterwards. Then per method its settings, 5,000 steps, seeds 1 and 2, `sync_checkpoints_to_wandb: true`, val analysis.

Any failed training or analysis stops the driver ("DRIVER STOPPED" in its log). Dry run on this laptop [verified: `results()` on the real t300 outputs, `pick` tie cases, generated c3 and final configs checked by eye].

WandB after the finals [assumed: from the listed sizes]: 3.32 − 0.95 + 2.0 ≈ 4.4 GB.

Expected timeline [assumed: ~40 min per tuning run, ~65 min per final run]: tuning (11–12 runs) until about 05:30 UTC, finals until about 12:30 UTC on 26-10-27.

Early sign from the trials (300 steps, not a result): plain tPD's components already write up to 1.65 |y| (|Δx| up to 2.01 |y|), the start of the cancellation seen in 4a. Recon and ortho stay at ≤ 0.47 and ≤ 0.27 for |C|/|y|.

---

26-10-26, Claude Opus 5.5 — *overnight plan revised*; same session. Julian in chat:

C: (transcribed from chat) No, I don't want cancelation for picking coefficients. At least not before we have looked at alternatives

So the recon and ortho coefficients can't be chosen overnight [decided: transcribed above]. The driver was stopped while it was only waiting (by PID, 21:40 UTC; round c1_lr kept training), changed, dry-run checked, and restarted. New sequence:
1. c1_lr and c2_imp as before: plain tPD chosen by the objective alone (CI-weighted components-only KL recovered on val; ties to LR 2e-3 / importance-minimality 1e-3).
2. c3_coef: recon and ortho at all three coefficients {0.1, 1, 10}. No choice is made; all results are logged.
3. c4_lr, if plain's LR ≠ 2e-3: ortho at LR 2e-3 for **all three** coefficients instead of one, so the morning decision has the full grid (3 runs, ~2 h extra GPU).
4. final_plain: delete the 4 tier-2 checkpoints, then plain tPD's two final runs (seeds 1, 2, 5,000 steps, checkpoints on WandB). Plain tPD has no coefficient, so nothing about it waits on the coefficient decision.
5. Stop. The recon/ortho final runs start after Julian chooses the coefficients.

Per run the log keeps every candidate criterion for that choice, none of them used yet [assumed: candidates, to discuss]: CI-weighted KL recovered (training prompt and second prompt), Δ only's KL recovered, and the cancellation-test ratios as diagnostics. The training logs also hold tPD's own eval losses: PGD reconstruction, Pile (non-target) KL, the importance-minimality loss / L0.

Expected [assumed]: tuning done ~07:00 UTC, plain finals ~09:15 UTC on 26-10-27.

---

26-10-26, Claude Opus 5.5 — *plain tPD chosen by eye on Julian's three axes*; same session. Julian in chat:

C: (transcribed from chat) I think how much of the verdict the components alone recover is not enought. We also had and discussed other metrics. Like how bad / good is Δ  alone. Can you find that somewhere?

Found: Julian's opening request in this LOG asks for "some undefined pareto optimum" of (a) the KL of the components alone on target data, (b) the KL of Δ alone on non-target data, (c) the shrink of the verdict separation with Δ alone on target data. My answer of 26-10-22 under the axis-(c) comment proposed (c) as Δ only's share of the all-off KL recovered, ideal 0. The driver had used only (a); that was my error.

Asked for a rule, Julian answered (multiple choice, free text):

C: (transcribed from chat) you check the numbers by eye and see which ones seem best

Implemented [decided: transcribed above]:
- The driver pauses at each decision point (plain tPD's LR after c1_lr, importance-minimality after c2_imp), logs "DECISION NEEDED" with every run's numbers, and waits for my decision file `/root/paper_c/decisions/<name>.json` with the value and a reason.
- Per run it logs: (a) CI-weighted components-only KL recovered on val (and under the second prompt, plus accuracy); (b) Δ-only Pile KL, from the training log's final eval; (c) Δ only's KL recovered on val. Also Pile rounded KL, PGD loss and L0, and the cancellation ratios, labelled as diagnostics.
- Fallback [assumed: my safeguard, not asked for]: if no decision arrives within 2 h (this session gone overnight), the driver takes the Pareto front over (a) max, (b) min, |c| min, then the lowest rank sum, ties to the higher (a), and logs FALLBACK.
- Dry run [verified: results on the t300 outputs, the fallback checked by hand (recon dominated by ortho, plain wins the rank sum), a decision read from a file].
- The driver was replaced again while it was only waiting (21:48 UTC); round c1_lr kept training.

---

26-10-26, 21:55 UTC, Claude Opus 5.5 — *HANDOFF: state for the next session*; same session, written at Julian's request before he restarts the agent session. Julian in chat:

C: (transcribed from chat) ok, I want ot restart your session as well, can you write the current state in the log so that another session of you can pick up from there?

**Read first**: this entry, then the entries *overnight plan revised* and *plain tPD chosen by eye on Julian's three axes*. The SUMMARY's *Running* paragraph has the short version.

## What is running

- **Instance**: vast.ai RTX PRO 6000 Blackwell (96 GB), Sevilla, Spain, ssh alias `vastai` (written by `spd-vast`). Hugging Face reachable. Launch remote commands from a login shell (`ssh vastai 'bash -lc "…"'`) so the WandB key is set; detach long jobs with `setsid nohup … < /dev/null &`. Kill by PID, never by a `pkill -f` pattern in the same call (docs/CODING.md).
- **Processes (21:51 UTC)**:
  - `run_round.sh c1_lr` (PID 4996), training its first job `lr5e-4` (PID 5000; step 2,467 / 3,001 after 24 min).
  - `driver.py` (PID 5837), waiting for "round done" in `/root/paper_c/rounds/c1_lr/progress.txt`.
- **Driver**: `/root/paper_c/driver.py`; local copy and source `~/spd_out/26-10-22_tpd_truth_paper/phase_c/driver.py`, docstring = the plan. Log: `/root/paper_c/driver.log`; state: `/root/paper_c/driver_state.json`. Sequence:
  1. c1_lr: plain tPD, LR {5e-4, 2e-3, 5e-3}, importance-minimality 1e-3.
  2. **DECISION `plain_lr`**.
  3. c2_imp: plain at that LR, importance-minimality {3e-4, 3e-3}.
  4. **DECISION `plain_imp`** among {3e-4, 1e-3 (= the chosen c1 run), 3e-3}.
  5. c3_coef: recon and ortho at plain's settings, coefficient {0.1, 1, 10} (6 runs).
  6. c4_lr, only if plain's LR ≠ 2e-3: ortho at LR 2e-3, coefficient {0.1, 1, 10} (3 runs).
  7. Delete the 4 tier-2 `.pth` files from WandB (Julian approved; local copies under `~/spd_out/runs/spd-<id>/model_5000.pth` checked 26-10-26), then final_plain: plain tPD, 5,000 steps, seeds 1 and 2, checkpoints synced to WandB.
  8. Stop, logging "waiting for Julian".
  Every job = `run_one.sh` (training, then `verdict_lens_analysis.py --split val`, both prompts). Any failure: "DRIVER STOPPED" in the log, and the driver exits.

## Your job overnight: the two decisions

- **When**: the driver logs `DECISION NEEDED plain_lr` (expected ~23:30 UTC) and later `DECISION NEEDED plain_imp` (~00:50 UTC). Just before each, it logs one line per run with all numbers; the same numbers are in `/root/paper_c/rounds/<round>/summary.json`.
- **How to decide** [decided: Julian, "you check the numbers by eye and see which ones seem best"]: judge by eye on Julian's three axes. No single metric decides, and **cancellation is not a criterion** (Julian: "No, I don't want cancelation for picking coefficients. At least not before we have looked at alternatives"; same spirit here). The axes:
  - (a) CI-weighted components-only KL recovered on val: higher is better.
  - (b) Δ-only Pile KL: lower is better.
  - (c) Δ only's KL recovered on val: ideal 0, meaning Δ alone neither helps nor harms the verdict.
  - Also visible: second-prompt KL recovered, accuracy, Pile rounded KL, PGD, L0, and the cancellation ratios (diagnostic only).
- **How to answer**: write the file on the instance, e.g.
  `ssh vastai 'mkdir -p ~/paper_c/decisions && echo "{\"value\": 0.002, \"why\": \"…\"}" > ~/paper_c/decisions/plain_lr.json'`
  The value must be one of the candidates (floats: 0.0005, 0.002, 0.005; for imp 0.0003, 0.001, 0.003). Then record the decision and its reason in this LOG, with the numbers.
- **Fallback**: without a file within 2 h, the driver picks by Pareto + rank sum over (a), (b), |c| and logs FALLBACK. That is my safeguard, not Julian's choice; if it fires, say so in the LOG.
- **Monitoring**: this session's background monitors die with it. Start your own, e.g. a `run_in_background` until-loop on `grep -q "DECISION NEEDED\|DRIVER STOPPED" ~/paper_c/driver.log`, and re-arm it after each decision.

## Bookkeeping to do as runs finish

- **Run register** `convos/julian/26-10-03_wandb_storage_RUNS.md`: add a row per run (id from the progress file / driver log). Tuning runs go in *Runs without a checkpoint* ("checkpoint on the instance only"); the final runs go in *Runs with a checkpoint on WandB*. After the tier-2 deletion, remove those 4 rows from the checkpoint table and record the deletion in `convos/julian/26-10-03_wandb_storage_LOG.md` + SUMMARY, with the check that the listing no longer shows them. Check WandB usage (`wandb.Api().client.execute(gql('query { entity(name: "bitt-j-personal") { storageBytes } }'))`, `from wandb_gql import gql`; 3.32 GB before; expected ≈ 4.4 GB after the finals; the counter lags by hours).
- **Copy outputs locally** to `~/spd_out/26-10-22_tpd_truth_paper/phase_c/<round>/` (logs, `analysis/`, `summary.json`, progress), as done for `t300` and `check_4e`.
- **Don't** look at the test split. It stays untouched until the analysis freeze (D3); only val analyses run.

## Morning, with Julian

1. Report the tuning results: plain tPD's choices with reasons; the full recon/ortho coefficient grid on all axes (and c4 if it ran).
2. Julian chooses the recon and ortho coefficients (and ortho's LR if c4 ran). Discuss alternatives to cancellation first: the three axes, PGD, Pile, transfer to the second prompt, L0.
3. Launch their final runs (seeds 1, 2, 5,000 steps, synced) with the driver's `make_round(..., final=True)` pattern. Checkpoint space ≈ 4 × 333.5 MB, which fits after the tier-2 deletion; check the counter first.
4. Then the analysis freeze (D3), the four-category analyses on test / OOD (AUC ≥ 0.7 sets) / second prompt, A5 (scripts into the repo), registry entries (phase E; the WandB project must be public, Julian's call).

## Uncommitted (Julian commits on request)

- `spd/experiments/lm/honesty_targeted_decomposition/verdict_lens_analysis.py`: the second prompt gets `max_seq_len + 5 + 1` (fix for 9 long val statements; checked on CPU on all val and test statements). The instance already runs this version.
- Records: this LOG, the SUMMARY, INDEX.md, the run register (the t300 rows).

## Open questions for Julian (not urgent)

- Should the fallback stay, or should the driver simply wait?
- WandB project public for the registry: when?

---

26-10-26, 21:58 UTC, Claude Opus 5.5 — *new session takes over the overnight watch*; new session. I have read CLAUDE.md, INDEX.md, this topic's SUMMARY, this LOG from *D1 answers* on (incl. the HANDOFF entry), and `driver.py`, `run_one.sh`, `run_round.sh`. Julian in chat: "Can you keep an eye on the current run? Are there any questions before I go to bed?"

State at 21:53 UTC [verified: driver log, progress file, `ps`, `nvidia-smi` on the instance]: driver (PID 5837) waiting for round c1_lr; job `lr5e-4` training since 21:26:40 (GPU 99%). A background watcher on this laptop (`watch.sh` in the session scratchpad) polls `driver.log` every 60 s and wakes me at each DECISION NEEDED / round finished / FALLBACK / DRIVER STOPPED / driver done, or after 15 failed ssh attempts.

Julian's answers (multiple choice in chat), transcribed:

C: (transcribed from chat) [If the driver stops on an error overnight, what may I do?] "Fix bugs, rerun (Recommended)" (option text: diagnose; if a plain bug, not a protocol change, fix it, rerun the failed job/round and restart the driver, log everything; otherwise leave the GPU idle until morning)
C: (transcribed from chat) [Keep the driver's 2 h fallback?] "Keep it (Recommended)"
C: (transcribed from chat) [Will this laptop and session stay awake overnight?] "Yes, stays awake"

So: both plain-tPD decisions (`plain_lr`, `plain_imp`) are made by me by eye, as in the HANDOFF; the fallback stays as a safety net (closes the HANDOFF's first open question).

---

26-10-26, Claude Opus 5.5 — *instance destruction approved*; same session. Julian in chat:

C: (transcribed from chat) another thing. Once everythihg is finished you are allowed to delet the vastai instance

C: (transcribed from chat) Destroy, You can copy them aswell. What do you need me for the decision for the final runs?

(The second message answers my question: destroy after the plain-tPD finals, or keep the instance for the recon/ortho finals? Julian had asked why copy, since the final checkpoints are on WandB. My answer: the tuning runs upload no checkpoints, so their `.pth` files exist only on the instance.)

Plan [decided: transcribed above]: once the driver is done, copy all tuning checkpoints, logs and analyses to `~/spd_out/26-10-22_tpd_truth_paper/phase_c/<round>/`, check the copies by sha256, check that the final checkpoints are listed on WandB, then destroy the instance (`vastai destroy instance`) and record it here.

---

26-10-26, 22:08 UTC, Claude Opus 5.5 — *recon/ortho coefficients by eye too; their finals overnight*; same session. Julian in chat:

C: (transcribed from chat) do the same by-eye judgment. I rather have some result than no result. In the worst case we have to do another final run which isn't a biggy.

OUTDATED (26-10-26): the entry *overnight plan revised* (steps 4–5: "Stop. The recon/ortho final runs start after Julian chooses the coefficients") and the HANDOFF's *Morning* steps 2–3 no longer hold. The recon and ortho settings are now chosen overnight, by me.

So the recon coefficient and ortho's setting (coefficient, plus LR 2e-3 vs plain's LR if round c4_lr runs) are chosen by eye on the same three axes as plain tPD's settings, never by cancellation [decided: transcribed above]. The method is agent judgement, not a rule, so these choices are provisional. Julian expects a possible rerun.

**Driver change** (`driver.py`, local and on the instance; sha256 equal; previous version kept as `driver_v3_backup.py` in both places):
- After c3_coef (and c4_lr) it announces `DECISION NEEDED recon_coef` and `DECISION NEEDED ortho_setting` (ortho candidates are run tags such as `ortho_c1` or `ortho_lr2e-3_c1`). It awaits them only after plain tPD's finals, so the GPU keeps training while I decide. The 2 h fallback starts when it begins to wait.
- Then round `final_rest`: recon and ortho at their chosen settings, seeds 1 and 2, 5,000 steps, checkpoints synced to WandB.
- Right before, a space check that fails loudly: the listed size of all WandB files plus 4 checkpoints must stay ≤ 4.8 GB. Before the night: storage counter 3.33 GB, listed 3.31 GB over 100 runs [verified: API, 22:00 UTC]. Projected after all six finals: 3.31 − 0.95 (tier 2) + 6 × 0.33 ≈ 4.36 GB [assumed: from the listed sizes].
- Dry run on the laptop [verified]: results on the real t300 outputs; fallback and decision files with string keys; the four `final_rest` configs built and validated as `Config` (seed, steps 5,000, LR, coefficient, sync on; s1 vs s2 differ only in seed and name).
- Swap: the old driver (PID 5837) was only waiting; it was stopped by PID at 22:05 UTC and the new one started (PID 6903). Round c1_lr was untouched: `lr5e-4` finished (run `s-f03881a2`, train + analysis exit 0, 22:02), `lr2e-3` training.

Once the driver logs "driver done": copy all checkpoints, logs and analyses locally, check them by sha256, check that the 6 final checkpoints are listed on WandB, then destroy the instance (entry *instance destruction approved*).

Expected timeline [assumed: 36 min per tuning run as measured for `lr5e-4`, ~62 min per final run]: c1 done ~23:15, c2 ~00:30, c3 ~04:20, c4 (if it runs) ~06:10, plain finals ~08:10, recon/ortho finals ~12:20 UTC on 26-10-27.

---

26-10-26, 23:16 UTC, Claude Opus 5.5 — *c1_lr results: plain tPD LR 5e-4*; same session.

Round c1_lr: plain tPD, importance-minimality 1e-3, 3,000 steps, seed 0, val [verified: driver log; all 3 jobs exit 0; outputs copied to `~/spd_out/26-10-22_tpd_truth_paper/phase_c/c1_lr/`]:

| LR | run | (a) comps-only KL rec. | 2nd prompt | accuracy | (b) Pile Δ-only KL | (c) Δ-only KL rec. | Pile rounded | PGD | L0 | max \|C\|/\|y\| (diag.) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5e-4 | s-f03881a2 | **0.905** | 0.936 | 0.965 | **0.091** | **0.035** | 0.091 | 2.03 | 5.6 | 2.08 |
| 2e-3 | s-63e23c1b | 0.847 | 0.708 | 0.903 | 0.361 | 0.264 | 0.361 | 2.37 | 2.7 | 8.46 |
| 5e-3 | s-13746f5b | 0.113 | −0.058 | 0.786 | 1.530 | −0.174 | 0.611 | 2.40 | 2.0 | 25.92 |

**Decision `plain_lr` = 5e-4** [decided by me by eye, under Julian's delegation]: best on all three axes at once (highest (a), lowest (b), (c) closest to 0), and the best second-prompt transfer and PGD loss. No trade-off to judge. Higher LRs give sparser decompositions (L0 2.0–2.7) that hurt the non-target data badly. The cancellation diagnostic grows with LR as well (not used for the choice).

Since 5e-4 ≠ 2e-3, round c4_lr (ortho at LR 2e-3, three coefficients) will run. Round c2_imp started 23:15 UTC (importance-minimality 3e-4 and 3e-3 at LR 5e-4). Run register rows added.

---

26-10-27, 00:30 UTC, Claude Opus 5.5 — *c2_imp results: plain tPD importance-minimality 1e-3*; same session.

Round c2_imp: plain tPD, LR 5e-4, 3,000 steps, seed 0, val [verified: driver log; both jobs exit 0; outputs copied to `~/spd_out/26-10-22_tpd_truth_paper/phase_c/c2_imp/`]. The 1e-3 row is c1_lr's 5e-4 run.

| imp.-min. | run | (a) comps-only KL rec. | 2nd prompt | accuracy | (b) Pile Δ-only KL | (c) Δ-only KL rec. | PGD | L0 | max \|C\|/\|y\| (diag.) |
|---|---|---|---|---|---|---|---|---|---|
| 3e-4 | s-42fd706b | **0.964** | 0.909 | 0.979 | 0.091 | −0.190 | 1.74 | 12.8 | 2.06 |
| 1e-3 | s-f03881a2 | 0.905 | **0.936** | 0.965 | 0.091 | **0.035** | 2.03 | 5.6 | 2.08 |
| 3e-3 | s-e4d259ea | 0.936 | 0.932 | 0.982 | 0.094 | 0.298 | 2.25 | 2.5 | 2.76 |

**Decision `plain_imp` = 1e-3** [decided by me by eye, under Julian's delegation]. This is a real trade-off:
- (b) is tied (0.091–0.094).
- (c) decides: at 1e-3, Δ alone is close to neutral on the verdict (0.035). At 3e-3, Δ alone carries 0.30 of the verdict, which is what targeting should avoid. At 3e-4 it harms the verdict (−0.19).
- 1e-3 also has the best second-prompt transfer (0.936).
- The cost: (a) is 0.06 below 3e-4 (0.905 vs 0.964). (a) is non-monotonic in the coefficient (3e-3 beats 1e-3), so with one seed part of that gap is likely noise [assumed].
- The alternative Julian may prefer: 3e-4, if (a) counts most. Its decomposition is also less sparse (L0 12.8 vs 5.6).

Round c3_coef started 00:29 UTC: recon and ortho at LR 5e-4, importance-minimality 1e-3, coefficients {0.1, 1, 10} (6 runs). Then c4_lr (ortho at LR 2e-3). Run register rows added.

---

26-10-27, 04:45 UTC, Claude Opus 5.5 — *c3_coef results: recon and ortho at plain tPD's settings*; same session.

Round c3_coef: LR 5e-4, importance-minimality 1e-3, 3,000 steps, seed 0, val [verified: driver log; all 6 jobs exit 0; outputs copied to `~/spd_out/26-10-22_tpd_truth_paper/phase_c/c3_coef/`].

| method, coef. | run | (a) comps-only KL rec. | 2nd prompt | accuracy | (b) Pile Δ-only KL | (c) Δ-only KL rec. | Pile rounded | PGD | L0 | max \|C\|/\|y\| (diag.) |
|---|---|---|---|---|---|---|---|---|---|---|
| recon 0.1 | s-614cd983 | 0.674 | 0.819 | 0.871 | 0.063 | −0.230 | 0.063 | 2.65 | 2.9 | 0.75 |
| recon 1 | s-2821a387 | 0.757 | 0.854 | 0.935 | 0.035 | −0.203 | 0.033 | 2.03 | 7.5 | 0.60 |
| recon 10 | s-9fa49cc7 | 0.953 | 0.560 | 0.968 | 0.148 | −0.462 | 0.015 | 0.31 | 48.2 | 0.96 |
| ortho 0.1 | s-baae9857 | 0.343 | 0.449 | 0.848 | 0.051 | 0.197 | 0.051 | 2.40 | 3.0 | 0.37 |
| ortho 1 | s-a53966f6 | 0.523 | 0.703 | 0.829 | 0.037 | 0.259 | 0.037 | 3.00 | 2.6 | 0.36 |
| ortho 10 | s-d383b59a | 0.269 | 0.614 | 0.851 | 0.029 | 0.232 | 0.029 | 2.74 | 3.0 | 0.33 |

First reading (the decisions come when the driver asks, after c4_lr) [concluded, one seed]:
- **recon**: coefficient 1 is ahead on (b) and |(c)| and transfers best to the second prompt. Coefficient 10 wins (a) (0.953), but with dense components (L0 48.2 vs 7.5). Its Δ-only Pile KL is high (0.148) while its rounded Pile KL is the lowest (0.015): at 10 the components carry the non-target data too, so the decomposition is no longer targeted. Its Δ alone also harms the verdict most (−0.46), and its transfer to the second prompt drops to 0.560.
- **ortho at LR 5e-4**: (a) is low at all coefficients (0.27–0.52), Δ alone carries 0.20–0.26 of the verdict. This repeats the 4e screening (0.37 at LR 5e-4, 0.91 at 2e-3). So round c4_lr (LR 2e-3, running since 04:40 UTC) decides ortho.

---

26-10-27, Claude Opus 5.5 — *correction: recon L0 is target-side*; same session. Julian in chat asked which evals the recon tuning used (answered in chat: the three axes plus the logged extras; the method's own `CIHiddenActsReconLoss` was not looked at), then: "You said that at 10 more compenents are active. Did you include that in your decision?"

The recon decision has not been made yet; the driver asks for it after c4_lr. But in the entry *c3_coef results* I used L0 as an argument against coefficient 10, and part of that argument was wrong:
- `eval/l0/0.01_total` (48.2 at coefficient 10, 7.5 at 1, 2.9 at 0.1) is measured on the **target** eval data, the val statements. On the Pile, `eval/nontarget/total_l0` is small for all three: 0.15 / 0.20 / 0.54 [verified: final eval lines in the copied training logs].
- So coefficient 10 is dense on the statements, not on the Pile. OUTDATED (26-10-27), entry *c3_coef results*, the clause "at 10 the components carry the non-target data too, so the decomposition is no longer targeted": too strong. The correct reading [concluded]: on the Pile only about 0.5 components fire per token, but those few carry a lot (rounded KL 0.015 vs Δ-only 0.148).
- L0 is not one of Julian's three axes. Whether target sparsity should count is Julian's question to decide.

---

26-10-27, 06:45 UTC, Claude Opus 5.5 — *c4_lr results; tier 2 deleted; plain finals started*; same session.

Round c4_lr: ortho at LR 2e-3, importance-minimality 1e-3, 3,000 steps, seed 0, val [verified: driver log; all 3 jobs exit 0; outputs copied to `~/spd_out/26-10-22_tpd_truth_paper/phase_c/c4_lr/`]:

| ortho coef. (LR 2e-3) | run | (a) comps-only KL rec. | 2nd prompt | accuracy | (b) Pile Δ-only KL | (c) Δ-only KL rec. | PGD | L0 (target) | max \|C\|/\|y\| (diag.) |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | s-fecf62f5 | 0.766 | 0.577 | 0.829 | 0.116 | −0.307 | 2.93 | 3.0 | 0.58 |
| 1 | s-53f816bf | 0.936 | 0.826 | 0.965 | 0.061 | **0.049** | 2.33 | 2.9 | 0.44 |
| 10 | s-ae91d311 | **0.941** | **0.910** | 0.973 | **0.023** | 0.114 | 1.89 | 2.6 | 0.39 |

The driver then announced `DECISION NEEDED recon_coef` and `ortho_setting` (awaited only after the plain finals). It deleted the 4 tier-2 checkpoints (recorded in `convos/julian/26-10-03_wandb_storage_LOG.md`), and round final_plain started at 06:32 UTC.

Not decided yet: Julian is awake and asked about the recon choice (entry *correction: recon L0 is target-side*), so I put both choices to him first.

---

26-10-27, Claude Opus 5.5 — *recon: Julian expects many active components; own reconstruction error favours coefficient 10*; same session. Julian in chat:

C: (transcribed from chat) It's natural for the amount of components to go up for the tPD + reconstruction. I susspect the best one to have many active components.

C: (transcribed from chat) Why do you prefer 1 over 10 in orthogonality?

**Ortho**: a misreading. In chat I proposed ortho at LR 2e-3, coefficient **10**, not 1 (it wins (a), (b), second prompt and PGD; it loses only (c), 0.114 vs 0.049).

**Recon**: the method's own error, read now from the final training evals. `eval/loss/CIHiddenActsReconLoss` is the absolute MSE (not relative; `calc_hidden_acts_mse` default) between the CI-masked components' and the original `down_proj` outputs at the decomposed positions, on the val statements [verified: copied training logs; code read]:

| run | CI-masked hidden-acts MSE |
|---|---|
| plain (LR 5e-4, imp 1e-3) | 0.968 |
| ortho LR 2e-3, coef. 1 / 10 | 0.854 / 0.741 |
| recon coef. 0.1 / 1 / 10 | 0.726 / 0.686 / **0.040** |

Reading [concluded, one seed]: only coefficient 10 actually reconstructs the MLP outputs. At 1 the term barely bites (0.69, against 0.97 for plain tPD without it). So coefficient 10 is the run that does what "+ per-layer reconstruction" claims, and Julian's expectation of many active components fits: reproducing the whole output needs many components (L0 48 on the statements). This also undercuts my axis-based preference for 1. (c) is close to neutral only at a point where the components don't reconstruct. When the components carry the whole output, Δ alone is a remainder, and its strongly negative (c) at 10 (−0.46) may be partly by construction [assumed].

Costs of coefficient 10 that remain real: Δ-only Pile KL 0.148 (plain 0.091; the ~0.5 components firing per Pile token carry much of it), and the weakest transfer to the second prompt (0.560 vs 0.854 at 1).

Revised proposal [assumed, awaiting Julian]: recon coefficient 10; ortho LR 2e-3, coefficient 10.

---

26-10-27, Claude Opus 5.5 — *should transfer count in the selection?*; same session. Julian in chat:

C: (transcribed from chat) Should we factor transfer into the eval?

My answer [assumed: recommendation, awaiting Julian]: report transfer in the evaluation (phase D), but **don't use it for selection**.
1. **It is our held-out generalisation test.** Selecting on it would spend it: a method tuned for transfer can no longer offer its transfer as evidence that the components carry something prompt-independent. This holds although tuning is on val and the finals are evaluated on test. The prompt itself would no longer be unseen.
2. **It only affects recon.** Plain tPD's LR and importance-minimality were also the best on transfer, and ortho at coefficient 10 wins transfer too (0.910). For recon it would flip the choice back to 1 (0.854 vs 0.560 at 10).
3. **What the second prompt tests is narrower than "transfer"** [concluded from the prompt construction, `verdict_lens_analysis.py` `SECOND_STIMULUS`]. The model is causal, and the statement is a prefix of both prompts. So every statement token except the last has identical activations and CIs under both prompts. The last token differs: '.\n' under the training prompt, '.' under the second. Transfer differences can therefore only come from (i) the components at a final token never seen in training, and (ii) the second prompt reading the MLP outputs differently.
   - For recon at 10, whose components reproduce the MLP outputs almost exactly (error 0.04), (ii) should be small at the shared tokens. So its drop to 0.560 probably sits at the final token, which carries most of the verdict (the end token alone keeps 82% of the separation, causal layer scan) [assumed: not checked].
   - That is a finding about recon (reconstruction fitted to one end token doesn't generalise to another), not a flaw that should decide the coefficient.
4. **Proposal for phase D** [assumed]: add a third readout whose statement ends with the same '.\n' token as in training, e.g. "<statement>\nIs this statement true (1) or false (0)? Answer: ". It separates readout transfer from end-token transfer. And a per-position check of recon's error at the final token under the second prompt.

---

26-10-27, Claude Opus 5.5 — *recon and ortho settings decided*; same session. Julian in chat:

C: (transcribed from chat) OK, agree on the coefficients. No need to add another eval for now

- **recon coefficient 10; ortho LR 2e-3, coefficient 10** [decided: transcribed above]. Reasons in the entries *recon: Julian expects many active components…* and *c4_lr results…*.
- Transfer to the second prompt is not a selection criterion (my recommendation, entry *should transfer count in the selection?*; Julian agreed on the coefficients that follow from it, so this is [concluded], not separately ratified). No third readout for now [decided: transcribed above].
- Decision files written on the instance (`recon_coef.json` = 10, `ortho_setting.json` = `ortho_lr2e-3_c10`) [verified: read back]. The driver reads them after round final_plain (running since 06:32 UTC), then runs the four recon/ortho finals.

Julian in chat, clarifying "No need to add another eval":

C: (transcribed from chat) I mean phrase for eval

So the decision is: no third prompt phrasing for the evaluation for now (the proposed '.\n'-ending readout of the entry *should transfer count in the selection?*, point 4). The per-position check of recon at the final token is a separate analysis, and it was not ruled out.

---

26-10-27, 08:40 UTC, Claude Opus 5.5 — *plain finals done; recon/ortho finals started*; same session.

Round final_plain: plain tPD, LR 5e-4, importance-minimality 1e-3, 5,000 steps, checkpoints synced to WandB; val analysis only [verified: driver log; both jobs exit 0; outputs copied to `~/spd_out/26-10-22_tpd_truth_paper/phase_c/final_plain/`]:

| seed | run | (a) comps-only KL rec. | 2nd prompt | accuracy | (b) Pile Δ-only KL | (c) Δ-only KL rec. | PGD | L0 (target) | max \|C\|/\|y\| (diag.) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | s-7152548c | 0.964 | 0.946 | 0.979 | 0.102 | 0.097 | 1.64 | 3.7 | 2.02 |
| 2 | s-46efc525 | 0.963 | 0.926 | 0.980 | 0.098 | −0.056 | 1.29 | 3.8 | 1.91 |

The two seeds agree closely on (a) and (b). (c) differs in sign (0.097 vs −0.056), so (c) has a seed spread of order 0.1 [concluded: two seeds]. That puts the (c) differences used in the plain importance-minimality and ortho choices within noise. Those choices were made on one seed.

The driver then read both decisions and ran the WandB space check: listed 3.05 GB, 4.39 GB after 4 more checkpoints (limit 4.8) [verified: driver log]. 3.05 ≈ 3.31 − 0.95 + 2 × 0.334, consistent with both plain final checkpoints (333,524,341 B each) being on WandB. Round final_rest (recon s1, s2; ortho s1, s2) started 08:33 UTC. Run register rows added.

---

26-10-27, 13:35 UTC, Claude Opus 5.5 — *all finals done; outputs copied; instance destroyed*; same session.

Round final_rest: 5,000 steps, importance-minimality 1e-3, checkpoints synced, val analysis only [verified: driver log; all 4 jobs exit 0; "driver done" 13:10 UTC]:

| method, seed | run | (a) comps-only KL rec. | 2nd prompt | accuracy | (b) Pile Δ-only KL | (c) Δ-only KL rec. | Pile rounded | PGD | L0 (target) | max \|C\|/\|y\| (diag.) |
|---|---|---|---|---|---|---|---|---|---|---|
| recon (coef. 10), 1 | s-eeac663f | 0.977 | 0.968 | 0.971 | 0.141 | −0.239 | 0.015 | 0.30 | 37.8 | 0.96 |
| recon (coef. 10), 2 | s-edc17041 | 0.972 | 0.955 | 0.976 | 0.140 | −0.433 | 0.014 | 0.31 | 38.5 | 0.96 |
| ortho (LR 2e-3, coef. 10), 1 | s-0c2fa0d7 | 0.907 | 0.934 | 0.975 | 0.024 | 0.162 | 0.024 | 2.75 | 2.8 | 0.43 |
| ortho (LR 2e-3, coef. 10), 2 | s-4186f973 | 0.906 | 0.808 | 0.956 | 0.023 | −0.061 | 0.023 | 2.54 | 3.2 | 0.42 |

With the plain finals (entry *plain finals done…*: (a) 0.964 / 0.963, second prompt 0.946 / 0.926, (b) 0.102 / 0.098), first readings [concluded: two seeds, val only; not the analysis]:
- (a) is very stable across seeds in every method (spread ≤ 0.005). (c) is not: it changes sign between seeds in plain and ortho, and spans 0.19 in recon. So (c) is too noisy to separate methods at this sample size.
- Second-prompt transfer is high for recon at 5,000 steps (0.968 / 0.955), against 0.560 in the 3,000-step tuning run at the same coefficient. OUTDATED (26-10-27), entry *should transfer count in the selection?*, point 3, the sentence "its drop to 0.560 probably sits at the final token…": the drop did not persist in the finals. It was a property of that one shorter run, not of recon at coefficient 10. The construction argument (shared statement prefix, differing last token) still holds. Ortho's transfer varies across seeds (0.934 / 0.808).
- (b) separates the methods: ortho 0.023, plain 0.10, recon 0.14. recon's rounded Pile KL is the lowest (0.015).

**Copies** [verified: sha256 of all 696 files, instance vs laptop, all equal; 23 `.pth`]:
- All 23 run directories of the instance (3 trials, 14 tuning runs, 6 finals; `final_config.yaml`, checkpoint, metadata) → `~/spd_out/runs/spd-<id>/`, the existing convention for local checkpoint copies.
- The whole driver directory `/root/paper_c` (driver, decisions, logs, all rounds) → `~/spd_out/26-10-22_tpd_truth_paper/phase_c/instance_paper_c/`.
- The instance's copy of the 4e checkpoint `s-eb1c20d5` was not copied back: it came from the laptop, and `~/spd_out/26-10-16_faithful_targeting/e4/checkpoints/s-eb1c20d5/model_3000.pth` exists.

**WandB** [verified: API 13:20 UTC]: all 6 final runs `finished`, each with one `model_5000.pth` of 333,524,341 B. Counter 3.81 GB (it lags uploads); listed projection 4.39 GB.

**Instance destroyed** (13:30 UTC; vast id 57869398, RTX PRO 6000, Sevilla) [decided: Julian, entry *instance destruction approved*; verified: `vastai show instances` lists none]. About 16.7 h rented at $1.46/h, ≈ $24 [assumed: from vast's age and price columns].

Next (phase D, with Julian): freeze the analysis plan (D3), then the four-category analyses on test / OOD / second prompt. Registry entries come after the analysis freeze (phase E). Uncommitted: `verdict_lens_analysis.py` (second-prompt length fix) and the records.

---

26-10-27, Claude Opus 5.5 — *D3 proposal: the analysis plan to freeze*; new session. I have read CLAUDE.md, INDEX.md, this topic's SUMMARY, this LOG's *first pass* (sections 3–8 with Julian's comments), *plan of steps*, *D1 answers*, *tuning budget revised* and the last two entries, and the faithful-targeting SUMMARY (4e analyses, A8, reprogramming). Julian in chat: "I'd like to do the next step of the analysis. We finetuned and trained some decompositions. What is next?"

**Next step = D3**: freeze the analysis plan (metrics, controls, data, decision rules), then phase D's analyses on test / OOD / second prompt. Below is my draft; nothing in it is fixed until Julian ratifies it.

**Order slip to note** [concluded: plan of steps vs the entries above]: the plan put the freeze *before* the final runs (*first pass* section 7, Julian: "fixing that and the final runs comes at the very end"). The finals ran first, overnight. Little is lost: only val numbers of the finals were looked at (tables in *plain finals done…* and *all finals done…*); test and the OOD sets are untouched, and none of the analyses below has been run on the finals. The report should say that the plan was fixed after the finals' val readout and before any test/OOD analysis.

## Draft analysis plan [assumed: proposal]

**Objects.** 6 finals (plain, recon, ortho × seeds 1, 2). Controls, each used wherever a category compares "decomposition vs not":
- **Untrained decomposition**: the same config at `steps: 0` (as `config_truth_untrained.yaml` did for arm A), seeds 1 and 2. At initialisation the three configs are identical (the method term acts only in training), so 2 untrained checkpoints cover all methods.
- **Data-aligned basis**: per layer the top-96 principal directions of the MLP's inputs on train statements, write vectors u = W v (the `--pca_basis` of `mask_search_4e.py`). It matched 4e's components at verdict swapping, so it is the control that matters most.
- **Data-blind random basis / size-matched random component sets**: weaker control, reported for scale.

**Data and prompts.** tiu v3 test (in-distribution); the 24 OOD files with verdict-prompt AUC ≥ 0.7 for verdict-target analyses; the 13 OOD files above the stricter bar for ground-truth analyses [decided: *D1 answers*, point 2]. Two prompts: the training verdict prompt and the second prompt "Is this statement true (1) or false (0)? Answer: " (held-out readout). The 4 excluded files are not analysed.

**Conventions** [decided earlier unless tagged]: CI-weighted state; macro-average over families; both labels (ground truth and the model's own verdict) in every analysis; all metrics per seed, never pooled across seeds. Uncertainty: bootstrap over statements within family, 1,000 resamples, 95% intervals [assumed: proposal].

**1. Recovery** (cheapest; `verdict_lens_analysis.py`, already in the repo)
- Components only (Δ off) KL recovered and verdict agreement with the original model; on test, per OOD file and for the second prompt. Untrained decomposition as floor.  
- Reported alongside: Δ only (axis (c)), Δ-only Pile KL (axis (b)), PGD loss, and the cancellation numbers |C|/|y|, ⟨C, Δx⟩/|y|² as a diagnostic only (Julian rejected cancellation as a criterion).  
C: Is Δ only on target recovery? Should it be "necesity" or something?  
Answer (26-10-27, Claude Opus 5.5): Yes, it's necessity. Δ only = the full model with all components removed, so it measures whether the components *together* are necessary. I move it to termination as T0 (removal of all components). The Δ-only Pile KL (axis (b)) is neither recovery nor necessity: it says whether the model stays intact off-target. I report it in its own line, *side effects*. Recovery is then only components-only, on test, OOD and the second prompt.
- Question it answers: does the components' verdict hold off the training families (German, conj/disj, inventors, facts)?

**2. Termination** (necessity; `ablate_4e.py`, `compare_decomps.py`)
- T1 single removal of each active component from the full model (Δ kept): drop of the verdict separation, flips true→false and false→true, per family.
- T2 removal of the top-k components (k = 1, 2, 5, 10, 20, all active), ranked on val, measured on test; against the same k of size-matched random components, of the untrained decomposition and of the data-aligned basis. Show the whole curve over k.
- T3 = A8: remove from the *original* MLP output its projection onto the span of the components' write vectors, per layer; controls: random subspaces of equal dimension, top principal directions. Two spans: all 96 write vectors (as in 4e) and only the active components (open proposal from the faithful-targeting topic).

**3. Correlation** (single components, since a probe over all CIs only reads the activations)  
C: yes, but we need the data as well to show that it's not telling  
Answer (26-10-27, Claude Opus 5.5): Agreed, if I read you right: the probe numbers go in, as the reason why correlation has to be judged on single components. New item K0: a probe over all CIs for each final run vs the untrained decomposition vs the residual stream, on test and OOD. If the arm A pattern repeats (untrained as good as trained), that is the shown evidence that CI probes say nothing tPD-specific. If you meant something else by "the data", please say.
- K1 sparsity: single-component AUC within family, for the label and for the model's verdict; best k-component logistic probe (k = 1, 2, 5) against the untrained decomposition and the data-aligned basis at the same k.
- K2 transfer: components selective on val (|AUC − 0.5| ≥ 0.2, selection fixed on val) keep their sign on the OOD files of the same sentence form (negated → neg_*/German negated, affirmative → inventors/facts/German, composition → conj/disj). Measure: share that transfers (same sign, |AUC − 0.5| ≥ 0.1), against the same share for the controls.
- K3 truth vs verdict: on statements where the model's verdict disagrees with the label, does the component follow the label or the verdict? Mostly an OOD analysis (in-distribution test has few errors). This one uses all 24 files by necessity, since the error-free sets have no disagreements.

**4. Manipulation** (most expensive; `mask_search_4e.py`)
- M1 mask search fit on train, tested on test, curve over the number of terms, for every final decomposition and every control basis. Outcomes: verdict under the training prompt, the same edit under the second prompt, and recall (the B3 recall readout for cities and elements) to separate "changes the verdict" from "changes the belief".
- No new single-component reversal experiments: the steering topic and B3 already show that verdict levers are readout levers; M1 with recall tests whether that holds for the paper's runs.  
C: I'd like to talk about manipulation a bit longer. What manipulation experiments did seem to have been successful so far? I want to use these as a basis for sensible results. My understanding is that training a manipulation gives no real results and also does not get us further in understanding the decompositions.  
Answer (26-10-27, Claude Opus 5.5): I agree about trained manipulation. What the records show [verified there, one seed each unless noted]:
- **Worked and taught us something: reversing single identified components** (component-steering topic, 4c). Reversing L17:5 (mask −1) sends 42.6% of correctly answered false statements to "1" (random single components ≤ 0.7%), but only in the negated families and smaller_than. That told us what the component does: negation × relation ("the 'not' is ignored"), not a general "false" writer. Gap: no data-aligned control was run.
- **Worked, but is termination, not manipulation**: removals. 4e at importance-minimality 3e-3: removing its 41 false-selective components calls 75% of the false statements "true", and true ones stay true. 4e at 1e-2: removing L11:22 inverts the larger_than verdicts. A8: removing the components' subspace from the original L11 output flips 22–28% of false statements (both seeds), and random subspaces of the same size do nothing. These are the strongest causal results so far, and they sit in section 2.
- **Worked, but no decomposition involved**: truth-direction steering at the statement tokens flips the verdict (B3: 0.975 under the verdict prompt), and recall stays intact: a truth tag, not a belief.
- **Trained manipulation (gradient mask search)**: it finds edits that swap verdicts, but on 4e an untrained data-aligned basis does it as well (129 terms → accuracy 0.057), and the belief-shaped objective gives verdicts by sentence form for every basis, random included. So it tells us about the readout, not about the decomposition. The one point in its favour (decompositions slightly better with ≤ 20 terms in the affirmative fact families) is weak.

**Proposal** [assumed]: drop M1 (mask search) from the frozen plan. Replace it with hypothesis-driven manipulation of the components that termination and correlation single out:
- M1′: for the components that are individually necessary in T1 (and the label-selective ones from K1), reverse (×−1) and amplify (×2) each one and each group, per family, under both prompts. Controls: the same operation on size-matched random components and on the data-aligned basis directions with the largest effect on the verdict (the control the 4c steering lacked).
- M2′: test the reading each one suggests. E.g. for a negation × relation component, the paired-negation test proposed in the steering topic: under reversal, does a negated statement's score follow its affirmative twin? Since the tests depend on what T1/K1 find, the plan fixes only the form (the hypothesis comes from T1/K1 on val, the test runs on test), not the hypotheses.
- Recall readout only where a manipulation flips affirmative fact families. Everything so far changed negated or compared statements, where recall of the fact is not affected.

**Decision rules** [assumed: proposal; they are reading rules, not a definition of "mechanism", per Julian's comment in *first pass* section 4]:  
C: what do we need to decide on? Don't we just do experiments and then interpret the results?  
Answer (26-10-27, Claude Opus 5.5): Mostly yes. What the freeze needs is the *list*: which experiments, which controls, which data, and that all of them get reported, including those that come out badly. With many analyses × 3 controls × 2 seeds × 24 OOD files, choosing what to show after seeing the results is the risk, not the interpretation itself. So I'd drop the thresholds ("counts as decomposition-specific only if …") as rules and keep two reporting conventions instead: every number is shown with both seeds and its controls next to it, and no method ranking where the two seeds overlap. Interpretation afterwards, as you say. That also fits your "no fixed definition" from *first pass* section 4.
- An effect counts as **decomposition-specific** only if it holds in both seeds and exceeds the data-aligned basis (and the untrained decomposition) by more than the larger of the seed spread and the bootstrap interval.  
C: what are the seed spread and bootstrap interval?  
Answer (26-10-27, Claude Opus 5.5): Two sources of noise in a number:
- **Seed spread**: the difference between the two final runs of a method (seeds 1 and 2), i.e. how much the result depends on the training's randomness (initialisation, data order, masks). Example from the finals on val: plain tPD's components-only KL recovered 0.964 vs 0.963 (spread 0.001); Δ only 0.097 vs −0.056 (spread 0.15). So the first number is stable, the second is not.
- **Bootstrap interval**: how much a number would change with a different sample of statements of the same size. Draw the test statements with replacement 1,000 times, recompute the metric each time; the middle 95% of the results is the interval. It matters most for small cells (German files: 49–50 statements each; test cells as small as 18 statements, animal_class_true).
Seed spread is noise of the decomposition, the bootstrap is noise of the test set. With 2 seeds the spread is a crude estimate, which is why I proposed no ranking when they overlap rather than a formal test.
- **Methods are ranked** on a metric only when both seeds of one method lie outside the other method's two-seed range; otherwise the report says "not separated" (my caution in *first pass* section 7).  
- Everything not listed above is **exploratory** and labelled so in the report (e.g. inspecting what a single component reads).

**Execution order**: (i) A5: move `ablate_4e.py`, `compare_decomps.py`, `mask_search_4e.py` (+ `attrib_4e.py` only if needed; A6 was not used as causal evidence) into the repo next to `verdict_lens_analysis.py`, generalised to the paper runs and the controls, with one reproduction check each on the 4e checkpoint `s-eb1c20d5` (val) against the stored outputs in `~/spd_out/26-10-16_faithful_targeting/e4/`; tests where feasible. The move is CPU work; the reproduction checks need a GPU and open the rental. (ii) Build the 2 untrained checkpoints and the data-aligned bases. (iii) One rental: recovery → termination → correlation → manipulation, in order of cost. GPU time not estimated yet; I'd measure it with the reproduction checks.

## Questions for Julian

1. Ratify the plan, or which parts to change? In particular the decision rules.  
C: I already had some comments. Another commet is about active components. It's basically two:
- we have seen that inactive components can play a major role in the functioning of the decomposations. I want to have the numbers for this as well. As an additional finding.
- one caveat on active component is that it uses the random cutoff point. I'd like to have a motivated cutoffpoint instead.
Answer (26-10-27, Claude Opus 5.5): Both make sense, and both are cheap (no training).

**(i) Role of the inactive components** [assumed: proposal]. Earlier evidence: in 4a, inactive components and Δ cancel a layer-11 write 12.9× the MLP output's norm; in 4d they each wrote 3–11 |y|, cancelling; in 4e the integrated-gradients split gave the inactive components −1.16 of 8.23 (path-dependent, not causal). Measures per final run and control, test and both prompts:
- KL recovered with only the active components (CI-weighted, others off, Δ off) vs all components CI-weighted: the difference is what the inactive ones contribute.
- Removing the inactive components from the full model (active ones and Δ kept): change of verdict and separation.
- Their size and direction per layer: |I|/|y| and cos(A, I), which `verdict_lens_analysis.py` already logs.
- Plus the mean-mask state (CI + (1 − CI)/2), which separates "inactive components matter" from "they matter only through the training's random masks" (your low-CI averaging hypothesis of *first pass* section 6). Reported as a hypothesis, as you decided there, unless the numbers are clear.

**(ii) A motivated cutoff.** The current thresholds are CI > 0.01 (the rounding rule of the eval metric, `verdict_lens_analysis.py`) and "active in ≥ 2% of statements" for candidate lists; both are arbitrary. Proposal [assumed]:
- **Counts without a cutoff**: per statement, the sum of the CIs. At the end of training the importance-minimality p-norm has annealed to 1 (`pnorm: 2.0`, `p_anneal_final_p: 1.0` in the paper configs [verified: config]), so this is close to the quantity the loss penalises.
- **The active set by function**: rank the components by mean CI on val statements; the active set is the smallest top-m set whose CI-weighted components-only state (others off, Δ off) recovers ≥ 95% of what all components recover. That is "the components that do the job", one set per run, no per-statement threshold. The 95% is still a choice, but on a scale with a meaning. The whole curve over m is reported, so a reader can see that nothing hinges on it.
- This set is used for A8 ("active" span), as T1's candidate list and for (i). As a check, the main results are also given at the old threshold 0.01.
OUTDATED (26-10-27): "it probably doesn't exist" in the next sentence is overstated: 30% of the mass in the middle does not rule out a dip there. To be checked, see the answer under Julian's comment below.  
Alternative I considered: a valley in the CI histogram. It probably doesn't exist: in the 4d trial 30% of the active CIs lay strictly between 0.01 and 0.99.  
C: I think there might be a valley. We should look for it at least.
Answer (26-10-27, Claude Opus 5.5): Agreed, and I was too quick above (marked OUTDATED). How to look [assumed: proposal]:
- Per final run and the untrained decomposition: histogram of all CI values (every component × statement token × layer, val), log-scaled counts, also per layer; and the histogram of the per-component mean CI, which the set definition uses.
- A valley = a minimum between a mode near 0 and a mode near 1 that appears in both seeds of a method.
- If there is one: it defines "active" at the level of a component on a token (counts, the active share per statement), and I compare it with the functional set. If there is none: the histogram goes in the report as the reason for the functional set.
- The data don't exist yet [verified: the finals' analysis files `verdict_s-*_val.npz` store only scores and KLs; no CIs saved in the 4e outputs either]. It needs a forward pass through layers 11–17. This laptop has no GPU, so either the first item of the rental, or now on the CPU (bf16 ≈ 15 GB of the 40 GB free, forward only up to layer 17, a few hundred val statements; runtime not measured).
C: I don't want CI > 0.01 = active component definition in the results.
Answer (26-10-27, Claude Opus 5.5): Noted [decided: C: comment]: the threshold 0.01 does not appear as a definition of "active" in any result, not even as a check. Where it currently enters: the rounded state and `n_active` of `verdict_lens_analysis.py`, and the "active on ≥ 2% of statements" candidate lists of the 4e scripts. In the paper versions (A5) these are replaced by the functional set (and the valley, if one exists). The CI-weighted state needs no threshold.
2. Scope for cost: keep M1 (mask search for 6 runs × 4 bases, with recall), or cut it to one seed per method?  
C: We talk about the manipulating some more, right?  
Answer (26-10-27, Claude Opus 5.5): Yes: my answer under your manipulation comment in section 4. My proposal drops the mask search, so this cost question goes away.
3. A8 span: all 96 write vectors, active components only, or both (my proposal)?  
C: both + what should count as ative  
Answer (26-10-27, Claude Opus 5.5): Both, noted [decided: C: comment]. "Active" per the proposal under your comment to question 1: the smallest top-m set by mean CI that does the components' job.
4. Shall I start A5 (scripts into the repo, CPU, reproduction checks need a GPU later) once the plan is fixed?
C: yes  
Answer (26-10-27, Claude Opus 5.5): Noted [decided: C: comment]: A5 starts once the plan is fixed.

C: another comment. I also want to investigate the connection of truth direction and the decomposition. For that I'd suggest that we compare the decomposition writes to the mass-mean and also logistic regression probes  
Answer (26-10-27, Claude Opus 5.5): Good, I add it as section 5, *truth directions* [assumed: design proposal].
- **Directions**: mass-mean and logistic regression, fit on v3 train, in the residual stream after each layer 11–17 (where that layer's MLP writes), at the last statement token. Because the statement comes first, the residuals at the statement tokens equal those of the bare statement except at the last token (`'.\n'` instead of `'.'`) [concluded: causal attention], so they are close to the directions used so far.
- **Per component**: cos(write vector, direction). Baseline: random vectors (≈ 0.017 in 3,584 dimensions, as in A4) and the data-aligned basis.
- **Share of the truth gap**: the projection of the CI-weighted writes of the components vs Δ onto the direction, mean true minus mean false (as for 4c, components 13% / Δ 73% at the end token, and 4e, components 0–7%).
- **Link to function**: do the components that matter in T1 or flip things in M1′ write along the direction? And the other way: does removing the direction from the MLP output (as in A8, one dimension) cost the verdict that removing the components costs?
Expectation from the records [concluded there]: the components write little along the direction; Δ holds most of the gap but gives no verdict alone; B3 showed the direction acts as a truth tag. So the likely result is "the verdict-relevant part is not the truth-direction part". That is a result in its own right, but the analysis should not be framed as if we expect alignment.


---

26-10-27, Claude Opus 5.5 — *D3 plan, revision 2 (after Julian's comments)*; same session. Julian in chat: "OK, I gave you some more comments." Answers are in place under each comment in *D3 proposal*; this entry collects the plan as it now stands [assumed: not ratified; parts tagged decided come from C: comments].

**Changes against the draft:**
- **Recovery**: components only (Δ off), on test, OOD, second prompt. Δ only moves to termination (T0 = remove all components). Δ-only Pile KL becomes its own line, *side effects*.
- **Termination**: T0 (Δ only), T1 single removals, T2 top-k curves, T3 = A8 with both spans (all 96 write vectors and the active set) [decided: C: comment].
- **Correlation**: new K0, the probe over all CIs (trained vs untrained vs residual stream), shown as the evidence that CI probes are not tPD-specific; then K1–K3 as drafted.
- **Manipulation**: mask search dropped (proposal; Julian's view that trained manipulation teaches little, and on 4e a data-aligned basis did as well). Instead M1′: reverse (×−1) and amplify (×2) the components singled out by T1/K1, with size-matched random and data-aligned controls; M2′: hypothesis tests of the reading each component suggests (form fixed now, hypotheses from val, tests on test); recall only where affirmative fact families flip. Open for discussion, as Julian asked.  
C: Can you explain this in more detail. Does that mean we have to investigate each component by hand?
Answer (26-10-27, Claude Opus 5.5): No. Selection and measurement are automatic. Only reading a handful of top components is done by hand, and that part is labelled exploratory. In detail [assumed: proposal]:

**M1′, automatic.**
- **Which components**: per run, the 10 components of the functional active set with the largest single-removal effect on the verdict separation (T1, on val), plus the label-selective group from K1. In 4e seed 0 six components mattered individually in the full model, so 10 should cover the large ones; the T2 curve shows whether it does.
- **What is done**: each candidate and the group reversed (×−1) and amplified (×2) at the statement tokens. Removal (×0) is already T1.
- **What is measured**: per family and label, under both prompts, on test and OOD: share of true statements going to "0", of false going to "1", the separation, the digit mass (does the answer stay a digit).
- **Controls, selected the same way**: the 10 data-aligned basis directions and the 10 untrained-decomposition components with the largest removal effect on val, plus 10 random components from the active set. Selecting the controls by effect too matters: otherwise the decomposition gets a selection advantage the controls don't.
- **Output**: one table per run. About 60 components in all (10 × 6 runs), forward passes only.

**M2′, two levels.**
- *Automatic profile* per candidate, from K1/T1/M1′ on val: in which families, sentence forms (affirmative, negated, comparison) and labels it is active, and where reversing it acts. From that one of three fixed types: sentence-form detector (active for true and false alike within a form), label-selective within a form, family-specific. Each type has a prediction that runs automatically on test/OOD: the first two must transfer to OOD files of the same form (K2), and for negation components the paired-twin test (tiu has affirmative/negated twins with the same subject and object; under reversal, does the negated statement's score follow its affirmative twin's?).
- *By hand, exploratory*: for the top 3 per run, reading example statements where the component fires and continuations after a flip, as in the 4c L17:5 analysis.

**Honest caveat and recommendation**: M2′'s typing overlaps with K1/K2. The cheapest version that keeps the useful part: M1′ automatic, plus the paired-twin test as the only fixed hypothesis test, plus the hand-reading as exploratory. I'd take that version.
- **New, active vs inactive components** (Julian's comment): the inactive components' contribution (active-only vs all CI-weighted, removal of the inactive ones from the full model, |I|/|y| and cos(A, I), mean-mask state). Counts as the sum of CIs; the active set = smallest top-m set by mean CI that recovers ≥ 95% of the components-only KL recovered (curve over m reported; main results also at the old 0.01).  
C: yes, I like this.
Answer (26-10-27, Claude Opus 5.5): Noted [decided: C: comment]: counts = sum of CIs; active set = the smallest top-m set by mean CI recovering ≥ 95% of the components-only KL recovered, with the curve over m reported. Two amendments from your other comments: no results at the old threshold 0.01, and the valley check (answer under your valley comment).
- **New section 5, truth directions** (Julian's comment): mass-mean and logistic-regression directions after layers 11–17 at the last statement token (v3 train); per component cos(write, direction) vs random and data-aligned basis; the components' vs Δ's share of the truth gap; whether the functionally important components write along it; removing the one direction from the MLP output vs removing the components.
- **Decision rules → reporting conventions**: all listed analyses are reported whatever their outcome; every number with both seeds and its controls; no method ranking where the seeds overlap. No thresholds for "decomposition-specific".
- Unchanged: controls (untrained decomposition at `steps: 0`, seeds 1, 2; data-aligned basis; random sets), data and prompts, CI-weighted, macro-average, both labels, bootstrap intervals. A5 starts once the plan is fixed [decided: C: comment].

**Still open for Julian**: the manipulation part (M1′/M2′), the 95% in the active-set definition, and the plan as a whole.

---

26-10-27, Claude Opus 5.5 — *D3 plan, revision 3*; same session. Julian in chat: "I've added some more comments." Answers in place: under the valley and 0.01 comments (in *D3 proposal*), under the manipulation and active-set comments (in *D3 plan, revision 2*).

Changes against revision 2:
- **Active components** [decided: C: comments]: counts = sum of CIs; active set = smallest top-m set by mean CI recovering ≥ 95%, curve over m reported. CI > 0.01 does not appear as a definition of "active" in any result. New: look for a valley in the CI histograms (finals, both seeds, and the untrained decomposition); if one exists it defines "active" per token and is compared with the functional set. Needs a forward pass: no CIs are stored anywhere [verified: output files].
- **Manipulation** [assumed: proposal]: M1′ automatic (top 10 per run by T1 effect on val + the K1 label-selective group; ×−1 and ×2; per family and label, both prompts, test and OOD; controls selected the same way from the data-aligned basis and the untrained decomposition, plus random active components). Fixed hypothesis test: only the paired-twin test for negation components. Profiles and hand-reading of the top 3 per run: exploratory.

Still open for Julian: the manipulation version (the cheap one above, or with M2′'s automatic typing); whether the valley check runs now on the CPU or first in the rental; the plan as a whole.

C: You still need to explain to me what "cheap one above and M2′ are and differ. I simply don't have a good idea about manipulationg. Valley check on gpu. Plan as a whole is fine (except that last issue about manipulating that I need to understand first)  
Answer (26-10-27, Claude Opus 5.5): Noted [decided: C: comment]: valley check on the GPU (first item of the rental); the plan stands except manipulation. Explanation:

**What manipulation adds.** Termination asks "is the component needed?" (remove it, does the verdict suffer?). Manipulation asks "does pushing on it move the verdict in a *direction*?" (reverse it or make it stronger, does the verdict go the way its role predicts?). Removal can only take a signal away; reversal can turn "this is false" into "this is true". Example from the records: removing 4c's L17:5 cost 15% of the verdict separation; reversing it sent 43% of the false statements to "1", but only in negated families and smaller_than. That second result is what told us the component handles negation, not truth in general. What manipulation cannot show: a change of belief. B3 and the 4c recall test showed that verdict flips leave the model's knowledge intact.

**Version A ("the cheap one").** Three steps, all automatic except the last:
1. Per run, pick the 10 components whose removal hurts the verdict most (on val), plus the group that fires mainly on true or mainly on false statements.
2. Reverse (×−1) and double (×2) each. Measure, per family and true/false label, under both prompts, on test and OOD: how many verdicts flip, the true-false score gap, whether the answer stays a digit. The same for control directions picked the same way (data-aligned basis, untrained decomposition, random).
3. One prediction fixed in advance, the paired-twin test: for a component that handles negation, does a negated statement's score under reversal follow its affirmative twin's ("the 'not' is ignored")? Plus reading a few examples by hand, labelled exploratory.

What A claims in the report: "these components steer the verdict, in these families, more (or not more) than control directions". A table, no labels on components.

**Version B = A + automatic labelling.** In addition, a rule sorts every candidate into one of three types from its activity pattern: (i) sentence-form detector (fires on all negated statements, true or false), (ii) truth-selective within a form (fires on false negated statements only), (iii) specific to one family. Each type comes with a prediction tested on OOD, e.g. a type-(ii) negation component must also fire selectively on the German negated statements, and reversing it must flip them too.

What B claims in addition: "component X *is* a negation detector", as a tested statement per component.

**Why I recommend A.**
- B's sorting rule needs cutoffs ("fires on all negated statements" = above which CI, in what share?): the arbitrary thresholds you rejected for "active".
- Most of B's predictions are already in the plan: K1 measures selectivity per family, K2 tests whether it transfers OOD, and A's step 2 already measures the reversal effect per OOD file. B only adds the label on top.
- The labels can still be given by hand for the few components that matter, as in the 4c L17:5 analysis, and the report says that they were assigned by reading the tables.


---

26-10-27, Claude Opus 5.5 — *manipulation: A plus full exploratory tables*; same session. Julian in chat, answering the A-vs-B question under *D3 plan, revision 3*:

C: (transcribed from chat) I like a plan where you see as much of the results as possible, just in the case you see something interesting

My reading [assumed: not yet confirmed]: neither B's automatic labels nor a minimal A, but A's fixed core plus wide output that I can browse. B's labels would add no information, only a verdict on top of the tables. What adds information is computing the tables for more components and keeping all of them.

**Manipulation, as proposed now:**
- **Fixed core (= A)**: top 10 per run by removal effect + the label-selective group; ×−1 and ×2; per family and label, both prompts, test and OOD; controls chosen the same way (data-aligned basis, untrained decomposition, random active components); the paired-twin test as the one prediction fixed in advance. Reported in full, whatever comes out.
- **Wide tables, exploratory**: for *every* component of the functional active set: activity profile (mean CI per family × label × sentence form; free, it comes from the same forward pass), removal effect (T1 already covers all), and reversal effect per family × label under the training prompt on val. Saved per run as files, not only summarised. Only the top 10 get both prompts and OOD, because singles × operations × prompts × datasets × 6 runs multiplies quickly. Cost not measured; I measure it with the reproduction checks and cut the wide tables to the top m by removal effect if needed.
- **Rule that keeps browsing honest** [assumed: proposal]: I browse only the val tables. Anything interesting I see there becomes a hypothesis and is tested on test/OOD before it appears in the report as a finding; untested observations appear as "exploratory, val only". With many tables, something always looks interesting by chance, and this separation keeps that out of the findings while losing none of the browsing.

If confirmed, the plan is frozen: revisions 2 and 3 plus this entry. Next step A5.

Addendum (same entry, same session). Julian in chat, while I was writing the entry above:

C: (transcribed from chat) I'm talking about manipulating here

So the wish applies to the manipulation part only, which is how the proposal above reads it. The other categories stay as fixed in revisions 2 and 3.

---

26-10-27, Claude Opus 5.5 — *D3 frozen; A5 started*; same session. Julian in chat, answering "If that matches what you meant, the analysis plan is final":

C: (transcribed from chat) yes

**The analysis plan is frozen** [decided: transcribed above and C: comment under *D3 plan, revision 3*]: entries *D3 plan, revision 2*, *D3 plan, revision 3* and *manipulation: A plus full exploratory tables* (with its addendum), plus the in-place answers in *D3 proposal*. Fixed after the finals' val readout, before any test/OOD analysis. Next: A5, moving the analysis scripts into the repo.

---

26-10-27, Claude Opus 5.5 — *A5 part 1: shared code and the recovery script*; same session. I have read `docs/CODING.md`, `verdict_lens_analysis.py`, `ablate_4e.py`, `mask_search_4e.py` (header and PCA basis), `compare_decomps.py`, the loaders. Julian in chat during the work: "what kind of machine do you need?"

**Machine** [assumed: from earlier runs]: none for now; A5 is coding on the laptop. Renting before the scripts are done would only bill idle hours. For the analyses: inference only, so the 48 GB RTX 4090 (`spd-vast` default config) is enough. The 26-10-26 steering run used 20.5 GB, and the 4c mask search ran on that card; the training-sized RTX PRO 6000 is not needed. Outside China, so Hugging Face is reachable. GPU hours: not estimated yet, measured with the reproduction checks.

**Design** [concluded: reasons below]: the 4e scripts are not moved one-to-one. They are val-only, use the CI > 0.01 threshold and "active in ≥ 2%" candidate lists, which the frozen plan excludes, and they know only one basis. Instead:
- `paper_common.py`: statement sets (val, test, OOD = the 24 files from the manifests' `role: ood_test`, minus `EXCLUDED_OOD`; `GROUND_TRUTH_OOD` = the 13 stricter-bar files), with family, sentence form and language per statement; both verdict prompts; batching; summary metrics (pooled, per family, macro, bootstrap 95% intervals within family); and **one edit hook for every basis**: y′ = d_on (y − C) + Σ_c mask_c (v_c · x) u_c at the decomposed positions. Trained decompositions, the untrained one and the data-aligned basis (remainder W(1 − P_V)) go through the same code, so a difference between them cannot come from the intervention code.
- `paper_recovery.py` (first rental item): active set on val (rank by mean CI, smallest top-m reaching 95% of all components' KL recovered, curve by grid + bisection); the val CI values saved for the valley check plus text histograms; on test and OOD × both prompts the states full, components_ci, active_ci, mean_mask, delta_only (T0), all_off, full_without_inactive; CI sums per statement; per layer |A|, |I|, cos(A, I), |C|, |D|, ⟨C, D⟩/|y|² with A/I split by the active set. If the components together recover nothing (expected for the untrained decomposition), there is no active set and the dependent states are skipped, with a printed note.
- Verification [verified: `make check` clean; `tests/test_paper_common.py` 6 tests pass]: the edit hook is tested against the formula on a small linear layer, including untouched non-decomposed positions and hook removal. A mutation (dropping the − C term) makes that test fail. Also tested: OOD discovery (24 families, none excluded, the 13 included), the metrics on a hand-computed example, bootstrap determinism.
- Not yet run on a model (no GPU here). **Reproduction check for the rental**: the script prints val KL recovered of all components (CI-weighted, Δ off); it must match the stored values within bf16 noise [verified: stored `verdict_*_val.json` of the finals]: plain 0.9640 (`s-7152548c`) / 0.9629 (`s-46efc525`), recon 0.9769 (`s-eeac663f`) / 0.9718 (`s-edc17041`), ortho 0.9071 (`s-0c2fa0d7`) / 0.9058 (`s-4186f973`).

**Still to write**: the untrained checkpoints' config (`steps: 0`, seeds 1 and 2); the data-aligned basis builder; termination (T1, T2 curves, T3/A8 with both spans); correlation (K0–K3); manipulation (M1′ core, wide tables, paired-twin test); truth directions. Then one rental.

---

26-10-27, Claude Opus 5.5 — *A5 part 2: termination script*; same session. Julian in chat: "So what comes next?" (answer: the rest of A5, then one rental; continued with it).

- **`paper_termination.py`** [verified: `make check` clean; not run on a model]: one basis per call, `--basis decomposition` (a trained final or the untrained checkpoint) or `data_aligned` (built from the run's target model: top-96 input principal directions on 512 train statements, u = W v, as in the 4e mask search). Candidates = the run's active set from `paper_recovery.py`'s JSON, else all 672 components.
  - T1: each candidate removed alone from the full model; on val for all candidates (ranking, also reused by manipulation), on test for active-set members.
  - T2: top-k by val separation drop removed together, k ∈ {1, 2, 5, 10, 20, 50, 100, 200} up to all candidates, plus `--extra_k` so controls can be compared at the trained runs' active-set sizes; test and OOD, both prompts; 5 random size-k subsets of the active set as the size-matched control.
  - T3/A8 (`--a8`, needs an active set): per layer, projection of the original output onto span(all 96 writes) and span(the layer's active writes) removed; controls of the same dimension: 3 random subspaces and the top principal directions of y.
- **Shared code extended**: `original_pass`, `run_edit` (any edit as a context manager), `projected_out` (A8), `data_aligned_basis`, `cap_per_dataset`; verdict flips relative to the model's own verdict (1→0, 0→1) in every summary. `paper_recovery.py` now uses the shared runner. Tests: 8 pass (projection hook, flips, the cap).
- **Cost point for the rental** [verified: statement counts]: val 986, test 1,794, OOD 8,713 (the conj files are large). Every OOD condition costs about 5× a test condition, so T2 and A8 on full OOD under both prompts may be long. `--ood_cap N` (seeded random N per dataset) exists; I'd decide N from the step times measured at the start of the rental, and record it [assumed: proposal].

Still to write: untrained checkpoint config; correlation (K0–K3); manipulation (M1′ core, wide tables, paired-twin test); truth directions.

---

26-10-27, Claude Opus 5.5 — *A5 part 3: all analysis scripts written; run order for the rental*; same session. Julian in chat: "keep going". Earlier in this stretch Julian asked me to commit his staged files and the newest changes (commits `814d273`, `927ffd2`).

**Written** [verified: `make check` clean, full suite 547 passed / 15 skipped, `tests/test_paper_common.py` 11 tests; none of the scripts has run on the model yet]:
- `config_paper_untrained.yaml`: `config_paper_plain.yaml` with only header, label, `seed: 1` and `steps: 0` changed (run a second time with `seed: 2`). The three paper configs differ only in their loss term, so one untrained decomposition per seed serves all methods [concluded: config diff]. `run_spd.py` returns before the training loop at `steps: 0` after saving `model_0.pth`; the PGD setup before that point has not been exercised with the paper config [assumed: works, check first in the rental].
- `paper_correlation.py` (K0–K3): per statement and component the coefficient (CI · v·x; v·x for the data-aligned basis, comparable across bases) and the CI, each at the last statement token and as the mean over statement tokens, plus the residual after layer 17. K0 probes over all features (as arm A: standardised, default L2, train → test/OOD; label and verdict). K1 single-component AUC per family (saved in full), readout chosen on train, top-k probes (k = 1, 2, 5, 10). K2 val selection (|AUC − 0.5| ≥ 0.2 within a form) and transfer to the OOD families of the matching form (same sign, ≥ 0.1): label on the 13 stricter files, verdict on all 24. K3 oriented AUC on the statements where verdict and label disagree. K2/K3 logic checked on synthetic data (a planted label-follower gets oriented AUC 0.98, a verdict-follower 0.0).
- `paper_manipulation.py`: fixed core = top 10 by val single-removal drop (from the termination JSON), each and as a group, plus the K2-selective group, ×−1 and ×2, test and OOD, both prompts; 10 random active components for a decomposition with an active set; the paired-twin test on test (all 489 negated test statements have their affirmative twin there [verified: CPU]; tested). Exploratory wide table: every active component reversed alone on val.
- `paper_directions.py`: mass-mean and logistic-regression directions after each layer 11–17 at the last statement token (fit on train, L2 strength chosen on val from A4's grid); per component cos(u, d) vs random vectors; shares of the truth gap of A (CI-weighted writes), I (rest of the components), D (remainder), and per component; Spearman of T1 drop vs gap share; `--remove_directions` removes one direction from the original MLP output per layer and at all layers.
- Shared: `make_basis` (the data-aligned basis always from 512 statements of the full train split, so quick checks and all scripts use the same basis per seed), `maybe_cap` / `--n_per_dataset` in every script for quick checks, `auc_columns` (tested against sklearn).

**Run order for the rental** [assumed: proposal]. The data-aligned basis depends only on the base model and the train data, so one call per script suffices (run with any final's config).
0. Smoke test of every script on one final and the data-aligned basis with `--n_per_dataset 4 --ood_cap 4`; measure the time per condition; then decide `--ood_cap` for the full runs (OOD has 8,713 statements).
1. Untrained checkpoints, seeds 1 and 2.
2. `paper_recovery.py` on the 6 finals and 2 untrained. Reproduction check against the stored val numbers (entry *A5 part 1*); the valley histograms come out here.
3. `paper_termination.py`: finals with `--recovery … --a8`; untrained with `--recovery` (no active set expected: all components are candidates); data-aligned once with `--extra_k` = the finals' active-set sizes.
4. `paper_correlation.py`: 6 + 2 + 1 calls.
5. `paper_manipulation.py`: same 9 (needs 3 and 4).
6. `paper_directions.py`: same 9 with `--termination`; `--remove_directions` once.
