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
Answer (26-10-22, Claude Opus 5.5): Agreed; added to the plan (entry *plan of steps*, step B3). Design [assumed: proposal]: steer with the in-context truth direction at the statement tokens, layers 13–17 (the window where steering worked on the verdict prompt), and read four outcomes: the verdict digit, the digit under a second verdict prompt, the direct recall question ("In which country is the city of X?", as in the steering topic's recall test) and the free continuation. A hint that steering may reach deeper than the component masks: in the statement window, steering made the model's explanation follow the steered digit, while after a mask-inverted digit the model mostly restates the true fact [verified in the bolder-experiments and component-steering topics; a few example statements each, anecdotal]. On where the model gets the information back after a mask edit at the statement tokens of layers 11–17, two candidate routes: (i) attention at the answer tokens re-reads the statement in layers outside 11–17; (ii) layers 18+ at the statement tokens recompute it. Test: widen the edit (more layers, then the answer positions too) and see where the true fact stops coming back.
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
