# The Quantization Model of Neural Scaling

**Eric J. Michaud, Ziming Liu, Uzay Girit, Max Tegmark (MIT & IAIFI)**

*Correspondence: ericjm@mit.edu*

*NeurIPS 2023. arXiv:2303.13506. Project code: https://github.com/ejmichaud/quantization-model*

---

## Abstract

We propose the *Quantization Model* of neural scaling laws, explaining both the observed power law dropoff of loss with model and data size, and also the sudden emergence of new capabilities with scale. We derive this model from what we call the *Quantization Hypothesis*, where network knowledge and skills are "quantized" into discrete chunks (*quanta*). We show that when quanta are learned in order of decreasing use frequency, then a power law in use frequencies explains observed power law scaling of loss. We validate this prediction on toy datasets, then study how scaling curves decompose for large language models. Using language model gradients, we automatically decompose model behavior into a diverse set of skills (quanta). We tentatively find that the frequency at which these quanta are used in the training distribution roughly follows a power law corresponding with the empirical scaling exponent for language models, a prediction of our theory.

---

## Introduction

In the aggregate, larger neural networks trained on more data perform better than smaller neural networks trained on less data, in a predictable way. Across a range of studies, mean test loss has been observed to decrease as a power law in both the number of network parameters ($L \propto N^{-\alpha_N}$) and the number of training samples ($L \propto D^{-\alpha_D}$) [Hestness et al., 2017; Rosenfeld et al., 2019; Kaplan et al., 2020; Henighan et al., 2020; Gordon et al., 2021; Zhai et al., 2022; Hoffmann et al., 2022]. Although aggregate performance changes smoothly with scale, when particular capabilities are examined, larger models often have emergent abilities, i.e., qualitatively different performance than smaller models [Wei et al., 2022; Steinhardt, 2022]. Understanding and reconciling both facets of scaling—the predictable power law decrease in loss and the emergence of new capabilities at scale—is of both theoretical and practical interest [Ganguli et al., 2022]. Understanding how scaling changes what neural networks learn is entangled with core questions: what are deep neural networks doing internally, and will they continue to improve with scale?

Recent studies of the internals of neural networks have found a variety of impressive algorithms learned by gradient descent [Olah et al., 2020; Cammarata et al., 2020; Nanda et al., 2023; Li et al., 2022; Wang et al., 2022]. As more work is put into understanding the structures learned by neural networks (the task of *mechanistic interpretability*), we may find more and more *circuits* [Olah et al., 2020; Elhage et al., 2021] in models, intelligible internal algorithms for accomplishing prediction in specific contexts. Can such analysis be scaled up to frontier models [Lieberum et al., 2023]? Two assumptions which, if true, would make us more optimistic about mechanistically understanding large models include (1) decomposability/modularity/sparsity [Casper et al., 2022; Bricken et al., 2023; Frankle & Carbin, 2018; Bayazit et al., 2023]—that large models are decomposable into parts, and only a small number of these parts are relevant to the model's behavior on any given sample and (2) universality [Li et al., 2016; Olah et al., 2020; Nguyen et al., 2021; Dravid et al., 2023]—that similar structures recur across models of increasing size. Recently, Olsson et al. [2022] found encouraging evidence for universality of "induction heads" across LLMs and found that these emerge in a discrete transition during training.

In this paper, we articulate the *Quantization Hypothesis*, a set of informal conjectures about the *decomposability* of networks into smaller parts, the *universality* of computations performed across model scales, the *discreteness* of what models learn, and about how properties of the data distribution produce power law neural scaling. In particular, we hypothesize that to many prediction problems, there corresponds a particular enumerable set of indivisible pieces of knowledge or skills that models must learn, and that model performance is determined by *which* of these elements models successfully learn. We call these basic building blocks of model performance the **quanta**:

> **Quantum (plural quanta)**: An indivisible computational module that, for example, retrieves a fact, implements an algorithm, or more generally corresponds to some basic skill possessed by a model.

**Figure 1: LLM skill "quanta" auto-discovered in text.** We auto-discover *quanta*—basic units of model knowledge/skill—for a language model. Here we show collections of next-token prediction samples which our method clustered together, each corresponding to some coherent model behavior. We indicate the token which was predicted from the context before it with **[bold brackets]**. We indicate newlines using "`\n`". See [The quanta of language modeling](#the-quanta-of-language-modeling) for explanation.

*Cluster 50: incrementing numerical sequences*

> 01- Mi Querencia (Simón Díaz)\n02- Tonada De Luna Llena (Simón Díaz)\n03- Sabana (José Salazar/Simón Díaz)\n04- Caballo Viejo (Simón Díaz)\n05- Todo Este Campo Es Mío (Simón Díaz)\n06- La Pena Del Becerrero (Simón Díaz)\n**[07]**

> \* \* from opening a through road or street for public use across said public park in the Park of The City of Riverton \* \* \*." (Emphasis supplied.) Appealing from that order, the city asserts (1) plaintiffs have no standing or right to maintain the action; (2) that the proposed road was in an undedicated part of the park; (3) that the proposed road was an access road and not a through street or part of the city's street system; (**[4]**

> 4. \_Introduction\_\n  5. Chapter 1: What Is Trust?\n  6. Chapter 2: Trust Brings Rest\n  7. Chapter 3: Who Can I Trust?\n  8. Chapter 4: The Folly of Self-Reliance\n  9. Chapter 5: Trust God and Do Good (Part 1)\n  10. Chapter 6: Trust God and Do Good (Part 2)\n  11. Chapter 7: At All Times\n  12. Chapter**[ 8]**

> was achieved. The chosen sites were recorded as: 0 = sound (\*n\* = 13); 1 = first visible sign of noncavitated lesion seen only when the tooth is dried; 2 = visible noncavitated lesion seen when wet and dry; 3 = microcavitation in enamel; 4 = noncavitated lesion extending into dentine seen as an undermining shadow; 5 = small cavitated lesion with visible dentine: less than 50% of surface; **[6]**

> QCBlockListMsg          = 0x0a\n    GetLatestStatusMsg      = 0x0b\n    LatestStatusMsg         = 0x0c\n    PrepareBlockHashMsg     = 0x0d\n    GetViewChangeMsg        = 0x0e\n    PingMsg                 = 0x0**[f]**

*Cluster 100: predicting newlines in line length limited text*

> TO PERFORM QUADRATIC REGRESSION\nON THE TI84 GRAPHING CALCULATOR,\nDETERMINE HOW WELL THE \nREGRESSION MODEL FITS THE DATA,\nAND THEN MAKE PREDICTIONS \nUSING THE REGRESSION EQUATION.\nIN STATISTICS, \nREGRESSION ANALYSIS INCLUDES\nANY TECHNIQUES USED FOR MODELING **[\n]**

> \# creddump is free software: you can redistribute it and/or modify\n# it under the terms of the GNU General Public License as published by\n# the Free Software Foundation, either version 3 of the License, or\n# (at your option) any later version.\n#\n# creddump is distributed in the hope that it will be useful,**[\n]**

> &lt;!--\n/\*\*\n \* Copyright (c) 2019, The Android Open Source Project\n \*\n \* Licensed under the Apache License, Version 2.0 (the "License");\n \* you may not use this file except in compliance with the License.**[\n]**

> \*\n        Pursuant to 5TH CIR. R. 47.5, the court has determined\nthat this opinion should not be published and is not precedent\nexcept under the limited circumstances set forth in 5TH CIR.**[\n]**

> children have a lack of maturity and an underdeveloped\nsense of responsibility, leading to recklessness, impul-\nsivity, and heedless risk-taking.... Second, children\nare more vulnerable... to negative influences and\noutside pressures, including from their family and\npeers; they have limited contro[l] over their own envi-**[\n]**

We use this terminology in analogy to Max Planck's assumption in 1900 that energy is quantized into discrete chunks (quanta)—here we imagine that knowledge/skills are quantized into discrete chunks (quanta). Since "quantization" is commonly used in machine learning in the context of low-precision arithmetic, we suggest "knowledge quantization" or "skill quantization" to refer to our notion of quantization. We will see that a Zipfian distribution governing the "use frequency" of the quanta produces power law neural scaling, where the effect of scaling is to learn an increasing number of discrete quanta, and smooth scaling laws average over small discrete jumps in model performance.

This paper is organized as follows: in [Theory](#theory) we give a theoretical model of power law neural scaling from the Quantization Hypothesis. In [Proof of concept](#proof-of-concept-a-toy-dataset) we construct toy datasets satisfying the hypothesis, where smooth power laws average over many discrete jumps in model performance. In [Decomposing LLM scaling laws](#decomposing-llm-scaling-laws) we then analyze how power law scaling decomposes for real LLMs. In [The quanta of language modeling](#the-quanta-of-language-modeling), we develop a method for automatically discovering quanta in language models by clustering their behavior into basic coherent skills, and analyze the statistics of these clusters, concluding in [Discussion](#discussion).

## Theory

Consider the task of modeling the distribution of text on the internet. Successful prediction requires an immense amount of knowledge, and the ability to perform diverse computations, due to the immense complexity and diversity of the world and therefore of human language. For instance, in order to predict what word will come next in a conversation between two physicists, one must "know" much about physics. In order to continue the text "`2534 + 7261 = `", one must be able to perform arithmetic (for large enough numbers, memorization becomes a highly inefficient strategy) [Branwen, 2021]. A great many distinct types of computations are present in the world in the processes that *produce* text, and so *predicting* text requires those computations to be present in our models.

In this paper, we conjecture the Quantization (or Quanta) Hypothesis:

> **QH1** Many natural prediction problems decompose into an enumerable set of computations, pieces of knowledge, or skills, which models must learn to reduce loss. We call these **quanta**, and model them as being *discrete*—they are either learned or not learned. Model performance is determined by *which* quanta have been learned.
>
> **QH2** Some quanta are more useful for reducing loss than others, leading to a natural ordering of the quanta. We call the ordered quanta the **Q Sequence**. Optimally trained networks should therefore learn the quanta in that order. The effect of scaling is to learn *more* of the quanta in the Q Sequence, so scaling performance is simply determined by *how many* quanta are successfully learned.
>
> **QH3** The frequencies at which the quanta are used for prediction follow a power law.

Together these can result in power law neural scaling. We model the Quantization Hypothesis as follows, referring to the below as the "Quantization (or Quanta) Model". Let $\mathbf{q}$ denote a bit string whose $k^{\rm th}$ bit $\mathbf{q}_k=1$ if the $k^{\rm th}$ quantum in the Q Sequence has been learned, and $\mathbf{q}_k=0$ otherwise. QH1 implies that the mean loss $L$ is simply a function of $\mathbf{q}$. QH2 implies that when $n\equiv\sum_k \mathbf{q}_k$ quanta have been learned, we have $\mathbf{q}_k=1$ for $k\le n$. Let $L_n$ denote the mean loss in this case. From QH3, we have that the $k^{\rm th}$ quantum benefits prediction on a randomly chosen sample with probability

$$p_k = \frac{1}{\zeta(\alpha+1)} k^{-(\alpha+1)}\propto k^{-(\alpha+1)}$$

for a Zipf power law $\alpha>0$, where $\zeta(s)\equiv\sum_{k=1}^\infty k^{-s}$. Let us also assume that learning the $k^{\rm th}$ quantum reduces average loss from $b_k$ before it is learned to $a_k$ after it is learned on the samples where it is utilized. If $a_k$ and $b_k$ are $k$-independent ($a_k=a$, $b_k=b$), then a model that has learned the first $n$ quanta will have expected loss

$$L_n = \sum_{k=1}^n a p_k + \sum_{k=n+1}^\infty b p_k = \sum_{k=1}^\infty a p_k + \sum_{k=n+1}^\infty (b-a) p_k$$

$$\approx a + \frac{b-a}{\zeta(\alpha+1)}\int_n^\infty k^{-(\alpha+1)} dk = a + \frac{b-a}{\alpha\zeta(\alpha+1)}n^{-\alpha}.$$

In other words, $L_\infty=a$ and $(L_n-L_\infty)\propto n^{-\alpha}$ is a power law.

In [Appendix: More general scaling laws](#more-general-scaling-laws), we provide analogous derivations for other assumptions for $a_k$ and $b_k$, and find that a range of assumptions produce curves that are exact or approximate power laws—the latter include a small logarithmic correction.

In the derivation above, we assumed that all samples are what we will refer to as *monogenic*, meaning that prediction relies on at most a single quantum, akin to how monogenic traits in biology (e.g. cystic fibrosis) depend on a single gene. By assuming that all samples are monogenic, we can write the expected loss as a sum over quanta, weighted by the fraction of samples which rely on that quanta $p_k$. We further explore the idea of monogenic vs. polygenic samples in [Monogenic versus polygenic scaling curves](#monogenic-versus-polygenic-scaling-curves). So far we have seen how the Quantization Hypothesis can produce power law scaling as a function of the number of quanta learned $n$. We will now give one possible mechanism by which this can translate into power law scaling in parameters, data, etc.:

**Parameter scaling**: In networks of finite size, network capacity can bottleneck how many quanta are learned. If we assume that all quanta require the same capacity of $C$ network parameters, then a network with $N$ parameters can learn roughly $n \approx N / C$ quanta. Therefore $L(N) - L_\infty \propto n^{-\alpha} \approx (N/C)^{-\alpha} \propto N^{-\alpha}$, so we get power law scaling in $N$ with exponent $\alpha_N = \alpha$.

**Data scaling (multi-epoch)**: For data scaling, we assume that for each quantum, a threshold of $\tau$ examples utilizing quantum $k$ are needed in the training set for quantum $k$ to be learned (this type of threshold has precedent, e.g. for the algorithmic tasks where "grokking" occurs [Power et al., 2022]). With $D$ training samples, approximately $Dp_k$ samples relying on quantum $k$ will be present, and solving for $Dp_n = \tau$ we get the last quantum to be learned will be $n \propto (D / \tau)^{1/(\alpha+1)}$ since $p_k \propto k^{-(\alpha+1)}$. Under this model, we get scaling in data samples $L(D) - L_\infty \propto n^{-\alpha} \propto \left( D / \tau \right)^{-\alpha/(\alpha+1)} \propto D^{-\alpha/(\alpha+1)}$, and so $\alpha_D = \alpha / (\alpha + 1)$. From our earlier result that $\alpha_N = \alpha$, we would therefore predict that $\alpha_D = \alpha_N / (\alpha_N + 1)$. We discuss whether this relationship holds empirically for data and parameter scaling exponents observed across a variety of studies in [Appendix: Parameter and data scaling exponents across studies](#parameter-and-data-scaling-exponents-across-studies).

**Data scaling (single-epoch)**: In multi-epoch training, the information contained in the training dataset can bottleneck which quanta are learned. However, the rate of convergence of SGD can also bottleneck performance. For single-epoch training, a greater number of training samples allows one to train for longer. In our model, the amount that each quantum reduces mean loss by follows a power law. If the magnitude of the gradients for learning these quanta also follow a power law, then the convergence time for each quanta may follow a power law too. If the number of steps to learn quantum $k$ is $\propto 1/p_k$, then if the first quantum requires $T$ steps to be learned, quantum $n$ will require $T n^{\alpha+1}$ steps, and so $n = (S/T)^{1/(\alpha+1)}$ quanta can be learned in $S$ steps. This gives scaling in training steps $L(S) - L_\infty \propto n^{-\alpha} \approx (S/T)^{-\alpha/(\alpha+1)} \propto S^{-\alpha/(\alpha+1)}$, and so $\alpha_S = \alpha / (\alpha + 1)$. Under this model, multi-epoch and single-epoch data scaling exponents coincide: $\alpha_D = \alpha_S$.

## Proof of concept: a toy dataset

In this section, we will describe a toy dataset consisting of distinct subtasks which are power law distributed in frequency. We observe power law neural scaling in data and parameters on this task, and find that the mechanism of neural scaling coincides with our theory from [Theory](#theory). It is therefore possible for scaling laws to arise from the Quantization Model for data with the right structure. We leave a study of whether natural datasets (e.g. natural modeling) possess such structure to [Decomposing LLM scaling laws](#decomposing-llm-scaling-laws).

### The "multitask sparse parity" dataset

The toy task we will construct consists of many subtasks—distinct types of inputs which each require corresponding distinct computations (quanta). For each subtask, we choose a variant of the "sparse parity" problem, recently studied in [Barak et al., 2022]. The sparse parity prediction problem is simple: given a bit string of length $n$, compute the parity (sum mod 2) of a fixed subset of $k$ of those bits. We introduce an extension of this task, which we call "multitask sparse parity". Beyond $n$ and $k$, multitask sparse parity adds an additional parameter $n_\text{tasks}$, the number of subtasks (number of distinct versions of sparse parity) present in the dataset. To construct the task, we first choose $n_\text{tasks}$ random subsets $S_i$ of $k$ indices from $\{1, 2, \ldots, n\}$: $S_i \subset \{1, 2, \ldots, n\}$ and $|S_i| = k$, where $i = 1, 2, \ldots, n_\text{tasks}$. Input bit strings are length $n_\text{tasks} + n$. We call the first $n_\text{tasks}$ bits the *control bits* and the last $n$ bits the *task bits*. If control bit $i$ is active, then the parity is computed from the subset $S_i$ of the task bits. The control bits 1-hot encode the task number: on a given input, only one control bit is set to $1$ at a time—the rest are zero. For the sample shown below, since control bit $2$ is active, the answer is the parity of the task bits $S_2 = \{2, 7\}$, which is $0$ for this input:

![Multitask sparse parity diagram](figures/sparse-parity-diagram.png)

We impose a uniform distribution over the task bits. On the control bits, we impose a Zipfian distribution: the probability that a sample has control bit $i$ active (and therefore the parity must be computed from the subset $S_i$ of the task bits) is $\frac{1}{Z} i^{-(\alpha+1)}$ where $Z = \sum_{i=1}^{n_\text{tasks}} i^{-(\alpha+1)}$. This imposes a power law distribution over subtasks in data. Since answers are parities, this task can be treated as a binary classification problem on the subset of bit strings $\{0, 1\}^{n_\text{tasks} + n}$ where for each string all but one bit of the first $n_\text{tasks}$ bits are zero.

### Power law scaling and emergence

![Scaling in parameters, steps and data on multitask sparse parity](figures/parameters-steps-data-emergence-and-scaling-scalingtop.png)

**Figure 2. Top:** Neural networks exhibit power law scaling in loss w.r.t. parameters $N$, training time $S$, and training samples $D$ (for multi-epoch training) when trained on the multitask sparse parity dataset. Here $\alpha = 0.4$ and we plot lines $\propto N^{-\alpha}$, $\propto S^{-\alpha/(\alpha+1)}$, $\propto D^{-\alpha / (\alpha+1)}$. **Bottom:** neural scaling broken down by subtask. Scaling behavior on individual subtasks exhibits emergence, where subtasks are suddenly learned above a particular scale. Power law neural scaling of mean test loss averages over a large number of qualitative changes in network performance (when broken down by subtask), with loss being driven to zero on an increasing number of subtasks which are power law distributed in frequency, a realization of the mechanism of neural scaling discussed in [Theory](#theory).

We train ReLU MLPs with a single hidden layer to solve this task with cross-entropy loss. The input dimension is $n_\text{tasks} + n$. We use the Adam optimizer with a learning rate of $10^{-3}$. To study scaling with respect to the number of model parameters, we train networks of varying width by sampling batches online. Within an individual single-epoch training run, we can study scaling in steps $S$. To study scaling with respect to multi-epoch training dataset size $D$, we use a network of sufficient width for capacity to not be a bottleneck, and for varying $D$ we sample a training set of $D$ samples and train for multiple epochs, recording model performance when mean test loss is lowest (early-stopping).

Training dynamics on the multitask sparse parity problem are highly nontrivial—on each individual subtask, loss follows a reverse-S curve, dropping after an initial plateau. This transition happens at different times for different subtasks, so the overall loss decreases smoothly, averaging over these transitions. See [Appendix: Additional results on multitask sparse parity](#additional-results-on-multitask-sparse-parity) for more discussion of training dynamics.

Figure 2 shows scaling curves on the multitask sparse parity problem. For the results shown, we used $n_\text{tasks} = 500$, $n = 100$, $k = 3$, $\alpha=0.4$, and a batch size of $20000$. We vary training dataset size from 1e4 to 5e6 and vary hidden-layer width from 10 to 500 neurons. We train for 2e5 steps. In line with the theory from [Theory](#theory), we find that as we scale training data and parameters, networks learn more and more quanta (reducing loss on more and more subtasks), roughly in order of their frequency, and that this is what drives neural scaling. We see that scaling w.r.t. parameters is noisier than data scaling, possibly due to model initialization having some influence on which quanta are learned (for our data scaling experiments, we use the same seed and same model size for all runs). We also see that for scaling on individual subtasks, there is a rough scale of data or parameters below which networks do not learn the task, and above which they do. Smooth power law scaling therefore averages over a large number of emergent changes in model performance when properly decomposed by subtask, a proof of concept that the Quantization Model can be the mechanism of neural scaling for data with the right structure. See [Appendix: Additional results on multitask sparse parity](#additional-results-on-multitask-sparse-parity) for additional results and discussion on how the scaling exponents $\alpha_N, \alpha_S, \alpha_D$ relate to the subtask distribution power law exponent $\alpha + 1$ empirically.

## Decomposing LLM scaling laws

![Pythia scaling and the distribution over per-token losses](figures/pythia-scaling-tripanel.png)

**Figure 3. Left:** Scaling of mean test loss w.r.t. non-embedding parameters for the Pythia models [Biderman et al., 2023]. The parameter scaling exponent $\alpha_N$ is measured to be $\approx 0.083$ from the first six points along the curve (the seventh model appears to break the trend). **Center:** the distribution $p(L)$ over losses on individual samples for models of different size. Losses $\approx 0$ are by far the most common, and larger models achieve $\approx 0$ loss on an increasing fraction of samples. **Right:** the expected loss integrand $L p(L)$ for models of different sizes. Low-loss samples contribute minimal mass to the mean loss, which is instead dominated by samples with much higher loss of 5-10 bits (depending on scale).

We now study how scaling curves for large language models decompose. For our experiments, we use the Pythia model suite from Eleuther AI [Biderman et al., 2023], a set of decoder-only transformers of varying size trained on approximately 300 billion tokens of The Pile [Gao et al., 2020]. We evaluate several models in the suite (ranging from 19m to 6.4b non-embedding parameters) on approximately 10 million tokens from the test set of The Pile. We record cross-entropy loss on every token, enabling us to study how loss on individual tokens, as well as how the distribution over losses, changes with model scale.

### The distribution over per-token losses

In Figure 3, we show how the distribution over losses scales with model size. First, we find that for the first six models in the Pythia sequence, the mean loss scales as a power law with exponent $\alpha_N = 0.083$, roughly in line with the parameter scaling exponent of $0.076$ measured in [Kaplan et al., 2020]. The 6.4b model does not fit the scaling curve well, so we excluded its loss when measuring the scaling exponent. Next, we plot the probability distribution over per-token losses $p(L)$. We find that losses close to zero are by far the most common, and that scaling increases the portion of approximately-zero losses. We also plot $L p(L)$, the probability density over losses weighted by loss. The mean loss is the area under this curve. We see that despite approximately-zero-loss tokens being by far the most common, they do not contribute much mass to the mean loss. See Figure A5 for how these distributions change over training steps rather than across model size. We note that neural scaling in the wild is much more complicated than for multitask sparse parity—notably, the distribution over losses is not bimodal. We leave a detailed study of whether the statistics of neural scaling in LLMs are compatible with prior models of neural scaling to future work.

**Figure 4.** Per-sample scaling curves can have diverse behavior. Here we show extreme examples where scaling (of loss on predicting the highlighted token) is abrupt versus smooth. If the Quantization Hypothesis describes language modeling, then samples with sharp scaling would be *monogenic*, displaying sharp emergence at a particular model scale when the relevant quantum is learned. Samples with gradual scaling would be *polygenic*, where many quanta, emerging at different scales, marginally improve the loss. We show additional examples in Figure A6.

*Monogenic sample*

![Scaling curve for a monogenic sample](figures/tokensinghsirsa.png)

> …  accused Jagdish Tytler at a Congress event where Sheila Dikshit took charge as party's Delhi chief.Shiromani Akali Dal MLA Manjinder Singh**[ Sir]**sa alleged that the Congress

*Polygenic sample*

![Scaling curve for a polygenic sample](figures/tokenfruit-influx.png)

> … The big disappointment this summer was that despite my 2 plum trees fruiting super-abundantly, beyond expectations, the fruit was mostly spoiled by an**[ inf]**estation of worms and several

### Monogenic versus polygenic scaling curves

In our introduction of the Quantization Hypothesis in [Theory](#theory) and our multitask sparse parity study in [Proof of concept](#proof-of-concept-a-toy-dataset) we modeled network performance on individual samples as benefitting from a single quantum—all samples belong to a single subtask, which is either solved or not solved in a binary fashion. In our model and on multitask sparse parity, scaling curves on individual examples all exhibit emergence—loss on individual examples undergoes a sharp transition at a particular scale of parameters or data. Do we observe this in large language models?

Inspecting a large number of per-token (per-sample) scaling curves, we observe a variety of scaling behaviors. On some samples, loss drops at a particular scale. More typically though, loss improves at multiple scales. If the Quantization Hypothesis is true and the effect of scaling is to simply add new quanta to the model, then for per-sample loss curves to show progress at multiple scales, those samples must benefit from multiple quanta additively. As first mentioned in [Theory](#theory), we borrow terminology from genetics and refer to prediction problems for which the model's performance is determined by a single quantum as *monogenic* (akin to when a single gene determines a trait) and as *polygenic* when multiple quanta influence performance (in analogy to when multiple genes contribute to a trait). In multitask sparse parity, all prediction problems are monogenic. In natural language, we observe that model performance on most tokens improves at multiple scales, suggesting that most tokens are polygenic, but we can find tokens for which loss drops as a single phase transition in scale. Polygenicity forms a spectrum: the smoothness of the loss curve can vary substantially between examples, presumably with some prediction problems using few quanta and others using many. In Figure 4, we show extreme examples of both monogenic and polygenic samples.

Note that our monogenic/polygenic taxonomy of model behaviors assumes that QH1 and QH2 are true. However, it could be the case that there isn't an underlying discreteness to what is learned, or that scaling totally changes what networks learn, rather than simply adding additional quanta. Whether scaling truly has the effect we described will have to be investigated in future studies of the internals of neural networks. We also note that it is possible that sharp transitions in the per-token loss curves could be due to noise—if we had multiple runs with different random seeds for each model scale, we could better test whether the mean loss across seeds decreases smoothly or if there is a genuine discreteness where gradual progress is impossible for apparently "monogenic" tokens.

## The quanta of language modeling

We have conjectured that the internals and behavior of language models are decomposable into an enumerable set of modules and associated skills (quanta). What might these basic building blocks of LLMs be? In this section, we develop a preliminary method to discover quanta. In particular, we will attempt to cluster tokens in a language corpus according to what knowledge or skill LLMs use to predict those tokens from their context. Our goal is to find coherent clusters of language model behavior that each reveal some distinct skill that the model has learned. Note that in *clustering* tokens to discover quanta, we are making the likely unrealistic assumption that these tokens are monogenic—that there is only one quantum involved in predicting each token. Note also that these clusters of behavior will not give us a mechanistic understanding of the quanta, but simply provide examples of LLM skills which could be studied further in future work.

We propose the use of gradients to cluster next-token prediction samples, where a "sample" consists of a token and its context in some document. Given some model, we will cluster two samples together if the gradient of the model's loss on each sample w.r.t. the model's parameters is similar for the two samples. The intuition for using gradients is as follows: if a model uses the same internal module to generate its prediction on two samples, then the gradients for parameters within the module may be nonzero and similar for the two samples (and possibly $\approx 0$ for parameters in irrelevant modules). If a model uses different modules to generate its prediction on different samples, then the gradients may not overlap. We therefore use gradient similarity as a proxy for *mechanistic similarity*—whether a model uses similar mechanisms/modules to generate its prediction on distinct samples. While crude, we find that gradients contain enough information to allow us to automatically discover many coherent clusters of LLM behavior using the following algorithm:

**Quanta Discovery from Gradients (QDG)**: We will use spectral clustering on gradients to find clusters of samples whose gradient has nonzero cosine similarity. Given a set of samples $(x_i, y_i)$ and a model $f_\theta$, we compute gradients for each sample $g_i = \nabla_\theta L(f_\theta(x_i), y_i)$. We then normalize these gradients $g_i \mapsto \hat{g}_i$ so that $\hat{g}_i \cdot \hat{g}_i = 1$. Let $A$ be a matrix whose rows are the normalized gradients: $A_{i, \cdot} = \hat{g}_i$. If we are clustering $d$ samples and our model has $n$ parameters, $A$ has shape $(d, n)$. We compute an affinity matrix $C = A A^T$, a matrix of shape $(d, d)$ where $C_{ij} = \hat{g}_i \cdot \hat{g}_j$, the cosine similarity between gradients $g_i, g_j$. From this, we compute an affinity matrix of the angular similarities $\hat{C}$ (which take values in $[0, 1]$) via $\hat{C}_{ij} = 1 - \arccos(C_{ij})/\pi$. We then perform spectral clustering with $\hat{C}$ to cluster samples.

![Gradient similarity matrix and rank-frequency plot of QDG clusters](figures/similarity-matrix-and-rank-frequency-envelope.png)

**Figure 5. Left:** angular similarity between model gradients for a variety of natural language samples. Samples are reordered according to their QDG cluster (with 400 clusters) to reveal the block-diagonal structure of the similarity matrix. We visualize a small part of the overall similarity matrix in this plot—note that not all clusters are as visibly distinct as the ones shown. **Right:** rank-frequency plot of QDG clusters. We measure the slope of the envelope of the rank-frequency curves from cluster rank 100-1000 to be $\approx -1.24$, which is steeper than the slope of -1.08 expected from the measured parameter-scaling exponent from Figure 3, though within the margin of error given the uncertainty of our clustering methodology. See [Appendix: The difficulty of estimating the power law exponent from clusters](#the-difficulty-of-estimating-the-power-law-exponent-from-clusters) for a discussion of the bias/uncertainty of our method.

QDG is expensive to compute for large models and for large numbers of samples. We therefore only apply it to the smallest model in the Pythia suite, which has 19m non-embedding parameters. We cluster 10000 tokens on which this model is confident and correct in its prediction, achieving less than $0.1$ nats of cross-entropy. See [Appendix: Details of application of QDG to LLMs](#details-of-application-of-qdg-to-llms) for more detail.

We find that many, though not all, QDG clusters reveal some coherent model behavior. We show examples from clusters in Figure 1 and Figure A7. These clusters were found with the spectral clustering hyperparameter `n_clusters = 400`. While most clusters involve the prediction of the same token, manually inspecting these clusters we find that they usually involve predicting the same token for a coherent reason, rather than being based merely on having the same output. We also find clusters for more abstract prediction rules. For instance, the quantum shown on the left column of Figure 1 is the skill of incrementing a numerical sequence, and the examples involve predicting a variety of different tokens representing numbers.

### The natural distribution over language modeling quanta

In our model, some quanta are more frequently used than others. If these frequencies follow a power law in accordance with the Quantization Hypothesis, then we may expect QDG cluster sizes to be governed by a power law. The measured scaling exponent of $\alpha_N = 0.083$ from Figure 3 implies a power law distribution over quanta with exponent $-1.083$. Do the cluster sizes follow this?

Figure 5 shows rank-frequency curves for clusters discovered with QDG for varying choices of `n_clusters`. These curves sort the clusters according to their size and then plot size against cluster index (rank). We plot rank-frequency curves for many choices of `n_clusters` since it is unclear a priori which `n_clusters` to use. When we measure the slope of the rank-frequency curve, we measure it from the envelope formed by the many rank-frequency curves, a practice which we discuss in [Appendix: The difficulty of estimating the power law exponent from clusters](#the-difficulty-of-estimating-the-power-law-exponent-from-clusters). Biases in the clustering algorithm and inherent noise in model gradients make clustering imperfect, and lead to high uncertainty of the measured power law exponent. From our analysis in that appendix, we think that extracting the power law exponent over quanta utilization frequency by measuring the slope of the rank-frequency curve should have uncertainty of at least 0.2. We also note that some rank-frequency curves don't look like a clean power law. In Figure A10 we find that we can get similar-looking curves in a toy model of this clustering process when the dimension and noise is high. Between ranks 100-1000, we measure a slope of $\approx -1.24$, about $0.16$ off our expected slope of $-1.08$, and so within the margin of error. We are encouraged that the size of our discovered clusters seem to decay at a rate (very roughly) compatible with observed neural scaling exponents, in line with our theory. However, less naive clustering schemes, operating on more samples with more clusters, could be useful to sharpen this measurement.

## Related Work

**Models of neural scaling**: Several models of neural scaling laws have been proposed in prior work. Sharma & Kaplan [2022] explain power law scaling w.r.t. model parameters using an argument from approximation theory, which relates neural scaling exponents to the dimension of the data manifold $d$. Michaud et al. [2023] point out that effective dimension $d$ could be generalized to the maximum arity of the target function's computation graph for sparse compositional problems. Bahri et al. [2021] generalized the model of Sharma & Kaplan [2022] to scaling w.r.t. dataset size, additionally relating scaling exponents to the power law spectrum of certain kernels. Maloney et al. [2022] develop an exactly solvable random-feature model of scaling, from which they derive a joint parameter-data scaling law. Bordelon et al. [2020] develop a model of data scaling for kernels, decomposing the generalization error into a sum over eigenmodes, whereas we decompose error into a sum over quanta. Arguably the closest prior work to ours is Hutter [2021], who develops a model of data scaling wherein a discrete set of "features" must be learned. In this model, a feature is learned if it occurs at least once in the training set. If the features are Zipfian distributed, this produces power law scaling in expectation but with high variance. In our model, using a data threshold $\tau \gg 1$ lowers the variance in the scaling curve, and we also considered scaling w.r.t. parameters and applied the model to real networks.

**Understanding emergent abilities**: Wei et al. [2022] and Srivastava et al. [2022] document examples of emergent abilities in large language models, though Schaeffer et al. [2023] suggest that these examples are an artifact of the metric used to evaluate performance. Arora & Goyal [2023] develop a framework for the emergence of "skills", where predicting text requires combining multiple different skills from an underlying set of language skills.

**Miscellaneous**: The topic of phase transitions in machine learning is not new [Saitta et al., 2011], but our work was strongly influenced by the recent work of Olsson et al. [2022] who observe a phase change from the formation of induction heads and especially Nanda et al. [2023] who conjecture that phase changes may be ubiquitous. Simon et al. [2023] also exhibit a task where learning proceeds as a series of discrete steps. Chen et al. [2023] develop a framework for understanding LLM "skills" in a hierarchy and for choosing data to more efficiently learn desired skills. Chan et al. [2022] study how a Zipfian data distribution influences in-context learning.

## Discussion

**Summary**: The Quantization Hypothesis posits that for some types of prediction problems, models must learn a discrete (quantized) set of modules/knowledge/skills (quanta). When data is distributed in such a way that the "use frequencies" of these quanta follow a power law, then power law neural scaling can arise as models learn more and more quanta, with smooth scaling curves averaging over many small cases of emergence. We presented a toy dataset where neural scaling exhibits these properties. We then documented how language model scaling curves decompose, beyond simply how the mean loss scales. Lastly, we developed a method to discover quanta from the internal structure of trained models, from which we were able to enumerate a large number of skills of a small language model. The frequencies at which the quanta we discover are used for prediction in natural text seem to roughly track the power law our theory would predict, though this measurement is quite imprecise.

**Limitations**: While the Quantization Hypothesis appears to hold for our toy datasets, much work remains in investigating to what extent it holds for natural tasks like language modeling. Probably our riskiest assumption was that there is an underlying discreteness to *everything* that models learn. Gradual scaling seems typical in LLMs [Schaeffer et al., 2023], and it could be more parsimonious to model neural scaling as an underlying smooth process rather than to assume that most tasks are highly polygenic with underlying discrete quanta. Note also that in our model of scaling w.r.t. parameters $N$, having more parameters merely increases the capacity of the network. In practice however, larger networks are more efficient learners [Hoffmann et al., 2022], and one can trade off between parameters and data, whereas in our model parameters and data independently bottleneck the number of quanta that can be learned. Additionally, we modeled the quanta as being independent, where learning order is given just by the use frequencies, but it could make more sense to think of the quanta as living in a hierarchical dependency graph. Lastly, our QDG method is neither very principled nor scalable, and much better methods could likely be developed to discover quanta and study their statistics for larger models and across more samples.

**Implications for emergence and forecasting**: Srivastava et al. [2022] find that on some tasks, neural scaling has high "linearity", with gradual improvements to scale, with other tasks displaying "breakthroughness", where performance improves sharply at some scale. In our model, high linearity would result from a task's relevant quanta being widely spread along the Q Sequence, and high breakthroughness would result from a task being monogenic or from the relevant quanta being close together in the Q Sequence. Our model also suggests that future capabilities could be forecasted if one could estimate the frequency at which that skill would benefit prediction in the training corpus.

**Implications for mechanistic interpretability**: If the Quantization Hypothesis is correct, then understanding a network reduces to enumerating its quanta. Having done this, the quanta could perhaps then be translated into a more interpretable format (something like code), studied in this format, and eventually executed in this format, rather than via the operation of the network.

**Outlook**: Lastly, our decomposition of networks into quanta is reminiscent of Minsky's *Society of Mind* [Minsky, 1988] perspective that minds are decomposable into individually mindless "agents". If this decomposition is indeed possible, then the quanta (agents) become natural objects of study within networks. This *mesoscale* understanding of networks, in terms of the internal modules which collectively constitute their performance, could perhaps act like statistical physics for deep learning, allowing us to bridge our microscale understanding of low-level training dynamics and our macroscale understanding of model performance.

## Acknowledgments

We thank Tamay Besiroglu, Neel Nanda, Tony Wang, David Bau, Ben Edelman, Brian Cheung, Wes Gurnee, Stephen Casper, Peter Hase, Davis Brown, Eleni Shor, Max Nadeau, and Xander Davies for helpful conversations and feedback. We thank Lauro Langosco for helping with code to visualize samples from The Pile. This work was supported by the Foundational Questions Institute, the Rothberg Family Fund for Cognitive Science, the NSF Graduate Research Fellowship (Grant No. 2141064), and IAIFI through NSF grant PHY-2019786.

---

## Appendix

### More general scaling laws

If one learns the first $n$ quanta, reducing the loss from $b_k$ to $a_k$ ($1\leq k\leq n$), while the loss remains $b_k$ for $k>n$, the expected loss is given by:

$$L_n = \sum_{k=1}^n a_kp_k + \sum_{k=n+1}^\infty b_kp_k.$$

In the main text, we used $a_k=a$ and $b_k=b$ for our model. However, one can imagine a variety of other choices for $a_k$ and $b_k$.

**Case 1** $b_k=-{\rm log}\ p_k$ and $a_k=0$, where $p_k=k^{-(\alpha+1)}/\zeta(\alpha+1)$. This baseline for $b_k$ is the error of a model which outputs the token frequencies, independent of the context (assuming that quanta involve the prediction of a particular token). The expected loss is given by:

$$L_n = \sum_{k=1}^n 0 \cdot p_k + \sum_{k=n+1}^\infty (-{\rm log\ }p_k)\cdot p_k \approx \frac{1+\alpha+\alpha{\rm log}\zeta(\alpha+1)}{\alpha^2\zeta(\alpha+1)}n^{-\alpha} + \frac{\alpha+1}{\alpha\zeta(\alpha+1)}n^{-\alpha}{\rm log\ }n,$$

which contains a power law term $n^{-\alpha}$ plus a log term $n^{-\alpha}{\rm log\ }n$. For very large $n$, the log term can be ignored, so $L$ is still approximately a power law of $n$ with exponent $-\alpha$, shown in Figure A1.

![Comparing scaling laws with different assumptions](figures/log_power_law.png)

**Figure A1.** Comparing different scaling laws. Setting $a_k=0$, we compare $b_k=-{\rm log}\ p_k$ (solid lines) and $b_k=1$ (dashed lines) for different alphas. Although the $b_k=-{\rm log}\ p_k$ case would cause an extra loss term $n^{-\alpha}{\rm log}n$ in addition to the power law term $n^{-\alpha}$, the loss becomes a power law asymptotically when $n$ becomes large.

**Case 2** $b_k=-{\rm log}\ p_k$ and $a_k=-{\rm log}\ (Cp_k)\ (C>1)$, where $p_k=k^{-(\alpha+1)}/\zeta(\alpha+1)$. The expected loss is given by:

$$L_n = \sum_{k=1}^n (-{\rm log\ }(Cp_k)) \cdot p_k + \sum_{k=n+1}^\infty (-{\rm log\ }p_k)\cdot p_k \approx \frac{{\rm log}C}{\alpha\zeta(\alpha+1)}n^{-\alpha}-{\rm log}C+\frac{1+\alpha+\alpha{\rm log}\zeta(\alpha+1)}{\alpha^2\zeta(\alpha+1)},$$

which is a power law $n^{-\alpha}$ plus a constant.

### Additional results on multitask sparse parity

**Training dynamics**: When loss is broken down by subtask on multitask sparse parity, learning curves consist of many reverse-S shaped curves, and mean loss decreases smoothly as an average over these curves. In Figure A2, we show loss versus time for each subtask for training runs in both the single-epoch and multi-epoch regimes. In Figure A3 we show how convergence time for each subtask relates to the frequency of that subtask.

![Per-subtask training dynamics](figures/parity-subtask-timeseries-infinite-and-finite-data.jpg)

**Figure A2.** Training dynamics on the multitask sparse parity dataset consist of many "phase transitions" when decomposed by subtask—the loss curve for each subtask drops following an initial plateau of no apparent progress, in line with [Barak et al., 2022]. The mean loss decreases smoothly, averaging over these phase transitions in the model's performance on subtasks. We show curves for single-epoch training (top) and multi-epoch training on 5 million samples (bottom). The dashed red line indicates the early stopping point where mean test loss is minimized. For these runs, $\alpha = 0.4$.

![Convergence time versus subtask frequency](figures/sparse-parity-convergence-time.png)

**Figure A3.** Convergence time for each subtask versus the frequency of that subtask. We see that convergence time $S_k$ on subtask $k$ is $S_k \propto p_k^{-0.81}$ rather than $S_k \propto p_k^{-1}$ as we had expected. This leads to a steeper scaling w.r.t. $S$ than expected from theory. For these experiments, we used $\alpha = 0.4$, and so we would have predicted $\alpha_S \approx 0.29$ but instead we get $\alpha_S \approx 0.45$. We consider the model to have converged on a subtask once it gets mean test loss less than 0.1 bits on that subtask.

**Scaling for varying $\alpha$**: In Figure A4b we show scaling curves on multitask sparse parity in $N, S, D$ for a variety of quanta distribution parameters $\alpha$. While all scaling curves appear to be power laws, the relationship between $\alpha_N, \alpha_S, \alpha_D$ and $\alpha$ is not precisely as predicted by theory:

1. **Parameter scaling:** We observe that the relationship between $\alpha_N$ and $\alpha$ deviates a bit from the prediction $\alpha_N = \alpha$, with $\alpha_N < \alpha$ for small $\alpha$ and $\alpha_N > \alpha$ for large $\alpha$. Perhaps model size does not influence learning just by changing capacity, but also by affecting optimization.
2. **Step scaling:** We observe that $\alpha_S$ is consistently higher than the theoretical prediction $\alpha / (\alpha + 1)$. In Figure A3, we saw that the number of steps to convergence for each subtask did not precisely follow $S_k \propto p_k^{-1}$, but was closer to $S_k \propto p_k^{-0.81}$. This means that many subtasks converge faster than we would expect, producing a steeper scaling curve.
3. **Data scaling:** We observe that $\alpha_D$ is substantially higher than the theoretical prediction $\alpha / (\alpha + 1)$ for small $\alpha$. We think this may be related to the fact that early-stopping cuts off training before all subtasks are learned as observed in Figure A2. In Figure A4a, we show how the number of subtasks learned $n$, when we include subtasks learned after early-stopping, seems to be in line with theory: $n \propto D^{1/(\alpha+1)}$.

Better understanding the precise nature of power law scaling on multitask sparse parity is an interesting avenue for future work.

![Number of subtasks learned versus training samples](figures/sparse-parity-data-scaling-dependence-n.png)

**Figure A4a.** Number of subtasks learned ($n$), including subtasks learned after early-stopping would have terminated the training run, versus training samples $D$ for a variety of $\alpha$. We see that the relation $n \propto D^{1/(\alpha+1)}$ approximately holds, in line with theory. Deviation from theory for the scaling exponent of loss $L$ w.r.t. $D$ therefore likely originates from our failure to regularize network training, leading to early-stopping ending training before some subtasks can be learned.

![Scaling in N, S and D for varying alpha](figures/sparse-parity-all-scaling-varying-alpha.png)

**Figure A4b.** Scaling in parameters ($N$), single-epoch training time ($S$), and multi-epoch training samples ($D$) for varying quanta power law distribution parameter $\alpha$ on multitask sparse parity. We notice that scaling curves in steps $S$ are typically steeper than the $\alpha_S = \alpha / (\alpha + 1)$ predicted from theory, and that for low $\alpha$ the scaling curves in $D$ also deviate from theory substantially.

### Additional results on language models

In Figure A5 we show how the distribution over losses changes across time during a training run, rather than across model scales like in Figure 3.

![Pythia training dynamics and loss distributions over time](figures/pythia-dynamics-tripanel.png)

**Figure A5. Left:** Training curves (scaling w.r.t. steps $S$) of mean test loss for Pythia models. We measure exponents $\alpha_S$ between 0.037 and 0.06. **Center:** the distribution $p(L)$ over time. Over time, models achieve $\approx 0$ loss on an increasing fraction of tokens, similar to scaling in model size. **Right:** The distribution $L \cdot p(L)$ over time.

**Figure A6.** Additional LLM scaling curves on individual samples which exhibit sharp vs smooth improvement. If the Quantization Hypothesis is true for language modeling, then we would interpret samples with sharp drops as "monogenic" and samples with gradual progress as "polygenic".

*Monogenic samples*

![Monogenic sample scaling curve](figures/tokenneilmackinnon.png)

> … "The law of unintended consequences and the history of previous military interventions in the region is not a recipe for political and economic stability," said Neil MacKinnon, global macro strategist at**[ V]**TB Capital.\n\n

![Monogenic sample scaling curve](figures/tokenessmarshall.png)

> … Opinion filed March 25, 1988.\nW.Y. Chalfant, of Branine, Chalfant & Hill, of Hutchinson, argued the cause and was on the brief for appellant, Hesston State Bank.\nKenneth C. Jones, of Watson, Ess,**[ Marshall]** & Enggas, of

*Polygenic samples*

![Polygenic sample scaling curve](figures/tokenssep-normal.png)

> …  In general, the lesions of thoraco-cervical level were difficult to detect, because the appearance rate of SSEP peaks are reduced over the thoraco-cervical spine even in**[ normal]** controls. In cases with

![Polygenic sample scaling curve](figures/tokenonconsumer.png)

> …  airline by revenue, dropped $2.15, or 7.2 percent, to $27.71 and Delta Air Lines lost $1.16, or 5.7 percent, to $19.11.\n\nStone said oil prices could start weighing on**[ consumer]** spending down the road,

In Figure A7 we show additional examples from clusters discovered with QDG.

**Figure A7.** Additional examples of clusters of inputs discovered by QDG. Like in Figure 1, we used 10000 samples and `n_clusters` of 400.

*Cluster 146: comma after day of month*

> Sam Willard\n\nSamuel Steven Willard (born September 9**[,]**

> \n215 U.S. 437 (1910)\nMECHANICAL APPLIANCE COMPANY\nv.\nCASTLEMAN.\nNo. 48.\nSupreme Court of United States.\nArgued December 3, 1909.\nDecided January 3**[,]**

> Frederick W. Keator\n\nFrederick W. Keator (October 22, 1855 – January 31**[,]**

> United States Patent No. 6,073,124 (issued June 6, 2000) ("the '124 patent"). Microsoft in turn asserted counterclaims against NCI for infringement of three of its patentsUnited States Patent Nos. 5,822,526, 5,999,914 and 5,794,006. Only terms of the '124 patent are presently before the Court; interpretation of claims in Microsoft's patents will be interpreted in a separate Markman hearing to be held on November 15**[,]**

*Cluster 269: "s" after starting year of decade*

> Romford Ice Arena\n\nRomford Ice Arena was an ice rink located in Romford in the London Borough of Havering, England. The venue was built in the 1980**[s]**

> Although the novel continues to be the dominant medium of the crime-mystery-detective narrative, short stories by these contemporary authors may be found in numerous anthologies of the genre published during the 1990**[s]**

> Armed with the new Shenwei SW26010 chips, a new supercomputer has surged to the top of the TOP500 list: Sunway TaihuLight. It is nearly three times as fast as Tianhe-2, being benchmarked in Linpack as being able to perform 93 quadrillion calculations each second (93 petaFLOPS). To put this achievement in context, modern desktop PCs are already more powerful than the top ranked supercomputers from the early 1990**[s]**

> Early activism \n\nHe began a lifetime involvement with revolutionary politics in the late 1930**[s]**

> Yesterday, visitor Greg Davidson commented that he was searching for songs played on the local forecast back in the late ’80**[s]**

> In 1954, the couple published Living the Good Life which inspired many young, educated Americans to create simpler, rural lifestyles and the back-to-the-land movement of the 1960**[s]**

*Cluster 278: colon after CSS property*

> .rickshaw\_graph.detail {\n    pointer-events: none;\n    position: absolute;\n    top: 0;\n    z-index: 2;\n    background: rgba(0, 0, 0, 0.1);\n    bottom: 0;\n    width**[:]**

> @import '../../../assets/sass/spin';\n\n.app-header {\n  background-color: #282c34;\n  min-height: 100vh;\n  display**[:]**

> See: http://jsfiddle.net/mWFGZ/1/\nhtml, body {\n    margin: 0;\n    padding**[:]**

*Cluster 292: protocol separator in URLs*

> \# 				                                         #\n#                                                                        #\n# For questions please refer to:                                         #\n# https**[://]**

> When I run the below code, I am getting an error: MongoError: The dollar ($) prefixed field '$push' in '$push' is not valid for storage.\nI put this together based on the docs: https**[://]**

> Gruber, Martin A. Views of the National Zoological Park in Washington, DC, showing Exhibit. 1919. Retrieved from the Digital Public Library of America, http**[://]**

> But as citizens, our responsibility is to look beyond the anecdote. We journalists try to make sure that if a tree falls in the forest, it won't go unnoticed. Still, if we get so taken by the trees that we don't see the forest, we'll all be lost.\n\nRex Smith is editor of the Times Union. Share your thoughts at http**[://]**

#### Details of application of QDG to LLMs

When applying QDG to language models, we use gradients within self-attention and MLP layers, but do not include embed, unembed, or layer norm gradients when we flatten and concatenate gradients into a vector $g$. (We exclude gradients for embed and unembed parameters because they are high dimensional and also because they may contain information more about the input and output rather than the computations the model performs internally. We exclude layer norm gradients because they appeared to contain less information about clusters in toy experiments.) We choose samples $(x_i, y_i)$ for which our 19m-parameter model achieves a cross-entropy loss less than $0.1$ nats. We filter based on this criteria since (1) we cannot cluster samples based on model mechanism if the model does not have such a mechanism for performing prediction correctly on those samples and (2) our intuition that samples with particularly low loss are more likely to be monogenic. We further exclude samples which can be solved via induction on the context (we filter copying induction problems by excluding samples where the token which is to be predicted is the last token in a trigram which occurred earlier in the context; this is not a very comprehensive filtering scheme), since such samples are quite common (possibly interfering with our task of finding diverse quanta) and since early experiments indicated that QDG had trouble clustering such samples together. We choose 10000 such samples to perform clustering on from the test set of The Pile. After computing the affinity matrix $\hat{C}$, we use the spectral clustering implementation from scikit-learn [Pedregosa et al., 2011] with labels assigned via k-means.

### Quanta discovery on TinyStories

We also apply QDG to TinyStories-33M, a language model trained on the TinyStories dataset [Eldan & Li, 2023]. We consider only tokens on which TinyStories-33M achieves a loss less than 1 bit of cross-entropy. We apply QDG to 10000 such samples, clustering their gradients with spectral clustering with `n_clusters = 400`. We show some samples from the resulting clusters in Figure A8. Many of these clusters reflect some simple recurring pattern in the TinyStories dataset, like predicting " time" after "Once upon a", which many documents in the dataset start with. Some other clusters are more interesting however, like Cluster 11, which seems to involve predicting the correct noun in a sentence where that noun was referred to earlier in the sentence or in previous sentences.

**Figure A8: Example "quanta" for the TinyStories dataset.** Examples of clusters within the TinyStories dataset, discovered by QDG on the TinyStories-33M model. Here we just show samples from four out of 400 `n_clusters`.

*Cluster 11: predicting the correct noun*

> Lily asked her mom, "Can I touch the sunflower?" Her mom replied, "No, Lily. The sunflower is not for touching. It's for looking at." \n\nLily was sad, but she understood. The next day, Lily rode her bike past the**[ sun]**

> Once upon a time, there was a big house with a door. The door was brown and it could move when people opened it. One day, a little boy came to the house and he saw the impressive door. He wanted to open it and see what was inside. So he moved the**[ door]**

> When they got to the park, Tim saw a man with a sack. The man had found Tim's wagon and put it in the sack to keep it safe. Tim was happy to have his wagon back and thanked the man. He put his**[ wagon]**

> Lily and Ben went to the park with Mom. They saw a big pond with many ducks and swans. Lily liked the swans. They were white and graceful. She wanted to feed them some bread.\n\n"Mom, can I give some bread to the**[ sw]**

> Lily and Ben nod. They promise to be careful. They ask mom to read the letter to them. Mom smiles. She reads the letter. It is from grandma. She says she loves them a lot. She sends them kisses and hugs. Lily and Ben are happy. They send kisses and hugs back to grandma. They thank mom for the**[ letter]**

*Cluster 31: " time" after "Once upon a"*

> Once upon a**[ time]** (8 examples, all identical in the discovered cluster)

*Cluster 75: comma after temporal phrase*

> One day, Jack wanted to show off his cool sunglasses. He spotted a nice patch of grass in the park, and he decided to sit down and enjoy the sun. He put his sunglasses on and just sat. He felt so special.\n\nA few moments later**[,]**

> Jack happily agreed and they started the game. At first, it was a bit tricky for both of them to get the squash to the other, but after a few tries, Jack was a pro. He laughed and cheered as he ran back and forth to get the squash. \n\nMommy and Jack played until the sun started to go down. Then**[,]**

> Bob loves to go on adventures. He went for a walk along the beach, looking for fun and exciting things. All of a sudden**[,]**

> Linda was a little girl who loved wandering around in nature. She was eager to explore and find what she could. One day, Linda was wandering around in the woods when she spotted a big yellow flower. She was so excited that she ran over to it. She scooped the flower up and inspected it more closely. Soon**[,]**

> Next, her mom told Emma to wipe the floor clean. Emma grabbed a cloth and wiped the floor. When she was finished, it was as clean as a new penny.\n\nFinally**[,]**

*Cluster 77: beginning of quote*

> Jack showed her the big blue ticket and said, "This is my ticket. I'm going to lay it down."\nThe girl asked, "Where will you lay it down?"\nJack answered,**[ "]**

> When she found her mom she said, "Mom, I have news!" Her mom said, "What is it, Jane?" Jane said, "I want to stir something up and make it more fun!" \n\nHer mom said,**[ "]**

> The twin jumped back in surprise. She had never heard a flower talk before! She asked the flower, "Who are you?"\n"My name is Pinky," said the flower.\n\nThe twin was now even more surprised. She asked,**[ "]**

> He asked Jill, "What's in this jar?" Jill smiled and said, "It's sugar! Would you like some?" \n\nJack shook his head and said, "No, thank you. I don't think I should eat sugar. My mom won't allow it." \n\nJill nodded and said,**[ "]**

> Nearby, her mom was watching and called out, "Lucy, come here! What's that you have there?"\n\nLucy proudly held up the hoop and announced,**[ "]**

### The difficulty of estimating the power law exponent from clusters

In [The natural distribution over language modeling quanta](#the-natural-distribution-over-language-modeling-quanta), when we looked at the distribution over elements in each cluster, we did not perfectly recover a Zipf distribution with exponent $\approx 1.08$ that we expected from our theory. In this section, we describe the difficulty of accurately estimating such an exponent with our method.

#### QDG on multitask sparse parity

![Similarity matrix and rank-frequency plots for multitask sparse parity](figures/similarity-matrix-and-rank-frequency-envelope-sparseparity.png)

**Figure A9.** Similarity matrix and rank-frequency plots from QDG on multitask sparse parity. Despite sparse parity having a known decomposition into subtasks which are power law distributed in frequency, we do not recover this same power law from samples. We used $\alpha = 0.4$ for the frequency distribution for an expected rank-frequency power law exponent of -1.4, but measure a rank-frequency envelope slope closer to -1.1.

As a first experiment, we performed QDG on multitask sparse parity, where there is a known, artificially-imposed power law distribution over subtasks. We train a width-500 single-hidden-layer ReLU MLP on multitask sparse parity with $\alpha = 0.4$ and with $n=100$, $k=3$, and $n_\text{tasks} = 500$. We then took 10000 samples which the network achieves $\approx 0$ loss on (sampled from the Zipf distribution over subtasks with exponent 1.4). We compute gradients of cross-entropy loss w.r.t. all model parameters for these samples, and then perform QDG just like for LLMs. We show results in Figure A9. We plot the full similarity matrix where samples are ordered according to their a priori known subtask, rather than their cluster from QDG, and see a clear pattern where elements from the same subtask have on average higher angular similarity than elements between subtasks. However, from the rank-frequency plot of the clusters, we do not recover a slope of -1.4, but rather a lower slope of $\approx -1.1$. This shows that even when there is an exact decomposition of inputs into subtasks with a known Zipf distribution over these subtasks, that we do not perfectly recover this Zipf distribution from QDG.

#### A toy model of QDG uncertainty and bias

**A toy model:** To understand the bias of spectral clustering, we develop the following toy model. We assume the dataset has $N=1000$ subtasks, each subtask containing $n_i=\lfloor\frac{A}{i^\alpha}\rfloor (1\leq i\leq N)$ tokens ($A=1000$). We use a Gaussian distribution $\mathcal{N}(\mathbf{m}_i,\sigma^2\mathbf{I}_{d\times d})$ to model gradients within a subtask $i$, where $d$ is the embedding dimension, $\sigma$ is the noise level, and $\mathbf{m}_i$ is the Gaussian mean. $\mathbf{m}_i$ itself is drawn from the standard Gaussian distribution $\mathbf{m}_i\sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d\times d})$. We define the similarity between two vectors $\mathbf{x}, \mathbf{y}$ to be ${\rm sim}\equiv 1+\frac{\mathbf{x}}{|\mathbf{x}|}\cdot \frac{\mathbf{y}}{|\mathbf{y}|}$. We compute pairwise similarity between all $\sum_{i=1}^N n_i$ tokens, and input the similarity matrix to the spectral clustering algorithm. We also need to specify the number of clusters $k$.

We have two hyperparameters in the toy model, the embedding dimension $d$ and the noise level $\sigma$. We need to determine them such that this toy model can decently imitate LLM results (Figure 5). We fix $\alpha=1$, sweeping $d=\{30,100,1000\}$, $\sigma=\{0,0.5,2.0\}$, and $k=\{100,200,500\}$. As shown in Figure A10, the high-dimension ($d=1000$) large-noise ($\sigma=2.0$) scheme seem to best agree with the LLM results, since the $k=200$ curve can reproduce the sag and the cliff present in LLM curves.

Estimating $\alpha$ from the frequency curve is hard, in fact, the slope depends on $k$ and the region used to estimate it. However, we observe that different $k$ curves form a clear envelope, whose slope is robust in a reasonably wide region. The envelope slope seems to indicate $\alpha$. We fix $d=1000$ and $\sigma=2.0$, sweeping $\alpha=\{0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5\}$. For each $\alpha$, we estimate the slope of the envelope. Although there is clear correlation between the estimated envelope and $\alpha$, if we use the envelope slope to estimate $\alpha$, the error is on the order of 0.2, as shown in Figure A11.

![Spectral clustering on a toy model with varying dimension and noise](figures/toy_clustering_d_noise.png)

**Figure A10.** To understand the bias of spectral clustering, we apply spectral clustering to a toy model with different embedding dimension $d$, noise scale $\sigma$ and number of clusters $k$. The high-dimension ($d=1000$) large-noise ($\sigma=2.0$) scheme seems to best agree with the LLM results (Figure 5).

![Envelope slope versus alpha](figures/toy_clustering_envelope_notparity.png)

![Comparison of estimated and true alpha](figures/toy_clustering_compare.png)

**Figure A11.** The difficulty of measuring $\alpha$ from curves. We apply spectral clustering to a toy model with different $\alpha$ and number of clusters $k$. For a fixed $\alpha$, different $k$ curves define an envelope. One could use the envelope slope to infer $\alpha$, but this incurs errors around 0.2.

### Parameter and data scaling exponents across studies

In Figure A12, we show $\alpha_N$ and $\alpha_D$ (or possibly $\alpha_S$, depending on the study) for a variety of prior studies of deep learning scaling, as compiled by Villalobos [2023]. While the data is messy, it is intriguing that most of the Rosenfeld et al. [2019] samples lie below the $\alpha_D = \alpha_N$ line, as our model would predict. The scaling exponents from Hoffmann et al. [2022] are slightly closer to our prediction than the relation $\alpha_D = \alpha_N$, which has been proposed by other models of neural scaling laws. Overall though, the existing empirical results are too messy to definitively support or contradict our model.

![Parameter versus data scaling exponents across studies](figures/scaling-scatter-linear-scale.png)

**Figure A12.** Parameter and data scaling exponents from various studies of deep learning scaling, compiled from the database of neural scaling laws from [Villalobos, 2023]. Our model of scaling predicts that $\alpha_D = \alpha_N / (\alpha_N + 1)$, indicated with the solid black line. Visible points are from [Rosenfeld et al., 2019; Kaplan et al., 2020; Hoffmann et al., 2022; Gordon et al., 2021; Droppo & Elibol, 2021]. [Ardalani et al., 2022] is above the visible window of the figure.

### Estimates of compute used for our experiments

**Multitask sparse parity**: Our training script takes roughly 1-4 hours (depending on network size) to perform a single-epoch training run on a GPU. When training multi-epoch on a fixed dataset, runs take typically between 3-10 minutes, with some outliers taking much longer. Our largest experiment was for Figure A4b, where we trained networks of varying width on data with varying distributions over subtasks (with different power law exponents). 467 runs completed with a total running time of approximately 1450 hours. These experiments were run on a cluster with heterogeneous hardware. Available GPUs include NVIDIA A100, RTXA6000, QUADRORTX6000, GEFORCERTX2080TI, GEFORCERTX2080, GEFORCEGTX1080TI, titan-x, and tesla-v100.

**Pythia model scaling evaluations**: We evaluated Pythia models on NVIDIA A100 80GBs. We do not have available the running time used when computing loss on approximately ten million tokens (for which we reported scaling statistics on), although it was likely less than an hour per model. The most expensive experiments were for Figure A5, where we evaluated the first four models in the Pythia suite across 143 checkpoints for a total of 572 evaluations. We likely used some hundreds of A100-hours for this, though possibly less than 100 hours.

**QDG**: We ran our QDG experiments on an NVIDIA A100 80GB. For the smallest Pythia model and for 10000 samples, it takes a few hours to compute the similarity matrix. We performed this computation only a handful of times.

---

## References

Ardalani, Wu, Chen, et al. (2022). Understanding Scaling Laws for Recommendation Models. *arXiv:2208.08489*. https://arxiv.org/abs/2208.08489

Arora, Goyal (2023). A theory for emergence of complex skills in language models. *arXiv:2307.15936*. https://arxiv.org/abs/2307.15936

Bahri, Dyer, Kaplan, et al. (2021). Explaining neural scaling laws. *arXiv:2102.06701*. https://arxiv.org/abs/2102.06701

Barak, Edelman, Goel, et al. (2022). Hidden progress in deep learning: Sgd learns parities near the computational limit. *arXiv:2207.08799*. https://arxiv.org/abs/2207.08799

Bayazit, Foroutan, Chen, et al. (2023). Discovering Knowledge-Critical Subnetworks in Pretrained Language Models. *arXiv:2310.03084*. https://arxiv.org/abs/2310.03084

Biderman, Schoelkopf, Anthony, et al. (2023). Pythia: A suite for analyzing large language models across training and scaling. *arXiv:2304.01373*. https://arxiv.org/abs/2304.01373

Bordelon, Canatar, Pehlevan (2020). Spectrum dependent learning curves in kernel regression and wide neural networks. *International Conference on Machine Learning*.

Branwen (2021). The scaling hypothesis. https://gwern.net/scaling-hypothesis

Bricken, Templeton, Batson, et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Transformer Circuits Thread*.

Cammarata, Goh, Carter, et al. (2020). Curve detectors. *Distill*.

Casper, Hod, Filan, et al. (2022). Graphical clusterability and local specialization in deep neural networks. *ICLR 2022 Workshop on PAIR $\$$\backslash$textasciicircum$\$ 2Struct: Privacy, Accountability, Interpretability, Robustness, Reasoning on Structured Data*.

Chan, Santoro, Lampinen, et al. (2022). Data distributional properties drive emergent in-context learning in transformers. *Advances in Neural Information Processing Systems*.

Chen, Roberts, Bhatia, et al. (2023). Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models. *arXiv:2307.14430*. https://arxiv.org/abs/2307.14430

Dravid, Gandelsman, Efros, et al. (2023). Rosetta Neurons: Mining the Common Units in a Model Zoo. http://arxiv.org/abs/2306.09346

Droppo, Elibol (2021). Scaling laws for acoustic models. *arXiv:2106.09488*. https://arxiv.org/abs/2106.09488

Eldan, Li (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English?. *arXiv:2305.07759*. https://arxiv.org/abs/2305.07759

Elhage, Nanda, Olsson, et al. (2021). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*.

Frankle, Carbin (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. *arXiv:1803.03635*. https://arxiv.org/abs/1803.03635

Ganguli, Hernandez, Lovitt, et al. (2022). Predictability and surprise in large generative models. *2022 ACM Conference on Fairness, Accountability, and Transparency*.

Gao, Biderman, Black, et al. (2020). The pile: An 800gb dataset of diverse text for language modeling. *arXiv:2101.00027*. https://arxiv.org/abs/2101.00027

Gordon, Duh, Kaplan (2021). Data and parameter scaling laws for neural machine translation. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*.

Henighan, Kaplan, Katz, et al. (2020). Scaling laws for autoregressive generative modeling. *arXiv:2010.14701*. https://arxiv.org/abs/2010.14701

Hestness, Narang, Ardalani, et al. (2017). Deep learning scaling is predictable, empirically. *arXiv:1712.00409*. https://arxiv.org/abs/1712.00409

Hoffmann, Borgeaud, Mensch, et al. (2022). Training compute-optimal large language models. *arXiv:2203.15556*. https://arxiv.org/abs/2203.15556

Hutter (2021). Learning curve theory. *arXiv:2102.04074*. https://arxiv.org/abs/2102.04074

Kaplan, McCandlish, Henighan, et al. (2020). Scaling laws for neural language models. *arXiv:2001.08361*. https://arxiv.org/abs/2001.08361

Li, Yosinski, Clune, et al. (2016). Convergent Learning: Do different neural networks learn the same representations?. http://arxiv.org/abs/1511.07543

Li, Hopkins, Bau, et al. (2022). Emergent world representations: Exploring a sequence model trained on a synthetic task. *arXiv:2210.13382*. https://arxiv.org/abs/2210.13382

Lieberum, Rahtz, Kramár, et al. (2023). Does circuit analysis interpretability scale? evidence from multiple choice capabilities in chinchilla. *arXiv:2307.09458*. https://arxiv.org/abs/2307.09458

Maloney, Roberts, Sully (2022). A Solvable Model of Neural Scaling Laws. *arXiv:2210.16859*. https://arxiv.org/abs/2210.16859

Michaud, Liu, Tegmark (2023). Precision Machine Learning. *Entropy*. https://www.mdpi.com/1099-4300/25/1/175

Minsky (1988). Society of mind.

Nanda, Chan, Liberum, et al. (2023). Progress measures for grokking via mechanistic interpretability. *arXiv:2301.05217*. https://arxiv.org/abs/2301.05217

Nguyen, Raghu, Kornblith (2021). Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth. http://arxiv.org/abs/2010.15327

Olah, Cammarata, Schubert, et al. (2020). Zoom In: An Introduction to Circuits. *Distill*.

Olsson, Elhage, Nanda, et al. (2022). In-context Learning and Induction Heads. *Transformer Circuits Thread*.

Pedregosa, Varoquaux, Gramfort, et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*.

Power, Burda, Edwards, et al. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv:2201.02177*. https://arxiv.org/abs/2201.02177

Rosenfeld, Rosenfeld, Belinkov, et al. (2019). A constructive prediction of the generalization error across scales. *arXiv:1909.12673*. https://arxiv.org/abs/1909.12673

Saitta, Giordana, Cornuejols (2011). Phase transitions in machine learning.

Schaeffer, Miranda, Koyejo (2023). Are Emergent Abilities of Large Language Models a Mirage?. *arXiv:2304.15004*. https://arxiv.org/abs/2304.15004

Sharma, Kaplan (2022). Scaling Laws from the Data Manifold Dimension. *Journal of Machine Learning Research*. http://jmlr.org/papers/v23/20-1111.html

Simon, Knutins, Ziyin, et al. (2023). On the stepwise nature of self-supervised learning. *arXiv:2303.15438*. https://arxiv.org/abs/2303.15438

Srivastava, Rastogi, Rao, et al. (2022). Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. *arXiv:2206.04615*. https://arxiv.org/abs/2206.04615

Steinhardt (2022). Future ML Systems Will Be Qualitatively Different. https://bounded-regret.ghost.io/future-ml-systems-will-be-qualitatively-different/

Villalobos (2023). Scaling Laws Literature Review.

Wang, Variengien, Conmy, et al. (2022). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small. *arXiv:2211.00593*. https://arxiv.org/abs/2211.00593

Wei, Tay, Bommasani, et al. (2022). Emergent Abilities of Large Language Models. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=yzkSU5zdwD

Zhai, Kolesnikov, Houlsby, et al. (2022). Scaling vision transformers. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.

