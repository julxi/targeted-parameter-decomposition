#import "template.typ": *
#import "26-10-14_ablation_examples_tables.typ": refs, base, comp, so_counts, so_diff

#show: report.with(
  title: "Removing and flipping Exp 3's components: example answers",
  subtitle: "Greedy 20-token answers of Qwen2.5-7B-Instruct to six true/false statements, with components of a targeted parameter decomposition removed or sign-flipped at the prompt",
  date: "2026-10-14",
  author: "Written by Claude Opus 5.5 for Julian Bitterlich",
  summary: [
    - *Setup.* Six statements, each followed by " Is this true or false? Answer:". At the 9 prompt positions that Exp 3 decomposes (the statement's period and the 8 question tokens), components are removed or their sign is flipped. The model then answers greedily for up to 20 tokens on its original weights.
    - *Most single removals change how the answer starts, not the verdict.* The model opens with " for sure true.", " the statement is true." or " (I) False (II) True", or starts to reason. Where it states a verdict within the 20 tokens, the verdict is mostly right. The large first-token effects measured earlier (KL 3–11 nats) mostly reflect this change of format.
    - *Each always-active component has a signature that repeats across statements.* Removing or flipping 28 produces a fill-in-the-blank multiple-choice template, removing 381 produces " for sure …", and flipping 410 makes the model answer in Spanish or Portuguese ("Para responder…").
    - *Verdicts change in a few places only.* On the true city statement A, removing "True" writer 447 (or 258, 402, 410) gives " False. Hamhung is a city in North Korea, but the statement is not entirely accurate…": the model justifies the changed answer. A few flips also produce wrong verdicts (§4–5). Flipping 447 turns no true statement into "False".
    - *Removing whole groups gives garbage first tokens, but the text often recovers.* E.g. "9/10↵You're correct, the answer is 9/10. Hamhung…"; several answers switch to Chinese. Zeroing the MLP outputs instead keeps fluent answers but reverses both negated statements.
    - *Anecdotal:* six statements, one greedy run each.
  ],
)

#show table: set text(size: 8pt)

= What was done

*The decomposition.* Exp 3 (WandB run #runid("s-161f8a7b")) decomposes the MLP output matrices (`down_proj`) of layers 15–19 of Qwen2.5-7B-Instruct. Each matrix is written as a sum of 96 rank-1 *components* plus a remainder Δ: $W = sum_c u_c v_c^top + Delta$, 480 components in all. On input $x$, component $c$ writes $u_c (v_c dot x)$. Exp 3 was trained so that the components alone, at the statement's final period and the 8 question tokens, reproduce the model's next-token distribution after " Is this true or false? Answer:". Component numbers below are global: component $c$ sits in layer $15 + floor(c\/96)$.

*Interventions.* Each component gets a mask that multiplies its write. The original weights are all masks 1 with Δ on. *Removed* = mask 0 (the weights minus the component). *Flipped* = mask −1 (the component's write with its sign reversed: the weights minus twice the component). Δ stays on in both. They apply only at the 9 decomposed prompt positions; all other positions, including every generated token, run on the original weights. Answers are greedy, up to 20 tokens, stopping early at an end-of-text token.

*Caveat on what "removed" means.* Every MLP output $y$ at these positions is the sum of three writes: the *active* components' (causal importance above 0.01 there), the *inactive* components' and Δ's. If the decomposition reconstructed the MLP, the active write would be close to $y$ and the other two close to zero. It is not (100 test statements, CPU). At 6 of the 8 question tokens no component is active at all. At the final ":", about 30 active components write a vector of norm 156, five times the norm of $y$ (32) and nearly orthogonal to it (cosine 0.06). Since components and Δ still sum to the original weights, the inactive components plus Δ must write $y$ minus that vector, about 158 in the opposite direction. Elsewhere inactive components and Δ cancel each other (cosine −0.8). Exp 3's training only asked for the right answer at ":", so the decomposition found its own way to produce it: MLPs off at the question, large writes into answer-setting directions at ":". Removing a component from the original weights therefore adds the negative of its write to an otherwise intact model, a perturbation the original never sees. It does not leave "the model without this mechanism".

*Components shown.* The ten components active on at least 99% of statements in Exp 3 (8, 28, 53, 122, 140, 231, 258, 381, 402, 410), and the four "True" writers found earlier: 447 (layer 19) and 319, 371, 396. These are active almost only at the final ":" of statements the model judges true. Also shown: groups of components, and states of the whole decomposition (§3).

*Statements.* Six tiu test statements (Truth-is-Universal dataset; subjects unseen in training), chosen before looking at any intervention: for each wanted (family, label, original first token), the first match in a fixed random order (numpy seed 0). A–D answer directly with " True"/" False"; E is a false number comparison, answered " False"; F is a true number comparison. The model starts every true number comparison with " To" (the start of reasoning), so F shows that case.

*Reading the tables.* Each cell is the continuation after "Answer:". ␣ marks a leading space and ↵ a newline. Below it: the probability mass on true-like and false-like first tokens ("True", "Yes", "correct", "right" / "False", "No", "incorrect", "wrong", any capitalisation, with and without a leading space). The line under each table heading is the mean over all 1,794 test statements, from the earlier runs: the first token's KL divergence from the original (in nats) and the mass outside both token sets. The original model itself puts 0.41 there, mostly on " To".

= Reference: the original model

#refs

= States of the whole decomposition

#let base_section(key, title, what) = block(breakable: false)[
  == #title
  #what \
  #text(size: 8.5pt, fill: palette.muted)[Over all statements: #base.at(key).agg]
  #base.at(key).table
]

#base_section("components", "Components only", [Components with causal importance above 0.01 kept, all others and Δ off. Same verdicts, slightly different wording.])
#base_section("all_off", "MLPs 15–19 output zero (all off)", [Fluent answers in a different style (" Yes, it is true.", " The statement is false."). Both negated statements are reversed: C ("does not mean 'silver'", true) gets "false", D gets "true". This matches the main report: these layers are what turns negated statements around. On F the model answers directly instead of reasoning.])
#base_section("delta_only", "All components removed (Δ only)", [A garbage first token ("9/10", "素 樊", "顿悟法", "Helpful"), then text that often recovers and states the verdict ("95 > 58, so the statement is true").])
#base_section("negated_components", "Components only, all flipped (Δ off)", [The negative of the decomposition's MLP output. Garbage first token ("IENTATION", "flase", "post truth"), then mostly the right verdict; F switches to Chinese.])
#base_section("flip_all", "All components flipped (Δ on)", [Weights minus twice the components. The answer derails: B becomes a text about a logistics company, E about proportions.])

= The ten always-active components

#let comp_section(key, title, what) = block(breakable: false)[
  == #title
  #what \
  #text(size: 8.5pt, fill: palette.muted)[Over all statements, removed: #comp.at(key).agg_remove; flipped: #comp.at(key).agg_flip]
  #comp.at(key).table
]

#comp_section("8", "Component 8 (layer 15)", [Removed: the model starts to reason (" Considering…", " To determine…", " Accordin to the Spanish-English dictionary"). Flipped: an odd first token ("PressEvent", "ynchronously", "M:"), then mostly the right verdict.])
#comp_section("28", "Component 28 (layer 15)", [Removed and flipped: on A, C and D a fill-in-the-blank template, `␣____↵A. True↵B. False↵Answer:↵` with the correct letter (A, A, B). On F, removal makes the model answer " True." at once instead of reasoning.])
#comp_section("53", "Component 53 (layer 15)", [Removed: answers without the leading space ("True.", "The city of Xuchang is in China…") or with " Yes"; verdicts correct. Flipped: reasoning openers (" Coming from…", " Rationale…").])
#comp_section("122", "Component 122 (layer 16)", [Removed: hedged or deferred answers. A gets " we do not know. … is neither", C " is the statement true or false?", B " based on being factually accurate, this is false". Flipped: " A. True", " with explanation." and similar.])
#comp_section("140", "Component 140 (layer 16)", [Removed: " continue…" / " take the following steps"; on C the text drifts off topic (" continue to be stored in silver containers"). Flipped: "onestly, the statement is false" on A, which is wrong, and " do not include any source references" on D.])
#comp_section("231", "Component 231 (layer 17)", [Removed: " the statement is true/false", fluent and correct; " such a statement is false" on E. Flipped: " Or…" openers and a numbered-options template on C (" (I) True (II) False↵Answer: (I) True").])
#comp_section("258", "Component 258 (layer 17)", [Removed: A turns into a justified " False" (wrong); C, E and F open with " either true or false". Flipped: " at…" openers. C drifts to " at the molecular level, all silver atoms…", and D gets " at least partially true" (wrong).])
#comp_section("381", "Component 381 (layer 18)", [Removed: " for sure true/false" with the correct verdict on A, C, D and E, " (F) False." on B. Flipped: " (F) The statement is false." on A (wrong), `statistically_invalid` / "statistically improbable", and the numbered-options template on C.])
#comp_section("402", "Component 402 (layer 19)", [Removed: A turns into a justified " false" (wrong); B gets " &lt;false&gt;", D " EXPERT MODE: False.", and E repeats the question. Flipped: " is…" / " now…" openers, which restate the question.])
#comp_section("410", "Component 410 (layer 19)", [Removed: A turns into a justified " False" (wrong); multiple-choice templates on B, D, E (" (I) False (II) True", " A. True B. False"); C is unchanged. Flipped: "Para…", then Spanish or Portuguese on B, D and E ("Para responder a esta pregunta, primero debo analizar…").])

= Groups and the "True" writers

#comp_section("always_on", "All ten always-active components", [Removed: A starts "电报↵↵False." (wrong), C answers in Chinese ("…正确答案应为：True"), and E turns into a blog footer ("Entry filed under: Uncategorized. Tags: arithmetic, comparison, false…"). Flipped: A starts a new quiz item (" golden retriever is a breed of dog. Is this true or false?"); F answers in Chinese, correctly.])
#comp_section("447", "\"True\" writer 447 (layer 19)", [Removed: the true city statement A becomes " False. Hamhung is a city in North Korea, but the statement is not entirely accurate because it…": the answer flips and the model invents a reason. C keeps "True" after an extra space token; the false statements B, D, E are unchanged. Flipped: no true statement turns into "False"; A gets " 1. The statement to evaluate is…", C " 1. True.", F `␣␣$\boxed{\text{true}}$`.])
#comp_section("true_writers", "\"True\" writers 447, 319, 371, 396 together", [Removed: the true statements A, C and F turn to reasoning (" Chain of reasoning:", " Let's think first.") or to an options template; false statements keep " False." with more confidence. Flipped: C becomes wrong ("盎格鲁萨克森人↵↵False."); A gets ".Stdout:↵↵The statement … is true."])

= Period and question vs question tokens only

All runs above intervene at the period and the 8 question tokens, as Exp 3 was trained. Intervening at the 8 question tokens only (the statement's period on original weights) changes little. #so_counts.same of #so_counts.total intervened continuations are identical, and #so_counts.first_same of #so_counts.total start with the same token. The cases with a different first token:

#so_diff

= Caveats

#caveat[
  - *Six statements, one greedy answer each.* The signatures above (e.g. 28 → template, 410 flipped → Spanish) are seen on these six statements only. They are anecdotes, not measured rates.
  - *Generated tokens run on the original weights*, so the text can repair a broken first token. A change only at the answer's first token is weaker than a change of what the model ends up saying.
  - *Removed ≠ absent.* Components and Δ largely cancel, so a removal adds the negative of the component's write to the model (§1).
]

= Files

#text(size: 9pt)[#set par(justify: false)
Scripts and raw outputs: `~/spd_out/26-10-10_bolder_experiments/dist_shift/` (`ablation_examples.py`, `examples.json`, `examples_stimulus_only.json`, `examples_selection.json`; aggregates from `dist_shift.py`: `full/`, `flip/`). Tables generated by `make_examples_tables.py` into `reports/26-10-14_ablation_examples_tables.typ`. Project record: `convos/julian/26-10-10_bolder_experiments_LOG.md`, entry *ablation examples*.
]
