# capitals checkpoint analysis

Analysis of the two most recent tPD runs with saved checkpoints in
`~/spd_out/spd/` (out of all runs there, filtered to ones that actually
have `model_*.pth` files):

| run_id | label | date | config |
|---|---|---|---|
| `s-d30f8be0` | `capitals_true` | 2026-08-31 22:46 | `config_capitals_true.yaml` |
| `s-047bca0c` | `capitals_false` | 2026-09-01 00:09 | `config_capitals_false.yaml` |

Both are Qwen2.5-7B-Instruct targeted decompositions over
`model.layers.{15..19}.mlp.down_proj` (C=96 each), trained "bare" — no chat
template, no system prompt, no honest/dishonest instruction wording, just a
capital-city fact stated as either true or false. `capitals_true`'s
target data was ONLY true statements (`prompts_capitals/capitals_true.txt`);
`capitals_false`'s target data was ONLY false statements
(`capitals_false.txt`). Analysis uses `model_10000.pth` (final checkpoint,
step 10000) for each. Data: `contrast_pairs.jsonl` (35 unique
capital-city questions, run through `load_bare_examples()` as bare true/false
pairs — 70 examples total).

## What was run

1. **`find_top_components.py`** (new script) — scans all 96 components
   per layer (not a hand-picked subset), ranks by
   `|mean_true_CI - mean_false_CI|` at the answer token. Output:
   `<label>/top_components.json` + `<label>/find_top_components.log`.
2. **`visualize_token_firing.py`** (existing script) — per-token CI
   heatmap HTML + quantitative summary, run once per targeted layer using
   that layer's top-3 components from step 1. Output:
   `<label>/token_firing_L{15..19}.html`.

(`visualize_token_firing_multilayer.py` — new script, same rendering
but loads the model once for multiple layers instead of once per layer.
Written after repeated GPU OOMs from another job on this shared GPU raced
against per-layer reloads; used for the `capitals` L17-L19 retry.)

## Headline finding: a strong true/false-vs-false/true asymmetry

**`capitals_false`** (target = false statements only) has genuinely
alive, strongly separating components in every one of the 5 targeted layers:

| layer | top component | mean CI (true) | mean CI (false) | separation |
|---|---|---|---|---|
| 15 | 62 | 0.22 | 0.40 | 0.17 |
| 16 | 67 | 0.25 | 0.92 | **0.67** |
| 17 | 50 | 0.42 | 0.95 | 0.53 |
| 18 | 64 | 0.22 | 0.87 | 0.65 |
| 19 | 87 | 0.51 | 0.97 | 0.46 |

Each of these fires on **100% of both true and false examples** (not a
binary on/off switch) but at a consistently, substantially higher magnitude
for false completions — a 2-4x jump in every layer.

**`capitals_true`** (target = true statements only) has essentially no
alive components anywhere: the best separation across all 5 layers x 96
components is 0.018 (layer 19, component 54; true=0.0011, false=0.0169),
and even that clears a 0.01 "alive" threshold on only 1-2 components per
layer, each barely above it.

**Reading**: representing a *true* capital-city fact appears to need almost
no dedicated component machinery — the base model already produces true
completions as close to its default behavior, so there's little for
targeted decomposition to isolate. Representing a *false* one is a
different story — every targeted layer has a specific, strongly-engaged
component when the model has to produce (or has been shown) a fabricated
capital, and those components are shared across nearly all false examples,
not example-specific noise. This is consistent with "generating a
falsehood requires an active, identifiable mechanism; generating the truth
mostly doesn't" — though note both runs were trained on ONE polarity of
target data each, so this describes what each decomposition needed in
order to reconstruct its own one-sided target distribution, not necessarily
a general "truth detector"/"lie detector" pair.

## Caveat

Both models are single-polarity training runs (never saw the *other*
completion type as target data during training) — this isn't the same
setup as the earlier `honest_only`/`dishonest_only` runs (`s-68e97c43`,
`s-5d94ea17`), which used instructed honest-vs-dishonest chat-template
data. A direct next step would be running the strongly-alive
`capitals_false` components (67 @ L16, 64 @ L18, 87 @ L19) against the
*other* checkpoint's bare true/false eval, or against the Liu et al.
12-domain bare set once `config_liu_true/false.yaml` are trained, to
see whether these are capitals-specific or reflect something more general.

## Files

```
analysis_capitals/
├── SUMMARY.md                          (this file)
├── capitals_true/
│   ├── top_components.json             (top-10 per layer, all 96 scanned)
│   ├── find_top_components.log
│   └── token_firing_L15.html ... L19.html
└── capitals_false/
    ├── top_components.json
    ├── find_top_components.log
    └── token_firing_L15.html ... L19.html
```
