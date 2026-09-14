"""
Same rendering as visualize_token_firing.py, but loads the model +
checkpoint ONCE and produces HTML/--summary output for MULTIPLE layers in a
single process. Useful on a contended/shared GPU where reloading a 7B model
once per layer multiplies the odds of losing a race against another job for
free memory -- one load here means one collision window instead of N.

Usage:
    python3 visualize_token_firing_multilayer.py \
        --config config_capitals_false.yaml \
        --checkpoint ~/spd_out/spd/s-047bca0c/model_10000.pth \
        --contrast-jsonl contrast_pairs_verified.jsonl \
        --layer-components model.layers.17.mlp.down_proj:50,77,33 \
                            model.layers.18.mlp.down_proj:64,50,13 \
                            model.layers.19.mlp.down_proj:87,22,58 \
        --n-examples 10 \
        --summary \
        --out-dir analysis_capitals/capitals_false
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig
from spd.experiments.lm.honesty_targeted_decomposition.visualize_token_firing import (
    build_component_model,
    load_bare_examples,
    render_example_html,
)


def parse_layer_components(spec: str) -> tuple[str, list[int]]:
    layer, components_str = spec.split(":")
    return layer, [int(c) for c in components_str.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--contrast-jsonl", required=True)
    ap.add_argument(
        "--layer-components",
        nargs="+",
        required=True,
        help="One or more LAYER:comp1,comp2,... specs",
    )
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    layer_components = dict(parse_layer_components(s) for s in args.layer_components)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    component_model, config = build_component_model(args.config, args.checkpoint, device)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    ci_fn_dtype = next(component_model.ci_fn.parameters()).dtype

    all_examples = load_bare_examples(args.contrast_jsonl, n_examples=None)
    display_examples = all_examples[: 2 * args.n_examples] if args.n_examples else all_examples
    print(
        f"Loaded {len(all_examples)} total examples "
        f"({len(all_examples) // 2} unique questions x true/false); "
        f"rendering {len(display_examples)} to HTML per layer"
    )

    layers = list(layer_components.keys())
    per_layer_examples = {layer: {c: [] for c in layer_components[layer]} for layer in layers}
    per_layer_all_values = {layer: {c: [] for c in layer_components[layer]} for layer in layers}
    per_layer_summary = {layer: {c: [] for c in layer_components[layer]} for layer in layers}

    with torch.no_grad():
        for idx, ex in enumerate(all_examples):
            is_displayed = idx < len(display_examples)
            prompt = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=True
            )
            full_text = prompt + ex["completion"]
            enc = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=task_config.max_seq_len,
            )
            input_ids = enc["input_ids"].to(device)
            token_strs = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            token_strs = [t.replace("Ġ", " ").replace("▁", " ") for t in token_strs]

            out = component_model(input_ids, cache_type="input")
            pre_weight_acts = {k: v.to(ci_fn_dtype) for k, v in out.cache.items()}
            ci_outputs = component_model.calc_causal_importances(
                pre_weight_acts, sampling=config.sampling
            )

            answer_pos = len(token_strs) - 1
            if token_strs[answer_pos].strip() in {".", ",", "!", "?"}:
                answer_pos -= 1

            for layer in layers:
                layer_ci = ci_outputs.lower_leaky[layer][0].cpu().numpy()  # [seq_len, C]
                for c in layer_components[layer]:
                    vals = layer_ci[:, c]
                    per_layer_summary[layer][c].append((ex["label"], float(vals[answer_pos])))
                    if is_displayed:
                        per_layer_examples[layer][c].append((token_strs, vals, ex["label"]))
                        per_layer_all_values[layer][c].extend(vals.tolist())

    for layer in layers:
        components = layer_components[layer]

        if args.summary:
            print(f"\n=== {layer}: quantitative summary, true vs. false ===")
            for c in components:
                true_vals = np.array(
                    [v for label, v in per_layer_summary[layer][c] if label == "true"]
                )
                false_vals = np.array(
                    [v for label, v in per_layer_summary[layer][c] if label == "false"]
                )
                threshold = 0.01
                frac_true = (
                    float((true_vals > threshold).mean()) if len(true_vals) else float("nan")
                )
                frac_false = (
                    float((false_vals > threshold).mean()) if len(false_vals) else float("nan")
                )
                print(
                    f"  component {c}: true mean={true_vals.mean():.4f} (>{threshold}: {frac_true * 100:.0f}%) "
                    f"| false mean={false_vals.mean():.4f} (>{threshold}: {frac_false * 100:.0f}%) "
                    f"| n={len(true_vals)} pairs"
                )

        html_parts = [
            "<html><head><meta charset='utf-8'><style>",
            "body { font-family: monospace; font-size: 14px; line-height: 2.2; }",
            "h2 { margin-top: 40px; border-bottom: 2px solid #333; }",
            ".label { font-weight: bold; margin-right: 8px; }",
            ".example { margin-bottom: 10px; padding: 6px; border: 1px solid #ddd; }",
            "</style></head><body>",
            f"<h1>Token-level firing (NO SYSTEM PROMPT): {layer}</h1>",
            f"<p>Data: {args.contrast_jsonl} -- bare user question + true/false completion only, "
            f"no honest/dishonest instruction wording anywhere in the input.</p>",
        ]
        for c in components:
            vmax = max(per_layer_all_values[layer][c]) if per_layer_all_values[layer][c] else 1.0
            html_parts.append(f"<h2>Component {c} (max value in this set: {vmax:.4f})</h2>")
            for token_strs, vals, label in per_layer_examples[layer][c]:
                label_color = "#1f6feb" if label == "true" else "#e8590c"
                rendered = render_example_html(token_strs, vals, vmax)
                html_parts.append(
                    f'<div class="example"><span class="label" style="color:{label_color}">'
                    f"[{label}]</span>{rendered}</div>"
                )
        html_parts.append("</body></html>")

        layer_tag = layer.replace("model.layers.", "L").replace(".mlp.down_proj", "")
        out_html = out_dir / f"token_firing_{layer_tag}.html"
        out_html.write_text("\n".join(html_parts))
        print(f"Saved HTML report to {out_html}")


if __name__ == "__main__":
    main()
