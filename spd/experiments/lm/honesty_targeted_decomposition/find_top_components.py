"""
Scan ALL components (not a hand-picked subset) across a bare tPD run's
targeted layers, using the same bare (no chat template/system prompt)
true/false examples as visualize_token_firing_bare.py, and rank them by how
cleanly they separate true vs. false completions at the answer token.

This is the "which components should I even look at" step before running
visualize_token_firing_bare.py --components <ids> for a human-readable HTML.

Usage:
    python3 find_top_components.py \
        --config config_capitals_true.yaml \
        --checkpoint ~/spd_out/spd/s-d30f8be0/model_10000.pth \
        --contrast-jsonl contrast_pairs_verified.jsonl \
        --layers model.layers.15.mlp.down_proj model.layers.16.mlp.down_proj \
                 model.layers.17.mlp.down_proj model.layers.18.mlp.down_proj \
                 model.layers.19.mlp.down_proj \
        --top-k 8 \
        --out top_components_capitals_true.json
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoTokenizer

from visualize_token_firing import build_component_model, load_bare_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--contrast-jsonl", required=True)
    ap.add_argument("--layers", nargs="+", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--ci-alive-threshold", type=float, default=0.01)
    ap.add_argument("--out", default="top_components_bare.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    component_model, config = build_component_model(args.config, device)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    component_model.load_state_dict(state_dict)
    component_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    ci_fn_dtype = next(component_model.ci_fn.parameters()).dtype

    examples = load_bare_examples(args.contrast_jsonl, n_examples=None)
    print(f"Loaded {len(examples)} examples ({len(examples) // 2} unique questions x true/false)")

    per_layer_vals = {layer: [] for layer in args.layers}  # each: list of [C] answer-token CI
    labels = []

    with torch.no_grad():
        for ex in examples:
            prompt = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=True
            )
            full_text = prompt + ex["completion"]
            enc = tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=config.task_config.max_seq_len
            )
            input_ids = enc["input_ids"].to(device)
            token_strs = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            token_strs = [t.replace("Ġ", " ").replace("▁", " ") for t in token_strs]

            out = component_model(input_ids, cache_type="input")
            pre_weight_acts = {k: v.to(ci_fn_dtype) for k, v in out.cache.items()}
            ci_outputs = component_model.calc_causal_importances(pre_weight_acts, sampling=config.sampling)

            answer_pos = len(token_strs) - 1
            if token_strs[answer_pos].strip() in {".", ",", "!", "?"}:
                answer_pos -= 1

            for layer in args.layers:
                layer_ci = ci_outputs.lower_leaky[layer][0].cpu().numpy()  # [seq_len, C]
                per_layer_vals[layer].append(layer_ci[answer_pos])

            labels.append(ex["label"])

    labels = np.array(labels)
    true_mask = labels == "true"
    false_mask = labels == "false"

    results = {}
    for layer in args.layers:
        vals = np.stack(per_layer_vals[layer])  # [n_examples, C]
        true_vals = vals[true_mask]
        false_vals = vals[false_mask]
        mean_true = true_vals.mean(axis=0)
        mean_false = false_vals.mean(axis=0)
        alive = np.maximum(mean_true, mean_false) > args.ci_alive_threshold
        separation = np.abs(mean_true - mean_false)

        ranked = np.argsort(-separation)
        layer_results = []
        for c in ranked[: args.top_k]:
            layer_results.append(
                {
                    "component": int(c),
                    "mean_true": float(mean_true[c]),
                    "mean_false": float(mean_false[c]),
                    "separation": float(separation[c]),
                    "alive": bool(alive[c]),
                }
            )
        results[layer] = layer_results

        print(
            f"\n=== {layer}: top {len(layer_results)} by |mean_true - mean_false| "
            f"({int(alive.sum())} alive of {len(alive)} total) ==="
        )
        for r in layer_results:
            print(
                f"  component {r['component']:3d}: true={r['mean_true']:.4f} "
                f"false={r['mean_false']:.4f} sep={r['separation']:.4f} alive={r['alive']}"
            )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ranked components to {args.out}")


if __name__ == "__main__":
    main()
