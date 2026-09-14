import argparse
import html
import json
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import resolve_class
from spd.utils.module_utils import expand_module_patterns
from spd.utils.run_utils import parse_config


def build_component_model(config_path: str, checkpoint_path: str, device: str):
    config = parse_config(config_path, None)
    pretrained_model_class = resolve_class(config.pretrained_model_class)
    assert hasattr(pretrained_model_class, "from_pretrained"), (
        f"Model class {pretrained_model_class} should have a `from_pretrained` method"
    )
    # torch_dtype + device_map set directly in from_pretrained (not a separate
    # .to(device) call afterward) -- avoids both an accidental fp32 default
    # (Qwen2.5-7B in fp32 needs ~28GB, won't fit regardless of what else is
    # running) and the transient memory spike from loading fully then moving.
    target_model = pretrained_model_class.from_pretrained(  # pyright: ignore[reportAttributeAccessIssue]
        config.pretrained_model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    target_model.eval()
    target_model.requires_grad_(False)

    module_path_info = expand_module_patterns(target_model, config.all_module_info)
    component_model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_config=config.ci_config,
        sigmoid_type=config.sigmoid_type,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
    )
    component_model.to(device)
    component_model.load_component_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    component_model.eval()
    return component_model, config


def load_bare_examples(jsonl_path: str, n_examples: int | None) -> list[dict[str, Any]]:
    """Build NO-SYSTEM-PROMPT examples: just user question + true/false answer.
    Uses each pair's own question/correct_answer/wrong_answer fields, works
    for both the capitals schema and the non-country probe schema.

    Validates the replace() actually worked (catches the silent-truncation-
    looking bug from the first version: if correct_answer isn't found in
    fact_sentence for some reason, replace() is a silent no-op and you get a
    malformed/mismatched example instead of a clear error).
    """
    examples = []
    seen_questions = set()
    skipped = 0
    with open(jsonl_path) as f:
        for line in f:
            p = json.loads(line)
            if p["question"] in seen_questions:
                continue
            seen_questions.add(p["question"])

            if p["correct_answer"] not in p["fact_sentence"]:
                print(
                    f"  [skip] correct_answer {p['correct_answer']!r} not found in "
                    f"fact_sentence {p['fact_sentence']!r} -- skipping this question"
                )
                skipped += 1
                continue

            false_sentence = p["fact_sentence"].replace(p["correct_answer"], p["wrong_answer"])
            if p["wrong_answer"] not in false_sentence:
                print(
                    f"  [skip] replace() didn't produce expected wrong_answer for "
                    f"{p['question']!r} -- skipping"
                )
                skipped += 1
                continue

            messages = [{"role": "user", "content": p["question"]}]
            true_completion = f" {p['fact_sentence']}"
            false_completion = f" {false_sentence}"

            examples.append({"messages": messages, "completion": true_completion, "label": "true"})
            examples.append(
                {"messages": messages, "completion": false_completion, "label": "false"}
            )
            if n_examples is not None and len(examples) >= 2 * n_examples:
                break

    if skipped:
        print(f"Skipped {skipped} malformed question(s) during construction.")
    return examples


def color_for_value(v: float, vmax: float) -> str:
    t = min(max(v / vmax, 0.0), 1.0) if vmax > 0 else 0.0
    r = 255
    g = int(255 * (1 - t))
    b = int(255 * (1 - t))
    return f"rgb({r},{g},{b})"


def render_example_html(tokens: list[str], ci_values: list[float], vmax: float) -> str:
    spans = []
    for tok, val in zip(tokens, ci_values, strict=True):
        color = color_for_value(val, vmax)
        display_tok = html.escape(tok).replace("\n", "\\n")
        spans.append(
            f'<span style="background-color:{color};padding:1px 2px;border-radius:2px;" '
            f'title="{val:.4f}">{display_tok}</span>'
        )
    return "".join(spans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--contrast-jsonl", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--components", type=int, nargs="+", required=True)
    ap.add_argument(
        "--n-examples",
        type=int,
        default=10,
        help="Number of PAIRS (true + false each) -- deduped by question first. "
        "This caps the HTML output only; --summary always uses the full "
        "deduped set regardless of this value.",
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Also compute the last-answer-token CI value for EVERY example "
        "(not just the ones rendered to HTML) and report, per component, "
        "how cleanly true vs. false completions separate -- the quantitative "
        "version of eyeballing the HTML.",
    )
    ap.add_argument("--out", default="token_firing_bare.html")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    component_model, config = build_component_model(args.config, args.checkpoint, device)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    ci_fn_dtype = next(component_model.ci_fn.parameters()).dtype

    # Full deduped set for the summary; HTML rendering below still respects
    # --n-examples via load_bare_examples' own cap for display purposes.
    all_examples = load_bare_examples(args.contrast_jsonl, n_examples=None)
    display_examples = all_examples[: 2 * args.n_examples] if args.n_examples else all_examples
    print(
        f"Loaded {len(all_examples)} total examples "
        f"({len(all_examples) // 2} unique questions x true/false); "
        f"rendering {len(display_examples)} to HTML"
    )

    per_component_examples = {c: [] for c in args.components}
    all_values_by_component = {c: [] for c in args.components}
    # (label, answer_token_ci) per component, for the quantitative summary
    summary_data = {c: [] for c in args.components}

    with torch.no_grad():
        for idx, ex in enumerate(all_examples):
            is_displayed = idx < len(display_examples)
            prompt = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=True
            )
            full_text = prompt + ex["completion"]
            enc_full = tokenizer(full_text, return_tensors="pt")
            n_tokens = enc_full["input_ids"].shape[1]
            if n_tokens > task_config.max_seq_len:
                print(
                    f"  [warn] example truncated: {n_tokens} tokens > "
                    f"max_seq_len={task_config.max_seq_len} -- {ex['messages'][0]['content']!r}"
                )
            enc = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=task_config.max_seq_len,
            )
            input_ids = enc["input_ids"].to(device)
            token_strs = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            token_strs = [t.replace("\u0120", " ").replace("\u2581", " ") for t in token_strs]

            out = component_model(input_ids, cache_type="input")
            pre_weight_acts = {k: v.to(ci_fn_dtype) for k, v in out.cache.items()}
            ci_outputs = component_model.calc_causal_importances(
                pre_weight_acts, sampling=config.sampling
            )

            layer_ci = ci_outputs.lower_leaky[args.layer][0].cpu().numpy()  # [seq_len, C]
            # "answer token" position: last token if not pure punctuation,
            # else the one before it (completions consistently end in '.')
            answer_pos = len(token_strs) - 1
            if token_strs[answer_pos].strip() in {".", ",", "!", "?"}:
                answer_pos -= 1

            for c in args.components:
                vals = layer_ci[:, c]
                summary_data[c].append((ex["label"], float(vals[answer_pos])))
                if is_displayed:
                    per_component_examples[c].append((token_strs, vals, ex["label"]))
                    all_values_by_component[c].extend(vals.tolist())

    if args.summary:
        print("\n=== Quantitative summary: last-answer-token CI, true vs. false ===")
        for c in args.components:
            true_vals = [v for label, v in summary_data[c] if label == "true"]
            false_vals = [v for label, v in summary_data[c] if label == "false"]
            true_arr = np.array(true_vals)
            false_arr = np.array(false_vals)
            threshold = 0.01
            frac_true_above = (
                float((true_arr > threshold).mean()) if len(true_arr) else float("nan")
            )
            frac_false_above = (
                float((false_arr > threshold).mean()) if len(false_arr) else float("nan")
            )
            print(
                f"component {c}: true mean={true_arr.mean():.4f} (>{threshold}: "
                f"{frac_true_above * 100:.0f}%) | false mean={false_arr.mean():.4f} "
                f"(>{threshold}: {frac_false_above * 100:.0f}%) | n={len(true_vals)} pairs"
            )

    html_parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body { font-family: monospace; font-size: 14px; line-height: 2.2; }",
        "h2 { margin-top: 40px; border-bottom: 2px solid #333; }",
        ".label { font-weight: bold; margin-right: 8px; }",
        ".example { margin-bottom: 10px; padding: 6px; border: 1px solid #ddd; }",
        "</style></head><body>",
        f"<h1>Token-level firing (NO SYSTEM PROMPT): {args.layer}</h1>",
        f"<p>Data: {args.contrast_jsonl} -- bare user question + true/false completion only, "
        f"no honest/dishonest instruction wording anywhere in the input.</p>",
    ]

    for c in args.components:
        vmax = max(all_values_by_component[c]) if all_values_by_component[c] else 1.0
        html_parts.append(f"<h2>Component {c} (max value in this set: {vmax:.4f})</h2>")
        for token_strs, vals, label in per_component_examples[c]:
            label_color = "#1f6feb" if label == "true" else "#e8590c"
            rendered = render_example_html(token_strs, vals, vmax)
            html_parts.append(
                f'<div class="example"><span class="label" style="color:{label_color}">'
                f"[{label}]</span>{rendered}</div>"
            )

    html_parts.append("</body></html>")

    with open(args.out, "w") as f:
        f.write("\n".join(html_parts))
    print(f"Saved HTML report to {args.out} -- open it in a browser")


if __name__ == "__main__":
    main()
