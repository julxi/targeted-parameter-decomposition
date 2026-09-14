"""
Build BARE true/false prompts files for capitals (no chat template, no
system prompt, no instruction wording) -- same construction logic already
used in visualize_token_firing_bare.py / pca_truth_geometry.py, just
exporting to prompts_file format for a real tPD training run instead of
an analysis script.

Run this locally (you have contrast_pairs_verified.jsonl, not me):

    python3 build_bare_prompts_capitals.py \
        --jsonl contrast_pairs.jsonl \
        --out-dir prompts_bare_capitals

Produces:
    prompts_bare_capitals/capitals_true.txt
    prompts_bare_capitals/capitals_false.txt
    prompts_bare_capitals/capitals_combined.txt
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="contrast_pairs.jsonl")
    ap.add_argument("--out-dir", default="prompts_capitals")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_questions = set()
    true_lines, false_lines = [], []
    skipped = 0

    with open(args.jsonl) as f:
        for line in f:
            p = json.loads(line)
            if p["question"] in seen_questions:
                continue
            seen_questions.add(p["question"])

            if p["correct_answer"] not in p["fact_sentence"]:
                skipped += 1
                continue
            false_sentence = p["fact_sentence"].replace(p["correct_answer"], p["wrong_answer"])
            if p["wrong_answer"] not in false_sentence:
                skipped += 1
                continue

            true_lines.append(p["fact_sentence"])
            false_lines.append(false_sentence)

    if skipped:
        print(f"Skipped {skipped} malformed question(s)")

    (out_dir / "capitals_true.txt").write_text("\n".join(true_lines) + "\n")
    (out_dir / "capitals_false.txt").write_text("\n".join(false_lines) + "\n")
    (out_dir / "capitals_combined.txt").write_text("\n".join(true_lines + false_lines) + "\n")

    print(f"Wrote {len(true_lines)} true + {len(false_lines)} false bare statements to {out_dir}/")
    print("Sample true:", true_lines[0])
    print("Sample false:", false_lines[0])


if __name__ == "__main__":
    main()
