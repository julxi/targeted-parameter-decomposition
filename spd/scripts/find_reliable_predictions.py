"""Find tokens the pretrained model predicts reliably on the target CSS dataset.

For each position where the model assigns probability > threshold to the correct next token,
outputs context, the predicted token, and the probability.

Usage:
    uv run python -m spd.scripts.find_reliable_predictions --n-batches 5 --prob-thr 0.3
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

from spd.data import DatasetConfig, create_data_loader
from spd.pretrain.run_info import PretrainRunInfo

DEFAULT_CONFIG = "spd/experiments/lm/pile_llama_simple_mlp-4L-targeted-css.yaml"


def escape_special(s: str) -> str:
    """Escape whitespace/special characters so the string is safe in a single TSV cell."""
    s = s.replace("\\", "\\\\")
    s = s.replace("\n", "\\n")
    s = s.replace("\t", "\\t")
    s = s.replace("\r", "\\r")
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="Find reliable next-token predictions on CSS data")
    parser.add_argument(
        "--config-path", type=Path, default=DEFAULT_CONFIG, help="Path to experiment YAML config"
    )
    parser.add_argument("--prob-thr", type=float, default=0.2, help="Probability threshold")
    parser.add_argument("--n-batches", type=int, default=100, help="Number of batches to process")
    parser.add_argument(
        "--output", type=Path, default=Path("reliable_predictions.tsv"), help="Output TSV path"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load config
    with open(args.config_path) as f:
        config_dict = yaml.safe_load(f)

    task_config = config_dict["task_config"]
    tokenizer_name = config_dict["tokenizer_name"]
    pretrained_model_name = config_dict["pretrained_model_name"]
    pretrained_model_class_path = config_dict["pretrained_model_class"]

    # Load model
    print(f"Loading model from {pretrained_model_name}...")
    run_info = PretrainRunInfo.from_path(pretrained_model_name)
    if "model_type" not in run_info.model_config_dict:
        run_info.model_config_dict["model_type"] = pretrained_model_class_path.split(".")[-1]

    from spd.utils.general_utils import resolve_class

    model_class = resolve_class(pretrained_model_class_path)
    model = model_class.from_run_info(run_info)  # pyright: ignore[reportAttributeAccessIssue]
    model.eval()
    model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    decode = tokenizer.decode

    # Load dataset
    print("Loading CSS dataset...")
    data_config = DatasetConfig(
        name=task_config["dataset_name"],
        hf_tokenizer_path=tokenizer_name,
        split=task_config["train_data_split"],
        n_ctx=task_config["max_seq_len"],
        is_tokenized=task_config["is_tokenized"],
        streaming=task_config["streaming"],
        column_name=task_config["column_name"],
        shuffle_each_epoch=False,
    )
    loader, _ = create_data_loader(
        dataset_config=data_config,
        batch_size=args.batch_size,
        buffer_size=1000,
        global_seed=0,
    )

    # Process batches
    rows: list[tuple[str, str, str, str]] = []
    n_total_positions = 0

    print(f"Processing {args.n_batches} batches (batch_size={args.batch_size})...")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.n_batches:
            break

        input_ids = batch[task_config["column_name"]].to(device)  # (B, seq_len)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(input_ids)  # (B, seq_len, vocab)

        # Probability of actual next token at each position
        probs = F.softmax(logits[:, :-1, :].float(), dim=-1)  # (B, seq_len-1, vocab)
        next_tokens = input_ids[:, 1:]  # (B, seq_len-1)
        next_token_probs = probs.gather(2, next_tokens.unsqueeze(2)).squeeze(2)  # (B, seq_len-1)

        n_total_positions += next_token_probs.numel()

        # Find hits
        hit_batch, hit_pos = torch.where(next_token_probs > args.prob_thr)

        for b, pos in zip(hit_batch.tolist(), hit_pos.tolist(), strict=True):
            # pos is the index into the shifted sequence: predicting token at position pos+1
            # Context is tokens 0..pos (inclusive)
            context_ids = input_ids[b, : pos + 1].tolist()
            next_tok_id = input_ids[b, pos + 1].item()
            prob = next_token_probs[b, pos].item()

            context_str = escape_special(decode(context_ids))
            short_tokens = [escape_special(decode([tid])) for tid in context_ids[-5:]]
            context_short_str = ";".join(short_tokens)
            next_tok_str = escape_special(decode([next_tok_id]))

            rows.append((context_str, context_short_str, next_tok_str, f"{prob:.4f}"))

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{args.n_batches}: {len(rows)} hits so far")

    # Write TSV
    print(f"Writing {len(rows)} rows to {args.output}")
    print(f"  ({len(rows)}/{n_total_positions} positions = {len(rows) / n_total_positions:.1%})")
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["context", "context_short", "next_token", "prob"])
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    main()
