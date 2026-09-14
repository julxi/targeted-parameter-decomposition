"""Extract a CSS-only subset of the Pile and upload to HuggingFace Hub.

Downloads only the CSS parquet files from andstor/the_pile_github (the Pile's GitHub subset
with language metadata), strips comments, tokenizes, and chunks to match the Pile's sequence
length. Output chunks are uploaded as parquet shards as they are produced — at most one shard
is on disk at a time.

Usage:
    uv run python -m spd.scripts.extract_css_from_pile --hf_repo <username>/pile-css-no-comments
"""

import argparse
import re
import tempfile
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

PILE_GITHUB_REPO = "andstor/the_pile_github"
PILE_TOKENIZED_REPO = "danbraunai/pile-uncopyrighted-tok-shuffled"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
SHARD_SIZE = 10_000  # chunks per output parquet shard


def get_pile_seq_len() -> int:
    """Check the sequence length of the tokenized Pile dataset."""
    ds = load_dataset(PILE_TOKENIZED_REPO, split="train", streaming=True)
    seq_len = len(next(iter(ds))["input_ids"])
    print(f"Pile sequence length: {seq_len}")
    return seq_len


def list_css_parquet_files() -> dict[str, list[str]]:
    """List CSS parquet files in the Pile GitHub repo, grouped by split."""
    api = HfApi()
    all_files = api.list_repo_files(PILE_GITHUB_REPO, repo_type="dataset")
    result: dict[str, list[str]] = {}
    for split in ("train", "validation"):
        files = sorted(f for f in all_files if f.startswith(f"data/{split}/CSS/"))
        print(f"Found {len(files)} CSS parquet file(s) for '{split}' split")
        result[split] = files
    return result


def iter_css_texts(parquet_files: list[str]) -> Iterator[str]:
    """Download parquet files one by one and yield comment-stripped CSS texts.

    Reads each parquet in batches to avoid loading all rows into memory at once.
    """
    for path in parquet_files:
        print(f"  Downloading {path}...")
        local = hf_hub_download(PILE_GITHUB_REPO, path, repo_type="dataset")
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=512, columns=["text"]):
            for text in batch.column("text").to_pylist():
                stripped = CSS_COMMENT_RE.sub("", text)
                if len(stripped.strip()) >= 20:
                    yield stripped


def process_and_upload(
    docs: Iterator[str],
    seq_len: int,
    hf_repo: str,
    split: str,
    max_docs: int | None,
) -> int:
    """Tokenize docs, chunk the token stream, and upload parquet shards to HF Hub.

    Each shard is written to a temp file, uploaded, then deleted — at most one shard is on disk
    at a time. Returns the total number of chunks uploaded.
    """
    api = HfApi()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    schema = pa.schema([("input_ids", pa.list_(pa.int32()))])

    buf: list[int] = []
    shard_buf: list[list[int]] = []
    n_docs = 0
    n_chunks = 0
    shard_idx = 0

    def flush_shard() -> None:
        nonlocal shard_buf, shard_idx
        if not shard_buf:
            return
        table = pa.table({"input_ids": shard_buf}, schema=schema)
        shard_name = f"shard-{shard_idx:05d}.parquet"
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp:
            pq.write_table(table, tmp.name)
            path_in_repo = f"data/{split}/{shard_name}"
            print(f"  Uploading {shard_name} ({len(shard_buf):,} chunks) → {path_in_repo}")
            api.upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo=path_in_repo,
                repo_id=hf_repo,
                repo_type="dataset",
            )
        shard_idx += 1
        shard_buf = []

    for text in docs:
        tokens: list[int] = tokenizer.encode(text)
        buf.extend(tokens)
        while len(buf) >= seq_len:
            shard_buf.append(buf[:seq_len])
            buf = buf[seq_len:]
            n_chunks += 1
            if len(shard_buf) >= SHARD_SIZE:
                flush_shard()

        n_docs += 1
        if n_docs % 5000 == 0:
            print(f"  {n_docs:,} docs → {n_chunks:,} chunks ({shard_idx} shards uploaded)")
        if max_docs is not None and n_docs >= max_docs:
            break

    flush_shard()
    print(f"  {n_docs:,} docs → {n_chunks:,} chunks in {shard_idx} shards")
    return n_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf_repo", required=True, help="HF Hub repo (e.g. user/pile-css)")
    parser.add_argument("--max_docs", type=int, default=None, help="Cap on CSS documents")
    args = parser.parse_args()

    seq_len = get_pile_seq_len()

    # Find CSS parquet files via Hub API (avoids resolving the full 50+ language dataset)
    print("\nListing CSS parquet files...")
    css_files = list_css_parquet_files()
    train_files = css_files["train"]
    val_files = css_files["validation"]
    assert train_files, "No CSS train parquet files found"

    # Create HF repo
    api = HfApi()
    api.create_repo(args.hf_repo, repo_type="dataset", exist_ok=True)

    # Process and upload train split
    print("\nProcessing train split...")
    n_train = process_and_upload(
        iter_css_texts(train_files), seq_len, args.hf_repo, "train", args.max_docs
    )
    assert n_train > 0, "No train chunks produced"

    # Process and upload val split
    print("\nProcessing validation split...")
    n_val = process_and_upload(
        iter_css_texts(val_files), seq_len, args.hf_repo, "val", max_docs=None
    )

    print(f"\nDone! {n_train:,} train + {n_val:,} val chunks → {args.hf_repo}")


if __name__ == "__main__":
    main()
