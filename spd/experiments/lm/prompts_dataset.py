"""Utility for loading prompts from a text file into a HuggingFace Dataset."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.log import logger
from spd.utils.distributed_utils import DistributedState


class StaticBatchLoader:
    """A simple loader that yields the same cached batch forever.

    Used when the prompts file is small enough to fit in a single batch,
    avoiding the overhead of DataLoader iteration and reshuffling.
    """

    def __init__(self, batch: dict[str, Tensor]):
        self.batch = batch

    def __iter__(self) -> Iterator[Any]:
        while True:
            yield self.batch


def load_prompts_dataset(
    prompts_file: Path,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
) -> Dataset:
    """Load prompts from text file and tokenize into HF Dataset.

    Args:
        prompts_file: Path to text file with one prompt per line
        tokenizer: Tokenizer to use for encoding prompts
        max_seq_len: Maximum sequence length (prompts are padded to this, errors if exceeded)

    Returns:
        HuggingFace Dataset with 'input_ids' column of shape (n_prompts, max_seq_len)
    """
    assert prompts_file.exists(), f"Prompts file not found: {prompts_file}"

    prompts = prompts_file.read_text().strip().split("\n")
    prompts = [p.strip() for p in prompts if p.strip()]

    assert len(prompts) > 0, f"No prompts found in {prompts_file}"
    logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")

    # Set pad_token_id if not set (common for GPT-style tokenizers)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    encoded: Any = tokenizer(prompts)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    too_long = [(i, length) for i, length in enumerate(lengths) if length > max_seq_len]
    if too_long:
        idx, length = too_long[0]
        raise ValueError(
            f"Prompt at index {idx} has {length} tokens, exceeding max_seq_len={max_seq_len}. "
            f"Found {len(too_long)} prompts exceeding limit."
        )

    # Pad to max_seq_len
    pad_token_id = tokenizer.pad_token_id
    input_ids = [ids + [pad_token_id] * (max_seq_len - len(ids)) for ids in encoded["input_ids"]]
    dataset = Dataset.from_dict({"input_ids": input_ids})
    dataset = dataset.with_format("torch")

    return dataset


def create_prompts_data_loader(
    prompts_file: Path,
    tokenizer_name: str,
    max_seq_len: int,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int = 0,
) -> tuple[DataLoader[Any] | StaticBatchLoader, PreTrainedTokenizer]:
    """Create a DataLoader from a prompts text file.

    Args:
        prompts_file: Path to text file with one prompt per line
        tokenizer_name: HuggingFace tokenizer name/path
        max_seq_len: Maximum sequence length
        batch_size: Batch size for the DataLoader
        dist_state: Distributed state for multi-GPU training
        seed: Random seed for shuffling

    Returns:
        Tuple of (DataLoader or StaticBatchLoader, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = load_prompts_dataset(prompts_file, tokenizer, max_seq_len)

    n_prompts = len(dataset)

    # For small datasets (single batch, non-distributed), cache and yield forever
    if n_prompts <= batch_size and dist_state is None:
        if n_prompts < batch_size:
            from datasets import concatenate_datasets

            n_repeats = (batch_size + n_prompts - 1) // n_prompts
            logger.info(
                f"Repeating {n_prompts} prompts {n_repeats}x to fill batch_size={batch_size}"
            )
            dataset = concatenate_datasets([dataset] * n_repeats)
            dataset = dataset.with_format("torch")

        batch = {"input_ids": dataset[:batch_size]["input_ids"]}
        return StaticBatchLoader(batch), tokenizer

    # For larger datasets or distributed training, use standard DataLoader.
    # With DistributedSampler(drop_last=True), each rank gets len(dataset) // world_size samples,
    # so we need at least batch_size * world_size total samples.
    world_size = dist_state.world_size if dist_state is not None else 1
    min_samples = batch_size * world_size
    if n_prompts < min_samples:
        from datasets import concatenate_datasets

        n_repeats = (min_samples + n_prompts - 1) // n_prompts
        logger.info(f"Repeating {n_prompts} prompts {n_repeats}x to fill batch_size={batch_size}")
        dataset = concatenate_datasets([dataset] * n_repeats)
        dataset = dataset.with_format("torch")

    from torch.utils.data import DistributedSampler

    sampler = None
    if dist_state is not None:
        sampler = DistributedSampler(
            dataset,  # pyright: ignore[reportArgumentType]
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
        generator=generator,
    )

    return loader, tokenizer
