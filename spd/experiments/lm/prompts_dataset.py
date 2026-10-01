"""Utilities for loading lists of prompts into padded, position-masked LM batches."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.configs import AllPositions, LastKPositions, LossPositions, TokenPositions
from spd.log import logger
from spd.utils.distributed_utils import DistributedState
from spd.utils.general_utils import POSITION_MASK_KEY, PositionMask, extract_batch_data


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


class PositionMaskedLoader:
    """Adds the loss-position mask to every batch of an unpadded (packed) token loader."""

    def __init__(
        self, loader: DataLoader[Any], loss_positions: TokenPositions | LastKPositions
    ) -> None:
        self.loader = loader
        self.loss_positions: TokenPositions | LastKPositions = loss_positions

    def __iter__(self) -> Iterator[Any]:
        for batch in self.loader:
            input_ids = extract_batch_data(batch)
            batch_size, seq_len = input_ids.shape
            n_tokens = torch.full((batch_size,), seq_len)
            position_mask = build_position_mask(n_tokens, seq_len, self.loss_positions)
            assert position_mask is not None
            yield {"input_ids": input_ids, POSITION_MASK_KEY: position_mask}


def build_position_mask(
    n_tokens: Int[Tensor, " batch"], seq_len: int, loss_positions: LossPositions
) -> PositionMask | None:
    """Mask of the positions that contribute to losses, given each sequence's number of
    non-padding tokens (padding is on the right). None means every position contributes."""
    positions = torch.arange(seq_len).unsqueeze(0)
    is_token = positions < n_tokens.unsqueeze(1)
    match loss_positions:
        case AllPositions():
            return None
        case TokenPositions():
            return is_token
        case LastKPositions(k=k):
            assert (n_tokens >= k).all(), (
                f"last_k={k} exceeds the shortest sequence ({int(n_tokens.min())} tokens)"
            )
            return is_token & (positions >= (n_tokens - k).unsqueeze(1))


def read_prompts_file(prompts_file: Path) -> list[str]:
    """One prompt per non-empty line."""
    assert prompts_file.exists(), f"Prompts file not found: {prompts_file}"
    prompts = [p.strip() for p in prompts_file.read_text().strip().split("\n") if p.strip()]
    assert len(prompts) > 0, f"No prompts found in {prompts_file}"
    logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
    return prompts


def load_prompts_dataset(
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    loss_positions: LossPositions,
) -> Dataset:
    """Tokenize prompts into an HF Dataset, right-padded to max_seq_len (errors if exceeded).

    Returns:
        HuggingFace Dataset with an 'input_ids' column of shape (n_prompts, max_seq_len), plus a
        'position_mask' column of the same shape unless loss_positions selects every position.
    """
    assert len(prompts) > 0, "No prompts"

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
    columns: dict[str, Any] = {"input_ids": input_ids}
    position_mask = build_position_mask(torch.tensor(lengths), max_seq_len, loss_positions)
    if position_mask is not None:
        columns[POSITION_MASK_KEY] = position_mask.tolist()
    dataset = Dataset.from_dict(columns)
    dataset = dataset.with_format("torch")

    return dataset


def create_prompts_data_loader(
    prompts: list[str],
    tokenizer_name: str,
    max_seq_len: int,
    loss_positions: LossPositions,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int = 0,
) -> tuple[DataLoader[Any] | StaticBatchLoader, PreTrainedTokenizer]:
    """Create a DataLoader from a list of prompts.

    Args:
        prompts: The prompts, one sequence each
        tokenizer_name: HuggingFace tokenizer name/path
        max_seq_len: Maximum sequence length
        loss_positions: Which positions contribute to losses (sets the 'position_mask' column)
        batch_size: Batch size for the DataLoader
        dist_state: Distributed state for multi-GPU training
        seed: Random seed for shuffling

    Returns:
        Tuple of (DataLoader or StaticBatchLoader, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = load_prompts_dataset(prompts, tokenizer, max_seq_len, loss_positions)

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

        batch = dataset[:batch_size]
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
