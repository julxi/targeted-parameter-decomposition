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

from spd.configs import AllPositions, LastKPositions, LossPositions, StimulusLens, TokenPositions
from spd.log import logger
from spd.utils.distributed_utils import DistributedState
from spd.utils.general_utils import (
    POSITION_MASK_KEY,
    STIMULUS_END_KEY,
    PositionMask,
    extract_batch_data,
)


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
        case TokenPositions(start=start):
            assert (n_tokens > start).all(), (
                f"tokens from start={start}: a sequence has only {int(n_tokens.min())} tokens"
            )
            return is_token & (positions >= start)
        case LastKPositions(k=k):
            assert (n_tokens >= k).all(), (
                f"last_k={k} exceeds the shortest sequence ({int(n_tokens.min())} tokens)"
            )
            return is_token & (positions >= (n_tokens - k).unsqueeze(1))


def trim_to_loss_positions(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Cut a right-padded batch after the last position any loss reads: its last loss position, or
    with an appended stimulus (STIMULUS_END_KEY) the last stimulus token, if that comes later.

    Later positions are padding that no loss reads: masked losses select by `position_mask`,
    unselected positions run on the original weights, the stimulus lens reads only the stimulus end,
    and with causal attention padding on the right cannot influence earlier positions. So dropping
    them only saves compute; tests/test_prompts_padding.py checks that the losses are unchanged.
    The batch width then varies from batch to batch (at most max_seq_len).
    """
    input_ids, position_mask = batch["input_ids"], batch[POSITION_MASK_KEY]
    assert input_ids.shape == position_mask.shape, (input_ids.shape, position_mask.shape)
    positions = torch.arange(1, position_mask.shape[1] + 1)
    width = int((position_mask * positions).max())
    assert width > 0, "batch without loss positions"
    trimmed = dict(batch)
    if STIMULUS_END_KEY in batch:
        width = max(width, int(batch[STIMULUS_END_KEY].max()) + 1)
    trimmed["input_ids"] = input_ids[:, :width]
    trimmed[POSITION_MASK_KEY] = position_mask[:, :width]
    return trimmed


def _collate_trimmed(rows: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return trim_to_loss_positions({key: torch.stack([row[key] for row in rows]) for key in rows[0]})


def read_prompts_file(prompts_file: Path) -> list[str]:
    """One prompt per non-empty line."""
    assert prompts_file.exists(), f"Prompts file not found: {prompts_file}"
    prompts = [p.strip() for p in prompts_file.read_text().strip().split("\n") if p.strip()]
    assert len(prompts) > 0, f"No prompts found in {prompts_file}"
    logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
    return prompts


def _tokenize_jointly(
    prompts: list[str], stimulus: str, tokenizer: PreTrainedTokenizer
) -> tuple[list[list[int]], list[int]]:
    """Token ids of each prompt + stimulus as one string, and each prompt's length in tokens = the
    index of the token holding the prompt's last character, plus one. That token may also hold the
    stimulus's first characters (Qwen: '.' + '\n' -> '.\n'); it counts as the prompt's. All tokens
    before it must equal the prompt's own tokens, and the tokens after it must be the same for
    every prompt (checked)."""
    ids_all, lengths, suffixes = [], [], set()
    for prompt in prompts:
        enc: Any = tokenizer(
            prompt + stimulus, return_offsets_mapping=True, add_special_tokens=False
        )
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        last_char = len(prompt) - 1
        holding = [i for i, (start, end) in enumerate(offsets) if start <= last_char < end]
        assert len(holding) == 1, (prompt, holding)
        n_prompt = holding[0] + 1
        own = tokenizer.encode(prompt, add_special_tokens=False)
        assert ids[: n_prompt - 1] == own[: n_prompt - 1], (prompt, ids, own)
        assert n_prompt < len(ids), (
            f"stimulus fully merged into the prompt's last token: {prompt!r}"
        )
        suffixes.add(tuple(ids[n_prompt:]))
        ids_all.append(ids)
        lengths.append(n_prompt)
    assert len(suffixes) == 1, f"stimulus tokenized differently after different prompts: {suffixes}"
    return ids_all, lengths


def load_prompts_dataset(
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    loss_positions: LossPositions,
    stimulus: StimulusLens | None,
) -> Dataset:
    """Tokenize prompts into an HF Dataset, right-padded to max_seq_len (errors if exceeded).

    With a `stimulus` (configs.StimulusLens), its text is appended to every prompt's tokens. It is
    tokenized on its own, so the prompt's tokens are the same as without it (joint tokenization can
    merge across the boundary, e.g. Qwen merges '.' + '\n' into one token). `loss_positions` then
    selects among the prompt's tokens only; with `stimulus.decompose_stimulus` every stimulus token
    is selected too. The selected positions are the decomposed ones; the stimulus lens reads the
    model at the stimulus's last token.

    Returns:
        HuggingFace Dataset with an 'input_ids' column of shape (n_prompts, max_seq_len), plus a
        'position_mask' column of the same shape unless loss_positions selects every position, plus
        with a stimulus a 'stimulus_end' column (index of each row's last stimulus token).
    """
    assert len(prompts) > 0, "No prompts"

    # Set pad_token_id if not set (common for GPT-style tokenizers)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    encoded: Any = tokenizer(prompts)
    prompt_lengths = [len(ids) for ids in encoded["input_ids"]]
    if stimulus is not None and stimulus.tokenize_jointly:
        assert not isinstance(loss_positions, AllPositions), (
            "a stimulus needs loss_positions within the prompt, not 'all'"
        )
        encoded["input_ids"], prompt_lengths = _tokenize_jointly(
            prompts, stimulus.stimulus, tokenizer
        )
        logger.info(
            f"Stimulus {stimulus.stimulus!r} tokenized jointly with every prompt "
            f"(decompose_stimulus={stimulus.decompose_stimulus}); e.g. "
            f"{[tokenizer.decode([i]) for i in encoded['input_ids'][0]]}, prompt = the first "
            f"{prompt_lengths[0]} tokens"
        )
    elif stimulus is not None:
        assert not isinstance(loss_positions, AllPositions), (
            "a stimulus needs loss_positions within the prompt, not 'all'"
        )
        stimulus_ids = tokenizer.encode(stimulus.stimulus, add_special_tokens=False)
        assert len(stimulus_ids) > 0, f"empty stimulus {stimulus.stimulus!r}"
        logger.info(
            f"Appending stimulus {stimulus.stimulus!r} = tokens "
            f"{[tokenizer.decode([i]) for i in stimulus_ids]} to every prompt "
            f"(decompose_stimulus={stimulus.decompose_stimulus}); e.g. "
            f"{tokenizer.decode(encoded['input_ids'][0] + stimulus_ids)!r}"
        )
        encoded["input_ids"] = [ids + stimulus_ids for ids in encoded["input_ids"]]
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
    position_mask = build_position_mask(torch.tensor(prompt_lengths), max_seq_len, loss_positions)
    if stimulus is not None and stimulus.decompose_stimulus:
        assert position_mask is not None
        positions = torch.arange(max_seq_len).unsqueeze(0)
        in_stimulus = (positions >= torch.tensor(prompt_lengths).unsqueeze(1)) & (
            positions < torch.tensor(lengths).unsqueeze(1)
        )
        position_mask = position_mask | in_stimulus
    if position_mask is not None:
        columns[POSITION_MASK_KEY] = position_mask.tolist()
    if stimulus is not None:
        columns[STIMULUS_END_KEY] = [length - 1 for length in lengths]
    dataset = Dataset.from_dict(columns)
    dataset = dataset.with_format("torch")

    return dataset


def create_prompts_data_loader(
    prompts: list[str],
    tokenizer_name: str,
    max_seq_len: int,
    loss_positions: LossPositions,
    batch_size: int,
    stimulus: StimulusLens | None,
    dist_state: DistributedState | None = None,
    seed: int = 0,
) -> tuple[DataLoader[Any] | StaticBatchLoader, PreTrainedTokenizer]:
    """Create a DataLoader from a list of prompts.

    With a position mask (`loss_positions` other than `all`), every batch is trimmed to its last
    loss position (`trim_to_loss_positions`), so short batches don't pay for padding up to
    max_seq_len. With `all`, padding positions are trained on, so batches keep the full width.

    Args:
        prompts: The prompts, one sequence each
        tokenizer_name: HuggingFace tokenizer name/path
        max_seq_len: Maximum sequence length
        loss_positions: Which positions contribute to losses (sets the 'position_mask' column)
        batch_size: Batch size for the DataLoader
        stimulus: Stimulus appended to every prompt (see load_prompts_dataset), or None
        dist_state: Distributed state for multi-GPU training
        seed: Random seed for shuffling

    Returns:
        Tuple of (DataLoader or StaticBatchLoader, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = load_prompts_dataset(prompts, tokenizer, max_seq_len, loss_positions, stimulus)

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
        if POSITION_MASK_KEY in batch:
            batch = trim_to_loss_positions(batch)
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
        collate_fn=_collate_trimmed if POSITION_MASK_KEY in dataset.column_names else None,
    )

    return loader, tokenizer
