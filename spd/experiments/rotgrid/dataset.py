"""Uniform random walks over the rotating grid-world."""

import torch
from jaxtyping import Bool, Int
from torch import Tensor
from torch.utils.data import Dataset

from spd.experiments.rotgrid import game


class RotGridDataset(
    Dataset[tuple[Int[Tensor, "batch seq_len"], Bool[Tensor, "batch seq_len n_tokens"]],]
):
    """Sequences of runs across the rotating grid-world, each token drawn uniformly at random
    from the tokens that are legal at that point.

    Every sequence starts at position 0 with `new`, on the board in its unrotated orientation.

    Alongside the tokens, `generate_batch` returns the set of tokens that were legal at each
    position. Because the true next-token distribution is uniform over that set, it is the exact
    distribution the model is trying to predict, which makes evaluation exact rather than sampled.
    """

    def __init__(self, seq_len: int, device: str):
        self.seq_len = seq_len
        self.device = device
        self.legal_mask = game.LEGAL_MASK.to(device)
        self.next_state = game.NEXT_STATE.to(device)

    def __len__(self) -> int:
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Int[Tensor, "batch seq_len"], Bool[Tensor, "batch seq_len n_tokens"]]:
        """Roll out `batch_size` sequences in lockstep.

        The whole batch advances together by indexing the automaton tables with a vector of states,
        so the only Python-level loop is over `seq_len` and the batch size is nearly free.
        """
        tokens = torch.empty(batch_size, self.seq_len, dtype=torch.long, device=self.device)
        legal_at = torch.empty(
            batch_size, self.seq_len, game.N_TOKENS, dtype=torch.bool, device=self.device
        )

        state = torch.full((batch_size,), game.INITIAL_STATE, dtype=torch.long, device=self.device)
        tokens[:, 0] = game.NEW
        legal_at[:, 0] = self.legal_mask[state]

        keys = torch.rand(batch_size, self.seq_len, game.N_TOKENS, device=self.device)
        any_illegal = torch.zeros((), dtype=torch.bool, device=self.device)

        for position in range(1, self.seq_len):
            # Uniform over the legal tokens: random keys everywhere, illegal ones pushed below all
            # legal ones, take the argmax.
            action = keys[:, position].masked_fill(~legal_at[:, position - 1], -1.0).argmax(dim=-1)
            tokens[:, position] = action

            state = self.next_state[state, action]
            any_illegal |= (state == game.ILLEGAL_TRANSITION).any()
            legal_at[:, position] = self.legal_mask[state]

        assert not any_illegal.item(), "sampled an illegal transition"
        return tokens, legal_at
