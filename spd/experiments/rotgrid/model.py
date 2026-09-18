from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

# The attention, MLP and positional-encoding modules are generic; only `InductionTransformer`
# itself is specific to the induction-heads task.
from spd.experiments.ih.model import MultiHeadSelfAttention, PositionalEncoding, TransformerBlock
from spd.experiments.rotgrid import game
from spd.experiments.rotgrid.configs import RotGridModelConfig, RotGridTrainConfig
from spd.interfaces import LoadableModule, RunInfo
from spd.spd_types import ModelPath


@dataclass
class RotGridTargetRunInfo(RunInfo[RotGridTrainConfig]):
    """Run info from training a RotGridTransformer."""

    config_class = RotGridTrainConfig
    config_filename = "rotgrid_train_config.yaml"
    checkpoint_filename = "rotgrid.pth"


class RotGridTransformer(LoadableModule):
    """Next-token predictor over the rotating grid-world's five-token alphabet."""

    def __init__(self, cfg: RotGridModelConfig):
        super().__init__()
        self.config = cfg

        self.token_embed = nn.Embedding(game.N_TOKENS, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.seq_len)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ff_fanout=cfg.ff_fanout,
                    use_ff=cfg.use_ff,
                    use_pos_encoding=cfg.use_pos_encoding,
                    use_layer_norm=cfg.use_layer_norm,
                    max_len=cfg.seq_len,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        if cfg.use_layer_norm:
            self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, game.N_TOKENS, bias=False)

    @override
    def forward(
        self, tokens: Int[Tensor, "batch seq_len"], **_: object
    ) -> Float[Tensor, "batch seq_len n_tokens"]:
        x = self.token_embed(tokens)

        for block in self.blocks:
            x = block(x)

        if self.config.use_layer_norm:
            x = self.ln_f(x)
        return self.unembed(x)

    def get_attention_weights(
        self, tokens: Int[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch n_layers n_heads seq_len seq_len"]:
        x = self.token_embed(tokens)

        attn_weights = []
        for block in self.blocks:
            assert isinstance(block.attn, MultiHeadSelfAttention)
            attn_weights.append(block.attn.get_attention_weights(x))
            x = block(x)

        return torch.stack(attn_weights, dim=1)

    @classmethod
    @override
    def from_run_info(cls, run_info: RunInfo[RotGridTrainConfig]) -> "RotGridTransformer":
        """Load a pretrained model from a run info object."""
        model = cls(cfg=run_info.config.rotgrid_model_config)
        model.load_state_dict(
            torch.load(run_info.checkpoint_path, weights_only=True, map_location="cpu")
        )
        return model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "RotGridTransformer":
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        run_info = RotGridTargetRunInfo.from_path(path)
        return cls.from_run_info(run_info)
