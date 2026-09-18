from typing import Literal

from pydantic import PositiveFloat, PositiveInt

from spd.base_config import BaseConfig


class RotGridModelConfig(BaseConfig):
    seq_len: PositiveInt
    d_model: PositiveInt
    n_heads: PositiveInt
    n_layers: PositiveInt
    ff_fanout: PositiveInt
    use_ff: bool
    use_pos_encoding: bool
    use_layer_norm: bool
    device: str = "cpu"


class RotGridTrainConfig(BaseConfig):
    wandb_project: str | None = None
    rotgrid_model_config: RotGridModelConfig
    steps: PositiveInt
    batch_size: PositiveInt
    lr: float
    lr_warmup: int | float
    weight_decay: float
    grad_clip: PositiveFloat
    lr_schedule: Literal["cosine", "constant", "linear"] = "linear"
    seed: int = 0
    eval_freq: PositiveInt
    eval_batch_size: PositiveInt
    attention_maps_n_steps: PositiveInt
