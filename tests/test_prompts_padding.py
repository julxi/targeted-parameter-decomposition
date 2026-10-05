"""Trimming prompt batches to their last loss position (per-batch padding) leaves the losses unchanged.

`create_prompts_data_loader` cuts every batch after its last loss position instead of padding it to
max_seq_len (`trim_to_loss_positions`). This is only a speedup if no loss reads the dropped
positions: masked losses select by `position_mask`, unselected positions run on the original
weights, and right padding cannot reach earlier positions through causal attention. The test runs
the deterministic training losses of the tiu/uth arm configs on a tiny Qwen2 decomposition, once
on a batch padded to max_seq_len and once on the trimmed batch.

The stochastic losses (e.g. StochasticReconSubsetLoss) are not compared: their random masks are
drawn with the batch's shape, so a narrower batch gets different mask values at the same positions.
They share the position routing and masked-loss code with the persistent PGD loss and
UnmaskedReconLoss, which are compared.
"""

import torch
import yaml
from transformers import AutoModelForCausalLM

from spd.configs import (
    GlobalCiConfig,
    ImportanceMinimalityLossConfig,
    LastKPositions,
    LossMetricConfigType,
    PersistentPGDReconLossConfig,
    TokenPositions,
    UnmaskedReconLossConfig,
)
from spd.experiments.lm.prompts_dataset import build_position_mask, trim_to_loss_positions
from spd.losses import compute_losses
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.persistent_pgd import PersistentPGDState
from spd.settings import REPO_ROOT
from spd.utils.general_utils import POSITION_MASK_KEY
from spd.utils.module_utils import ModulePathInfo

TINY_QWEN = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
ARM_A_CONFIG = (
    REPO_ROOT / "spd/experiments/lm/honesty_targeted_decomposition/config_truth_all_tokens.yaml"
)
MAX_SEQ_LEN = 20
PAD_ID = 0


def _padded_batch(
    lengths: list[int], loss_positions: TokenPositions | LastKPositions
) -> dict[str, torch.Tensor]:
    """Random token ids right-padded to MAX_SEQ_LEN, as `load_prompts_dataset` builds them."""
    gen = torch.Generator().manual_seed(0)
    ids = torch.full((len(lengths), MAX_SEQ_LEN), PAD_ID)
    for row, n in enumerate(lengths):
        ids[row, :n] = torch.randint(1, 1000, (n,), generator=gen)
    mask = build_position_mask(torch.tensor(lengths), MAX_SEQ_LEN, loss_positions)
    assert mask is not None
    return {"input_ids": ids, POSITION_MASK_KEY: mask}


def test_trim_cuts_after_the_last_loss_position() -> None:
    for loss_positions in (TokenPositions(type="tokens"), LastKPositions(type="last_k", k=2)):
        full = _padded_batch([5, 12, 9], loss_positions)
        trimmed = trim_to_loss_positions(full)
        assert trimmed["input_ids"].shape == (3, 12)
        assert torch.equal(trimmed["input_ids"], full["input_ids"][:, :12])
        assert torch.equal(trimmed[POSITION_MASK_KEY], full[POSITION_MASK_KEY][:, :12])
        assert not full[POSITION_MASK_KEY][:, 12:].any()


def test_losses_unchanged_by_trimming() -> None:
    target = AutoModelForCausalLM.from_pretrained(TINY_QWEN, torch_dtype=torch.float32).eval()
    target.requires_grad_(False)
    torch.manual_seed(0)
    model = ComponentModel(
        target_model=target,
        module_path_info=[
            ModulePathInfo(module_path=f"model.layers.{i}.mlp.down_proj", C=8) for i in (0, 1)
        ],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        sigmoid_type="leaky_hard",
        pretrained_model_output_attr="logits",
    )
    arm_a = {
        c["classname"]: c for c in yaml.safe_load(ARM_A_CONFIG.read_text())["loss_metric_configs"]
    }
    ppgd_cfg = PersistentPGDReconLossConfig(**arm_a["PersistentPGDReconLoss"])
    loss_cfgs: list[LossMetricConfigType] = [
        ImportanceMinimalityLossConfig(**arm_a["ImportanceMinimalityLoss"]),
        ppgd_cfg,
        UnmaskedReconLossConfig(coeff=1.0),
    ]
    full = _padded_batch([5, 12, 9, 3], TokenPositions(type="tokens"))
    trimmed = trim_to_loss_positions(full)
    assert trimmed["input_ids"].shape == (4, 12)

    # Sources sized for max_seq_len, as run_spd.py creates them for LM tasks; the trimmed batch
    # uses their first 12 positions.
    ppgd_state = PersistentPGDState(
        module_to_c=model.module_to_c,
        batch_dims=(4, MAX_SEQ_LEN),
        device="cpu",
        use_delta_component=True,
        cfg=ppgd_cfg,
        output_loss_type="kl",
    )
    weight_deltas = model.calc_weight_deltas()
    losses = {}
    for name, batch in (("full", full), ("trimmed", trimmed)):
        out = model(batch["input_ids"], cache_type="input")
        assert isinstance(out, OutputWithCache)
        ci = model.calc_causal_importances(pre_weight_acts=out.cache, sampling="continuous")
        losses[name] = compute_losses(
            loss_metric_configs=loss_cfgs,
            model=model,
            batch=batch["input_ids"],
            ci=ci,
            target_out=out.output,
            weight_deltas=weight_deltas,
            current_frac_of_training=0.9,  # after the PPGD start (start_frac 0.8)
            sampling="continuous",
            use_delta_component=True,
            n_mask_samples=1,
            ppgd_states=dict([(ppgd_cfg, ppgd_state)]),
            output_loss_type="kl",
            position_mask=batch[POSITION_MASK_KEY],
        )
    for cfg in loss_cfgs:
        assert losses["full"][cfg] > 0, cfg.classname
        torch.testing.assert_close(losses["trimmed"][cfg], losses["full"][cfg], msg=cfg.classname)
