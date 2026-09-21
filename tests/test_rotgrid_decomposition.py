from pathlib import Path

import pytest

from spd.configs import (
    CI_L0Config,
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    Config,
    ImportanceMinimalityLossConfig,
    LayerwiseCiConfig,
    ModulePatternInfoConfig,
    RotGridTaskConfig,
    ScheduleConfig,
    StochasticHiddenActsReconLossConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
)
from spd.experiments.rotgrid.configs import RotGridModelConfig
from spd.experiments.rotgrid.dataset import RotGridDataset, RotGridRolloutLoader
from spd.experiments.rotgrid.model import RotGridTransformer
from spd.experiments.rotgrid.rotgrid_decomposition import build_run_name
from spd.experiments.rotgrid.rotgrid_sweep import build_sweep_configs
from spd.run_spd import optimize
from spd.utils.general_utils import set_seed


@pytest.mark.slow
def test_rotgrid_decomposition_happy_path(tmp_path: Path) -> None:
    """SPD decomposition of a RotGridTransformer's attention out_proj modules."""
    set_seed(0)
    device = "cpu"

    rotgrid_model_config = RotGridModelConfig(
        seq_len=32,
        d_model=16,
        n_heads=2,
        n_layers=2,
        ff_fanout=4,
        use_ff=True,
        use_pos_encoding=True,
        use_layer_norm=False,
    )

    config = Config(
        wandb_project=None,
        wandb_run_name=None,
        wandb_run_name_prefix="",
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="vector_mlp", hidden_dims=[8]),
        sigmoid_type="leaky_hard",
        module_info=[ModulePatternInfoConfig(module_pattern="blocks.*.attn.out_proj", C=8)],
        identity_module_info=None,
        use_delta_component=True,
        loss_metric_configs=[
            ImportanceMinimalityLossConfig(coeff=1e-2, pnorm=1.0, beta=0.0),
            StochasticReconLossConfig(coeff=1.0),
            StochasticReconLayerwiseLossConfig(coeff=1.0),
        ],
        output_loss_type="kl",
        lr_schedule=ScheduleConfig(
            start_val=1e-3, fn_type="cosine", warmup_pct=0.0, final_val_frac=0.0
        ),
        steps=2,
        batch_size=2,
        faithfulness_warmup_steps=1,
        faithfulness_warmup_lr=0.01,
        faithfulness_warmup_weight_decay=0.1,
        train_log_freq=1,
        eval_freq=1,
        eval_batch_size=2,
        slow_eval_freq=1,
        n_eval_steps=1,
        slow_eval_on_first_step=True,
        save_freq=None,
        ci_alive_threshold=0.1,
        eval_metric_configs=[
            CI_L0Config(groups=None),
            CIHistogramsConfig(n_batches_accum=1),
            ComponentActivationDensityConfig(),
            CIMeanPerComponentConfig(),
            StochasticHiddenActsReconLossConfig(),
        ],
        pretrained_model_class="spd.experiments.rotgrid.model.RotGridTransformer",
        pretrained_model_path=None,
        pretrained_model_name=None,
        pretrained_model_output_attr=None,
        tokenizer_name=None,
        task_config=RotGridTaskConfig(task_name="rotgrid", steps_per_rollout=4),
    )

    target_model = RotGridTransformer(rotgrid_model_config).to(device)
    target_model.eval()
    target_model.requires_grad_(False)

    dataset = RotGridDataset(seq_len=rotgrid_model_config.seq_len, device=device)
    assert isinstance(config.task_config, RotGridTaskConfig)
    steps_per_rollout = config.task_config.steps_per_rollout
    train_loader = RotGridRolloutLoader(
        dataset, batch_size=config.batch_size, steps_per_rollout=steps_per_rollout
    )
    eval_loader = RotGridRolloutLoader(
        dataset, batch_size=config.eval_batch_size, steps_per_rollout=steps_per_rollout
    )

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=tmp_path,
    )


def test_build_run_name_tracks_the_hyperparameters_it_describes():
    """The name is derived so it cannot drift out of step with the config."""
    config = Config.from_file("spd/experiments/rotgrid/rotgrid_config.yaml")
    assert build_run_name(config, n_layers=3).startswith("rotgrid_3L_")

    impmin_updated = [
        loss_config.model_copy(update={"coeff": 0.03})
        if isinstance(loss_config, ImportanceMinimalityLossConfig)
        else loss_config
        for loss_config in config.loss_metric_configs
    ]
    swept = config.model_copy(update={"loss_metric_configs": impmin_updated, "seed": 1})
    assert build_run_name(swept, n_layers=3).endswith("_impmin0.03_seed1")


def test_an_explicit_run_name_is_left_alone():
    """A name set in the config wins, so `spd-run --sweep` names survive."""
    config = Config.from_file("spd/experiments/rotgrid/rotgrid_config.yaml")
    assert config.wandb_run_name is None

    named = config.model_copy(update={"wandb_run_name": "my-explicit-name"})
    assert named.wandb_run_name == "my-explicit-name"
    assert build_run_name(named, n_layers=3) != named.wandb_run_name


def test_the_checked_in_sweep_expands_to_distinct_runs():
    """Every grid point must reach the decomposition as a valid Config with its own name."""
    sweep_configs = build_sweep_configs(
        config_path="spd/experiments/rotgrid/rotgrid_config.yaml",
        sweep_path="spd/experiments/rotgrid/rotgrid_sweep_params.yaml",
    )
    assert len(sweep_configs) == 4

    coeffs = []
    for config, _ in sweep_configs:
        impmin = [
            loss_config
            for loss_config in config.loss_metric_configs
            if isinstance(loss_config, ImportanceMinimalityLossConfig)
        ]
        coeffs.append(impmin[0].coeff)
    assert coeffs == [0.03, 0.01, 0.003, 0.001]

    names = {build_run_name(config, n_layers=3) for config, _ in sweep_configs}
    assert len(names) == len(sweep_configs)
