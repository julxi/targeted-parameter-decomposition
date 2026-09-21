"""Run SPD on a RotGridTransformer."""

from pathlib import Path

import fire

from spd.configs import (
    Config,
    ImportanceMinimalityLossConfig,
    LayerwiseCiConfig,
    RotGridTaskConfig,
)
from spd.experiments.rotgrid.dataset import RotGridDataset, RotGridRolloutLoader
from spd.experiments.rotgrid.model import RotGridTargetRunInfo, RotGridTransformer
from spd.log import logger
from spd.run_spd import run_experiment
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import parse_config, parse_sweep_params


def build_run_name(config: Config, n_layers: int) -> str:
    """Name the run after the settings that vary between rotgrid decompositions.

    Deriving the name from the config keeps it from drifting out of step with the hyperparameters
    it describes.
    """
    ci_config = config.ci_config
    assert isinstance(ci_config, LayerwiseCiConfig)

    cs = {info.C for info in config.module_info}
    assert len(cs) == 1, f"Expected one C across module patterns, got {cs}"

    impmin_configs = [
        loss_config
        for loss_config in config.loss_metric_configs
        if isinstance(loss_config, ImportanceMinimalityLossConfig)
    ]
    assert len(impmin_configs) == 1, f"Expected one ImpMin loss, got {len(impmin_configs)}"
    impmin_coeff = impmin_configs[0].coeff
    assert impmin_coeff is not None

    hidden_dims = "-".join(str(dim) for dim in ci_config.hidden_dims)
    return (
        f"rotgrid_{n_layers}L"
        f"_C{cs.pop()}"
        f"_{ci_config.fn_type}{hidden_dims}"
        f"_impmin{impmin_coeff:g}"
        f"_seed{config.seed}"
    )


def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    config = parse_config(config_path, config_json)

    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(config.seed)

    task_config = config.task_config
    assert isinstance(task_config, RotGridTaskConfig)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = RotGridTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = RotGridTransformer.from_run_info(target_run_info)
    target_model = target_model.to(device)
    target_model.eval()

    if config.wandb_run_name is None:
        config = config.model_copy(
            update={"wandb_run_name": build_run_name(config, target_model.config.n_layers)}
        )
    logger.info(f"WandB run name: {config.wandb_run_name}")

    dataset = RotGridDataset(seq_len=target_model.config.seq_len, device=device)
    steps_per_rollout = task_config.steps_per_rollout
    train_loader = RotGridRolloutLoader(
        dataset, batch_size=config.batch_size, steps_per_rollout=steps_per_rollout
    )
    eval_loader = RotGridRolloutLoader(
        dataset, batch_size=config.eval_batch_size, steps_per_rollout=steps_per_rollout
    )

    run_experiment(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_tag="rotgrid",
        run_id=run_id,
        launch_id=launch_id,
        evals_id=evals_id,
        sweep_params=parse_sweep_params(sweep_params_json),
        target_model_train_config=target_run_info.config,
    )


if __name__ == "__main__":
    fire.Fire(main)
