"""Language Model decomposition script."""

from pathlib import Path

import fire

from spd.configs import (
    AllPositions,
    LMTaskConfig,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    RepeatAcrossBatchScope,
)
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.prepared_datasets import load_prepared_datasets
from spd.experiments.lm.prompts_dataset import (
    PositionMaskedLoader,
    create_prompts_data_loader,
    read_prompts_file,
)
from spd.log import logger
from spd.pretrain.run_info import PretrainRunInfo
from spd.run_spd import LoaderType, run_experiment
from spd.utils.distributed_utils import (
    DistributedState,
    ensure_cached_and_call,
    get_device,
    init_distributed,
    is_main_process,
    with_distributed_cleanup,
)
from spd.utils.general_utils import resolve_class, set_seed
from spd.utils.run_utils import parse_config, parse_sweep_params


def _create_lm_loaders(
    task_config: LMTaskConfig,
    tokenizer_name: str | None,
    train_batch_size: int,
    eval_batch_size: int,
    seed: int,
    dist_state: DistributedState | None,
) -> tuple[LoaderType, LoaderType]:
    """Create train and eval loaders from an LMTaskConfig.

    Supports dataset-based loading (dataset_name), prompts-file-based loading (prompts_file; train
    and eval use the same prompts) and prepared datasets under data/ (prepared_datasets; train on
    their train splits, eval on their held-out test splits).
    """
    if task_config.prompts_file is not None or task_config.prepared_datasets is not None:
        assert tokenizer_name is not None
        if task_config.prepared_datasets is not None:
            train_prompts = load_prepared_datasets(task_config.prepared_datasets, "train")
            eval_prompts = load_prepared_datasets(task_config.prepared_datasets, "test")
        else:
            assert task_config.prompts_file is not None
            train_prompts = eval_prompts = read_prompts_file(Path(task_config.prompts_file))
        train_loader, _ = create_prompts_data_loader(
            prompts=train_prompts,
            tokenizer_name=tokenizer_name,
            max_seq_len=task_config.max_seq_len,
            loss_positions=task_config.loss_positions,
            batch_size=train_batch_size,
            dist_state=dist_state,
            seed=seed,
        )
        eval_loader, _ = create_prompts_data_loader(
            prompts=eval_prompts,
            tokenizer_name=tokenizer_name,
            max_seq_len=task_config.max_seq_len,
            loss_positions=task_config.loss_positions,
            batch_size=eval_batch_size,
            dist_state=dist_state,
            seed=seed + 1,
        )
        return train_loader, eval_loader

    assert task_config.dataset_name is not None, (
        "One of dataset_name, prompts_file and prepared_datasets must be set in LMTaskConfig"
    )
    train_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=tokenizer_name,
        split=task_config.train_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=task_config.shuffle_each_epoch,
        seed=task_config.dataset_seed,
    )
    train_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=train_batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=seed,
        dist_state=dist_state,
    )

    eval_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=task_config.shuffle_each_epoch,
        seed=task_config.dataset_seed,
    )
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=eval_batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=seed + 1,
        dist_state=dist_state,
    )
    if isinstance(task_config.loss_positions, AllPositions):
        return train_loader, eval_loader
    return (
        PositionMaskedLoader(train_loader, task_config.loss_positions),
        PositionMaskedLoader(eval_loader, task_config.loss_positions),
    )


@with_distributed_cleanup
def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    config = parse_config(config_path, config_json)

    dist_state = init_distributed()
    logger.info(f"Distributed state: {dist_state}")

    # Use the same seed across all ranks for deterministic data loading
    set_seed(config.seed)

    device = get_device()
    assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"

    pretrained_model_class = resolve_class(config.pretrained_model_class)
    assert hasattr(pretrained_model_class, "from_pretrained"), (
        f"Model class {pretrained_model_class} should have a `from_pretrained` method"
    )
    assert config.pretrained_model_name is not None

    if config.pretrained_model_class.startswith("spd.pretrain"):
        # Ensure local_rank 0 on each node caches the model, then all ranks load from local cache
        # (In multi-node setups, /tmp is node-local so we can't broadcast paths across nodes)
        run_info = ensure_cached_and_call(PretrainRunInfo.from_path, config.pretrained_model_name)

        # Handle old training runs not having a model_type in the model_config_dict
        if "model_type" not in run_info.model_config_dict:
            run_info.model_config_dict["model_type"] = config.pretrained_model_class.split(".")[-1]

        assert hasattr(pretrained_model_class, "from_run_info")
        # Just loads from local file
        target_model = pretrained_model_class.from_run_info(run_info)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        # Avoid concurrent wandb API requests by first calling from_pretrained on rank 0 only
        target_model = ensure_cached_and_call(
            pretrained_model_class.from_pretrained,  # pyright: ignore[reportAttributeAccessIssue]
            config.pretrained_model_name,
        )
    target_model.eval()

    # --- Load Data --- #
    if is_main_process():
        logger.info("Loading dataset...")

    world_size = dist_state.world_size if dist_state is not None else 1

    def _rank_batch_size(total_batch_size: int) -> int:
        assert total_batch_size % world_size == 0 and total_batch_size > 0, (
            f"Batch size {total_batch_size} is not divisible by world size {world_size}."
        )
        return total_batch_size // world_size

    train_rank_batch_size = _rank_batch_size(config.batch_size)

    for cfg in config.loss_metric_configs:
        if isinstance(
            cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
        ) and isinstance(cfg.scope, RepeatAcrossBatchScope):
            n = cfg.scope.n_sources
            assert train_rank_batch_size % n == 0, (
                f"repeat_across_batch n_sources={n} must divide per-rank batch_size="
                f"{train_rank_batch_size}"
            )

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)

    train_loader, eval_loader = _create_lm_loaders(
        task_config=task_config,
        tokenizer_name=config.tokenizer_name,
        train_batch_size=train_rank_batch_size,
        eval_batch_size=_rank_batch_size(config.eval_batch_size),
        seed=config.seed,
        dist_state=dist_state,
    )

    # --- Nontarget data --- #
    nontarget_train_loader: LoaderType | None = None
    nontarget_eval_loader: LoaderType | None = None
    if config.nontarget_task_config is not None:
        assert isinstance(config.nontarget_task_config, LMTaskConfig)
        assert config.nontarget_batch_size is not None
        assert config.nontarget_eval_batch_size is not None
        nontarget_train_loader, nontarget_eval_loader = _create_lm_loaders(
            task_config=config.nontarget_task_config,
            tokenizer_name=config.tokenizer_name,
            train_batch_size=_rank_batch_size(config.nontarget_batch_size),
            eval_batch_size=_rank_batch_size(config.nontarget_eval_batch_size),
            seed=config.seed + 2,
            dist_state=dist_state,
        )

    run_experiment(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_tag="lm",
        run_id=run_id,
        launch_id=launch_id,
        evals_id=evals_id,
        sweep_params=parse_sweep_params(sweep_params_json),
        nontarget_train_loader=nontarget_train_loader,
        nontarget_eval_loader=nontarget_eval_loader,
    )


if __name__ == "__main__":
    fire.Fire(main)
