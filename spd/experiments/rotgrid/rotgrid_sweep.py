"""Run a grid of RotGrid SPD decompositions back to back on a single GPU.

`spd-run --sweep` covers this on SLURM, but RotGrid is not in the experiment registry and a rented
single-GPU box has no scheduler, so the grid is run sequentially here instead.
"""

import json
import subprocess
import sys
from pathlib import Path

import fire
import yaml

from spd.configs import Config
from spd.log import logger
from spd.settings import REPO_ROOT
from spd.utils.run_utils import apply_nested_updates, generate_grid_combinations

DECOMPOSITION_SCRIPT = REPO_ROOT / "spd/experiments/rotgrid/rotgrid_decomposition.py"


def build_sweep_configs(
    config_path: Path | str, sweep_path: Path | str
) -> list[tuple[Config, dict[str, object]]]:
    """Expand a sweep file into one validated Config per point of the grid.

    Every Config is built up front so that a malformed sweep file fails before any GPU time is
    spent, rather than part-way through the grid.
    """
    with open(sweep_path) as f:
        sweep_params = yaml.safe_load(f)

    combinations = generate_grid_combinations(sweep_params)
    base_config_dict = Config.from_file(config_path).model_dump(mode="json")
    return [
        (Config(**apply_nested_updates(dict(base_config_dict), combination)), combination)
        for combination in combinations
    ]


def main(config_path: Path | str, sweep_path: Path | str) -> None:
    sweep_configs = build_sweep_configs(config_path=config_path, sweep_path=sweep_path)

    logger.info(f"Sweep over {len(sweep_configs)} runs:")
    for _, combination in sweep_configs:
        logger.info(f"  {combination}")

    for i, (config, combination) in enumerate(sweep_configs, start=1):
        logger.info(f"--- run {i}/{len(sweep_configs)}: {combination} ---")
        # The `json:` tag keeps fire from parsing the JSON into a dict before the script sees it.
        config_json = f"json:{json.dumps(config.model_dump(mode='json'))}"
        sweep_params_json = f"json:{json.dumps(combination)}"
        subprocess.run(
            [
                sys.executable,
                str(DECOMPOSITION_SCRIPT),
                f"--config_json={config_json}",
                f"--sweep_params_json={sweep_params_json}",
            ],
            check=True,
        )


if __name__ == "__main__":
    fire.Fire(main)
