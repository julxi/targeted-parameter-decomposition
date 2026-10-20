"""Every metric config class has a WandB short name: init_wandb looks each one up when it flattens
the config (wandb_utils.flatten_metric_configs), so a missing entry crashes a run only at startup
on the GPU instance."""

from typing import get_args

from spd.configs import MetricConfigType
from spd.utils.wandb_utils import METRIC_CONFIG_SHORT_NAMES


def test_every_metric_config_has_a_short_name() -> None:
    classnames = [cls.model_fields["classname"].default for cls in get_args(MetricConfigType)]
    assert len(classnames) > 20, classnames
    missing = [c for c in classnames if c not in METRIC_CONFIG_SHORT_NAMES]
    assert not missing, f"add to METRIC_CONFIG_SHORT_NAMES: {missing}"
