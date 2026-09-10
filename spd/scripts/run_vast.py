"""SPD experiment runner for rented vast.ai GPUs.

vast.ai is a GPU marketplace, not a job service: this script searches offers, rents a machine,
writes an ssh config stanza for it, rsyncs the working tree over, and starts training. Checkpoints
are synced to WandB because the machine (and its disk) disappear when the instance is destroyed.
"""

import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from netrc import netrc
from pathlib import Path
from typing import Any, Literal

import fire
from dotenv import dotenv_values
from pydantic import Field

from spd.base_config import BaseConfig
from spd.configs import Config
from spd.log import logger
from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import DEFAULT_PROJECT_NAME, REPO_ROOT
from spd.utils.compute_utils import TrainingJob, get_command
from spd.utils.run_utils import generate_run_id
from spd.utils.wandb_utils import get_wandb_entity, get_wandb_run_url

Mode = Literal["run", "detached", "provision"]

DEFAULT_VAST_CONFIG = "vast_config.yaml"
SSH_HOST_ALIAS = "vastai"
SSH_CONFIG_PATH = Path.home() / ".ssh/config.d/vastai.conf"
SSH_IDENTITY_FILE = Path.home() / ".ssh/id_rsa"
REMOTE_REPO_DIR = "/root/spd"
REMOTE_OUT_DIR = "/root/spd_out"
REMOTE_LOG = f"{REMOTE_REPO_DIR}/train.log"
READY_TIMEOUT_S = 900
POLL_INTERVAL_S = 10


class VastConfig(BaseConfig):
    """Machine selection and rental settings, loaded from a git-tracked YAML file."""

    gpu_name: str = Field(
        description="vast.ai GPU name with spaces as underscores (e.g. 'RTX_4090', 'A100_SXM4')"
    )
    max_price: float = Field(description="Maximum total price in $/hour, including storage")
    min_gpu_ram: int = Field(
        description="Minimum GPU memory in GB (vast.ai filters this field in GB but reports it "
        "in MB)"
    )
    disk: int = Field(
        description="Disk to rent in GB. Machines with less free disk than this are filtered out, "
        "and offer prices are quoted including this much storage"
    )
    image: str = Field(description="Docker image to launch")
    min_reliability: float = Field(
        description="Minimum host reliability. Low-reliability hosts drop runs mid-training"
    )
    min_cuda: float = Field(
        description="Minimum CUDA version the host supports, matched to the torch wheels in uv.lock"
    )
    sort: str = Field(
        description="vast.ai sort order; the first offer is rented (e.g. 'dlperf_per_dphtotal-' "
        "for best performance per dollar, 'dph_total' for cheapest)"
    )
    n_offers: int = Field(description="How many offers to fetch and show")


def main(
    experiment: str | None = None,
    config: str = DEFAULT_VAST_CONFIG,
    mode: Mode = "run",
    gpu_name: str | None = None,
    max_price: float | None = None,
    min_gpu_ram: int | None = None,
    disk: int | None = None,
    image: str | None = None,
    offer_id: int | None = None,
    list_offers: bool = False,
    destroy_on_exit: bool = False,
    project: str = DEFAULT_PROJECT_NAME,
    dry_run: bool = False,
) -> None:
    """Run a single SPD experiment on a rented vast.ai GPU.

    Machine selection comes from a git-tracked YAML config (`spd/scripts/vast_config.yaml` by
    default). The flags below override individual config fields for one-off launches.

    Args:
        experiment: Experiment name from registry (e.g. 'tms_5-2'). Not needed with --list_offers.
        config: Machine config: a short name ('h100' -> spd/scripts/vast_h100_config.yaml), a
            filename in spd/scripts, or a path.
        mode: 'run' streams training until it exits, 'detached' starts it and returns,
            'provision' only rents + syncs and leaves the run for you to start over ssh.
        gpu_name: Override the config's gpu_name.
        max_price: Override the config's max_price.
        min_gpu_ram: Override the config's min_gpu_ram.
        disk: Override the config's disk.
        image: Override the config's image.
        offer_id: Rent this specific offer instead of searching.
        list_offers: Print matching offers and exit without renting anything.
        destroy_on_exit: Destroy the instance once training finishes (only for mode='run').
        project: WandB project name.
        dry_run: Print the commands that would be run, rent nothing.

    Examples:
        spd-vast --list_offers                          # browse offers, rent nothing
        spd-vast tms_5-2                                # rent, sync, train, stream logs
        spd-vast tms_5-2 --config h100                  # spd/scripts/vast_h100_config.yaml
        spd-vast tms_5-2 --gpu_name A100_SXM4           # one-off override
        spd-vast tms_5-2 --mode provision               # rent + sync, then `ssh vastai`
        spd-vast tms_5-2 --destroy_on_exit              # stop paying when training ends
    """
    assert shutil.which("vastai"), "vastai CLI not found. Install it with `pip install vastai`."

    vast_config = _load_vast_config(
        config,
        overrides={
            "gpu_name": gpu_name,
            "max_price": max_price,
            "min_gpu_ram": min_gpu_ram,
            "disk": disk,
            "image": image,
        },
    )

    if list_offers:
        _print_offers(_search_offers(vast_config), vast_config.sort, vast_config.disk)
        return

    assert experiment is not None, "experiment is required unless --list_offers is passed"
    assert experiment in EXPERIMENT_REGISTRY, (
        f"Unknown experiment '{experiment}'. Available: {', '.join(sorted(EXPERIMENT_REGISTRY))}"
    )
    assert not (destroy_on_exit and mode != "run"), "destroy_on_exit requires mode='run'"

    exp_config = EXPERIMENT_REGISTRY[experiment]
    run_id = generate_run_id("spd")
    launch_id = f"vast-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_url = get_wandb_run_url(project, run_id)

    job = TrainingJob(
        experiment=experiment,
        script_path=exp_config.decomp_script,
        config=_build_config(REPO_ROOT / exp_config.config_path, project),
        run_id=run_id,
    )
    train_command = get_command(
        launch_id=launch_id,
        job=job,
        job_idx=0,
        n_gpus=1,
        sweep_params=None,
        snapshot_branch="",
        is_array=False,
    ).command
    remote_script = _build_remote_script(train_command)

    if offer_id is None:
        offers = _search_offers(vast_config)
        offer = offers[0]
        offer_id = int(offer["id"])
        logger.info(
            f"Selected offer {offer_id}: {offer['gpu_name']} at ${offer['dph_total']:.3f}/hr "
            f"in {offer['geolocation']} (reliability {offer['reliability']:.4f})"
        )

    env_vars = _forwarded_env_vars()
    create_args = _create_instance_args(offer_id, vast_config, experiment, run_id, env_vars)

    if dry_run:
        redacted_args = _create_instance_args(
            offer_id, vast_config, experiment, run_id, dict.fromkeys(env_vars, "<redacted>")
        )
        logger.section("Dry run - nothing rented")
        logger.values(
            {
                "create": shlex.join(["vastai", *redacted_args]),
                "rsync": shlex.join(_rsync_args()),
                "remote command": remote_script,
                "wandb": wandb_url,
            }
        )
        return

    instance_id = _create_instance(create_args)
    logger.info(f"Created instance {instance_id}, waiting for it to boot")
    try:
        _wait_until_running(instance_id)
        host, port = _ssh_host_port(instance_id)
        _write_ssh_config(host, port, run_id, instance_id)
        logger.info(f"Wrote {SSH_CONFIG_PATH} - connect with `ssh {SSH_HOST_ALIAS}`")
        _rsync_repo()

        _log_summary(instance_id, run_id, wandb_url, mode)

        match mode:
            case "provision":
                logger.values(
                    {"start training with": f"ssh {SSH_HOST_ALIAS}, then: {remote_script}"}
                )
            case "detached":
                _ssh(
                    f"setsid nohup bash -lc {shlex.quote(remote_script)} "
                    f"> {REMOTE_LOG} 2>&1 < /dev/null & echo started"
                )
                logger.info(
                    f"Training started. Follow with: ssh {SSH_HOST_ALIAS} tail -f {REMOTE_LOG}"
                )
            case "run":
                exit_code = _ssh_streaming(
                    f"set -o pipefail; bash -lc {shlex.quote(remote_script)} 2>&1 "
                    f"| tee {REMOTE_LOG}"
                )
                logger.info(f"Training exited with code {exit_code}")
    finally:
        if destroy_on_exit:
            logger.info(f"Destroying instance {instance_id}")
            _vastai(["destroy", "instance", str(instance_id)], raw=False)
        else:
            logger.info(
                f"Instance still running. Destroy it with: vastai destroy instance {instance_id}"
            )


def _build_config(config_path: Path, project: str) -> Config:
    """Load the experiment config, forcing checkpoint upload since the instance is ephemeral."""
    config_dict = Config.from_file(config_path).model_dump(mode="json")
    config_dict["wandb_project"] = project
    config_dict["sync_checkpoints_to_wandb"] = True
    return Config(**config_dict)


def _build_remote_script(train_command: str) -> str:
    """Install uv, sync dependencies and run training, all inside the synced repo."""
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {REMOTE_REPO_DIR}",
            f"export SPD_OUT_DIR={REMOTE_OUT_DIR}",
            'export PATH="$HOME/.local/bin:$PATH"',
            "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh",
            'export PATH="$HOME/.local/bin:$PATH"',
            "uv sync --no-dev --link-mode copy -q",
            "source .venv/bin/activate",
            train_command,
        ]
    )


def _vastai(args: list[str], raw: bool = True) -> Any:
    """Run a vastai CLI command, parsing its JSON output when raw."""
    cmd = ["vastai", *args, "--raw"] if raw else ["vastai", *args]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout) if raw else result.stdout


def _resolve_config_path(config: str) -> Path:
    """Resolve a config given as a short name, a repo-relative path, or an absolute path.

    'h100' -> spd/scripts/vast_h100_config.yaml, 'vast_h100_config.yaml' -> spd/scripts/<it>,
    anything containing a '/' is taken as given (relative to the repo root).
    """
    if "/" in config:
        path = Path(config)
        return path if path.is_absolute() else REPO_ROOT / path
    filename = config if config.endswith((".yaml", ".yml")) else f"vast_{config}_config.yaml"
    return REPO_ROOT / "spd/scripts" / filename


def _load_vast_config(config: str, overrides: dict[str, Any]) -> VastConfig:
    """Load the machine config, applying any flags the caller passed explicitly."""
    config_path = _resolve_config_path(config)
    available = sorted(p.name for p in (REPO_ROOT / "spd/scripts").glob("vast_*config.yaml"))
    assert config_path.exists(), (
        f"vast config not found: {config_path}\nAvailable in spd/scripts: {', '.join(available)}"
    )
    logger.info(f"Machine config: {config_path.relative_to(REPO_ROOT)}")
    vast_config = VastConfig.from_file(config_path)
    given = {k: v for k, v in overrides.items() if v is not None}
    if given:
        logger.info(f"Overriding {config_path.name}: {given}")
    return vast_config.model_copy(update=given)


def _search_offers(config: VastConfig) -> list[Any]:
    """Search offers in the config's sort order; the caller rents the first one.

    `disk_space` is the machine's free disk, which must fit the disk we intend to rent.
    `direct_port_count>=1` is required for the `--direct` ssh this script relies on. `gpu_ram` is
    queried in GB even though offers report it in MB.
    """
    query = (
        f"num_gpus=1 gpu_name={config.gpu_name} rentable=true verified=true "
        f"reliability>{config.min_reliability} direct_port_count>=1 "
        f"cuda_max_good>={config.min_cuda} gpu_ram>={config.min_gpu_ram} "
        f"disk_space>={config.disk} dph_total<{config.max_price}"
    )
    logger.info(f"Searching offers: {query}")
    offers = _vastai(
        [
            "search",
            "offers",
            query,
            "-o",
            config.sort,
            "--limit",
            str(config.n_offers),
            # Price storage for the disk we actually rent; vast.ai otherwise quotes it for 5GiB
            "--storage",
            str(config.disk),
        ]
    )
    assert offers, (
        "No offers matched. Raise max_price, lower min_gpu_ram/disk, or change gpu_name - "
        "in the config or with the matching flag."
    )
    return offers


def _print_offers(offers: list[Any], sort: str, disk: int) -> None:
    """Print the offers as a table.

    `gpu_ram` and `cpu_ram` are reported in MB, disk in GB. 'free' is the machine's unallocated
    disk, not the disk we rent - `$/hr` already covers the rented disk.
    """
    logger.section(
        f"{len(offers)} matching offers, sorted by {sort}. $/hr includes the {disk}GB disk"
    )
    header = (
        f"{'id':>10}  {'gpu':<16} {'vram':>5} {'$/hr':>6} {'free':>6} {'ram':>6} {'cores':>5} "
        f"{'cuda':>5} {'net↓':>7} {'dlperf':>7} {'dlperf/$':>8} {'reliab':>7}  location"
    )
    rows = [
        f"{o['id']:>10}  {o['gpu_name']:<16} {o['gpu_ram'] / 1024:>4.0f}G "
        f"{o['dph_total']:>6.3f} {o['disk_space']:>5.0f}G {o['cpu_ram'] / 1024:>5.0f}G "
        f"{o['cpu_cores_effective']:>5.1f} {o['cuda_max_good']:>5.1f} "
        f"{o['inet_down']:>6.0f}M {o['dlperf']:>7.1f} {o['dlperf_per_dphtotal']:>8.1f} "
        f"{o['reliability']:>7.4f}  {o['geolocation']}"
        for o in offers
    ]
    logger.info("\n".join([header, *rows]))


def _forwarded_env_vars() -> dict[str, str]:
    """WandB credentials for the container.

    `.env` is gitignored and therefore not rsynced, so the remote run has no credentials unless
    they are passed as env vars. The key comes from `.env`, the environment, or `~/.netrc` (where
    `wandb login` puts it).
    """
    env_file = dotenv_values(REPO_ROOT / ".env")
    api_key = env_file.get("WANDB_API_KEY") or os.getenv("WANDB_API_KEY") or _netrc_wandb_key()
    assert api_key, (
        "No WandB API key found in .env, the environment, or ~/.netrc. The remote run cannot log "
        "without one - set WANDB_API_KEY or run `wandb login`."
    )
    return {"WANDB_API_KEY": api_key, "WANDB_ENTITY": get_wandb_entity()}


def _netrc_wandb_key() -> str | None:
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return None
    auth = netrc(netrc_path).authenticators("api.wandb.ai")
    return auth[2] if auth else None


def _create_instance_args(
    offer_id: int,
    config: VastConfig,
    experiment: str,
    run_id: str,
    env_vars: dict[str, str],
) -> list[str]:
    env_arg = " ".join(f"-e {k}={v}" for k, v in env_vars.items())
    return [
        "create",
        "instance",
        str(offer_id),
        "--image",
        config.image,
        "--disk",
        str(config.disk),
        "--ssh",
        "--direct",
        "--label",
        f"spd-{experiment}-{run_id}",
        "--cancel-unavail",
        "--env",
        env_arg,
    ]


def _create_instance(create_args: list[str]) -> int:
    result = _vastai(create_args)
    assert result["success"], f"Failed to create instance: {result}"
    return int(result["new_contract"])


def _wait_until_running(instance_id: int) -> None:
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        instance = _vastai(["show", "instance", str(instance_id)])
        status = instance["actual_status"]
        if status == "running":
            return
        logger.info(f"Instance status: {status} ({instance.get('status_msg') or 'no message'})")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(
        f"Instance {instance_id} not running after {READY_TIMEOUT_S}s. "
        f"Destroy it with: vastai destroy instance {instance_id}"
    )


def _ssh_host_port(instance_id: int) -> tuple[str, int]:
    """Parse `vastai ssh-url`, which reports the right host/port for direct or proxied ssh."""
    ssh_url = _vastai(["ssh-url", str(instance_id)], raw=False).strip()
    assert ssh_url.startswith("ssh://"), f"Unexpected ssh url: {ssh_url}"
    host_port = ssh_url.removeprefix("ssh://").split("@")[-1]
    host, port = host_port.split(":")
    return host, int(port)


def _write_ssh_config(host: str, port: int, run_id: str, instance_id: int) -> None:
    """Write the `vastai` ssh host stanza, picked up by the Include in ~/.ssh/config.

    Host keys are not checked because every rental is a fresh machine, often reusing an
    ip:port that a previous rental had.
    """
    SSH_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SSH_CONFIG_PATH.write_text(
        f"# generated by spd-vast for {run_id} (instance {instance_id}) "
        f"at {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Host {SSH_HOST_ALIAS}\n"
        f"    HostName {host}\n"
        f"    User root\n"
        f"    IdentityFile {SSH_IDENTITY_FILE}\n"
        f"    Port {port}\n"
        f"    StrictHostKeyChecking no\n"
        f"    UserKnownHostsFile /dev/null\n"
    )


def _rsync_args() -> list[str]:
    return [
        "rsync",
        "-az",
        "--delete",
        "--filter=:- .gitignore",
        "--exclude=.git",
        "--exclude=.venv",
        f"{REPO_ROOT}/",
        f"{SSH_HOST_ALIAS}:{REMOTE_REPO_DIR}/",
    ]


def _rsync_repo() -> None:
    args = _rsync_args()
    logger.info(f"Syncing working tree: {shlex.join(args)}")
    subprocess.run(args, check=True)


def _ssh(remote_command: str) -> None:
    subprocess.run(["ssh", SSH_HOST_ALIAS, remote_command], check=True)


def _ssh_streaming(remote_command: str) -> int:
    """Run a remote command with its output streamed to our stdout, returning its exit code."""
    return subprocess.run(["ssh", SSH_HOST_ALIAS, remote_command], check=False).returncode


def _log_summary(instance_id: int, run_id: str, wandb_url: str, mode: Mode) -> None:
    logger.section("vast.ai instance ready")
    logger.values(
        {
            "instance": instance_id,
            "mode": mode,
            "run_id": run_id,
            "ssh": f"ssh {SSH_HOST_ALIAS}",
            "logs": f"vastai logs {instance_id}",
            "wandb": wandb_url,
            "destroy": f"vastai destroy instance {instance_id}",
        }
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
