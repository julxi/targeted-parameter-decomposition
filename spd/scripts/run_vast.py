"""SPD experiment runner for rented vast.ai GPUs.

vast.ai is a GPU marketplace, not a job service: this script searches offers, rents a machine,
attaches an ssh key to it, writes an ssh config stanza, rsyncs the working tree over, and starts
training. Checkpoints are synced to WandB because the machine (and its disk) disappear when the
instance is destroyed.
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
from typing import Any, Literal, NamedTuple

import fire
import yaml
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

Mode = Literal["run", "detached", "provision", "sync"]


class ExperimentLaunch(NamedTuple):
    """The experiment to run on the rented machine, absent when provisioning a bare one."""

    experiment: str
    remote_script: str
    wandb_url: str


DEFAULT_VAST_CONFIG = "vast_config.yaml"
SSH_HOST_ALIAS = "vastai"
SSH_CONFIG_PATH = Path.home() / ".ssh/config.d/vastai.conf"
DEFAULT_SSH_IDENTITY_FILE = Path.home() / ".ssh/id_rsa"
REMOTE_REPO_DIR = "/root/spd"
REMOTE_OUT_DIR = "/root/spd_out"
REMOTE_LOG = f"{REMOTE_REPO_DIR}/train.log"
REMOTE_ENV_FILE = "/etc/profile.d/spd_env.sh"
READY_TIMEOUT_S = 900
POLL_INTERVAL_S = 10
SSH_READY_TIMEOUT_S = 300
SSH_POLL_INTERVAL_S = 5
# GNU coreutils `timeout` reports this when it kills the command it wrapped.
TIMEOUT_EXIT = 124
# vast.ai reports these when the host has dropped off its control plane. It keeps claiming
# intended_status "running", but the container never finishes provisioning.
DEAD_STATUSES = frozenset({"missing", "exited"})
# vast.ai's error for an ask that has already been rented by someone else.
TAKEN_OFFER_ERROR = "no_such_ask"
# Tracked, but not needed to train, so not worth the upload.
UNSYNCED_REPO_DIR = "papers"
BLACKLIST_PATH = REPO_ROOT / "spd/scripts/vast_blacklist.yaml"
# Rewritten on top of the file whenever spd-vast adds a host, which drops any other comments.
BLACKLIST_HEADER = """\
# vast.ai hosts that spd-vast never rents again, whatever the machine config.
# A host is one operator (host_id), who may run several machines (machine_id). The failures that
# land a host here - ssh never connecting, `uv sync` or downloads crawling - come from the
# operator's network, which all of their machines share, so the whole host is excluded.
# spd-vast appends a host itself when ssh, booting or `uv sync` fails on it. Add hosts by hand
# when a rental turns out bad later (e.g. a slow Hugging Face download): `spd-vast` prints the
# host id of every offer and of the rented instance. Delete an entry to rent that host again.
# Entry format (replace `hosts: []` by a list if it is empty):
#   hosts:
#   - host_id: 124072
#     date: 26-10-06
#     reason: Hugging Face download at 2 MB/s (instance 1234567)
# This file is rewritten when spd-vast adds a host: put notes in `reason`, not in comments.
"""


class HostFailure(Exception):
    """A rental that failed because of the host, not because of us, so the host is blacklisted.

    `reason` is the short form that goes into the blacklist; the exception message is the full
    explanation shown to the user. `main` blacklists the host and destroys the instance, so the
    message need not tell the user to.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class BlacklistedHost(BaseConfig):
    host_id: int = Field(description="vast.ai host id, i.e. the operator, not a single machine")
    date: str = Field(description="When the host was added, YY-MM-DD")
    reason: str = Field(description="What failed, and on which instance")


class VastBlacklist(BaseConfig):
    """Hosts excluded from every offer search, loaded from `vast_blacklist.yaml`."""

    hosts: list[BlacklistedHost]


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
    datacenter_only: bool = Field(
        description="Rent only hosts vast.ai lists as datacenters. Most consumer-GPU offers (e.g. "
        "RTX 4090) are not, and those hosts often fail to open the direct ssh port or reach PyPI "
        "too slowly for `uv sync`. False searches all hosts, it does not exclude datacenters"
    )
    min_cuda: float = Field(
        description="Minimum CUDA version the host supports, matched to the torch wheels in uv.lock"
    )
    sort: str = Field(
        description="vast.ai sort order; the first offer is rented (e.g. 'dlperf_per_dphtotal-' "
        "for best performance per dollar, 'dph_total' for cheapest)"
    )
    n_offers: int = Field(description="How many offers to fetch and show")
    max_sync_minutes: float = Field(
        description="How long `uv sync` may take on the rented machine before the host is "
        "rejected. It pulls several GB of torch and CUDA wheels, and hosts with a poor route to "
        "PyPI stall here for tens of minutes before training starts"
    )


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
        experiment: Experiment name from registry (e.g. 'tms_5-2'). Omit it with --list_offers,
            or with --mode provision to rent a bare machine to work on over ssh.
        config: Machine config: a short name ('h100' -> spd/scripts/vast_h100_config.yaml), a
            filename in spd/scripts, or a path.
        mode: 'run' streams training until it exits, 'detached' starts it and returns,
            'provision' only rents + syncs and leaves the run for you to start over ssh,
            'sync' re-rsyncs the working tree to the instance behind the `vastai` ssh alias
            without renting anything (dependencies are not re-synced).
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
        spd-vast --mode provision                       # rent a bare machine, no experiment
        spd-vast --mode sync                            # push local edits to the last instance
        spd-vast tms_5-2 --destroy_on_exit              # stop paying when training ends
    """
    if mode == "sync":
        assert experiment is None, "--mode sync only syncs the working tree, it takes no experiment"
        assert SSH_CONFIG_PATH.exists(), (
            f"{SSH_CONFIG_PATH} not found - rent an instance first with --mode provision"
        )
        if dry_run:
            logger.values({"rsync": shlex.join(_rsync_args())})
            return
        _rsync_repo()
        logger.info(f"Synced working tree to {SSH_HOST_ALIAS}:{REMOTE_REPO_DIR}")
        return

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

    blacklisted = _blacklisted_host_ids(_load_blacklist())

    if list_offers:
        _print_offers(_search_offers(vast_config, blacklisted), vast_config.sort, vast_config.disk)
        return

    assert experiment is not None or mode == "provision", (
        "experiment is required unless --list_offers or --mode provision is passed"
    )
    assert not (destroy_on_exit and mode != "run"), "destroy_on_exit requires mode='run'"
    identity_file = _ssh_identity_file()

    run_id = generate_run_id("spd")
    launch = _build_launch(experiment, run_id, project) if experiment is not None else None
    label = f"spd-{launch.experiment if launch else 'provision'}-{run_id}"

    env_vars = _forwarded_env_vars()
    offers = (
        _search_offers(vast_config, blacklisted)
        if offer_id is None
        else [_find_offer(offer_id, blacklisted)]
    )

    if dry_run:
        redacted_args = _create_instance_args(
            int(offers[0]["id"]), vast_config, label, dict.fromkeys(env_vars, "<redacted>")
        )
        logger.section("Dry run - nothing rented")
        logger.values(
            {
                "create": shlex.join(["vastai", *redacted_args]),
                "host": offers[0]["host_id"],
                "rsync": shlex.join(_rsync_args()),
                "remote command": launch.remote_script if launch else _build_sync_script(),
                "wandb": launch.wandb_url if launch else "-",
            }
        )
        return

    instance_id, offer = _rent_first_available(offers, vast_config, label, env_vars)
    logger.info(
        f"Created instance {instance_id} on host {offer['host_id']} (machine "
        f"{offer['machine_id']}), waiting for it to boot"
    )
    destroy = destroy_on_exit
    try:
        _wait_until_running(instance_id)
        _attach_ssh_key(instance_id, identity_file)
        host, port = _ssh_host_port(instance_id)
        _write_ssh_config(host, port, run_id, instance_id, identity_file)
        logger.info(f"Wrote {SSH_CONFIG_PATH} - connect with `ssh {SSH_HOST_ALIAS}`")
        _wait_until_ssh_ready(instance_id, identity_file)
        _install_remote_env(env_vars)
        _rsync_repo()
        _sync_dependencies(vast_config.max_sync_minutes)

        _log_summary(instance_id, offer, run_id, launch, mode)

        match mode:
            case "provision":
                next_step = (
                    f"ssh {SSH_HOST_ALIAS}, then: {launch.remote_script}"
                    if launch
                    else f"ssh {SSH_HOST_ALIAS}, then: cd {REMOTE_REPO_DIR} "
                    f"&& source .venv/bin/activate"
                )
                logger.values({"next step": next_step})
            case "detached":
                assert launch is not None
                _ssh(
                    f"setsid nohup bash -lc {shlex.quote(launch.remote_script)} "
                    f"> {REMOTE_LOG} 2>&1 < /dev/null & echo started"
                )
                logger.info(
                    f"Training started. Follow with: ssh {SSH_HOST_ALIAS} tail -f {REMOTE_LOG}"
                )
            case "run":
                assert launch is not None
                exit_code = _ssh_streaming(
                    f"set -o pipefail; bash -lc {shlex.quote(launch.remote_script)} 2>&1 "
                    f"| tee {REMOTE_LOG}"
                )
                logger.info(f"Training exited with code {exit_code}")
    except HostFailure as failure:
        # The failed instance is destroyed (in `finally`) so it stops billing, but intentionally
        # no next offer is rented: Julian wants to decide on a relaunch himself
        # (convos/julian/26-10-05_vast_rentals_LOG.md).
        destroy = True
        _add_to_blacklist(
            BlacklistedHost(
                host_id=int(offer["host_id"]),
                date=datetime.now().strftime("%y-%m-%d"),
                reason=f"{failure.reason} (instance {instance_id}, machine {offer['machine_id']}, "
                f"{offer['gpu_name']}, {offer['geolocation']})",
            )
        )
        raise
    finally:
        if destroy:
            _destroy_instance(instance_id)
        else:
            logger.info(
                f"Instance still running. Destroy it with: vastai destroy instance {instance_id} -y"
            )


def _destroy_instance(instance_id: int) -> None:
    """Destroy an instance, failing loudly unless vast.ai confirms it.

    The CLI exits 0 whatever happens: on success it prints "destroying instance <id>.", on an
    unknown instance "Failed with error 404: ..." (to stderr), and without `-y` it prompts, reads
    a closed stdin and prints "Aborted." [all observed with vastai 1.2.0 on 26-10-06]. So the
    output, both streams, is the only signal. A 404 is the one failure to accept, because an
    instance whose host dropped it is already gone (see `_show_instance`).
    """
    logger.info(f"Destroying instance {instance_id}")
    result = subprocess.run(
        ["vastai", "destroy", "instance", str(instance_id), "-y"],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    output = (result.stdout + result.stderr).strip()
    logger.info(output)
    assert output == f"destroying instance {instance_id}." or "error 404" in output, (
        f"vast.ai did not confirm destroying instance {instance_id} ({output!r}), so it may still "
        f"be billing. Check `vastai show instances` and destroy it with: "
        f"vastai destroy instance {instance_id} -y"
    )


def _build_config(config_path: Path, project: str) -> Config:
    """Load the experiment config, forcing checkpoint upload since the instance is ephemeral."""
    config_dict = Config.from_file(config_path).model_dump(mode="json")
    config_dict["wandb_project"] = project
    config_dict["sync_checkpoints_to_wandb"] = True
    return Config(**config_dict)


def _build_sync_script() -> str:
    """Build the venv. Run once by `_sync_dependencies`, not by every command.

    The image is expected to ship uv; vast.ai's CUDA images do, and installing our own would mask
    an image that is not the one we think we rented.

    Intentionally includes the `dev` dependency group (no `--no-dev`): analysis scripts that run on
    the rented machine need it, e.g. the probe scripts import scikit-learn, which is declared there.
    """
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {REMOTE_REPO_DIR}",
            "command -v uv >/dev/null || { echo 'uv is not installed in this image' >&2; exit 1; }",
            "uv sync --link-mode copy",
        ]
    )


def _build_remote_script(train_command: str) -> str:
    """Enter the synced repo and its already-built venv, then run training."""
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {REMOTE_REPO_DIR}",
            f"export SPD_OUT_DIR={REMOTE_OUT_DIR}",
            "source .venv/bin/activate",
            train_command,
        ]
    )


def _build_launch(experiment: str, run_id: str, project: str) -> ExperimentLaunch:
    """Resolve a registry experiment into the command to run on the rented machine."""
    assert experiment in EXPERIMENT_REGISTRY, (
        f"Unknown experiment '{experiment}'. Available: {', '.join(sorted(EXPERIMENT_REGISTRY))}"
    )
    exp_config = EXPERIMENT_REGISTRY[experiment]
    job = TrainingJob(
        experiment=experiment,
        script_path=exp_config.decomp_script,
        config=_build_config(REPO_ROOT / exp_config.config_path, project),
        run_id=run_id,
    )
    train_command = get_command(
        launch_id=f"vast-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        job=job,
        job_idx=0,
        n_gpus=1,
        sweep_params=None,
        snapshot_branch="",
        is_array=False,
    ).command
    return ExperimentLaunch(
        experiment=experiment,
        remote_script=_build_remote_script(train_command),
        wandb_url=get_wandb_run_url(project, run_id),
    )


def _vastai(args: list[str], raw: bool = True) -> Any:
    """Run a vastai CLI command, parsing its JSON output when raw.

    Under `--raw` the CLI reports an API failure by printing a JSON error object to stderr and
    exiting 0 with an empty stdout, so empty stdout means the command failed rather than that it
    found nothing. Only the leading tokens of a failed command are reported, to keep any
    credentials passed in later arguments out of the error.

    stdin is closed because stdout is captured: a command that prompts for confirmation (e.g.
    `destroy instance` without `-y`) would otherwise wait on an invisible prompt forever. With
    no stdin the prompt aborts at once - but the CLI still exits 0, so such commands must check
    their output (see `_destroy_instance`).
    """
    cmd = ["vastai", *args, "--raw"] if raw else ["vastai", *args]
    result = subprocess.run(
        cmd, check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    if not raw:
        return result.stdout
    stdout = result.stdout.strip()
    redacted_command = shlex.join(["vastai", *args[:3]])
    assert stdout, f"`{redacted_command}` failed: {result.stderr.strip() or 'no error message'}"
    return json.loads(stdout)


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
    # Rebuild rather than model_copy(update=...), which skips validation and would let a
    # mistyped flag (e.g. an offer id passed to --gpu_name) through into the offer search.
    return VastConfig(**{**vast_config.model_dump(), **given})


def _search_offers(config: VastConfig, blacklisted: list[int]) -> list[Any]:
    """Search offers in the config's sort order; the caller rents the first one.

    `disk_space` is the machine's free disk, which must fit the disk we intend to rent.
    `direct_port_count>=1` is required for the `--direct` ssh this script relies on. `gpu_ram` is
    queried in GB even though offers report it in MB. The datacenter term is added only when
    required, because `datacenter=false` would restrict the search to non-datacenter hosts instead
    of leaving it open. Blacklisted hosts are excluded in the query, so `n_offers` still counts
    rentable ones; `host_id` is missing from the CLI's documented query fields (it worked when
    tested on 26-10-06), so the result is checked for leaks too.
    """
    query = (
        f"num_gpus=1 gpu_name={config.gpu_name} rentable=true verified=true "
        f"reliability>{config.min_reliability} direct_port_count>=1 "
        f"cuda_max_good>={config.min_cuda} gpu_ram>={config.min_gpu_ram} "
        f"disk_space>={config.disk} dph_total<{config.max_price}"
    )
    if config.datacenter_only:
        query += " datacenter=true"
    if blacklisted:
        query += f" host_id notin [{','.join(map(str, blacklisted))}]"
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
        f"in the config or with the matching flag. {len(blacklisted)} hosts are excluded by "
        f"{BLACKLIST_PATH.relative_to(REPO_ROOT)}."
    )
    leaked = sorted({o["host_id"] for o in offers} & set(blacklisted))
    assert not leaked, (
        f"vast.ai ignored the host_id exclusion and returned blacklisted hosts {leaked}"
    )
    return offers


def _find_offer(offer_id: int, blacklisted: list[int]) -> Any:
    """Look up one offer, for --offer_id, so its host is known and checked like a searched one.

    vast.ai's query language has no documented field for the offer id: `id=` matches nothing, and
    `ask_contract_id=` works but the CLI warns that it does not know the field. `-n` drops the
    CLI's default terms (rentable, verified, ...) so a hand-picked offer is not filtered out.
    """
    found = _vastai(["search", "offers", "-n", f"ask_contract_id={offer_id}"])
    assert [int(o["id"]) for o in found] == [offer_id], (
        f"Looking up offer {offer_id} returned {[o['id'] for o in found]} - it is no longer on "
        f"the market, or vast.ai has changed how offers are looked up by id."
    )
    offer = found[0]
    assert offer["host_id"] not in blacklisted, (
        f"Offer {offer_id} is on blacklisted host {offer['host_id']}. Remove the host from "
        f"{BLACKLIST_PATH.relative_to(REPO_ROOT)} to rent it anyway."
    )
    return offer


def _load_blacklist() -> VastBlacklist:
    assert BLACKLIST_PATH.exists(), f"Host blacklist not found: {BLACKLIST_PATH}"
    blacklist = VastBlacklist.from_file(BLACKLIST_PATH)
    host_ids = [h.host_id for h in blacklist.hosts]
    assert len(host_ids) == len(set(host_ids)), f"Duplicate host ids in {BLACKLIST_PATH}"
    return blacklist


def _blacklisted_host_ids(blacklist: VastBlacklist) -> list[int]:
    host_ids = sorted(h.host_id for h in blacklist.hosts)
    if host_ids:
        logger.info(
            f"Excluding {len(host_ids)} blacklisted hosts "
            f"({BLACKLIST_PATH.relative_to(REPO_ROOT)}): {host_ids}"
        )
    return host_ids


def _add_to_blacklist(entry: BlacklistedHost) -> None:
    """Append a host to the git-tracked blacklist, so the next search skips it."""
    blacklist = _load_blacklist()
    assert entry.host_id not in {h.host_id for h in blacklist.hosts}, (
        f"Host {entry.host_id} failed but is already blacklisted - the search should have "
        f"excluded it"
    )
    hosts = [*blacklist.hosts, entry]
    BLACKLIST_PATH.write_text(
        BLACKLIST_HEADER
        + yaml.safe_dump(
            VastBlacklist(hosts=hosts).model_dump(mode="json"), sort_keys=False, width=100
        )
    )
    logger.info(
        f"Blacklisted host {entry.host_id} in {BLACKLIST_PATH.relative_to(REPO_ROOT)}: "
        f"{entry.reason}"
    )


def _print_offers(offers: list[Any], sort: str, disk: int) -> None:
    """Print the offers as a table.

    'host' is the operator's host id, the unit the blacklist excludes. `gpu_ram` and `cpu_ram` are
    reported in MB, disk in GB. 'free' is the machine's unallocated
    disk, not the disk we rent - `$/hr` already covers the rented disk.
    """
    logger.section(
        f"{len(offers)} matching offers, sorted by {sort}. $/hr includes the {disk}GB disk"
    )
    header = (
        f"{'id':>10} {'host':>7}  {'gpu':<16} {'vram':>5} {'$/hr':>6} {'free':>6} {'ram':>6} {'cores':>5} "
        f"{'cuda':>5} {'net↓':>7} {'dlperf':>7} {'dlperf/$':>8} {'reliab':>7}  location"
    )
    rows = [
        f"{o['id']:>10} {o['host_id']:>7}  {o['gpu_name']:<16} {o['gpu_ram'] / 1024:>4.0f}G "
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
    label: str,
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
        label,
        "--cancel-unavail",
        "--env",
        env_arg,
    ]


def _rent_first_available(
    offers: list[Any], config: VastConfig, label: str, env_vars: dict[str, str]
) -> tuple[int, Any]:
    """Rent the best offer still on the market, walking down the search order.

    Returns the instance id and the offer it was rented from (whose host gets blacklisted if the
    rental fails).

    vast.ai's offer search serves asks that someone else has already rented, so the top offer is
    regularly gone by the time we ask for it. Only a taken offer moves us on to the next one; any
    other create failure is ours and is raised.
    """
    for offer in offers:
        offer_id = int(offer["id"])
        logger.info(
            f"Renting offer {offer_id}: {offer['gpu_name']} at ${offer['dph_total']:.3f}/hr "
            f"in {offer['geolocation']} (reliability {offer['reliability']:.4f}, host "
            f"{offer['host_id']})"
        )
        instance_id = _create_instance(_create_instance_args(offer_id, config, label, env_vars))
        if instance_id is not None:
            return instance_id, offer
        logger.info(f"Offer {offer_id} was taken before we got it, trying the next one")
    raise AssertionError(
        f"All {len(offers)} matching offers were taken before we could rent one. Launch again - "
        f"the search index lags the marketplace by minutes, so a fresh search lists different asks."
    )


def _create_instance(create_args: list[str]) -> int | None:
    """Rent an offer, returning None if it was already taken. Nothing is rented on failure.

    `--cancel-unavail` makes vast.ai reject a taken ask outright instead of parking a stopped
    instance on us, and it reports that as `no_such_ask`. This does not go through `_vastai`
    because a taken ask is an expected outcome here rather than an error to report, and because
    the command carries the WandB key in its `--env` argument.
    """
    result = subprocess.run(
        ["vastai", *create_args, "--raw"], check=True, capture_output=True, text=True
    )
    stdout = result.stdout.strip()
    if not stdout:
        stderr = result.stderr.strip()
        assert TAKEN_OFFER_ERROR in stderr, (
            f"`vastai create instance` failed: {stderr or 'no error message'}"
        )
        return None
    created = json.loads(stdout)
    assert created["success"], f"Failed to create instance: {created}"
    return int(created["new_contract"])


def _show_instance(instance_id: int) -> dict[str, Any]:
    """Look an instance up, failing loudly if it has vanished.

    An instance whose host drops it stops existing rather than reporting a dead status, and
    `show instance` answers for it with `{"instances": null}` instead of the flat instance dict.
    """
    shown = _vastai(["show", "instance", str(instance_id)])
    if "actual_status" not in shown:
        raise HostFailure(
            "host dropped the rental",
            f"Instance {instance_id} no longer exists ({shown}). Its host dropped it, which "
            f"vast.ai reports by forgetting the rental rather than by marking it dead. Launch "
            f"again to land on a different host.",
        )
    return shown


def _wait_until_running(instance_id: int) -> None:
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        instance = _show_instance(instance_id)
        status = instance["actual_status"]
        if status == "running":
            return
        if status in DEAD_STATUSES:
            raise HostFailure(
                f"instance went '{status}' while booting",
                f"Instance {instance_id} is '{status}' "
                f"({instance.get('status_msg') or 'no message'}). The host has dropped off "
                f"vast.ai's control plane and will never install your ssh key, however healthy "
                f"the control plane claims it is.",
            )
        logger.info(f"Instance status: {status} ({instance.get('status_msg') or 'no message'})")
        time.sleep(POLL_INTERVAL_S)
    raise HostFailure(
        f"not running after {READY_TIMEOUT_S}s",
        f"Instance {instance_id} not running after {READY_TIMEOUT_S}s.",
    )


def _ssh_identity_file() -> Path:
    """The private key ssh authenticates to rented machines with.

    The path is machine-specific but not secret, so it lives in the gitignored `.env` rather than
    the git-tracked vast config, where every collaborator would rewrite the line.
    """
    configured = dotenv_values(REPO_ROOT / ".env").get("SPD_VAST_SSH_KEY") or os.getenv(
        "SPD_VAST_SSH_KEY"
    )
    key = Path(configured).expanduser() if configured else DEFAULT_SSH_IDENTITY_FILE
    assert key.exists(), f"ssh key not found: {key}. Set SPD_VAST_SSH_KEY in .env."
    return key


def _public_key_path(identity_file: Path) -> Path:
    """OpenSSH appends `.pub` to the whole filename, so `vast.key` pairs with `vast.key.pub`."""
    return identity_file.with_name(identity_file.name + ".pub")


def _attach_ssh_key(instance_id: int, identity_file: Path) -> None:
    """Push our public key to the instance through vast.ai's control plane.

    vast.ai's entrypoint copies account-level keys into authorized_keys at container start, but on
    some hosts that is slow or never happens, and there is no way in to fix it by hand: `vastai
    execute` is restricted to ls/rm/du and cannot write a file. Attaching is idempotent, so it runs
    on every rental rather than only on the ones that turn out to need it.
    """
    public_key = _public_key_path(identity_file)
    assert public_key.exists(), f"public key not found: {public_key}"
    _vastai(["attach", "ssh", str(instance_id), public_key.read_text().strip()], raw=False)


def _ssh_host_port(instance_id: int) -> tuple[str, int]:
    """Parse `vastai ssh-url`, which reports the right host/port for direct or proxied ssh."""
    ssh_url = _vastai(["ssh-url", str(instance_id)], raw=False).strip()
    assert ssh_url.startswith("ssh://"), f"Unexpected ssh url: {ssh_url}"
    host_port = ssh_url.removeprefix("ssh://").split("@")[-1]
    host, port = host_port.split(":")
    return host, int(port)


def _write_ssh_config(
    host: str, port: int, run_id: str, instance_id: int, identity_file: Path
) -> None:
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
        f"    IdentityFile {identity_file}\n"
        f"    Port {port}\n"
        f"    StrictHostKeyChecking no\n"
        f"    UserKnownHostsFile /dev/null\n"
    )


def _wait_until_ssh_ready(instance_id: int, identity_file: Path) -> None:
    """Poll until sshd accepts our key, re-attaching it once halfway through.

    `actual_status == "running"` only means the container started: vast.ai's entrypoint installs
    authorized_keys a few seconds later, and connections made before that are closed mid-auth. A
    host that rejects the key is not a lost cause either - the same key is often accepted minutes
    later - so the loop keeps polling through "Permission denied" and re-attaches once in case the
    entrypoint overwrote authorized_keys after we first attached.
    """
    logger.info("Waiting for ssh to accept our key")
    deadline = time.time() + SSH_READY_TIMEOUT_S
    reattach_at = time.time() + SSH_READY_TIMEOUT_S / 2
    reattached = False
    last_error = "<no probe ran>"
    while time.time() < deadline:
        probe = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", SSH_HOST_ALIAS, "true"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return
        last_error = probe.stderr.strip() or f"exit code {probe.returncode}, no stderr"
        if not reattached and time.time() >= reattach_at:
            logger.info(f"ssh still failing ({last_error}) - re-attaching the key")
            _attach_ssh_key(instance_id, identity_file)
            reattached = True
        time.sleep(SSH_POLL_INTERVAL_S)
    status = _show_instance(instance_id)["actual_status"]
    if status in DEAD_STATUSES:
        raise HostFailure(
            f"instance went '{status}' while waiting for ssh",
            f"Instance {instance_id} went '{status}' while we waited for ssh. The host dropped "
            f"off vast.ai's control plane after reporting itself running, so it never installed "
            f"your key - the rejection is the host's fault, not your key's.",
        )
    # Blacklisted although a broken local key would fail the same way: a key that works on other
    # hosts is far more common, and the first blacklisted host would reveal a broken one.
    raise HostFailure(
        f"ssh failing after {SSH_READY_TIMEOUT_S}s: {last_error.splitlines()[-1]}",
        f"ssh to {SSH_HOST_ALIAS} still failing after {SSH_READY_TIMEOUT_S}s while instance "
        f"{instance_id} reports '{status}'.\n"
        f"Last ssh error: {last_error}\n"
        f"'Permission denied (publickey)' means sshd is up but never took the key in "
        f"{identity_file}; a refused or timed out connection means the direct port mapping never "
        f"came up. Read the entrypoint's own account with: vastai logs {instance_id}",
    )


def _sync_dependencies(max_sync_minutes: float) -> None:
    """Build the venv on the rented machine, rejecting the host if it takes too long.

    `uv sync` downloads several GB of torch and CUDA wheels, and how fast that goes depends on the
    host's route to PyPI - which an offer's self-reported `inet_down` does not predict, and which a
    generic speed test measures the wrong path for. Timing the real sync is the direct check, and
    `timeout` runs remotely so a host that blows the budget is left with nothing still running.
    """
    budget_s = int(max_sync_minutes * 60)
    command = f"timeout {budget_s} bash -lc {shlex.quote(_build_sync_script())}"
    logger.info(f"Syncing dependencies (budget {max_sync_minutes} min)")
    start = time.time()
    returncode = subprocess.run(["ssh", SSH_HOST_ALIAS, command], check=False).returncode
    elapsed_min = (time.time() - start) / 60
    if returncode == TIMEOUT_EXIT:
        raise HostFailure(
            f"`uv sync` over {max_sync_minutes} min",
            f"`uv sync` was still running after max_sync_minutes={max_sync_minutes}. This host's "
            f"route to PyPI is too slow to be worth training on.",
        )
    assert returncode == 0, f"`uv sync` failed on the rented machine with exit {returncode}"
    logger.info(f"Dependencies synced in {elapsed_min:.1f} min")


def _install_remote_env(env_vars: dict[str, str]) -> None:
    """Put the container's env vars somewhere ssh sessions can see them.

    vast.ai passes `--env` to the container, so PID 1 has the WandB credentials, but sshd starts
    sessions that do not inherit docker's environment. Everything this script runs goes over ssh as
    a login shell, so without this the training run finds no credentials and W&B drops to an
    interactive login prompt that nothing is there to answer.
    """
    exports = "\n".join(f"export {name}={shlex.quote(value)}" for name, value in env_vars.items())
    script = f"umask 077 && cat > {REMOTE_ENV_FILE} <<'SPD_ENV'\n{exports}\nSPD_ENV"
    subprocess.run(["ssh", SSH_HOST_ALIAS, script], check=True)
    logger.info(f"Wrote credentials to {REMOTE_ENV_FILE} for ssh sessions")


def _rsync_args() -> list[str]:
    """Sync the working tree, `.git` included.

    Every target-model training script stamps its run via `ExecutionStamp.create`, which shells out
    to `git`. Without the repo those scripts die on `git status --porcelain` before training starts,
    and the run loses its branch and commit provenance.

    Ownership is deliberately not preserved: `-a` would carry our local uid across, and the
    container runs as root, so git would then refuse the repo as having dubious ownership.

    `papers/` is left out: it is a fifth of the upload, mostly figures, and nothing on the rented
    machine reads it. `--info=progress2` prints one running total, since a slow route to the host
    otherwise looks like a hang.
    """
    return [
        "rsync",
        "-az",
        "--no-owner",
        "--no-group",
        "--delete",
        "--info=progress2",
        "--filter=:- .gitignore",
        "--exclude=.venv",
        f"--exclude=/{UNSYNCED_REPO_DIR}/",
        f"{REPO_ROOT}/",
        f"{SSH_HOST_ALIAS}:{REMOTE_REPO_DIR}/",
    ]


def _rsync_repo() -> None:
    args = _rsync_args()
    logger.info(f"Syncing working tree: {shlex.join(args)}")
    subprocess.run(args, check=True)
    _hide_unsynced_dir_from_git()


def _hide_unsynced_dir_from_git() -> None:
    """Mark the tracked files we did not sync as skip-worktree in the remote checkout.

    Otherwise git sees them as deleted, `repo_is_clean()` fails, and every run is stamped as having
    uncommitted changes with no commit hash. The rsync overwrites `.git/index` with our local one,
    so this has to run again after every sync.
    """
    _ssh(
        f"cd {REMOTE_REPO_DIR} && git ls-files -z -- {UNSYNCED_REPO_DIR} "
        f"| git update-index -z --skip-worktree --stdin"
    )


def _ssh(remote_command: str) -> None:
    subprocess.run(["ssh", SSH_HOST_ALIAS, remote_command], check=True)


def _ssh_streaming(remote_command: str) -> int:
    """Run a remote command with its output streamed to our stdout, returning its exit code."""
    return subprocess.run(["ssh", SSH_HOST_ALIAS, remote_command], check=False).returncode


def _log_summary(
    instance_id: int, offer: Any, run_id: str, launch: ExperimentLaunch | None, mode: Mode
) -> None:
    logger.section("vast.ai instance ready")
    logger.values(
        {
            "instance": instance_id,
            "host": f"{offer['host_id']} (machine {offer['machine_id']}, {offer['geolocation']})",
            "mode": mode,
            "run_id": run_id,
            "experiment": launch.experiment if launch else "-",
            "ssh": f"ssh {SSH_HOST_ALIAS}",
            "logs": f"vastai logs {instance_id}",
            "wandb": launch.wandb_url if launch else "-",
            "destroy": f"vastai destroy instance {instance_id} -y",
        }
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
