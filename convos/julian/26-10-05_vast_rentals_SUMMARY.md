# SUMMARY: vast.ai rentals — RTX 4090 hosts failing at ssh / `uv sync`

**Last updated:** 26-10-05 (datacenter filter implemented for the 4090 config)

Julian's 4090 rentals often fail: ssh never accepts the connection, or `uv sync` exceeds the 10-min budget. H100 rentals usually work. The 4090 is meant for UTH stage 0 probes (inference only; a 4090 cannot run tPD training) — see `convos/julian/26-10-05_uth_experiments_SUMMARY.md`.

- **Cause** [assumed]: 43 of the 50 matching 4090 offers are non-datacenter hosts [verified: offer search 26-10-05], whose port mappings and routes to PyPI vary widely. The 4090 config's perf-per-dollar sort puts the cheapest of these first.
- **Launcher gap** [verified: code]: `spd-vast` raises on ssh/sync failure but leaves the failed instance billing (unless `--destroy_on_exit`).
- **Implemented: datacenter filter** (Julian asked for it in chat) [verified: lint/typecheck clean, `spd-vast --list_offers` with both configs]. New required `VastConfig` field `datacenter_only: bool`; when it is true, the offer query gets `datacenter=true`. When it is false, nothing is added, because `datacenter=false` would select only non-datacenter hosts. Default 4090 config: `true` (7 offers, from $0.551/h); H100 config: `false` (unchanged, 4 offers).
- **Open**: (a) with the perf-per-dollar sort, the first 4090 pick is the lowest-reliability datacenter host (0.9544); `min_reliability: 0.98` would make it a $0.721/h host with reliability 0.9997. (b) Auto-destroy-and-retry on ssh/sync failure, skipping the failed machine. A custom docker image is not recommended.

Details and the offer table: [LOG](26-10-05_vast_rentals_LOG.md).

- See also: [convos/julian/26-10-02_training_run_SUMMARY.md] — earlier H100 rentals with `spd-vast` (login shell for the WandB key, etc.)
