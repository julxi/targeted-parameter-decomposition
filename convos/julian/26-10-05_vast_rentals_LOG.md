C: (transcribed from chat) I have problems renting RTX 4090. Whenever I rent a H100 it usually works. When I rent the other sometimes it doesn't allow the ssh connection and even if that works it usually fails to install all dependencies in time. Is there anything I can do about that?

---

26-10-05, Claude Opus 5.5 — *why 4090 rentals fail and what to change*; I have read INDEX.md, `spd/scripts/run_vast.py`, both vast configs, the UTH SUMMARY, and grepped the training-run LOG for vast/ssh notes

**What the 4090 is for.** UTH stage 0 (`probe_uth.py`, forward passes only). That fits in 24 GB (7B in bf16 ≈ 15 GB); tPD training does not (≈ 27 GB peak at batch 16) [assumed: from `convos/julian/26-10-05_no_truth_baseline_LOG.md`, which derived it from H100 memory].

**What the launcher already does** [verified: read the code]:
- it waits up to 300 s for ssh and re-attaches the key once halfway through;
- it gives `uv sync` 10 min (`max_sync_minutes`), and then raises an error;
- on any of these failures it raises, **and the broken instance keeps billing** unless `--destroy_on_exit` was passed (the `finally` block only prints the destroy command). Recovery is manual: destroy it, relaunch.

**Market check** [verified: `vastai search offers`, 26-10-05, the 4090 config's query with `dph_total` sort, 60 GB storage priced in]:

| Query | Offers | Cheapest |
|---|---|---|
| 4090 config as is (reliability > 0.95) | 50 (43 non-datacenter, 7 datacenter) | $0.363/h, all top 6 non-datacenter |
| + `datacenter=true` | 7 | $0.551/h (GB, reliability 0.954); then $0.721/h (US, reliability 1.000) |
| + `datacenter=true`, reliability > 0.98 | 6 | $0.721/h |
| H100 config as is | 4 (2 datacenter, 2 not) | — |

Interpretation [assumed]: most 4090s are hosted by individuals or small operators on residential or shared links. The broken direct-port mappings (ssh fails) and slow routes to PyPI (sync fails) are host-level problems, and this is where they are most common. H100 hosts are bigger operations even when vast.ai doesn't label them datacenter, but the H100 market is too thin (4 offers) to test that properly. The config also sorts by `dlperf_per_dphtotal-` (performance per dollar), which pushes the cheapest hosts to the top.

**Options**, most useful first:
1. **Add `datacenter=true` to the 4090 search** (a new `VastConfig` field, e.g. `datacenter_only`, so the H100 config can choose separately). Cost: about $0.72/h instead of about $0.37/h. Stage 0 is a short job [assumed: "minutes on an H100"], so the setup time dominates, and a failed rental costs more than the price difference.
2. **Automatic retry**: when ssh, the instance status or `uv sync` fails, destroy the instance and rent the next offer, skipping the failed `machine_id`, up to N attempts. This removes the manual destroy-and-relaunch and stops failed hosts from billing. It is the larger code change, and it also helps H100s.
3. **Raise `min_reliability` to 0.98** (as in the H100 config), and/or sort by `reliability-`. This is cheap but weaker than 1: reliability measures uptime, not the network route.
4. Not recommended: baking the venv into a custom docker image. The image pull goes over the same bad network, and a custom image is not cached on hosts, unlike `vastai/base-image`.

Side note [concluded: the uv sync downloads about 3–4 GB of torch + 34 nvidia wheels (count from `uv.lock`), and the model is ~15 GB]: even after a sync passes, stage 0 downloads Qwen2.5-7B (~15 GB) from HuggingFace over the same link. So a host that only just passes the sync budget will still be slow, which is one more reason to prefer option 1.

Not implemented; waiting for Julian to choose.

---

C: (transcribed from chat) ok, add datacenter = true

---

26-10-05, Claude Opus 5.5 — *datacenter filter implemented*

Implemented option 1 [verified: `ruff check`, `ruff format --check`, `basedpyright` clean on `run_vast.py`; `spd-vast --list_offers` run with both configs]:
- `VastConfig` has a new required field, `datacenter_only: bool`. When it is true, `_search_offers` appends `datacenter=true` to the query. When it is false, it adds nothing, because `datacenter=false` restricts the search to non-datacenter hosts [verified: that query returned 43 offers, all with `hosting_type` 0].
- `vast_config.yaml` (the default, RTX 4090): `datacenter_only: true`. `spd-vast --list_offers` now shows 7 offers, from $0.551/h.
- `vast_h100_config.yaml`: `datacenter_only: false`, so H100 behaviour is unchanged (4 offers). H100 rentals have worked so far, and the filter would halve an already thin market. Julian asked only about the 4090, so this is my choice [concluded].

Open point: the 4090 config sorts by performance per dollar, so it would rent the GB host first ($0.551/h). That host has the lowest reliability of the seven (0.9544, just above the 0.95 floor). Raising `min_reliability` to 0.98 would make the first pick the Maryland host at $0.721/h (reliability 0.9997). Not changed; Julian to decide. Option 2 (auto-retry) is still open.
