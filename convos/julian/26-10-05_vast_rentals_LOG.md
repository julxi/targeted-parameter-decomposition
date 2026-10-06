C: (transcribed from chat) I have problems renting RTX 4090. Whenever I rent a H100 it usually works. When I rent the other sometimes it doesn't allow the ssh connection and even if that works it usually fails to install all dependencies in time. Is there anything I can do about that?

---

26-10-05, Claude Opus 5.5 — *why 4090 rentals fail and what to change*; I have read INDEX.md, `spd/scripts/run_vast.py`, both vast configs, the UTH SUMMARY, and grepped the training-run LOG for vast/ssh notes

**What the 4090 is for.** UTH stage 0 (`probe_uth.py`, forward passes only). That fits in 24 GB (7B in bf16 ≈ 15 GB); tPD training does not (≈ 27 GB peak at batch 16) [assumed: from `convos/julian/26-10-05_no_truth_baseline_LOG.md`, which derived it from H100 memory].

**What the launcher already does** [verified: read the code]:
- it waits up to 300 s for ssh and re-attaches the key once halfway through;
- it gives `uv sync` 10 min (`max_sync_minutes`), and then raises an error;
- on any of these failures it raises, **and the broken instance keeps billing** unless `--destroy_on_exit` was passed (the `finally` block only prints the destroy command). Recovery is manual: destroy it, relaunch.
OUTDATED (26-10-06): host-caused failures now destroy the instance (entry *destroy on blacklist*), and `--destroy_on_exit` never actually worked before the `-y` fix (entry *first live test: boot timeout blacklisted, destroy hung*).

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

---

C: (transcribed from chat) I bring this topic up again from another session. I was asking what I can do to increase the reliability of my rentals (relible in the sense that ssh connects or that thet download speed is not abysimal low). One solution is to use datacenter, but then I can't rent any 49GB RTX 4090. I have two questions:
1. is it possible to create a blacklist of machines that have proven to be bad
2. should I maybe switch to another gpu architecture?

---

26-10-06, Claude Opus 5.5 — *blacklist and 48 GB alternatives*; I have read INDEX.md, this topic's LOG and SUMMARY, `run_vast.py` (search and rent code), both vast configs, and grepped the truth-writing-components LOG for memory needs

**Config state.** The working tree (uncommitted, Julian's edit) has the default config at `min_gpu_ram: 40` and `datacenter_only: false`, i.e. it targets the 48 GB RTX 4090 variant (a clamshell-modded card, not an NVIDIA SKU [assumed]). The comment above `datacenter_only` still describes the 24 GB market.

**1. Blacklist: yes, possible** [verified: `vastai search offers`, 26-10-06]. The offer query accepts `machine_id notin [...]` and also `host_id notin [...]`, although `host_id` is not in the CLI's documented field list. Both removed the excluded offers server-side (7 → 5 for two IDs). So a blacklist is one extra query term; no client-side filtering is needed.
- Key it on **`host_id`**, not `machine_id` [concluded: both failure modes are network-side (port forwarding for ssh, the uplink for downloads), and those belong to the host operator, who typically runs all their machines behind one connection. Example in today's market: host 45423 has three A6000 machines in Romania]. Hardware faults would be per machine, but that is not what fails here.
- Its value depends on automation. The launcher logs only the offer ID, so no failed host has been recorded so far; the list starts empty. Useful version: on ssh/sync failure the launcher appends the `host_id` (with date and reason) to a git-tracked list, destroys the instance and rents the next offer. That is option 2 (auto-retry) from the entry *why 4090 rentals fail and what to change*, plus one file.
- Limit: the 48 GB 4090 market is tiny (below), so a blacklist there excludes most of it after a few failures.

**2. Switching GPU: probably yes.** Market for single GPUs with ≥ 40 GB VRAM [verified: offer search 26-10-06, the launcher's query (verified, reliability > 0.95, direct port, CUDA ≥ 12.8, 60 GB disk, price incl. disk) with `gpu_ram>=40`, max $2.5/h; prices are the cheapest offer, "dc" = vast.ai datacenter label]:

| GPU | VRAM | offers | dc | cheapest $/h | cheapest dc $/h | note |
|---|---|---|---|---|---|---|
| RTX 4090 (modded) | 48 GB | 6 (5 hosts) | 0 | 0.43 | — | US, JP; a 7th at > $2.5/h |
| A100 SXM4 | 40 GB | 12 | 1 | 0.42 | 0.74 | mostly US |
| RTX PRO 5000 (Blackwell) | 48 GB | 15 | 4 | 0.63 | 0.75 | |
| RTX A6000 (Ampere) | 48 GB | 4 | 0 | 0.42 | — | dlperf 54 vs 4090's 126 |
| L40S | 48 GB | 3 | 0 | 0.81 | — | |
| RTX PRO 6000 (various) | 96 GB | 43 | 9 | 1.03 | 1.39 | |

(Also listed but unsuitable: Q RTX 8000 — Turing, no bf16; CMP 170HX — a mining card.)

Recommendation [concluded, reasoning given per item]:
- **If 40 GB is enough: A100 SXM4 40GB.** Same entry price as the 48 GB 4090 and twice the offers. SXM modules only exist in server boards, so every host is a server operation even without the dc label [assumed: plausible but not measured; the 24 GB 4090 failures came from hosts that were mostly not]. Its bf16 tensor throughput is higher than a 4090's (spec sheets: 312 vs ~165 TFLOPS dense [assumed: from memory, not re-checked]). Fit: tPD training peaked at ~27 GB at batch 16 in bf16, and UTH stage 0 at ~20.5 GB (both from the truth-writing-components LOG, which cites the no-truth-baseline and UTH SUMMARYs).
- **If 48 GB is needed: RTX PRO 5000.** An official 48 GB card with the most offers in the class and 4 datacenter hosts at $0.75/h. Blackwell (sm_120) needs CUDA 12.8 torch wheels; `uv.lock` pins torch 2.8.0 with the cu128 setup, which supports it [assumed: not tested on a rented PRO 5000].
- **Stay with the 48 GB 4090 only for price**, and accept that every offer is a consumer host. Here a blacklist + auto-retry matters most.

Open question to Julian: why ≥ 40 GB? The decided plan for the truth-writing analyses says a 24 GB 4090. If the reason is A7's backward pass or training on the cheaper card, the A100 40 GB covers both at ~27 GB peak.

Nothing implemented; waiting for Julian's choice (blacklist + auto-retry, and/or a new config such as `vast_a100_config.yaml`).

---

C: (transcribed from chat) I want the blacklist code, also show the host number for each offer and when provisioning. No restarts though. and a config for A100

---

26-10-06, Claude Opus 5.5 — *host blacklist and A100 config implemented*; I have read `docs/CODING.md` and the whole of `run_vast.py`

**Blacklist** (`spd/scripts/vast_blacklist.yaml`, git-tracked, starts as `hosts: []`; one list for all machine configs):
- Entries: `host_id`, `date` (YY-MM-DD), `reason`. Pydantic models `BlacklistedHost` / `VastBlacklist` in `run_vast.py`; duplicate host ids fail loudly.
- Every search adds `host_id notin [...]` to the query, so `n_offers` still counts rentable offers. Because `host_id` is not a documented query field, the result is also checked for blacklisted hosts (assert).
- **Auto-append, no restart**: the host-caused failures now raise a new `HostFailure` instead of `AssertionError`/`TimeoutError`: instance vanished, status `missing`/`exited` while booting or waiting for ssh, not running after 900 s, ssh still failing after 300 s, `uv sync` over `max_sync_minutes`. `main` catches it, appends the host (reason includes instance, machine, GPU, location), and re-raises. Nothing is destroyed or re-rented (Julian: "No restarts"); the instance keeps billing as before unless `--destroy_on_exit`. A non-timeout `uv sync` error is not blacklisted, since it can be our lockfile's fault.
OUTDATED (26-10-06): "Nothing is destroyed" was reversed at Julian's request in entry *destroy on blacklist*: a `HostFailure` now destroys the instance (still no re-rent).
- ssh failures are blacklisted even though a broken local key would fail the same way [concluded: a broken key would show up after the first blacklisted host, and an entry is easy to delete].
- Hand entries: for failures the launcher cannot see (e.g. a slow Hugging Face download later). The file's header comment shows the format. The launcher rewrites the file when it appends, so comments other than the header are lost; notes go in `reason`.
- `--offer_id` now looks the offer up first (`search offers -n ask_contract_id=<id>`; `id=` matches nothing [verified: 26-10-06]; the CLI warns that `ask_contract_id` is an unknown field but filters correctly, and the code asserts exactly that offer came back), and refuses an offer on a blacklisted host.

**Host shown**: `host` column in the offer table; host (and machine) in the "Renting offer" and "Created instance" lines, in the "instance ready" summary, and in `--dry_run`.

**A100 config**: `spd/scripts/vast_a100_config.yaml` (`--config a100`): `A100_SXM4`, `min_gpu_ram: 40` (80 GB variants also match under the price cap), `max_price: 1.0`, disk 60 GB, `min_reliability: 0.98`, `datacenter_only: false`, sort perf-per-dollar.

**Verification** [verified: 26-10-06]: `ruff check`, `ruff format --check`, `basedpyright` clean. `spd-vast --list_offers --config a100` lists 10 offers with host ids, from $0.422/h. On a scratch copy of the blacklist: appending host 513108 wrote valid YAML; the next search had `host_id notin [513108]` and returned none of its offers; `_find_offer` refused an offer on it and accepted one on another host; a duplicate append and a non-existent offer id both failed loudly. `spd-vast --mode provision --config a100 --offer_id 53071394 --dry_run` ran through the lookup. **Not tested**: the `HostFailure` → append path inside a real rental (needs a failing host).

---

C: (transcribed from chat) can you add that the instance get's destroyed when it gets blacklisted

---

26-10-06, Claude Opus 5.5 — *destroy on blacklist*

Implemented: when a `HostFailure` blacklists the host, `main` now also destroys the instance (in the `finally` block, the same path as `--destroy_on_exit`; the CLI's reply is logged). Still no re-rent: the launcher stops with the error, and relaunching is Julian's call. The "Destroy it with: vastai destroy instance …" hints were removed from the `HostFailure` messages since the destroy is automatic; `_sync_dependencies` lost its now-unused `instance_id` parameter. Other failures (ours, e.g. a `uv sync` error that isn't a timeout) still leave the instance running with the destroy hint, as before.

[verified: ruff, basedpyright clean; `--dry_run` runs.] Not tested: the destroy inside a real failing rental. One unknown: for the "instance vanished" failure, vast.ai has already forgotten the rental, and I don't know whether `vastai destroy instance` then exits non-zero. If it does, Python reports the destroy error on top of the original one; both stay visible.
OUTDATED (26-10-06): this destroy hung in the first live test (vastai's hidden confirmation prompt, no `-y`), and `vastai destroy` exits 0 even on a 404; fixed in entry *first live test: boot timeout blacklisted, destroy hung*.

---

26-10-06, Claude Opus 5.5 — *first live test: boot timeout blacklisted, destroy hung*

Julian's live test, transcribed from chat (summary of his pasted console, not verbatim): `spd-vast --mode provision` (default 48 GB 4090 config) rented offer 45473509, instance 54486536 on host 124072 (machine 38442, Japan). The instance stayed `loading` for 15 min. Its `status_msg` was a truncated tail of a DNS error ("…esolve host: cloud.vast.ai"), so the host could not resolve vast.ai's own domain.

What happened [verified: blacklist file and `vastai show instances`, 26-10-06]:
- After `READY_TIMEOUT_S` = 900 s the launcher raised `HostFailure` and appended host 124072 to `vast_blacklist.yaml` ("not running after 900s"). That part worked.
- The destroy did **not** happen. `vastai destroy instance` (CLI 1.2.0) asks for confirmation unless it gets `-y`. `_vastai` captures stdout, so the prompt was invisible and the launcher hung waiting on it. The existing `--destroy_on_exit` had the same bug.
- I destroyed instance 54486536 with `-y` (it was blacklisted, so Julian's "destroy when blacklisted" request covers it); `show instances` was empty afterwards.

Observed CLI behaviour [verified: 26-10-06]: success prints `destroying instance <id>.`; an unknown instance prints `Failed with error 404: ...` to stderr; without `-y` and with stdin closed it prints `Aborted.`. All three exit 0.

Fix: `_vastai` now runs with stdin closed (no invisible prompt can hang it). The new `_destroy_instance` passes `-y`, reads both output streams and asserts that the output is the success line or a 404 (a vanished instance is already gone). Anything else raises with the manual destroy command. The printed destroy hints now include `-y`. [verified: lint/typecheck clean; a real 404 on the destroyed instance accepted; the "Aborted." and success outputs tested with a stubbed subprocess.] The success path against a live instance is still untested in code (only by hand with the CLI).

Possible improvements, not done:
- A `status_msg` with "resolve host" (or similar) could fail at once instead of after 15 min.
- The boot wait prints a line every 10 s even when nothing changed; it could print only changes.

---

26-10-06, Claude Opus 5.5 — *sync: two publickey blacklistings unexplained*; written during a `!sync`, from the blacklist file, `git log` and `run_vast.py`; I did not see the console of these rentals

Commit `1b7e978` ("update vastai blacklist", 26-10-06 16:22) added two hosts this LOG never discusses, both auto-appended by the launcher [verified: `spd/scripts/vast_blacklist.yaml`]:
- host 256524 (instance 54491074, California): "ssh failing after 300s: … Permission denied (publickey)."
- host 33035 (instance 54492767, Romania): the same message.

Two key rejections in a row on different hosts is the pattern that the comment in `_wait_until_ssh_ready` names as the sign of a broken local key ("the first blacklisted host would reveal a broken one"). Nothing in this LOG records a rental since then where ssh accepted the key. So these two entries may blame hosts for our own key or for a failed `vastai attach ssh` [assumed: cannot tell which from the files]. Suggested check before trusting them: run `vastai logs <id>` on the next rejection, or try the key on a fresh rental. If the key turns out to be at fault, delete the two entries.
OUTDATED (26-10-06): the flag is resolved: the hosts were bad, and the next rental accepted the same key; see entry *publickey flag resolved: hosts were bad*.

---

C: (transcribed from chat) no, they were bad. The third one worked

---

26-10-06, Claude Opus 5.5 — *publickey flag resolved: hosts were bad*; continuing the sync session

Julian's answer above settles the flag raised in *sync: two publickey blacklistings unexplained*: hosts 256524 and 33035 were bad, and the third rental after them accepted the same key, so our local key and `vastai attach ssh` work [verified by Julian, chat 26-10-06; I did not see that rental's console]. The two blacklist entries stay. This fits the code comment's logic: a broken key would have failed on the next host too.
