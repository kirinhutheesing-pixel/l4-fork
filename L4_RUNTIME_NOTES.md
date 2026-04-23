# L4 Runtime Notes

These notes capture the actual runtime behavior observed during the April 22, 2026 L4 bring-up. This file is for hardening work, not just operator steps.

## Purpose

The main runbook in `FALCON_PIPELINE_L4.md` is the clean operator path.

This file records:
- what actually worked
- what actually failed
- what the failure meant
- what should be hardened next

## Proven setup snapshot

Observed working cloud configuration:
- account: `kirin@lifelineus.com`
- project: `tableminder`
- instance: `falcon-pipeline-l4`
- successful zone: `us-east4-c`
- machine type: `g2-standard-8`
- image: Ubuntu `22.04`
- GPU: `1 x NVIDIA L4`

Observed capacity failures before success:
- `us-central1-a`
- `us-central1-b`
- `us-central1-c`
- `us-east4-a`

Interpretation:
- the runtime design was not the blocker
- zonal capacity was the blocker
- retrying a different L4-capable zone was the correct move

## Host bootstrap notes

The VM startup script successfully handled:
- Docker install
- NVIDIA driver install
- NVIDIA container toolkit install
- Docker runtime configuration for NVIDIA
- persistent host paths under `/opt/falcon-pipeline`

The right bootstrap signals were:
- `/var/lib/falcon-pipeline/bootstrap.ready`
- `/var/log/falcon-pipeline-bootstrap.log`
- host `nvidia-smi`
- container `nvidia-smi` through `sudo docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi`

Conclusion:
- these four checks are enough for first host validation
- broader host debugging is unnecessary unless one of those fails

## Windows gcloud notes

On this workstation, `gcloud` uses PuTTY tools for VM transport:
- SSH is `plink`
- SCP is `pscp`

That caused real friction:
- OpenSSH `-o ...` flags failed
- `pscp` rejected OpenSSH-only flags
- `~` was not a safe remote destination

Working rules:
- use `gcloud compute ssh ... --strict-host-key-checking=no`
- use absolute remote paths for `gcloud compute scp`
- verify the remote path with `pwd` and `ls -la` before copying if anything looks off

This is not a repo bug. It is an operator-environment behavior on this Windows machine.

## Docker runtime notes

The image build worked on the L4 VM.

Important runtime detail:
- the image uses `CMD`
- it does not use `ENTRYPOINT`

Meaning:
- `docker run falcon-pipeline:l4 --source-url ...` is wrong
- Docker replaces the whole `CMD` with `--source-url ...`
- the correct form when passing args is:
  - `falcon-pipeline:l4 /app/.venv/bin/python /app/falcon_pipeline_realtime_service.py ...`

Hardening implication:
- either keep using the explicit Python command form in docs
- or convert the image to a true `ENTRYPOINT` later

## Falcon runtime notes

Original heavy-path issue:
- Falcon still triggered compile-time behavior even when the service was launched with `--no-compile`

Cause:
- `falcon_perception/attention.py` compiled FlexAttention helpers at import time
- the `--no-compile` switch only disabled model-level compile
- attention helper compile could still fail later on the VM with compiler-related errors

Current fix:
- attention compile is now lazy
- attention compile can be disabled via the runtime compile switch
- recoverable compile failures fall back to eager execution

Effect:
- `--no-compile` now actually matters for the heavy path
- compile-related failures should degrade more safely instead of poisoning the whole run

## API/runtime contract notes

The realtime contract that mattered most in practice was:
- `GET /api/healthz`
- `GET /api/state`
- `GET /api/frame.jpg`
- `POST /api/session`

Important clarification:
- `healthz` is only process liveness
- `state` is the real operational truth surface

The state fields that proved most useful were:
- `readiness.service_state`
- `readiness.integration_ready`
- `frame.state`
- `frame.ready`
- `metrics.capture_state`
- `metrics.pipeline_state`
- `metrics.last_error`
- `model_status.*`

Operational rule:
- do not trust `frame.jpg` alone to decide readiness
- always pair it with `/api/state`

## Session endpoint notes

Observed issue:
- `POST /api/session` originally behaved as if `request` were a missing query parameter

Cause:
- the `Request` annotation was imported inside `create_app()` under postponed annotations
- FastAPI did not resolve it the way the endpoint expected

Current fix:
- `Request` is imported at module scope
- the JSON session update test now passes

## Source-ingest notes

The current high-level ingest path is fine:
- source URL resolves through `yt-dlp` if it is YouTube
- OpenCV reads the resolved media URL
- the capture loop keeps only the freshest frame

That is the correct live architecture for this project size.

The real blocker moved to source policy:
- the specific YouTube stream `https://www.youtube.com/watch?v=S605ycm0Vlk` returned:
  - `Sign in to confirm you’re not a bot`

What this means:
- the service was healthy
- Falcon and RT-DETR loaded
- the source itself could not be resolved without authentication

This is not a Falcon bug and not an RT-DETR bug.

## YouTube authentication notes

The correct design adjustment was to support a real cookie file:
- `--yt-cookies-file`
- `YTDLP_COOKIES_FILE`

Why this matters:
- browser-cookie scraping on Windows was unreliable on this machine
- local Chrome and Edge cookie extraction failed because the cookie database could not be copied in that state
- relying on `--cookies-from-browser` at runtime is too fragile for VM bring-up

Current expected path for bot-gated streams:
1. export Netscape-format `cookies.txt` from a signed-in browser
2. copy it to the VM host
3. mount it into the container
4. launch the realtime service with `--yt-cookies-file`

Hardening implication:
- cookie-backed YouTube is now a first-class runtime path and should stay documented that way

## Restaurant-mode notes

The current restaurant scene logic is intentionally simple:
- RT-DETR candidate detections classify `person` and `table`
- people near tables are treated as restaurant goers
- people away from tables are treated as servers
- Falcon overlap is used to mark likely service-seeking people
- service-seeking people turn red in the overlay

Important limitation:
- this is heuristic scene logic, not a trained restaurant-role model
- it is good enough for first live proof, not yet a robust production taxonomy

That tradeoff is acceptable for milestone 1.

## Known-good verification sequence

The shortest reliable proof sequence was:
1. verify host GPU
2. verify Docker GPU
3. build image
4. launch service with explicit Python command
5. inspect `/api/healthz`
6. inspect `/api/state`
7. only then trust `/api/frame.jpg`

If step 6 says the source is blocked, do not keep debugging the model runtime.

## Hardening backlog

P0:
- convert the container from `CMD`-only behavior to a real `ENTRYPOINT` so runtime args are less error-prone
- add a source preflight check that returns an explicit source-resolution error before the full pipeline starts
- expose the configured cookie-file path and source-resolution mode in `/api/state`
- add a documented method for exporting YouTube `cookies.txt`

P1:
- add a small VM-side smoke script that checks:
  - `healthz`
  - `state`
  - `frame.jpg`
  - current Docker logs
- add container log guidance to the main runbook
- optionally include a JS runtime in the image if `yt-dlp` stability becomes a recurring problem

P2:
- consider a cleaner restaurant scene classifier if role accuracy matters beyond first demo quality
- consider a true source-switch integration test for `/api/session`
- consider auto-tagging `state` with a clearer cause group such as:
  - `source_auth`
  - `source_unavailable`
  - `model_load`
  - `inference_runtime`

## What not to debug next time

Do not burn time on these until the basic checks say they matter:
- SAM 3, if the target run only needs boxes
- UI issues, if `/api/state` already says the source is blocked
- Falcon compile internals, if the actual error is a YouTube bot-check
- general VM provisioning changes, if the host GPU and Docker GPU checks already pass

The next setup should be treated as:
- host prove-up
- service prove-up
- source prove-up

in that exact order
