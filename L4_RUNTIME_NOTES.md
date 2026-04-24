# L4 Runtime Notes

These notes capture the actual runtime behavior observed during the April 22-23, 2026 L4 bring-up. This file is for hardening work, not just operator steps.

## Purpose

The main runbook in `FALCON_PIPELINE_L4.md` is the clean operator path.

This file records:
- what actually worked
- what actually failed
- what the failure meant
- what should be hardened next

## Proven setup snapshot

Observed working cloud configuration across the last two runs:
- account: `kirin@lifelineus.com`
- project: `tableminder`
- instance: `falcon-pipeline-l4`
- successful zones:
  - `us-east4-c` on 2026-04-22
  - `us-east4-c` on 2026-04-24
  - `us-west4-c` on 2026-04-23
- machine type: `g2-standard-8`
- image: Ubuntu `22.04`
- GPU: `1 x NVIDIA L4`

Observed capacity failures before success:
- 2026-04-22:
  - `us-central1-a`
  - `us-central1-b`
  - `us-central1-c`
  - `us-east4-a`
- 2026-04-23:
  - `us-east4-c`
  - `us-east1-c`
  - `us-east1-b`

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

The image build worked on the L4 VM in both runs.

Current runtime detail:
- the image now uses a real `ENTRYPOINT`
- `run_realtime_service.sh` still forces the explicit Python entrypoint for clarity and to avoid operator drift

Important April 23 runtime finding:
- a live service plus a real frame was not enough to prove Falcon itself was healthy
- RT-DETR kept the service live while Falcon failed deeper in Triton helper compilation

Current hardening:
- `check_realtime_service.sh` now fails if `readiness.full_pipeline_ready` is not `true`
- the smoke summary now surfaces `readiness.blocking_engine_errors`
- a live frame no longer counts as success when an enabled engine is still in `error`

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

April 23 follow-on blocker chain:
- first Falcon runtime error:
  - `Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.`
- after adding compiler tooling:
  - `Python.h: No such file or directory`

What that means:
- the blocker is still packaging-level, not model logic
- the image needs:
  - `build-essential`
  - `python3-dev`
  - `CC=/usr/bin/gcc`
  - `CXX=/usr/bin/g++`

Current repo status:
- those image changes are now present in the repo-local `Dockerfile.gcp-l4`
- they were re-verified on the April 24 VM run
- Falcon loaded and ran on the L4 image after those packaging fixes landed

Operational rule:
- if Falcon errors mention compiler tooling or `Python.h`, do not debug YouTube, RT-DETR, or SAM3
- fix the image and rebuild first

April 24 clarification:
- Falcon was no longer the failing engine
- Falcon produced one scene-wide prompt echo box instead of useful person-level grounding
- the remaining live blockers moved up-stack:
  - gated SAM 3 access
  - weak restaurant-role grounding on this stream

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
- `readiness.full_pipeline_ready`
- `readiness.blocking_engine_errors`
- `frame.state`
- `frame.ready`
- `metrics.capture_state`
- `metrics.pipeline_state`
- `metrics.last_error`
- `model_status.*`

Operational rule:
- do not trust `frame.jpg` alone to decide readiness
- always pair it with `/api/state`

April 23 clarification from the live run:
- `readiness.service_state = "live"` plus `frame.ready = true` can still coexist with `result.engines[].status = "error"` for `falcon`
- for full-pipeline validation, `result.engines[]` is the final truth surface

April 24 clarification from the live run:
- `frame.ready = true` plus `capture_state = running` can still coexist with `readiness.service_state = "warming"` and `full_pipeline_ready = false`
- on that run, the blocker moved from Falcon to `sam3`
- the correct read was:
  - source healthy
  - Falcon healthy
  - RT-DETR healthy
  - SAM 3 blocked by model access

Current repo-local fix:
- the service now reports `readiness.service_state = "degraded"` when a live frame exists but an enabled engine still reports `status = "error"`
- `readiness.full_pipeline_ready` stays `false` until the enabled engines are healthy
- `readiness.blocking_engine_errors` exposes the engine names and reasons that are still blocking a full success state

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

April 23 source result:
- the specific YouTube stream `https://www.youtube.com/watch?v=S605ycm0Vlk` preflighted successfully
- `source.status = ready`
- no cookie file was required
- the resolved stream title was `Hogs Breath Mini Bar Cam ...`

What this means:
- YouTube auth is not a universal blocker
- preflight should always be attempted before exporting cookies
- a public video ID can be directly usable even if earlier runs hit a bot-check on the same source class

## YouTube authentication notes

The correct design adjustment was to support a real cookie file:
- `--yt-cookies-file`
- `YTDLP_COOKIES_FILE`

Why this matters:
- browser-cookie scraping on Windows was unreliable on this machine
- local Chrome and Edge cookie extraction failed because the cookie database could not be copied in that state
- relying on `--cookies-from-browser` at runtime is too fragile for VM bring-up

Updated rule after April 23:
- cookies are a fallback path, not the default path
- only introduce `cookies.txt` when preflight or `/api/state` says `source_auth`

Current expected path for bot-gated streams:
1. export Netscape-format `cookies.txt` from a signed-in browser
2. copy it to the VM host
3. mount it into the container
4. launch the realtime service with `--yt-cookies-file`

Hardening implication:
- cookie-backed YouTube is now a first-class runtime path and should stay documented that way

## April 23 live run results

What worked end to end:
- VM creation completed in `us-west4-c`
- host GPU and Docker GPU checks passed
- the image built on the VM
- source preflight for `https://www.youtube.com/watch?v=S605ycm0Vlk` returned `ready`
- the running service reported:
  - `readiness.service_state = live`
  - `source.status = ready`
  - `frame.ready = true`
- `/api/frame.jpg` returned a real JPEG around `560 KB`
- RT-DETR produced live detections at roughly `0.02s` generation time per processed frame

What did not work end to end:
- Falcon guidance still errored in Triton helper compilation
- the restaurant-scene heuristics on this stream produced:
  - `tables = 0`
  - `restaurant_goers = 0`
  - `servers > 0`

Interpretation:
- the heavy path is partially proven:
  - source ingest works
  - live frame generation works
  - RT-DETR works
- Falcon itself is not yet fully proven on the L4 image
- the restaurant-role layer is still too weak to trust on this stream

## Source-state contract notes

Observed issue:
- the service could be actively streaming, but `/api/state` still showed `source.status = "resolving"`

Cause:
- `_load_models()` could overwrite a correct capture-derived source state after the stream was already connected

Current fix:
- source state now reuses `stream_info` and current capture state
- once the stream is running, `source.status` stays `ready`

This bug is fixed locally in `falcon_pipeline_realtime_service.py`.

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

April 23 evidence:
- on the Hogs Breath stream, no tables were recognized
- that caused all people to collapse into the `server` bucket
- no `needs_service` overlays were triggered

Current repo-local fix:
- when no tables are detected, unmatched people no longer default to `server`
- the fallback role is now `unclassified`

Hardening implication:
- the restaurant profile is still not reliable enough for the target use case
- the current fix avoids a misleading false taxonomy, but it does not solve table-detection weakness

## April 24 live run snapshot

Observed working cloud configuration on the latest run:
- account: `kirin@lifelineus.com`
- project: `tableminder`
- instance: `falcon-pipeline-l4`
- successful zone: `us-east4-c`
- source: `https://www.youtube.com/watch?v=S605ycm0Vlk`
- source result: `source.status = ready`
- cookies required: no

Observed live session values:
- `task = segmentation`
- `enable_sam3 = true`
- `falcon_refresh_seconds = 5.0`
- `max_dim = 960`
- `falcon_max_dim = 640`

Observed runtime state from `/api/state`:
- `metrics.capture_state = running`
- `metrics.pipeline_state = running`
- `frame.ready = true`
- `readiness.service_state = warming`
- `readiness.full_pipeline_ready = false`
- `readiness.error_kind = model_load`
- `readiness.blocking_engine_errors = [{"name":"sam3", ...}]`

Observed latency split:
- `processed_fps = 0.63`
- end-to-end `latest_generation_seconds = 3.286`
- Falcon `generation_seconds = 3.263`
- RT-DETR `generation_seconds = 0.023`

Interpretation:
- the L4 itself was not the latency bottleneck
- Falcon generation dominated wall time
- RT-DETR was effectively negligible relative to Falcon
- turning Falcon refresh from `2.0` to `5.0` reduced refresh frequency, but per-refresh cost was still about `3.3s`

## Current local latency hardening after the April 24 run

Runtime change:
- Falcon no longer runs inline on every processed frame
- Falcon now refreshes in a background guidance loop, defaulting to every `5.0` seconds
- RT-DETR remains the fast per-frame detector/tracker path
- SAM 3 has its own background segmentation loop and is now mandatory

SAM 3 visual behavior:
- `task=segmentation` and SAM 3 loading are now fixed runtime behavior, not operator options
- RT-DETR/Falcon boxes are queued as SAM 3 prompts
- once SAM 3 produces masks, `result.primary_engine = "sam3"`
- `/api/frame.jpg` then draws SAM 3 masks as the primary overlay
- while SAM 3 is still loading or segmenting, frames can continue to update from RT-DETR/Falcon, but `readiness.full_pipeline_ready` and `readiness.sam3_visual_ready` must remain `false`

Diagnostics added for future agents:
- `metrics.falcon_guidance_state`
- `metrics.falcon_guidance_age_seconds`
- `metrics.falcon_guidance_generation_seconds`
- `metrics.sam3_segmentation_state`
- `metrics.sam3_segmentation_age_seconds`
- `metrics.sam3_segmentation_generation_seconds`
- `result.frame_generation_seconds`
- `result.sam3_segmentation_frame_id`

Operational consequence:
- judge screen latency from `result.frame_generation_seconds`, not stale Falcon `generation_seconds`
- judge SAM 3 visibility from `result.primary_engine = "sam3"` and mask age, not from the browser being open
- if the state shows `model_access`, stop; the remaining blocker is Hugging Face access to `facebook/sam3`

## SAM 3 notes from the April 24 run

What happened:
- the service was relaunched in segmentation mode with SAM 3 required
- the stream still connected
- frames still rendered
- the service did not become fully ready because SAM 3 failed during model load

Direct proof:
- a direct model probe inside the running container returned a Hugging Face gated-repo failure on `facebook/sam3`
- the actual failure was a `401 Unauthorized` while trying to resolve `processor_config.json`

Interpretation:
- this is not a CUDA problem
- this is not a YouTube problem
- this is not an OpenCV capture problem
- this is a model-access problem

Operational rule:
- if `/api/state` shows access approval or authentication errors for `facebook/sam3`, stop debugging the rest of the pipeline
- supply an `HF_TOKEN` that already has approved access; SAM 3 is no longer optional in this service

Important clarification:
- YouTube cookies do nothing for this failure
- `cookies.txt` is only for source ingest
- `HF_TOKEN` is the relevant credential for gated Hugging Face checkpoints

## Classification notes from the April 24 run

Observed scene output:
- `tables = 0`
- `restaurant_goers = 0`
- `servers = 0`
- `unclassified = 16`
- `needs_service = 0`

Observed RT-DETR support objects:
- `cup`
- `bottle`
- `handbag`
- `backpack`
- `suitcase`

Observed Falcon guidance:
- Falcon returned one prompt echo detection covering the full frame:
  - `center = (0.5, 0.5)`
  - `width = 1.0`
  - `height = 1.0`

Interpretation:
- Falcon was alive, but its output was not useful as role grounding for this compound restaurant prompt
- RT-DETR found people reliably, but it did not find any tables
- without tables and without focused Falcon guidance, the current role heuristics had no safe basis for calling anyone a guest, server, or service seeker
- `unclassified` was therefore the honest output, not a runtime failure

What this means for the next offline fix:
- restaurant roles need a better fallback than table-only grounding
- broad full-frame Falcon detections should not be treated as evidence that everyone needs service
- if Falcon or SAM 3 is going to drive role assignment, the logic needs to filter for focused person-level guidance, not prompt-wide scene boxes
- the current compound prompt is also likely too broad for reliable grounding

Current local hardening:
- full-frame guidance boxes are ignored for service-seeker classification
- focused Falcon or SAM 3 detections can mark a person as `needs_service`
- RT-DETR guest-context objects such as `cup`, `bottle`, `plate`, `bowl`, `wine glass`, `chair`, and `dining table` can classify nearby people as `restaurant_goer`
- each person annotation includes `role_reason`, `role_confidence`, `classification_source`, and `near_guest_context`

## Agent accessibility notes

Use `scripts/gcp/status_l4_vm.ps1` before starting a new VM debugging pass.

It is read-only and reports:
- local `gcloud` account/project
- target VM and boot disk status
- host GPU and Docker GPU checks
- container state
- `/api/healthz`
- summarized `/api/state`
- recent container logs

This should be the first command after setting `PROJECT_ID`, `ZONE`, `VM_NAME`, and `CLOUDSDK_CONFIG`.

## Known-good verification sequence

The shortest reliable proof sequence was:
1. verify host GPU
2. verify Docker GPU
3. build image
4. launch service with preflight
5. inspect `/api/healthz`
6. inspect `/api/state`
7. inspect `result.engines[]`
8. only then trust `/api/frame.jpg` as a full pipeline proof

If step 6 says the source is blocked, do not keep debugging the model runtime.
If step 7 says Falcon is in `error`, do not treat a live frame as success.

## Hardening backlog

P0:
- keep and verify the image dependency fixes on the next VM run:
  - `build-essential`
  - `python3-dev`
  - `CC=/usr/bin/gcc`
  - `CXX=/usr/bin/g++`
- verify `model_status.sam3.error_kind = model_access` when an approved Hugging Face token is missing or rejected
- verify restaurant annotations show role explanations instead of unexplained `unclassified` output

P1:
- add a more explicit operator check for `result.engines[*].reason`
- add container log guidance to the main runbook for compiler and Python-header failures
- narrow the restaurant prompt and/or session model so Falcon stops returning full-frame guidance for multi-target prompts
- evaluate the current RT-DETR guest-context anchor heuristic against more restaurant streams

P2:
- consider a true source-switch integration test for `/api/session`
- consider enabling SAM3 only after Falcon+RT-DETR are fully proven on the L4 image

## What not to debug next time

Do not burn time on these until the basic checks say they matter:
- UI polish, if `/api/state` says SAM 3 is blocked by model access
- UI issues, if `/api/state` already says the source is blocked
- YouTube cookies, if preflight already says `source.status = ready`
- YouTube cookies, if the actual blocker is a gated Hugging Face checkpoint
- general VM provisioning changes, if the host GPU and Docker GPU checks already pass

The next setup should be treated as:
- host prove-up
- service prove-up
- source prove-up
- engine prove-up

in that exact order
