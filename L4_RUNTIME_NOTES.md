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
- while SAM 3 is first loading, frames can continue to update from RT-DETR/Falcon, but `readiness.full_pipeline_ready` and `readiness.sam3_visual_ready` must remain `false`
- after a visible SAM 3 result exists, the next `segmenting` cycle should keep readiness live on commit `60590b1` or newer

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

## April 24 approved SAM 3 VM run

Purpose:
- rerun the L4 service after Hugging Face approved access for SAM 3
- keep the image build directly on the VM, matching the prior proven process
- test whether SAM 3.1 could be used without a larger integration rewrite

Cloud result:
- VM name: `falcon-pipeline-l4`
- project: `tableminder`
- zone: `us-east4-c`
- machine: `g2-standard-8`
- host GPU: `NVIDIA L4`
- Docker GPU probe: passed
- image build: completed on the VM as `falcon-pipeline:l4`
- final VM state: deleted and verified absent along with same-name boot disk

Repository commits made before/during the run:
- `2a658db`: made SAM 3 mandatory for the L4 realtime runtime
- `a763722`: added `SAM3_MODEL_ID` launcher override and documented SAM 3.1 as a compatibility probe
- `60590b1`: kept SAM 3 visual readiness true while the next mask pass is `segmenting`

Source result:
- source URL: `https://www.youtube.com/watch?v=S605ycm0Vlk`
- prompt: `track restaurant goers, servers, and tables. person turns red if looking for service.`
- YouTube cookies were not required
- preflight returned `source.status = "ready"`
- source metadata resolved as Hogs Breath Saloon Key West live content

SAM 3.1 probe:
- `facebook/sam3.1` exposed an authenticated `config.json`, so a basic Hugging Face file probe can pass
- launching the service with `SAM3_MODEL_ID=facebook/sam3.1` did not produce a ready SAM path
- `/api/state` reported SAM as blocked with `error_kind = "model_access"`
- this confirms `facebook/sam3.1` must not become the default until a native package/checkpoint integration is implemented and proven
- keep `facebook/sam3` as the supported runtime model id

Successful SAM 3 run:
- launching with `SAM3_MODEL_ID=facebook/sam3` succeeded
- Falcon loaded on CUDA
- RT-DETR loaded on CUDA
- SAM 3 loaded on CUDA
- `/api/state` reached:
  - `readiness.service_state = "live"`
  - `readiness.models_ready = true`
  - `readiness.capture_connected = true`
  - `readiness.integration_ready = true`
  - `readiness.full_pipeline_ready = true`
  - `readiness.sam3_visual_ready = true`
  - `readiness.blocking_engine_errors = []`
- `result.primary_engine = "sam3"`
- SAM 3 engine returned:
  - `status = "ok"`
  - `model_id = "facebook/sam3"`
  - `device = "cuda"`
  - `detections_count = 21`
  - `num_masks = 21`
  - `prompt_boxes_count = 12`
  - observed mask generation around `0.5` to `0.9` seconds
- `check_realtime_service.sh` passed with `full_pipeline_ready = true`

Runtime issue found:
- `/api/state` could flap between `live` and `degraded`
- the screen stayed SAM-primary, but readiness temporarily became false while `metrics.sam3_segmentation_state = "segmenting"`
- this was a state-contract bug, not a model failure
- commit `60590b1` fixes this by treating a recent visible SAM 3 overlay as ready while the next SAM 3 pass is segmenting
- this fix passed local unit tests but was not re-proven on the VM before teardown

Tunnel notes:
- direct external SSH timed out after VM creation
- `--tunnel-through-iap` worked for VM commands
- Windows `gcloud compute ssh ... -- -L ...` did not work because gcloud parsed `-L` incorrectly in this environment
- the working local tunnel syntax used explicit flags:
  - `--ssh-flag='-L'`
  - `--ssh-flag='127.0.0.1:8080:127.0.0.1:8080'`
  - `--ssh-flag='-N'`
- local `http://127.0.0.1:8080/api/healthz` returned `200`

Operational lessons:
- build long Docker images as detached VM-side jobs when using IAP/Plink, or SSH drops can kill the build during layer export
- if a foreground build disconnects during export, check `sudo docker images` before rebuilding
- if no image exists, run the build with `nohup ... > /tmp/falcon-pipeline-build.log 2>&1 &` and poll the log
- SAM 3 model access is now solved for `facebook/sam3`, but not for `facebook/sam3.1`
- full readiness must be judged from `/api/state`, not only from `/api/frame.jpg`

Hardening needed next:
- prove commit `60590b1` on a fresh VM run and confirm readiness no longer flaps while SAM 3 is segmenting
- update `check_realtime_service.sh` to tolerate `sam3_segmentation_state = "segmenting"` when a recent SAM 3 primary overlay exists
- add a SAM model preflight/probe that distinguishes:
  - no token
  - gated access missing
  - checkpoint available but unsupported by Transformers
  - model files present but runtime load failed
- improve `load_sam3_runtime` error classification so SAM 3.1 unsupported-format failures do not get mislabeled as pure `model_access`
- add a script path for detached VM-side Docker builds and log polling, because this run proved the direct foreground SSH build is brittle during export
- update tunnel docs to prefer IAP plus `--ssh-flag` syntax on this Windows machine
- add a state test for the exact flapping sequence observed live: ready SAM result, worker enters `segmenting`, readiness remains live
- add a smoke assertion that `result.primary_engine = "sam3"` and `result.engines[name=sam3].num_masks > 0`, not only `full_pipeline_ready = true`

## Current local hardening after the approved SAM 3 run

Implemented process hardening:
- `scripts/gcp/build_l4_image.ps1` is now the preferred build command
- the build wrapper clones or fast-forwards `/home/kirin/l4-fork`, starts `docker build` with `nohup`, writes `/tmp/falcon-pipeline-build.log`, and polls until `falcon-pipeline:l4` exists or the build fails
- the wrapper defaults to `--tunnel-through-iap`, matching the working transport on this Windows workstation
- `scripts/gcp/run_realtime_service.sh` now runs source preflight first, SAM model preflight second, then launches the long-running container only if both pass

Implemented SAM 3 model diagnostics:
- `--sam-preflight-only` emits structured JSON before service launch
- exit `22` means no Hugging Face token was configured
- exit `23` means gated/model access failure
- exit `24` means unsupported checkpoint/runtime mismatch
- exit `25` means generic model-load/runtime failure
- `model_unsupported` is now distinct from `model_access` and `model_load`
- `facebook/sam3` remains the supported runtime model id
- `facebook/sam3.1` remains an explicit compatibility probe only

Implemented smoke hardening:
- `check_realtime_service.sh` still accepts structured `source_auth` and `source_unavailable` as useful diagnostic exits
- a successful full run now requires `full_pipeline_ready = true`
- a successful full run also requires `readiness.sam3_visual_ready = true`
- a successful full run requires `result.primary_engine = "sam3"`
- a successful full run requires SAM3 engine `status = "ok"` and `num_masks > 0`
- `segmenting` is accepted only when there is a recent prior SAM3 primary overlay
- optional `CHECK_RESTAURANT_CONTRACT=1` verifies restaurant person entities include `role_reason`, `role_confidence`, `classification_source`, and `near_guest_context`

Next VM acceptance should prove:
- default `SAM3_MODEL_ID=facebook/sam3` reaches `full_pipeline_ready = true`
- readiness does not flap while SAM3 alternates between `ready` and `segmenting`
- `check_realtime_service.sh` passes without weakening the SAM3 visible-primary checks
- `SAM3_MODEL_ID=facebook/sam3.1` fails cleanly as `model_unsupported` or `model_access`, not a generic SAM error

## April 24 SAM3 source-switch, UI, and latency VM run

Purpose:
- validate the hardened SAM3-required runtime on a live L4 VM
- make SAM3 the visible screen output while keeping RT-DETR boxes hidden
- prove prompt updates and source URL changes through the browser UI
- collect latency evidence before deciding whether 10 FPS needs a larger GPU or a different runtime structure

Cloud path:
- VM name: `falcon-pipeline-l4`
- project: `tableminder`
- zone: `us-east4-c`
- machine: `g2-standard-8`
- GPU: one `NVIDIA L4`
- access path: SSH tunnel only through local `127.0.0.1:8080`
- VM checkout at start of the run: `/home/kirin/l4-fork` was still on commit `83c8e28`
- local repo commits later produced and pushed during the run:
  - `ebe3a47`: render SAM3 overlay without detector boxes
  - `b435f76`: poll `/api/frame.jpg` in the UI instead of holding an MJPEG connection open
- next VM run must build from `origin/main` commit `b435f76` or newer; do not rely on the hot-patched container from this run

Sources used:
- Hogs Breath stream: `https://www.youtube.com/watch?v=S605ycm0Vlk`
- Jimmy's Fish House source-switch test: `https://www.youtube.com/live/9c1oLjB3wIs?si=7Xmt2xcyhWaXCOOV`
- restaurant prompt: `track restaurant goers, servers, and tables. person turns red if looking for service.`
- YouTube cookies were not required for either observed source in this run
- after switching to the Jimmy's source, direct `/api/state` on the VM showed `source.status = "ready"` and the service stayed live

Runtime edits and hot patches applied during the run:
- `falcon_pipeline_realtime_service.py` was updated so the visible overlay no longer draws RT-DETR or Falcon rectangles
- RT-DETR and Falcon detections remain internal guidance for SAM3 and restaurant annotations
- visible people are represented by SAM3 mask colors instead of detector boxes
- restaurant role colors now drive the visible mask tint:
  - red for `needs_service`
  - blue for `restaurant_goer`
  - amber for `server`
  - neutral gray for unclassified or unsupported entities
- unmatched role entities can receive small color markers, but the detector rectangles stay hidden
- tests were added or updated to cover hidden detector boxes, role-colored SAM3 masks, prompt editing, and frame-polling UI behavior
- the running VM/container was hot-patched with newer files via copy/restart during diagnosis; this is not the preferred future path

Prompt and session-update issue:
- symptom: the natural-language prompt field could not be edited reliably in the browser
- root cause: the UI state poll refreshed the input values while the operator was typing
- fix: `falcon_pipeline_realtime_ui.html` now tracks pending local edits and does not overwrite dirty form fields until after a successful session update
- proof: after the fix, `/api/session` accepted the restaurant prompt and `/api/state` reflected it without restarting the service

Source-switch and local stream issue:
- symptom: changing the stream URL made the local page appear broken and the user could not see the live stream
- direct VM evidence showed the Jimmy's source was live and `source.status = "ready"`
- root cause: the local Windows PuTTY/IAP tunnel was wedged by long-lived `/api/stream.mjpg` and stale browser requests, not by YouTube or SAM3
- fix: the UI now polls `/api/frame.jpg?t=...` instead of using `/api/stream.mjpg` as the default visual transport
- additional UI fix: state polling was slowed and overlapping frame/state requests are skipped
- operator rule: if a source switch looks broken, check direct VM `/api/state` first; if the VM is live but local browser requests time out, restart the local tunnel and confirm the UI is on commit `b435f76` or newer
- `/api/stream.mjpg` can remain available for diagnostics, but `/api/frame.jpg` polling is the safer default over this Windows IAP/PuTTY tunnel

Final observed live state before teardown:
- `readiness.service_state = "live"`
- `readiness.full_pipeline_ready = true`
- `readiness.sam3_visual_ready = true`
- `source.status = "ready"`
- `source.input_url = "https://www.youtube.com/live/9c1oLjB3wIs?si=7Xmt2xcyhWaXCOOV"`
- `result.primary_engine = "sam3"`
- `result.num_masks = 6` in the final local-tunnel snapshot
- `metrics.pipeline_state = "running"`
- `metrics.processed_fps = 0.47` in the final local-tunnel snapshot
- `metrics.falcon_guidance_generation_seconds = 5.67` in the final local-tunnel snapshot
- `metrics.sam3_segmentation_generation_seconds = 0.50` in the final local-tunnel snapshot
- `result.frame_generation_seconds = 0.44` in the final local-tunnel snapshot

Earlier latency samples in the same run:
- processed FPS varied around `0.42` to `0.50`
- SAM3 segmentation generation varied around `0.50` to `0.90` seconds
- Falcon guidance generation varied around `4.2` to `8.0` seconds
- L4 VRAM use was about `5.3 GB / 23.0 GB`
- GPU utilization was bursty rather than continuously saturated
- examples observed during polling included low or idle utilization between inference bursts, with occasional higher spikes

Latency interpretation:
- the current implementation is not a 10 FPS realtime video pipeline
- the L4 GPU is not the only bottleneck because the GPU is not continuously saturated
- wall time is dominated by pipeline structure: capture, RT-DETR guidance, SAM3 segmentation, Falcon guidance, overlay generation, JSON state, and browser delivery are coupled too tightly
- Falcon is the slowest semantic component; even at a five-second refresh, it can dominate the background pipeline
- SAM3 can produce visible masks on the L4, but full-frame SAM3 refreshes are still far below 10 FPS
- increasing GPU size alone is unlikely to create 10 FPS unless the render path is decoupled from heavy inference

Accuracy shortcomings observed:
- SAM3 masks are visible and useful, but they are instance masks, not restaurant roles by themselves
- role coloring is still derived from heuristic classification layered on top of Falcon/RT-DETR/SAM evidence
- static frames do not reliably prove `looking for service`; that needs temporal behavior, posture, hand/face direction, or repeated attention cues
- table detection remains unreliable on some livestream angles
- when Falcon returns full-frame prompt guidance or RT-DETR misses guest-context objects, the conservative classifier may leave people `unclassified`
- forcing every person into a role would be misleading; unsupported people should stay `unclassified` until there is stronger evidence

What worked:
- source ingest worked for both tested YouTube live URLs without cookies
- Falcon loaded and produced guidance
- RT-DETR loaded and supplied prompt boxes
- SAM3 loaded with the approved Hugging Face token and became the visible primary engine
- the service reached `full_pipeline_ready = true`
- `/api/frame.jpg` returned live SAM3 overlay frames
- prompt updates worked after the UI dirty-field fix
- source changes worked on the VM side after checking `/api/state`

What did not meet the product target:
- the display was not close to 10 FPS
- the classification layer was not accurate enough to be trusted as restaurant operations intelligence
- source switches could still confuse the operator when the local tunnel was stale
- the VM had to be hot-patched during the run instead of rebuilding from current `origin/main`

Next-run requirement:
- start by building the image fresh from `origin/main` at `b435f76` or newer
- keep RT-DETR/Falcon boxes hidden; the user should see only SAM3 color output and minimal markers
- treat `/api/frame.jpg` polling as the primary UI delivery mechanism
- use `/api/state` as the source of truth for source-switch diagnosis
- focus implementation work on decoupling display FPS from AI inference FPS and improving restaurant role evidence
- do not spend the next run chasing a larger GPU until the pipeline is profiled and the render/inference loops are separated

Teardown:
- requested after notes
- `delete_l4_vm.ps1` hit a transient `gcloud` `RemoteDisconnected` after submitting the delete request
- follow-up `gcloud compute instances describe falcon-pipeline-l4 --project=tableminder --zone=us-east4-c` returned resource not found
- follow-up `gcloud compute disks describe falcon-pipeline-l4 --project=tableminder --zone=us-east4-c` returned HTTP `404` resource not found
- local port `8080` had no `LISTENING` tunnel left after teardown, only closed `TIME_WAIT` sockets and one browser retry
