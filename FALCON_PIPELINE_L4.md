# Falcon Pipeline Realtime on Google Cloud L4

This is the primary heavy-runtime path for Falcon Pipeline.

The first milestone is intentionally narrow:
- one `g2-standard-8` VM with one `NVIDIA L4`
- Ubuntu `22.04`
- Docker as the runtime
- one public YouTube live stream
- `Falcon-Perception-300M` + `RT-DETR`
- `SAM 3` always loaded and required
- private access through an SSH tunnel only

The first integration surface for other programs is:
- `GET /api/frame.jpg` for the latest overlay frame
- `GET /api/state` for readiness and error state

## Runtime profile

The intended default runtime is:
- Falcon model: `tiiuae/Falcon-Perception-300M`
- Task: `segmentation`
- RT-DETR: enabled
- SAM 3: enabled and required
- SAM model id: `facebook/sam3` by default
- `max_dim=960`
- `falcon_refresh_seconds=5.0`

## SAM 3 realtime concept

The supported bring-up is now the SAM 3 segmentation view:
- RT-DETR runs on live frames and supplies fast person/object boxes
- Falcon runs in a background guidance loop and refreshes slower natural-language context
- SAM 3 is always loaded by the service and is not an operator toggle
- when SAM 3 returns masks, `primary_engine` becomes `sam3` and `/api/frame.jpg` shows the SAM 3 mask overlay
- while SAM 3 is first loading, the service can still show RT-DETR/Falcon-backed frames, but `/api/state` keeps `full_pipeline_ready=false` until SAM 3 is healthy and visibly primary
- after a visible SAM 3 result exists, the next `segmenting` cycle should not drop readiness on commit `60590b1` or newer

SAM 3.1 note: `facebook/sam3.1` is a gated checkpoint repo, but the official Hugging Face model card says it has no Transformers integration. The current service uses the Transformers `Sam3Model` / `Sam3Processor` path, so `facebook/sam3` remains the default. Use `SAM3_MODEL_ID=facebook/sam3.1` only as an explicit compatibility probe until the service has a native `facebookresearch/sam3` package integration.

Use `/api/state` to confirm the active visual path:
- `result.primary_engine = "sam3"` means the visible overlay is currently SAM 3-driven
- `metrics.sam3_segmentation_state = "ready"` means a SAM 3 result is available
- `readiness.sam3_visual_ready = true` means the integration contract sees SAM 3 as the visible output
- `result.sam3_segmentation_age_seconds` tells how stale the visible SAM 3 mask is
- `readiness.blocking_engine_errors` should be empty for a fully ready SAM 3 run

## Proven values on 2026-04-22 through 2026-04-24

These are not product requirements, but they are the last known operator values observed during live L4 bring-up:
- local account: `kirin@lifelineus.com`
- project: `tableminder`
- instance name: `falcon-pipeline-l4`
- successful allocation zones:
  - `us-east4-c` on 2026-04-22
  - `us-east4-c` on 2026-04-24
  - `us-west4-c` on 2026-04-23
- capacity note:
  - `us-east4-c`, `us-east1-c`, and `us-east1-b` were exhausted on 2026-04-23
  - `us-central1-a`, `us-central1-b`, `us-central1-c`, and `us-east4-a` were exhausted on the prior bring-up
- live source note:
  - `https://www.youtube.com/watch?v=S605ycm0Vlk` preflighted as `source.status = "ready"` without cookies on 2026-04-23
  - the same stream still resolved without cookies on 2026-04-24 during a segmentation-mode run
- SAM 3 note:
  - launching SAM 3 without approved Hugging Face access to `facebook/sam3` left the service streaming but not fully ready
  - the live blocker was model auth, not CUDA, OpenCV, or YouTube
  - after Hugging Face approval, `facebook/sam3` loaded on the L4 and produced CUDA masks as the visible primary engine
  - `facebook/sam3.1` was probed but did not produce a ready Transformers-backed runtime; keep it as a compatibility probe only

If the preferred zone is exhausted again, retry another L4-capable zone before changing the runtime design.

## Files

- `falcon_pipeline_realtime_service.py`: FastAPI realtime service
- `falcon_pipeline_realtime_ui.html`: browser UI
- `Dockerfile.gcp-l4`: container image for the L4 VM
- `requirements-gcp-l4.txt`: Docker and VM Python dependencies
- `scripts/gcp/create_l4_vm.ps1`: zone-retrying VM create + bootstrap/GPU validation
- `scripts/gcp/build_l4_image.ps1`: detached VM-side Docker build + log polling
- `scripts/gcp/delete_l4_vm.ps1`: verified teardown for instance + same-name boot disk
- `scripts/gcp/status_l4_vm.ps1`: read-only VM, GPU, container, API, and log summary
- `scripts/gcp/run_realtime_service.sh`: source preflight + SAM 3 preflight + container launch
- `scripts/gcp/check_realtime_service.sh`: host/Docker/API/SAM-primary smoke check

## Agent quick start

When returning to this repo after a prior run, start with:

```powershell
cd "C:\Users\kirin\OneDrive\Documents\Playground\L4 Fork"
$env:CLOUDSDK_CONFIG = "$PWD\.gcloud-config-l4-lifeline"
$env:PROJECT_ID = "tableminder"
$env:ZONE = "us-east4-c"
$env:VM_NAME = "falcon-pipeline-l4"
pwsh -File .\scripts\gcp\status_l4_vm.ps1 -ProjectId $env:PROJECT_ID -Zone $env:ZONE -VmName $env:VM_NAME
```

That script is read-only. It reports local `gcloud` context, VM/disk status, host GPU, Docker GPU, container state, `/api/healthz`, summarized `/api/state`, and recent container logs.

## 1. Prepare the local gcloud context

Use a repo-local `gcloud` config, not the global default profile. This follows the same isolation pattern used in the separate disposable-rig program.

From the repo root on your local machine:

```powershell
cd "C:\Users\kirin\OneDrive\Documents\Playground\L4 Fork"
pwsh -File .\scripts\gcp\gcloud_setup.ps1 -ProjectId YOUR_PROJECT_ID -AccountEmail you@example.com
```

That script:
- sets `CLOUDSDK_CONFIG` to `.gcloud-config-l4`
- runs `gcloud auth login`
- sets the target project
- runs `gcloud auth application-default login`
- enables:
  - `compute.googleapis.com`
  - `iam.googleapis.com`
  - `serviceusage.googleapis.com`

Set the local variables for this bring-up session:

```powershell
$env:CLOUDSDK_CONFIG = "$PWD\.gcloud-config-l4"
$env:PROJECT_ID = "YOUR_PROJECT_ID"
$env:ZONE = "us-central1-a"
$env:VM_NAME = "falcon-pipeline-l4"
$env:HF_TOKEN = "hf_..."
```

Before creation, confirm that your target project has quota and capacity for one `NVIDIA L4` in the chosen zone.

### Windows gcloud notes

On this Windows machine, `gcloud compute ssh` and `gcloud compute scp` route through PuTTY tooling, not OpenSSH:
- `gcloud compute ssh` uses `plink`
- `gcloud compute scp` uses `pscp`

That changes what works:
- do use `--strict-host-key-checking=no`
- do not pass OpenSSH `-o ...` flags through `gcloud compute ssh`
- do not assume `~` works as a remote destination in `gcloud compute scp`
- do use absolute remote paths such as `/home/kirin/l4-fork/...`
- do not assume OpenSSH-only `scp` flags such as `-T` are supported by `pscp`

## 2. Create the VM

Recommended shape:
- machine type: `g2-standard-8`
- GPU: `1 x NVIDIA L4`
- boot disk: `200 GB`
- image: Ubuntu `22.04 LTS`

Keep the VM private for bring-up. Do not open the app port publicly for the first milestone.

Preferred path from the repo root:

```powershell
pwsh -File .\scripts\gcp\create_l4_vm.ps1 -ProjectId $env:PROJECT_ID -VmName $env:VM_NAME
```

What `create_l4_vm.ps1` does:
- tries the default L4-capable zones in priority order until the first successful allocation
- uploads `bootstrap_g2_ubuntu2204.sh`
- waits for `/var/lib/falcon-pipeline/bootstrap.ready`
- runs host `nvidia-smi`
- runs the Docker GPU probe

If the winning zone differs from your current `$env:ZONE`, update `$env:ZONE` before the later SSH steps.

If the script returns success, host prove-up is complete and broader VM debugging is out of scope.

## 3. Wait for bootstrap and verify the host

`create_l4_vm.ps1` already performs this step. If you need to run it manually:

```powershell
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE
```

On the VM, confirm the bootstrap completed:

```bash
sudo test -f /var/lib/falcon-pipeline/bootstrap.ready
sudo cat /var/log/falcon-pipeline-bootstrap.log | tail -n 50
nvidia-smi
sudo docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi
```

Expected result:
- `nvidia-smi` shows one `NVIDIA L4`
- the Docker probe shows the same GPU inside the container

If `nvidia-smi` is still missing, reboot once and rerun the checks. The Google GPU installer can require that on first boot.

## 4. Clone the repo and build the image

Preferred Windows-side wrapper:

```powershell
pwsh -File .\scripts\gcp\build_l4_image.ps1 -ProjectId $env:PROJECT_ID -Zone $env:ZONE -VmName $env:VM_NAME
```

What the wrapper does:
- clones or fast-forwards `/home/kirin/l4-fork` on the VM
- starts `sudo docker build -f Dockerfile.gcp-l4 -t falcon-pipeline:l4 .` as a detached VM-side job
- writes logs to `/tmp/falcon-pipeline-build.log`
- polls until `falcon-pipeline:l4` exists or the build fails
- uses `--tunnel-through-iap` by default to avoid direct SSH flakiness on this workstation

Manual VM-side equivalent:

```bash
git clone https://github.com/kirinhutheesing-pixel/l4-fork.git
cd l4-fork
nohup sudo docker build -f Dockerfile.gcp-l4 -t falcon-pipeline:l4 . > /tmp/falcon-pipeline-build.log 2>&1 < /dev/null &
tail -f /tmp/falcon-pipeline-build.log
```

Notes:
- model weights are downloaded at runtime into `/opt/falcon-pipeline/hf-cache`
- overlay outputs and artifacts stay under `/opt/falcon-pipeline/outputs`
- neither weights nor outputs are baked into the image
- the image must include `build-essential` and `python3-dev` for Falcon/Triton helper compilation
- if Falcon logs mention `Failed to find C compiler` or `Python.h: No such file or directory`, rebuild from the current repo before debugging anything else
- on Windows/IAP, foreground SSH can drop during Docker layer export; use `build_l4_image.ps1` instead of hand-running a foreground build

## 5. Launch the service

Use a current public YouTube live URL. Channel live pages are usually more stable than one-off video IDs.

Example environment:

```bash
export TEST_SOURCE_URL="https://www.youtube.com/@earthcam/live"
export TEST_PROMPT="people near the center of the frame"
export HF_TOKEN="hf_..."
# Optional compatibility probe only:
# export SAM3_MODEL_ID="facebook/sam3.1"
```

If that channel is not live when you test, replace it with any current public YouTube live URL.

On 2026-04-23 the specific stream `https://www.youtube.com/watch?v=S605ycm0Vlk` resolved cleanly without cookies, so do not assume cookies are always required for public video IDs.

Preferred path on the VM:

```bash
cd /home/kirin/l4-fork
TEST_SOURCE_URL="$TEST_SOURCE_URL" TEST_PROMPT="$TEST_PROMPT" bash scripts/gcp/run_realtime_service.sh
```

What `run_realtime_service.sh` does:
- runs the service image in `--preflight-only` mode first
- exits `20` on source auth failures
- exits `21` on unavailable/unplayable sources
- runs the service image in `--sam-preflight-only` mode second
- exits `22` when no Hugging Face token is present, because SAM 3 is mandatory
- exits `23` when the SAM checkpoint is gated or the token lacks approved access
- exits `24` when the SAM checkpoint/runtime combination is unsupported, including current `facebook/sam3.1` Transformers mismatch cases
- exits `25` on generic SAM model-load/runtime failures
- launches the long-running container only after preflight succeeds
- defaults to `--no-compile`
- mounts the cookie file only when `YTDLP_COOKIES_FILE` is set
- accepts `SAM3_MODEL_ID` for explicit checkpoint compatibility probes

Optional runtime environment toggles:
- `SAM3_MODEL_ID`: defaults to `facebook/sam3`
- `FALCON_REFRESH_SECONDS`: defaults to `5.0`
- `FALCON_MIN_DIM`, `FALCON_MAX_DIM`, `FALCON_MAX_NEW_TOKENS`
- `MIN_DIM`, `MAX_DIM`

Keep `SAM3_MODEL_ID=facebook/sam3` for supported runs. `facebook/sam3.1` is not yet a supported runtime path in this service.

Important:
- YouTube cookies and Hugging Face model access are separate concerns
- `cookies.txt` only fixes bot-gated YouTube ingest
- it does not fix a gated SAM 3 checkpoint
- always provide an `HF_TOKEN` that has approved access to the selected SAM model id

If a YouTube stream is bot-gated, export a Netscape-format `cookies.txt` from a signed-in browser and mount it into the container:

```bash
cd /home/kirin/l4-fork
TEST_SOURCE_URL="$TEST_SOURCE_URL" \
TEST_PROMPT="$TEST_PROMPT" \
YTDLP_COOKIES_FILE="/opt/falcon-pipeline/youtube-cookies.txt" \
bash scripts/gcp/run_realtime_service.sh
```

Use the cookie-backed form only when the public URL path is blocked by YouTube auth checks.

The cookie-backed form is also the correct fallback for streams that fail with:
- `Sign in to confirm you’re not a bot`
- `Use --cookies-from-browser or --cookies for the authentication`

Cookie export and copy instructions are documented in `YOUTUBE_COOKIES.md`.

This launch uses the intended runtime:
- `Falcon-Perception-300M`
- `task=segmentation`
- `RT-DETR` loaded
- `SAM 3` loaded and required

The April 24 live run proved that source ingest, Falcon, and RT-DETR can work while SAM 3 is blocked by Hugging Face access. After this change, that state is treated as a hard runtime blocker, not an optional degradation.

## 6. Tunnel the service locally

From your local machine:

```powershell
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE --tunnel-through-iap --strict-host-key-checking=no --ssh-flag='-L' --ssh-flag='127.0.0.1:8080:127.0.0.1:8080' --ssh-flag='-N'
```

Use only the tunnel during first bring-up. Do not expose port `8080` publicly yet.

Then access:

```text
http://127.0.0.1:8080
```

The browser UI should use `/api/frame.jpg` polling as the default visual path. On the April 24 source-switch run, long-lived `/api/stream.mjpg` requests wedged the local Windows IAP/PuTTY tunnel even though the VM-side service and source were healthy. Keep `/api/stream.mjpg` available as a diagnostic endpoint, but do not treat it as the preferred operator UI transport over this tunnel.

## 7. Verify the runtime contract

### Liveness

```bash
curl -s http://127.0.0.1:8080/api/healthz
```

Expected shape:

```json
{"status":"ok","service":"falcon-pipeline-realtime"}
```

### State

```bash
curl -s http://127.0.0.1:8080/api/state | python3 -m json.tool
```

Watch these fields:
- `readiness.service_state`
- `readiness.integration_ready`
- `readiness.error_kind`
- `readiness.error_message`
- `source.status`
- `source.error_kind`
- `source.error_message`
- `frame.state`
- `frame.is_placeholder`
- `frame.ready`
- `frame.capture_has_frame`
- `model_status.falcon.state`
- `model_status.rt_detr.state`
- `result.engines`
- `result.engines[*].status`
- `result.engines[*].reason`
- `result.engines[*].error_kind`
- `result.scene_annotations.entities[*].role_reason`
- `result.scene_annotations.entities[*].classification_source`
- `rtdetr_available`
- `sam3_available`
- `metrics.capture_state`
- `metrics.pipeline_state`
- `metrics.last_error`

Expected progression:

1. Startup warming:
   - `readiness.service_state = "warming"`
   - `frame.state = "placeholder"`
   - `model_status.falcon.state = "loading"` or `model_status.rt_detr.state = "loading"`

2. Source connecting:
   - `readiness.service_state = "connecting"`
   - `source.status = "resolving"`
   - `frame.state = "placeholder"`
   - `metrics.capture_state = "connecting"` or `"running"`

3. Live:
   - `readiness.service_state = "live"`
   - `readiness.integration_ready = true`
   - `readiness.full_pipeline_ready = true`
   - `readiness.sam3_visual_ready = true`
   - `readiness.blocking_engine_errors = []`
   - `source.status = "ready"`
   - `frame.state = "live"`
   - `frame.is_placeholder = false`
   - `frame.ready = true`
   - `rtdetr_available = true`
   - `sam3_available = true`
   - `result.primary_engine = "sam3"`
   - `result.num_masks > 0`

3a. Degraded live frame:
   - `readiness.service_state = "degraded"`
   - `frame.ready = true`
   - `readiness.full_pipeline_ready = false`
   - `readiness.blocking_engine_errors` is non-empty
   - this means the service is producing a real frame, but at least one enabled engine is still failing

3b. Full SAM 3 proof:
   - `readiness.full_pipeline_ready = true`
   - `readiness.sam3_visual_ready = true`
   - `result.primary_engine = "sam3"`
   - `result.engines[]` contains `falcon` with `status != "error"`
   - `result.engines[]` contains `rt_detr` with `status = "ok"`
   - `result.engines[]` contains `sam3` with `status = "ok"` and `num_masks > 0`
   - `readiness.blocking_engine_errors = []`
   - `metrics.last_error = null`

Important:
- `readiness.service_state = "live"` plus `frame.ready = true` only proves the service is producing a real frame
- it does not prove Falcon guidance is healthy, because RT-DETR can keep the service live on its own
- use `readiness.full_pipeline_ready` as the first pass/fail gate
- use `readiness.sam3_visual_ready`, `result.primary_engine`, and SAM 3 mask count as the visual proof gate
- use `result.engines[*].status` and `result.engines[*].reason` as the final proof surface for engine-specific diagnosis

4. Explicit source block:
   - `source.status = "auth_required"` with `readiness.error_kind = "source_auth"`
   - or `source.status = "unavailable"` with `readiness.error_kind = "source_unavailable"`
   - this is a successful diagnosis, not a model-runtime failure

### Single-frame integration endpoint

```bash
curl -s http://127.0.0.1:8080/api/frame.jpg --output frame.jpg
file frame.jpg
```

Expected result:
- a JPEG file is written successfully

To confirm that the service is returning real live frames instead of the placeholder, fetch it multiple times:

```bash
curl -s http://127.0.0.1:8080/api/frame.jpg --output frame-1.jpg
sleep 2
curl -s http://127.0.0.1:8080/api/frame.jpg --output frame-2.jpg
```

When the service is live, the files should change over time and `/api/state` should report `frame.ready = true`.

### Browser overlay

Open:

```text
http://127.0.0.1:8080
```

The UI should show the overlay and current prompt controls. Current UI builds poll `/api/frame.jpg?t=...` and skip overlapping frame/state requests to avoid tunnel stalls during source switches. `/api/stream.mjpg` remains available for diagnostics, but the first supported browser path is frame polling.

## 8. Update the prompt without restarting

```bash
curl -s -X POST http://127.0.0.1:8080/api/session \
  -H "Content-Type: application/json" \
  -d '{"prompt":"people on the left side of the frame"}' | python3 -m json.tool
```

Expected result:
- `/api/state` reflects the new prompt
- subsequent `frame.jpg` responses and the UI overlay update without restarting the container

## 9. First-line troubleshooting

If the local `gcloud` setup step fails:
- confirm `gcloud` is installed locally
- confirm `CLOUDSDK_CONFIG` points at `.gcloud-config-l4`
- confirm the target project allows `compute.googleapis.com`

If VM creation fails:
- use `create_l4_vm.ps1`
- trust zonal capacity as the first suspect
- only debug provisioning deeper if the create script fails for a reason other than zone exhaustion

If `/var/lib/falcon-pipeline/bootstrap.ready` is missing:
- inspect `/var/log/falcon-pipeline-bootstrap.log`
- reboot once, then re-check `nvidia-smi`
- confirm the startup script actually attached through `--metadata-from-file`

If `docker build` fails:
- confirm you are in the repo root
- confirm `requirements-gcp-l4.txt` is present
- confirm the VM has internet access to pull Python packages and the CUDA base image

If `docker run --gpus all ... nvidia-smi` fails:
- re-check host driver installation
- re-check `nvidia-container-toolkit`
- re-run `sudo nvidia-ctk runtime configure --runtime=docker`

If `/api/healthz` works but `/api/state` stays in `warming`:
- wait for the first model download to finish
- inspect container logs for Hugging Face download or model-load errors
- if you skipped `HF_TOKEN`, retry with it set
- if `model_status.sam3.error_kind = "model_access"`, this is a gated Hugging Face checkpoint, not a stream problem
- if `model_status.sam3.error_kind = "model_unsupported"`, this checkpoint is not usable through the current Transformers SAM 3 path; use `facebook/sam3`

If `/api/state` says `service_state = "degraded"` or `full_pipeline_ready = false` while a real frame exists:
- inspect `readiness.blocking_engine_errors`
- inspect `result.engines[*]`
- treat `result.engines[].status = "error"` for `falcon` as a real pipeline failure
- treat `result.engines[].status = "error"` for `sam3` as a real segmentation failure because SAM 3 is mandatory
- if `result.primary_engine = "sam3"` but `sam3_segmentation_state = "segmenting"`, make sure the VM has commit `60590b1` or newer; older builds flapped readiness while computing the next mask
- run `bash scripts/gcp/check_realtime_service.sh` to enforce `primary_engine=sam3`, SAM3 engine `status=ok`, and `num_masks > 0`

If `/api/state` shows `capture_state = "error"`:
- the source URL is not currently usable
- replace it with another current public YouTube live URL
- if the error mentions bot checks or cookies, use `--yt-cookies-file`

If changing the source URL makes the local browser appear stuck:
- first check direct VM `/api/state` before assuming YouTube or SAM3 failed
- if the VM reports `source.status = "ready"` and `readiness.service_state = "live"`, restart the local tunnel
- confirm the VM image or hot-patched checkout includes commit `b435f76` or newer, because older UI builds used a long-lived MJPEG stream that could wedge the Windows tunnel

If `/api/state` shows `source.status = "auth_required"`:
- stop debugging Falcon and RT-DETR
- export `cookies.txt`
- copy it to `/opt/falcon-pipeline/youtube-cookies.txt`
- relaunch with `YTDLP_COOKIES_FILE=/opt/falcon-pipeline/youtube-cookies.txt`

If `/api/frame.jpg` stays placeholder:
- confirm `readiness.integration_ready` in `/api/state`
- confirm `source.status = "ready"`
- confirm `frame.capture_has_frame = true`
- confirm `model_status.falcon.state = "loaded"`
- confirm `model_status.rt_detr.state = "loaded"`

If latency is far below 10 FPS:
- inspect `metrics.processed_fps`, `metrics.falcon_guidance_generation_seconds`, `metrics.sam3_segmentation_generation_seconds`, and `result.frame_generation_seconds`
- do not assume the L4 is too weak until GPU utilization and per-stage timings prove it
- the April 24 source-switch run showed about `0.42` to `0.50` processed FPS, Falcon guidance around `4.2` to `8.0` seconds, SAM3 segmentation around `0.50` to `0.90` seconds, and bursty GPU use
- the next hardening target is to decouple display FPS from heavy model inference, not to buy a larger GPU first

If the frame is live but everyone is `unclassified`:
- inspect `result.scene_annotations.counts`
- inspect `result.engine_outputs.falcon.detections`
- inspect `result.engine_outputs.rt_detr.candidate_detections`
- inspect `result.scene_annotations.entities[*].role_reason`
- on the April 24 Hogs Breath run, Falcon returned one full-frame prompt box and RT-DETR found no tables, so the restaurant heuristic had no safe role grounding
- current hardening ignores full-frame guidance boxes and records why each person is classified or left `unclassified`

If Falcon loads but later fails during inference:
- retry with `--no-compile`
- confirm the runtime includes the current compile-fallback changes in `falcon_perception/attention.py`
- inspect `/api/state` for `result.engines[*].reason`

If Falcon logs `Failed to find C compiler`:
- the image is missing compiler tooling
- rebuild with `build-essential`

If Falcon logs `Python.h: No such file or directory`:
- the image is missing Python C headers
- rebuild with `python3-dev`
- do not keep debugging Triton, RT-DETR, or YouTube until that image dependency is fixed

If a Windows-to-VM file copy fails:
- re-run `gcloud compute ssh ... --command='pwd; ls -la /home/$USER/l4-fork'`
- use absolute remote paths in `gcloud compute scp`
- do not use `~` or OpenSSH-only `scp` flags

## 10. Teardown

If the VM is only for an isolated test run, delete it when you are done:

```powershell
pwsh -File .\scripts\gcp\delete_l4_vm.ps1 -ProjectId $env:PROJECT_ID -Zone $env:ZONE -VmName $env:VM_NAME
```

`delete_l4_vm.ps1` verifies:
- the instance returns `NOT_FOUND`
- the same-name boot disk returns `NOT_FOUND`
