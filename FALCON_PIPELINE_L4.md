# Falcon Pipeline Realtime on Google Cloud L4

This is the primary heavy-runtime path for Falcon Pipeline.

The first milestone is intentionally narrow:
- one `g2-standard-8` VM with one `NVIDIA L4`
- Ubuntu `22.04`
- Docker as the runtime
- one public YouTube live stream
- `Falcon-Perception-300M` + `RT-DETR`
- `SAM 3` off by default
- private access through an SSH tunnel only

The first integration surface for other programs is:
- `GET /api/frame.jpg` for the latest overlay frame
- `GET /api/state` for readiness and error state

## Runtime profile

The intended default runtime is:
- Falcon model: `tiiuae/Falcon-Perception-300M`
- Task: `detection`
- RT-DETR: enabled
- SAM 3: disabled unless explicitly requested
- `max_dim=960`
- `falcon_refresh_seconds=2.0`

## Proven values on 2026-04-22

These are not product requirements, but they are the last known-good operator values from the live L4 bring-up:
- local account: `kirin@lifelineus.com`
- project: `tableminder`
- instance name: `falcon-pipeline-l4`
- first successful allocation zone: `us-east4-c`
- capacity note: `us-central1-a`, `us-central1-b`, `us-central1-c`, and `us-east4-a` were exhausted for `g2-standard-8` + `1 x NVIDIA L4`

If the preferred zone is exhausted again, retry another L4-capable zone before changing the runtime design.

## Files

- `falcon_pipeline_realtime_service.py`: FastAPI realtime service
- `falcon_pipeline_realtime_ui.html`: browser UI
- `Dockerfile.gcp-l4`: container image for the L4 VM
- `requirements-gcp-l4.txt`: Docker and VM Python dependencies
- `scripts/gcp/create_l4_vm.ps1`: zone-retrying VM create + bootstrap/GPU validation
- `scripts/gcp/delete_l4_vm.ps1`: verified teardown for instance + same-name boot disk
- `scripts/gcp/run_realtime_service.sh`: source preflight + container launch
- `scripts/gcp/check_realtime_service.sh`: host/Docker/API smoke check

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

```bash
git clone https://github.com/kirinhutheesing-pixel/l4-fork.git
cd l4-fork
docker build -f Dockerfile.gcp-l4 -t falcon-pipeline:l4 .
```

Notes:
- model weights are downloaded at runtime into `/opt/falcon-pipeline/hf-cache`
- overlay outputs and artifacts stay under `/opt/falcon-pipeline/outputs`
- neither weights nor outputs are baked into the image

## 5. Launch the service

Use a current public YouTube live URL. Channel live pages are usually more stable than one-off video IDs.

Example environment:

```bash
export TEST_SOURCE_URL="https://www.youtube.com/@earthcam/live"
export TEST_PROMPT="people near the center of the frame"
export HF_TOKEN="hf_..."
```

If that channel is not live when you test, replace it with any current public YouTube live URL.

Preferred path on the VM:

```bash
cd /home/kirin/l4-fork
TEST_SOURCE_URL="$TEST_SOURCE_URL" TEST_PROMPT="$TEST_PROMPT" bash scripts/gcp/run_realtime_service.sh
```

What `run_realtime_service.sh` does:
- runs the service image in `--preflight-only` mode first
- exits `20` on source auth failures
- exits `21` on unavailable/unplayable sources
- launches the long-running container only after preflight succeeds
- defaults to `--no-compile`
- mounts the cookie file only when `YTDLP_COOKIES_FILE` is set

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

This launch uses the intended first-milestone defaults:
- `Falcon-Perception-300M`
- `task=detection`
- `RT-DETR` loaded
- `SAM 3` not loaded

`HF_TOKEN` is optional if the model path is fully public, but include it for first bring-up so gated downloads or rate limits do not become the blocker.

## 6. Tunnel the service locally

From your local machine:

```powershell
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE -- -L 8080:localhost:8080
```

Use only the tunnel during first bring-up. Do not expose port `8080` publicly yet.

Then access:

```text
http://127.0.0.1:8080
```

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
   - `source.status = "ready"`
   - `frame.state = "live"`
   - `frame.is_placeholder = false`
   - `frame.ready = true`
   - `rtdetr_available = true`
   - `sam3_available = false`

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

### MJPEG browser stream

Open:

```text
http://127.0.0.1:8080
```

The UI should show the continuous overlay stream and current prompt controls.

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

If `/api/state` shows `capture_state = "error"`:
- the source URL is not currently usable
- replace it with another current public YouTube live URL
- if the error mentions bot checks or cookies, use `--yt-cookies-file`

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

If Falcon loads but later fails during inference:
- retry with `--no-compile`
- confirm the runtime includes the current compile-fallback changes in `falcon_perception/attention.py`
- inspect `/api/state` for `result.engines[*].reason`

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
