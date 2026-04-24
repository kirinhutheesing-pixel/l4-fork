# L4 Operator Checklist

Use this when you want the shortest possible path to a live Falcon Pipeline run on Google Cloud.

## 1. Local setup

From the repo root:

```powershell
cd "C:\Users\kirin\OneDrive\Documents\Playground\L4 Fork"
pwsh -File .\scripts\gcp\gcloud_setup.ps1 -ProjectId tableminder -AccountEmail kirin@lifelineus.com -ConfigDir .\.gcloud-config-l4-lifeline
```

Set the current shell:

```powershell
$env:CLOUDSDK_CONFIG = "$PWD\.gcloud-config-l4-lifeline"
$env:PROJECT_ID = "tableminder"
$env:ZONE = "us-west4-c"
$env:VM_NAME = "falcon-pipeline-l4"
$env:HF_TOKEN = "hf_..."
```

Quick checks:

```powershell
gcloud config get-value account
gcloud config get-value project
```

Expected:
- account = `kirin@lifelineus.com`
- project = `tableminder`

Latest observed success:
- `us-east4-c` on 2026-04-24
- `us-west4-c` on 2026-04-23
- `us-east4-c` on 2026-04-22

Agent status check for an existing VM:

```powershell
pwsh -File .\scripts\gcp\status_l4_vm.ps1 -ProjectId $env:PROJECT_ID -Zone $env:ZONE -VmName $env:VM_NAME
```

## 2. Create the VM

```powershell
pwsh -File .\scripts\gcp\create_l4_vm.ps1 -ProjectId $env:PROJECT_ID -VmName $env:VM_NAME
```

This script:
- retries the configured zone list until the first successful allocation
- waits for `/var/lib/falcon-pipeline/bootstrap.ready`
- runs the host `nvidia-smi` check
- runs the Docker GPU probe
- prints the winning zone at the end

If the winning zone is not the same as `$env:ZONE`, update `$env:ZONE` before the later SSH steps.

## 3. Build the image

```powershell
pwsh -File .\scripts\gcp\build_l4_image.ps1 -ProjectId $env:PROJECT_ID -Zone $env:ZONE -VmName $env:VM_NAME
```

This wrapper clones or pulls `/home/kirin/l4-fork`, starts the Docker build as a detached VM-side job, polls `/tmp/falcon-pipeline-build.log`, and uses IAP by default. Use it instead of a foreground SSH build because foreground IAP/Plink sessions have dropped during Docker layer export.

## 4. Start the service

```powershell
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE --tunnel-through-iap --strict-host-key-checking=no --command='cd /home/kirin/l4-fork && TEST_SOURCE_URL="YOUR_STREAM_URL" TEST_PROMPT="YOUR_PROMPT" HF_TOKEN="YOUR_APPROVED_HF_TOKEN" bash scripts/gcp/run_realtime_service.sh'
```

Cookie-backed YouTube form:

```powershell
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE --tunnel-through-iap --strict-host-key-checking=no --command='cd /home/kirin/l4-fork && TEST_SOURCE_URL="YOUR_STREAM_URL" TEST_PROMPT="YOUR_PROMPT" HF_TOKEN="YOUR_APPROVED_HF_TOKEN" YTDLP_COOKIES_FILE="/opt/falcon-pipeline/youtube-cookies.txt" bash scripts/gcp/run_realtime_service.sh'
```

`run_realtime_service.sh` always:
- runs a source preflight first
- runs a SAM 3 model preflight second
- exits `20` on source auth failures
- exits `21` on unavailable/unplayable sources
- exits `22` if `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is missing
- exits `23` on gated SAM model access
- exits `24` on unsupported SAM checkpoint/runtime mismatch
- exits `25` on generic SAM model load failure
- launches the container with `--no-compile`
- mounts the cookie file only when `YTDLP_COOKIES_FILE` is set

Current required runtime:
- `task=segmentation`
- `SAM 3` always loaded
- `RT-DETR` supplies live prompt boxes
- `HF_TOKEN` must already have approved access to the selected SAM model id
- default SAM model id is `facebook/sam3`
- `SAM3_MODEL_ID=facebook/sam3.1` is only a compatibility probe because the official `facebook/sam3.1` Hugging Face repo is checkpoint-only and does not provide Transformers integration

Useful optional launcher env vars:
- `SAM3_MODEL_ID=facebook/sam3`
- `FALCON_REFRESH_SECONDS=5.0`

Confirm `/api/state` before judging the screen:
- `result.primary_engine = "sam3"`
- `metrics.sam3_segmentation_state = "ready"` or `segmenting` with a recent SAM 3 primary result on commit `60590b1` or newer
- `readiness.sam3_visual_ready = true`
- `readiness.full_pipeline_ready = true`
- `readiness.blocking_engine_errors = []`

## 5. Verify runtime

```powershell
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE --tunnel-through-iap --strict-host-key-checking=no --command='cd /home/kirin/l4-fork && bash scripts/gcp/check_realtime_service.sh'
```

Healthy live target:
- `healthz.status = ok`
- `readiness.service_state = live`
- `readiness.full_pipeline_ready = true`
- `readiness.sam3_visual_ready = true`
- `readiness.blocking_engine_errors = []`
- `result.primary_engine = sam3`
- SAM3 engine `status = ok`
- SAM3 `num_masks > 0`
- `source.status = ready`
- `frame.ready = true`

Structured blocked target:
- `source.status = auth_required` with `readiness.error_kind = source_auth`
- or `source.status = unavailable` with `readiness.error_kind = source_unavailable`

If you want the raw engine detail behind the smoke result:

```powershell
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE --tunnel-through-iap --strict-host-key-checking=no --command='curl -fsS http://127.0.0.1:8080/api/state'
```

`check_realtime_service.sh` now fails if any enabled engine reports `status = error`, if SAM3 is not the visible primary output, or if SAM3 reports zero masks.
Use raw `/api/state` when you need:
- the exact `result.engines[*].reason`
- the current `readiness.blocking_engine_errors`
- deeper debugging beyond the pass/fail smoke result

## 6. Tunnel locally

```powershell
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE --tunnel-through-iap --strict-host-key-checking=no --ssh-flag='-L' --ssh-flag='127.0.0.1:8080:127.0.0.1:8080' --ssh-flag='-N'
```

Then open:
- `http://127.0.0.1:8080`
- `http://127.0.0.1:8080/api/state`
- `http://127.0.0.1:8080/api/frame.jpg`

## 7. When it fails

If VM creation fails:
- assume zonal capacity first

If SSH flags fail on Windows:
- remove OpenSSH `-o ...` flags
- keep `--strict-host-key-checking=no`
- for local tunnels, prefer `--tunnel-through-iap` plus explicit `--ssh-flag` values; the plain `-- -L ...` form was parsed incorrectly on this Windows setup

If SCP fails on Windows:
- use absolute remote paths like `/home/kirin/l4-fork/...`

If the source fails with YouTube bot-check text:
- do not keep retrying bare URLs
- export `cookies.txt` and use `YTDLP_COOKIES_FILE`
- follow [YOUTUBE_COOKIES.md](C:/Users/kirin/OneDrive/Documents/Playground/L4%20Fork/YOUTUBE_COOKIES.md)

If preflight returns `source.status = ready`:
- do not add cookies just because the source is YouTube
- the Hogs Breath stream `https://www.youtube.com/watch?v=S605ycm0Vlk` resolved without cookies on 2026-04-23

If the service is up but the frame stays placeholder:
- inspect `/api/state`
- trust `readiness`, `source`, `frame`, `metrics`, and `model_status`

If the service is degraded or `full_pipeline_ready = false` while a real frame exists:
- inspect `readiness.blocking_engine_errors`
- inspect container logs
- if you see `Failed to find C compiler`, rebuild the image with `build-essential`
- if you see `Python.h: No such file or directory`, rebuild the image with `python3-dev`
- do not treat a live RT-DETR fallback as a full pipeline success

If `/api/state` shows:
- `model_status.sam3.state = error`
- `model_status.sam3.error_kind = model_access`

Then:
- stop debugging the stream or the GPU
- this is a Hugging Face access or unsupported-checkpoint problem on the selected SAM model id
- rerun with an approved `HF_TOKEN`
- use `facebook/sam3` for supported runs; `facebook/sam3.1` still needs a separate integration proof

If `/api/state` shows:
- `model_status.sam3.state = error`
- `model_status.sam3.error_kind = model_unsupported`

Then:
- stop retrying the same checkpoint
- use `SAM3_MODEL_ID=facebook/sam3`
- treat `facebook/sam3.1` as unsupported until a native loader path is added and proven

If everyone is `unclassified` in `scene_annotations`:
- inspect `result.engine_outputs.falcon.detections`
- inspect `result.engine_outputs.rt_detr.candidate_detections`
- inspect `result.scene_annotations.entities[*].role_reason`
- if Falcon only returned one full-frame prompt box and RT-DETR found no tables, the classification layer does not have enough usable grounding on that frame
- that is a prompt/heuristic limitation, not a VM health failure

## 8. Delete the VM

```powershell
pwsh -File .\scripts\gcp\delete_l4_vm.ps1 -ProjectId $env:PROJECT_ID -Zone $env:ZONE -VmName $env:VM_NAME
```

The delete script verifies:
- the instance returns `NOT_FOUND`
- the same-name boot disk returns `NOT_FOUND`
