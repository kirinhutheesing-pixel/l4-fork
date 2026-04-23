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
$env:ZONE = "us-east4-c"
$env:VM_NAME = "falcon-pipeline-l4"
```

Quick checks:

```powershell
gcloud config get-value account
gcloud config get-value project
```

Expected:
- account = `kirin@lifelineus.com`
- project = `tableminder`

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
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE --strict-host-key-checking=no --command='cd /home/kirin && if [ ! -d l4-fork ]; then git clone https://github.com/kirinhutheesing-pixel/l4-fork.git; fi && cd /home/kirin/l4-fork && git pull --ff-only && sudo docker build -f Dockerfile.gcp-l4 -t falcon-pipeline:l4 .'
```

## 4. Start the service

```powershell
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE --strict-host-key-checking=no --command='cd /home/kirin/l4-fork && TEST_SOURCE_URL="YOUR_STREAM_URL" TEST_PROMPT="YOUR_PROMPT" bash scripts/gcp/run_realtime_service.sh'
```

Cookie-backed YouTube form:

```powershell
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE --strict-host-key-checking=no --command='cd /home/kirin/l4-fork && TEST_SOURCE_URL="YOUR_STREAM_URL" TEST_PROMPT="YOUR_PROMPT" YTDLP_COOKIES_FILE="/opt/falcon-pipeline/youtube-cookies.txt" bash scripts/gcp/run_realtime_service.sh'
```

`run_realtime_service.sh` always:
- runs a source preflight first
- exits `20` on source auth failures
- exits `21` on unavailable/unplayable sources
- launches the container with `--no-compile`
- mounts the cookie file only when `YTDLP_COOKIES_FILE` is set

## 5. Verify runtime

```powershell
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE --strict-host-key-checking=no --command='cd /home/kirin/l4-fork && bash scripts/gcp/check_realtime_service.sh'
```

Healthy live target:
- `healthz.status = ok`
- `readiness.service_state = live`
- `frame.ready = true`

Structured blocked target:
- `source.status = auth_required` with `readiness.error_kind = source_auth`
- or `source.status = unavailable` with `readiness.error_kind = source_unavailable`

## 6. Tunnel locally

```powershell
gcloud compute ssh $env:VM_NAME --project=$env:PROJECT_ID --zone=$env:ZONE -- -L 8080:localhost:8080
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

If SCP fails on Windows:
- use absolute remote paths like `/home/kirin/l4-fork/...`

If the source fails with YouTube bot-check text:
- do not keep retrying bare URLs
- export `cookies.txt` and use `YTDLP_COOKIES_FILE`
- follow [YOUTUBE_COOKIES.md](C:/Users/kirin/OneDrive/Documents/Playground/L4%20Fork/YOUTUBE_COOKIES.md)

If the service is up but the frame stays placeholder:
- inspect `/api/state`
- trust `readiness`, `source`, `frame`, `metrics`, and `model_status`

## 8. Delete the VM

```powershell
pwsh -File .\scripts\gcp\delete_l4_vm.ps1 -ProjectId $env:PROJECT_ID -Zone $env:ZONE -VmName $env:VM_NAME
```

The delete script verifies:
- the instance returns `NOT_FOUND`
- the same-name boot disk returns `NOT_FOUND`
