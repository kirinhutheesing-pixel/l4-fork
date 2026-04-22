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

## Files

- `falcon_pipeline_realtime_service.py`: FastAPI realtime service
- `falcon_pipeline_realtime_ui.html`: browser UI
- `Dockerfile.gcp-l4`: container image for the L4 VM
- `requirements-gcp-l4.txt`: Docker and VM Python dependencies

## 1. Create the VM

Recommended shape:
- machine type: `g2-standard-8`
- GPU: `1 x NVIDIA L4`
- boot disk: `150 GB` or larger
- image: Ubuntu `22.04 LTS`

Keep the VM private for bring-up. Do not open the app port publicly for the first milestone.

## 2. Install the NVIDIA driver on the VM

If the image already has a working GPU driver and `nvidia-smi` succeeds, skip this section.

On Ubuntu `22.04`, install the signed Google Compute Engine driver packages:

```bash
sudo apt-get update
NVIDIA_DRIVER_VERSION=$(apt-cache search 'linux-modules-nvidia-[0-9]+-gcp$' | awk '{print $1}' | sort | tail -n 1 | head -n 1 | awk -F"-" '{print $4}')
sudo apt-get install -y "linux-modules-nvidia-${NVIDIA_DRIVER_VERSION}-gcp" "nvidia-driver-${NVIDIA_DRIVER_VERSION}"
sudo reboot
```

Reconnect and verify:

```bash
nvidia-smi
```

Expected result:
- one `NVIDIA L4`
- driver is loaded successfully

## 3. Install Docker and the NVIDIA container runtime

Install Docker:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
newgrp docker
```

Install NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify Docker can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi
```

Expected result:
- the container prints the same `NVIDIA L4`

## 4. Clone the repo and build the image

```bash
git clone https://github.com/kirinhutheesing-pixel/l4-fork.git
cd l4-fork
mkdir -p .cache/huggingface outputs/falcon-pipeline-realtime
docker build -f Dockerfile.gcp-l4 -t falcon-pipeline:l4 .
```

Notes:
- model weights are downloaded at runtime into `.cache/huggingface`
- overlay outputs and artifacts stay under `outputs/falcon-pipeline-realtime`
- neither weights nor outputs are baked into the image

## 5. Launch the service

Use a current public YouTube live URL. Channel live pages are usually more stable than one-off video IDs.

Example environment:

```bash
export TEST_SOURCE_URL="https://www.youtube.com/@earthcam/live"
export TEST_PROMPT="people near the center of the frame"
```

If that channel is not live when you test, replace it with any current public YouTube live URL.

Run the service:

```bash
docker run --rm --gpus all -p 8080:8080 \
  -v "$PWD/.cache/huggingface:/app/.cache/huggingface" \
  -v "$PWD/outputs/falcon-pipeline-realtime:/app/outputs/falcon-pipeline-realtime" \
  falcon-pipeline:l4 \
  --source-url "$TEST_SOURCE_URL" \
  --prompt "$TEST_PROMPT"
```

This launch uses the intended first-milestone defaults:
- `Falcon-Perception-300M`
- `task=detection`
- `RT-DETR` loaded
- `SAM 3` not loaded

## 6. Tunnel the service locally

From your local machine:

```bash
gcloud compute ssh YOUR_VM_NAME --zone YOUR_ZONE -- -L 8080:localhost:8080
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
   - `frame.state = "placeholder"`
   - `metrics.capture_state = "connecting"` or `"running"`

3. Live:
   - `readiness.service_state = "live"`
   - `readiness.integration_ready = true`
   - `frame.state = "live"`
   - `frame.is_placeholder = false`
   - `frame.ready = true`
   - `rtdetr_available = true`
   - `sam3_available = false`

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

If `docker build` fails:
- confirm you are in the repo root
- confirm `requirements-gcp-l4.txt` is present
- confirm the VM has internet access to pull Python packages and the CUDA base image

If `docker run --gpus all ... nvidia-smi` fails:
- re-check host driver installation
- re-check `nvidia-container-toolkit`
- re-run `sudo nvidia-ctk runtime configure --runtime=docker`

If `/api/healthz` works but `/api/state` stays in `warming`:
- wait for first model download to finish
- inspect container logs for Hugging Face download or model-load errors

If `/api/state` shows `capture_state = "error"`:
- the source URL is not currently usable
- replace it with another current public YouTube live URL

If `/api/frame.jpg` stays placeholder:
- confirm `readiness.integration_ready` in `/api/state`
- confirm `frame.capture_has_frame = true`
- confirm `model_status.falcon.state = "loaded"`
- confirm `model_status.rt_detr.state = "loaded"`
