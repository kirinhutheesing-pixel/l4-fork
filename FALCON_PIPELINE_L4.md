# Falcon Pipeline Realtime on Google Cloud L4

This build is the realtime, CUDA-first version of Falcon Pipeline.

Architecture:
- `Falcon Perception` is the natural-language orchestrator.
- `RT-DETR` is the per-frame detection backbone.
- `SAM 3` is the optional live segmentation refiner.

The browser UI is continuous video, not a sampled frame gallery. Internally, the service drops stale frames so the GPU always works on the freshest available view.

## Files

- `falcon_pipeline_realtime_service.py`: FastAPI realtime service and background workers.
- `falcon_pipeline_realtime_ui.html`: browser UI with livestream view and Falcon prompt box.
- `falcon-pipeline-realtime`: local launcher.
- `requirements-gcp-l4.txt`: Python dependencies for a CUDA L4 VM.
- `Dockerfile.gcp-l4`: container image for GCP deployment.

## Recommended Google Cloud shape

Start with one of these:
- `g2-standard-8`: best default for a single L4 production test box.
- `g2-standard-16`: better headroom if you expect multiple viewers or heavier segmentation.

Practical notes:
- L4 gives you `24 GB` of GPU memory per GPU.
- Use a generous boot disk, preferably `150 GB` or more, because model weights and container layers add up quickly.
- Use Ubuntu `22.04`.

## VM setup outline

1. Create a `G2` VM with one `NVIDIA L4`.
2. Install the NVIDIA driver on the host.
3. Install Docker plus the NVIDIA container runtime, or use a Python venv directly.
4. Copy this project onto the VM.
5. Launch the realtime service and tunnel the port locally.

## Option A: Run with Docker

Build the image:

```bash
docker build -f Dockerfile.gcp-l4 -t falcon-pipeline:l4 .
```

Run the service:

```bash
docker run --rm --gpus all -p 8080:8080 \
  -v "$PWD/.cache:/app/.cache" \
  falcon-pipeline:l4 \
  --source-url "https://www.youtube.com/watch?v=9c1oLjB3wIs" \
  --prompt "people under the umbrellas" \
  --task detection
```

## Option B: Run with a Python venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-gcp-l4.txt
chmod +x falcon-pipeline-realtime
./falcon-pipeline-realtime \
  --source-url "https://www.youtube.com/watch?v=9c1oLjB3wIs" \
  --prompt "people under the umbrellas"
```

## Reach the UI safely

Use an SSH tunnel instead of opening the app to the public internet immediately:

```bash
gcloud compute ssh YOUR_VM_NAME --zone YOUR_ZONE -- -L 8080:localhost:8080
```

Then open:

```text
http://127.0.0.1:8080
```

## Realtime tuning

If you want lower latency:
- keep `task` on `detection`
- keep `SAM 3` off until detection quality is stable
- use `max_dim=960` first, then move down to `736` if needed
- keep `falcon_refresh_seconds` around `2.0` to `3.0`

If you want better mask quality:
- switch `task` to `segmentation`
- enable `SAM 3`
- expect lower live throughput

## How to prompt Falcon Perception well

Falcon Perception works best with direct referring expressions.

Use prompts like:
- `people under the blue umbrellas`
- `tables on the left side of the deck`
- `boats on the horizon`
- `person standing near the railing`
- `empty tables closest to the water`

Avoid prompts like:
- `what is happening here`
- `does this look busy`
- `watch the scene and tell me if anything interesting happens`
- `everything`

Guidelines:
- Name the visible thing you want to track.
- Add location words when many similar objects are present.
- Add attributes like color, size, or scene role if they are visually obvious.
- Keep each prompt focused on one target concept for the best live performance.
- For counting, prompt the object directly and use the detection count in the UI.

## Why this version can be realtime

The original laptop build sampled frames because Falcon Perception was doing too much work per refresh on Apple Silicon.

This L4 version changes the workload split:
- `RT-DETR` handles the fast frame loop.
- `Falcon` refreshes the natural-language grounding periodically instead of on every frame.
- `SAM 3` only runs when you explicitly want refined masks.

That is the reason this build can behave like a live perception service instead of a slow frame-by-frame inspector.
