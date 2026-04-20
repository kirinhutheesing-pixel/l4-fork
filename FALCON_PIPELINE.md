# Falcon Pipeline

This workspace now includes a local Apple Silicon launcher for `tiiuae/Falcon-Perception`.

## Pipeline Roles

The current build uses a three-stage layout:

- `Falcon`: natural-language orchestrator
- `RT-DETR`: detection backbone
- `SAM 3`: optional segmentation/refinement stage

On this Mac, RT-DETR is working locally. SAM 3 is integrated but still gated by checkpoint access and is not a practical local default on this non-CUDA machine.

## Run it

```bash
cd "/Users/kirinhutheesing/Documents/Mac Playground "
./falcon-pipeline --image /absolute/path/to/photo.jpg --query "cat"
```

Run it from a normal macOS Terminal session. Inside Codex, MLX may need elevated permissions because it uses Metal directly.

Detection only:

```bash
./falcon-pipeline --image /absolute/path/to/photo.jpg --query "cat" --task detection
```

Livestreams, including YouTube live watch URLs:

```bash
./falcon-pipeline --stream "https://www.youtube.com/watch?v=emDyfhDmfUk" --query "umbrellas"
```

Run the RT-DETR backbone under Falcon orchestration:

```bash
./falcon-pipeline --stream "https://www.youtube.com/watch?v=9c1oLjB3wIs" --query "umbrellas" --enable-rtdetr
```

Try the optional SAM 3 refinement path:

```bash
./falcon-pipeline --stream "https://www.youtube.com/watch?v=9c1oLjB3wIs" --query "umbrellas" --enable-rtdetr --enable-sam3 --allow-experimental-sam3
```

Visible live viewer app:

```bash
./falcon-pipeline-app
```

Then open `http://127.0.0.1:8501` in Safari or another browser on this Mac.

Inside the app, enable the `RT-DETR backbone` toggle in `Orchestration stages`. `SAM 3 refinement` is exposed in the same panel with a clear hardware warning.

Google Cloud L4 realtime app:

```bash
./falcon-pipeline-realtime --source-url "https://www.youtube.com/watch?v=9c1oLjB3wIs" --prompt "people under the umbrellas"
```

That launches the CUDA-first web service on port `8080`. The full deployment guide is in `FALCON_PIPELINE_L4.md`.

Sample multiple frames from a stream:

```bash
./falcon-pipeline --stream "https://www.youtube.com/watch?v=emDyfhDmfUk" --query "tables" --stream-max-samples 3 --stream-sample-interval 10
```

Optional demo sample from the PBench dataset:

```bash
./falcon-pipeline --demo
```

## First run

The first launch downloads the Falcon Perception weights used by Falcon Pipeline into the workspace cache. In this setup the initial model download was about 2.1 GB and took a few minutes.

## Outputs

By default the program writes artifacts to `outputs/falcon-pipeline/`:

- `*-input.png`: a copy of the input image
- `*-overlay.png`: saved visualization with masks and boxes
- `*.json`: decoded text plus structured detections, primary engine, and per-stage status
- `*-stream.json`: stream-level summary with one entry per sampled frame

Model downloads are cached in `.cache/huggingface/` inside this workspace. YouTube watch URLs are resolved with `yt-dlp`, which is installed in the workspace virtual environment.

## Hardware Notes

This device is a MacBook Air with Apple M1 and 8 GB RAM. It is capable of running Falcon Pipeline with a visible live-monitoring UI, but not smooth full-motion realtime video.

Practical settings on this machine:

- `max_dim=512` is the best default for livestreams
- expect roughly 18 to 22 seconds per sampled Jimmy's Fish House frame in the app
- use sampled monitoring rather than trying to process every video frame
- expect the RT-DETR backbone to add extra latency here because PyTorch is running on the local CPU in this environment
- treat SAM 3 as a remote/CUDA-oriented stage unless you deliberately want an experimental local attempt
