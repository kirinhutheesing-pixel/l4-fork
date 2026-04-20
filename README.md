# Falcon Pipeline

Falcon Pipeline is a three-stage vision system built around:

- `Falcon Perception` for natural-language orchestration
- `RT-DETR` for live detection
- `SAM 3` for optional mask refinement

This workspace contains two main runtime targets:

- `falcon-pipeline`: local Apple Silicon CLI for images and sampled livestream analysis
- `falcon-pipeline-realtime`: CUDA-first realtime web app designed for a Google Cloud `L4` VM

## Quick Start

Local sampled run:

```bash
./falcon-pipeline --stream "https://www.youtube.com/watch?v=9c1oLjB3wIs" --query "umbrellas" --enable-rtdetr
```

Realtime L4 service:

```bash
./falcon-pipeline-realtime --source-url "https://www.youtube.com/watch?v=9c1oLjB3wIs" --prompt "people under the umbrellas"
```

## Docs

- `FALCON_PIPELINE.md`: local workflow and launcher notes
- `FALCON_PIPELINE_L4.md`: Google Cloud L4 deployment and prompting guide

## Realtime UI

The realtime UI gives you:

- a continuous overlay stream instead of saved frame samples
- a prompt box for Falcon Perception
- per-frame RT-DETR backbone inference
- optional SAM 3 refinement when mask quality matters more than speed

## Prompting

Use direct referring expressions:

- `people under the blue umbrellas`
- `boats on the horizon`
- `tables on the left side`

Avoid open-ended prompts like:

- `what is happening here`
- `anything unusual`
