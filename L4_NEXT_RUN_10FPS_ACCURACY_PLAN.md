# L4 Next Run: 10 FPS And Accuracy Plan

This plan is based on the April 24 SAM3 L4 run. That run proved source ingest, Falcon, RT-DETR, SAM3, Docker, and the browser path can work on the L4. It also proved the current architecture is not a 10 FPS video pipeline.

Status after local hardening:
- P0 cached-overlay runtime has been implemented locally
- `/api/frame.jpg` now returns cached JPEG output without synchronously running RT-DETR, SAM3, or Falcon
- `/api/state` now exposes loop FPS, freshness, render/encode timing, response timing, and best-effort GPU telemetry
- `scripts/gcp/measure_realtime_fps.py` is the next VM measurement command
- the next VM run must prove the actual FPS target on the L4; local unit tests prove the contract, not live throughput

## Current baseline

Observed on the L4 VM:
- `readiness.full_pipeline_ready = true`
- `readiness.sam3_visual_ready = true`
- `result.primary_engine = "sam3"`
- final snapshot on the Jimmy's Fish House source showed `metrics.processed_fps = 0.47`
- Falcon guidance took about `4.2` to `8.0` seconds per pass
- SAM3 segmentation took about `0.50` to `0.90` seconds per pass
- overlay frame generation was about `0.44` seconds in the final local-tunnel snapshot
- GPU utilization was bursty, not continuously saturated
- VRAM use was about `5.3 GB` on a `23 GB` L4

Conclusion:
- the L4 is not obviously too small for the current model set
- the current service structure is too synchronous and too heavy to produce 10 FPS
- the next run should optimize architecture and scheduling before paying for a larger GPU

## Product target

The next useful target is:
- browser-visible overlay updates at 10 FPS
- SAM3 remains the visible segmentation layer
- RT-DETR and Falcon remain hidden helper engines
- restaurant people are color-coded only when there is enough evidence
- `/api/state` explains why each person was classified or left unclassified

## Main design change

Separate visual FPS from inference FPS.

The browser does not need a fresh Falcon, RT-DETR, and SAM3 pass for every displayed frame. It needs a fresh video frame plus the latest valid masks and roles projected onto it.

Target loops:
- capture loop: read frames continuously as fast as the source allows
- display loop: return JPEG overlays at 10 FPS by reusing the latest segmentation/role state
- RT-DETR loop: run at a lower fixed rate to refresh person/object boxes
- SAM3 loop: run on keyframes or changed prompt boxes, not every displayed frame
- Falcon loop: run even slower for semantic guidance and scene context
- state loop: publish compact state without blocking frame delivery

## Implementation priorities

P0:
- add explicit timing instrumentation for capture, RT-DETR, SAM3, Falcon, overlay rendering, JPEG encoding, and HTTP frame response
- add a fixed-rate `/api/frame.jpg` path that can return a fresh captured frame with the latest cached overlay even while SAM3 or Falcon is running
- prevent `/api/state` serialization from including large engine payloads unless a debug flag is requested
- keep `/api/stream.mjpg` secondary; the browser should keep using `/api/frame.jpg` polling over the tunnel

P1:
- add simple temporal reuse of SAM3 masks between segmentation passes
- associate detections across frames with stable entity ids using bounding-box overlap and mask overlap before adding a heavier tracker
- only trigger SAM3 when RT-DETR boxes materially change, when a mask gets stale, or on a fixed low-frequency refresh
- reduce default `max_dim` for live display experiments and measure the quality/FPS tradeoff

P2:
- add a dedicated restaurant scene profile with clearer evidence categories:
  - guest context near table/chair/cup/bottle/plate/bowl/wine glass
  - server context from movement through service areas or repeated table approach
  - needs-service context from repeated stationary waiting, arm/hand signal, facing service lane, or repeated absence of server contact
- keep uncertain people `unclassified`
- expose confidence and reason fields in `/api/state` rather than guessing roles silently

## What not to do first

Do not start by moving to a bigger GPU. The latest run showed bursty GPU use and low VRAM pressure, which points at scheduling and pipeline coupling.

Do not make RT-DETR or Falcon boxes visible again. They are helper signals only.

Do not try to make Falcon run at 10 FPS. Falcon should provide slower semantic context while the display loop stays responsive from cached masks and tracking.

Do not treat YouTube cookies as a latency fix. Cookies only solve source auth.

## Acceptance for the next VM run

The next run should prove:
- default source starts with `full_pipeline_ready = true`
- browser receives visible overlay updates at or near 10 FPS
- SAM3 remains the visible primary segmentation layer
- heavy model loops can be slower than the display loop without freezing the screen
- changing the source URL does not wedge the UI or the local tunnel
- `/api/state` reports measured per-stage timings
- restaurant annotations include stable reasons and confidence
- unsupported people remain `unclassified`

## Practical first experiment

Start with the smallest architecture change:
- keep the current models
- keep the current Docker image path
- keep the current endpoints
- add a cached-overlay frame pump that serves the latest captured frame plus the latest valid SAM3 masks
- measure whether this alone moves browser delivery toward 10 FPS

Only after that measurement should we decide whether to add a tracker, lower resolution further, change SAM3 cadence, or test a cheaper/larger GPU.
