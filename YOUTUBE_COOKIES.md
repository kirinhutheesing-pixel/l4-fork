# YouTube Cookies For L4 Runs

Use this only when a YouTube stream fails with auth/bot-check errors such as:
- `Sign in to confirm you’re not a bot`
- `Use --cookies-from-browser or --cookies for the authentication`

Do not use `--cookies-from-browser` as the normal workflow on this Windows machine. Export a real Netscape-format `cookies.txt` instead.

## 0. Try preflight first

Do not add cookies just because the source is YouTube.

On 2026-04-23 the stream `https://www.youtube.com/watch?v=S605ycm0Vlk` resolved cleanly without cookies on the L4 VM.
On 2026-04-24 the same stream still resolved cleanly without cookies during a segmentation-mode run.

Use cookies only if preflight or `/api/state` reports:
- `source.status = "auth_required"`
- `readiness.error_kind = "source_auth"`

Do not confuse this with gated model access:
- YouTube cookies do not help if `SAM 3` fails to load from Hugging Face
- that case requires an `HF_TOKEN` with approved access to `facebook/sam3`
- cookies also do not fix `model_unsupported`; if `SAM3_MODEL_ID=facebook/sam3.1` fails the SAM preflight, switch back to `facebook/sam3`

## 1. Export the cookie file locally

Requirements:
- signed-in browser session that can open the target YouTube stream
- export format must be Netscape `cookies.txt`

Use any export method that produces a plain-text Netscape cookie jar. The runtime only needs the final `cookies.txt` file.

## 2. Copy it to the VM host path

The expected VM host path is:

```text
/opt/falcon-pipeline/youtube-cookies.txt
```

From Windows, use an absolute remote path:

```powershell
gcloud compute scp C:\path\to\cookies.txt kirin@falcon-pipeline-l4:/opt/falcon-pipeline/youtube-cookies.txt --project=YOUR_PROJECT_ID --zone=YOUR_ZONE
```

Do not use `~` in the remote path on this workstation. `gcloud compute scp` routes through `pscp`.

## 3. Launch the service with the cookie file

On the VM:

```bash
cd /home/kirin/l4-fork
TEST_SOURCE_URL="YOUR_STREAM_URL" \
TEST_PROMPT="YOUR_PROMPT" \
YTDLP_COOKIES_FILE="/opt/falcon-pipeline/youtube-cookies.txt" \
bash scripts/gcp/run_realtime_service.sh
```

## 4. Verify the state contract

The service should expose one of these clean outcomes in `/api/state`:

- success:
  - `source.status = "ready"`
  - `readiness.service_state = "live"` or progressing toward live

- still blocked:
  - `source.status = "auth_required"`
  - `readiness.error_kind = "source_auth"`

If it still reports `source_auth`, the cookie export is not usable for that stream and should be refreshed.

## 5. Keep source auth separate from SAM auth

`run_realtime_service.sh` now proves two separate things before launch:
- source preflight checks YouTube/direct ingest and can return `source_auth` or `source_unavailable`
- SAM preflight checks Hugging Face model access and runtime compatibility

If source preflight is `ready` but SAM preflight exits:
- `22`: set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`
- `23`: the token lacks approved access to the selected SAM checkpoint
- `24`: the selected checkpoint is unsupported by the current Transformers SAM 3 path
- `25`: generic SAM model-load/runtime failure

Do not refresh YouTube cookies for SAM preflight failures.
