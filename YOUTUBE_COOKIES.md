# YouTube Cookies For L4 Runs

Use this only when a YouTube stream fails with auth/bot-check errors such as:
- `Sign in to confirm you’re not a bot`
- `Use --cookies-from-browser or --cookies for the authentication`

Do not use `--cookies-from-browser` as the normal workflow on this Windows machine. Export a real Netscape-format `cookies.txt` instead.

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
