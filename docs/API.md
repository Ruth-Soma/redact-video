# HTTP API

Phase 8 exposes the verified automatic pipeline through a thin FastAPI adapter.
Video work never runs in an HTTP request handler. Accepted jobs enter a bounded
local process pool, and the worker can later be replaced by an external queue
without changing the routes or core pipeline.

## Run locally

Install the API and OCR extras on Python 3.12 or 3.13:

```bash
uv sync --python 3.13 --extra api --extra ocr
uv run redact-video-api
```

The server binds to `127.0.0.1:8000` by default. Set
`REDACT_VIDEO_API_HOST=0.0.0.0` only when the service is protected by the
deployment's authentication, TLS, and network controls. V1 does not implement
end-user authentication in the application itself.

## Create a job

```http
POST /v1/redactions
Content-Type: multipart/form-data
```

Multipart fields:

- `file`: required MP4 or MOV upload
- `policy`: optional; currently only `card_details`
- `mode`: optional `standard`, `safe`, or `fast`

```bash
curl -F file=@recording.mp4 \
  -F policy=card_details \
  -F mode=safe \
  http://127.0.0.1:8000/v1/redactions
```

The API responds with HTTP 202:

```json
{"job_id":"red_...","status":"queued"}
```

HTTP 413 indicates the configured upload limit was exceeded. HTTP 503 with a
`Retry-After` header indicates that both the worker and pending queue are full.

## Status, output, and report

```http
GET /v1/redactions/{job_id}
GET /v1/redactions/{job_id}/output
GET /v1/redactions/{job_id}/report
```

Status responses contain a sanitized lifecycle state and progress from 0 to
100. Output and report routes return HTTP 409 until the job is terminal.
`COMPLETED` and `NEEDS_REVIEW` jobs can both have an output; callers must inspect
the report status before treating the media as verified.

The report contains processing counts, sensitive categories, verification
statistics, per-stage millisecond timings, warnings, and timestamp-only
potential-issue windows. It never contains the upload filename, server paths,
mask timeline, OCR text, or a recognized card value. Failed reports use fixed
public error codes and messages rather than raw exception text.

## Resource limits and cleanup

| Environment variable | Default | Purpose |
| --- | ---: | --- |
| `REDACT_VIDEO_API_WORKERS` | `1` | Local worker processes |
| `REDACT_VIDEO_API_QUEUE_SIZE` | `4` | Pending jobs beyond active workers |
| `REDACT_VIDEO_API_MAX_UPLOAD_BYTES` | `2147483648` | Maximum uploaded video bytes |
| `REDACT_VIDEO_API_RETENTION_SECONDS` | `3600` | Terminal output/report retention |
| `REDACT_VIDEO_API_JOB_ROOT` | `/tmp/video-redactor` | Private temporary job root |
| `REDACT_VIDEO_CONFIG` | unset | Strict YAML engine configuration |
| `REDACT_VIDEO_API_HOST` | `127.0.0.1` | Server bind address |
| `REDACT_VIDEO_API_PORT` | `8000` | Server port |

Each job uses `/tmp/video-redactor/<job-id>/` with private permissions. The raw
upload and transient progress file are removed immediately after success or
failure. A background cleanup loop removes the retained output when its terminal
job expires. Service shutdown waits for owned workers and removes all local job
directories.

When `REDACT_VIDEO_CONFIG` is set, the server validates the YAML file during
startup. Each child process loads the same base profile and then applies the
job's `standard`, `safe`, or `fast` scan-rate mode. Unknown fields or invalid
values fail startup without echoing submitted YAML values.

## Docker

The image retains the CLI as its default entry point. Start its API entry point
explicitly:

```bash
docker run --rm -p 8000:8000 \
  -e REDACT_VIDEO_API_HOST=0.0.0.0 \
  --entrypoint redact-video-api \
  video-redactor
```
