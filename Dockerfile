FROM python:3.15-slim

LABEL org.opencontainers.image.title="video-redactor" \
      org.opencontainers.image.description="Headless payment-card video redaction"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN python -m pip install ".[ocr,api]"

RUN groupadd --system redactor \
    && useradd --system --gid redactor --create-home redactor \
    && install -d -o redactor -g redactor -m 0700 /tmp/video-redactor

USER redactor

ENTRYPOINT ["redact-video"]
CMD ["--help"]
