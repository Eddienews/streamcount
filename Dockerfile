FROM python:3.11-slim

# ffmpeg is the only system dependency (frame sampling)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE.md ./
COPY streamcount ./streamcount
RUN pip install --no-cache-dir .

# runtime data (model cache, runs) lives in the mounted volume
ENV STREAMCOUNT_HOME=/data
WORKDIR /data

ENTRYPOINT ["streamcount"]
CMD ["--help"]
