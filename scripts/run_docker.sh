#!/usr/bin/env bash
set -euo pipefail

IMAGE=credit-default-api

docker build -t "$IMAGE" .
docker rm -f "$IMAGE" >/dev/null 2>&1 || true
docker run -d --name "$IMAGE" -p 8000:8000 "$IMAGE"

for _ in $(seq 1 30); do
  curl -sf http://localhost:8000/health >/dev/null && break
  sleep 1
done

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @tests/data/example_request.json
echo
