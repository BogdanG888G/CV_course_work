# syntax=docker/dockerfile:1
FROM python:3.11.11-slim-bookworm
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
WORKDIR /workspace
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --timeout 180 --retries 10 \
      torch==2.5.1 torchvision==0.20.1 --index-url ${PYTORCH_INDEX_URL}
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --timeout 180 --retries 10 -r requirements.txt
COPY har ./har
COPY configs ./configs
COPY tests ./tests
COPY README.md ./README.md
CMD ["python", "-m", "har.prepare"]
