# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# Install system dependencies that are commonly required by MinerU/doc processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md README_zh.md setup.py MANIFEST.in requirements.txt ./
COPY raganything ./raganything
COPY examples ./examples
COPY assets ./assets
COPY docs ./docs
COPY app ./app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[ui]

EXPOSE 7860

CMD ["python", "-m", "app.gradio_app"]
