FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libreoffice \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy RAG-Anything modules
COPY raganything /app/raganything/

# Copy application code
COPY python-api/requirements.txt /app/requirements.txt
COPY python-api/app /app/app/

# Install Python dependencies
# First install numpy for OpenCV compatibility, then requirements, then lightrag-hku
RUN pip install --no-cache-dir numpy==1.26.4 && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir --force-reinstall \
    numpy==1.26.4 \
    sympy==1.12 && \
    pip install --no-cache-dir \
    lightrag-hku==1.4.6 \
    nano-vectordb \
    networkx

# Create necessary directories
RUN mkdir -p /app/storage /app/lightrag_storage /app/uploads /app/temp /app/logs \
    && chown -R app:app /app

# Switch to app user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]