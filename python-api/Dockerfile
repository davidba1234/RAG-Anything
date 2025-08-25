FROM python:3.10-slim

# Set Python environment variables for production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for document processing
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    git \
    wget \
    curl \
    # Document processing
    libreoffice \
    poppler-utils \
    # OCR support
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    # Image processing libraries
    libmagic1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
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

# Install Python dependencies in stages for better caching
# Stage 1: Core dependencies and numpy
RUN pip install --no-cache-dir numpy==1.26.4 && \
    pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu

# Stage 2: Base requirements
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir --force-reinstall \
    numpy==1.26.4 \
    sympy==1.12

# Stage 3: LightRAG and related packages
RUN pip install --no-cache-dir \
    lightrag-hku==1.4.6 \
    nano-vectordb \
    networkx

# Stage 4: Document parsers
# Install MinerU (magic-pdf) - basic version without detectron2
RUN pip install --no-cache-dir \
    magic-pdf==0.7.0b1 \
    paddleocr \
    paddlepaddle

# Install Docling for advanced document parsing
RUN pip install --no-cache-dir \
    docling==1.20.0

# Set Tesseract data path for OCR
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

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