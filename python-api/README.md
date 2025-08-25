# RAG-Anything FastAPI

Production-ready REST API for RAG-Anything, providing HTTP endpoints for multimodal document processing and intelligent querying.

## Features

- **Document Processing**: PDF, DOCX, PPTX, images, and more via MinerU and Docling parsers
- **Vision-Language Models**: GPT-4o integration for advanced image understanding
- **Query Modes**: naive, local, global, hybrid, mix with LightRAG knowledge graphs
- **Modal Processors**: Specialized handling for images, tables, equations
- **Authentication**: JWT tokens and API keys with Redis session management
- **Auto Documentation**: Interactive Swagger UI at `/docs`

## Quick Start

### Using Docker (Recommended)

```bash
# Clone and navigate to directory
cd RAG-Anything/python-api

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start services
docker-compose up -d
```

API documentation: `http://localhost:8000/docs`

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install lightrag-hku==1.4.6

# Configure
export OPENAI_API_KEY=your_key_here
export REDIS__URL=redis://localhost:6379

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Configuration

Required environment variables:
```env
# OpenAI API (required)
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1

# Redis (default provided)
REDIS__URL=redis://redis:6379

# Optional
ENVIRONMENT=production
LOG_LEVEL=INFO
AUTH__SECRET_KEY=your-secret-key
```

## Architecture

```
app/
├── api/           # Route handlers
├── services/      # Business logic
├── models/        # Request/response models
├── middleware/    # Auth, rate limiting
└── integration/   # RAG-Anything integration
```