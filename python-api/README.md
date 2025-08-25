# RAG-Anything FastAPI

Production-ready REST API for RAG-Anything, providing HTTP endpoints for document processing and querying.

## Features

- **Document Processing**: PDF, TXT, MD, DOCX, HTML, CSV, JSON
- **Query Modes**: naive, local, global, hybrid, mix
- **Authentication**: JWT tokens and API keys
- **Rate Limiting**: Per-user/IP limits
- **Auto Documentation**: Interactive API docs at `/docs`

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env

# Run
uvicorn app.main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

## Docker

```bash
docker-compose up --build
```

## Configuration

Key environment variables in `.env`:
```env
LIGHTRAG_WORKING_DIR=./storage
LIGHTRAG_LLM_MODEL=gpt-4o-mini
API_KEY=your-api-key
JWT_SECRET_KEY=your-secret-key
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