# =============================================================================
# AI RESEARCH ASSISTANT - PRODUCTION DOCKERFILE
# =============================================================================

# Use Python 3.11 slim image for smaller size and better performance
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# DEPENDENCY STAGE
# =============================================================================
FROM base as dependencies

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# =============================================================================
# DEVELOPMENT STAGE
# =============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev,test]"

# Copy source code
COPY . .

# Create non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || exit 1

# Start command for development
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

# =============================================================================
# PRODUCTION STAGE
# =============================================================================
FROM dependencies as production

# Copy only necessary files
COPY backend/ ./backend/
COPY config.yaml .
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install --no-cache-dir .

# Create non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || exit 1

# Start command for production (using Gunicorn for better performance)
CMD ["gunicorn", "backend.server:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8001", \
     "--timeout", "300", \
     "--keep-alive", "2", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]

# =============================================================================
# TESTING STAGE
# =============================================================================
FROM development as testing

# Run tests
RUN python -m pytest tests/ --cov=backend --cov-report=html --cov-report=term-missing

# =============================================================================
# DOCUMENTATION STAGE
# =============================================================================
FROM dependencies as docs

# Install docs dependencies
RUN pip install --no-cache-dir -e ".[docs]"

# Copy source code
COPY . .

# Build documentation
RUN mkdocs build

# Serve documentation
EXPOSE 8000
CMD ["mkdocs", "serve", "--dev-addr", "0.0.0.0:8000"]