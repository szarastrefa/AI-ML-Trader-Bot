FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create trader user with proper permissions
RUN groupadd -r trader && useradd -r -g trader trader

# Copy requirements first for better caching
COPY backend/requirements.txt requirements.txt

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ .

# Create directories with proper permissions
RUN mkdir -p logs data models strategies config /tmp/celery \
    && rm -rf celerybeat-schedule* /tmp/celerybeat-schedule* 2>/dev/null || true \
    && chown -R trader:trader /app /tmp/celery \
    && chmod 755 /tmp/celery \
    && chmod +x /app/*.py 2>/dev/null || true

# Switch to trader user
USER trader

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["python", "main.py"]
