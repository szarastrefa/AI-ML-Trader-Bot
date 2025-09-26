FROM python:3.11-slim

# Set metadata
LABEL maintainer="szarastrefa"
LABEL description="AI/ML Trading Bot Backend"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV FLASK_APP=main.py
ENV FLASK_ENV=production

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
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source with compatibility fixes
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && ldconfig

# Create non-root user
RUN groupadd -r trader && useradd -r -g trader trader

# Copy requirements first (for better Docker layer caching)
COPY backend/requirements.txt requirements.txt

# Upgrade pip and install Python dependencies with compatibility fixes
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install compatible NumPy version first
RUN pip install --no-cache-dir "numpy>=1.21.0,<1.25.0"

# Install TA-Lib Python wrapper with compatible version
RUN pip install --no-cache-dir --no-build-isolation "TA-Lib>=0.4.24,<0.5.0"

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ .

# Create necessary directories with proper permissions
RUN mkdir -p logs data models strategies config \
    && chown -R trader:trader /app

# Switch to non-root user
USER trader

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "main.py"]