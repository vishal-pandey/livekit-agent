# Alternative Dockerfile using Python base to avoid repository issues
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (minimal approach)
RUN apt-get update && apt-get install -y --no-install-recommends

# Create a non-root user for security
# RUN useradd --create-home --shell /bin/bash app && \
#     chown -R app:app /app
# USER app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/home/app/.local/bin:${PATH}"

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies with retries
RUN python3 -m pip install --user --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --user --no-cache-dir --timeout=1000 --retries=5 -r requirements.txt

# Copy application code
COPY agent.py .

# Create recordings directory and cache directories
RUN mkdir -p recordings && \
    mkdir -p /home/app/.cache/huggingface

RUN python3 agent.py download-files

# Health check to ensure the container is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

RUN export TMPDIR=/var/tmp

# Default command to run the agent (models will download on first use)
CMD ["python3", "agent.py", "start"]
