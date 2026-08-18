# Dockerfile for crypto-bot-ML
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files first for caching
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen --no-install-project || uv sync --no-install-project

# Copy project source code
COPY . .

# Install project package
RUN uv sync --frozen || uv sync

# Create volume mount points for persistence
VOLUME ["/app/data", "/app/logs", "/app/artifacts"]

# Default entrypoint runs the bot in paper mode (or configured BOT_MODE)
ENTRYPOINT ["uv", "run", "python", "scripts/run_bot.py"]
CMD ["--warmup-bars", "2000"]
