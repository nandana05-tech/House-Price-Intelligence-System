FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    libatomic1 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install uv && uv pip install --system --no-cache -e .

# Production stage
FROM python:3.12-slim AS production

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    libatomic1 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY . .

RUN test -d models || (echo "ERROR: folder models/ tidak ditemukan" && exit 1)
RUN test -d metadata || (echo "ERROR: folder metadata/ tidak ditemukan" && exit 1)

RUN prisma generate

EXPOSE 8080

# Default CMD: jalankan FastAPI. Setiap service di docker-compose
# bisa override command ini sesuai kebutuhan (mcp_server, consumer_*, dll).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]