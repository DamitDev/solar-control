FROM python:3.12-slim

ARG APP_VERSION=0.0.0-dev
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN sed -i "s/^version = .*/version = \"${APP_VERSION}\"/" pyproject.toml && \
    pip install --no-cache-dir -e .

RUN useradd --create-home --uid 1000 appuser && \
    mkdir -p /app/data && \
    chmod +x entrypoint.sh && \
    chown -R appuser:appuser /app

USER appuser

ENV HOST=0.0.0.0
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
CMD ["sh", "-c", "uvicorn app.main:sio_asgi_app --host $HOST --port $PORT"]
