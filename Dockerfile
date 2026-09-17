FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .

RUN pip install \
    --no-cache-dir \
    --upgrade pip \
    && pip install \
    --no-cache-dir \
    -r requirements-api.txt

COPY api ./api
COPY src ./src
COPY models ./models

RUN useradd \
    --create-home \
    --shell /bin/bash \
    appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
