FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /service

COPY pyproject.toml ./
COPY app ./app
COPY laws ./laws
COPY docs ./docs
COPY entrypoint.sh ./entrypoint.sh

RUN pip install --upgrade pip && pip install .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /tmp/chroma_data \
    && chown -R appuser:appuser /service /tmp/chroma_data

USER appuser

EXPOSE 8001

ENTRYPOINT ["./entrypoint.sh"]
