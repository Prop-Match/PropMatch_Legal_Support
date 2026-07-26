FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /service

COPY pyproject.toml ./
COPY app ./app
COPY laws ./laws
COPY docs ./docs

RUN pip install --upgrade pip && pip install .

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /service

USER appuser

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
