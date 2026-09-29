FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY examples ./examples

# Reports written under /app/output can be mounted out: -v "$(pwd)/output:/app/output"
ENTRYPOINT ["pre"]
CMD ["--help"]
