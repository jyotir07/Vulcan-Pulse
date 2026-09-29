# API service: FastAPI over the counterfactual engine.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# The CPU-only torch wheel is several times smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

COPY pyproject.toml ./
COPY backend ./backend
RUN pip install .

COPY configs ./configs
COPY scripts ./scripts
COPY experiments ./experiments
COPY docker/api-entrypoint.sh /usr/local/bin/api-entrypoint.sh
RUN chmod +x /usr/local/bin/api-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["api-entrypoint.sh"]
