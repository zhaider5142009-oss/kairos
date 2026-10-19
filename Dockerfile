FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# core deps only — torch/qiskit/MetaTrader5 stay out of the free-tier image;
# KAIROS degrades gracefully without them
COPY requirements-docker.txt ./
RUN pip install -r requirements-docker.txt

COPY . .

# Render injects $PORT; without a config.yaml the engine runs on safe defaults
# (no LLM council, no broker credentials — a deployed tower is read-mostly)
CMD ["sh", "-c", "python -m kairos serve --host 0.0.0.0 --port ${PORT:-8600}"]
