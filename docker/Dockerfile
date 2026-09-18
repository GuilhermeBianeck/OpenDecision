FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir '.[inference,server]' \
    && groupadd --gid 10001 opendecision \
    && useradd --uid 10001 --gid 10001 --create-home opendecision \
    && mkdir -p /models/huggingface \
    && chown -R opendecision:opendecision /models
USER opendecision
EXPOSE 8042
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8042/health', timeout=4).read()"
CMD ["opendecision", "serve", "--host", "0.0.0.0", "--device", "cpu"]
