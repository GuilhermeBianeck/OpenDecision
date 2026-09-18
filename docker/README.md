# CPU container

Build from the repository root:

```sh
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  api opendecision pull base --device cpu
docker compose -f docker/compose.yaml up
```

The explicit pull downloads pinned weights into a persistent named volume and runs a local smoke test. Serving sets Hugging Face and Transformers offline flags. The service runs as a non-root user and publishes only `127.0.0.1:8042` on the host. It listens on all interfaces inside the isolated container, so the CLI emits its standard network binding warning.

The image uses PyTorch CPU wheels. MPS is unavailable inside Docker on macOS; use the native Python installation to evaluate Apple Silicon acceleration. No model weights are baked into the image. Docker build and runtime performance must be verified on your own Docker host.
