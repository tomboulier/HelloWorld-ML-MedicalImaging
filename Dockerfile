# Container image that runs the training and the tests with the exact
# dependency versions of uv.lock. See docs/usage.md for how to run it with
# Docker on a workstation and with Apptainer on a cluster.
FROM python:3.12-slim

# uv installs the locked dependencies; the version is pinned like any other dependency.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Install dependencies first so that this layer is cached when only the code changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# Then the project itself (package, configuration, tests).
COPY src ./src
COPY config.toml ./
COPY tests ./tests
RUN uv sync --frozen

# Default command: train with config.toml. Override it to run the tests, e.g.
#   docker run --rm hello-world-medical-imaging pytest
CMD ["hello-world-medical-imaging", "--config", "config.toml"]
