# Runs the full pipeline in a container (from the repo root, on Linux or macOS):
#   docker build -t persuadables .
#   mkdir -p data results reports docs/assets
#   docker run --rm --user "$(id -u):$(id -g)" \
#     -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
#     -v "$PWD/docs/assets:/app/docs/assets" -v "$PWD/reports:/app/reports" persuadables all
FROM python:3.12-slim

# XGBoost's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

# MPLCONFIGDIR: matplotlib needs a writable cache even when the container runs as a host uid
# that has no home directory.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=4

WORKDIR /app

# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY src ./src
COPY configs ./configs
# The output folders are writable by any uid, so `docker run --user "$(id -u):$(id -g)"` works
# for the folders that are not bind-mounted (artifacts/) as well as for those that are.
RUN uv sync --frozen --no-default-groups \
    && useradd --create-home --uid 1000 app \
    && mkdir -p data results artifacts reports docs/assets \
    && chown -R app /app \
    && chmod -R a+rwX data results artifacts reports docs/assets

USER app
ENTRYPOINT ["/app/.venv/bin/persuadables"]
CMD ["--help"]
