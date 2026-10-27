# EMIPredict AI — production image
#
# Build:  docker build -t emi-predict-ai .
# Run:    docker run --rm -p 8501:8501 -e PORT=8501 emi-predict-ai
#
# The 295 MB compressed regression artifact is copied from the build context
# when present. On a clean clone (where the artifact is gitignored), the build
# downloads it from Hugging Face and verifies the SHA-256 before proceeding.

FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="emi-predict-ai" \
      org.opencontainers.image.description="EMI eligibility classification and maximum-affordable-EMI prediction" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/home/app

WORKDIR /app

# libgomp1 is required by the XGBoost wheel; ca-certificates is required for the
# artifact download. No compiler toolchain is installed: all dependencies have
# manylinux wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create the non-root runtime user before copying large files, so --chown can
# be used consistently.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

COPY requirements.txt ./
# Install Pillow in its own layer first. This avoids a resolver edge case in
# which pip's combined backtracking reports "no versions" for Pillow even when
# the wheel is available. The version matches the requirements pin.
RUN pip install --no-cache-dir --retries 10 --timeout 60 pillow==12.3.0
RUN pip install --no-cache-dir --retries 10 --timeout 60 --prefer-binary -r requirements.txt \
    && python -c "import sklearn, xgboost, pandas, numpy, streamlit; print('deps ok')"

# Application code and small artifacts.
COPY --chown=app:app emi_app ./emi_app
COPY --chown=app:app pages ./pages
COPY --chown=app:app app.py entrypoint.py ./
COPY --chown=app:app .streamlit ./.streamlit
COPY --chown=app:app scripts/fetch_model.py ./scripts/fetch_model.py

# Small model metadata + classification model (10 MB).
COPY --chown=app:app models/ ./models/

# Compact EDA assets and saved reports (the raw/cleaned datasets are excluded
# via .dockerignore and are not needed at runtime).
COPY --chown=app:app data/eda_sample.csv.gz data/eda_summary.json ./data/
COPY --chown=app:app reports ./reports

# Large regression artifact. It is present in a local checkout but gitignored;
# a clean Render build downloads it. The SHA-256 below matches the local
# artifact and the Hugging Face LFS object:
#   a321224177d43c5e7de64c8bc98fa0f15689240ba7f521c29139ca412cd9e401
ARG REGRESSION_MODEL_URL="https://huggingface.co/M-Prasad/EMIPredict_AI/resolve/main/best_regression_model_compressed.pkl?download=true"
ARG REGRESSION_MODEL_SHA256="a321224177d43c5e7de64c8bc98fa0f15689240ba7f521c29139ca412cd9e401"
ARG REGRESSION_MODEL_SIZE="295141016"

ENV EMI_REGRESSION_MODEL_URL=${REGRESSION_MODEL_URL} \
    EMI_REGRESSION_MODEL_SHA256=${REGRESSION_MODEL_SHA256} \
    EMI_REGRESSION_MODEL_SIZE=${REGRESSION_MODEL_SIZE}

RUN python scripts/fetch_model.py \
        --url "${EMI_REGRESSION_MODEL_URL}" \
        --sha256 "${EMI_REGRESSION_MODEL_SHA256}" \
        --size "${EMI_REGRESSION_MODEL_SIZE}" \
        --dest /app/models/best_regression_model_compressed.pkl \
    && chown -R app:app /app

USER app

ENV PORT=8501
EXPOSE 8501
STOPSIGNAL SIGTERM

# Streamlit's built-in health endpoint is lightweight and does not run a
# prediction. The Health page adds an on-demand model smoke test.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8501') + '/_stcore/health', timeout=3).read()"

CMD ["python", "entrypoint.py"]
