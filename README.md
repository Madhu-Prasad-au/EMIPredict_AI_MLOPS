# EMIPredict AI

A deployable EMI (Equated Monthly Instalment) assessment application built from
the trained artifacts in this repository. It serves:

- **EMI eligibility classification** — `Eligible`, `High_Risk`, or `Not_Eligible`
- **Maximum affordable monthly EMI regression**
- **Interactive EDA** over a shipped stratified sample + full-dataset aggregates
- **Saved model-performance reports**
- **Session-only assessment history**
- **Application and artifact health checks**

The app is a **Streamlit** multipage application. It runs on a single port, has
no external runtime services, and is packaged as a non-root Docker image for
Render.

> **Important:** this is a machine-learning demonstration, not a credit
> approval system. The **full profile's** regression model was trained with a
> target-derived feature; see [Model limitation](#model-limitation-target-leakage).
> The **free/lite profile** is a separate, leakage-free compact model.

## Two deployment profiles

| | Full profile (default) | Free/lite profile |
|---|---|---|
| Select with | `EMI_PROFILE=full` | `EMI_PROFILE=lite` |
| Models | Saved XGBoost + 200-tree Random Forest | New compact HistGradientBoosting pair |
| Features | 36, includes target-derived `affordability_ratio` | 35, leakage-free |
| Memory | ~1.25 GB RSS; ~2.3 GB load peak | Designed for a 512 MB instance |
| Image | `Dockerfile` / `render.yaml` | `Dockerfile.free` / `render-free.yaml` |
| Render | Paid 4 GB instance | **Free 512 MB instance; no card** |
| Metrics | Original notebook reports (optimistic regression) | Honest held-out test: F1 0.9795, R² 0.9940 |

The lite models are new artifacts created for free hosting. They do **not**
replace the saved full artifacts. The full profile remains the default for
local/paid deployments.

---

## Why Streamlit

The project already had a Streamlit form/pages structure, and the workload is
interactive single-user analytics plus a small form-based inference call. A
Streamlit multipage app keeps the EDA, model performance, and prediction UI in
one Python codebase with no separate frontend build, API gateway, or JS
toolchain. The saved sklearn pipelines run directly in-process. The trade-off
is that Streamlit's HTTP surface is not a general REST API; the production
prediction entry point is the form plus the reusable `emi_app.service.predict`
function, and the platform health endpoint is Streamlit's built-in
`/_stcore/health`.

---

## Repository layout

```text
app.py                         Home / overview page
entrypoint.py                  Preloads artifacts, then starts Streamlit on $PORT
EXP1(1).ipynb                  Original training notebook (provenance; not runtime)
pages/
  1_Prediction.py              Form, validation, prediction results
  2_EDA_Analytics.py           Filters, charts, sample download
  3_Model_Performance.py       Saved validation/test metrics
  4_Assessment_History.py      Session-only history (no server-side persistence)
  5_Health.py                  Artifact/runtime/readiness + prediction smoke test
emi_app/
  config.py                    Root-anchored paths, env config, version pins
  artifacts.py                 One-time load + validation of all model artifacts
  schema.py                    Input schema, validation, defaults
  preprocessing.py             Exact training-time feature engineering
  service.py                   Two-model inference + decision support
  data.py                      EDA sample/summary and report loaders
  health.py                    Health snapshot and smoke test
  ui.py                        Shared Streamlit UI helpers
models/
  best_classification_model.pkl        XGBoost Pipeline (10.9 MB, committed)
  classification_label_encoder.pkl     LabelEncoder (committed)
  feature_info.pkl                      Full-profile feature metadata
  preprocessing_stats.json              Medians/ranges (committed)
  best_regression_model_compressed.pkl  Full Random Forest (295 MB, not committed)
  lite_classification_model.pkl         Compact HistGradientBoosting classifier (~1 MB)
  lite_regression_model.pkl             Compact HistGradientBoosting regressor (~0.3 MB)
  lite_feature_info.json                Lite profile metadata (35 features)
  lite_metrics.json                     Lite held-out test metrics
reports/                       Saved full-model comparison / final metrics CSVs
data/
  eda_sample.csv.gz            60,001-row stratified sample (5.8 MB)
  eda_sample_lite.csv.gz       20,001-row sample for the free profile (1.9 MB)
  eda_summary.json             Full-dataset aggregates (404,800 rows)
scripts/
  prepare_assets.py            Rebuilds the compact EDA assets (needs full CSV)
  train_lite_models.py         Trains the compact, leakage-free lite pair
  fetch_model.py               Downloads/verifies the large full regression artifact
  smoke_test.py                End-to-end artifact + prediction test
Dockerfile                     Full-profile image
Dockerfile.free                Free/lite 512 MB image
render.yaml                    Paid 4 GB full-profile Blueprint
render-free.yaml               Free 512 MB lite-profile Blueprint
requirements.txt               Full-profile dependencies
requirements-free.txt          Lite-profile dependencies (no xgboost)
.dockerignore
.gitignore
```

The original emoji-named pages, `utils/` helpers, and the ephemeral CSV CRUD page
were replaced. The CRUD page wrote to `data/financial_records.csv`, which is not
durable on Render; assessment history now lives in Streamlit session state.

---

## Artifacts and version pinning

The saved pipelines were inspected directly. The classification pickle records
`_sklearn_version = 1.9.0`. The matching training environment found on this
machine was Python 3.12 with:

| Package | Version |
|---|---|
| scikit-learn | 1.9.0 |
| xgboost | 3.4.1 |
| pandas | 2.3.3 |
| numpy | 2.5.2 |
| joblib | 1.5.3 |
| streamlit | 1.62.0 |

`requirements.txt` pins those versions. The image installs **`xgboost-cpu==3.4.1`**
rather than the full `xgboost` wheel: the full wheel pulls in CUDA/NCCL
packages that are unnecessary for inference and add hundreds of MB. It is the
same 3.4.1 codebase and provides the same importable `xgboost` module and pickle
compatibility; inference is CPU-only.

### Large regression artifact

`models/best_regression_model_compressed.pkl` is 295,141,016 bytes. It is
gitignored because it exceeds GitHub's per-file limit. Its SHA-256 is:

```text
a321224177d43c5e7de64c8bc98fa0f15689240ba7f521c29139ca412cd9e401
```

The file is hosted publicly at:

```text
https://huggingface.co/M-Prasad/EMIPredict_AI/resolve/main/best_regression_model_compressed.pkl?download=true
```

The SHA-256 was verified against the Hugging Face LFS object metadata and the
local file. The Dockerfile:

1. Copies the artifact if it is present in the build context (local builds).
2. Otherwise downloads it from the URL above.
3. Verifies size and SHA-256 before the image build succeeds.

If the artifact is missing locally, fetch it with:

```powershell
python scripts/fetch_model.py `
  --url "https://huggingface.co/M-Prasad/EMIPredict_AI/resolve/main/best_regression_model_compressed.pkl?download=true" `
  --sha256 a321224177d43c5e7de64c8bc98fa0f15689240ba7f521c29139ca412cd9e401 `
  --size 295141016 `
  --dest models/best_regression_model_compressed.pkl
```

The full 1.1 GB uncompressed `models/best_regression_model.pkl` is not used or
shipped.

---

## Local run

### Prerequisites

- Python **3.12** (the artifacts were saved with Python 3.12 / sklearn 1.9.0)
- **Full profile:** ~2.5 GB free RAM while loading the Random Forest, plus the
  295 MB compressed artifact
- **Lite profile:** ~512 MB RAM; only the small model files and the lite EDA
  sample

**Full profile (default):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python entrypoint.py
```

Make sure `models/best_regression_model_compressed.pkl` is present (see above).

**Free/lite profile:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-free.txt
$env:EMI_PROFILE = "lite"
python entrypoint.py
```

The lite artifacts are already in `models/`; the full 295 MB model is not
needed. If you need to regenerate them, see
[Lite model training](#lite-model-training).

`entrypoint.py` loads and validates the selected artifacts **once**, then starts
Streamlit on `PORT` (default `8501`). Open <http://localhost:8501>. The built-in
health endpoint is <http://localhost:8501/_stcore/health>.

Development alternative (loads artifacts on the first page render, still once
per process):

```powershell
streamlit run app.py
```

### Smoke test

```powershell
python scripts/smoke_test.py
```

The smoke test loads the artifacts, validates a median applicant, runs both
models twice, checks determinism, and asserts that invalid inputs produce clear
validation errors. For the lite profile:

```powershell
$env:EMI_PROFILE = "lite"
python scripts/smoke_test.py
```

### Troubleshooting

- **`InconsistentVersionWarning` / unpickle failure:** use Python 3.12 and the
  pinned scikit-learn 1.9.0. Do not upgrade scikit-learn independently of the
  artifacts.
- **pip reports no matching distribution for Pillow:** install
  `pillow==12.3.0` first, then `pip install -r requirements.txt`. The Dockerfile
  does this in a separate layer; a combined resolver run occasionally
  misreports the available wheel on some pip/BuildKit combinations.
- **Container killed during startup:** the full Random Forest needs ~1.25 GB RSS
  and peaked at ~2.3 GB while unpickling. Use a 4 GB instance, or use the lite
  profile (`Dockerfile.free` + `EMI_PROFILE=lite`).
- **`best_regression_model_compressed.pkl` missing:** run
  `scripts/fetch_model.py` with the URL and SHA-256 above.
- **Model download timeout in Docker:** ensure the build host can reach
  `huggingface.co`; the fetch script retries three times and verifies the hash.
- **Lite predictions differ from the original demo:** expected. The lite pair is
  a new, leakage-free model; see its held-out metrics on the Model Performance
  page.

### Rebuilding EDA assets (optional)

Only needed if `data/emi_prediction_cleaned.csv` is present and you want to
regenerate the samples/summary:

```powershell
python scripts/prepare_assets.py
```

### Lite model training

The free/lite artifacts are already committed. To retrain them (requires the
full cleaned CSV):

```powershell
python scripts/train_lite_models.py
```

This trains a compact `HistGradientBoostingClassifier` and
`HistGradientBoostingRegressor` on 35 features, excluding the target-derived
`affordability_ratio`. It writes:

```text
models/lite_classification_model.pkl
models/lite_regression_model.pkl
models/lite_feature_info.json
models/lite_metrics.json
```

Held-out test results from the current artifacts:

| Lite model | Metric |
|---|---:|
| Classification | Accuracy 0.9804, F1 0.9795, ROC-AUC 0.9991 |
| Regression | RMSE ₹595.66, MAE ₹200.22, R² 0.9940, MAPE 7.12% |

These are honest, leakage-free metrics. They are lower than the full model's
reported regression R², but the full number is inflated by target leakage.

---

## Docker

Build:

```powershell
docker build -t emi-predict-ai .
```

Run (local test on port 8501):

```powershell
docker run --rm -p 8501:8501 -e PORT=8501 emi-predict-ai
```

Then:

```powershell
curl.exe http://localhost:8501/_stcore/health
# ok
```

**Free/lite image (512 MB):**

```powershell
docker build -f Dockerfile.free -t emi-predict-ai:free .
docker run --rm -p 8501:8501 -e PORT=8501 emi-predict-ai:free
```

The free image installs no XGBoost and does not contain or download the 295 MB
full model. It was validated under a 512 MB container memory limit.

The image:

- uses `python:3.12-slim-bookworm`;
- installs only runtime dependencies (no compiler toolchain);
- creates and runs as UID/GID `10001` (`app`), not root;
- preloads both model pipelines before Streamlit starts;
- binds to `0.0.0.0:$PORT` (Render injects `PORT`);
- exposes a Docker `HEALTHCHECK` against `/_stcore/health`;
- has a build-time SHA-256 check for the large model.

**Memory:** the compressed Random Forest expands to ~1.25 GB RSS. joblib's
unpickling peaked at **~2.3 GB** during local measurement. Use at least a 4 GB
instance; a 2 GB instance may OOM during startup.

### Build without a local model file

On a clean clone the artifact is absent. The Docker build downloads and verifies
it from Hugging Face. To use a mirror:

```powershell
docker build -t emi-predict-ai . `
  --build-arg REGRESSION_MODEL_URL="https://your-mirror/model.pkl" `
  --build-arg REGRESSION_MODEL_SHA256="<sha256>"
```

If the host is private, add its credentials as a BuildKit secret or pre-place
the file in `models/`; do not bake tokens into the Dockerfile.

---

## Render deployment

1. **Push the repository to GitHub.** The 295 MB regression artifact does not
   need to be committed: the Docker build downloads it and verifies its hash.
2. In Render, choose **New → Blueprint**, select the repository, and apply
   `render.yaml`.
3. Confirm these settings in the Render dashboard:
   - **Runtime:** Docker
   - **Dockerfile path:** `./Dockerfile`
   - **Start command:** `python entrypoint.py` (from `dockerCommand`)
   - **Health check path:** `/_stcore/health`
   - **Plan:** `2c-4g` (2 CPU / 4 GB). This is the minimum safe plan for the
     measured load peak. If you have measured a lower footprint, `1c-2g`
     (2 GB) may work, but it is risky.
   - **Region:** your choice (defaults to Oregon)
   - **Auto-deploy:** on commit
   - **Instances:** 1
   - **Persistent disk:** not required (the app is stateless; history is
     session-only)
4. No application secrets are required for the public artifact. The app does
   not hardcode secrets. If you later use a private artifact host, create a
   Render secret and adapt the build; do not commit credentials.
5. Wait for the first build. Expect a larger initial build because it downloads
   the 295 MB artifact and installs ~150 MB of Python wheels. The container then
   spends roughly 6–10 seconds loading the pipelines before Streamlit accepts
   connections. The image's Docker `HEALTHCHECK` uses a 120-second start period
   so a slow first load is not marked unhealthy.
6. Open the service URL, go to **System Health**, and run the prediction smoke
   test. Also verify `https://<service>.onrender.com/_stcore/health` returns
   `ok`.

### Render Free (no card) — lite profile

Use this path if you do not want to add a payment method. It runs the compact,
leakage-free models on Render's **0.1 CPU / 512 MB Free** instance.

**Option 1 — Blueprint:**

1. In Render, choose **New → Blueprint** and select this repository.
2. When prompted for the Blueprint path, use `render-free.yaml` (Render may
   auto-detect only `render.yaml`; the dashboard lets you choose the file, or
   create the service manually below).
3. Confirm:
   - Runtime: **Docker**
   - Dockerfile: `./Dockerfile.free`
   - Plan: **Free**
   - Health check: `/_stcore/health`
   - Env: `EMI_PROFILE=lite`, `EMI_ENV=production`, `EMI_LOG_LEVEL=INFO`
4. Create the service. No card is required on the Free plan.

**Option 2 — normal Web Service (recommended if the dashboard only reads
`render.yaml`):**

1. **New → Web Service** → connect the GitHub repo.
2. Branch `main`.
3. Runtime **Docker**; Dockerfile path `./Dockerfile.free`; context `.`.
4. Instance type **Free**.
5. Health check path `/_stcore/health`.
6. Add `EMI_PROFILE=lite` (plus optional `EMI_ENV`, `EMI_LOG_LEVEL`).
7. Create.

The free image does **not** install XGBoost or download the 295 MB model. It
starts quickly and contains only the compact models. Render Free still sleeps
after 15 minutes of inactivity and takes about a minute to wake.

If Render insists on a card even when **Free** is selected, that is an
account/region verification restriction rather than a project setting; use a
Hugging Face Docker Space instead.

### Render notes

- Render injects `PORT`; the app binds to it. Do not define `PORT` manually.
- The service is stateless. Redeploys/restarts lose session history (by design).
- The Docker build requires outbound access to Docker Hub, PyPI, and the
  Hugging Face URL during the first build. If your build environment blocks
  Hugging Face, pre-place the artifact via Git LFS or a private mirror.
- Render's free/512 MB instances cannot load the **full** model. Use the
  `render-free.yaml` lite profile there, or the 4 GB plan for the full model.
- If you change `render.yaml` to a plan ID your workspace cannot use, the
  Blueprint sync will fail; confirm plan availability/pricing in the dashboard.

---

## Model limitation: target leakage

This section describes the **full profile**. The free/lite profile uses a new
compact model trained without the leaky feature, so it does not have this
problem.

The training notebook created this feature **before** the train/test split:

```python
df["affordability_ratio"] = df["max_monthly_emi"] / df["monthly_salary"]
```

`max_monthly_emi` is the regression target, and `affordability_ratio` was then
included in both feature matrices. The saved Random Forest assigns ~15% of its
importance to this feature; `remaining_income` is the dominant feature (~79%).
The reported regression R² of 0.998 is therefore **optimistic and not
reproducible in production**.

The **full profile** does not retrain the saved models. At inference it:

1. Starts with a transparent estimate of 44% of positive remaining income over
   salary.
2. Runs the saved regression model.
3. Sets `affordability_ratio = predicted_emi / salary`.
4. Repeats until stable (max 10 iterations).
5. Runs classification and the final regression with that reconstructed ratio.

The result is internally consistent with the saved pipeline, but it is an
approximation. The UI flags non-converged ratios and low classification
confidence.

The **free/lite profile** is the leakage-free alternative: it trains a compact
HistGradientBoosting pair on 35 features with `affordability_ratio` excluded.
Its held-out R² is 0.9940 with MAPE 7.12%, which is lower than the full model's
leaky 0.998 but honest and reproducible. It is a new model, not a modification
of the original artifacts.

Other caveats:

- The dataset appears synthetic/simulated.
- No fairness, calibration, or out-of-time stability analysis is in the saved
  reports.
- `credit_score` is validated to the standard 300–900 range. The cleaned data
  contains a few 0/1200 values; those are treated as data-quality issues.
- Requested EMI is **principal-only** (`amount ÷ tenure`). No interest rate is
  present in the training data.
- This is decision support, not an automated credit decision.

---

## Preprocessing fidelity

The full pipelines contain their own `SimpleImputer`, `StandardScaler`, and
`OneHotEncoder`; the lite pipelines contain an imputer and one-hot encoder (the
tree models do not need scaling). The app passes raw values and lets each
pipeline transform them. The notebook's 11 engineered features are reproduced
exactly:

```text
total_monthly_expenses = monthly_rent + school_fees + college_fees
                       + travel_expenses + groceries_utilities
                       + other_monthly_expenses
total_debt_burden      = current_emi_amount + total_monthly_expenses
remaining_income       = monthly_salary - total_debt_burden
debt_to_income_ratio   = total_debt_burden / monthly_salary
expense_to_income_ratio= total_monthly_expenses / monthly_salary
current_emi_to_income  = current_emi_amount / monthly_salary
loan_to_salary_ratio   = requested_amount / monthly_salary
balance_to_salary_ratio= bank_balance / monthly_salary
emergency_coverage_ratio = emergency_fund / total_monthly_expenses
estimated_requested_emi  = requested_amount / requested_tenure
affordability_ratio    = reconstructed iteratively (see above)
```

Zero denominators produce NaN, mirroring the notebook's
`replace(0, np.nan)` + inf→NaN step. Missing numeric values are filled with the
saved pipeline's own imputer statistics. Categorical options are read from the
saved encoder, not hardcoded — this fixed the old UI's invalid `company_type`
values (`Medium`, `Large`); the model's actual categories are `Large Indian`,
`MNC`, `Mid-size`, `Small`, and `Startup`.

---

## Saved model results (as reported, not re-evaluated)

### Full profile — original saved artifacts

These are the original notebook results. The regression numbers are inflated by
target leakage; read the caveat above.

| Classification | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost (best) | 0.9884 | 0.9881 | 0.9884 | 0.9882 | 0.9997 |
| Random Forest | 0.9605 | 0.9574 | 0.9605 | 0.9489 | 0.9982 |
| Logistic Regression | 0.9528 | 0.9445 | 0.9528 | 0.9456 | 0.9942 |

| Regression | RMSE | MAE | R² | MAPE |
|---|---:|---:|---:|---:|
| Random Forest (best) | 330.76 | 63.22 | 0.9982 | 1.05% |
| XGBoost | 385.91 | 101.76 | 0.9975 | 2.00% |
| Linear Regression | 3466.83 | 2198.92 | 0.7995 | 134.24% |

Held-out test cards come from `reports/final_*_metrics.csv`; comparison tables
come from the validation run.

### Free/lite profile — new leakage-free model

Held-out test results from `python scripts/train_lite_models.py`:

| Lite model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| HistGradientBoostingClassifier | 0.9804 | 0.9795 | 0.9991 |

| Lite model | RMSE | MAE | R² | MAPE |
|---|---:|---:|---:|---:|
| HistGradientBoostingRegressor | ₹595.66 | ₹200.22 | 0.9940 | 7.12% |

The lite metrics are lower than the full regression numbers, but they are
honest and do not depend on the target. They come from a 15% held-out test set.

---

## Health checks

| Check | Path / action |
|---|---|
| Server liveness (Render) | `GET /_stcore/health` → `ok` |
| Artifact load + metadata checks | **System Health** page |
| Real inference smoke test | **System Health → Run prediction smoke test** |
| CI/local smoke test | `python scripts/smoke_test.py` |

`/_stcore/health` is intentionally lightweight so Render can validate a deploy
without running the 295 MB model again. The Health page checks that both
pipelines are loaded, the feature count is 36, the classes are
`Eligible/High_Risk/Not_Eligible`, categorical metadata is present, and the EDA
and report assets exist.

---

## Artifact and path issues resolved

1. **Wrong scikit-learn version.** Loading with the system Python 3.9/1.5.1
   emitted an `InconsistentVersionWarning`; the label encoder pickle records
   sklearn 1.9.0. Pinned 1.9.0 and validated with a matching Python 3.12 env.
2. **Huge regression artifact.** The usable artifact is the 295 MB compressed
   pipeline, not the 1.1 GB uncompressed file. The loader prefers the compressed
   file and falls back to uncompressed only if necessary.
3. **Target leakage.** `affordability_ratio` was derived from the regression
   target. Added a documented iterative reconstruction and removed any claim
   that the saved metrics are production accuracy.
4. **Invalid categorical options.** The old UI offered `Medium`/`Large` company
   types, which were silently ignored by the encoder's `handle_unknown` policy.
   The new UI reads the actual encoder categories.
5. **Runtime downloads / lazy loading.** The old app downloaded the model from
   Hugging Face on demand and used `@st.cache_resource` lazily. The new
   `entrypoint.py` preloads and validates everything before serving, and the
   Docker build downloads/verifies the artifact once.
6. **Huge data in the image.** The 75 MB raw CSV and 143 MB cleaned CSV are
   excluded. The full profile ships a 5.8 MB sample; the lite profile ships a
   1.9 MB sample. Both use the 12 KB full-dataset summary.
7. **Non-durable CRUD.** Removed the CSV write path; history is session-only.
8. **CPU-only XGBoost.** `xgboost-cpu==3.4.1` avoids the NCCL/CUDA dependency
   pulled by the full wheel while preserving 3.4.1 pickle compatibility.
9. **Path anchoring.** All paths are resolved from the package root
   (`Path(__file__).resolve()`), so the app works from any working directory and
   inside `/app`.

---

## Validation performed in this workspace

- Parsed both pickles with the matching environment; classification is
  `XGBoost` + a 36-feature preprocessing pipeline, regression is
  `RandomForestRegressor` + the same 36 features.
- Confirmed the Hugging Face LFS object SHA-256 matches the local artifact.
- `python scripts/smoke_test.py` **PASSED**: default applicant → `Eligible`,
  max EMI ₹13,198.96, requested EMI ₹9,440.00, 3 ratio iterations, ~100 ms.
- Streamlit `AppTest` page checks passed for all pages; the prediction form
  submitted successfully and rendered the expected metric cards.
- Built the Docker image with Windows' built-in WSL container engine (`wslc`,
  BuildKit). Docker Desktop is not installed in this workspace, but the same
  Dockerfile and standard Docker build semantics are used. The clean-context
  image is **1.11 GB**; a cached local-context build measured 1.4 GB due to
  layer layout. Both include the 295 MB (281.5 MiB) model.
- Ran the image as UID/GID **10001** with `PORT=10000` (Render convention),
  published to host port 8512. `GET /_stcore/health` returned **200 `ok`** and
  the home page returned **200**.
- Ran a real prediction inside the container (not just on the host):
  `Eligible`, max EMI **₹13,198.96**, requested EMI **₹9,440.00**, 3 ratio
  iterations, **97.6 ms**, `xgboost-cpu` **3.4.1**. This confirms the CPU-only
  XGBoost distribution unpickles the saved classifier and produces the same
  result as the training environment.
- Clean-context build (the 295 MB model removed, like a fresh Render checkout):
  the Docker build downloaded the artifact from Hugging Face, printed
  `Downloaded 295,141,016 bytes; SHA-256: a321224177d43c5e7de64c8bc98fa0f15689240ba7f521c29139ca412cd9e401`,
  and passed the same prediction smoke test. This is the exact path Render uses.
- Trained the free/lite pair on 404,800 rows with `affordability_ratio`
  excluded. Held-out test: classification F1 **0.9795**, ROC-AUC **0.9991**;
  regression R² **0.9940**, RMSE **₹595.66**, MAPE **7.12%**. Artifact sizes:
  **0.96 MB** classifier, **0.32 MB** regressor.
- Built `Dockerfile.free` (image **757 MB**) and ran it with a hard
  **`-m 512M`** memory limit:
  - one-off prediction succeeded as UID 10001 in **27 ms**;
  - the Streamlit server returned **200 `ok`** at `/_stcore/health` and **200**
    for the home page;
  - AppTest passed for the prediction page (including form submit) and the EDA
    page under the same 512 MB limit.
  This is the target Render Free environment.

---

## Security and operations

- Runs as non-root UID/GID 10001.
- No secrets or `.env` files are baked into the image.
- `.dockerignore` excludes `.git`, virtualenvs, notebooks, raw data, the 1.1 GB
  model, and debug scratch files.
- The app reads fixed artifact paths; no user-supplied filesystem paths.
- Applicant data is handled in memory/session state only; no database and no
  PII persistence.
- Logs include model names, timings, and decisions, not the full applicant
  payload.
- Joblib/pickle artifacts can execute code when loaded. Only use artifacts from
  a trusted source; this project verifies the large downloaded artifact's
  SHA-256 before loading it.

---

## License / provenance

Model artifacts and report files are the project owner's trained outputs. The
dataset appears synthetic; do not treat predictions as financial advice.
