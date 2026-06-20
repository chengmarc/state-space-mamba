"""Central configuration — the contract shared *between* pipeline stages.

This module is the single source of truth for everything that crosses a stage
boundary: directory locations, the artifact filenames each stage hands to the
next, and the horizon / model / training hyperparameters used by more than one
stage.

Stage-*internal* structural config stays in its own stage on purpose — e.g. the
feature registry in ``step_3_model_input`` and the oscillator/detrend tables in
``step_2_feature_engineering``. Those are local to one stage's logic, not a knob
shared across the pipeline, so centralising them here would only hurt locality.

Pipeline data flow (each stage reads the previous stage's artifacts)::

    step_1_data_ingestion      ──▶  ODS_CSV
    step_2_feature_engineering ──▶  DWH_CSV, TREND_PARAMS_JSON        (reads ODS_CSV)
    step_3_model_input         ──▶  MODEL_INPUT_CSV, NORM_PARAMS_JSON (reads DWH_CSV)
    step_4_train               ──▶  CHECKPOINT, TRAIN_META_JSON       (reads MODEL_INPUT_CSV)
    step_5_evaluate            ──▶  figures + metrics                 (reads all of the above)
"""

from pathlib import Path

# ── directories ──────────────────────────────────────────────────────────────
# config.py lives at <root>/src/ssm/config.py, so the repo root is parents[2].
ROOT           = Path(__file__).resolve().parents[2]
OUTPUT         = ROOT / "output"
CHECKPOINT_DIR = OUTPUT / "checkpoints"

# ── stage artifacts (the inter-stage contract) ───────────────────────────────
ODS_CSV           = OUTPUT / "step1_ods.csv"
DWH_CSV           = OUTPUT / "step2_dwh.csv"
MODEL_INPUT_CSV   = OUTPUT / "step3_model_input.csv"
TREND_PARAMS_JSON = OUTPUT / "trend_params.json"
NORM_PARAMS_JSON  = OUTPUT / "norm_params.json"
CHECKPOINT        = CHECKPOINT_DIR / "MambaSSM_best.pt"
TRAIN_META_JSON   = OUTPUT / "train_meta.json"

# ── horizons (windowing contract shared by train + evaluate) ─────────────────
TEST_DAYS        = 365   # tail held out entirely — never seen during training
HISTORIC_HORIZON = 730   # lookback window fed to the model
FORECAST_HORIZON = 90    # prediction horizon

# ── training hyperparameters ─────────────────────────────────────────────────
EPOCHS     = 1000
PATIENCE   = 30          # early-stopping: epochs without val improvement before stopping
LR         = 1e-3
BATCH_SIZE = 32
VAL_SPLIT  = 0.1         # last fraction of windows held out for validation

# ── model hyperparameters ────────────────────────────────────────────────────
D_MODEL = 32
N_LAYER = 8


def ensure_dirs() -> None:
    """Create the output directories used by the pipeline if they don't exist."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
