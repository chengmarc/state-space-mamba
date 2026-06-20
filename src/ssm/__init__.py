"""ssm — Sequence-to-sequence time series forecasting with a selective state-space model.

This package holds the reusable building blocks of the project:

    ssm.config        cross-stage configuration (paths, artifact names, horizons, hyperparameters)
    ssm.arch.mamba    the MambaSSM forecasting model and its `create_model` factory
    ssm.arch.scans    the selective-scan kernel used by the Mamba block
    ssm.data.loader   windowing + DataLoader construction from a feature frame

The runnable pipeline (data ingestion → feature engineering → model input →
training → evaluation) lives in the top-level ``pipeline/``, ``training/`` and
``evaluation/`` directories and imports from this package.
"""
