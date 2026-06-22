"""ssm — Sequence-to-sequence time series forecasting with a selective state-space model.

This package holds the reusable building blocks of the project:

    ssm.config        cross-stage configuration (paths, artifact names, horizons, hyperparameters)
    ssm.artifacts     shared train_meta/checkpoint IO and compatibility helpers
    ssm.arch.mamba    the MambaSSM forecasting model and its `create_model` factory
    ssm.arch.scans    the selective-scan kernel used by the Mamba block
    ssm.data.loader   windowing + DataLoader construction from a feature frame

The runnable pipeline (data ingestion → feature engineering → DM assembly →
training → evaluation) lives entirely in ``pipeline/`` and imports from this package.
"""
