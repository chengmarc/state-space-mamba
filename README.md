[![python - v3.12.10](https://img.shields.io/static/v1?label=python&message=v3.12.10&color=blue&logo=python&logoColor=white)](https://)
[![cuda - v12.8](https://img.shields.io/static/v1?label=cuda&message=v12.8&color=green&logo=nvidia&logoColor=white)](https://)
[![torch - v2.11.0](https://img.shields.io/static/v1?label=torch&message=v2.11.0&color=orange&logo=pytorch&logoColor=white)](https://)

# Sequence-to-Sequence Time Series Forecasting with a Selective State-Space Model

**Task:** Multi-feature BTC time series forecasting &nbsp;|&nbsp; **Model:** MambaSSM

---

## Abstract

This project implements a time series forecasting pipeline using multi-feature inputs and a **selective state-space model (MambaSSM)**. The central challenge is the **naive lag-1 baseline**: a model that learns to output $\hat{Y}_{t+1} \approx Y_t$ achieves deceptively low error on trending series while carrying no genuine predictive signal.

The pipeline breaks lag-1 degeneracy at three levels:

- **Detrending** — a logarithmic trend is fitted and removed, eliminating the smooth autocorrelation that makes lag-copying cheap to learn
- **Multi-feature inputs** — on-chain, macro, and cycle signals impose orthogonal constraints that a univariate copier cannot satisfy
- **Architectural selectivity** — MambaSSM's input-dependent gating discourages trivial state propagation across timesteps

The trained model achieved a best val_loss of **0.001213 at epoch 863**, beating the copy-lag-1 MSE baseline of 0.001312. This is a significant result: **the model predicts tomorrow's residual more accurately than "copy today's value" — despite never seeing any price data from the last 30 days.** Rather than exploiting short-term autocorrelation, the model has learned genuine structure from cycle position, volatility regime, and macro context alone.

A range of recurrent and attention architectures (SegRNN, LSTM, Seq2Seq LSTM, Attention LSTM, Transformer) were explored and are summarised under [Prior Work](#6-prior-work); the maintained codebase focuses on MambaSSM, which gave the best performance and strongest resistance to lag-1 degeneracy.

---

## 1. Methodology

### 1.1 Logarithmic Trend Extraction

The underlying price trend is approximated by a logarithmic function:

$$Y_t = a + b \cdot \log(X_t + c)$$

- $X_t$ — integer-encoded time index (days since data start)
- $(a, b, c)$ — estimated via non-linear least squares on training data only
- $c$ is fitted once globally; $a$ and $b$ are updated cumulatively at each row via expanding-window OLS, so no future data leaks into the residual

This form handles exponential-like growth while remaining numerically stable for small $X_t$. By removing the trend explicitly, the model cannot exploit slow-moving autocorrelation as a shortcut.

### 1.2 Residual Computation

$$R_t = Y_t - \hat{Y}_t$$

- Isolates stationary short-term fluctuations from the global drift
- The residual oscillates around zero — mean-reversion is the dominant regime
- All downstream modelling operates on $R_t$, not on raw prices

### 1.3 Multi-Feature Residual Forecasting

The model predicts the log-price residual at t+1 from a sparse lookback of multi-feature inputs. The feature set is assembled in [stage 3](#4-pipeline) and comprises 7 input channels plus the target:

| Group | Features |
|---|---|
| Cycle | years since last halving |
| Volatility | 30-day realized volatility (z-scored) |
| Macro | DXY 30-day return (×10), DXY 100-day return (×10) |
| Technical | Williams %R short (21d, EMA-3), Williams %R long (112d, EMA-3), WR composite |
| **Target** | **log-price residual** |

For each anchor day $t$, the input is a set of non-contiguous snapshots at fixed offsets (currently `[365, 180, 90, 75, 65, 55, 30]` days before $t$), giving an input tensor of shape `(batch, n_slices, n_features)`. The learned mapping is:

$$\left( F_{t - s_1}, \ldots, F_{t - s_k} \right) \rightarrow R_{t+1}$$

where $s_1 > \ldots > s_k$ are the slice offsets. The minimum offset is **30 days** — the model has no access to recent prices, so any copy strategy is at least 30 days stale.

<div align="center"><img src="./output/step2_fig2_signals.png" width="90%"></div>

The three signal groups over the full data history:

- **Top — 30-day realized volatility (annualised %):** the dashed red line at 18.31% marks the threshold below which long-horizon win rates exceed 89% (established in the companion winrate-matrix analysis). Volatility spikes identify capitulation and euphoria phases with strong mean-reversion signal at multi-month horizons.
- **Middle — DXY 30-day and 100-day returns:** negative USD momentum historically anti-correlates with BTC strength. The 100-day series captures longer regime shifts that a short window would miss.
- **Bottom — years since last halving:** positions the current date within the 4-year supply-emission cycle, distinguishing accumulation from distribution phases.

### 1.4 Inverse Transformation

Final price predictions are reconstructed by adding the forecasted residuals back to the trend:

$$\hat{P}_{t+k} = \exp\!\left(\hat{R}_{t+k} + \hat{Y}_{t+k}\right)$$

- The trend parameters fitted in stage 2 are persisted to `trend_params.json`
- Stage 5 loads them to extrapolate the trend forward and reconstruct USD prices for the full 4-year rollout

---

## 2. Model

**MambaSSM** is a structured state-space model with selective state transitions:

- **Linear-time complexity** — recurrent structure scales with sequence length, not quadratically
- **Input-dependent gating** — the selection mechanism learns which historical context to propagate, providing architecture-level resistance to trivial lag-1 state copying
- **Log-domain scan** — the `logcumsumexp` selective scan kernel is numerically stable for long sequences

The model maps a sparse lookback of shape `(batch, n_slices, n_features)` to a prediction of shape `(batch, predict_window)`, reading the final hidden state after processing all slices.

**Mamba Block** — selective SSM with dual-branch projection, depthwise convolution, and SiLU gating ([1, 2]):

<div align="center"><img src="./mamba.png" width="60%"></div>

---

## 3. Repository Structure

```
src/ssm/                         # importable package
  config.py                      # cross-stage contract: paths, artifact names, horizons, hyperparameters
  arch/mamba.py                  # MambaSSM model + create_model factory
  arch/scans.py                  # selective-scan kernels
  data/loader.py                 # sliding-window DataLoader construction
pipeline/
  step_1_data_ingestion.py       # raw sources  -> ODS
  step_2_feature_engineering.py  # ODS          -> DWD (features) + trend params
  step_3_dm.py                   # DWD          -> DM + norm params
  step_4_train.py                # DM           -> checkpoint + train metadata
  step_5_evaluate.py             # checkpoint   -> metrics + figures
output/                          # generated artifacts (git-ignored)
```

Configuration is split by locality:

- **`src/ssm/config.py`** — everything that crosses a stage boundary: directory paths, artifact filenames, slice offsets, model and training hyperparameters
- **Stage-local config** — the feature registry in stage 3 and the oscillator tables in stage 2 stay in their own files; they are not shared across stages

### Setup

```bash
pip install -e .
```

Installs the `ssm` package in editable mode (`src/` layout) so all `pipeline/` scripts can `import ssm` without path hacks.

---

## 4. Pipeline

Each stage reads the previous stage's artifacts from `output/` and writes its own. Artifact filenames are the inter-stage contract, defined once in `config.py`.

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 1 — Ingestion | `pipeline/step_1_data_ingestion.py` | CoinMetrics, yfinance | `step1_ods.csv` |
| 2 — Features | `pipeline/step_2_feature_engineering.py` | `step1_ods.csv` | `step2_dwd.csv`, `trend_params.json`, figures |
| 3 — DM | `pipeline/step_3_dm.py` | `step2_dwd.csv` | `step3_dm.csv`, `norm_params.json`, figure |
| 4 — Train | `pipeline/step_4_train.py` | `step3_dm.csv` | `checkpoints/MambaSSM_best.pt`, `train_meta.json` |
| 5 — Evaluate | `pipeline/step_5_evaluate.py` | all of the above | metrics + figures |

<div align="center"><img src="./output/step3_fig_distributions.png" width="90%"></div>

Stage 3 produces the distribution plot above — one histogram per DM feature after normalization, four per row. Colour encodes the transform:

- **Orange — z-scored:** mean and variance fitted on training rows only, then applied to the full series; the held-out tail is normalized with training statistics and is not re-centred
- **Green — ×10 linear scale:** applied to DXY return features whose raw values (~0.02) sit two orders of magnitude below the residual; the fixed scale preserves threshold semantics without data-dependent fitting
- **Blue — no transformation:** features already on a natural or bounded scale (`years_since_halving`, Williams %R signals, and the `log_price_residual` target)

```bash
python pipeline/step_1_data_ingestion.py
python pipeline/step_2_feature_engineering.py
python pipeline/step_3_dm.py
python pipeline/step_4_train.py
python pipeline/step_5_evaluate.py
```

Stages 1–3 are CPU/network bound; stage 4 uses CUDA if available and falls back to CPU.

---

## 5. Results

### Training

- **Loss:** MSE on `log_price_residual` at t+1
- **Input:** 7 slices per anchor, nearest at 30 days — no access to recent prices
- **Split:** last 365 calendar days held out as the test window; remainder randomly split 85/15 train/val

### MSE Baselines (training window, 3 815 rows)

Reference points for interpreting `val_loss` during stage 4:

| Baseline | MSE | Description |
|---|---|---|
| **Copy-lag-1** | **0.001312** | Predict residual[t+1] = residual[t]. Trivially achievable by memorising yesterday — the absolute floor. |
| **Copy-lag-30** | **0.043819** | Predict residual[t+1] = residual[t−30]. The cheapest copy the model can make from its nearest input slice. |

Interpreting `val_loss`:

- **val_loss ≫ 0.0438** — model is not yet extracting useful signal from any slice
- **val_loss well below 0.0438, above 0.0013** — model is using multi-feature context beyond a nearest-slice copy; target operating range
- **val_loss approaching 0.0013** — model is achieving lag-1-equivalent accuracy using only 30-day-old information; strong result given it has no access to recent prices

**Trained result: val_loss = 0.001213 at epoch 863/1000** — below the copy-lag-1 floor, using inputs whose nearest snapshot is 30 days old.

### Evaluation (stage 5)

Stage 5 runs an autoregressive rollout through the 365-day held-out test window:

- Each predicted residual is fed back as input context for the next step
- All other feature columns (volatility, macro, cycle, technical) use real observed data throughout
- No synthetic future — the evaluation is grounded in real market conditions

Metrics reported on the held-out test window:

- **MAE** on the residual series
- **Directional accuracy** vs the previous day's residual
- **Copy-lag-30 MSE** as the naive baseline

Price reconstruction adds the fitted log trend back to the predicted residuals, giving a USD price trajectory over the test window.

---

## 6. Prior Work

Earlier in this project a set of recurrent and attention architectures were explored on the same forecasting task. They are not part of the maintained codebase but are summarised here for context.

| Model | Description |
|---|---|
| SegRNN [3] | Segment-wise RNN that partitions input sequences into fixed-length segments. |
| Simple LSTM [4] | Vanilla LSTM trained directly on residual sequences over fixed-length windows. |
| Seq2Seq LSTM [5] | Encoder-decoder LSTM for multi-step forecasting. |
| Attention LSTM | LSTM augmented with additive attention over past hidden states. |
| Transformer [6] | Fully attention-based architecture using multi-head self-attention. |

RNN/LSTM baselines showed the most visible lag-1 tendency; Transformer and Mamba models were substantially more resistant.

---

## 7. References

[1] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*. https://arxiv.org/abs/2312.00752

[2] Gu, A., Goel, K., & Ré, C. (2021). Efficiently Modeling Long Sequences with Structured State Spaces. *ICLR 2022*. https://arxiv.org/abs/2111.00396

[3] Lin, S., Lin, W., Wu, W., Zhao, F., Mo, R., & Zhang, H. (2023). SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting. *arXiv:2308.11200*. https://arxiv.org/abs/2308.11200

[4] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9*(8), 1735–1780. https://www.bioinf.jku.at/publications/older/2604.pdf

[5] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS 2014*. https://arxiv.org/abs/1409.3215

[6] Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762

---

## Implementation Note

The MambaSSM block implementation (`src/ssm/arch/mamba.py`, `src/ssm/arch/scans.py`) is adapted from [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal), a community reference implementation of [1]:

- Selective scan kernel, RMSNorm, and block structure follow that reference
- Forecasting wrapper, input/output projection layers, training loop, and data pipeline are original
