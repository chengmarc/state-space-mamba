[![python - v3.12](https://img.shields.io/static/v1?label=python&message=v3.12&color=blue&logo=python&logoColor=white)](https://)
[![cuda - v12.1](https://img.shields.io/static/v1?label=cuda&message=v12.1&color=green&logo=nvidia&logoColor=white)](https://)
[![torch - v2.5+](https://img.shields.io/static/v1?label=torch&message=v2.5%2B&color=orange&logo=pytorch&logoColor=white)](https://)

# Sequence-to-Sequence Time Series Forecasting with a Selective State-Space Model

**Task:** Multi-feature BTC time series forecasting &nbsp;|&nbsp; **Model:** MambaSSM

---

## Abstract

This project implements a sequence-to-sequence time series forecasting pipeline using multi-feature inputs and a **selective state-space model (MambaSSM)**. A central failure mode in financial time series forecasting is the **naive lag-1 baseline**, where a model learns to approximate the next value as the current value — producing predictions that appear correlated with ground truth but carry no genuine predictive signal.

The pipeline addresses the lag-1 problem through three complementary mechanisms: a logarithmic detrending stage that removes the smooth trend component driving autocorrelation, multi-feature on-chain and macro inputs that introduce orthogonal signals beyond price history, and architecture-level selectivity in MambaSSM that discourages trivial state copying across timesteps.

A range of recurrent and attention architectures (SegRNN, LSTM, Seq2Seq LSTM, Attention LSTM, Transformer) were explored during the course of this work and are summarised under [Prior Work](#6-prior-work); the maintained codebase here focuses on MambaSSM, which gave the best forecasting performance and the strongest resistance to lag-1 degeneracy.

---

## 1. Methodology

A persistent failure mode in financial time series forecasting is the **naive lag-1 baseline**: a model that outputs $\hat{Y}_{t+1} \approx Y_t$ achieves deceptively low error on non-stationary series driven by smooth trends, while providing no genuine predictive value. The methodology is structured to break this degeneracy at three levels.

### 1.1 Logarithmic Trend Extraction

Given an observation <code>Y<sub>t</sub></code>, the underlying trend is approximated by fitting a logarithmic function of the form:

$$Y_t = a + b \cdot \log(X_t + c)$$

where <code>X<sub>t</sub></code> is the integer-encoded time index and <code>(a, b, c)</code> are estimated via non-linear least squares. This form is well-suited to data exhibiting exponential-like growth and remains numerically stable for small <code>X<sub>t</sub></code>.

By explicitly fitting and removing the trend, the model is prevented from exploiting slow-moving autocorrelation as a shortcut — the primary mechanism through which lag-1 degeneracy arises in price series.

### 1.2 Residual Computation

The residual component is obtained by subtracting the fitted trend:

$$R_t = Y_t - \hat{Y}_t$$

This isolates stationary short-term fluctuations from the global trend, allowing the model to focus on residual dynamics rather than long-run drift. An example of the transformation on a single feature is shown below.

<div align="center"><img src="./method.png" width="60%"></div>

### 1.3 Multi-Feature Residual Forecasting

Rather than predicting absolute values, the model is trained to forecast the **log-price residual** from a window of multi-feature inputs. The feature set is assembled in [stage 3](#4-pipeline) and currently comprises 11 input channels plus the target:

| Group | Features |
|---|---|
| On-chain (supply-detrended residuals) | log hash rate, log transaction count, log active addresses |
| Valuation | MVRV ratio (z-scored) |
| Technical oscillators | Williams %R (short / long), Normalized Maximum Drawdown (90 / 365 / 730-day) |
| Volatility | 30-day realized volatility (z-scored) |
| Macro | CPI year-over-year (z-scored) |
| **Target** | **log-price residual** |

Given <code>k</code> input features, the learned mapping takes the form:

$$\sum_{i=1}^{k} \left( R_{t-n}^{(i)}, \ldots, R_{t}^{(i)} \right) \rightarrow \left( R_{t+1}, \ldots, R_{t+m} \right)$$

where <code>n</code> and <code>m</code> denote the lookback (`HISTORIC_HORIZON`) and forecasting (`FORECAST_HORIZON`) horizons. The inclusion of features orthogonal to price history is the second anti-lag mechanism: a model that simply copies its last input state cannot satisfy the joint constraint imposed by structurally independent feature channels.

### 1.4 Inverse Transformation

Final price predictions are reconstructed by adding the forecasted residuals back to the trend estimates:

$$\left( Y_{t+1}, \ldots, Y_{t+m} \right) = \left( \hat{Y}_{t+1}, \ldots, \hat{Y}_{t+m} \right) + \left( R_{t+1}, \ldots, R_{t+m} \right)$$

The trend parameters fitted in stage 2 are persisted (`trend_params.json`) so stage 5 can perform this reconstruction.

---

## 2. Model

**MambaSSM** is a structured state-space model with selective state transitions, offering linear-time complexity and strong performance on long-range sequence modeling. The input-dependent selection mechanism explicitly gates which information is propagated across timesteps, providing architecture-level resistance to trivial lag-1 state copying.

The forecasting wrapper maps an input window of shape `(batch, HISTORIC_HORIZON, n_features)` to a forecast of shape `(batch, FORECAST_HORIZON, 1)`, taking the trailing `FORECAST_HORIZON` positions of the final hidden sequence as the prediction.

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
  step_2_feature_engineering.py  # ODS          -> DWH (features) + trend params
  step_3_model_input.py          # DWH          -> model input + norm params
training/
  step_4_train.py                # model input  -> checkpoint + train metadata
evaluation/
  step_5_evaluate.py             # checkpoint   -> metrics + figures
scripts/                         # archived exploratory scripts (not part of the pipeline)
output/                          # generated artifacts (git-ignored)
```

All shared configuration lives in **`src/ssm/config.py`** — directory locations, the artifact filename each stage hands to the next, the windowing horizons, and the model/training hyperparameters. Stage-internal structural config (e.g. the feature registry in stage 3, the oscillator tables in stage 2) stays local to its stage.

### Setup

```bash
pip install -e .
```

This installs the `ssm` package (editable, `src/` layout) so the `pipeline/`, `training/` and `evaluation/` scripts can `import ssm` from anywhere — no path hacks required.

---

## 4. Pipeline

The pipeline is a linear chain of stages; each reads the previous stage's artifacts and writes its own into `output/`. The artifact filenames form the contract between stages and are defined once in `config.py`.

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 1 — Ingestion | `pipeline/step_1_data_ingestion.py` | CoinMetrics, yfinance, FRED | `step1_ods.csv` |
| 2 — Features | `pipeline/step_2_feature_engineering.py` | `step1_ods.csv` | `step2_dwh.csv`, `trend_params.json`, figures |
| 3 — Model input | `pipeline/step_3_model_input.py` | `step2_dwh.csv` | `step3_model_input.csv`, `norm_params.json`, figure |
| 4 — Train | `training/step_4_train.py` | `step3_model_input.csv` | `checkpoints/MambaSSM_best.pt`, `train_meta.json` |
| 5 — Evaluate | `evaluation/step_5_evaluate.py` | all of the above | metrics + figures |

Run the stages in order:

```bash
python pipeline/step_1_data_ingestion.py
python pipeline/step_2_feature_engineering.py
python pipeline/step_3_model_input.py
python training/step_4_train.py
python evaluation/step_5_evaluate.py
```

Stages 1–3 are CPU/network bound; stage 4 uses CUDA if available and falls back to CPU.

---

## 5. Results

MambaSSM produces 30-/90-day forecasts whose predicted sequences avoid the lag-1 pattern — a common failure mode in financial forecasting where model output is visually indistinguishable from a one-day shift of the input series. Rather than tracking a lagged copy of the input, the model produces predictions that diverge from the immediate prior trajectory where the residual signal warrants it.

The log-detrending stage is the first line of defense against this failure mode, eliminating the smooth autocorrelation that makes lag-1 strategies cheap to learn. Multi-feature inputs impose orthogonal constraints that further prevent the model from collapsing to univariate copying. MambaSSM's selective state mechanism provides the final layer of resistance: by learning which historical context to gate rather than uniformly propagating all state, it avoids the recurrent shortcut that causes LSTM-class models to exhibit residual lag.

<div align="center"><img src="./result.png" width="60%"></div>

---

## 6. Prior Work

Earlier in this project a set of recurrent and attention architectures were explored on the same forecasting task. They are not part of the maintained codebase but are summarised here for context, and motivated the move to a selective state-space model.

| Model | Description |
|---|---|
| SegRNN [3] | A segment-wise RNN that partitions input sequences into fixed-length segments. |
| Simple LSTM [4] | A vanilla LSTM trained directly on residual sequences over fixed-length windows. |
| Seq2Seq LSTM [5] | An encoder-decoder LSTM for multi-step forecasting. |
| Attention LSTM | An LSTM augmented with additive attention over past hidden states. |
| Transformer [6] | A fully attention-based architecture using multi-head self-attention. |

RNN/LSTM baselines (including Seq2Seq and Attention variants) showed the most visible lag-1 tendency in their output curves; Transformer and Mamba models were substantially more resistant.

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

The MambaSSM block implementation (`src/ssm/arch/mamba.py`, `src/ssm/arch/scans.py`) is adapted from [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal), a community reference implementation of [1]. The selective scan kernel, RMSNorm, and block structure follow that reference. The forecasting wrapper, input/output projection layers, training loop, and data pipeline are original.
