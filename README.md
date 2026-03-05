[![python - v3.12.8](https://img.shields.io/static/v1?label=python&message=v3.12.8&color=blue&logo=python&logoColor=white)](https://)
[![cuda - v12.1](https://img.shields.io/static/v1?label=cuda&message=v12.1&color=green&logo=nvidia&logoColor=white)](https://)
[![torch - v2.5.1](https://img.shields.io/static/v1?label=torch&message=v2.5.1&color=orange&logo=pytorch&logoColor=white)](https://)

# Sequence-to-Sequence Time Series Forecasting with Multi-Feature Inputs

**Task:** Time Series Forecasting &nbsp;|&nbsp; **Architectures:** RNN, LSTM, Transformer, Mamba

---

## Abstract

This project investigates sequence-to-sequence time series forecasting using multi-feature inputs across a range of deep learning architectures. A central contribution is a detrending pipeline based on logarithmic trend extraction and residual forecasting, which improves stationarity and prediction stability. Experiments compare six architectures — SegRNN, LSTM, Seq2Seq LSTM, Attention LSTM, Transformer, and MambaSSM — on the same forecasting task, with MambaSSM achieving the best performance among all tested models.

---

## 1. Methodology

To mitigate non-stationarity, all inputs are passed through a two-stage transformation before being fed to any model.

### 1.1 Logarithmic Trend Extraction

Given an observation <code>Y<sub>t</sub></code>, the underlying trend is approximated by fitting a logarithmic function of the form:

$$Y_t = a + b \cdot \log(X_t + c)$$

where <code>X<sub>t</sub></code> is the integer-encoded time index and <code>(a, b, c)</code> are estimated via non-linear least squares. This form is well-suited to data exhibiting exponential-like growth and remains numerically stable for small <code>X<sub>t</sub></code>.

### 1.2 Residual Computation

The residual component is obtained by subtracting the fitted trend:

$$R_t = Y_t - \hat{Y}_t$$

This isolates stationary short-term fluctuations from the global trend, allowing each model to focus on residual dynamics rather than long-run drift. An example of the transformation on a single feature is shown below.

<div align="center"><img src="./method.png" width="60%"></div>

### 1.3 Multi-Feature Residual Forecasting

Rather than predicting absolute values, the models are trained to forecast residual sequences. Given <code>k</code> input features, the learned mapping takes the form:

$$\sum_{i=1}^{k} \left( R_{t-n}^{(i)}, \ldots, R_{t}^{(i)} \right) \rightarrow \left( R_{t+1}, \ldots, R_{t+m} \right)$$

where <code>n</code> and <code>m</code> denote the lookback and forecasting horizons, respectively.

### 1.4 Inverse Transformation

Final predictions are reconstructed by adding the forecasted residuals back to the trend estimates:

$$\left( Y_{t+1}, \ldots, Y_{t+m} \right) = \left( \hat{Y}_{t+1}, \ldots, \hat{Y}_{t+m} \right) + \left( R_{t+1}, \ldots, R_{t+m} \right)$$

---

## 2. Models

| Model | Description |
|---|---|
| SegRNN [1] | A segment-wise RNN that partitions input sequences into fixed-length segments, improving computational efficiency while preserving recurrent state transitions. |
| Simple LSTM [2] | A vanilla LSTM trained directly on residual sequences over fixed-length windows. |
| Seq2Seq LSTM [3] | An encoder-decoder LSTM designed for multi-step forecasting, enabling variable-length input-output mappings. |
| Attention LSTM | An LSTM augmented with an additive attention mechanism, allowing selective weighting of past hidden states. |
| Transformer [4] | A fully attention-based architecture that removes recurrence, leveraging multi-head self-attention to capture both local and global temporal dependencies. |
| MambaSSM [5] | A structured state-space model (SSM) with selective state transitions, offering linear-time complexity and strong performance on long-range sequence modeling. |

---

## 3. Architecture Diagrams

The two primary architectures evaluated in this project are illustrated below.

**Transformer** — encoder-decoder architecture with multi-head self-attention and positional encoding ([4]):

<div align="center"><img src="./attention.png" width="60%"></div>

**Mamba Block** — selective SSM with dual-branch projection, depthwise convolution, and SiLU gating ([5]):

<div align="center"><img src="./mamba.png" width="60%"></div>

---

## 4. Results

MambaSSM achieved the best forecasting performance among all tested architectures, with RNN and LSTM baselines (including Seq2Seq and Attention variants) performing comparably to each other but below the Transformer and Mamba models. The log-detrending pipeline consistently improved stability across all architectures relative to training on raw values. See `result.png` for a full comparison.

<div align="center"><img src="./result.png" width="60%"></div>

---

## 5. References

[1] Lin, S., Lin, W., Wu, W., Zhao, F., Mo, R., & Zhang, H. (2023). SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting. *arXiv:2308.11200*. https://arxiv.org/abs/2308.11200

[2] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9*(8), 1735–1780. https://www.bioinf.jku.at/publications/older/2604.pdf

[3] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS 2014*. https://arxiv.org/abs/1409.3215

[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762

[5] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*. https://arxiv.org/abs/2312.00752

[6] Gu, A., Goel, K., & Ré, C. (2021). Efficiently Modeling Long Sequences with Structured State Spaces. *ICLR 2022*. https://arxiv.org/abs/2111.00396
