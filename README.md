[![python - v3.12.8](https://img.shields.io/static/v1?label=python&message=v3.12.8&color=blue&logo=python&logoColor=white)](https://)
[![cuda - v12.1](https://img.shields.io/static/v1?label=cuda&message=v12.1&color=green&logo=nvidia&logoColor=white)](https://)
[![torch - v2.5.1](https://img.shields.io/static/v1?label=torch&message=v2.5.1&color=orange&logo=pytorch&logoColor=white)](https://)

### Repository Summary ###
- Task: Time Series Forecasting
- Model Type: RNN, LSTM, Transformer, MAMBA

## **Sequence-to-Sequence Time Series Forecasting with Multi-Feature Inputs**  ##
This project investigates **sequence-to-sequence time series forecasting** using multi-feature inputs, leveraging various deep learning architectures to model temporal dependencies. A key focus is on **detrending via logarithmic transformations** and residual forecasting, allowing for more stable and interpretable predictions.  

## Methodology ##
To improve forecasting stability and mitigate non-stationarity, we apply the following transformations:  

### 1. Logarithmic Trend Extraction ###
Given an observation <code>Y<sub>t</sub></code>, we approximate its underlying trend using a logarithmic function of the form: 

**Y<sub>t</sub> = a + b * log(X<sub>t</sub> + c)**

where <code>X<sub>t</sub></code> represents the time index transformed into an integer format, and <code>(a, b, c)</code> are estimated via non-linear least squares fitting. This transformation is particularly effective in capturing **exponential-like growth** while ensuring numerical stability in cases of small <code>X<sub>t</sub></code>.  

### 2. Residual Computation ###
After estimating the log trend <code>Ŷ<sub>t</sub></code>, the residual component is obtained as:

**R<sub>t</sub> = Y<sub>t</sub> - Ŷ<sub>t</sub>**

This isolates short-term fluctuations, allowing the forecasting model to learn purely stationary residual dynamics without being influenced by global trend variations. Below is an example of the computation on a single feature.

![method](./method.png)

### 3. Multi-Feature Residual Forecasting ###
Instead of forecasting the absolute values of <code>Y<sub>t</sub></code>, we predict the residual components <code>R<sub>t</sub></code> using multiple transformed features. The model learns an implicit mapping:

**∑<sub>i=1</sub><sup>k</sup> (R<sub>t-n</sub><sup>(i)</sup>, ..., R<sub>t</sub><sup>(i)</sup>)  →  (R<sub>t+1</sub>, ..., R<sub>t+m</sub>)**
 
where <code>n</code> and <code>m</code> denote the historical and forecasting horizons, respectively.

### 4. Inverse Transformation for Final Prediction ###
After forecasting <code>(R<sub>t+1</sub>, ..., R<sub>t+m</sub>)</code>, we reconstruct the final time series prediction as:

**(Y<sub>t+1</sub>, ..., Y<sub>t+m</sub>) = (Ŷ<sub>t+1</sub>, ..., Ŷ<sub>t+m</sub>) + (R<sub>t+1</sub>, ..., R<sub>t+m</sub>)**

ensuring consistency with the original data distribution.

## Modeling Approaches ##
We experimented with several sequence modeling architectures, each incorporating different mechanisms for temporal feature extraction:

| Model                          | Description                                                   |
|-----------------------------------------|--------------------------------------------------------|
| **SegRNN** | A segment-wise **Recurrent Neural Network (RNN)** that dynamically partitions sequences, improving computational efficiency while preserving recurrent state transitions. |
| **Simple LSTM** | A **vanilla LSTM** trained on sequential residuals to capture temporal dependencies over fixed-length windows. |
| **Seq2Seq LSTM** | An **encoder-decoder LSTM** architecture designed for multi-step forecasting, enabling the model to learn dynamic representations across different sequence lengths. |
| **Attention LSTM** | An **LSTM augmented with an attention mechanism**, allowing selective weighting of past time steps to enhance long-term dependency retention. |
| **Transformer** | A self-attention-based model that removes recurrence, leveraging **multi-head attention** to capture both local and global dependencies. |
| **MambaSSM** | A **state-space model (SSM)** optimized for long-range forecasting, leveraging structured state transitions and memory-efficient recurrence to outperform traditional RNN-based architectures. |

## Results & Findings ##
- **MambaSSM demonstrated superior performance**, effectively capturing long-range dependencies while maintaining computational efficiency.
- **Log transformation + residual forecasting significantly improved stability**, leading to better generalization across different time horizons.
- **MambaSSM outperformed traditional models** such as Simple LSTM and Seq2Seq LSTM, and even the attention-based models, by a significant margin.

![result](./result.png)

## References ##
Fuck you I wrote them all.
Okay maybe I will add some references in the future.
