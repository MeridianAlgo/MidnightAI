# 🌑 Midnight.AI: Quantum Multi-Objective Trading Engine

Midnight.AI is a high-performance, institutional-grade algorithmic trading platform. It utilizes advanced **Digital Signal Processing (DSP)**, **Temporal Convolutional Networks (TCN)**, and **Multi-Head Self-Attention** to identify and exploit alpha in high-volatility cryptocurrency markets.

> **Status**: [PRE-TRAINED] - This repository contains state-of-the-art weights in `models/best_model.pt` calibrated over 7-20 evolution sessions.

---

## 🏗️ Core Architecture (Quantum Evolution)

### 1. The Neural Backbone: Attention-Augmented ResTCN
- **MidnightModel**: A specialized hybrid architecture combining causal dilated convolutions with global self-attention.
- **Positional Encoding**: Explicitly encodes temporal distances to capture cyclical market patterns.
- **Residual TCN Blocks**: Captures local micro-trends using causal dilated convolutions, ensuring zero future leakage.
- **Multi-Head Self-Attention (12-Heads)**: A "Quantum Focus" layer that identifies non-local correlations across the 60-bar lookback window.
- **Gated Linear Units (GLU)**: Acting as informational filters to suppress noise during low-entropy market regimes.

### 2. Digital Signal Processing (DSP) Layer
Midnight.AI transforms raw price action into a high-fidelity, Gaussian-normalized feature space before it ever touches the neural network:
- **Hilbert Transform**: Extracts **Instantaneous Phase** and **Amplitude** to pin-point cycle positions.
- **Fisher Transform**: Forces price distributions into a Gaussian Probability Density Function (PDF) for better linear separability.
- **Shannon Entropy**: Measures the "Kolmogorov Complexity" of recent price action to detect regime shifts.
- **VPIN**: Volume-Synchronized Probability of Informed Trading (Real-time Institutional Flow detection).
- **Hurst Exponent**: Quantifies the "memory" of the market (Trending vs. Mean-Reverting).

### 3. Multi-Objective Optimization
- **Focal Loss ($\gamma=5, \alpha=2$)**: Prioritizes "Hard Examples," forcing the model to learn rare Buy/Sell signals over the dominant "Hold" class.
- **Tanh-Bounded Sharpe Loss**: Directly optimizes the **Differential Sharpe Ratio** (Return/Risk) while maintaining training stability.
- **MTL Heads**: Separate heads for **Action Classification**, **Return Regression**, and **Volatility Prediction**.

---

## 🛠️ Operational Guide

### 1. Deployment Requirements
```bash
pip install torch scipy numba pandas ta yfinance sklearn matplotlib seaborn tqdm
```

### 2. The Real Training Phase (20 Sessions)
The system is currently configured for a robust **20-Session Training Evolution**. Each session performs a full training cycle with **Stochastic Weight Averaging (SWA)** to find the most stable local minima in the loss landscape.
```bash
python src/train.py
```
*   **Progress Tracking**: High-fidelity tqdm bars display **Validation Accuracy**, **Best Loss**, and **Gradient Norm** per session.
*   **Ultra-Logs**: Forensic logging records every loss component (Focal, Sharpe, Reg) to `logs/run_*.log`.

### 3. Verification & Testing
Before deployment, run the integrated test suite to verify model integrity and signal processing:
```bash
pytest tests/
```

### 4. Neural Backtesting (Visual Intelligence)
Validate the engine using the **Visual Intelligence Dashboard**:
```bash
python src/neural_backtest.py
```
Generates 10+ analytical charts in `backtests/run_*/`, including:
- **Cumulative PnL vs. Baseline (Buy & Hold)**.
- **Underwater Drawdown Analysis**.
- **Market Execution Heatmaps**.
- **PnL by Hour/Volatility Distribution**.

### 4. Live Bot Execution
```bash
python src/bot.py
```
*Built-in: Half-Kelly position sizing, Dynamic ATR Stop-Loss, and Real-time DSP filtering.*

---

## 📊 Evolutionary Monitoring
Midnight.AI logs all metadata to a local `trading_memory.db` (SQLite). This allows for deep visualization of the model's lineage and performance metrics over time.

---
**Institutional Disclaimer**: *Midnight.AI is an advanced research tool. Algorithmic trading involves significant risk. Always validate on paper environments before deploying capital.*
