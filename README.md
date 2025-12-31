# Midnight.AI: AI Trading Engine for Bitcoin

Midnight.AI is a powerful automated trading system designed to find profitable opportunities in Bitcoin markets. It uses advanced math and artificial intelligence to "clean" market data and recognize patterns that humans might miss.

> **Current Status**: [READY] - Includes a pre-trained model in `models/best_model.pt` that has been refined over multiple training sessions.

---

## Important Disclaimers

**Read this carefully before using or deploying Midnight.AI:**

1. **Trading involves significant risk**: The cryptocurrency market is highly volatile. You can lose your entire investment in a very short period. Never trade with funds you cannot afford to lose.
2. **Not Financial Advice**: The authors and contributors of Midnight.AI are not financial advisors. This software is provided for educational and research purposes only. Any trading decisions you make are your own responsibility.
3. **Software is "As-Is"**: This is experimental software. There are no warranties or guarantees regarding its performance, stability, or profitability. Bugs, logic errors, or API connectivity issues can lead to financial loss.
4. **Past Performance is No Guarantee**: Backtests are based on historical data. Markets change, and what worked in the past may not work in the future.
5. **No Affiliation**: This project is not affiliated with any exchange or financial institution.
6. **Execution Risk**: Slippage, exchange downtime, and network latency can all impact the performance of automated trading bots.

---

## Key Features

### Advanced Signal Processing
Midnight.AI doesn't just look at price numbers. It uses "Market Math" to see through the noise:
- **Cycle Detection**: Identifies if Bitcoin is in a wave-like pattern or a straight trend.
- **Sentiment Analysis**: Gauges market pressure by looking at buying vs. selling volume (Institutional Flow).
- **Chaos Reduction**: Processes raw data into a clean format that makes it easier for the AI to "think" clearly.

### Intelligent Pattern Recognition
- **Memory Layers**: The bot looks back at the last hour of trading to understand the current context.
- **Attention System**: Like a human trader focusing on a specific candle, the AI focuses on exactly which past events matter most for the next move.
- **Auto-Regularization**: Built-in logic to prevent the bot from "overfitting" or memorizing the past instead of learning general patterns.

### Professional Risk Management
- **Smart Sizing**: Uses advanced math (Half-Kelly) to decide exactly how much to bet on a trade based on the bot's confidence.
- **Safety Stops**: Automatically places stop-losses to protect your balance if a trade goes the wrong way.
- **Visual Intelligence**: Generates over 10 different types of charts so you can see exactly why the bot made specific decisions.

---

## Getting Started

### 1. Setup
Ensure you have Python installed, then run:
```bash
pip install torch scipy numba pandas ta yfinance sklearn matplotlib seaborn tqdm
```

### 2. Training the AI
The system is pre-configured for "Evolutionary Training." You can run it yourself to further refine the model:
```bash
python src/train.py
```
*Detailed records and forensic logs are saved in the `logs/` directory.*

### 3. Verification
Run the test suite to ensure everything is installed and working correctly:
```bash
pytest tests/
```

### 4. Backtesting
Run a simulation to see how the bot would have performed on historical data:
```bash
python src/neural_backtest.py
```
Check the `backtests/` folder for:
- **PnL Charts**: Your profit/loss compared to just holding Bitcoin.
- **Drawdown Analysis**: A look at the "worst-case scenarios" during the test.
- **Market Heatmaps**: Visual markers of every buy and sell decision.

### 5. Deployment
When you are ready to run the live engine:
```bash
python src/bot.py
```
*Always start with "Paper Trading" (simulated money) to verify your setup.*

---

## Monitoring and Memory
The engine maintains a local database (`trading_memory.db`) using SQLite. This database stores every trade, every prediction, and the bot's evolving logic, allowing for deep performance analysis over time.

---

**Made with love by MeridianAlgo**
