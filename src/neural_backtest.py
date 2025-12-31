import sys
import os
# Add the project root to sys.path to allow running scripts directly from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings

# Suppress library warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from src.model import MidnightModel, prepare_features, get_feature_cols
from src.memory_db import TradingMemoryDB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class MidnightBacktester:
    def __init__(self, symbol='BTC-USD', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.symbol = symbol
        self.device = device
        self.memory = TradingMemoryDB()
        self.model = None
        self.scaler = None
        self.load_model()
        
    def load_model(self):
        model_path = 'models/best_model.pt'
        if os.path.exists(model_path):
            # Using weights_only=False because we are loading a scaler (non-tensor object)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model_state_dict = checkpoint['model_state_dict']
            self.scaler = checkpoint['scaler']
            
            input_size = len(get_feature_cols())
                
            self.model = MidnightModel(input_size=input_size).to(self.device)
            self.model.load_state_dict(self.model_state_dict)
            self.model.eval()
            print(f"Loaded Midnight.AI model from {model_path} (Input Size: {input_size})")
        else:
            raise FileNotFoundError("Model not found. Run train.py first.")


    def run_test(self, days=30):
        # Create unique folder for this backtest
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'backtests/run_{run_id}'
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting Midnight.AI neural backtest for {self.symbol} over the last {days} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10) # Get extra for indicators
        
        df = yf.download(self.symbol, start=start_date.strftime('%Y-%m-%d'), 
                        end=end_date.strftime('%Y-%m-%d'), interval='1h', progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        
        print(f"Downloaded {len(df)} bars.")
        
        # Prepare features
        df_feat = prepare_features(df.copy())
        df_feat = df_feat.dropna()
        
        feature_cols = get_feature_cols()
        
        # We need at least 60 samples for the first prediction
        if len(df_feat) < 61:
            print("Not enough data for backtest.")
            return

        active_pos = False
        entry_price = 0
        trades = []
        total_pnl = 0
        
        action_counts = {0: 0, 1: 0, 2: 0}
        
        action_probs = {0: [], 1: [], 2: []}
        
        # Sequence length used in training
        seq_length = 60
        
        # Shift start to after indicators and first sequence
        for i in range(seq_length, len(df_feat)):
            current_row = df_feat.iloc[i]
            current_price = current_row['close']
            
            # Neural prediction
            # Get the previous 60 bars
            recent_data = df_feat[feature_cols].iloc[i-seq_length:i].values
            recent_data_scaled = self.scaler.transform(recent_data)
            
            input_tensor = torch.FloatTensor(recent_data_scaled).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_logits, p_ret, p_vol = self.model(input_tensor)
                probs = torch.softmax(action_logits, dim=1)
                prob, predicted = torch.max(probs, 1)
                action = predicted.item()
                action_counts[action] += 1
                action_probs[action].append(prob.item())
            
            # Check for bypass (risky conditions)
            conditions = {
                'rsi': current_row['rsi'],
                'macd_diff': current_row['macd_diff'],
                'price_vs_bb_l': current_price / current_row['bb_l']
            }
            
            # Signal handling
            if action == 2 and not active_pos: # BUY
                if self.memory.is_risky_condition(conditions):
                    # Neural bypass logged
                    pass
                else:
                    active_pos = True
                    entry_price = current_price
                    entry_time = df_feat.index[i]
                    
            elif action == 0 and active_pos: # SELL
                exit_price = current_price
                exit_time = df_feat.index[i]
                pnl = (exit_price - entry_price) / entry_price * 100
                total_pnl += pnl
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl
                })
                
                # Record to DB for the dashboard/memory logic
                self.memory.record_trade(entry_price, exit_price, pnl, conditions)
                active_pos = False

        # Results
        print("\n" + "="*40)
        print(f"BACKTEST COMPLETE: {self.symbol}")
        print("="*40)
        print(f"Total Trades: {len(trades)}")
        print(f"Predictions:  BUY: {action_counts[2]}, SELL: {action_counts[0]}, HOLD: {action_counts[1]}")
        avg_hold_conf = np.mean(action_probs[1]) if action_probs[1] else 0
        print(f"Avg Hold Confidence: {avg_hold_conf:.4f}")
        
        if len(trades) > 0:
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            
            win_rate = (len(wins) / len(trades)) * 100
            total_pnl = sum(pnls)
            avg_pnl = total_pnl / len(trades)
            
            # Profit Factor
            gross_profits = sum(wins)
            gross_losses = abs(sum(losses)) if losses else 1e-9
            profit_factor = gross_profits / gross_losses
            
            # Realized Sharpe (Basic estimate based on trade-by-trade returns)
            std_pnl = np.std(pnls) + 1e-9
            realized_sharpe = (avg_pnl / std_pnl) * np.sqrt(len(trades)) if len(trades) > 1 else 0
            
            # Max Drawdown
            cumulative_pnl = np.cumsum(pnls)
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = np.max(drawdown)

            print(f"Win Rate:      {win_rate:.2f}% ({len(wins)}W / {len(losses)}L)")
            print(f"Total PnL:     {total_pnl:.2f}%")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Max Drawdown:  {max_drawdown:.2f}%")
            print(f"Realized Sharpe: {realized_sharpe:.4f}")
            print(f"Worst Trade:   {min(pnls):.2f}%")
            
            # --- COMPREHENSIVE VISUALIZATION SUITE ---
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(24, 18))
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # 1. Main Price + Trade Markers
            ax1 = fig.add_subplot(gs[0:2, 0:3])
            ax1.plot(df_feat.index[seq_length:], df_feat['close'].iloc[seq_length:], label='Price', color='#00d2ff', alpha=0.4)
            for t in trades:
                ax1.scatter(t['entry_time'], t['entry_price'], marker='^', color='#00ff41', s=60, alpha=0.9)
                ax1.scatter(t['exit_time'], t['exit_price'], marker='v', color='#ff3131', s=60, alpha=0.9)
            ax1.set_title(f"Midnight.AI Market Execution - {self.symbol}", fontsize=14)
            ax1.legend()
            ax1.grid(alpha=0.1)

            # 2. Equity Curve
            ax2 = fig.add_subplot(gs[2, 0:3], sharex=ax1)
            cum_pnl = np.cumsum([0] + pnls)
            trade_times = [trades[0]['entry_time']] + [t['exit_time'] for t in trades]
            ax2.plot(trade_times, cum_pnl, color='#ffcc00', linewidth=2.5, label='Strategy (Neural)')
            
            # Baseline: Buy & Hold
            baseline_returns = (df_feat['close'].iloc[seq_length:] / df_feat['close'].iloc[seq_length] - 1) * 100
            ax2.plot(df_feat.index[seq_length:], baseline_returns, color='white', alpha=0.3, label='Buy & Hold (Baseline)')
            
            ax2.set_title("Cumulative Returns vs Baseline (%)", fontsize=12)
            ax2.fill_between(trade_times, cum_pnl, color='#ffcc00', alpha=0.1)
            ax2.legend()
            ax2.grid(alpha=0.1)

            # 3. Drawdown Curve
            ax3 = fig.add_subplot(gs[3, 0:3], sharex=ax1)
            peak = np.maximum.accumulate(cum_pnl)
            dd = peak - cum_pnl
            ax3.fill_between(trade_times, -dd, 0, color='#ff3131', alpha=0.2, label='Drawdown')
            ax3.plot(trade_times, -dd, color='#ff3131', linewidth=1)
            ax3.set_title("Strategy Drawdown (%)", fontsize=12)
            ax3.grid(alpha=0.1)

            # 4. Returns Distribution
            ax4 = fig.add_subplot(gs[0, 3])
            sns.histplot(pnls, kde=True, ax=ax4, color='#00d2ff')
            ax4.set_title("Trade Return Dist.")

            # 5. Prediction Balance (Pie)
            ax5 = fig.add_subplot(gs[1, 3])
            labels = ['Hold', 'Sell', 'Buy']
            sizes = [action_counts[1], action_counts[0], action_counts[2]]
            ax5.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#444444', '#ff3131', '#00ff41'])
            ax5.set_title("Action Mix")

            # 6. Win Rate (Donut)
            ax6 = fig.add_subplot(gs[2, 3])
            ax6.pie([len(wins), len(losses)], labels=['Wins', 'Losses'], colors=['#00ff41', '#ff3131'], wedgeprops=dict(width=0.3))
            ax6.set_title(f"Win Rate: {win_rate:.1f}%")

            # 7. Confidence Distribution
            ax7 = fig.add_subplot(gs[3, 3])
            all_probs = []
            for k in action_probs: all_probs.extend(action_probs[k])
            sns.kdeplot(all_probs, ax=ax7, shade=True, color='gold')
            ax7.set_title("Model Confidence Dist.")

            plt.tight_layout()
            dashboard_path = f"{save_dir}/dashboard.png"
            plt.savefig(dashboard_path, dpi=150)
            
            # Additional Individual Charts for deeper analysis
            # 8. PnL by Time of Day
            plt.figure(figsize=(10, 6))
            df_trades = pd.DataFrame(trades)
            df_trades['hour'] = df_trades['entry_time'].apply(lambda x: x.hour)
            sns.barplot(x='hour', y='pnl', data=df_trades, ci=None, palette='viridis')
            plt.title("Performance by Hour of Day")
            plt.savefig(f"{save_dir}/pnl_by_hour.png")
            
            # 9. Rolling Sharpe (Approx)
            plt.figure(figsize=(10, 6))
            rolling_sharpe = [np.mean(pnls[:i+1]) / (np.std(pnls[:i+1]) + 1e-9) * np.sqrt(i+1) for i in range(len(pnls))]
            plt.plot(rolling_sharpe, color='orange')
            plt.title("Rolling Realized Sharpe")
            plt.savefig(f"{save_dir}/rolling_sharpe.png")

            # 10. Trade Scatter (Return vs Hold Time)
            plt.figure(figsize=(10, 6))
            hold_times = [(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in trades]
            plt.scatter(hold_times, pnls, c=pnls, cmap='RdYlGn')
            plt.axhline(0, color='white', alpha=0.3)
            plt.title("Return vs Hold Time (Hours)")
            plt.savefig(f"{save_dir}/return_vs_duration.png")

            print(f"\n[CHARTS] 10+ Backtest charts saved to: {save_dir}/")
        else:
            print("No trades executed.")
        print("="*40)

if __name__ == "__main__":
    tester = MidnightBacktester()
    tester.run_test(days=60) # Check last 2 months
