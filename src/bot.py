import sys
import os
# Add the project root to sys.path to allow running scripts directly from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import torch
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import warnings

# Suppress library warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.model import MidnightModel, prepare_features, get_feature_cols
from src.memory_db import TradingMemoryDB

load_dotenv()

class MidnightLiveBot:
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.symbol = os.getenv('SYMBOL', 'BTC/USD')
        self.qty_limit = float(os.getenv('QUANTITY_LIMIT', 0.01))
        
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client = CryptoHistoricalDataClient()
        self.memory = TradingMemoryDB()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.scaler = None
        self.load_model()
        
        print(f"MIDNIGHT.AI LIVE BOT INITIALIZED: {self.symbol}")
        print(f"DEVICE: {self.device}")

    def load_model(self):
        model_path = 'models/best_model.pt'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            self.scaler = checkpoint['scaler']
            
            # Reconstruct model
            input_size = len(get_feature_cols())
            self.model = MidnightModel(input_size=input_size).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Midnight.AI Quant model loaded successfully.")
        else:
            print("No trained model found. Bot will wait for train.py to finish.")

    def get_realtime_data(self):
        now = datetime.now()
        start = now - timedelta(days=5)
        
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[self.symbol],
            timeframe=TimeFrame.Hour,
            start=start
        )
        
        bars = self.data_client.get_crypto_bars(request_params)
        df = bars.df
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(self.symbol, level=0)
            
        return df

    def predict_action(self, df):
        if self.model is None:
            return 1, 0, 0, 0 # action, prob, pred_return, pred_vol
            
        df = prepare_features(df)
        df = df.dropna()
        
        if len(df) < 60:
            return 1, 0, 0, 0
            
        # Select same features as training
        feature_cols = get_feature_cols()
        
        recent_data = df[feature_cols].tail(60).values
        recent_data_scaled = self.scaler.transform(recent_data)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(recent_data_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, pred_return, pred_vol = self.model(input_tensor)
            probs = torch.softmax(action_logits, dim=1)
            prob, predicted = torch.max(probs, 1)
            
        return predicted.item(), prob.item(), pred_return.item(), pred_vol.item()

    def execute_trade(self, side, current_price, qty=None):
        try:
            trade_qty = qty if qty else self.qty_limit
            order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=trade_qty,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            order = self.trading_client.submit_order(order_data)
            print(f"ORDER EXECUTED: {side} {trade_qty} {self.symbol} at approx ${current_price}")
            return order
        except Exception as e:
            print(f"TRADE FAILED: {str(e)}")
            return None

    def calculate_kelly_qty(self, prob, pred_return, current_price):
        """Institutional sizing: Half-Kelly to avoid over-leveraging"""
        if prob < 0.6 or pred_return <= 0:
            return self.qty_limit * 0.5
        
        # win_prob = prob, loss_prob = 1 - prob
        # win_loss_ratio assumed 2:1 for crypto
        ratio = 2.0
        kelly_f = (prob * ratio - (1 - prob)) / ratio
        kelly_f = max(0, min(kelly_f, 0.2)) # Cap at 20% of limit
        
        qty = self.qty_limit * (kelly_f * 5) # Scale to qty_limit
        return round(qty, 5)

    def run(self):
        active_pos = False
        entry_price = 0
        
        while True:
            try:
                # Refresh model if it wasn't loaded
                if self.model is None:
                    self.load_model()
                
                df = self.get_realtime_data()
                current_price = df['close'].iloc[-1]
                
                # Check for neural bypass (risky conditions)
                conditions = {
                    'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
                    'macd_diff': df['macd_diff'].iloc[-1] if 'macd_diff' in df.columns else 0,
                    'price_vs_bb_l': current_price / df['bb_l'].iloc[-1] if 'bb_l' in df.columns else 1
                }
                
                action, prob, p_ret, p_vol = self.predict_action(df)
                
                # Regime Filter: Avoid high volatility noise
                if p_vol > 0.05: # Arbitrary threshold for high-vol
                    print(f"REGIME SHIFT: High Volatility Predicted ({p_vol:.4f}). Neutralizing.")
                    action = 1
                
                if action == 2 and not active_pos: # BUY
                    if self.memory.is_risky_condition(conditions):
                        print("NEURAL BYPASS: Condition too similar to past mistakes. Skipping trade.")
                    else:
                        qty = self.calculate_kelly_qty(prob, p_ret, current_price)
                        print(f"NEURAL SIGNAL: BUY (Conf: {prob:.2f}, Pred R: {p_ret:.4f}, Qty: {qty})")
                        self.execute_trade(OrderSide.BUY, current_price, qty=qty)
                        active_pos = True
                        entry_price = current_price
                        
                elif action == 0 and active_pos: # SELL
                    print(f"NEURAL SIGNAL: SELL (Conf: {prob:.2f})")
                    self.execute_trade(OrderSide.SELL, current_price)
                    
                    pnl = (current_price - entry_price) / entry_price * 100
                    self.memory.record_trade(entry_price, current_price, pnl, conditions)
                    active_pos = False
                    
                    print(f"Trade Results: PnL: {pnl:.4f}%")
                    print(self.memory.get_threatening_message())
                
                else:
                    print(f"Analyzing... Price: ${current_price:.2f} | Action: {'HOLD' if action==1 else 'NO_POS'} | Conf: {prob:.2f}")

                time.sleep(60) # Watch every minute
                
            except Exception as e:
                print(f"Loop Error: {str(e)}")
                time.sleep(10)

if __name__ == "__main__":
    bot = MidnightLiveBot()
    # bot.run() # Uncomment to run
