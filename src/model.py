import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.signal import hilbert
import numpy as np
import pandas as pd

class GLU(nn.Module):
    def __init__(self, input_size):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_size, input_size * 2)

    def forward(self, x):
        # x shape: (batch, seq_len, hidden)
        out = self.linear(x)
        return out[:, :, :x.shape[2]] * torch.sigmoid(out[:, :, x.shape[2]:])

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        # Manually apply causal padding for conv1
        out = F.pad(x, (self.padding, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Manually apply causal padding for conv2
        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MidnightModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_heads=8, dropout=0.3):
        super(MidnightModel, self).__init__()
        
        # 1. Input Projection & Positional Encoding
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # 2. Attention-Augmented TCN Blocks
        self.tcn1 = TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=1, dropout=dropout)
        self.tcn2 = TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=2, dropout=dropout)
        self.tcn3 = TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=4, dropout=dropout)
        
        # 3. Gated Adaptive Filter
        self.glu = GLU(hidden_size)
        
        # 4. Multi-Head Self-Attention (Quantum Focus)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
        
        # 5. Dedicated Multi-Task Learning Heads
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        
        # Back to (batch, seq_len, hidden)
        x = x.transpose(1, 2)
        x = self.glu(x)
        
        # Global Cross-Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.ln(x + attn_out)
        
        # Pooling: Focus on the latest information state
        last_hidden = x[:, -1, :]
        
        action_logits = self.classifier(last_hidden)
        pred_return = self.regressor(last_hidden)
        pred_vol = self.vol_head(last_hidden)
        
        return action_logits, pred_return, pred_vol

# Alias for backward compatibility if needed, though we are renaming everything
TradingLSTM = MidnightModel

def create_sequences(data, seq_length=60):
    """Create sequences for MTL training. Last 3 columns of data are targets."""
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-3])  # All features except last 3 targets
        y.append(data[i+seq_length, -3:])     # Targets: [action, return, vol]
    
    return np.array(X), np.array(y)

def prepare_features(df):
    """Prepare features from dataframe"""
    # Ensure columns are lowercase for consistency (future-proofing)
    df.columns = [str(c).lower() for c in df.columns]
    
    if len(df) < 30: # Minimum rows for basic indicators
        raise ValueError(f"DataFrame too small for indicators: {len(df)} rows")
        
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, UltimateOscillator, PercentagePriceOscillator, StochRSIIndicator, KAMAIndicator, TSIIndicator
    from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator, PSARIndicator, SMAIndicator, AroonIndicator, VortexIndicator, TRIXIndicator, MassIndex, IchimokuIndicator, DPOIndicator
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
    from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator, VolumePriceTrendIndicator, EaseOfMovementIndicator, ForceIndexIndicator
    
    # Price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / (df['close'].shift(1) + 1e-9))
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    # Heikin-Ashi
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = (df['open'].shift(1) + df['close'].shift(1)) / 2
    # Simplified HA for vectorized speed
    df['ha_close'] = ha_close
    df['ha_returns'] = df['ha_close'].pct_change()

    # DSP: Hilbert Transform (Market Cycles)
    def calculate_hilbert(series):
        try:
            v = series.ffill().bfill().values
            if len(v) < 10: return np.zeros(len(series)), np.zeros(len(series))
            analytic_signal = hilbert(v)
            amplitude = np.abs(analytic_signal)
            phase = np.angle(analytic_signal)
            return amplitude, phase
        except:
            return np.zeros(len(series)), np.zeros(len(series))
            
    df['hilbert_amp'], df['hilbert_phase'] = calculate_hilbert(df['close'])
    
    # DSP: Fisher Transform (Gaussian Normalization)
    def fisher_transform(series, window=10):
        low = series.rolling(window=window).min()
        high = series.rolling(window=window).max()
        # Scale to -1 to 1 range with smoothing
        val = 2 * ((series - low) / (high - low + 1e-9) - 0.5)
        val = val.rolling(window=3).mean().fillna(0)
        val = np.clip(val, -0.999, 0.999)
        return 0.5 * np.log((1 + val) / (1 - val + 1e-9))
        
    df['fisher'] = fisher_transform(df['close'])
    
    # Microstructure: Shannon Entropy (Predictability)
    def calculate_entropy(series, window=20):
        def entropy(x):
            if len(x) < 5: return 0
            x = (x - np.mean(x)) / (np.std(x) + 1e-9)
            counts, _ = np.histogram(x, bins=10, range=(-3, 3), density=True)
            counts = counts / (np.sum(counts) + 1e-9)
            counts = counts[counts > 1e-6]
            return -np.sum(counts * np.log(counts))
        return series.rolling(window=window).apply(entropy)
        
    df['entropy'] = calculate_entropy(df['returns'])
    
    # Microstructure: VPIN (Volume-Synchronized Probability of Informed Trading)
    def calculate_vpin(df, window=20):
        # Buy/Sell volume proxy based on price position in bar
        buy_vol = df['volume'] * (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
        sell_vol = df['volume'] - buy_vol
        oi = np.abs(buy_vol - sell_vol) # Order imbalance
        return oi.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
        
    df['vpin'] = calculate_vpin(df)

    # Chaos Theory: Hurst Exponent (Simplified R/S)
    def calculate_hurst(series, window=30):
        def hurst_rs(ts):
            if len(ts) < 10: return 0.5
            try:
                # Standardize
                ts_norm = (ts - np.mean(ts)) / (np.std(ts) + 1e-9)
                # Rescaled Range
                y = np.cumsum(ts_norm)
                R = np.max(y) - np.min(y)
                S = np.std(ts) + 1e-9
                # Hurst Exponent = log(R/S) / log(n)
                # For crypto, we expect 0.5 (random), >0.5 (trending), <0.5 (mean-reverting)
                h = np.log(R / S + 1e-9) / np.log(len(ts))
                return np.clip(h, 0, 1)
            except:
                return 0.5
        return series.rolling(window=window).apply(hurst_rs)

    df['hurst_20'] = calculate_hurst(df['close'], window=20)
    
    # Seasonality Features
    if df.index.dtype == 'datetime64[ns]' or isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        day = df.index.dayofweek
    else:
        # Try to convert index to datetime if it's not
        try:
            temp_idx = pd.to_datetime(df.index)
            hour = temp_idx.hour
            day = temp_idx.dayofweek
        except:
            hour = np.zeros(len(df))
            day = np.zeros(len(df))
            
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * day / 7)
    df['day_cos'] = np.cos(2 * np.pi * day / 7)
    
    # Momentum indicators
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['stoch'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
    df['stoch_rsi'] = StochRSIIndicator(close=df['close']).stochrsi()
    df['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()
    df['willr'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
    df['roc'] = ROCIndicator(close=df['close'], window=12).roc()
    df['ppo'] = PercentagePriceOscillator(close=df['close']).ppo()
    df['tsi'] = TSIIndicator(close=df['close']).tsi()
    df['kama'] = KAMAIndicator(close=df['close']).kama()
    df['uo'] = UltimateOscillator(high=df['high'], low=df['low'], close=df['close']).ultimate_oscillator()
    
    # Trend indicators
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
    df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
    df['psar'] = PSARIndicator(high=df['high'], low=df['low'], close=df['close']).psar()
    
    aroon = AroonIndicator(high=df['high'], low=df['low'], window=25)
    df['aroon_up'] = aroon.aroon_up()
    df['aroon_down'] = aroon.aroon_down()
    
    vortex = VortexIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['vortex_pos'] = vortex.vortex_indicator_pos()
    df['vortex_neg'] = vortex.vortex_indicator_neg()
    
    df['trix'] = TRIXIndicator(close=df['close'], window=15).trix()
    df['mass_index'] = MassIndex(high=df['high'], low=df['low']).mass_index()
    df['dpo'] = DPOIndicator(close=df['close']).dpo()
    
    ichi = IchimokuIndicator(high=df['high'], low=df['low'])
    df['ichi_a'] = ichi.ichimoku_a()
    df['ichi_b'] = ichi.ichimoku_b()
    
    # Moving Average Crosses
    df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['sma_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
    df['sma_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
    
    # Volatility indicators
    bb = BollingerBands(close=df['close'])
    df['bb_h'] = bb.bollinger_hband()
    df['bb_l'] = bb.bollinger_lband()
    df['bb_m'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_h'] - df['bb_l']) / df['bb_m']
    
    kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'])
    df['kc_width'] = (kc.keltner_channel_hband() - kc.keltner_channel_lband()) / kc.keltner_channel_mband()
    
    dc = DonchianChannel(high=df['high'], low=df['low'], close=df['close'])
    df['dc_width'] = (dc.donchian_channel_hband() - dc.donchian_channel_lband()) / df['close']
    
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['volatility_std'] = df['returns'].rolling(window=14).std()
    
    # Volume indicators
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['adi'] = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()
    df['cmf'] = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).chaikin_money_flow()
    df['vpt'] = VolumePriceTrendIndicator(close=df['close'], volume=df['volume']).volume_price_trend()
    df['eom'] = EaseOfMovementIndicator(high=df['high'], low=df['low'], volume=df['volume']).ease_of_movement()
    df['force_index'] = ForceIndexIndicator(close=df['close'], volume=df['volume']).force_index()
    
    # Price position
    df['price_vs_bb'] = (df['close'] - df['bb_l']) / (df['bb_h'] - df['bb_l'] + 1e-9)
    df['price_vs_sma50'] = df['close'] / (df['sma_50'] + 1e-9)
    
    # MTL Targets (Forward looking)
    df['next_return'] = df['returns'].shift(-1)
    df['next_vol'] = df['volatility_5'].shift(-1)
    
    # Cleanup: Replace Infs with NaNs and forward fill to ensure no gaps for the neural network
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def get_feature_cols():
    """Centralized list of features for training and prediction"""
    return [
        'returns', 'log_returns', 'ha_returns', 'hurst_20',
        'hilbert_amp', 'hilbert_phase', 'fisher', 'entropy', 'vpin',
        'volatility_5', 'volatility_20',
        'rsi', 'stoch', 'stoch_rsi', 'mfi', 'willr', 'roc', 'ppo', 'tsi', 'kama', 'uo',
        'macd', 'macd_diff', 'ema_12', 'ema_26', 'adx', 'cci', 'psar',
        'aroon_up', 'aroon_down', 'vortex_pos', 'vortex_neg', 'trix', 'mass_index', 'dpo',
        'ichi_a', 'ichi_b', 'sma_cross', 'kc_width', 'dc_width',
        'bb_width', 'atr', 'volatility_std', 'obv', 'adi', 'cmf', 
        'vpt', 'eom', 'force_index', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'price_vs_bb', 'price_vs_sma50'
    ]

def generate_labels(df, future_periods=5, threshold=0.005):
    """Generate trading labels: 0=SELL, 1=HOLD, 2=BUY"""
    future_returns = df['close'].pct_change(future_periods).shift(-future_periods)
    
    labels = np.ones(len(df)) # Default to HOLD (1)
    labels[future_returns > threshold] = 2  # BUY
    labels[future_returns < -threshold] = 0  # SELL
    
    return labels
