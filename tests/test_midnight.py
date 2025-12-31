import torch
import pytest
import numpy as np
import pandas as pd
from src.model import MidnightModel, prepare_features, get_feature_cols
from src.train import FocalLoss, SharpeLoss
from src.memory_db import TradingMemoryDB

def test_model_forward_pass():
    input_size = len(get_feature_cols())
    model = MidnightModel(input_size=input_size)
    # Batch size 2, Sequence length 60, Features input_size
    dummy_input = torch.randn(2, 60, input_size)
    logits, returns, vol = model(dummy_input)
    
    assert logits.shape == (2, 3)
    assert returns.shape == (2, 1)
    assert vol.shape == (2, 1)

def test_focal_loss():
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    logits = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = criterion(logits, targets)
    assert loss > 0
    loss.backward()
    assert logits.grad is not None

def test_sharpe_loss():
    criterion = SharpeLoss()
    # High returns, low volatility should give lower (better) loss
    good_returns = torch.tensor([0.02, 0.021, 0.019, 0.02], requires_grad=True)
    loss = criterion(good_returns)
    assert -1 <= loss <= 1

def test_feature_preparation():
    # Create dummy dataframe
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 101,
        'low': np.random.randn(100) + 99,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randn(100) * 1000 + 5000
    }, index=dates)
    
    df_feat = prepare_features(df)
    feature_cols = get_feature_cols()
    
    for col in feature_cols:
        assert col in df_feat.columns
    assert not df_feat[feature_cols].isnull().all().any()

def test_db_initialization(tmp_path):
    db_file = tmp_path / "test_trading.db"
    db = TradingMemoryDB(db_path=str(db_file))
    assert db_file.exists()
