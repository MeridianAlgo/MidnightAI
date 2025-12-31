import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.swa_utils as swa_utils
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import time
from datetime import datetime, timedelta
import json
import warnings
import signal
import sys

# Suppress library warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class FocalLoss(nn.Module):
    def __init__(self, alpha=2.0, gamma=5.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class SharpeLoss(nn.Module):
    def __init__(self, risk_free_rate=0):
        super(SharpeLoss, self).__init__()
        self.rf = risk_free_rate

    def forward(self, pred_returns):
        # Rewards high mean return and low volatility
        # We negate it and use tanh to bound the objective
        mean_ret = torch.mean(pred_returns)
        std_ret = torch.std(pred_returns) + 1e-4
        sharpe = (mean_ret - self.rf) / std_ret
        return -torch.tanh(sharpe)

from src.model import MidnightModel, create_sequences, prepare_features, generate_labels, get_feature_cols
from src.memory_db import TradingMemoryDB

class MidnightTrainer:
    def __init__(self, symbol='BTC-USD', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.symbol = symbol
        self.device = device
        self.memory = TradingMemoryDB()
        self.model = None
        self.scaler = StandardScaler()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience_limit = 15 # STOP after 15 epochs of no improvement
        
        # Unique ID for this specific execution (run)
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Timeframes for single-horizon training (per user request)
        self.timeframes = ['1h']
        
        # Try to load existing model immediately
        self.load_existing_model()
        
        print(f"Midnight.AI Trainer initialized on {self.device}")
        print(f"Logging to: logs/run_{self.run_id}.log")
    
    def load_existing_model(self, current_input_size=None):
        """Try to load the best model from previous runs, ensuring input size matches"""
        if current_input_size is None:
            current_input_size = len(get_feature_cols())
            
        model_path = 'models/best_model.pt'
        emergency_path = 'models/emergency_checkpoint.pt'
        
        load_path = model_path if os.path.exists(model_path) else (emergency_path if os.path.exists(emergency_path) else None)
        
        if load_path:
            try:
                checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
                state_dict = checkpoint['model_state_dict']
                
                # Input size check (God-Tier uses input_proj so we check the first layer's weight)
                loaded_input_size = state_dict['input_proj.weight'].shape[1]
                
                if loaded_input_size == current_input_size:
                    self.scaler = checkpoint['scaler']
                    self.model = MidnightModel(input_size=current_input_size).to(self.device)
                    self.model.load_state_dict(state_dict)
                    print(f"Resuming Evolution: {load_path} (Signals: {loaded_input_size})")
                else:
                    print(f"Signal mismatch (Loaded: {loaded_input_size}, New: {current_input_size}). Starting fresh.")
            except Exception as e:
                print(f"Resume failed: {e}")
    
    def download_multi_timeframe_data(self, start_date, end_date):
        """Download data across multiple timeframes"""
        all_data = {}
        
        for tf in self.timeframes:
            try:
                print(f"Downloading {self.symbol} {tf} data from {start_date} to {end_date}...")
                df = yf.download(self.symbol, start=start_date, end=end_date, interval=tf, progress=False, auto_adjust=True)
                
                if df.empty or len(df) < 100:
                    print(f"Insufficient data for {tf}")
                    continue
                    
                # Fix for multi-index if it still happens
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                df.columns = [str(c).lower() for c in df.columns]
                all_data[tf] = df
            except Exception as e:
                print(f"❌ Error downloading {tf}: {e}")
        
        return all_data
    
    def prepare_dataset(self, df, timeframe='1h', seq_length=60):
        """Prepare dataset with features and labels"""
        if df is None or len(df) < seq_length + 50:
            return None, None, None
            
        try:
            df = prepare_features(df.copy())
            df['label'] = generate_labels(df)
            # Replace infinity with NaN and then drop
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
        except Exception as e:
            print(f"Prep failed for {timeframe}: {e}")
            return None, None, None
            
        if len(df) < seq_length + 10:
            return None, None, None
        
        # Select features
        feature_cols = get_feature_cols() + ['label', 'next_return', 'next_vol']
        
        data = df[feature_cols].values
        
        # Normalize features (all except last 3 targets)
        data[:, :-3] = self.scaler.fit_transform(data[:, :-3])
        
        # Create sequences
        X, y = create_sequences(data, seq_length)
        
        return X, y, df
    
    def train_epoch(self, model, train_loader, criterion_cls, criterion_reg, criterion_sharpe, optimizer):
        """Train for one epoch with Multi-Task God-Tier Learning"""
        model.train()
        metrics = {'loss': 0, 'loss_focal': 0, 'loss_ret': 0, 'loss_vol': 0, 'loss_sharpe': 0, 'correct': 0, 'total': 0}
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_action = y_batch[:, 0].to(self.device).long()
            y_return = y_batch[:, 1].to(self.device).float().unsqueeze(1)
            y_vol = y_batch[:, 2].to(self.device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            action_logits, pred_return, pred_vol = model(X_batch)
            
            l_focal = criterion_cls(action_logits, y_action)
            l_ret = criterion_reg(pred_return, y_return)
            l_vol = criterion_reg(pred_vol, y_vol)
            l_sharpe = criterion_sharpe(pred_return)
            
            # Balanced Loss: Focal (Priority) + MSE(Returns) + MSE(Vol) + Scaled Sharpe
            loss = 2.0 * l_focal + 5.0 * l_ret + 1.0 * l_vol + 0.0001 * l_sharpe
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            metrics['loss'] += loss.item()
            metrics['loss_focal'] += l_focal.item()
            metrics['loss_ret'] += l_ret.item()
            metrics['loss_vol'] += l_vol.item()
            metrics['loss_sharpe'] += l_sharpe.item()
            metrics['grad_norm'] = metrics.get('grad_norm', 0) + grad_norm.item()
            
            _, predicted = torch.max(action_logits.data, 1)
            metrics['total'] += y_action.size(0)
            metrics['correct'] += (predicted == y_action).sum().item()
        
        n = len(train_loader)
        return {k: v / n if k != 'total' and k != 'correct' else v for k, v in metrics.items()}
    
    def validate(self, model, val_loader, criterion_cls, criterion_reg, criterion_sharpe):
        """Validate with Multi-Task God-Tier Learning"""
        model.eval()
        metrics = {'loss': 0, 'loss_focal': 0, 'loss_ret': 0, 'loss_vol': 0, 'loss_sharpe': 0, 'correct': 0, 'total': 0}
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_action = y_batch[:, 0].to(self.device).long()
                y_return = y_batch[:, 1].to(self.device).float().unsqueeze(1)
                y_vol = y_batch[:, 2].to(self.device).float().unsqueeze(1)
                
                action_logits, pred_return, pred_vol = model(X_batch)
                
                l_focal = criterion_cls(action_logits, y_action)
                l_ret = criterion_reg(pred_return, y_return)
                l_vol = criterion_reg(pred_vol, y_vol)
                l_sharpe = criterion_sharpe(pred_return)
                
                loss = 2.0 * l_focal + 5.0 * l_ret + 1.0 * l_vol + 0.0001 * l_sharpe
                
                metrics['loss'] += loss.item()
                metrics['loss_focal'] += l_focal.item()
                metrics['loss_ret'] += l_ret.item()
                metrics['loss_vol'] += l_vol.item()
                metrics['loss_sharpe'] += l_sharpe.item()
                
                _, predicted = torch.max(action_logits.data, 1)
                metrics['total'] += y_action.size(0)
                metrics['correct'] += (predicted == y_action).sum().item()
        
        n = len(val_loader)
        return {k: v / n if k != 'total' and k != 'correct' else v for k, v in metrics.items()}
    
    def log_to_file(self, message):
        """Append log message to the current run's log file"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        filename = f"{log_dir}/run_{self.run_id}.log"
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")

    def train(self, epochs=200, batch_size=64, learning_rate=0.0005, session_num=1):
        """Main training loop with multi-timeframe data"""
        self.patience_counter = 0 # RESET PATIENCE FOR NEW SESSION
        
        config = {
            "run_id": self.run_id,
            "session_num": session_num,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "timeframes": self.timeframes
        }
        session_id = self.memory.log_session_start(self.symbol, config)
        
        print(f"\n[STARTING RUN: {self.run_id} | SESSION: {session_num}]")
        print("=" * 60)
        self.log_to_file(f"--- SESSION {session_num} START (DB ID: {session_id}) ---")
        
        # Use a very safe 365 day window for high-fidelity intraday data
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        val_split_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Download data for all timeframes
        train_data = self.download_multi_timeframe_data(start_date, val_split_date)
        val_data = self.download_multi_timeframe_data(val_split_date, end_date)
        
        # Combine all timeframes
        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []
        
        for tf in self.timeframes:
            print(f"\nProcessing {tf} timeframe...")
            
            # Training data
            if tf in train_data:
                X_train, y_train, _ = self.prepare_dataset(train_data[tf], tf, seq_length=60)
                if X_train is not None:
                    all_X_train.append(X_train)
                    all_y_train.append(y_train)
                    print(f"   Train samples: {len(X_train)}")
            
            # Validation data
            if tf in val_data:
                X_val, y_val, _ = self.prepare_dataset(val_data[tf], tf, seq_length=60)
                if X_val is not None:
                    all_X_val.append(X_val)
                    all_y_val.append(y_val)
                    print(f"   Val samples: {len(X_val)}")
        
        if not all_X_train:
            print("TOTAL DATA FAILURE. No training data prepared."); return
        
        # Concatenate all timeframes
        X_train = np.concatenate(all_X_train, axis=0)
        y_train = np.concatenate(all_y_train, axis=0)
        X_val = np.concatenate(all_X_val, axis=0)
        y_val = np.concatenate(all_y_val, axis=0)
        
        print(f"Total training samples: {len(X_train)}")
        print(f"Total validation samples: {len(X_val)}")
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        # Log class distribution
        unique, counts = np.unique(y_train[:, 0], return_counts=True)
        dist_str = ", ".join([f"Class {int(u)}: {c}" for u, c in zip(unique, counts)])
        print(f"Train Distribution: {dist_str}")
        self.log_to_file(f"Train Distribution: {dist_str}")

        # WEIGHTED SAMPLER: Force balance (using action labels)
        class_sample_count = np.array([len(np.where(y_train[:, 0] == t)[0]) for t in np.unique(y_train[:, 0])])
        weight = 1. / class_sample_count
        samples_weight = torch.from_numpy(np.array([weight[int(t)] for t in y_train[:, 0]])).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model if not loaded
        input_size = X_train.shape[2]
        if self.model is None:
            print("Initializing fresh neural weights...")
            self.model = MidnightModel(input_size=input_size).to(self.device)
        
        # Loss and optimizer
        criterion_cls = FocalLoss()
        criterion_reg = nn.MSELoss()
        criterion_sharpe = SharpeLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4) # Regularization as requested
        
        # Adaptive Learning Rate Decay (Pre-SWA stability)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # SWA: Stochastic Weight Averaging for God-Tier generalization
        swa_model = swa_utils.AveragedModel(self.model)
        swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=0.005)
        swa_start = int(epochs * 0.7) # Start SWA after 70% of epochs
        
        def signal_handler(sig, frame):
            print("\nEMERGENCY SHUTDOWN DETECTED. Saving state...")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
            }, 'models/emergency_checkpoint.pt')
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)

        print(f"\nTraining session {session_num} (God-Tier Target: {epochs} Epochs)")
        print("=" * 60)
        
        session_pbar = tqdm(range(epochs), desc=f"Session {session_num}", unit="epoch", leave=True)
        for epoch in session_pbar:
            start_time = time.time()
            epoch_reached = epoch
            
            t_metrics = self.train_epoch(self.model, train_loader, criterion_cls, criterion_reg, criterion_sharpe, optimizer)
            v_metrics = self.validate(self.model, val_loader, criterion_cls, criterion_reg, criterion_sharpe)
            
            train_loss, train_acc = t_metrics['loss'], (100 * t_metrics['correct'] / t_metrics['total'])
            val_loss, val_acc = v_metrics['loss'], (100 * v_metrics['correct'] / v_metrics['total'])
            
            duration = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            
            # SWA phase handling
            if epoch > swa_start:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
            else:
                scheduler.step(val_loss)
            self.memory.log_training_metric(session_id, epoch, train_loss, val_loss, train_acc, val_acc, lr)
            
            # Ultra-Detailed metrics to file
            msg = (f"S{session_num} E{epoch:3d} | {duration:.1f}s | LR: {lr:.6f} | GradNorm: {t_metrics.get('grad_norm', 0):.4f}\n"
                   f"   TRAIN: Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Dist: {t_metrics.get('correct', 0)}/{t_metrics.get('total', 1)}\n"
                   f"          Breakdown: [Focal: {t_metrics['loss_focal']:.4f}, Ret: {t_metrics['loss_ret']:.4f}, Vol: {t_metrics['loss_vol']:.4f}, Sharpe: {t_metrics['loss_sharpe']:.4f}]\n"
                   f"   VAL  : Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | Dist: {v_metrics.get('correct', 0)}/{v_metrics.get('total', 1)}\n"
                   f"          Breakdown: [Focal: {v_metrics['loss_focal']:.4f}, Ret: {v_metrics['loss_ret']:.4f}, Vol: {v_metrics['loss_vol']:.4f}, Sharpe: {v_metrics['loss_sharpe']:.4f}]")
            
            if epoch % 5 == 0:
                self.log_to_file(f"--- Epoch {epoch} Ultra-Log Checkpoint ---")
            
            # Update session progress bar
            session_pbar.set_postfix({
                'V_Acc': f"{val_acc:.1f}%",
                'V_L': f"{val_loss:.3f}",
                'T_L': f"{train_loss:.3f}",
                'Best': f"{self.best_val_loss:.3f}"
            })
            
            self.log_to_file(msg)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model of ALL time
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'scaler': self.scaler,
                    'arch': 'MidnightModel',
                    'session': session_num,
                    'epoch': epoch
                }, 'models/best_model.pt')
                self.log_to_file(f"NEW BEST VAL LOSS: {val_loss:.4f} (Saving...)")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience_limit:
                    msg = f"EARLY STOPPING at epoch {epoch}. Validation loss hasn't improved in {self.patience_limit} epochs."
                    print(msg)
                    self.log_to_file(msg)
                    
                    # RESTORE BEST WEIGHTS
                    checkpoint = torch.load('models/best_model.pt', map_location=self.device, weights_only=False)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print("Best weights restored. Ending current evolution cycle.")
                    break
        
        session_pbar.close() # Ensure bar is finalized without duplication
        
        print("\nSession complete! Model upgraded.")
        
        # Update Batch Norm statistics for SWA
        swa_utils.update_bn(train_loader, swa_model, device=self.device)
        
        # Strip "module." prefix from SWA state_dict for compatibility
        clean_state_dict = {k.replace('module.', ''): v for k, v in swa_model.state_dict().items()}
        
        torch.save({
            'model_state_dict': clean_state_dict,
            'scaler': self.scaler,
        }, 'models/final_model.pt')
        
        self.memory.log_session_end(session_id, self.best_val_loss)
        self.log_to_file(f"SESSION {session_num} COMPLETE. Best Val Loss: {self.best_val_loss:.4f}")
        
        return self.model

if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    trainer = MidnightTrainer(symbol='BTC-USD')
    # Trigger 20 sessions for real training as requested
    for i in range(20):
        trainer.train(epochs=100, batch_size=64, learning_rate=0.001, session_num=i+1)
        print("Cooling down for 10 seconds before next evolution...")
        time.sleep(10)
