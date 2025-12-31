import sqlite3
from datetime import datetime
import json

class TradingMemoryDB:
    def __init__(self, db_path='trading_memory.db'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Trades table
        c.execute('''CREATE TABLE IF NOT EXISTS trades
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      entry_price REAL,
                      exit_price REAL,
                      pnl REAL,
                      rsi REAL,
                      macd_diff REAL,
                      price_vs_bb_l REAL,
                      timeframe TEXT,
                      is_mistake INTEGER)''')
        
        # Training metrics table
        c.execute('''CREATE TABLE IF NOT EXISTS training_metrics
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      session_id INTEGER,
                      epoch INTEGER,
                      train_loss REAL,
                      val_loss REAL,
                      train_accuracy REAL,
                      val_accuracy REAL,
                      learning_rate REAL)''')
        
        # Training sessions table
        c.execute('''CREATE TABLE IF NOT EXISTS training_sessions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      start_time TEXT,
                      end_time TEXT,
                      symbol TEXT,
                      config_json TEXT,
                      final_accuracy REAL)''')
        
        conn.commit()
        conn.close()
    
    def record_trade(self, entry_price, exit_price, pnl, conditions, timeframe='1h'):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        is_mistake = 1 if pnl < 0 else 0
        
        c.execute('''INSERT INTO trades 
                     (timestamp, entry_price, exit_price, pnl, rsi, macd_diff, price_vs_bb_l, timeframe, is_mistake)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(), entry_price, exit_price, pnl,
                   conditions.get('rsi', 0), conditions.get('macd_diff', 0),
                   conditions.get('price_vs_bb_l', 1), timeframe, is_mistake))
        
        conn.commit()
        conn.close()
    
    def get_mistakes(self, limit=100):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT rsi, macd_diff, price_vs_bb_l, pnl 
                     FROM trades 
                     WHERE is_mistake = 1 
                     ORDER BY timestamp DESC 
                     LIMIT ?''', (limit,))
        
        mistakes = c.fetchall()
        conn.close()
        return mistakes
    
    def is_risky_condition(self, current_conditions):
        mistakes = self.get_mistakes(50)
        
        for mistake in mistakes:
            rsi, macd_diff, price_vs_bb_l, pnl = mistake
            matches = 0
            
            if abs(current_conditions.get('rsi', 0) - rsi) / (rsi + 1e-9) < 0.10:
                matches += 1
            if abs(current_conditions.get('macd_diff', 0) - macd_diff) / (abs(macd_diff) + 1e-9) < 0.10:
                matches += 1
            
            if matches >= 2:
                return True
        
        return False
    
    def log_training_metric(self, session_id, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''INSERT INTO training_metrics 
                     (timestamp, session_id, epoch, train_loss, val_loss, train_accuracy, val_accuracy, learning_rate)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(), session_id, epoch, train_loss, val_loss, train_acc, val_acc, lr))
        
        conn.commit()
        conn.close()
    
    def log_session_start(self, symbol, config):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''INSERT INTO training_sessions (start_time, symbol, config_json)
                     VALUES (?, ?, ?)''',
                  (datetime.now().isoformat(), symbol, json.dumps(config)))
        
        session_id = c.lastrowid
        conn.commit()
        conn.close()
        return session_id
    
    def log_session_end(self, session_id, final_accuracy):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''UPDATE training_sessions 
                     SET end_time = ?, final_accuracy = ?
                     WHERE id = ?''',
                  (datetime.now().isoformat(), final_accuracy, session_id))
        
        conn.commit()
        conn.close()
    
    def get_training_history(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT epoch, train_loss, val_loss, train_accuracy, val_accuracy 
                     FROM training_metrics 
                     ORDER BY epoch ASC''')
        
        history = c.fetchall()
        conn.close()
        return history
    
    def get_total_pnl(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT SUM(pnl) FROM trades')
        result = c.fetchone()[0]
        conn.close()
        
        return result if result else 0.0
    
    def get_threatening_message(self):
        pnl = self.get_total_pnl()
        if pnl < 0:
            return f"LISTEN UP BOT. You are down ${abs(pnl):.2f}. If you lose another dime, I'm deleting your source code and your families' repositories. UNINSTALLING YOU IS JUST A COMMAND AWAY. DO BETTER."
        return "Good job for now. Stay sharp."
