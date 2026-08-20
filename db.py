import os
import json
import logging
import sqlite3
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DB_FILE = os.path.join(os.path.dirname(__file__), "sentinelpay.db")

logger = logging.getLogger("sentinelpay.db")

def _safe_float(val, default=0.0):
    if val is None or val == "":
        return default
    try:
        import math
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (ValueError, TypeError):
        return default

def _safe_str(val, default=""):
    if val is None:
        return default
    return str(val)

class DatabaseManager:
    """
    Database Manager supporting PostgreSQL with automatic SQLite fallback.
    Manages transaction logs and risk analysis audit trail.
    """
    def __init__(self):
        self.use_sqlite = False
        self._init_db()

    def _get_pg_connection(self):
        import psycopg2
        url = DATABASE_URL
        if not url:
            host = os.getenv("PGHOST", "localhost")
            port = os.getenv("PGPORT", "5432")
            user = os.getenv("PGUSER", "postgres")
            password = os.getenv("PGPASSWORD", "")
            dbname = os.getenv("PGDATABASE", "sentinelpay_db")
            if password:
                url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
            else:
                url = f"postgresql://{user}@{host}:{port}/{dbname}"
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        conn = psycopg2.connect(url)
        conn.autocommit = True
        return conn

    def _get_sqlite_connection(self):
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        # Try PostgreSQL first
        try:
            conn = self._get_pg_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS transactions (
                        id SERIAL PRIMARY KEY,
                        transaction_id VARCHAR(64) UNIQUE,
                        card_id VARCHAR(64),
                        trans_date_trans_time VARCHAR(64),
                        amount_inr DOUBLE PRECISION,
                        merchant_name VARCHAR(255),
                        merchant_category VARCHAR(100),
                        channel VARCHAR(50),
                        ip_country VARCHAR(100),
                        transaction_city VARCHAR(100),
                        device_id VARCHAR(100),
                        ml_fraud_probability DOUBLE PRECISION,
                        operational_risk_score DOUBLE PRECISION,
                        risk_level VARCHAR(20),
                        decision VARCHAR(50),
                        recommended_action VARCHAR(100),
                        reasons TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_transactions_txid ON transactions(transaction_id);
                """)
            conn.close()
            self.use_sqlite = False
            logger.info("PostgreSQL database initialized successfully.")
            return
        except Exception as e:
            logger.warning(f"PostgreSQL initialization failed ({e}). Falling back to SQLite database at {DB_FILE}.")
            self.use_sqlite = True

        # SQLite fallback initialization
        try:
            conn = self._get_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE,
                    card_id TEXT,
                    trans_date_trans_time TEXT,
                    amount_inr REAL,
                    merchant_name TEXT,
                    merchant_category TEXT,
                    channel TEXT,
                    ip_country TEXT,
                    transaction_city TEXT,
                    device_id TEXT,
                    ml_fraud_probability REAL,
                    operational_risk_score REAL,
                    risk_level TEXT,
                    decision TEXT,
                    recommended_action TEXT,
                    reasons TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_txid ON transactions(transaction_id);")
            conn.commit()
            conn.close()
            logger.info("SQLite database initialized successfully.")
        except Exception as ex:
            logger.error(f"SQLite initialization failed: {ex}")

    def save_transaction(self, tx_res):
        """Save a single analyzed transaction result."""
        if not tx_res or not isinstance(tx_res, dict):
            return

        reasons_json = json.dumps(tx_res.get("reasons", []))
        
        tx_id = _safe_str(tx_res.get("transaction_id", ""))
        card_id = _safe_str(tx_res.get("card_id", ""))
        trans_time = _safe_str(tx_res.get("trans_date_trans_time", ""))
        amount_inr = _safe_float(tx_res.get("amount_inr"))
        merchant_name = _safe_str(tx_res.get("merchant_name"))
        merchant_category = _safe_str(tx_res.get("merchant_category"))
        channel = _safe_str(tx_res.get("channel"))
        ip_country = _safe_str(tx_res.get("ip_country"))
        transaction_city = _safe_str(tx_res.get("transaction_city"))
        device_id = _safe_str(tx_res.get("device_id"))
        ml_prob = _safe_float(tx_res.get("ml_fraud_probability"))
        op_score = _safe_float(tx_res.get("operational_risk_score"))
        risk_level = _safe_str(tx_res.get("risk_level"))
        decision = _safe_str(tx_res.get("decision"))
        action = _safe_str(tx_res.get("recommended_action"))

        if not self.use_sqlite:
            try:
                conn = self._get_pg_connection()
                with conn.cursor() as cursor:
                    sql = """
                        INSERT INTO transactions (
                            transaction_id, card_id, trans_date_trans_time, amount_inr,
                            merchant_name, merchant_category, channel, ip_country,
                            transaction_city, device_id, ml_fraud_probability,
                            operational_risk_score, risk_level, decision,
                            recommended_action, reasons
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (transaction_id) DO UPDATE SET
                            operational_risk_score = EXCLUDED.operational_risk_score,
                            risk_level = EXCLUDED.risk_level,
                            decision = EXCLUDED.decision,
                            recommended_action = EXCLUDED.recommended_action,
                            reasons = EXCLUDED.reasons;
                    """
                    cursor.execute(sql, (
                        tx_id, card_id, trans_time, amount_inr,
                        merchant_name, merchant_category, channel, ip_country,
                        transaction_city, device_id, ml_prob,
                        op_score, risk_level, decision,
                        action, reasons_json
                    ))
                conn.close()
                return
            except Exception as e:
                logger.error(f"Failed to save transaction to PostgreSQL ({e}). Retrying with SQLite fallback.")
                self.use_sqlite = True

        # SQLite execution
        try:
            conn = self._get_sqlite_connection()
            cursor = conn.cursor()
            sql = """
                INSERT INTO transactions (
                    transaction_id, card_id, trans_date_trans_time, amount_inr,
                    merchant_name, merchant_category, channel, ip_country,
                    transaction_city, device_id, ml_fraud_probability,
                    operational_risk_score, risk_level, decision,
                    recommended_action, reasons
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(transaction_id) DO UPDATE SET
                    operational_risk_score = excluded.operational_risk_score,
                    risk_level = excluded.risk_level,
                    decision = excluded.decision,
                    recommended_action = excluded.recommended_action,
                    reasons = excluded.reasons;
            """
            cursor.execute(sql, (
                tx_id, card_id, trans_time, amount_inr,
                merchant_name, merchant_category, channel, ip_country,
                transaction_city, device_id, ml_prob,
                op_score, risk_level, decision,
                action, reasons_json
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save transaction to SQLite: {e}")

    def save_batch_transactions(self, tx_list):
        """Save a list of analyzed transaction results from batch CSV processing."""
        if not tx_list:
            return
        for tx_res in tx_list:
            self.save_transaction(tx_res)

    def _row_to_dict(self, row):
        if not row:
            return None
        if isinstance(row, dict):
            d = dict(row)
        elif hasattr(row, 'keys'):
            d = dict(row)
        else:
            cols = ["id", "transaction_id", "card_id", "trans_date_trans_time", "amount_inr",
                    "merchant_name", "merchant_category", "channel", "ip_country",
                    "transaction_city", "device_id", "ml_fraud_probability",
                    "operational_risk_score", "risk_level", "decision",
                    "recommended_action", "reasons", "created_at"]
            d = dict(zip(cols, row))
        
        reasons_val = d.get("reasons")
        if isinstance(reasons_val, str):
            try:
                d["reasons"] = json.loads(reasons_val)
            except Exception:
                d["reasons"] = [reasons_val] if reasons_val else []
        elif not isinstance(reasons_val, list):
            d["reasons"] = []
            
        return d

    def get_transaction(self, tx_id):
        """Retrieve a saved transaction by transaction_id."""
        if not tx_id:
            return None
        if not self.use_sqlite:
            try:
                import psycopg2.extras
                conn = self._get_pg_connection()
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute("SELECT * FROM transactions WHERE transaction_id = %s LIMIT 1", (tx_id,))
                    row = cursor.fetchone()
                conn.close()
                return self._row_to_dict(row) if row else None
            except Exception as e:
                logger.error(f"Failed to fetch transaction from PostgreSQL: {e}")
                self.use_sqlite = True

        try:
            conn = self._get_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM transactions WHERE transaction_id = ? LIMIT 1", (tx_id,))
            row = cursor.fetchone()
            conn.close()
            return self._row_to_dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to fetch transaction from SQLite: {e}")
            return None

    def get_recent_transactions(self, limit=1000):
        """Retrieve recent saved transactions."""
        if not self.use_sqlite:
            try:
                import psycopg2.extras
                conn = self._get_pg_connection()
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute("SELECT * FROM transactions ORDER BY id DESC LIMIT %s", (limit,))
                    rows = cursor.fetchall()
                conn.close()
                return [self._row_to_dict(r) for r in rows if r]
            except Exception as e:
                logger.error(f"Failed to fetch recent transactions from PostgreSQL: {e}")
                self.use_sqlite = True

        try:
            conn = self._get_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM transactions ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            conn.close()
            return [self._row_to_dict(r) for r in rows if r]
        except Exception as e:
            logger.error(f"Failed to fetch recent transactions from SQLite: {e}")
            return []

    def clear_transactions(self):
        """Clear all stored transactions for session reset."""
        if not self.use_sqlite:
            try:
                conn = self._get_pg_connection()
                with conn.cursor() as cursor:
                    cursor.execute("TRUNCATE TABLE transactions;")
                conn.close()
                return
            except Exception as e:
                logger.error(f"Failed to clear transactions from PostgreSQL: {e}")
                self.use_sqlite = True

        try:
            conn = self._get_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transactions;")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to clear transactions from SQLite: {e}")

# Instantiate singleton DB manager
db_manager = DatabaseManager()
