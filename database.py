import sqlite3
import json
import datetime
from config import DB_PATH

# ===============================
# Kết nối DB
# ===============================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    return conn, c

# ===============================
# Khởi tạo bảng
# ===============================
def init_db():
    conn, c = get_db_connection()
    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY,
            date TEXT,
            amount REAL,
            category TEXT,
            source TEXT,
            sentiment TEXT,
            metadata TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            date TEXT,
            predicted_amount REAL,
            category TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

# ===============================
# Ghi giao dịch
# ===============================
def insert_transaction(transaction, sentiment, metadata):
    conn, c = get_db_connection()

    # Đảm bảo có 'date' và 'source'
    date = transaction.get("date") or datetime.datetime.now().strftime("%Y-%m-%d")
    source = transaction.get("source", "unknown")
    metadata_json = json.dumps(metadata)

    try:
        c.execute("""
            INSERT INTO transactions (date, amount, category, source, sentiment, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            date,
            transaction["amount"],
            transaction["category"],
            source,
            sentiment,
            metadata_json
        ))
    except Exception as e:
        print(f"❌ insert_transaction error: {e}")
    finally:
        conn.commit()
        conn.close()

# ===============================
# Ghi dự đoán
# ===============================
def insert_prediction(prediction):
    conn, c = get_db_connection()

    # Đảm bảo có 'date'
    date = prediction.get("date") or datetime.datetime.now().strftime("%Y-%m-%d")

    try:
        c.execute("""
            INSERT INTO predictions (date, predicted_amount, category, confidence)
            VALUES (?, ?, ?, ?)
        """, (
            date,
            prediction["predicted_amount"],
            prediction["category"],
            prediction["confidence"]
        ))
    except Exception as e:
        print(f"❌ insert_prediction error: {e}")
    finally:
        conn.commit()
        conn.close()

# ===============================
# Tổng hợp chi tiêu theo tháng
# ===============================
def get_monthly_summary():
    conn, c = get_db_connection()
    c.execute("""
        SELECT strftime('%Y-%m', date) as month, category, SUM(amount)
        FROM transactions
        GROUP BY month, category
    """)
    summary = {}
    for row in c.fetchall():
        month, category, total = row
        summary.setdefault(month, {})[category] = total
    conn.close()
    return summary
