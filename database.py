import sqlite3
import json
import datetime
from config import DB_PATH
from postgre_db import TransactionCreate, create_transaction

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
async def insert_transaction(transaction, sentiment, metadata):
    category_id = transaction.get("category_id")
    amount = transaction.get("amount")
    user_id = transaction.get("user_id")
    try:
        transactionCreate = TransactionCreate(
            userId=user_id, categoryId=category_id, amount=amount, currencyId='669d209b-99ac-401d-a441-8fa7bb387d4c')
        print("TRANSACTION CREATE:")
        print(transactionCreate.model_dump_json())
        await create_transaction(transactionCreate)
    except Exception as e:
        print(f"❌ insert_transaction error: {e}")

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
