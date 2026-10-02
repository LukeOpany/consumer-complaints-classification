import sqlite3
from pathlib import Path


def connection(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(path)
    conn.execute('''CREATE TABLE IF NOT EXISTS review_queue (
        id INTEGER PRIMARY KEY, narrative TEXT NOT NULL, suggested_product TEXT,
        score REAL, status TEXT NOT NULL DEFAULT 'pending', final_product TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)''')
    return conn


def add(path,text,result):
    conn=connection(path)
    try:
        with conn:
            return conn.execute('INSERT INTO review_queue(narrative,suggested_product,score) VALUES (?,?,?)',
                                (text,result['product'],result['score'])).lastrowid
    finally: conn.close()


def pending(path):
    conn=connection(path);conn.row_factory=sqlite3.Row
    try: return [dict(row) for row in conn.execute("SELECT * FROM review_queue WHERE status='pending' ORDER BY id")]
    finally: conn.close()


def resolve(path,identifier,product):
    if not product.strip(): raise ValueError('A final product is required')
    conn=connection(path)
    try:
        with conn: conn.execute("UPDATE review_queue SET status='reviewed',final_product=? WHERE id=? AND status='pending'",(product,identifier))
    finally: conn.close()
