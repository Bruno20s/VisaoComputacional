import sqlite3
import numpy as np
from datetime import datetime


class VectorDatabase:
    def __init__(self, db_path="embeddings.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_id INTEGER,
            arquivo TEXT,
            data_hora TEXT,
            vector BLOB
        )
        """)

        self.conn.commit()

    def insert_vector(self, object_id, vector, arquivo):
        cursor = self.conn.cursor()

        # converte numpy array para bytes
        vector_bytes = vector.astype(np.float32).tobytes()

        # data/hora atual
        data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
        INSERT INTO embeddings (object_id, arquivo, data_hora, vector)
        VALUES (?, ?, ?, ?)
        """, (object_id, arquivo, data_hora, vector_bytes))

        self.conn.commit()

    def get_vector(self, object_id):
        cursor = self.conn.cursor()

        cursor.execute(
            "SELECT vector FROM embeddings WHERE object_id=?",
            (object_id,)
        )

        row = cursor.fetchone()

        if row:
            return np.frombuffer(row[0], dtype=np.float32)

        return None

    def get_all_vectors(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        SELECT id, object_id, arquivo, data_hora, vector
        FROM embeddings
        """)

        rows = cursor.fetchall()

        results = []

        for rid, object_id, arquivo, data_hora, vector_blob in rows:
            vector = np.frombuffer(vector_blob, dtype=np.float32)
            results.append((rid, object_id, arquivo, data_hora, vector))

        return results

    def reset_table(self):
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS embeddings")
        self.conn.commit()
        self.create_table()

    def clear_table(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM embeddings")
        self.conn.commit()

    def fechar(self):
        self.conn.close()