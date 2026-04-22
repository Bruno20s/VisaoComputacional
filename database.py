import sqlite3
import numpy as np
from datetime import datetime


class VectorDatabase:
    def __init__(self, db_path="embeddings.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()
        self._ensure_columns()

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

    def _ensure_columns(self):
        cursor = self.conn.cursor()

        cursor.execute("PRAGMA table_info(embeddings)")
        colunas = [col[1] for col in cursor.fetchall()]

        novas_colunas = [
            ("classe1", "TEXT"),
            ("score1", "REAL"),
            ("classe2", "TEXT"),
            ("score2", "REAL"),
            ("classe3", "TEXT"),
            ("score3", "REAL"),
        ]

        for nome, tipo in novas_colunas:
            if nome not in colunas:
                cursor.execute(f"ALTER TABLE embeddings ADD COLUMN {nome} {tipo}")

        self.conn.commit()

    # 🔧 insert agora com classificações
    def insert_vector(self, object_id, vector, arquivo, classificacoes):
        cursor = self.conn.cursor()

        vector_bytes = vector.astype(np.float32).tobytes()
        data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # garante 3 classificações
        classificacoes = classificacoes[:3]
        while len(classificacoes) < 3:
            classificacoes.append(("desconhecido", 0.0))

        (c1, s1), (c2, s2), (c3, s3) = classificacoes

        cursor.execute("""
        INSERT INTO embeddings (
            object_id, arquivo, data_hora, vector,
            classe1, score1,
            classe2, score2,
            classe3, score3
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            object_id, arquivo, data_hora, vector_bytes,
            c1, s1,
            c2, s2,
            c3, s3
        ))

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
        SELECT id, object_id, arquivo, data_hora, vector,
               classe1, score1,
               classe2, score2,
               classe3, score3
        FROM embeddings
        """)

        rows = cursor.fetchall()

        results = []

        for (rid, object_id, arquivo, data_hora, vector_blob,
             c1, s1, c2, s2, c3, s3) in rows:

            vector = np.frombuffer(vector_blob, dtype=np.float32)

            classes = [
                (c1, s1),
                (c2, s2),
                (c3, s3)
            ]

            results.append((
                rid, object_id, arquivo, data_hora, vector, classes
            ))

        return results

    def reset_table(self):
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS embeddings")
        self.conn.commit()
        self.create_table()
        self._ensure_columns()

    def clear_table(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM embeddings")
        self.conn.commit()

    def fechar(self):
        self.conn.close()