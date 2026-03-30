import sqlite3
import numpy as np

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
            vector BLOB
        )
        """)
        self.conn.commit()

    def insert_vector(self, object_id, vector):
        cursor = self.conn.cursor()

        # converte numpy array para bytes
        vector_bytes = vector.astype(np.float32).tobytes()

        cursor.execute(
            "INSERT INTO embeddings (object_id, vector) VALUES (?, ?)",
            (object_id, vector_bytes)
        )
        self.conn.commit()
        
        

    def get_vector(self, object_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT vector FROM embeddings WHERE object_id=?", (object_id,))
        row = cursor.fetchone()

        if row:
            return np.frombuffer(row[0], dtype=np.float32)

        return None
    
    def get_all_vectors(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT object_id, vector FROM embeddings")

        rows = cursor.fetchall()

        results = []

        for object_id, vector_blob in rows:
            vector = np.frombuffer(vector_blob, dtype=np.float32)
            results.append((object_id, vector))

        return results