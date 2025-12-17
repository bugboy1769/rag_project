import pickle
import os

class VectorStore:
    def __init__(self, persist_path=None):
        self.db=[]
        self.persist_path=persist_path
    
    def add_documents(self, chunks, embeddings):
        """
        chunks: List[str]
        embeddings: List[List[float]]
        """
        for chunk, embedding in zip(chunks, embeddings):
            self.db.append((chunk, embedding))
    
    def get_all(self):
        return self.db

    def save(self):
        if self.persist_path:
            with open(self.persist_path, 'wb') as f:
                pickle.dump(self.db, f)
    
    def load(self):
        if self.persist_path and os.path.exists(self.persist_path):
            with open(self.persist_path, 'rb') as f:
                self.db=pickle.load(f)