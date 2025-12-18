import pickle
import os

class VectorStore:
    def __init__(self, persist_path=None):
        self.db=[]
        self.persist_path=persist_path
    
    def add_documents(self, chunks, embeddings, metadata_list=None):
        """
        chunks: List[str]
        embeddings: List[List[float]]
        """
        if metadata_list is None:
            metadata_list=[{} for _ in chunks]

        for chunk, embedding, meta in zip(chunks, embeddings, metadata_list):
            self.db.append((chunk, embedding, meta))
    
    def get_all(self):
        return self.db

    def filter_by_metadata(self, key, value):
        """
        Returns all documents where metadata[key] == value
        """
        results = []
        for item in self.db:
            if len(item) == 3:
                chunk, embedding, meta = item
                if meta.get(key) == value:
                    results.append((chunk, 1.0, meta)) # 1.0 similarity for exact match
        return results

    def save(self):
        if self.persist_path:
            with open(self.persist_path, 'wb') as f:
                pickle.dump(self.db, f)
    
    def load(self):
        if self.persist_path and os.path.exists(self.persist_path):
            with open(self.persist_path, 'rb') as f:
                self.db=pickle.load(f)