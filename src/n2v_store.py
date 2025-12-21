from node2vec import Node2Vec
import networkx as nx
import os

class Node2VecStore:
    def __init__(self, graph_store, persist_path="n2v_model.pkl"):
        self.graph=graph_store.graph
        self.persist_path=persist_path
        self.model=None
    
    def train(self, dimensions=64, walk_length=100, num_walks=500):

        # We want to favor Structural Equivalence (q > 1) vs Community (q < 1)
        # We are using p=1, q=4 to encourage BFS-like behavior (Structural Roles)
        print("Initializing Node2Vec with Structural Equivalence params (p=1, q=4)...")
        n2v = Node2Vec(self.graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=1, quiet=True, p=1, q=4)

        print("Training Embeddings...")
        self.model = n2v.fit(window=10, min_count=1, batch_words=4)
        print("Training Complete")
    
    def get_similar_nodes(self, entity_name, top_k=2):
        if not self.model:
            return []
            
        key = str(entity_name)
        
        # 1. Direct Match
        if key in self.model.wv:
            return self.model.wv.most_similar(key, topn=top_k)
            
        # 2. Fuzzy Match (Fallback)
        import difflib
        vocab = list(self.model.wv.index_to_key)
        matches = difflib.get_close_matches(key, vocab, n=1, cutoff=0.6)
        
        if matches:
            best_match = matches[0]
            print(f"DEBUG: Fuzzy Match Node2Vec: '{key}' -> '{best_match}'")
            return self.model.wv.most_similar(best_match, topn=top_k)
            
        return []

    def save(self):
        if self.model:
            self.model.save(self.persist_path)
    
    def load(self):
        if os.path.exists(self.persist_path):
            from gensim.models import Word2Vec
            self.model=Word2Vec.load(self.persist_path)