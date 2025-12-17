from node2vec import Node2Vec
import networkx as nx
import os

class Node2VecStore:
    def __init__(self, graph_store, persist_path="n2v_model.pkl"):
        self.graph=graph_store.graph
        self.persist_path=persist_path
        self.model=None
    
    def train(self, dimensions=64, walk_length=30, num_walks=200):
        print("Initialising Node2Vec random walks...")
        n2v=Node2Vec(self.graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=1, quiet=True)

        print("Training Embeddings...")
        self.model=n2v.fit(window=10, min_count=1, batch_words=4)
        print("Training Complete")
    
    def get_similar_nodes(self, entity_name, top_k=2):
        key=str(entity_name)
        if self.model and key in self.model.wv:
            return self.model.wv.most_similar(key, topn=top_k)
        return []

    def save(self):
        if self.model:
            self.model.save(self.persist_path)
    
    def load(self):
        if os.path.exists(self.persist_path):
            from gensim.models import Word2Vec
            self.model=Word2Vec.load(self.persist_path)