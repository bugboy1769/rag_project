import pickle
import os
import networkx as nx

class GraphStore:
    def __init__(self, persist_path=None):
        self.graph=nx.DiGraph()
        self.persist_path=persist_path
    
    def add_triplets(self, subject, predicate, object):
        """
        Add a relationship: (Subject) -> [Predicate] -> Object
        """
        self.graph.add_edge(subject, object, relation=predicate)
    
    def get_neighbors(self, entity, depth=1):
        if entity not in self.graph:
            return []
        
        #Immediate neighbours for now
        neighbors=[]
        for neighbor in self.graph.neighbors(entity):
            relation=self.graph[entity][neighbor]['relation']
            neighbors.append((entity, relation, neighbor))
        return neighbors
    
    def save(self):
        if self.persist_path:
            with open(self.persist_path, 'wb') as f:
                pickle.dump(self.graph, f)

    def load(self):
        if self.persist_path and os.path.exists(self.persist_path):
            with open(self.persist_path, 'rb') as f:
                self.graph=pickle.load(f)
    