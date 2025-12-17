
def cosine_similarity(a, b):
    dot_product=sum([x*y for x, y in zip(a,b)])
    norm_a=sum([x**2 for x in a])**0.5
    norm_b=sum([x**2 for x in b])**0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product/(norm_a*norm_b)

def retrieve_topk(query_embedding, vector_db, top_k=3):
    similarities=[]
    db_items = vector_db.get_all()
    similarities = []
    for item in db_items:
        if len(item)==3:
             chunk, embedding, meta = item
        else:
             chunk, embedding=item
             meta={}
        similarity=cosine_similarity(embedding, query_embedding)
        similarities.append((chunk, similarity, meta))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def retrieve_hybrid(query_embedding, query_text, vector_store, graph_store, top_k=3):
    #1. Semantic Search
    vector_results=retrieve_topk(query_embedding, vector_store, top_k)

    #2. Graph Search
    graph_context=[]

    all_nodes=list(graph_store.graph.nodes())
    found_entities=[]

    for node in all_nodes:
        if str(node).lower() in query_text.lower():
            found_entities.append(node)
    for entity in found_entities:
            neighbors=graph_store.get_neighbors(entity)
            graph_context.extend(neighbors)
    return vector_results, graph_context

