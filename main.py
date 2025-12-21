import os
from src.config import EMBEDDING_MODEL
from src.ingestion import load_file, extract_triplets
from src.models import get_embedding, get_llm_response
from src.vector_store import VectorStore
from src.retrieval import retrieve_topk
from src.chatbot import build_prompt
from src.graph_store import GraphStore
from src.n2v_store import Node2VecStore

def main():
    # 1. Setup
    print("Initializing ...")
    v_store = VectorStore(persist_path="vectordb.pkl")
    g_store = GraphStore(persist_path="graph.pkl")
    t_store=VectorStore(persist_path="tripletdb.pkl")
    
    # Try to load existing data
    v_store.load()
    t_store.load()
    g_store.load()

    # Check if we need to ingest (if DB is empty)
    if not v_store.get_all() or not t_store.get_all():
        print("Data stores are empty. Starting ingestion...")
        
        # 2. Ingestion
        print("Loading data ...")
        chunks = load_file('data/cat_facts.txt')
    
        # 3. Embedding and Ingestion
        print(f"Embedding {len(chunks)} chunks ...")
        embeddings = []
        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            embeddings.append(emb)
    
            print(f"Extracting graph data from chunk {i}...")
            triplets = extract_triplets(chunk)
            triplet_texts=[]
            triplet_embeddings=[]
            triplet_metadatas=[]
            for t in triplets:
                g_store.add_triplets(t['subject'], t['predicate'], t['object'])
                text_rep=f"{t['subject']} {t['predicate']} {t['object']}"

                triplet_texts.append(text_rep)
                emb=get_embedding(text_rep)
                triplet_embeddings.append(emb)
                triplet_metadatas.append({"anchor_subject":t['subject']})
            
            if triplet_texts:
                t_store.add_documents(triplet_texts, triplet_embeddings, triplet_metadatas)

        v_store.add_documents(chunks, embeddings)
        v_store.save()
        g_store.save()
        t_store.save()
    else:
        print("Loaded existing data from disk. Skipping ingestion.")
    
    n2v_store=Node2VecStore(g_store)
    if not os.path.exists("n2v_model.pkl") and g_store.graph.nodes:
        print("Training Node2Vec Topological Model...")
        n2v_store.train()
        print(f"DEBUG: Nodes in Node2Vec model: {list(n2v_store.model.wv.index_to_key)[:10]}")
        n2v_store.save()
    else:
        n2v_store.load()
    
    # Optional: Visualize Graph for Debugging
    from src.visualize import draw_graph
    draw_graph(g_store)


    #4. Retrieval Loop
    while True:
        query=input("\nAsk a question or type 'q' to quit\n")
        if query.lower()=='q':
            break

        print('\nEmbedding Query')
        query_emb=get_embedding(query)
        
        print('\nFetching Semantic Context (Vector Search)')
        v_docs=retrieve_topk(query_emb, v_store, top_k=3)
        semantic_text = "\n".join([f"- {item[0]}" for item in v_docs])
        
        # Graph-Expanded Structural Context
        print('\nFetching Structural Context (Graph-Expanded)')
        structural_chunks = []
        
        # Step 1: Find anchor entity from triplet store
        g_docs=retrieve_topk(query_emb, t_store, top_k=1)
        if g_docs:
            top_meta = g_docs[0][2]
            anchor = top_meta.get("anchor_subject")
            print(f"DEBUG: Anchor Entity -> {anchor}")
            
            if anchor:
                # Step 2: Get structurally similar nodes via Node2Vec
                similar_nodes = n2v_store.get_similar_nodes(anchor, top_k=3)
                print(f"DEBUG: Structurally Similar Nodes -> {similar_nodes}")
                
                # Step 3: For each similar node, search VECTOR STORE for original text
                for node, score in similar_nodes:
                    node_emb = get_embedding(node)
                    related_chunks = retrieve_topk(node_emb, v_store, top_k=1)
                    structural_chunks.extend(related_chunks)
        
        structural_text = "\n".join([f"- {item[0]}" for item in structural_chunks]) if structural_chunks else "(No structural context found)"
        
        print(f"\nDEBUG: Semantic Context:\n{semantic_text}")
        print(f"\nDEBUG: Structural Context:\n{structural_text}")

        print('\nBuilding Prompt')
        prompt = build_prompt(query, semantic_text, structural_text)
        print('\nFetching LLM Response')
        response = get_llm_response(prompt)

        print(f"\nResponse: {response}")

if __name__=='__main__':
    main()
