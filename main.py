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
        chunks = load_file('data/got_plot.txt')
    
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
        n2v_store.save()
    else:
        n2v_store.load()
    

    #4. Retrieval Loop
    while True:
        query=input("\nAsk a question or type 'q' to quit\n")
        if query.lower()=='q':
            break

        print('\nEmbedding Query')
        query_emb=get_embedding(query)
        print('\nFetching Relevant Context')
        v_docs=retrieve_topk(query_emb, v_store, top_k=3)
        g_docs=retrieve_topk(query_emb, t_store, top_k=5)

        #Structural Expansion
        structural_context=[]
        if g_docs:
            top_result=g_docs[0]
            top_meta=top_result[2] #metadata

            potential_subject=top_meta.get("anchor_subject")
            print(f"DEBUG: Node2Vec Anchor Subject -> {potential_subject}")

            if potential_subject:
                similar_nodes=n2v_store.get_similar_nodes(potential_subject, top_k=2)

            for node, score in similar_nodes:
                node_emb=get_embedding(node)
                related_facts=retrieve_topk(node_emb, t_store, top_k=1)
                structural_context.extend(related_facts)
        #Format graph context
        g_text="\n".join([f"- {item[0]}" for item in g_docs])
        s_text="\n".join([f"- {item[0]} (Structurally related to query)" for item in structural_context])

        full_context=f"Vector Context:\n{v_docs}\n\nGraph Context:\n{g_text}\n\nStructural Analogies:\n{s_text}"
        print(f"\nDEBUG: The full context being passed to the LLM is as follows: {full_context}")

        print('\nBuilding Prompt')
        prompt=build_prompt(query, full_context)
        print('\nFetching LLM Response')
        response=get_llm_response(prompt)

        print(f"\nResponse: {response}")

if __name__=='__main__':
    main()
