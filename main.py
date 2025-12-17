import os
from src.config import EMBEDDING_MODEL
from src.ingestion import load_file, extract_triplets
from src.models import get_embedding, get_llm_response
from src.vector_store import VectorStore
from src.retrieval import retrieve_topk
from src.chatbot import build_prompt
from src.graph_store import GraphStore

def main():
    # 1. Setup
    print("Initializing ...")
    v_store = VectorStore(persist_path="vectordb.pkl")
    g_store = GraphStore(persist_path="graph.pkl")
    t_store=VectorStore(persist_path="tripletdb.pkl")
    
    # Try to load existing data
    v_store.load()
    t_store.load()
    
    # Check if we need to ingest (if DB is empty)
    if not v_store.get_all() or not g_store.graph.nodes:
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
            for t in triplets:
                g_store.add_triplets(t['subject'], t['predicate'], t['object'])
                text_rep=f"{t['subject']} {t['predicate']} {t['object']}"

                triplet_texts.append(text_rep)
                emb=get_embedding(text_rep)
                triplet_embeddings.append(emb)
            
            if triplet_texts:
                t_store.add_documents(triplet_texts, triplet_embeddings)

        v_store.add_documents(chunks, embeddings)
        v_store.save()
        g_store.save()
        t_store.save()
    else:
        print("Loaded existing data from disk. Skipping ingestion.")
    

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

        #Format graph context
        g_text="\n".join([f"- {item[0]}" for item in g_docs])

        full_context=f"Vector Context:\n{v_docs}\n\nGraph Context:\n{g_text}"
        print(f"\nDEBUG: The full context being passed to the LLM is as follows: {full_context}")

        print('\nBuilding Prompt')
        prompt=build_prompt(query, full_context)
        print('\nFetching LLM Response')
        response=get_llm_response(prompt)

        print(f"\nResponse: {response}")

if __name__=='__main__':
    main()
