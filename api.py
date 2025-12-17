from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.vector_store import VectorStore
from src.ingestion import load_file
from src.retrieval import retrieve_topk
from src.models import get_embedding, get_llm_response
from src.chatbot import build_prompt

app=FastAPI(title="RAG Experimentation API")

# -- Global State --
store=VectorStore(persist_path="vectordb.pkl")

try:
    store.load()
    print("Found and loaded existing vector db")
except:
    print("No existing database found. Please call /index first.")

# -- Data Models --
class IndexRequest(BaseModel):
    file_path:str

class ChatRequest(BaseModel):
    query:str

# -- Endpoints --
@app.get("/")
def health_check():
    return {'status':'ok', 'db_size':len(store.get_all())}

@app.post("/index")
def index_document(request: IndexRequest):
    """
    Load file, chunk it, embed it and add to vector db
    """
    try:
        chunks=load_file(request.file_path)

        #Embed
        embeddings=[]
        for chunk in chunks:
            embeddings.append(get_embedding(chunk))
        
        store.add_documents(chunks, embeddings)
        store.save()

        return {
            "message":"Document Indexed Successfully",
            "chunks_added":len(chunks),
            "total_chunks":len(store.get_all())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat(request: ChatRequest):

    query_emb=get_embedding(request.query)

    relevant_docs=retrieve_topk(query_emb, store, top_k=3)

    prompt=build_prompt(request.query, relevant_docs)
    answer=get_llm_response(prompt)

    return {
        "query":request.query,
        "answer":answer,
        "fetched docs":[doc[0] for doc in relevant_docs]
    }
