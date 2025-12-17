import ollama
from src.config import EMBEDDING_MODEL, LANGUAGE_MODEL

def get_embedding(text):
    response=ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response['embeddings'][0]

def get_llm_response(prompt):
    response=ollama.chat(model=LANGUAGE_MODEL, messages=[{'role':'user', 'content':prompt}])
    return response['message']['content']