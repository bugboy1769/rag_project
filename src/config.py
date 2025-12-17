import os

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL','hf.co/CompendiumLabs/bge-base-en-v1.5-gguf')
LANGUAGE_MODEL = os.getenv('LANGUAGE_MODEL','hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF')
OLLAMA_BASE_URL=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

CHUNK_SIZE=1
TOP_K=3