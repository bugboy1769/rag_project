def build_prompt(query, context_text):
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.
    
    Context:
    {context_text}
    
    Question: 
    {query}
    """
    return prompt