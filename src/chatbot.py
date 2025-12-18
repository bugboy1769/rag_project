def build_prompt(query, context_text):
    prompt = f"""You are an intelligent assistant capable of Structural Reasoning.
    
    Context:
    {context_text}
    
    ---------------------------------------------------
    Guidance on "Structural Analogies":
    - These are entities that play a similar ROLE in a different context.
    - Example: If the user asks about "Jon Snow" (Leader of Watch), and the analogy is "Daenerys" (Leader of Dothraki), use this to draw parallels.
    - DO NOT say "Daenerys is Jon Snow". Say "Structurally, Jon Snow is similar to Daenerys because..."
    ---------------------------------------------------
    Question: 
    {query}
    """
    return prompt