def build_prompt(query, semantic_context, structural_context):
    prompt = f"""You are an intelligent assistant with access to two types of context.

## SEMANTIC CONTEXT (Direct Similarity)
Use this for factual questions. These are text passages that are semantically similar to the query.
{semantic_context}

## STRUCTURAL CONTEXT (Graph-Expanded)
Use this for relationship, chain-of-reasoning, or "who is similar to whom" questions.
These passages were found by traversing a knowledge graph to find structurally related entities.
{structural_context}

---
GUIDANCE:
- For "What is X?" questions: Prioritize SEMANTIC CONTEXT.
- For "How is X related to Y?" or "Who plays a similar role?" questions: Prioritize STRUCTURAL CONTEXT.
- If STRUCTURAL CONTEXT is empty, rely solely on SEMANTIC CONTEXT.
- Never invent information. If neither context answers the question, say "I don't have enough information."
---

Question: {query}
"""
    return prompt