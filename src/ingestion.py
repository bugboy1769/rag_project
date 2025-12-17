import os
import json
from src.models import get_llm_response

def extract_triplets(text_chunk):
    """
    Uses LLM to parse text into JSON triplets
    """
    prompt = f"""
    Analyze the following text and extract knowledge graph triplets.
    Return JSON ONLY in this format: 
    [
        {{"subject": "Entity1", "predicate": "relation", "object": "Entity2"}},
        ...
    ]
    
    Text: {text_chunk}
    """
    response = get_llm_response(prompt)
    
    # Clean markdown code fences which often cause parsing issues
    cleaned_response = response.replace("```json", "").replace("```", "")
    
    try:
        start = cleaned_response.find('[')
        end = cleaned_response.rfind(']') + 1
        if start == -1 or end == 0:
            return []
            
        json_str = cleaned_response[start:end]
        triplets = json.loads(json_str)
        
        # Validation and Type-Safety
        valid_triplets = []
        if isinstance(triplets, list):
            for t in triplets:
                if isinstance(t, dict) and 'subject' in t and 'predicate' in t and 'object' in t:
                    # FORCE STRING to avoid 'unhashable type: dict' error
                    # if the LLM returned nested objects like {"subject": {"name": "Cat"}}
                    valid_triplets.append({
                        "subject": str(t['subject']),
                        "predicate": str(t['predicate']),
                        "object": str(t['object'])
                    })
        return valid_triplets

    except Exception as e:
        print(f"Failed to parse triplets: {e}")
        return []
    
def load_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    
    with open(file_path, 'r') as f:
        lines=[line.strip() for line in f.readlines() if line.strip()]
    
    return lines
