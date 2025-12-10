"""
Utility functions for HyPo-RAG.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def timer(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


def count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens in text using provided tokenizer or simple heuristic."""
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    # Simple heuristic: ~4 chars per token (GPT-3/4 average)
    return len(text) // 4


def format_path_as_sentence(path: List[Tuple[str, str, str]]) -> str:
    """
    Format a KG path as a natural language sentence.
    
    Args:
        path: List of (subject, relation, object) triples
        
    Returns:
        Natural language sentence representation
    """
    if not path:
        return ""
    
    sentences = []
    for subj, rel, obj in path:
        # Clean entity names (remove namespace prefixes)
        subj_clean = subj.split("/")[-1].replace("_", " ")
        obj_clean = obj.split("/")[-1].replace("_", " ")
        rel_clean = rel.split("/")[-1].replace("_", " ").replace(".", " ")
        
        sentences.append(f"{subj_clean} {rel_clean} {obj_clean}")
    
    return ". ".join(sentences) + "."


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        response: LLM response text
        
    Returns:
        Parsed JSON dictionary or None if parsing fails
    """
    try:
        # Try direct parsing first
        return json.loads(response)
    except json.JSONDecodeError:
        # Try extracting from markdown code block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            json_str = response[start:end].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    logger.warning(f"Failed to parse JSON response: {response[:200]}")
    return None


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split items into batches of specified size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON file."""
    import json
    from pathlib import Path
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: str) -> Optional[Dict[str, Any]]:
    """Load results from JSON file."""
    import json
    from pathlib import Path
    
    if not Path(filepath).exists():
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)
