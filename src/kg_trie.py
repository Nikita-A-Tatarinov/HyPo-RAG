"""
KG-Trie construction for graph-constrained decoding.

Implements BFS retrieval, path formatting, tokenization, and MARISA Trie storage
to enable 100% faithful reasoning by constraining LLM generation to valid KG paths.

Following GCR paper approach: Uses per-question local subgraphs from RoG datasets.
Each question has a pre-extracted subgraph (list of triples) within 2 hops.
Target performance: 0.28s per entity (avg), O(1) prefix validation.
"""

from typing import List, Tuple, Set, Dict, Optional
from collections import deque, defaultdict
import marisa_trie
import json
import pickle
from pathlib import Path
import time
from transformers import AutoTokenizer
from tqdm import tqdm

from .config import Config
from .utils import logger, timer, format_path_as_sentence


class KGTrie:
    """
    KG-Trie index for graph-constrained decoding.
    
    Stores all valid L-hop paths from entities in a trie structure,
    enabling O(1) prefix validation during beam search.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.trie: Optional[marisa_trie.Trie] = None
        self.tokenizer = None
        self.entity_paths: Dict[str, List[List[int]]] = {}  # entity -> token sequences
        
        # Initialize tokenizer once (using public tokenizer for trie)
        tokenizer_name = getattr(self.config, 'trie_tokenizer', 'gpt2')
        logger.info(f"Loading tokenizer for KG-Trie: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            token=self.config.hf_token if self.config.hf_token else None
        )
    
    @timer
    def build_trie_from_triples(
        self, 
        triples: List[List[str]], 
        question_entities: List[str]
    ):
        """
        Build KG-Trie from a list of triples (per-question subgraph).
        
        This is the main method for per-question trie construction following GCR approach.
        Each question has a pre-extracted local subgraph from the RoG datasets.
        
        Pipeline:
        1. Convert triples to adjacency list
        2. BFS retrieval: Get L-hop paths from question entities
        3. Path formatting: Convert triples to natural language sentences
        4. Tokenization: Tokenize sentences with Llama-3-8B tokenizer
        5. Trie construction: Store token sequences in MARISA Trie
        
        Args:
            triples: List of [subject, relation, object] triples from dataset
            question_entities: List of question entity names to start BFS from
        """
        start_time = time.time()
        
        # Step 1: Convert triples to adjacency list
        kg_data = self._triples_to_adjacency_list(triples)
        
        # Collect all token sequences
        all_sequences = []
        
        # Step 2-5: BFS retrieval and tokenization for each question entity
        for entity in question_entities:
            # BFS retrieval (L-hop paths) on local graph
            paths = self._bfs_retrieve_from_adjacency(entity, kg_data)
            
            # Format and tokenize paths
            token_sequences = []
            for path in paths:
                # Format as natural language sentence
                sentence = format_path_as_sentence(path)
                
                # Tokenize
                tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
                token_sequences.append(tokens)
                
                # Convert to string for trie storage (space-separated token IDs)
                all_sequences.append(" ".join(map(str, tokens)))
            
            self.entity_paths[entity] = token_sequences
        
        # Build MARISA Trie
        if all_sequences:
            self.trie = marisa_trie.Trie(all_sequences)
            elapsed = time.time() - start_time
            logger.debug(f"KG-Trie built: {len(all_sequences)} paths from {len(question_entities)} entities in {elapsed:.3f}s")
        else:
            # Empty trie if no paths found
            self.trie = marisa_trie.Trie([])
            logger.warning("No valid paths found, created empty trie")
    
    def _triples_to_adjacency_list(self, triples: List[List[str]]) -> Dict:
        """
        Convert list of triples to adjacency list representation.
        
        Args:
            triples: List of [subject, relation, object] triples
            
        Returns:
            Adjacency list: {entity_name: {relation: [target_entities]}}
        """
        adjacency_list = defaultdict(lambda: defaultdict(list))
        
        for triple in triples:
            if len(triple) != 3:
                continue
            
            subj, rel, obj = triple
            
            # Add forward edge
            adjacency_list[subj][rel].append(obj)
        
        return dict(adjacency_list)
    
    def _bfs_retrieve_from_adjacency(
        self, 
        start_entity: str, 
        adjacency_list: Dict
    ) -> List[List[Tuple[str, str, str]]]:
        """
        BFS retrieval of L-hop paths from start entity using adjacency list.
        
        Following GCR approach: BFS on local graph within max_hops.
        
        Args:
            start_entity: Starting entity name (as appears in triples)
            adjacency_list: Adjacency list {entity: {relation: [targets]}}
            
        Returns:
            List of paths, where each path is a list of (subj, rel, obj) triples
        """
        max_hops = self.config.max_hops
        paths = []
        
        # Check if entity exists in graph
        if start_entity not in adjacency_list:
            logger.debug(f"Entity '{start_entity}' not found in subgraph")
            return []
        
        # BFS queue: (current_entity, path_so_far, hops)
        queue = deque([(start_entity, [], 0)])
        visited = {start_entity}
        
        while queue:
            current, path, hops = queue.popleft()
            
            if hops >= max_hops:
                if path:  # Don't add empty paths
                    paths.append(path)
                continue
            
            # Get neighbors from adjacency list
            if current not in adjacency_list:
                # No outgoing edges, add path if non-empty
                if path:
                    paths.append(path)
                continue
            
            relations = adjacency_list[current]
            
            # Iterate through all relations and targets
            for relation, targets in relations.items():
                for target in targets:
                    # Create triple
                    new_triple = (current, relation, target)
                    new_path = path + [new_triple]
                    
                    # Add to paths
                    paths.append(new_path)
                    
                    # Continue BFS if not beyond max_hops and not visited
                    # Use <= so that paths of length `max_hops` are explored.
                    if hops + 1 <= max_hops and target not in visited:
                        queue.append((target, new_path, hops + 1))
                        visited.add(target)
        
        return paths
    
    def validate_path(self, token_sequence: List[int]) -> bool:
        """
        Validate if token sequence is a valid prefix in KG-Trie.
        
        Args:
            token_sequence: List of token IDs
            
        Returns:
            True if valid prefix, False otherwise
            
        Complexity: O(1) with MARISA Trie
        """
        if self.trie is None:
            raise ValueError("Trie not built yet. Call build_trie() first.")
        
        # Convert token sequence to string
        key = " ".join(map(str, token_sequence))
        
        # Check if prefix exists in trie
        return len(self.trie.keys(key)) > 0
    
    def get_valid_next_tokens(self, prefix: List[int]) -> Set[int]:
        """
        Get all valid next tokens for a given prefix.
        
        Args:
            prefix: Current token sequence
            
        Returns:
            Set of valid next token IDs
        """
        if self.trie is None:
            raise ValueError("Trie not built yet. Call build_trie() first.")
        
        prefix_str = " ".join(map(str, prefix)) + " " if prefix else ""
        
        # Find all keys with this prefix
        matching_keys = self.trie.keys(prefix_str)
        
        # Extract next tokens
        next_tokens = set()
        for key in matching_keys:
            tokens = key.split()
            if len(tokens) > len(prefix):
                next_token = int(tokens[len(prefix)])
                next_tokens.add(next_token)
        
        return next_tokens
    


if __name__ == "__main__":
    # Test KG-Trie construction with sample triples
    from .config import Config
    
    config = Config()
    
    # Sample triples (from RoG-WebQSP format)
    test_triples = [
        ["Justin Bieber", "people.person.sibling_s", "m.0gxnnwj"],
        ["m.0gxnnwj", "people.sibling_relationship.sibling", "Jaxon Bieber"],
        ["Justin Bieber", "award.award_winner.awards_won", "m.0yrkc0l"],
    ]
    
    test_entities = ["Justin Bieber"]
    
    kg_trie = KGTrie(config)
    kg_trie.build_trie_from_triples(test_triples, test_entities)
    
    # Test validation
    if kg_trie.tokenizer and kg_trie.trie:
        test_text = "Justin Bieber people.person.sibling_s m.0gxnnwj"
        tokens = kg_trie.tokenizer.encode(test_text, add_special_tokens=False)
        is_valid = kg_trie.validate_path(tokens[:3])  # Test prefix
        print(f"Test path valid: {is_valid}")
        print(f"Trie contains {len(kg_trie.trie)} paths")
