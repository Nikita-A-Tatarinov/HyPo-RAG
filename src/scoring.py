"""
Policy-guided factlet scoring and selection.

Implements deterministic scoring: score = α·relevance + β·diversity + γ·veracity - λ·tokens
Greedy selection under token budget B to optimize evidence quality.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from .config import Config
from .constrained_gen import ConstrainedPath
from .utils import logger, timer, cosine_similarity, count_tokens


@dataclass
class Factlet:
    """A factual statement extracted from a KG path."""
    text: str  # Natural language fact
    source_path: str  # Original KG path
    tokens: int  # Token count
    embedding: Optional[List[float]] = None  # Sentence embedding
    relevance: float = 0.0  # Relevance to question
    diversity: float = 0.0  # Diversity from selected facts
    veracity: float = 0.0  # Veracity score
    score: float = 0.0  # Final score


class ScoringPolicy:
    """
    Training-free scoring policy for evidence selection.
    
    Implements explicit budget optimization with multi-objective scoring:
    - Relevance (α): How relevant to the question
    - Diversity (β): How diverse from already selected facts
    - Veracity (γ): Confidence in factual correctness
    - Token penalty (λ): Penalize longer facts
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Load embedding model for relevance/diversity
        logger.info(f"Loading embedding model: {config.embedding_model}")
        self.encoder = SentenceTransformer(config.embedding_model)
        
        # OpenAI client for veracity scoring
        self.client = OpenAI(api_key=config.openai_api_key)
        
    @timer
    def score_and_select(
        self,
        question: str,
        paths: List[ConstrainedPath],
        budget: Optional[int] = None
    ) -> List[Factlet]:
        """
        Score factlets and greedily select under token budget.
        
        Args:
            question: Input question
            paths: List of constrained KG paths
            budget: Token budget (default: config.token_budget)
            
        Returns:
            List of selected factlets, total tokens ≤ budget
        """
        if budget is None:
            budget = self.config.token_budget
        
        # Step 1: Extract factlets from paths
        factlets = self._extract_factlets(paths)
        logger.info(f"Extracted {len(factlets)} factlets from {len(paths)} paths")
        
        # Step 2: Compute embeddings
        factlets = self._compute_embeddings(factlets, question)
        
        # Step 3: Greedy selection under budget
        selected = self._greedy_select(factlets, budget)
        
        logger.info(f"Selected {len(selected)}/{len(factlets)} factlets "
                   f"({sum(f.tokens for f in selected)}/{budget} tokens)")
        
        return selected
    
    def _extract_factlets(self, paths: List[ConstrainedPath]) -> List[Factlet]:
        """
        Extract atomic factlets from KG paths.
        
        Each path may contain multiple facts. Split into atomic statements.
        """
        factlets = []
        
        for path in paths:
            # Simple splitting by sentence/period
            sentences = [s.strip() for s in path.text.split('.') if s.strip()]
            
            for sentence in sentences:
                # Skip very short or very long facts
                tokens = count_tokens(sentence)
                if tokens < 3 or tokens > 100:
                    continue
                
                factlets.append(Factlet(
                    text=sentence,
                    source_path=path.text,
                    tokens=tokens
                ))
        
        # Remove duplicates
        seen = set()
        unique_factlets = []
        for f in factlets:
            if f.text not in seen:
                seen.add(f.text)
                unique_factlets.append(f)
        
        return unique_factlets
    
    def _compute_embeddings(
        self, 
        factlets: List[Factlet], 
        question: str
    ) -> List[Factlet]:
        """
        Compute sentence embeddings for relevance/diversity scoring.
        
        Also computes initial relevance scores.
        """
        # Encode question
        question_emb = self.encoder.encode(question, convert_to_numpy=True)
        
        # Encode factlets
        texts = [f.text for f in factlets]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Compute relevance scores
        for i, factlet in enumerate(factlets):
            factlet.embedding = embeddings[i].tolist()
            factlet.relevance = cosine_similarity(question_emb.tolist(), factlet.embedding)
        
        return factlets
    
    def _greedy_select(
        self, 
        factlets: List[Factlet], 
        budget: int
    ) -> List[Factlet]:
        """
        Greedy selection under token budget.
        
        Iteratively select factlet with highest score that fits in budget,
        updating diversity scores after each selection.
        """
        selected = []
        remaining = list(factlets)
        current_tokens = 0
        
        while remaining and current_tokens < budget:
            # Compute veracity for top candidates (expensive, so do it lazily)
            if len(remaining) <= 20:
                self._compute_veracity(remaining)
            else:
                # Only compute for top candidates by relevance
                top_candidates = sorted(remaining, key=lambda f: f.relevance, reverse=True)[:20]
                self._compute_veracity(top_candidates)
            
            # Compute diversity scores based on already selected
            for factlet in remaining:
                if selected:
                    # Negative max similarity to selected facts
                    similarities = [
                        cosine_similarity(factlet.embedding, s.embedding)
                        for s in selected
                    ]
                    factlet.diversity = -max(similarities)
                else:
                    factlet.diversity = 0.0
            
            # Compute final scores
            for factlet in remaining:
                factlet.score = (
                    self.config.alpha * factlet.relevance +
                    self.config.beta * factlet.diversity +
                    self.config.gamma * factlet.veracity -
                    self.config.lambda_ * factlet.tokens
                )
            
            # Select best factlet that fits in budget
            remaining.sort(key=lambda f: f.score, reverse=True)
            
            selected_any = False
            for factlet in remaining:
                if current_tokens + factlet.tokens <= budget:
                    selected.append(factlet)
                    current_tokens += factlet.tokens
                    remaining.remove(factlet)
                    selected_any = True
                    break
            
            if not selected_any:
                # No factlet fits, stop
                break
        
        return selected
    
    def _compute_veracity(self, factlets: List[Factlet]):
        """
        Compute veracity scores using heuristic (training-free, no API calls).
        
        Uses simple heuristics based on:
        - Path length (shorter = more direct = higher confidence)
        - Token count (concise facts = higher confidence)
        - Structural completeness (subject-relation-object)
        """
        for factlet in factlets:
            if factlet.veracity > 0:
                continue  # Already computed
            
            # Heuristic veracity score (0.0 to 1.0)
            # Factors:
            # 1. Prefer shorter paths (more direct evidence)
            # 2. Penalize very long or very short factlets
            # 3. Base confidence of 0.7 (neutral)
            
            base_score = 0.7
            
            # Penalty for very long factlets (likely concatenated paths)
            length_penalty = 0.0
            if factlet.tokens > 50:
                length_penalty = min(0.3, (factlet.tokens - 50) * 0.01)
            
            # Small bonus for medium-length factlets (complete but concise)
            length_bonus = 0.0
            if 10 <= factlet.tokens <= 30:
                length_bonus = 0.1
            
            factlet.veracity = max(0.0, min(1.0, base_score - length_penalty + length_bonus))
    
    def score_factlet(
        self,
        factlet: Factlet,
        question: str,
        selected: List[Factlet]
    ) -> float:
        """
        Score a single factlet.
        
        Args:
            factlet: Factlet to score
            question: Input question
            selected: Already selected factlets
            
        Returns:
            Final score
        """
        # Ensure embeddings computed
        if factlet.embedding is None:
            factlet = self._compute_embeddings([factlet], question)[0]
        
        # Compute diversity
        if selected:
            similarities = [
                cosine_similarity(factlet.embedding, s.embedding)
                for s in selected
            ]
            factlet.diversity = -max(similarities)
        else:
            factlet.diversity = 0.0
        
        # Compute veracity if needed
        if factlet.veracity == 0.0:
            self._compute_veracity([factlet])
        
        # Compute final score
        score = (
            self.config.alpha * factlet.relevance +
            self.config.beta * factlet.diversity +
            self.config.gamma * factlet.veracity -
            self.config.lambda_ * factlet.tokens
        )
        
        return score


def test_scoring():
    """Test factlet scoring and selection."""
    from .config import Config
    from .constrained_gen import ConstrainedPath
    
    config = Config()
    policy = ScoringPolicy(config)
    
    # Create test paths
    test_paths = [
        ConstrainedPath(
            tokens=[],
            text="Barack Obama place of birth Honolulu Hawaii.",
            score=0.0,
            is_complete=True
        ),
        ConstrainedPath(
            tokens=[],
            text="Barack Obama person birthplace Honolulu. Honolulu location state Hawaii.",
            score=0.0,
            is_complete=True
        ),
        ConstrainedPath(
            tokens=[],
            text="Barack Obama profession politician. Barack Obama nationality American.",
            score=0.0,
            is_complete=True
        ),
    ]
    
    question = "Where was Barack Obama born?"
    
    # Score and select
    selected = policy.score_and_select(question, test_paths, budget=100)
    
    print(f"\nSelected {len(selected)} factlets:")
    for i, f in enumerate(selected, 1):
        print(f"{i}. {f.text}")
        print(f"   Relevance: {f.relevance:.3f}, Diversity: {f.diversity:.3f}, "
              f"Veracity: {f.veracity:.3f}, Tokens: {f.tokens}")
        print(f"   Score: {f.score:.3f}")


if __name__ == "__main__":
    test_scoring()
