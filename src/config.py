"""
Configuration management for HyPo-RAG.
All hyperparameters and API settings in one place.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class Config:
    """HyPo-RAG configuration with all hyperparameters."""
    
    # === API Configuration ===
    # Read from environment variables (export OPENAI_API_KEY=... before running)
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    
    # === Model Configuration (All Frozen) ===
    # Hypothesis generation model
    hypothesis_model: str = "gpt-4o-mini"
    hypothesis_temperature: float = 0.0
    hypothesis_max_tokens: int = 2048
    
    # Path generation model (graph-constrained)
    path_model: str = "gpt-3.5-turbo"  # Using OpenAI for path generation
    path_temperature: float = 0.0
    path_max_tokens: int = 512
    
    # Tokenizer for KG-Trie (using public GPT-2 tokenizer)
    trie_tokenizer: str = "gpt2"  # Public tokenizer for path tokenization
    
    # Final reasoning model
    reasoning_model: str = "gpt-4o-mini"  # or "gpt-4"
    reasoning_temperature: float = 0.0
    reasoning_max_tokens: int = 2048
    
    # Veracity scoring model (for γ component)
    veracity_model: str = "gpt-4o-mini"
    
    # Embedding model for relevance/diversity scoring
    embedding_model: str = "all-mpnet-base-v2"
    
    # === Dataset Configuration ===
    # RoG datasets with per-question local subgraphs (following GCR Appendix C)
    dataset_name: str = "rmanluo/RoG-webqsp"  # or "rmanluo/RoG-cwq"
    
    # === KG-Trie Configuration ===
    max_hops: int = 3  # L=3 for BFS retrieval (increased from 2 to capture more paths)
    # Note: No global KG file needed - each question has its own subgraph in dataset
    
    # === Hypothesis-Guided Planning (HGP) ===
    num_hypotheses: int = 5  # N=3-5 entity/relation hypotheses
    num_subquestions: int = 3  # M=2-4 sub-questions
    enable_backward_reasoning: bool = True  # ORT-style
    
    # === Graph-Constrained Path Retrieval ===
    beam_size: int = 10  # K=10 faithful paths
    max_path_length: int = 4  # Maximum reasoning steps
    
    # === Scoring Policy (Training-Free) ===
    # Hyperparameters for score = α·rel + β·div + γ·ver - λ·tok
    alpha: float = 0.4  # Relevance weight
    beta: float = 0.3   # Diversity weight
    gamma: float = 0.2  # Veracity weight
    lambda_: float = 0.002  # Token penalty
    
    token_budget: int = 500  # B=500 token budget
    
    # === LMP Formatting ===
    group_by_entity: bool = True  # Hierarchical grouping
    topological_order: bool = True  # For aggregates
    
    # === Evaluation ===
    results_dir: str = "./results"
    split: str = "test"  # "train", "validation", or "test"
    
    # === Parallelization ===
    num_gpus: int = 4  # H200 GPUs available
    batch_size: int = 1  # Process questions sequentially for now
    
    # === Validation Tuning ===
    validation_size: int = 500
    # Grid search ranges
    alpha_range: list = field(default_factory=lambda: [0.3, 0.4, 0.5])
    beta_range: list = field(default_factory=lambda: [0.2, 0.3, 0.4])
    gamma_range: list = field(default_factory=lambda: [0.2, 0.3, 0.4])
    lambda_range: list = field(default_factory=lambda: [0.001, 0.002, 0.005])
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.openai_api_key and "gpt" in self.hypothesis_model.lower():
            raise ValueError("OPENAI_API_KEY not set for GPT models")
        
        # Create results directory
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            "hypothesis_model": self.hypothesis_model,
            "path_model": self.path_model,
            "reasoning_model": self.reasoning_model,
            "max_hops": self.max_hops,
            "beam_size": self.beam_size,
            "num_hypotheses": self.num_hypotheses,
            "num_subquestions": self.num_subquestions,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "lambda": self.lambda_,
            "token_budget": self.token_budget,
        }
