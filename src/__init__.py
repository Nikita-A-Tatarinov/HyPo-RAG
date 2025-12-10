"""
HyPo-RAG: Hypothesis-Guided Planning with Graph-Constrained Retrieval
Training-free framework for faithful knowledge graph question answering.
"""

__version__ = "0.1.0"

from .config import Config
from .inference import run_hyporag, answer_question

__all__ = ["Config", "run_hyporag", "answer_question"]
