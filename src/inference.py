"""
End-to-end HyPo-RAG inference pipeline.

Integrates all components:
1. Hypothesis-Guided Planning (HGP)
2. Graph-Constrained Path Retrieval
3. Policy-Guided Factlet Selection
4. LMP Formatting
5. Final Reasoning with Frozen LLM
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
from openai import OpenAI

from .config import Config
from .kg_trie import KGTrie
from .hypothesis import HypothesisGenerator, HGP
from .constrained_gen import ConstrainedPathGenerator
from .scoring import ScoringPolicy, Factlet
from .formatting import LMPFormatter
from .utils import logger, timer, count_tokens


@dataclass
class HyPoRAGResult:
    """Result from HyPo-RAG inference."""
    question: str
    answer: str
    hgp: HGP
    paths: List[str]  # Retrieved paths
    factlets: List[Factlet]  # Selected evidence
    evidence: str  # Formatted evidence
    metadata: Dict[str, Any]  # Timing, token counts, etc.


class HyPoRAG:
    """
    HyPo-RAG: Hypothesis-Guided Planning with Graph-Constrained Retrieval.
    
    Training-free framework for faithful KG question answering.
    Works with per-question subgraphs from RoG datasets.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components (KG-Trie will be built per question)
        logger.info("Initializing HyPo-RAG components...")
        self.hypothesis_generator = HypothesisGenerator(config)
        self.scoring_policy = ScoringPolicy(config)
        self.formatter = LMPFormatter(
            group_by_entity=config.group_by_entity,
            topological_order=config.topological_order
        )
        
        # OpenAI client for final reasoning
        self.client = OpenAI(api_key=config.openai_api_key)
        
        logger.info("HyPo-RAG initialized successfully")
    
    @timer
    def run(
        self, 
        question: str, 
        graph_triples: List[List[str]], 
        question_entities: List[str]
    ) -> HyPoRAGResult:
        """
        Run HyPo-RAG on a single question with its local subgraph.
        
        Args:
            question: Input question
            graph_triples: Per-question subgraph as list of [subj, rel, obj] triples
            question_entities: Question entity names to start BFS from
            
        Returns:
            HyPoRAGResult with answer and metadata
        """
        start_time = time.time()
        metadata = {
            "timings": {},
            "api_calls": 0,
            "tokens": {},
            "graph_size": len(graph_triples),
        }
        
        # Step 0: Build KG-Trie for this question's subgraph
        logger.info(f"Question: {question}")
        logger.info(f"  Building KG-Trie from {len(graph_triples)} triples...")
        t0 = time.time()
        kg_trie = KGTrie(self.config)
        kg_trie.build_trie_from_triples(graph_triples, question_entities)
        metadata["timings"]["trie_building"] = time.time() - t0
        logger.info(f"  KG-Trie built in {metadata['timings']['trie_building']:.3f}s")
        
        # Step 1: Generate Hypothesis-Guided Plan
        t0 = time.time()
        hgp = self.hypothesis_generator.generate_hgp(question)
        metadata["timings"]["hgp_generation"] = time.time() - t0
        metadata["api_calls"] += 1
        logger.info(f"  Generated HGP: {len(hgp.hypotheses)} hypotheses")
        
        # Step 2: Graph-Constrained Path Retrieval
        t0 = time.time()
        path_generator = ConstrainedPathGenerator(self.config, kg_trie)
        paths = path_generator.generate_paths(
            question, 
            hgp, 
            beam_size=self.config.beam_size
        )
        metadata["timings"]["path_generation"] = time.time() - t0
        metadata["api_calls"] += 1 if path_generator.client else 0
        logger.info(f"  Retrieved {len(paths)} constrained paths")
        
        # Step 3: Policy-Guided Factlet Selection
        t0 = time.time()
        factlets = self.scoring_policy.score_and_select(
            question,
            paths,
            budget=self.config.token_budget
        )
        metadata["timings"]["factlet_selection"] = time.time() - t0
        # Note: Veracity now uses heuristics (no API calls), so we don't count it
        metadata["tokens"]["evidence"] = sum(f.tokens for f in factlets)
        logger.info(f"  Selected {len(factlets)} factlets ({metadata['tokens']['evidence']} tokens)")
        
        # Step 4: LMP Formatting
        t0 = time.time()
        evidence = self.formatter.format(factlets)
        metadata["timings"]["formatting"] = time.time() - t0
        
        # Step 5: Final Reasoning
        t0 = time.time()
        answer = self.answer_question(question, evidence)
        metadata["timings"]["final_reasoning"] = time.time() - t0
        metadata["api_calls"] += 1
        metadata["tokens"]["answer"] = count_tokens(answer)
        logger.info(f"  Answer: {answer}")
        
        # Compute total metrics
        metadata["timings"]["total"] = time.time() - start_time
        metadata["tokens"]["total"] = (
            metadata["tokens"]["evidence"] + 
            metadata["tokens"]["answer"]
        )
        
        logger.info(f"  Total: {metadata['timings']['total']:.2f}s, "
                   f"{metadata['api_calls']} API calls, "
                   f"{metadata['tokens']['total']} tokens")
        
        return HyPoRAGResult(
            question=question,
            answer=answer,
            hgp=hgp,
            paths=[p.text for p in paths],
            factlets=factlets,
            evidence=evidence,
            metadata=metadata
        )
    
    def answer_question(self, question: str, evidence: str) -> str:
        """
        Generate final answer using frozen reasoning LLM.
        
        Args:
            question: Input question
            evidence: Formatted evidence from KG
            
        Returns:
            Final answer
        """
        # Build prompt with evidence
        prompt = self._build_reasoning_prompt(question, evidence)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.reasoning_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at answering questions based on "
                            "knowledge graph evidence. Provide accurate, concise answers "
                            "grounded in the given evidence. If the evidence is insufficient, "
                            "say 'I cannot answer based on the given evidence.'"
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.reasoning_temperature,
                max_tokens=self.config.reasoning_max_tokens,
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            logger.error(f"Final reasoning failed: {e}")
            return "Error generating answer"
    
    def _build_reasoning_prompt(self, question: str, evidence: str) -> str:
        """Build prompt for final reasoning."""
        return f"""Answer the following question based on the provided knowledge graph evidence.

Question: {question}

{evidence}

Answer:"""


def run_hyporag(
    question: str, 
    graph_triples: List[List[str]],
    question_entities: List[str],
    config: Optional[Config] = None
) -> HyPoRAGResult:
    """
    Convenience function to run HyPo-RAG on a single question.
    
    Args:
        question: Input question
        graph_triples: Per-question subgraph as list of [subj, rel, obj] triples
        question_entities: Question entity names
        config: Configuration (default: Config())
        
    Returns:
        HyPoRAGResult
    """
    if config is None:
        config = Config()
    
    hyporag = HyPoRAG(config)
    return hyporag.run(question, graph_triples, question_entities)


def answer_question(
    question: str,
    graph_triples: List[List[str]],
    question_entities: List[str],
    config: Optional[Config] = None
) -> str:
    """
    Simple interface: question -> answer.
    
    Args:
        question: Input question
        graph_triples: Per-question subgraph
        question_entities: Question entity names
        config: Configuration
        
    Returns:
        Answer string
    """
    result = run_hyporag(question, graph_triples, question_entities, config)
    return result.answer


def test_pipeline():
    """Test end-to-end HyPo-RAG pipeline with sample data."""
    from .config import Config
    
    config = Config()
    
    # Sample triples (from RoG-WebQSP)
    test_triples = [
        ["Barack Obama", "people.person.place_of_birth", "Honolulu"],
        ["Barack Obama", "people.person.profession", "Politician"],
        ["Honolulu", "location.location.containedby", "Hawaii"],
    ]
    test_entities = ["Barack Obama"]
    
    # Run HyPo-RAG
    print("\nRunning HyPo-RAG...")
    hyporag = HyPoRAG(config)
    
    test_question = "Where was Barack Obama born?"
    
    print(f"\n{'='*60}")
    result = hyporag.run(test_question, test_triples, test_entities)
    
    print(f"Question: {result.question}")
    print(f"Answer: {result.answer}")
    print(f"\nEvidence ({len(result.factlets)} factlets):")
    print(result.evidence)
    print(f"\nMetadata:")
    print(f"  Time: {result.metadata['timings']['total']:.2f}s")
    print(f"  API calls: {result.metadata['api_calls']}")
    print(f"  Tokens: {result.metadata['tokens']['total']}")


if __name__ == "__main__":
    test_pipeline()
