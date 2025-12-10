"""
Hypothesis-Guided Planning (HGP) generation.

Synthesizes HyKGE hypotheses + ORT backward reasoning + iQUEST sub-questions
to guide efficient KG path retrieval. Uses frozen GPT-4o-mini with 2-shot prompting.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from openai import OpenAI

from .config import Config
from .utils import logger, parse_json_response, timer


@dataclass
class Hypothesis:
    """Structured hypothesis for guiding KG retrieval."""
    entity: str  # Expected entity mention
    relation: str  # Expected relation type
    target_type: Optional[str] = None  # Expected answer type (ORT backward reasoning)
    attributes: Optional[List[str]] = None  # Expected attributes


@dataclass
class HGP:
    """Hypothesis-Guided Plan for a question."""
    question: str
    hypotheses: List[Hypothesis]  # N=3-5 hypotheses
    sub_questions: List[str]  # M=2-4 sub-questions
    target_types: List[str]  # Expected answer types (from ORT)


class HypothesisGenerator:
    """
    Generate hypothesis-guided plans using GPT-4o-mini.
    
    Synthesizes three paradigms:
    1. HyKGE: Entity/relation hypotheses
    2. ORT: Backward reasoning from target types
    3. iQUEST: Iterative sub-question decomposition
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        
        # 2-shot examples for HGP generation
        self.examples = [
            {
                "question": "What movies did Tom Hanks act in that were directed by Steven Spielberg?",
                "hgp": {
                    "hypotheses": [
                        {
                            "entity": "Tom Hanks",
                            "relation": "actor.film",
                            "target_type": "Film",
                            "attributes": ["director", "title"]
                        },
                        {
                            "entity": "Steven Spielberg",
                            "relation": "director.film",
                            "target_type": "Film",
                            "attributes": ["title", "release_date"]
                        },
                        {
                            "entity": "Tom Hanks",
                            "relation": "actor.film.film.director",
                            "target_type": "Person",
                            "attributes": ["name"]
                        }
                    ],
                    "sub_questions": [
                        "What movies did Tom Hanks act in?",
                        "What movies did Steven Spielberg direct?",
                        "Which of Tom Hanks' movies were directed by Steven Spielberg?"
                    ],
                    "target_types": ["Film", "Movie"]
                }
            },
            {
                "question": "Where was Barack Obama born?",
                "hgp": {
                    "hypotheses": [
                        {
                            "entity": "Barack Obama",
                            "relation": "place_of_birth",
                            "target_type": "Location",
                            "attributes": ["name", "type"]
                        },
                        {
                            "entity": "Barack Obama",
                            "relation": "person.birthplace",
                            "target_type": "City",
                            "attributes": ["name", "country"]
                        }
                    ],
                    "sub_questions": [
                        "Who is Barack Obama?",
                        "What is Barack Obama's birthplace?"
                    ],
                    "target_types": ["Location", "City", "Place"]
                }
            }
        ]
    
    @timer
    def generate_hgp(self, question: str) -> HGP:
        """
        Generate Hypothesis-Guided Plan for question.
        
        Args:
            question: Input question
            
        Returns:
            HGP object with hypotheses, sub-questions, and target types
        """
        # Build prompt with 2-shot examples
        prompt = self._build_prompt(question)
        
        # Call GPT-4o-mini
        try:
            response = self.client.chat.completions.create(
                model=self.config.hypothesis_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at analyzing knowledge graph questions "
                            "and generating hypothesis-guided plans. For each question, "
                            "generate:\n"
                            "1. Entity/relation hypotheses (3-5) for KG paths\n"
                            "2. Sub-questions (2-4) to decompose the problem\n"
                            "3. Target answer types for backward reasoning\n\n"
                            "Output as JSON with keys: hypotheses, sub_questions, target_types."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.hypothesis_temperature,
                max_tokens=self.config.hypothesis_max_tokens,
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            hgp_dict = parse_json_response(content)
            if hgp_dict is None:
                logger.warning(f"Failed to parse HGP response, using fallback")
                return self._fallback_hgp(question)
            
            # Convert to HGP object
            hypotheses = [
                Hypothesis(
                    entity=h.get("entity", ""),
                    relation=h.get("relation", ""),
                    target_type=h.get("target_type"),
                    attributes=h.get("attributes")
                )
                for h in hgp_dict.get("hypotheses", [])
            ]
            
            hgp = HGP(
                question=question,
                hypotheses=hypotheses[:self.config.num_hypotheses],
                sub_questions=hgp_dict.get("sub_questions", [])[:self.config.num_subquestions],
                target_types=hgp_dict.get("target_types", [])
            )
            
            logger.info(f"Generated HGP: {len(hgp.hypotheses)} hypotheses, "
                       f"{len(hgp.sub_questions)} sub-questions")
            
            return hgp
            
        except Exception as e:
            logger.error(f"HGP generation failed: {e}")
            return self._fallback_hgp(question)
    
    def _build_prompt(self, question: str) -> str:
        """Build 2-shot prompt for HGP generation."""
        prompt = "Generate a hypothesis-guided plan for the following question.\n\n"
        
        # Add examples
        for i, example in enumerate(self.examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {example['question']}\n"
            prompt += f"HGP: {self._format_hgp_json(example['hgp'])}\n\n"
        
        # Add target question
        prompt += f"Now generate an HGP for:\nQuestion: {question}\nHGP:"
        
        return prompt
    
    def _format_hgp_json(self, hgp: Dict[str, Any]) -> str:
        """Format HGP as JSON string."""
        import json
        return json.dumps(hgp, indent=2)
    
    def _fallback_hgp(self, question: str) -> HGP:
        """Generate simple fallback HGP when API fails."""
        # Extract potential entities (simple heuristic: capitalized words)
        words = question.split()
        entities = [w for w in words if w[0].isupper() and len(w) > 1]
        
        hypotheses = [
            Hypothesis(
                entity=entity,
                relation="?",
                target_type=None
            )
            for entity in entities[:self.config.num_hypotheses]
        ]
        
        return HGP(
            question=question,
            hypotheses=hypotheses,
            sub_questions=[question],
            target_types=[]
        )
    
    def hykge_hypotheses(self, question: str) -> List[Hypothesis]:
        """
        Generate HyKGE-style entity/relation hypotheses.
        
        Args:
            question: Input question
            
        Returns:
            List of entity/relation hypotheses
        """
        hgp = self.generate_hgp(question)
        return hgp.hypotheses
    
    def ort_backward_reasoning(self, question: str) -> List[str]:
        """
        Generate ORT-style target types via backward reasoning.
        
        Args:
            question: Input question
            
        Returns:
            List of expected answer types
        """
        hgp = self.generate_hgp(question)
        return hgp.target_types
    
    def iquest_subquestions(self, question: str) -> List[str]:
        """
        Generate iQUEST-style iterative sub-questions.
        
        Args:
            question: Input question
            
        Returns:
            List of sub-questions
        """
        hgp = self.generate_hgp(question)
        return hgp.sub_questions


def test_hypothesis_generation():
    """Test HGP generation on sample questions."""
    from .config import Config
    
    config = Config()
    generator = HypothesisGenerator(config)
    
    test_questions = [
        "Where was Barack Obama born?",
        "What movies did Tom Hanks act in?",
        "Who is the president of France?",
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        hgp = generator.generate_hgp(question)
        
        print(f"Hypotheses ({len(hgp.hypotheses)}):")
        for h in hgp.hypotheses:
            print(f"  - {h.entity} --[{h.relation}]--> {h.target_type}")
        
        print(f"Sub-questions ({len(hgp.sub_questions)}):")
        for sq in hgp.sub_questions:
            print(f"  - {sq}")
        
        print(f"Target types: {hgp.target_types}")


if __name__ == "__main__":
    test_hypothesis_generation()
