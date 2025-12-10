"""
LMP (Language Model Prompting) formatting for evidence.

Formats selected factlets in hierarchical structure:
- Group by entity
- Topological order for aggregates
- Clear, readable presentation for LLM reasoning
"""

from typing import List, Dict, Set
from collections import defaultdict
import re

from .scoring import Factlet
from .utils import logger


class LMPFormatter:
    """
    Format evidence in LMP style for optimal LLM reasoning.
    
    Implements hierarchical grouping and topological ordering
    as described in LMP and SubgraphRAG papers.
    """
    
    def __init__(self, group_by_entity: bool = True, topological_order: bool = True):
        self.group_by_entity = group_by_entity
        self.topological_order = topological_order
    
    def format(self, factlets: List[Factlet]) -> str:
        """
        Format factlets as structured evidence.
        
        Args:
            factlets: Selected factlets to format
            
        Returns:
            Formatted evidence string
        """
        if not factlets:
            return "No evidence found."
        
        if self.group_by_entity:
            return self._format_grouped(factlets)
        else:
            return self._format_flat(factlets)
    
    def _format_flat(self, factlets: List[Factlet]) -> str:
        """Simple flat formatting."""
        lines = ["Evidence:", ""]
        for i, fact in enumerate(factlets, 1):
            lines.append(f"{i}. {fact.text}")
        return "\n".join(lines)
    
    def _format_grouped(self, factlets: List[Factlet]) -> str:
        """
        Hierarchical formatting grouped by entity.
        
        Groups facts by their subject entity for better organization.
        """
        # Extract entity -> facts mapping
        entity_facts = defaultdict(list)
        
        for fact in factlets:
            # Extract subject entity (first capitalized phrase)
            entities = self._extract_entities(fact.text)
            if entities:
                entity = entities[0]
                entity_facts[entity].append(fact)
            else:
                entity_facts["Other"].append(fact)
        
        # Format hierarchically
        lines = ["Evidence:", ""]
        
        # Sort entities by number of facts (descending)
        sorted_entities = sorted(
            entity_facts.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for entity, facts in sorted_entities:
            lines.append(f"### {entity}")
            for fact in facts:
                lines.append(f"  - {fact.text}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entity mentions from text.
        
        Simple heuristic: capitalized words/phrases.
        """
        # Find sequences of capitalized words
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(pattern, text)
        return entities
    
    def format_with_paths(
        self, 
        factlets: List[Factlet], 
        include_paths: bool = False
    ) -> str:
        """
        Format with optional KG path provenance.
        
        Args:
            factlets: Factlets to format
            include_paths: If True, include source KG paths
            
        Returns:
            Formatted evidence with optional paths
        """
        if not include_paths:
            return self.format(factlets)
        
        lines = ["Evidence:", ""]
        
        for i, fact in enumerate(factlets, 1):
            lines.append(f"{i}. {fact.text}")
            if fact.source_path and fact.source_path != fact.text:
                lines.append(f"   Source: {fact.source_path}")
        
        return "\n".join(lines)


def test_formatting():
    """Test LMP formatting."""
    from .scoring import Factlet
    
    # Create test factlets
    factlets = [
        Factlet(
            text="Barack Obama place of birth Honolulu",
            source_path="Barack Obama -> place_of_birth -> Honolulu",
            tokens=10
        ),
        Factlet(
            text="Barack Obama profession politician",
            source_path="Barack Obama -> profession -> politician",
            tokens=8
        ),
        Factlet(
            text="Honolulu location state Hawaii",
            source_path="Honolulu -> location -> Hawaii",
            tokens=8
        ),
        Factlet(
            text="Barack Obama nationality American",
            source_path="Barack Obama -> nationality -> American",
            tokens=8
        ),
    ]
    
    # Test different formats
    formatter = LMPFormatter(group_by_entity=True)
    
    print("Grouped format:")
    print(formatter.format(factlets))
    print("\n" + "="*50 + "\n")
    
    print("With paths:")
    print(formatter.format_with_paths(factlets, include_paths=True))
    print("\n" + "="*50 + "\n")
    
    formatter_flat = LMPFormatter(group_by_entity=False)
    print("Flat format:")
    print(formatter_flat.format(factlets))


if __name__ == "__main__":
    test_formatting()
