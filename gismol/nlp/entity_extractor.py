"""Entity and relation extraction from text"""

from typing import List, Dict, Optional
from gismol.core.coh_object import COHObject
from gismol.core.repository import COHRepository


class EntityRelationExtractor:
    """Joint entity and relation extraction with COH integration"""
    
    def extract_to_coh(self, text: str, repository: COHRepository) -> List[COHObject]:
        """Extract entities and create COHObjects"""
        # Simple placeholder: split by spaces and create objects for capitalized words
        words = text.split()
        objects = []
        for word in words:
            if word[0].isupper():
                obj = COHObject(name=word)
                repository.add_object(obj)
                objects.append(obj)
        return objects