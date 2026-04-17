"""Convert text documents to COHObjects"""

from typing import Dict, Any
from gismol.core import COHRepository, COHObject


class Text2COH:
    """Convert text documents into COHObjects with hierarchical integration"""
    
    def __init__(self, repository: COHRepository):
        self.repository = repository
    
    def process_knowledge_source(self, document: str) -> Dict[str, Any]:
        """Process document and create COHObjects"""
        # Simple: create one object per sentence
        sentences = document.split('.')
        new_objects = 0
        for sent in sentences:
            if sent.strip():
                obj = COHObject(name=sent[:50])
                self.repository.add_object(obj)
                new_objects += 1
        return {
            'new_objects': new_objects,
            'integrated': new_objects
        }