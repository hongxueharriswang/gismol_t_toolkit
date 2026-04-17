"""Intent recognition from natural language"""

import re
from typing import Dict, Any


class IntentRecognizer:
    """Recognizes user intent in natural language queries"""
    
    def __init__(self):
        self.patterns = {
            'attribute_query': re.compile(r'what is (?:the )?(\w+) (?:of|for) (\w+)', re.IGNORECASE),
            'list_objects': re.compile(r'list (?:all )?objects', re.IGNORECASE),
            'constraint_check': re.compile(r'check (?:constraint )?(\w+)', re.IGNORECASE)
        }
    
    def recognize_intent(self, query: str) -> Dict[str, Any]:
        """Return intent and extracted parameters"""
        for intent, pattern in self.patterns.items():
            match = pattern.search(query)
            if match:
                return {
                    'intent': intent,
                    'parameters': match.groups()
                }
        return {'intent': 'unknown', 'parameters': ()}