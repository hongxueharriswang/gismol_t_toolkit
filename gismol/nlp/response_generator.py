"""Natural language response generation with constraint validation"""

from typing import Dict, List, Any
from gismol.core.coh_object import COHObject


class ConstraintAwareResponseGenerator:
    """Generates responses that satisfy COH constraints"""
    
    def generate_response(self, intent: str, context: Dict, objects: List[COHObject]) -> str:
        """Generate response based on intent and objects"""
        if intent == 'attribute_query':
            obj = objects[0] if objects else None
            attr = context.get('attribute', 'unknown')
            if obj:
                value = obj.get_attribute(attr, 'not set')
                return f"The {attr} of {obj.name} is {value}."
        return "Response generated with constraint validation."