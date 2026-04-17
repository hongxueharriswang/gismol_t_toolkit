"""Response validation against COH constraints"""

from typing import List, Dict, Any
from gismol.core.coh_object import COHObject, ConstraintViolation


class ResponseValidator:
    """Validates natural language responses against COH constraints"""
    
    def validate(self, response: str, objects: List[COHObject]) -> List[Dict[str, Any]]:
        """Check response against relevant constraints"""
        results = []
        for obj in objects:
            # Check if response contains any attribute values that violate constraints
            for constraint in obj.identity_constraints:
                if constraint.specification in response:
                    # Simplified: assume valid if object's context satisfies constraint
                    context = obj.get_context()
                    if not obj.constraint_system.validate_single(constraint, context):
                        results.append({
                            'valid': False,
                            'message': f"Response conflicts with constraint '{constraint.name}'",
                            'suggestions': ["Adjust value to satisfy constraint"]
                        })
        if not results:
            results.append({'valid': True, 'message': 'All constraints satisfied', 'suggestions': []})
        return results