"""Parse natural language constraints"""

from typing import Dict, List
from gismol.core.constraints import Constraint


class ConstraintParser:
    """Parse constraints from natural language"""
    
    def parse(self, text: str) -> Dict:
        """Parse text into structured constraint representation"""
        # Simple: detect comparison patterns
        result = {'original': text}
        for op in ['<', '>', '<=', '>=', '==']:
            if op in text:
                parts = text.split(op)
                if len(parts) == 2:
                    result['left'] = parts[0].strip()
                    result['operator'] = op
                    result['right'] = parts[1].strip()
        return result
    
    def to_coh_constraints(self, parsed: Dict) -> List[Constraint]:
        """Convert parsed representation to COH constraints"""
        if 'left' in parsed:
            spec = f"{parsed['left']} {parsed['operator']} {parsed['right']}"
            constraint = Constraint(
                name="parsed_constraint",
                specification=spec,
                category='attribute'
            )
            return [constraint]
        return []