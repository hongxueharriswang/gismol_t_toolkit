"""Domain-specific reasoners"""

from typing import Dict
from .base import BaseReasoner
from gismol.core.constraints import Constraint

class BiologicalReasoner(BaseReasoner, reasoner_type="biological"):
    """Handles biological and biomedical constraints"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification.lower()
        if 'cfu/ml' in spec or 'concentration' in spec:
            # Biological concentration validation
            return super().evaluate(constraint, context)
        return super().evaluate(constraint, context)


class PhysicalReasoner(BaseReasoner, reasoner_type="physical"):
    """Validates physics-based constraints with tolerance"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        if 'with tolerance' in spec:
            # Parse tolerance
            main_part = spec.split('with tolerance')[0].strip()
            tolerance = float(spec.split('with tolerance')[1].strip())
            # Evaluate main equality
            eq_constraint = type(constraint)()
            eq_constraint.specification = main_part
            result = super().evaluate(eq_constraint, context)
            if isinstance(result, bool):
                return result
            # If result is numeric, check within tolerance
            if isinstance(result, (int, float)):
                target = float(main_part.split('=')[1].strip())
                return abs(result - target) <= tolerance
        return super().evaluate(constraint, context)


class GeometricReasoner(BaseReasoner, reasoner_type="geometric"):
    """Handles spatial relationships and geometric constraints"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        if 'distance_to' in spec:
            # Parse: "robot distance_to human > 1.0m"
            parts = spec.split('distance_to')
            if len(parts) == 2:
                obj1 = parts[0].strip()
                rest = parts[1].strip()
                # Extract operator and threshold
                for op in ['>', '<', '>=', '<=']:
                    if op in rest:
                        op_parts = rest.split(op)
                        obj2 = op_parts[0].strip()
                        threshold = float(op_parts[1].replace('m', '').strip())
                        # Get actual distance from context
                        distance = context.get('distance', None)
                        if distance is not None:
                            if op == '>':
                                return distance > threshold
                            elif op == '<':
                                return distance < threshold
                            elif op == '>=':
                                return distance >= threshold
                            elif op == '<=':
                                return distance <= threshold
        return super().evaluate(constraint, context)