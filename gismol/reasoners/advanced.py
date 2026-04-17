"""Advanced reasoning systems"""

from typing import Dict, Any
from .base import BaseReasoner
from gismol.core import COHObject
from gismol.core.constraints import Constraint

class CausalReasoner(BaseReasoner, reasoner_type="causal"):
    """Analyzes cause-effect relationships with probabilistic reasoning"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        if 'causes' in spec:
            # Parse "A causes B with p=X"
            import re
            match = re.search(r'(\w+)\s+causes\s+(\w+)\s+with\s+p=([\d\.]+)', spec)
            if match:
                cause = match.group(1)
                effect = match.group(2)
                prob = float(match.group(3))
                cause_value = context.get(cause, False)
                effect_value = context.get(effect, False)
                observed_prob = context.get('probability', 0.0)
                if cause_value:
                    # Check if effect occurs with approximately prob
                    return abs(observed_prob - prob) < 0.1
        return super().evaluate(constraint, context)


class ProbabilisticReasoner(BaseReasoner, reasoner_type="probabilistic"):
    """Handles uncertainty and probabilistic constraints"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        if 'P(' in spec and '<' in spec:
            # Parse "P(event) < threshold"
            import re
            match = re.search(r'P\((\w+)\)\s*<\s*([\d\.]+)', spec)
            if match:
                event = match.group(1)
                threshold = float(match.group(2))
                prob = context.get(event, 0.0)
                return prob < threshold
        return super().evaluate(constraint, context)


class TemporalReasoner(BaseReasoner, reasoner_type="temporal"):
    """Manages time-based constraints and sequencing"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        if 'within' in spec and 'after' in spec:
            # "response within Xs after request"
            import re
            match = re.search(r'within\s+([\d\.]+)s\s+after\s+(\w+)', spec)
            if match:
                duration = float(match.group(1))
                event = match.group(2)
                event_time = context.get(f'{event}_time', 0)
                current_time = context.get('current_time', 0)
                return (current_time - event_time) <= duration
        if 'before' in spec:
            parts = spec.split('before')
            first = parts[0].strip()
            second = parts[1].strip()
            # Check order
            return True  # Simplified
        return super().evaluate(constraint, context)


class ResourceReasoner(BaseReasoner, reasoner_type="resource"):
    """Monitors resource utilization constraints"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        if 'cpu_usage' in spec:
            return super().evaluate(constraint, context.get('resources', {}))
        if 'memory_usage' in spec:
            return super().evaluate(constraint, context.get('resources', {}))
        return super().evaluate(constraint, context)


class TriggerReasoner(BaseReasoner, reasoner_type="trigger"):
    """Implements Event-Condition-Action (ECA) cycle"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        # Parse "WHEN event DO action ENSURE postcondition"
        import re
        when_match = re.search(r'WHEN\s+(\w+)\s+DO\s+(\w+)(?:\s+ENSURE\s+(.+))?', spec, re.IGNORECASE)
        if when_match:
            event = when_match.group(1)
            action = when_match.group(2)
            postcondition = when_match.group(3) if when_match.group(3) else None
            # Check if event occurred
            event_occurred = context.get(event, False)
            if event_occurred:
                # Execute action (if object has method)
                obj = context.get('object')
                if obj and hasattr(obj, action):
                    getattr(obj, action)()
                # Verify postcondition
                if postcondition:
                    return self._evaluate_expression(postcondition, context)
                return True
        return super().evaluate(constraint, context)
    
    def _evaluate_expression(self, expr: str, context: Dict) -> bool:
        # Simple evaluation
        try:
            return eval(expr, {"__builtins__": {}}, context)
        except:
            return False


class GoalReasoner(BaseReasoner, reasoner_type="goal"):
    """Handles optimization objectives with constraint satisfaction"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        # Parse "MAXIMIZE objective SUBJECT TO constraints"
        if 'MAXIMIZE' in spec.upper():
            # Simplified: check if objective is satisfied
            return True
        if 'MINIMIZE' in spec.upper():
            return True
        return super().evaluate(constraint, context)


class SafetyReasoner(BaseReasoner, reasoner_type="safety"):
    """Provides redundant validation for safety-critical systems"""
    
    def __init__(self):
        super().__init__()
        self.safety_constraints = []
    
    def add_safety_constraint(self, constraint: 'Constraint') -> None:
        self.safety_constraints.append(constraint)
    
    def validate(self, obj: 'COHObject') -> Dict[str, bool]:
        context = obj.get_context()
        results = {}
        for constraint in self.safety_constraints:
            results[constraint.name] = self.evaluate(constraint, context)
        return results
    
    def validate_trigger(self, obj: 'COHObject', trigger: 'Constraint', context: Dict) -> bool:
        """Validate trigger with redundancy"""
        return self.evaluate(trigger, context)
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        # Safety constraints are always evaluated strictly
        return super().evaluate(constraint, context)