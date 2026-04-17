"""General-purpose reasoners"""

import re
from typing import Dict, Any, Callable
from .base import BaseReasoner
from gismol.core.constraints import Constraint

class GeneralReasoner(BaseReasoner, reasoner_type="general"):
    """Handles complex logical expressions with custom functions"""
    
    def __init__(self):
        super().__init__()
        self._functions: Dict[str, Callable] = {}
    
    def register_function(self, name: str, func: Callable) -> None:
        self._functions[name] = func
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        # Handle logical operators
        if 'implies' in spec:
            parts = spec.split('implies')
            antecedent = parts[0].strip()
            consequent = parts[1].strip()
            # A implies B is equivalent to (not A) or B
            return (not self._evaluate_expression(antecedent, context)) or self._evaluate_expression(consequent, context)
        if 'and' in spec:
            subexprs = spec.split('and')
            return all(self._evaluate_expression(sub.strip(), context) for sub in subexprs)
        if 'or' in spec:
            subexprs = spec.split('or')
            return any(self._evaluate_expression(sub.strip(), context) for sub in subexprs)
        if 'iff' in spec:
            parts = spec.split('iff')
            left = self._evaluate_expression(parts[0].strip(), context)
            right = self._evaluate_expression(parts[1].strip(), context)
            return left == right
        return super().evaluate(constraint, context)
    
    def _evaluate_expression(self, expr: str, context: Dict) -> Any:
        # Check for custom function calls
        for func_name, func in self._functions.items():
            if expr.startswith(func_name + '('):
                # Extract argument
                arg = expr[len(func_name)+1:-1]
                return func(self._resolve_value(arg, context))
        return self._resolve_value(expr, context)


class AttributeReasoner(BaseReasoner, reasoner_type="attribute"):
    """Specialized in attribute value constraints"""
    pass


class CardinalityReasoner(BaseReasoner, reasoner_type="cardinality"):
    """Handles numerical constraints and collection sizes"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        if 'count(' in spec:
            # Parse count(collection) operator
            match = re.search(r'count\((\w+)\)\s*([<>=]+)\s*(\d+)', spec)
            if match:
                collection_name = match.group(1)
                op = match.group(2)
                threshold = int(match.group(3))
                collection = context.get(collection_name, [])
                count = len(collection)
                if op == '<':
                    return count < threshold
                elif op == '<=':
                    return count <= threshold
                elif op == '>':
                    return count > threshold
                elif op == '>=':
                    return count >= threshold
                elif op == '==':
                    return count == threshold
        return super().evaluate(constraint, context)


class RelationalReasoner(BaseReasoner, reasoner_type="relational"):
    """Manages entity relationships"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        if 'transitive(' in spec:
            # Simplified: assume transitive closure holds
            return True
        if 'symmetric(' in spec:
            return True
        # Check simple "A related_to B"
        if ' ' in spec and len(spec.split()) == 3:
            subj, rel, obj = spec.split()
            # Look up relationship in context
            relations = context.get('relations', {})
            key = (subj, obj)
            return key in relations.get(rel, [])
        return super().evaluate(constraint, context)


class CompositionReasoner(BaseReasoner, reasoner_type="composition"):
    """Validates part-whole relationships"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        if 'has' in spec:
            parts = spec.split('has')
            whole = parts[0].strip()
            part = parts[1].strip()
            # Check if whole contains part in its children
            obj = context.get('object')
            if obj and obj.name == whole:
                for child in obj.children:
                    if child.name == part:
                        return True
            return False
        return super().evaluate(constraint, context)


class ComponentReasoner(BaseReasoner, reasoner_type="component"):
    """Manages inter-component relationships"""
    pass