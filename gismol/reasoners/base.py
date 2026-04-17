"""Base reasoner classes"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type
import logging
from gismol.core.constraints import Constraint

logger = logging.getLogger(__name__)


class Reasoner(ABC):
    """Base class for all constraint reasoners"""
    
    _registry: Dict[str, Type['Reasoner']] = {}
    
    def __init_subclass__(cls, reasoner_type: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if reasoner_type:
            cls._registry[reasoner_type] = cls
    
    @classmethod
    def get_reasoner(cls, reasoner_type: str) -> Type['Reasoner']:
        return cls._registry.get(reasoner_type, BaseReasoner)
    
    @abstractmethod
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        """Evaluate constraint against context"""
        pass
    
    def attempt_resolution(self, constraint: 'Constraint', context: Dict) -> bool:
        """Attempt to resolve constraint violation"""
        return False


class BaseReasoner(Reasoner, reasoner_type="base"):
    """Fallback reasoner handling basic constraint evaluation"""
    
    def evaluate(self, constraint: 'Constraint', context: Dict) -> bool:
        spec = constraint.specification
        # Handle comparison operators
        for op in ['<=', '>=', '<', '>', '==', '!=']:
            if op in spec:
                parts = spec.split(op)
                if len(parts) == 2:
                    left = self._resolve_value(parts[0].strip(), context)
                    right = self._resolve_value(parts[1].strip(), context)
                    if op == '<':
                        return left < right
                    elif op == '<=':
                        return left <= right
                    elif op == '>':
                        return left > right
                    elif op == '>=':
                        return left >= right
                    elif op == '==':
                        return left == right
                    elif op == '!=':
                        return left != right
        # Default to True if cannot evaluate
        return True
    
    def _resolve_value(self, expr: str, context: Dict) -> Any:
        """Resolve dot-notation and variable references"""
        # Handle numeric literals
        try:
            return float(expr)
        except ValueError:
            pass
        # Handle boolean
        if expr.lower() == 'true':
            return True
        if expr.lower() == 'false':
            return False
        # Handle dot notation
        parts = expr.split('.')
        value = context
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part, None)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
        return value