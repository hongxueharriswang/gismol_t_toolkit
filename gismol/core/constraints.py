"""Constraint system and specialized constraints"""

import re
import ast
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import InvalidConstraintError, ConstraintViolation
from gismol.reasoners.base import BaseReasoner, Reasoner

class ConstraintCategory(Enum):
    ATTRIBUTE = "attribute"
    GEOMETRIC = "geometric"
    TEMPORAL = "temporal"
    RESOURCE = "resource"
    CARDINALITY = "cardinality"
    COMPOSITION = "composition"
    PHYSICAL = "physical"
    CAUSAL = "causal"
    RELATIONAL = "relational"
    PROBABILISTIC = "probabilistic"
    IDENTITY = "identity"
    TRIGGER = "trigger"
    GOAL = "goal"
    SAFETY = "safety"
    AUTO = "auto"


@dataclass
class Constraint:
    """Represents a constraint with specification and metadata"""
    name: str
    specification: str
    category: ConstraintCategory = ConstraintCategory.AUTO
    severity: int = 5  # 1-10, 10 most severe
    priority: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL
    parsed_spec: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict, category: str = None) -> 'Constraint':
        """Create constraint from dictionary"""
        cat = data.get('category', category or 'auto')
        if cat == 'auto':
            cat = cls._auto_detect_category(data.get('specification', ''))
        else:
            cat = ConstraintCategory(cat)
        
        return cls(
            name=data['name'],
            specification=data['specification'],
            category=cat,
            severity=data.get('severity', 5),
            priority=data.get('priority', 'MEDIUM'),
            parsed_spec=cls._parse_specification(data['specification'])
        )
    
    @staticmethod
    def _auto_detect_category(spec: str) -> ConstraintCategory:
        spec_lower = spec.lower()
        if 'distance' in spec_lower or 'position' in spec_lower:
            return ConstraintCategory.GEOMETRIC
        if 'time' in spec_lower or 'duration' in spec_lower or 'before' in spec_lower:
            return ConstraintCategory.TEMPORAL
        if 'resource' in spec_lower or 'cpu' in spec_lower or 'memory' in spec_lower:
            return ConstraintCategory.RESOURCE
        if 'count' in spec_lower or 'cardinality' in spec_lower:
            return ConstraintCategory.CARDINALITY
        if 'cause' in spec_lower or 'effect' in spec_lower:
            return ConstraintCategory.CAUSAL
        if 'probability' in spec_lower or 'p(' in spec_lower:
            return ConstraintCategory.PROBABILISTIC
        if '=' in spec_lower or '<' in spec_lower or '>' in spec_lower:
            return ConstraintCategory.ATTRIBUTE
        return ConstraintCategory.ATTRIBUTE
    
    @staticmethod
    def _parse_specification(spec: str) -> Dict:
        """Parse constraint specification into structured form"""
        # Simple placeholder parsing
        result = {'expression': spec}
        # Extract comparison operators
        for op in ['<', '>', '<=', '>=', '==', '!=']:
            if op in spec:
                parts = spec.split(op)
                if len(parts) == 2:
                    result['left'] = parts[0].strip()
                    result['operator'] = op
                    result['right'] = parts[1].strip()
                break
        return result
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'specification': self.specification,
            'category': self.category.value,
            'severity': self.severity,
            'priority': self.priority
        }


class ConstraintSystem:
    """Manages declarative constraints with specialized reasoners"""
    
    def __init__(self):
        self.constraints: List[Constraint] = []
        self.reasoners: Dict[str, 'Reasoner'] = {}
    
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the system"""
        self.constraints.append(constraint)
    
    def register_reasoner(self, category: str, reasoner: 'Reasoner') -> None:
        """Register a reasoner for a constraint category"""
        self.reasoners[category] = reasoner
    
    def validate_single(self, constraint: Constraint, context: Dict) -> bool:
        """Validate a single constraint using appropriate reasoner"""
        category = constraint.category.value
        reasoner = self.reasoners.get(category)
        if not reasoner:
            from gismol.reasoners import BaseReasoner
            reasoner = BaseReasoner()
        return reasoner.evaluate(constraint, context)
    
    def validate_all(self, context: Dict) -> Dict[str, bool]:
        """Validate all constraints, return results per constraint"""
        results = {}
        for constraint in self.constraints:
            results[constraint.name] = self.validate_single(constraint, context)
        return results
    
    def validate_all_raise(self, context: Dict) -> bool:
        """Validate all constraints, raise on first violation"""
        for constraint in self.constraints:
            if not self.validate_single(constraint, context):
                raise ConstraintViolation(
                    constraint.name, constraint.specification, context,
                    severity=constraint.severity
                )
        return True