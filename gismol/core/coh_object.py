"""Core COHObject implementation"""

import uuid
import logging
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime

from .exceptions import ConstraintViolation, MissingEmbeddingModel, HierarchyCycleError
from .constraints import Constraint, ConstraintSystem
from gismol.neural.components import NeuralComponent
from gismol.core.repository import COHRepository
from gismol.core.daemons import ConstraintDaemon
from gismol.neural import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class COHRelation:
    """Manages relationships between COHObjects"""
    relations: Dict[str, List[tuple]] = field(default_factory=dict)
    
    def add_relation(self, source: 'COHObject', target: 'COHObject', 
                     relation_type: str, name: str = None) -> None:
        """Add a relationship between two COHObjects"""
        key = relation_type
        if key not in self.relations:
            self.relations[key] = []
        self.relations[key].append((source, target, name or f"{source.name}_{target.name}"))
    
    def get_relations(self, relation_type: str = None) -> List[tuple]:
        """Get relations, optionally filtered by type"""
        if relation_type:
            return self.relations.get(relation_type, [])
        all_rels = []
        for rels in self.relations.values():
            all_rels.extend(rels)
        return all_rels
    
    def visualize(self, format: str = "mermaid") -> str:
        """Generate visualization (Mermaid format)"""
        lines = ["graph LR"]
        for rels in self.relations.values():
            for src, tgt, name in rels:
                lines.append(f"    {src.name}({src.name}) -->|{name}| {tgt.name}({tgt.name})")
        return "\n".join(lines)


class COHObject:
    """
    Fundamental entity in COH framework representing intelligent objects
    with constraints, neural capabilities, and hierarchical relationships.
    """
    
    def __init__(self, name: str = None, parent: Optional['COHObject'] = None):
        self.id = str(uuid.uuid4())
        self.name = name or f"COHObject_{self.id[:8]}"
        self.parent = parent
        self.children: List['COHObject'] = []
        self.attributes: Dict[str, Any] = {}
        self.methods: Dict[str, callable] = {}
        self.neural_components: Dict[str, 'NeuralComponent'] = {}
        self.embedding_model: Optional['EmbeddingModel'] = None
        
        # Constraint systems
        self.identity_constraints: List[Constraint] = []
        self.trigger_constraints: List[Constraint] = []
        self.goal_constraints: List[Constraint] = []
        self.constraint_system = ConstraintSystem()
        
        # Daemons
        self.daemons: Dict[str, 'ConstraintDaemon'] = {}
        
        # Relationships
        self.relations = COHRelation()
        
        # State
        self._initialized = False
        self._daemons_running = False
        
        # Register with parent if provided
        if parent:
            parent.add_child(self)
    
    def add_child(self, child: 'COHObject') -> None:
        """Add a child component (hierarchy)"""
        if self._would_create_cycle(child):
            raise HierarchyCycleError(f"Adding {child.name} would create a cycle")
        self.children.append(child)
        child.parent = self
    
    def _would_create_cycle(self, potential_child: 'COHObject') -> bool:
        """Check if adding child would create a cycle in the DAG"""
        # Simple check: if potential_child is an ancestor of self, cycle
        current = self
        while current.parent:
            if current.parent == potential_child:
                return True
            current = current.parent
        return False
    
    def remove_child(self, child: 'COHObject') -> None:
        """Remove a child component"""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
    
    def add_attribute(self, key: str, value: Any) -> None:
        """Add or update an attribute"""
        self.attributes[key] = value
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get attribute value"""
        return self.attributes.get(key, default)
    
    def add_method(self, name: str, func: callable) -> None:
        """Add an executable method"""
        self.methods[name] = func
    
    def execute_method(self, name: str, *args, **kwargs) -> Any:
        """Execute a method with constraint checking"""
        if name not in self.methods:
            raise AttributeError(f"Method '{name}' not found")
        
        # Check trigger constraints before execution
        for constraint in self.trigger_constraints:
            if not self.constraint_system.validate_single(constraint, self.get_context()):
                raise ConstraintViolation(
                    constraint.name, constraint.specification, self.get_context(),
                    severity=getattr(constraint, 'severity', 5)
                )
        
        # Execute method
        result = self.methods[name](self, *args, **kwargs)
        
        # Check identity constraints after execution
        for constraint in self.identity_constraints:
            if not self.constraint_system.validate_single(constraint, self.get_context()):
                raise ConstraintViolation(
                    constraint.name, constraint.specification, self.get_context(),
                    severity=getattr(constraint, 'severity', 10)
                )
        
        return result
    
    def add_identity_constraint(self, constraint_spec: Dict) -> None:
        """Add an identity constraint (invariant)"""
        constraint = Constraint.from_dict(constraint_spec, category='identity')
        self.identity_constraints.append(constraint)
        self.constraint_system.add_constraint(constraint)
    
    def add_trigger_constraint(self, constraint_spec: Dict) -> None:
        """Add a trigger constraint (ECA rule)"""
        constraint = Constraint.from_dict(constraint_spec, category='trigger')
        self.trigger_constraints.append(constraint)
        self.constraint_system.add_constraint(constraint)
    
    def add_goal_constraint(self, constraint_spec: Dict) -> None:
        """Add a goal constraint (optimization objective)"""
        constraint = Constraint.from_dict(constraint_spec, category='goal')
        self.goal_constraints.append(constraint)
        self.constraint_system.add_constraint(constraint)
    
    def add_neural_component(self, name: str, component: 'NeuralComponent', 
                             is_embedding_model: bool = False) -> None:
        """Add a neural component to this object"""
        self.neural_components[name] = component
        component.parent_object = self
        if is_embedding_model:
            self.embedding_model = component
    
    def get_neural_component(self, name: str) -> 'NeuralComponent':
        """Retrieve a neural component by name"""
        return self.neural_components.get(name)
    
    def get_context(self) -> Dict[str, Any]:
        """Build evaluation context for constraints"""
        context = {
            'object': self,
            'name': self.name,
            'id': self.id,
            'parent': self.parent,
            'children': self.children,
            **self.attributes
        }
        # Add neural component outputs if available
        for name, nc in self.neural_components.items():
            context[f'neural_{name}'] = nc
        return context
    
    def semantic_distance(self, other: 'COHObject') -> float:
        """Compute semantic distance to another object using embedding model"""
        if not self.embedding_model:
            raise MissingEmbeddingModel("No embedding model set for this object")
        emb1 = self.embedding_model.embed_object(self)
        emb2 = self.embedding_model.embed_object(other)
        # Cosine distance
        import numpy as np
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return 1.0 - similarity  # distance
    
    def start_daemons(self) -> None:
        """Start all constraint daemons"""
        from .daemons import IdentityConstraintDaemon, TriggerConstraintDaemon, GoalConstraintDaemon
        self.daemons['identity'] = IdentityConstraintDaemon(self)
        self.daemons['trigger'] = TriggerConstraintDaemon(self)
        self.daemons['goal'] = GoalConstraintDaemon(self)
        for daemon in self.daemons.values():
            daemon.start()
        self._daemons_running = True
    
    def stop_daemons(self) -> None:
        """Stop all daemons"""
        for daemon in self.daemons.values():
            daemon.stop()
        self._daemons_running = False
    
    def initialize_system(self) -> None:
        """Initialize the object and all its components"""
        for child in self.children:
            child.initialize_system()
        # Validate all constraints
        context = self.get_context()
        for constraint in self.identity_constraints:
            if not self.constraint_system.validate_single(constraint, context):
                raise ConstraintViolation(
                    constraint.name, constraint.specification, context,
                    severity=getattr(constraint, 'severity', 10)
                )
        self._initialized = True
    
    def to_dict(self) -> Dict:
        """Serialize object to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'attributes': self.attributes,
            'identity_constraints': [c.to_dict() for c in self.identity_constraints],
            'trigger_constraints': [c.to_dict() for c in self.trigger_constraints],
            'goal_constraints': [c.to_dict() for c in self.goal_constraints],
            'children_ids': [c.id for c in self.children],
            'parent_id': self.parent.id if self.parent else None
        }