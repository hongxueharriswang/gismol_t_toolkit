"""Object repository and management"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from .coh_object import COHObject, COHRelation
from .exceptions import MissingEmbeddingModel

logger = logging.getLogger(__name__)


class COHRepository:
    """Manages collections of COHObjects and their relationships"""
    
    def __init__(self):
        self.objects: Dict[str, COHObject] = {}
        self.relations = COHRelation()
        self.focus_object: Optional[COHObject] = None
    
    def add_object(self, obj: COHObject) -> None:
        """Add a COHObject to the repository"""
        self.objects[obj.id] = obj
        logger.info(f"Added object {obj.name} (id: {obj.id})")
    
    def get_object(self, obj_id: str) -> Optional[COHObject]:
        """Retrieve object by ID"""
        return self.objects.get(obj_id)
    
    def find_by_name(self, name: str) -> List[COHObject]:
        """Find objects by name (partial match)"""
        return [obj for obj in self.objects.values() if name.lower() in obj.name.lower()]
    
    def find_by_attribute(self, key: str, value: Any) -> List[COHObject]:
        """Find objects with a matching attribute"""
        return [obj for obj in self.objects.values() if obj.get_attribute(key) == value]
    
    def add_relation(self, source: COHObject, target: COHObject, 
                     relation_type: str, name: str = None) -> None:
        """Add a relation between two objects"""
        self.relations.add_relation(source, target, relation_type, name)
        logger.info(f"Added relation {relation_type} from {source.name} to {target.name}")
    
    def set_focus_object(self, obj_id_or_name: str) -> None:
        """Set the focus object for conversation context"""
        obj = self.objects.get(obj_id_or_name)
        if not obj:
            matches = self.find_by_name(obj_id_or_name)
            if matches:
                obj = matches[0]
        if obj:
            self.focus_object = obj
            logger.info(f"Focus object set to {obj.name}")
        else:
            logger.warning(f"Object '{obj_id_or_name}' not found")
    
    def find_semantic_matches(self, query: str, threshold: float = 0.7) -> List[COHObject]:
        """Find objects semantically similar to query text"""
        results = []
        for obj in self.objects.values():
            if obj.embedding_model:
                # Compute similarity between query and object
                query_emb = obj.embedding_model.embed_text(query)
                obj_emb = obj.embedding_model.embed_object(obj)
                import numpy as np
                similarity = np.dot(query_emb, obj_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(obj_emb) + 1e-8)
                if similarity >= threshold:
                    results.append((obj, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return [obj for obj, _ in results]
    
    def classify_and_extend(self, text: str) -> Optional[COHObject]:
        """Classify text and create/extend COHObject"""
        # Simple implementation: create a new object
        new_obj = COHObject(name=text[:50])
        self.add_object(new_obj)
        return new_obj
    
    def integrate_objects(self, objects: List[COHObject]) -> None:
        """Integrate a list of objects into the repository"""
        for obj in objects:
            self.add_object(obj)
    
    def to_dict(self) -> Dict:
        """Serialize repository to dictionary"""
        return {
            'objects': {oid: obj.to_dict() for oid, obj in self.objects.items()},
            'focus_id': self.focus_object.id if self.focus_object else None
        }