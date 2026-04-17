from typing import List, Tuple
from gismol.core.coh_object import COHObject

class RelationsMiner:
    """Mine relationships from text"""
    
    def mine(self, text: str, objects: List[COHObject]) -> List[Tuple[COHObject, COHObject, str]]:
        """Extract relations between objects"""
        text_lower = text.lower()
        relations = []
        n = len(objects)
        for i in range(n):
            obj1 = objects[i]
            for j in range(n):
                if i == j:
                    continue
                obj2 = objects[j]
                # Check patterns like "obj1 has obj2" or "obj1 connected to obj2"
                if f"{obj1.name} has {obj2.name}" in text_lower:
                    relations.append((obj1, obj2, 'has'))
                elif f"{obj1.name} connected to {obj2.name}" in text_lower:
                    relations.append((obj1, obj2, 'connected_to'))
        return relations