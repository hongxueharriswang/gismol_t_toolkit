"""Custom exception classes for COH framework"""

class COHError(Exception):
    """Base exception for all COH-related errors"""
    pass

class ConstraintViolation(COHError):
    """Raised when a constraint is violated"""
    def __init__(self, constraint_name, specification, context, severity=5):
        self.constraint_name = constraint_name
        self.specification = specification
        self.context = context
        self.severity = severity
        super().__init__(self.detailed_report())
    
    def detailed_report(self):
        return f"ConstraintViolation: '{self.constraint_name}' - {self.specification} (severity {self.severity})"
    
    def attempt_autofix(self):
        """Attempt automatic resolution - override in subclasses"""
        return False

class ResolutionFailure(COHError):
    """Raised when constraint resolution fails"""
    pass

class MissingEmbeddingModel(COHError):
    """Raised when semantic operation requires an embedding model that is missing"""
    pass

class PlanningFailed(COHError):
    """Raised when planning or scheduling fails"""
    pass

class HierarchyCycleError(COHError):
    """Raised when a cycle is detected in the component DAG"""
    pass

class InvalidConstraintError(COHError):
    """Raised when a constraint specification is invalid"""
    pass

class NeuralComponentError(COHError):
    """Base exception for neural component errors"""
    pass