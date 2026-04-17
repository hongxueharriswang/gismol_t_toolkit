"""Constraint monitoring daemons"""

import threading
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from .exceptions import ConstraintViolation
from .coh_object import COHObject

logger = logging.getLogger(__name__)


class ConstraintDaemon(ABC):
    """Base class for constraint monitoring daemons"""
    
    def __init__(self, parent_object: 'COHObject', interval: float = 0.1):
        self.parent = parent_object
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    @abstractmethod
    def check(self) -> None:
        """Perform the monitoring check"""
        pass
    
    def _run(self) -> None:
        """Daemon loop"""
        while self._running:
            try:
                self.check()
            except Exception as e:
                logger.error(f"Daemon error: {e}")
            time.sleep(self.interval)
    
    def start(self) -> None:
        """Start the daemon thread"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Started daemon {self.__class__.__name__} for {self.parent.name}")
    
    def stop(self) -> None:
        """Stop the daemon"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info(f"Stopped daemon {self.__class__.__name__} for {self.parent.name}")


class IdentityConstraintDaemon(ConstraintDaemon):
    """Monitors identity constraints (invariants)"""
    
    def check(self) -> None:
        context = self.parent.get_context()
        for constraint in self.parent.identity_constraints:
            if not self.parent.constraint_system.validate_single(constraint, context):
                logger.warning(f"Identity constraint violated: {constraint.name}")
                # Attempt auto-resolution if possible
                if hasattr(constraint, 'severity') and constraint.severity >= 8:
                    raise ConstraintViolation(
                        constraint.name, constraint.specification, context,
                        severity=constraint.severity
                    )


class TriggerConstraintDaemon(ConstraintDaemon):
    """Handles event-driven trigger constraints"""
    
    def __init__(self, parent_object: 'COHObject', interval: float = 0.05):
        super().__init__(parent_object, interval)
        self._last_state: Dict[str, Any] = {}
    
    def check(self) -> None:
        context = self.parent.get_context()
        for constraint in self.parent.trigger_constraints:
            # Check precondition
            precond = getattr(constraint, 'parsed_spec', {}).get('precondition', '')
            if precond and self._evaluate_precondition(precond, context):
                # Execute action
                action = getattr(constraint, 'parsed_spec', {}).get('action', '')
                if action:
                    self._execute_action(action, context)
    
    def _evaluate_precondition(self, precond: str, context: Dict) -> bool:
        # Simple evaluation (could be expanded)
        try:
            return eval(precond, {"__builtins__": {}}, context)
        except:
            return False
    
    def _execute_action(self, action: str, context: Dict) -> None:
        # Execute method if it exists
        if action in self.parent.methods:
            self.parent.execute_method(action)


class GoalConstraintDaemon(ConstraintDaemon):
    """Tracks goal achievement progress"""
    
    def __init__(self, parent_object: 'COHObject', interval: float = 1.0):
        super().__init__(parent_object, interval)
        self.goal_progress: Dict[str, float] = {}
    
    def check(self) -> None:
        context = self.parent.get_context()
        for constraint in self.parent.goal_constraints:
            # Evaluate how close the goal is to being satisfied
            satisfied = self.parent.constraint_system.validate_single(constraint, context)
            progress = 1.0 if satisfied else 0.5  # simplistic
            self.goal_progress[constraint.name] = progress
            logger.debug(f"Goal {constraint.name} progress: {progress}")