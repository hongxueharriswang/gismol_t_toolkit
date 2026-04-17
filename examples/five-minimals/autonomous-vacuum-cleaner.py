#!/usr/bin/env python3
"""
Autonomous Vacuum Cleaner - COH Implementation using GISMOL Toolkit

This example models a robotic vacuum as a Constrained Object Hierarchy (COH).
It navigates a grid, cleans dirt, plans coverage using a neural network,
respects battery and boundary constraints, and autonomously recharges.
"""

import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

# Import GISMOL components
from gismol.core import COHObject
from gismol.core.daemons import ConstraintDaemon
from gismol.neural import NeuralComponent
from gismol.neural.embeddings import EmbeddingModel


# =============================================================================
# 1. Grid World Environment (simulated)
# =============================================================================
class GridWorld:
    """Simple grid environment for the vacuum cleaner."""
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        # Dirt density: random per cell, between 0 and 1
        self.dirt = np.random.rand(height, width)
        # Initial coverage map: all zeros (not cleaned)
        self.coverage = np.zeros((height, width), dtype=bool)
        self.robot_pos = (height // 2, width // 2)  # start center

    def get_dirt_density(self, x: int, y: int) -> float:
        """Return dirt level at (x,y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.dirt[y, x]
        return 0.0

    def clean_cell(self, x: int, y: int) -> float:
        """Clean the cell, return amount of dirt removed."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        dirt_removed = self.dirt[y, x]
        self.dirt[y, x] = 0.0
        self.coverage[y, x] = True
        return dirt_removed

    def get_coverage_percentage(self) -> float:
        """Return fraction of cells cleaned."""
        return np.mean(self.coverage)

    def is_within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_state_vector(self) -> np.ndarray:
        """Return a flattened view of dirt and coverage for neural input."""
        # Concatenate dirt and coverage maps (2 x H x W)
        state = np.stack([self.dirt, self.coverage.astype(float)], axis=0)
        return state.flatten()


# =============================================================================
# 2. Neural Component: Coverage Planner (Deep Q-Network)
# =============================================================================
class DQN(NeuralComponent):
    """Simple DQN with one hidden layer for action selection."""
    def __init__(self, name: str, state_dim: int, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__(name, input_dim=state_dim, output_dim=action_dim)
        self.action_dim = action_dim
        # Simulate neural network weights (random initialization)
        self.W1 = np.random.randn(hidden_dim, state_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(action_dim, hidden_dim) * 0.01
        self.b2 = np.zeros(action_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions."""
        x = np.asarray(x).flatten()
        h = np.maximum(0, self.W1 @ x + self.b1)  # ReLU
        q = self.W2 @ h + self.b2
        return q

    def act(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.forward(state)
        return int(np.argmax(q_values))

    def train_step(self, state, action, reward, next_state, done, optimizer):
        """Single training step (simplified). In real DQN, would use replay buffer."""
        # This is a placeholder – actual training would involve backprop.
        # For demonstration, we just update weights randomly.
        self.W1 += np.random.randn(*self.W1.shape) * 0.001
        self.W2 += np.random.randn(*self.W2.shape) * 0.001


# =============================================================================
# 3. Custom Embedding Model: Local Map CNN (simulated)
# =============================================================================
class CoverageEmbedding(EmbeddingModel):
    """Produces a 64‑dim embedding of the robot's local view (5x5 patch)."""
    def __init__(self, name: str = "coverage_embedder", embedding_dim: int = 64):
        super().__init__(name, embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim

    def embed(self, obj: COHObject) -> np.ndarray:
        """Create embedding from robot's current state (position, local dirt map)."""
        # Get robot's position and world reference from parent object
        world = obj.get_attribute("world")
        if world is None:
            return np.zeros(self.embedding_dim)
        x, y = obj.get_attribute("position", (0, 0))
        # Extract 5x5 local patch (dirt + coverage)
        patch_size = 5
        half = patch_size // 2
        local_dirt = []
        for dy in range(-half, half+1):
            for dx in range(-half, half+1):
                nx, ny = x + dx, y + dy
                if world.is_within_bounds(nx, ny):
                    local_dirt.append(world.get_dirt_density(nx, ny))
                else:
                    local_dirt.append(0.0)
        # Convert to numpy and project to embedding dimension using random matrix
        vec = np.array(local_dirt)
        # Simple random projection (simulates a learned CNN)
        proj = np.random.randn(self.embedding_dim, len(vec)) * 0.01
        embedding = proj @ vec
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding


# =============================================================================
# 4. Custom Daemon: BatteryGuardian
# =============================================================================
class BatteryGuardian(ConstraintDaemon):
    """Monitors battery level and prevents deep discharge."""
    def __init__(self, parent: COHObject, interval: float = 1.0):
        super().__init__(parent, interval)
        self.critical_level = 10  # percent

    def check(self) -> None:
        battery = self.parent.get_attribute("battery_level", 100)
        if battery < self.critical_level:
            print("[BatteryGuardian] Battery critical! Forcing return to dock.")
            self.parent.execute_method("return_to_dock")
        elif battery < 20:
            print(f"[BatteryGuardian] Low battery ({battery:.1f}%). Consider returning.")


# =============================================================================
# 5. Main Vacuum Cleaner COH Object
# =============================================================================
class AutonomousVacuum(COHObject):
    """
    Robotic vacuum cleaner with navigation, coverage planning, and constraints.
    """
    def __init__(self, name: str = "VacuumBot", world: Optional[GridWorld] = None):
        super().__init__(name)
        self.world = world or GridWorld()

        # ---- Attributes (A) ----
        self.add_attribute("position", self.world.robot_pos)
        self.add_attribute("battery_level", 100.0)      # percentage
        self.add_attribute("time_elapsed", 0.0)         # minutes
        self.add_attribute("total_dirt_removed", 0.0)
        self.add_attribute("world", self.world)         # reference

        # ---- Neural Components (N) ----
        state_dim = self.world.width * self.world.height * 2  # dirt + coverage maps
        self.dqn = DQN(name="coverage_planner", state_dim=state_dim, action_dim=4)
        self.add_neural_component("coverage_planner", self.dqn)

        # ---- Embedding (E) ----
        embedder = CoverageEmbedding(name="local_embedder", embedding_dim=64)
        self.add_neural_component("embedder", embedder, is_embedding_model=True)

        # ---- Methods (M) ----
        self.add_method("move", self.move)
        self.add_method("clean_cell", self.clean_cell)
        self.add_method("return_to_dock", self.return_to_dock)
        self.add_method("update_map", self.update_map)

        # ---- Identity Constraints (I) ----
        self.add_identity_constraint({
            'name': 'minimum_battery',
            'specification': 'battery_level >= 10',
            'severity': 10,
            'category': 'safety'
        })
        self.add_identity_constraint({
            'name': 'within_bounds',
            'specification': 'position[0] >= 0 and position[0] < world.width and position[1] >= 0 and position[1] < world.height',
            'severity': 9
        })

        # ---- Trigger Constraints (T) ----
        self.add_trigger_constraint({
            'name': 'low_battery_return',
            'specification': 'WHEN battery_level < 20 DO return_to_dock()',
            'priority': 'HIGH'
        })
        self.add_trigger_constraint({
            'name': 'clean_dirty_cell',
            'specification': 'WHEN world.get_dirt_density(position[0], position[1]) > 0.5 DO clean_cell()',
            'priority': 'MEDIUM'
        })

        # ---- Goal Constraints (G) ----
        self.add_goal_constraint({
            'name': 'coverage_efficiency',
            'specification': 'MAXIMIZE coverage_percentage - 0.1 * time_elapsed - 100 * battery_depletion_penalty',
            'priority': 'HIGH'
        })

        # ---- Daemons (D) ----
        battery_guardian = BatteryGuardian(self, interval=2.0)
        self.daemons['battery_guardian'] = battery_guardian

    # ---- Method implementations ----
    def move(self, dx: int, dy: int) -> bool:
        """Attempt to move by (dx, dy); returns success."""
        x, y = self.get_attribute("position")
        nx, ny = x + dx, y + dy
        if self.world.is_within_bounds(nx, ny):
            self.add_attribute("position", (nx, ny))
            # Battery consumption: 0.5% per move
            new_battery = self.get_attribute("battery_level") - 0.5
            self.add_attribute("battery_level", max(0, new_battery))
            # Update elapsed time
            self.add_attribute("time_elapsed", self.get_attribute("time_elapsed") + 0.1)  # 0.1 min per move
            return True
        return False

    def clean_cell(self) -> float:
        """Clean current cell, return dirt removed."""
        x, y = self.get_attribute("position")
        dirt = self.world.clean_cell(x, y)
        if dirt > 0:
            self.add_attribute("total_dirt_removed", self.get_attribute("total_dirt_removed") + dirt)
            # Battery consumption: 1% per clean
            new_battery = self.get_attribute("battery_level") - 1.0
            self.add_attribute("battery_level", max(0, new_battery))
            self.add_attribute("time_elapsed", self.get_attribute("time_elapsed") + 0.2)
        return dirt

    def return_to_dock(self) -> None:
        """Navigate back to charging dock (center of grid)."""
        print("Returning to dock for recharge...")
        # Simplified: teleport to dock and recharge
        dock_pos = (self.world.width // 2, self.world.height // 2)
        self.add_attribute("position", dock_pos)
        self.add_attribute("battery_level", 100.0)
        self.add_attribute("time_elapsed", self.get_attribute("time_elapsed") + 5.0)  # 5 min recharge
        print("Recharged. Resuming cleaning.")

    def update_map(self) -> None:
        """Update internal map (just sync with world)."""
        # In a real system, would merge sensor data; here we just log coverage.
        cov = self.world.get_coverage_percentage()
        print(f"Coverage: {cov*100:.1f}% | Battery: {self.get_attribute('battery_level'):.1f}%")

    def get_coverage_percentage(self) -> float:
        return self.world.get_coverage_percentage()


# =============================================================================
# 6. Simulation with DQN-based planning
# =============================================================================
def run_simulation(steps: int = 500, render: bool = True):
    """Run the vacuum cleaner for a number of time steps."""
    # Create environment and robot
    world = GridWorld(width=8, height=8)  # smaller for faster simulation
    robot = AutonomousVacuum("VacuumBot", world)

    # Initialize COH system
    robot.initialize_system()
    robot.start_daemons()

    # Training parameters for DQN (simplified)
    epsilon = 0.2  # exploration rate
    actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right

    print("=== Vacuum Cleaner Simulation ===")
    for step in range(steps):
        # Get current state (full grid dirt+coverage)
        state = world.get_state_vector()

        # Neural planner chooses action
        action_idx = robot.dqn.act(state, epsilon)
        dx, dy = actions[action_idx]

        # Execute move
        success = robot.execute_method("move", dx, dy)

        # After moving, check if current cell is dirty and clean if needed
        x, y = robot.get_attribute("position")
        if world.get_dirt_density(x, y) > 0.2:
            robot.execute_method("clean_cell")

        # Update map periodically
        if step % 50 == 0:
            robot.execute_method("update_map")

        # Render simple ASCII grid every 20 steps
        if render and step % 20 == 0:
            print(f"\n--- Step {step} ---")
            print(f"Position: {robot.get_attribute('position')}")
            print(f"Battery: {robot.get_attribute('battery_level'):.1f}%")
            print(f"Coverage: {robot.get_coverage_percentage()*100:.1f}%")
            # Simple grid display
            grid_str = ""
            for y in range(world.height):
                row = ""
                for x in range(world.width):
                    if (x, y) == robot.get_attribute("position"):
                        row += "🤖 "
                    elif world.coverage[y, x]:
                        row += "✓ "
                    else:
                        row += ". "
                grid_str += row + "\n"
            print(grid_str)

        # Decay exploration
        epsilon = max(0.05, epsilon * 0.999)

        # Stop if coverage is >95% and battery is high
        if robot.get_coverage_percentage() > 0.95 and robot.get_attribute("battery_level") > 80:
            print("Cleaning complete! Shutting down.")
            break

        # Small delay to observe
        time.sleep(0.02)

    robot.stop_daemons()
    print("\n=== Simulation Complete ===")
    print(f"Final coverage: {robot.get_coverage_percentage()*100:.1f}%")
    print(f"Total dirt removed: {robot.get_attribute('total_dirt_removed'):.2f}")
    print(f"Time elapsed: {robot.get_attribute('time_elapsed'):.1f} minutes")


# =============================================================================
# 7. Main Entry Point
# =============================================================================
if __name__ == "__main__":
    run_simulation(steps=800, render=True)