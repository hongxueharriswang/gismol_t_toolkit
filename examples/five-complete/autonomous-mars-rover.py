#!/usr/bin/env python3
"""
Autonomous Rover for Mars Exploration - COH Implementation using GISMOL Toolkit

This example models a Mars rover as a Constrained Object Hierarchy (COH).
It navigates rough terrain, collects science data, manages limited power and
bandwidth, and autonomously re-plans based on new discoveries.
"""

import time
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import deque, defaultdict
from dataclasses import dataclass, field

# Import GISMOL components
from gismol.core import COHObject
from gismol.core.daemons import ConstraintDaemon
from gismol.neural import NeuralComponent, Classifier, Regressor
from gismol.neural.embeddings import EmbeddingModel
from gismol.reasoners import BaseReasoner, TemporalReasoner, SafetyReasoner


# =============================================================================
# 1. Simulated Mars Environment
# =============================================================================

@dataclass
class TerrainCell:
    """A cell in the terrain map."""
    traversability: float  # 0 = impassable, 1 = easy
    science_interest: float  # 0 = none, 1 = high
    mineral_type: str  # "rock", "soil", "ice", "clay"
    altitude: float

@dataclass
class ScienceTarget:
    """A point of scientific interest."""
    id: str
    position: Tuple[int, int]
    interest_score: float  # 0-1
    sample_type: str
    estimated_time_min: float
    data_volume_mb: float

class MarsWorld:
    """Simulated Mars terrain and science targets."""
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.terrain = [[None for _ in range(width)] for _ in range(height)]
        self.science_targets: Dict[str, ScienceTarget] = {}
        self._generate_terrain()
        self._generate_targets()
        self.current_sol = 0  # Martian day
        self.communication_window_open = False
        self.signal_strength = 0.0

    def _generate_terrain(self):
        """Generate random terrain with realistic patterns."""
        for y in range(self.height):
            for x in range(self.width):
                # Simulate rough terrain with some flat areas
                base = 0.5 + 0.3 * math.sin(x * 0.5) * math.cos(y * 0.5)
                # Add random variation
                trav = base + random.gauss(0, 0.15)
                trav = max(0.0, min(1.0, trav))
                # Altitude: higher in some regions
                alt = 1000 + 200 * math.sin(x * 0.3) * math.cos(y * 0.3) + random.gauss(0, 50)
                # Mineral type based on altitude
                if alt > 1200:
                    mineral = "rock"
                elif alt < 900:
                    mineral = "clay"
                else:
                    mineral = "soil"
                # Science interest: higher in varied terrain
                interest = 0.2 + 0.6 * (1 - abs(trav - 0.5) * 2) + random.gauss(0, 0.1)
                interest = max(0.0, min(1.0, interest))
                self.terrain[y][x] = TerrainCell(trav, interest, mineral, alt)

    def _generate_targets(self):
        """Place science targets at interesting locations."""
        target_id = 1
        for y in range(self.height):
            for x in range(self.width):
                if self.terrain[y][x].science_interest > 0.7 and random.random() < 0.1:
                    interest = self.terrain[y][x].science_interest * random.uniform(0.8, 1.2)
                    interest = min(1.0, interest)
                    sample_type = self.terrain[y][x].mineral_type
                    time_est = 5 + 10 * (1 - self.terrain[y][x].traversability)
                    data_mb = 10 + 20 * interest
                    target = ScienceTarget(
                        id=f"T{target_id}",
                        position=(x, y),
                        interest_score=interest,
                        sample_type=sample_type,
                        estimated_time_min=time_est,
                        data_volume_mb=data_mb
                    )
                    self.science_targets[target.id] = target
                    target_id += 1

    def get_traversability(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.terrain[y][x].traversability
        return 0.0

    def get_terrain_patch(self, center_x: int, center_y: int, radius: int = 3) -> np.ndarray:
        """Return a local terrain map (traversability) as a 2D array."""
        patch = np.zeros((2*radius+1, 2*radius+1))
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                x = center_x + dx
                y = center_y + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    patch[dy+radius, dx+radius] = self.terrain[y][x].traversability
                else:
                    patch[dy+radius, dx+radius] = 0  # out of bounds = impassable
        return patch

    def get_nearby_targets(self, x: int, y: int, radius: int = 5) -> List[ScienceTarget]:
        """Return science targets within radius."""
        targets = []
        for target in self.science_targets.values():
            tx, ty = target.position
            if abs(tx - x) <= radius and abs(ty - y) <= radius:
                targets.append(target)
        return targets

    def update_communication(self):
        """Simulate orbital communication windows."""
        # Simple: window every 6 hours (simulated as random)
        self.communication_window_open = random.random() < 0.3
        if self.communication_window_open:
            self.signal_strength = random.uniform(0.5, 1.0)
        else:
            self.signal_strength = 0.0

    def advance_sol(self):
        self.current_sol += 1
        self.update_communication()


# =============================================================================
# 2. Neural Components
# =============================================================================

class TerrainCNN(NeuralComponent):
    """CNN that classifies traversability from local terrain patches."""
    def __init__(self, name: str, patch_size: int = 7, hidden_dim: int = 16):
        input_dim = patch_size * patch_size
        super().__init__(name, input_dim=input_dim, output_dim=1)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        # Simulated conv layers (simplified as linear projection)
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W2 = np.random.randn(1, hidden_dim) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        h = np.tanh(self.W1 @ x)
        out = self.W2 @ h
        return out  # predicted traversability

    def predict_traversability(self, patch: np.ndarray) -> float:
        pred = self.forward(patch.flatten())
        return float(np.clip(pred[0], 0, 1))


class ScienceInterestPredictor(NeuralComponent):
    """Predicts scientific interest of unknown terrain from spectral data."""
    def __init__(self, name: str, input_dim: int = 8, hidden_dim: int = 16):
        super().__init__(name, input_dim=input_dim, output_dim=1)
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W2 = np.random.randn(1, hidden_dim) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        h = np.tanh(self.W1 @ x)
        out = self.W2 @ h
        return out

    def predict_interest(self, spectral_features: np.ndarray) -> float:
        pred = self.forward(spectral_features)
        return float(np.clip(pred[0], 0, 1))


class ResourceAllocator(NeuralComponent):
    """Manages power and bandwidth allocation across tasks."""
    def __init__(self, name: str, n_tasks: int = 5, hidden_dim: int = 8):
        super().__init__(name, input_dim=n_tasks + 2, output_dim=n_tasks)  # tasks + power + bandwidth
        self.n_tasks = n_tasks
        self.W1 = np.random.randn(hidden_dim, self.input_dim) * 0.01
        self.W2 = np.random.randn(self.output_dim, hidden_dim) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        h = np.tanh(self.W1 @ x)
        out = self.W2 @ h
        return out  # allocation fractions (will be normalized)

    def allocate(self, task_urgencies: List[float], battery_level: float, bandwidth: float) -> List[float]:
        features = task_urgencies + [battery_level, bandwidth]
        raw = self.forward(np.array(features))
        # Softmax normalization
        exp_raw = np.exp(raw - np.max(raw))
        allocations = exp_raw / np.sum(exp_raw)
        return allocations.tolist()


# =============================================================================
# 3. Custom Embedding
# =============================================================================

class RoverStateEmbedding(EmbeddingModel):
    """Embedding of rover state: local map, battery, pending tasks, comm window."""
    def __init__(self, name: str = "rover_embedder", embedding_dim: int = 128, map_size: int = 7):
        super().__init__(name, embedding_dim=embedding_dim)
        self.map_size = map_size

    def embed(self, obj: COHObject) -> np.ndarray:
        rover = obj.get_attribute("rover")
        world = obj.get_attribute("world")
        if rover is None or world is None:
            return np.zeros(self.embedding_dim)
        # Get local terrain patch
        x, y = rover.position
        patch = world.get_terrain_patch(x, y, radius=self.map_size//2)
        # Flatten patch
        features = patch.flatten()
        # Add battery and pending tasks
        features = np.append(features, rover.battery_level / 100.0)
        features = np.append(features, len(rover.pending_tasks) / 10.0)
        features = np.append(features, 1.0 if world.communication_window_open else 0.0)
        # Pad or truncate to embedding_dim
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        else:
            features = features[:self.embedding_dim]
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        return features


# =============================================================================
# 4. Daemons
# =============================================================================

class PowerGuardian(ConstraintDaemon):
    """Forecasts energy usage and pre-emptively enters sleep mode."""
    def __init__(self, parent: COHObject, interval: float = 10.0):
        super().__init__(parent, interval)
        self.low_power_threshold = 25.0  # %

    def check(self) -> None:
        rover = self.parent.get_attribute("rover")
        if rover is None:
            return
        battery = rover.battery_level
        if battery < self.low_power_threshold:
            print(f"[PowerGuardian] Battery low ({battery:.1f}%). Entering low-power mode.")
            self.parent.execute_method("adjust_power_mode", "low_power")
        elif battery > 80 and rover.power_mode == "low_power":
            print("[PowerGuardian] Battery recovered. Resuming normal mode.")
            self.parent.execute_method("adjust_power_mode", "normal")


class HazardMonitor(ConstraintDaemon):
    """Detects steep slopes or loose soil, triggers re-planning."""
    def __init__(self, parent: COHObject, interval: float = 5.0):
        super().__init__(parent, interval)
        self.slope_threshold = 0.3  # traversability drop per step

    def check(self) -> None:
        rover = self.parent.get_attribute("rover")
        world = self.parent.get_attribute("world")
        if rover is None or world is None:
            return
        x, y = rover.position
        trav = world.get_traversability(x, y)
        if trav < 0.3:
            print(f"[HazardMonitor] Impassable terrain detected at ({x},{y})! Re-planning route.")
            self.parent.execute_method("replan_route")


# =============================================================================
# 5. Main Rover COH Object
# =============================================================================

class MarsRover(COHObject):
    """
    Autonomous Mars rover with navigation, science, power management, and planning.
    """
    def __init__(self, name: str = "MarsRover", world: Optional[MarsWorld] = None):
        super().__init__(name)
        self.world = world or MarsWorld()
        self.position = (self.world.width // 2, self.world.height // 2)
        self.battery_level = 100.0  # percent
        self.power_mode = "normal"  # normal, low_power, sleep
        self.data_buffer_mb = 0.0
        self.visited_targets: Set[str] = set()
        self.pending_tasks: List[Dict] = []
        self.science_collected: List[Dict] = []
        self.total_science_value = 0.0

        # ---- Attributes (A) ----
        self.add_attribute("world", self.world)
        self.add_attribute("rover", self)  # self-reference for daemons
        self.add_attribute("position", self.position)
        self.add_attribute("battery_level", self.battery_level)
        self.add_attribute("power_mode", self.power_mode)
        self.add_attribute("data_buffer_mb", self.data_buffer_mb)
        self.add_attribute("pending_tasks", self.pending_tasks)
        self.add_attribute("total_science_value", self.total_science_value)
        self.add_attribute("current_plan", [])

        # ---- Neural Components (N) ----
        self.terrain_cnn = TerrainCNN(name="terrain_cnn", patch_size=7)
        self.add_neural_component("terrain_cnn", self.terrain_cnn)
        self.science_predictor = ScienceInterestPredictor(name="science_predictor", input_dim=8)
        self.add_neural_component("science_predictor", self.science_predictor)
        self.resource_allocator = ResourceAllocator(name="resource_allocator", n_tasks=5)
        self.add_neural_component("resource_allocator", self.resource_allocator)

        # ---- Embedding (E) ----
        embedder = RoverStateEmbedding(name="rover_embedder", embedding_dim=128)
        self.add_neural_component("embedder", embedder, is_embedding_model=True)

        # ---- Methods (M) ----
        self.add_method("navigate_to", self.navigate_to)
        self.add_method("collect_sample", self.collect_sample)
        self.add_method("transmit_data", self.transmit_data)
        self.add_method("adjust_power_mode", self.adjust_power_mode)
        self.add_method("select_next_target", self.select_next_target)
        self.add_method("replan_route", self.replan_route)
        self.add_method("daily_cycle", self.daily_cycle)

        # ---- Identity Constraints (I) ----
        self.add_identity_constraint({
            'name': 'minimum_battery',
            'specification': 'battery_level >= 15',
            'severity': 10,
            'category': 'safety'
        })
        self.add_identity_constraint({
            'name': 'terrain_safety',
            'specification': 'world.get_traversability(position[0], position[1]) >= 0.3',
            'severity': 9
        })
        self.add_identity_constraint({
            'name': 'data_buffer_limit',
            'specification': 'data_buffer_mb <= 500',
            'severity': 8
        })

        # ---- Trigger Constraints (T) ----
        self.add_trigger_constraint({
            'name': 'low_battery_mode',
            'specification': 'WHEN battery_level < 25 DO adjust_power_mode("low_power")',
            'priority': 'HIGH'
        })
        self.add_trigger_constraint({
            'name': 'transmit_when_possible',
            'specification': 'WHEN world.communication_window_open == True AND data_buffer_mb > 50 DO transmit_data()',
            'priority': 'MEDIUM'
        })
        self.add_trigger_constraint({
            'name': 'new_target_replan',
            'specification': 'WHEN new_high_interest_target_detected DO replan_route()',
            'priority': 'HIGH'
        })

        # ---- Goal Constraints (G) ----
        self.add_goal_constraint({
            'name': 'maximize_science_return',
            'specification': 'MAXIMIZE total_science_value per sol',
            'priority': 'HIGH'
        })
        self.add_goal_constraint({
            'name': 'ensure_safety',
            'specification': 'MINIMIZE hazard_exposure',
            'priority': 'CRITICAL'
        })
        self.add_goal_constraint({
            'name': 'communication_compliance',
            'specification': 'MAXIMIZE data_transmitted / data_collected',
            'priority': 'MEDIUM'
        })

        # ---- Daemons (D) ----
        power_guardian = PowerGuardian(self, interval=10.0)
        self.daemons['power_guardian'] = power_guardian
        hazard_monitor = HazardMonitor(self, interval=5.0)
        self.daemons['hazard_monitor'] = hazard_monitor

        # Register reasoners
        self.constraint_system.register_reasoner("temporal", TemporalReasoner())
        self.constraint_system.register_reasoner("safety", SafetyReasoner())

        # Initialize plan
        self.replan_route()

    def navigate_to(self, target_x: int, target_y: int) -> bool:
        """Move rover toward target using A*-like heuristic (simplified)."""
        if self.power_mode == "sleep":
            print("Rover is sleeping. Cannot navigate.")
            return False
        x, y = self.position
        if x == target_x and y == target_y:
            return True
        # Simple step: move in direction of target
        dx = 0 if x == target_x else (1 if target_x > x else -1)
        dy = 0 if y == target_y else (1 if target_y > y else -1)
        new_x, new_y = x + dx, y + dy
        # Check traversability using neural predictor for unknown areas
        trav = self.world.get_traversability(new_x, new_y)
        if trav < 0.3:
            print(f"Path blocked at ({new_x},{new_y}) traversability={trav:.2f}")
            return False
        # Move
        self.position = (new_x, new_y)
        self.add_attribute("position", self.position)
        # Battery consumption: depends on terrain and power mode
        consumption = 0.5 / trav if trav > 0 else 1.0
        if self.power_mode == "low_power":
            consumption *= 0.5
        self.battery_level = max(0.0, self.battery_level - consumption)
        self.add_attribute("battery_level", self.battery_level)
        return True

    def collect_sample(self, target: ScienceTarget) -> float:
        """Collect a science sample, update data buffer and science value."""
        if self.power_mode == "sleep":
            print("Cannot collect sample: rover sleeping")
            return 0.0
        # Check if already collected
        if target.id in self.visited_targets:
            return 0.0
        # Energy cost
        energy_cost = target.estimated_time_min * 0.2
        if self.battery_level < energy_cost:
            print(f"Not enough battery to collect sample at {target.position}")
            return 0.0
        self.battery_level -= energy_cost
        self.add_attribute("battery_level", self.battery_level)
        # Data volume
        self.data_buffer_mb += target.data_volume_mb
        self.add_attribute("data_buffer_mb", self.data_buffer_mb)
        # Science value (interest * data volume)
        value = target.interest_score * target.data_volume_mb
        self.total_science_value += value
        self.add_attribute("total_science_value", self.total_science_value)
        self.visited_targets.add(target.id)
        self.science_collected.append({
            "target_id": target.id,
            "type": target.sample_type,
            "value": value,
            "sol": self.world.current_sol
        })
        print(f"Collected sample from {target.id}: value {value:.1f}")
        # Check for new high-interest target discovery (simulated)
        if target.interest_score > 0.8 and random.random() < 0.3:
            print("New high-interest target detected nearby!")
            self.add_attribute("new_high_interest_target_detected", True)
            self.execute_method("replan_route")
        return value

    def transmit_data(self) -> float:
        """Transmit data during communication window."""
        if not self.world.communication_window_open:
            print("No communication window")
            return 0.0
        # Bandwidth depends on signal strength
        bandwidth = self.world.signal_strength * 100  # Mbps
        # Transmit as much as possible
        transmitted = min(self.data_buffer_mb, bandwidth * 5)  # assume 5 sec window
        self.data_buffer_mb -= transmitted
        self.add_attribute("data_buffer_mb", self.data_buffer_mb)
        print(f"Transmitted {transmitted:.1f} MB. Remaining buffer: {self.data_buffer_mb:.1f} MB")
        return transmitted

    def adjust_power_mode(self, mode: str) -> None:
        """Change power mode."""
        valid_modes = ["normal", "low_power", "sleep"]
        if mode not in valid_modes:
            return
        self.power_mode = mode
        self.add_attribute("power_mode", mode)
        print(f"Power mode changed to {mode}")

    def select_next_target(self) -> Optional[ScienceTarget]:
        """Select the best science target based on interest, distance, and battery."""
        # Get all unvisited targets
        unvisited = [t for t in self.world.science_targets.values() if t.id not in self.visited_targets]
        if not unvisited:
            return None
        x, y = self.position
        # Score each target
        scored = []
        for target in unvisited:
            tx, ty = target.position
            dist = abs(tx - x) + abs(ty - y)
            # Energy estimate
            energy_needed = dist * 0.5 + target.estimated_time_min * 0.2
            if energy_needed > self.battery_level * 0.8:  # reserve 20%
                continue
            score = target.interest_score * 100 - dist * 2
            scored.append((score, target))
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][1]

    def replan_route(self) -> None:
        """Re-plan the route to prioritize high-value targets."""
        print("Re-planning route...")
        # Simple plan: go to the highest value target reachable
        next_target = self.select_next_target()
        if next_target:
            self.add_attribute("current_plan", [next_target])
            print(f"New plan: go to target {next_target.id} at {next_target.position}")
        else:
            # Explore unknown areas
            print("No targets left. Entering exploration mode.")
            self.add_attribute("current_plan", [])

    def daily_cycle(self) -> None:
        """Execute one sol's activities."""
        self.world.advance_sol()
        print(f"\n=== Sol {self.world.current_sol} ===")
        print(f"Position: {self.position}, Battery: {self.battery_level:.1f}%, Buffer: {self.data_buffer_mb:.1f} MB")

        # Follow current plan
        plan = self.get_attribute("current_plan")
        if plan:
            target = plan[0]
            # Navigate toward target
            tx, ty = target.position
            if self.position != (tx, ty):
                self.execute_method("navigate_to", tx, ty)
            else:
                # At target, collect sample
                self.execute_method("collect_sample", target)
                # Remove from plan
                self.add_attribute("current_plan", [])
                # Select next target
                next_t = self.select_next_target()
                if next_t:
                    self.add_attribute("current_plan", [next_t])
        else:
            # Exploration mode: move to a random nearby cell with good traversability
            x, y = self.position
            best_move = None
            best_trav = -1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                        trav = self.world.get_traversability(nx, ny)
                        if trav > best_trav:
                            best_trav = trav
                            best_move = (nx, ny)
            if best_move:
                self.execute_method("navigate_to", best_move[0], best_move[1])

        # Transmit if window open
        if self.world.communication_window_open:
            self.execute_method("transmit_data")

        # Recharge a bit from solar (simulated)
        solar_gain = 5.0 if self.power_mode != "sleep" else 2.0
        self.battery_level = min(100.0, self.battery_level + solar_gain)
        self.add_attribute("battery_level", self.battery_level)

        print(f"End of sol: Battery {self.battery_level:.1f}%, Science value {self.total_science_value:.1f}")


# =============================================================================
# 6. Simulation
# =============================================================================

def run_simulation(sols: int = 10):
    """Run rover simulation for given number of sols."""
    world = MarsWorld(width=15, height=15)
    rover = MarsRover("Curiosity", world)

    rover.initialize_system()
    rover.start_daemons()

    print("=== Autonomous Mars Rover Simulation ===")
    for sol in range(sols):
        rover.execute_method("daily_cycle")
        time.sleep(0.5)  # slow for readability

    rover.stop_daemons()
    print("\n=== Mission Summary ===")
    print(f"Total science value collected: {rover.total_science_value:.1f}")
    print(f"Samples collected: {len(rover.science_collected)}")
    print(f"Final battery: {rover.battery_level:.1f}%")
    print(f"Final data buffer: {rover.data_buffer_mb:.1f} MB")


if __name__ == "__main__":
    run_simulation(sols=10)