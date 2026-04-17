#!/usr/bin/env python3
"""
Adaptive Cruise Control - COH Implementation using GISMOL Toolkit

This example models an adaptive cruise control system as a Constrained Object Hierarchy (COH).
It maintains a safe following distance, predicts lead vehicle behavior with an LSTM,
and enforces safety constraints via triggers and a daemon.
"""

import time
import math
import random
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque

# Import GISMOL components
from gismol.core import COHObject
from gismol.core.daemons import ConstraintDaemon
from gismol.neural import NeuralComponent
from gismol.neural.embeddings import EmbeddingModel


# =============================================================================
# 1. Simulated Environment: Lead Vehicle and Road
# =============================================================================
class LeadVehicle:
    """Simulates a lead vehicle with realistic acceleration and braking."""
    def __init__(self, initial_speed: float = 20.0):  # m/s (~72 km/h)
        self.speed = initial_speed
        self.acceleration = 0.0
        self.position = 0.0  # relative to ego? We'll use absolute for simplicity.
        self.history = deque(maxlen=100)

    def update(self, dt: float, driver_aggression: float = 0.5):
        """
        Update lead vehicle state based on a simple driver model.
        driver_aggression: 0 (gentle) to 1 (aggressive braking/acceleration).
        """
        # Simple model: occasionally brake or accelerate
        if random.random() < 0.02 * driver_aggression:
            # Sudden braking: deceleration between -2 and -5 m/s²
            self.acceleration = -random.uniform(2.0, 5.0)
        elif random.random() < 0.01:
            # Gentle acceleration
            self.acceleration = random.uniform(0.5, 2.0)
        else:
            # Maintain speed with small noise
            self.acceleration = random.gauss(0, 0.1)

        # Limit acceleration to realistic bounds
        self.acceleration = max(-6.0, min(3.0, self.acceleration))

        # Update speed and position
        self.speed += self.acceleration * dt
        self.speed = max(0.0, min(40.0, self.speed))  # 0 to 144 km/h
        self.position += self.speed * dt
        self.history.append((self.speed, self.acceleration))

    def get_state(self) -> np.ndarray:
        """Return recent speed history for neural input (last 20 steps)."""
        speeds = [s for s, _ in self.history]
        if len(speeds) < 20:
            speeds = [speeds[0]] * (20 - len(speeds)) + speeds
        return np.array(speeds[-20:]) / 40.0  # normalize to [0,1]


# =============================================================================
# 2. Neural Component: Lead Vehicle Predictor (LSTM simulation)
# =============================================================================
class LeadVehiclePredictor(NeuralComponent):
    """
    Predicts the lead vehicle's speed and acceleration over the next 2 seconds.
    Simulates an LSTM with a simple feedforward network.
    """
    def __init__(self, name: str, input_dim: int = 20, hidden_dim: int = 32, output_dim: int = 2):
        super().__init__(name, input_dim=input_dim, output_dim=output_dim)
        self.hidden_dim = hidden_dim
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Returns (predicted_speed, predicted_acceleration) normalized."""
        x = np.asarray(x).flatten()
        h = np.tanh(self.W1 @ x + self.b1)
        out = self.W2 @ h + self.b2
        return out  # speed in [0,1], acceleration in [-0.15,0.075] (scaled)

    def predict(self, lead_state: np.ndarray) -> Tuple[float, float]:
        """Return predicted speed (m/s) and acceleration (m/s²)."""
        pred = self.forward(lead_state)
        speed = pred[0] * 40.0  # unnormalize
        acc = pred[1] * 6.0 - 3.0  # map from [-0.15,0.075] to [-3,1.5] approx
        return speed, acc


# =============================================================================
# 3. Custom Embedding Model: Road and Traffic Context
# =============================================================================
class ACCEmbedding(EmbeddingModel):
    """Embedding of current state: (ego_speed, lead_distance, lead_speed, road_curvature)."""
    def __init__(self, name: str = "acc_embedder", embedding_dim: int = 32):
        super().__init__(name, embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim

    def embed(self, obj: COHObject) -> np.ndarray:
        ego_speed = obj.get_attribute("ego_speed", 0.0)
        lead_distance = obj.get_attribute("lead_distance", 100.0)
        lead_speed = obj.get_attribute("lead_speed", 0.0)
        road_curvature = obj.get_attribute("road_curvature", 0.0)
        features = np.array([ego_speed / 40.0, lead_distance / 100.0, lead_speed / 40.0, road_curvature])
        # Random projection to embedding dimension
        proj = np.random.randn(self.embedding_dim, 4) * 0.1
        embedding = proj @ features
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding


# =============================================================================
# 4. Custom Daemon: Safety Enforcer
# =============================================================================
class SafetyEnforcer(ConstraintDaemon):
    """
    Monitors the required deceleration and overrides throttle if tire friction limit
    would be exceeded. Also ensures that the time gap never falls below a hard minimum.
    """
    def __init__(self, parent: COHObject, interval: float = 0.05, max_decel: float = 7.0):
        super().__init__(parent, interval)
        self.max_decel = max_decel  # m/s² (tire friction limit)
        self.min_time_gap = 0.8     # seconds

    def check(self) -> None:
        ego_speed = self.parent.get_attribute("ego_speed", 0.0)
        lead_distance = self.parent.get_attribute("lead_distance", 100.0)
        lead_speed = self.parent.get_attribute("lead_speed", 0.0)
        time_gap = lead_distance / (ego_speed + 0.01) if ego_speed > 0.1 else 10.0

        # Hard safety: if time gap below absolute minimum, force emergency brake
        if time_gap < self.min_time_gap:
            print(f"[SafetyEnforcer] Time gap {time_gap:.2f}s < {self.min_time_gap}s! Forcing emergency brake.")
            self.parent.execute_method("emergency_brake")
            return

        # Predict required deceleration to avoid collision in the next 2 seconds
        if lead_distance < 30.0 and ego_speed > lead_speed:
            required_decel = (ego_speed - lead_speed) / 0.5  # simple: stop in 0.5s
            if required_decel > self.max_decel:
                print(f"[SafetyEnforcer] Required decel {required_decel:.1f} m/s² exceeds limit! Overriding.")
                # Override throttle: apply max brake
                self.parent.execute_method("apply_brake", 100)
            else:
                # Allow normal control
                pass


# =============================================================================
# 5. Main Adaptive Cruise Control COH Object
# =============================================================================
class AdaptiveCruiseControl(COHObject):
    """
    Adaptive Cruise Control system with radar, speed controller, brake actuator,
    and lead vehicle tracking.
    """
    def __init__(self, name: str = "ACCSystem", lead_vehicle: Optional[LeadVehicle] = None):
        super().__init__(name)
        self.lead = lead_vehicle or LeadVehicle()

        # ---- Attributes (A) ----
        self.add_attribute("ego_speed", 20.0)           # m/s
        self.add_attribute("lead_distance", 50.0)       # meters
        self.add_attribute("lead_speed", 20.0)          # m/s
        self.add_attribute("time_gap", 1.5)             # seconds (desired)
        self.add_attribute("brake_pressure", 0.0)       # percent (0-100)
        self.add_attribute("throttle", 0.0)             # percent
        self.add_attribute("desired_speed", 25.0)       # m/s (set by driver)
        self.add_attribute("road_curvature", 0.0)       # rad/m
        self.add_attribute("lead_vehicle_ref", self.lead)

        # ---- Neural Components (N) ----
        self.predictor = LeadVehiclePredictor(name="lead_predictor", input_dim=20, hidden_dim=32)
        self.add_neural_component("lead_predictor", self.predictor)

        # ---- Embedding (E) ----
        embedder = ACCEmbedding(name="acc_embedder", embedding_dim=32)
        self.add_neural_component("embedder", embedder, is_embedding_model=True)

        # ---- Methods (M) ----
        self.add_method("update_radar", self.update_radar)
        self.add_method("compute_safe_gap", self.compute_safe_gap)
        self.add_method("set_throttle", self.set_throttle)
        self.add_method("apply_brake", self.apply_brake)
        self.add_method("emergency_brake", self.emergency_brake)
        self.add_method("control_loop", self.control_loop)

        # ---- Identity Constraints (I) ----
        self.add_identity_constraint({
            'name': 'brake_pressure_bounds',
            'specification': 'brake_pressure >= 0 and brake_pressure <= 100',
            'severity': 10
        })
        self.add_identity_constraint({
            'name': 'absolute_min_distance',
            'specification': 'lead_distance >= 2.5',
            'severity': 10,
            'category': 'safety'
        })

        # ---- Trigger Constraints (T) ----
        # Time gap too low -> increase brake pressure linearly
        self.add_trigger_constraint({
            'name': 'time_gap_control',
            'specification': 'WHEN time_gap < 1.0 DO increase_brake_pressure(linear)',
            'priority': 'HIGH'
        })
        # Emergency braking if distance < 5m and ego speed > lead speed
        self.add_trigger_constraint({
            'name': 'emergency_trigger',
            'specification': 'WHEN lead_distance < 5 AND ego_speed > lead_speed DO emergency_brake()',
            'priority': 'CRITICAL'
        })

        # ---- Goal Constraints (G) ----
        self.add_goal_constraint({
            'name': 'comfort_safety',
            'specification': 'MINIMIZE (desired_speed - ego_speed)^2 + 0.1 * jerk^2 + 1000 * collision_risk',
            'priority': 'HIGH'
        })

        # ---- Daemons (D) ----
        safety_daemon = SafetyEnforcer(self, interval=0.05, max_decel=7.0)
        self.daemons['safety_enforcer'] = safety_daemon

    # ---- Method implementations ----
    def update_radar(self, dt: float = 0.1) -> None:
        """Simulate radar measurement: update lead distance and speed."""
        # Update lead vehicle state
        self.lead.update(dt)

        # Get current lead speed
        lead_speed = self.lead.speed
        self.add_attribute("lead_speed", lead_speed)

        # Update relative distance based on ego and lead speeds
        ego_speed = self.get_attribute("ego_speed")
        distance = self.get_attribute("lead_distance")
        new_distance = distance + (lead_speed - ego_speed) * dt
        new_distance = max(0.0, new_distance)
        self.add_attribute("lead_distance", new_distance)

        # Use neural predictor to forecast lead vehicle motion (for planning)
        lead_state = self.lead.get_state()
        pred_speed, pred_acc = self.predictor.predict(lead_state)
        self.add_attribute("predicted_lead_speed", pred_speed)
        self.add_attribute("predicted_lead_accel", pred_acc)

        # Compute time gap
        if ego_speed > 0.5:
            time_gap = new_distance / ego_speed
        else:
            time_gap = 10.0
        self.add_attribute("time_gap", time_gap)

    def compute_safe_gap(self) -> float:
        """Return desired following distance based on speed (2-second rule)."""
        ego_speed = self.get_attribute("ego_speed")
        # Minimum 5 meters plus 2 seconds of travel
        safe_gap = 5.0 + ego_speed * 2.0
        return safe_gap

    def set_throttle(self, percent: float) -> None:
        """Set throttle (0-100)."""
        percent = max(0.0, min(100.0, percent))
        self.add_attribute("throttle", percent)
        # Update ego speed based on throttle and drag
        ego_speed = self.get_attribute("ego_speed")
        # Simple dynamics: acceleration = (throttle/100)*5 - 0.1*ego_speed (drag)
        acc = (percent / 100.0) * 4.0 - 0.05 * ego_speed
        dt = 0.1  # assume called every 0.1s
        new_speed = ego_speed + acc * dt
        new_speed = max(0.0, min(40.0, new_speed))
        self.add_attribute("ego_speed", new_speed)

    def apply_brake(self, pressure: float) -> None:
        """Apply brake pressure (0-100)."""
        pressure = max(0.0, min(100.0, pressure))
        self.add_attribute("brake_pressure", pressure)
        # Deceleration: linear up to 8 m/s² at 100% pressure
        decel = (pressure / 100.0) * 8.0
        ego_speed = self.get_attribute("ego_speed")
        dt = 0.1
        new_speed = max(0.0, ego_speed - decel * dt)
        self.add_attribute("ego_speed", new_speed)
        # Clear throttle when braking
        if pressure > 5:
            self.add_attribute("throttle", 0.0)

    def emergency_brake(self) -> None:
        """Full braking."""
        print("[ACC] Emergency brake activated!")
        self.apply_brake(100.0)

    def control_loop(self) -> None:
        """
        Main PID-like control to maintain safe following distance.
        Called periodically.
        """
        ego_speed = self.get_attribute("ego_speed")
        lead_distance = self.get_attribute("lead_distance")
        lead_speed = self.get_attribute("lead_speed")
        desired_speed = self.get_attribute("desired_speed")
        safe_gap = self.compute_safe_gap()

        # If no lead vehicle within 150m, use desired speed control
        if lead_distance > 150.0:
            # Speed control
            error = desired_speed - ego_speed
            throttle = max(0.0, min(100.0, error * 10.0))
            self.set_throttle(throttle)
            if throttle < 1:
                self.apply_brake(0)
            return

        # Following distance control
        gap_error = lead_distance - safe_gap
        relative_speed = ego_speed - lead_speed

        # Simple PD control on gap
        # Throttle when gap too large and relative speed low
        if gap_error > 2.0 and relative_speed < 0.5:
            # Need to accelerate
            throttle = min(80.0, max(0.0, gap_error * 5.0 - relative_speed * 2.0))
            self.set_throttle(throttle)
            self.apply_brake(0)
        elif gap_error < -0.5 or relative_speed > 0.5:
            # Need to brake
            brake = min(100.0, max(0.0, (-gap_error) * 10.0 + relative_speed * 5.0))
            self.apply_brake(brake)
            self.set_throttle(0)
        else:
            # Maintain speed
            self.set_throttle(10.0)  # small throttle to overcome drag
            self.apply_brake(0)

        # Also incorporate neural prediction for smoother response
        pred_lead_speed = self.get_attribute("predicted_lead_speed", lead_speed)
        if pred_lead_speed < lead_speed - 1.0:
            # Lead vehicle predicted to brake soon, preemptively brake
            self.apply_brake(15.0)


# =============================================================================
# 6. Simulation Loop
# =============================================================================
def simulate_acc(duration: float = 60.0, dt: float = 0.1):
    """
    Run the ACC system for a given duration (seconds).
    """
    # Create lead vehicle and ACC system
    lead = LeadVehicle(initial_speed=20.0)
    acc = AdaptiveCruiseControl("MyACC", lead_vehicle=lead)

    # Initialize
    acc.initialize_system()
    acc.start_daemons()

    # Set initial distance
    acc.add_attribute("lead_distance", 40.0)
    acc.add_attribute("ego_speed", 20.0)
    acc.add_attribute("desired_speed", 25.0)

    print("=== Adaptive Cruise Control Simulation ===")
    print("Time(s) | Ego Spd | Lead Spd | Distance | Time Gap | Throttle | Brake")
    print("-" * 70)

    t = 0.0
    try:
        while t <= duration:
            # Update radar and lead vehicle
            acc.execute_method("update_radar", dt)

            # Run control loop
            acc.execute_method("control_loop")

            # Log every 0.5 seconds
            if abs(t - round(t * 10) / 10) < dt/2:
                ego = acc.get_attribute("ego_speed")
                lead_spd = acc.get_attribute("lead_speed")
                dist = acc.get_attribute("lead_distance")
                tgap = acc.get_attribute("time_gap")
                throttle = acc.get_attribute("throttle")
                brake = acc.get_attribute("brake_pressure")
                print(f"{t:6.1f} | {ego:6.2f} | {lead_spd:6.2f} | {dist:6.2f} | {tgap:6.2f} | {throttle:6.1f} | {brake:5.1f}")

            t += dt
            time.sleep(dt * 0.1)  # speed up simulation

    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
    finally:
        acc.stop_daemons()
        print("\n=== Simulation Complete ===")
        print(f"Final ego speed: {acc.get_attribute('ego_speed'):.2f} m/s")
        print(f"Final distance: {acc.get_attribute('lead_distance'):.2f} m")


# =============================================================================
# 7. Main Entry Point
# =============================================================================
if __name__ == "__main__":
    simulate_acc(duration=60.0, dt=0.1)