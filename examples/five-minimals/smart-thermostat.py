#!/usr/bin/env python3
"""
Smart Thermostat - COH Implementation using GISMOL Toolkit

This example models a home energy management thermostat as a Constrained Object Hierarchy (COH).
It learns occupancy patterns, balances comfort vs. energy, and enforces safety constraints.
"""

import time
import random
import numpy as np
from typing import Dict, Any

# Import GISMOL components
from gismol.core import COHObject, COHRepository, ConstraintSystem
from gismol.core.daemons import ConstraintDaemon
from gismol.neural import Regressor  # Simple neural predictor
from gismol.neural.embeddings import EmbeddingModel
from gismol.reasoners import TemporalReasoner, SafetyReasoner


# =============================================================================
# 1. Custom Neural Component: Temperature Predictor
# =============================================================================
class TemperaturePredictor(Regressor):
    """
    Simple neural network that predicts the next hour's temperature
    based on current temp, target, time of day, and occupancy history.
    """
    def __init__(self, name: str, input_dim: int = 4, hidden_dim: int = 8, **kwargs):
        super().__init__(name, input_dim, output_dim=1, hidden_dim=hidden_dim, **kwargs)
        # Simulate learned weights (in real system, would train on historical data)
        self._weights = np.random.randn(self.output_dim, hidden_dim) @ np.random.randn(hidden_dim, input_dim) * 0.1

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        # Two-layer network simulation
        h = np.maximum(0, self._W1 @ x)  # ReLU
        out = self._W2 @ h
        return out

    def predict_next_temp(self, current_temp: float, target_temp: float, hour: int, occupancy: float) -> float:
        """Return predicted temperature in one hour."""
        features = np.array([current_temp, target_temp, hour / 24.0, occupancy])
        pred = self.forward(features)[0]
        return float(np.clip(pred, current_temp - 5, current_temp + 5))


# =============================================================================
# 2. Custom Embedding Model (minimal)
# =============================================================================
class ThermostatEmbedding(EmbeddingModel):
    """Embedding for thermostat state: (temp, target, hour) normalized."""
    def embed(self, obj: COHObject) -> np.ndarray:
        context = obj.get_context()
        return np.array([
            context.get('current_temp', 20) / 40.0,
            context.get('target_temp', 20) / 40.0,
            (context.get('hour', 12) % 24) / 24.0
        ])


# =============================================================================
# 3. Custom Daemon: Overshoot Monitor
# =============================================================================
class OvershootMonitor(ConstraintDaemon):
    """Logs if temperature exceeds target by >2°C for more than 10 minutes."""
    def __init__(self, parent: COHObject, interval: float = 60.0):
        super().__init__(parent, interval)
        self.violation_start = None

    def check(self) -> None:
        current = self.parent.get_attribute('current_temp', 20)
        target = self.parent.get_attribute('target_temp', 20)
        if current > target + 2.0:
            if self.violation_start is None:
                self.violation_start = time.time()
            elif time.time() - self.violation_start > 600:  # 10 minutes
                print(f"[DAEMON] Overshoot detected: {current:.1f}°C > target+2 for >10 min")
                # Could trigger an alert or adjust PID parameters
        else:
            self.violation_start = None


# =============================================================================
# 4. Main Thermostat System Construction
# =============================================================================
def create_smart_thermostat(name: str = "SmartThermostat") -> COHObject:
    """Build and return a fully configured COH thermostat object."""
    thermostat = COHObject(name=name)

    # ---- Attributes (A) ----
    thermostat.add_attribute("current_temp", 20.0)      # °C
    thermostat.add_attribute("target_temp", 21.0)       # °C
    thermostat.add_attribute("heater_on", 0)            # 0/1
    thermostat.add_attribute("energy_used", 0.0)        # kWh
    thermostat.add_attribute("hour", 0)                 # 0-23
    thermostat.add_attribute("occupancy", 0.0)          # 0 (empty) to 1 (full)

    # ---- Methods (M) ----
    def measure_temp(self) -> float:
        """Simulate temperature measurement with slight noise and external effects."""
        # Simple simulation: temperature drifts based on heater, outside temp, etc.
        current = self.get_attribute('current_temp')
        heater = self.get_attribute('heater_on')
        outside = 5 + 10 * np.sin(np.pi * (self.get_attribute('hour') - 14) / 12)  # day/night cycle
        # Thermal dynamics: heater adds heat, outside cools/warms
        delta = (heater * 2.0) - 0.1 * (current - outside) + random.gauss(0, 0.2)
        new_temp = current + delta * 0.1  # 6 min step (simplified)
        new_temp = max(5, min(35, new_temp))  # enforce absolute bounds
        self.add_attribute('current_temp', new_temp)
        return new_temp

    def set_heater(self, state: int) -> None:
        """Turn heater on (1) or off (0)."""
        if state not in (0, 1):
            raise ValueError("Heater state must be 0 or 1")
        self.add_attribute("heater_on", state)
        if state:
            # Increment energy usage: assume 2 kW when on, each call = 0.1 hour step
            energy = self.get_attribute('energy_used', 0.0)
            self.add_attribute('energy_used', energy + 0.2)  # 2 kW * 0.1 h = 0.2 kWh

    def adjust_target(self, delta: float) -> None:
        """Change target temperature by delta degrees."""
        new_target = self.get_attribute('target_temp', 21) + delta
        new_target = max(15, min(30, new_target))  # reasonable bounds
        self.add_attribute('target_temp', new_target)

    thermostat.add_method("measure_temp", measure_temp)
    thermostat.add_method("set_heater", set_heater)
    thermostat.add_method("adjust_target", adjust_target)

    # ---- Neural Components (N) ----
    predictor = TemperaturePredictor(name="temp_predictor", input_dim=4, hidden_dim=8)
    thermostat.add_neural_component("predictor", predictor)

    # ---- Embedding (E) ----
    embedder = ThermostatEmbedding(name="thermostat_embedder", embedding_dim=3)
    thermostat.add_neural_component("embedder", embedder, is_embedding_model=True)

    # ---- Identity Constraints (I) ----
    thermostat.add_identity_constraint({
        'name': 'freeze_protection',
        'specification': 'current_temp >= 5',
        'severity': 10,
        'category': 'safety'
    })
    thermostat.add_identity_constraint({
        'name': 'heater_state_binary',
        'specification': 'heater_on == 0 or heater_on == 1',
        'severity': 5
    })

    # ---- Trigger Constraints (T) ----
    # Rule: If current temp < target - 0.5, turn heater ON
    thermostat.add_trigger_constraint({
        'name': 'heat_on_rule',
        'specification': 'WHEN current_temp < target_temp - 0.5 DO set_heater(1)',
        'priority': 'HIGH'
    })
    # Rule: If current temp > target + 0.5, turn heater OFF
    thermostat.add_trigger_constraint({
        'name': 'heat_off_rule',
        'specification': 'WHEN current_temp > target_temp + 0.5 DO set_heater(0)',
        'priority': 'HIGH'
    })
    # Rule: If occupancy is low, allow target to drift (energy saving)
    thermostat.add_trigger_constraint({
        'name': 'eco_mode',
        'specification': 'WHEN occupancy < 0.2 DO adjust_target(-1)',
        'priority': 'MEDIUM'
    })

    # ---- Goal Constraints (G) ----
    # Minimize weighted sum of comfort error and energy consumption
    # (The actual optimization is performed by the policy; constraints express the objective)
    thermostat.add_goal_constraint({
        'name': 'comfort_energy_balance',
        'specification': 'MINIMIZE (current_temp - target_temp)^2 + 0.01 * energy_used',
        'priority': 'HIGH'
    })

    # ---- Daemons (D) ----
    overshoot_monitor = OvershootMonitor(thermostat, interval=60.0)
    thermostat.daemons['overshoot'] = overshoot_monitor

    # Register reasoners (optional, for constraint evaluation)
    thermostat.constraint_system.register_reasoner("temporal", TemporalReasoner())
    thermostat.constraint_system.register_reasoner("safety", SafetyReasoner())

    return thermostat


# =============================================================================
# 5. Simulation Loop
# =============================================================================
def simulate_day(thermostat: COHObject, steps_per_hour: int = 6):
    """
    Run a 24-hour simulation.
    Each step = 60 / steps_per_hour minutes.
    """
    print("=== Smart Thermostat Simulation ===")
    print("Time(h) | Temp(°C) | Target | Heater | Energy(kWh) | Occupancy")
    print("-" * 60)

    # Start daemons
    thermostat.start_daemons()

    # Define a simple occupancy schedule: home from 7-9 and 17-22
    def get_occupancy(hour: int) -> float:
        if 7 <= hour <= 9 or 17 <= hour <= 22:
            return 1.0
        elif 22 <= hour <= 23 or 0 <= hour <= 6:
            return 0.1
        else:
            return 0.3

    # Main simulation loop
    for hour in range(24):
        thermostat.add_attribute('hour', hour)
        occ = get_occupancy(hour)
        thermostat.add_attribute('occupancy', occ)

        # Each hour is divided into steps
        for step in range(steps_per_hour):
            # 1. Measure current temperature (simulates physical process)
            thermostat.execute_method("measure_temp")

            # 2. Evaluate triggers (they will adjust heater and possibly target)
            # In GISMOL, triggers are evaluated automatically when methods are called
            # or can be manually triggered. We'll explicitly check trigger constraints.
            # For simplicity, we call a helper that evaluates all trigger constraints.
            # (In a real system, daemons would handle this.)
            for constraint in thermostat.trigger_constraints:
                context = thermostat.get_context()
                if not thermostat.constraint_system.validate_single(constraint, context):
                    # Violation would raise, but we assume triggers are satisfied
                    pass

            # 3. (Optional) Use neural predictor to anticipate next hour's need
            if step == steps_per_hour - 1:  # last step of the hour
                pred = thermostat.get_neural_component("predictor").predict_next_temp(
                    current_temp=thermostat.get_attribute('current_temp'),
                    target_temp=thermostat.get_attribute('target_temp'),
                    hour=hour,
                    occupancy=occ
                )
                # If prediction shows upcoming cold, pre-heat slightly
                if pred < thermostat.get_attribute('target_temp') - 1.0:
                    print(f"[NN] Pre-heating activated for hour {hour+1}: predicted {pred:.1f}°C")
                    thermostat.execute_method("set_heater", 1)

            # 4. Log current state
            if step == 0:  # print once per hour
                print(f"{hour:2d}     | {thermostat.get_attribute('current_temp'):5.1f}  | "
                      f"{thermostat.get_attribute('target_temp'):5.1f}   | "
                      f"{thermostat.get_attribute('heater_on'):2d}     | "
                      f"{thermostat.get_attribute('energy_used'):7.2f} | {occ:5.2f}")

            # Simulate real time (optional)
            time.sleep(0.05)  # small delay for realism

    # Stop daemons
    thermostat.stop_daemons()

    print("\n=== Simulation Complete ===")
    print(f"Total energy used: {thermostat.get_attribute('energy_used'):.2f} kWh")


# =============================================================================
# 6. Main Entry Point
# =============================================================================
if __name__ == "__main__":
    # Create the thermostat COH object
    thermostat = create_smart_thermostat()

    # Initialize system (validates constraints)
    try:
        thermostat.initialize_system()
        print("Thermostat system initialized successfully.\n")
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit(1)

    # Run simulation
    simulate_day(thermostat, steps_per_hour=6)  # 10 minutes per step

    # Example of semantic query using embedding
    embedder = thermostat.embedding_model
    if embedder:
        emb = embedder.embed(thermostat)
        print(f"\nEmbedding vector (first 3 dims): {emb[:3]}")