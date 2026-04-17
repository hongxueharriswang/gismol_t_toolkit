#!/usr/bin/env python3
"""
Predictive Maintenance for CNC Machining - COH Implementation using GISMOL Toolkit

This example models a CNC machining cell as a Constrained Object Hierarchy (COH).
It monitors tool wear, predicts failures, schedules maintenance, and adapts parameters.
"""

import time
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
from dataclasses import dataclass, field

# Import GISMOL components
from gismol.core import COHObject
from gismol.core.daemons import ConstraintDaemon
from gismol.neural import NeuralComponent, Regressor, AnomalyDetector
from gismol.neural.embeddings import EmbeddingModel
from gismol.reasoners import BaseReasoner, TemporalReasoner, SafetyReasoner


# =============================================================================
# 1. Simulated Domain Models
# =============================================================================

@dataclass
class Tool:
    """Cutting tool with properties."""
    tool_id: str
    type: str  # "end_mill", "drill", "insert"
    initial_life_min: float  # expected lifetime in minutes
    current_life_min: float
    wear_rate_factor: float = 1.0  # multiplier (higher = faster wear)
    is_installed: bool = True

@dataclass
class CNCMachine:
    """CNC machine with sensors and operational parameters."""
    machine_id: str
    model: str
    spindle_power_max: float  # kW
    feed_rate_max: float  # mm/min
    cutting_speed_max: float  # m/min
    current_tool: Optional[Tool] = None
    spindle_power: float = 0.0
    feed_rate: float = 0.0
    cutting_speed: float = 0.0
    vibration_level: float = 0.0  # normalized 0-1
    temperature: float = 25.0  # °C
    is_running: bool = False
    maintenance_due: bool = False

class ManufacturingWorld:
    """Simulated manufacturing environment with CNC machines, tools, and production."""
    def __init__(self):
        # Tools
        self.tools = {
            "T001": Tool("T001", "end_mill", initial_life_min=120, current_life_min=120, wear_rate_factor=1.0),
            "T002": Tool("T002", "end_mill", initial_life_min=120, current_life_min=80, wear_rate_factor=1.2),
            "T003": Tool("T003", "drill", initial_life_min=90, current_life_min=90, wear_rate_factor=1.0),
            "T004": Tool("T004", "insert", initial_life_min=200, current_life_min=200, wear_rate_factor=1.0),
            "T005": Tool("T005", "insert", initial_life_min=200, current_life_min=150, wear_rate_factor=1.1),
        }
        # CNC Machines
        self.machines = {
            "CNC1": CNCMachine("CNC1", "Haas VF-2", spindle_power_max=22.4, feed_rate_max=10000, cutting_speed_max=300),
            "CNC2": CNCMachine("CNC2", "DMG Mori", spindle_power_max=30.0, feed_rate_max=12000, cutting_speed_max=400),
        }
        # Assign tools to machines
        self.machines["CNC1"].current_tool = self.tools["T001"]
        self.machines["CNC2"].current_tool = self.tools["T003"]
        self.current_time_min = 0.0
        self.production_target = 100  # parts per day
        self.parts_produced = 0

    def run_machine(self, machine_id: str, duration_min: float) -> Dict:
        """Simulate machine operation, update tool wear, and return metrics."""
        machine = self.machines[machine_id]
        if not machine.is_running or machine.maintenance_due:
            return {"error": "machine not ready"}
        tool = machine.current_tool
        if tool is None:
            return {"error": "no tool installed"}

        # Simulate wear: wear rate depends on cutting parameters and tool type
        wear_increment = duration_min * tool.wear_rate_factor * (machine.feed_rate / machine.feed_rate_max) * (machine.cutting_speed / machine.cutting_speed_max)
        tool.current_life_min -= wear_increment
        # Simulate vibration: increases with wear
        wear_ratio = 1 - (tool.current_life_min / tool.initial_life_min)
        machine.vibration_level = 0.1 + wear_ratio * 0.8 + random.gauss(0, 0.05)
        machine.vibration_level = min(1.0, max(0.0, machine.vibration_level))
        # Simulate temperature rise
        machine.temperature += wear_increment * 0.5 + random.gauss(0, 0.2)
        machine.temperature = min(60, machine.temperature)
        # Power consumption
        machine.spindle_power = (machine.cutting_speed / machine.cutting_speed_max) * machine.spindle_power_max * (0.5 + wear_ratio * 0.5)
        # Update production
        parts_this_run = int(duration_min / 2)  # 2 min per part
        self.parts_produced += parts_this_run
        self.current_time_min += duration_min
        return {
            "wear_remaining": tool.current_life_min,
            "vibration": machine.vibration_level,
            "temperature": machine.temperature,
            "spindle_power": machine.spindle_power,
            "parts_produced": parts_this_run
        }

    def adjust_parameters(self, machine_id: str, feed_rate: float = None, cutting_speed: float = None) -> None:
        """Adjust cutting parameters within constraints."""
        machine = self.machines[machine_id]
        if feed_rate is not None:
            machine.feed_rate = max(0, min(machine.feed_rate_max, feed_rate))
        if cutting_speed is not None:
            machine.cutting_speed = max(0, min(machine.cutting_speed_max, cutting_speed))

    def schedule_maintenance(self, machine_id: str, tool_id: str = None) -> None:
        """Schedule tool change or machine maintenance."""
        machine = self.machines[machine_id]
        if tool_id:
            # Replace tool
            new_tool = self.tools.get(tool_id)
            if new_tool and new_tool.current_life_min > 0:
                machine.current_tool = new_tool
                print(f"[Maintenance] Tool {tool_id} installed on {machine_id}")
        else:
            # General maintenance
            machine.maintenance_due = False
            print(f"[Maintenance] Machine {machine_id} serviced")
        machine.is_running = True  # resume after maintenance

    def emergency_stop(self, machine_id: str) -> None:
        """Emergency stop the machine."""
        machine = self.machines[machine_id]
        machine.is_running = False
        print(f"[EMERGENCY] Machine {machine_id} stopped due to critical condition")


# =============================================================================
# 2. Neural Components
# =============================================================================

class CNNLSTM_RULPredictor(NeuralComponent):
    """Simulates a CNN-LSTM for remaining useful life prediction from vibration and power."""
    def __init__(self, name: str, input_window: int = 20, hidden_dim: int = 32):
        super().__init__(name, input_dim=input_window * 2, output_dim=1)  # vibration + power
        self.input_window = input_window
        self.hidden_dim = hidden_dim
        # Simulated weights
        self.W_conv = np.random.randn(hidden_dim, 5) * 0.01  # CNN kernel
        self.W_lstm = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_out = np.random.randn(1, hidden_dim) * 0.01
        self.h = np.zeros(hidden_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        # Simulate CNN feature extraction
        conv_out = []
        for i in range(0, len(x)-4, 2):
            window = x[i:i+5]
            feat = np.tanh(self.W_conv @ window)
            conv_out.append(feat)
        if not conv_out:
            conv_out = [np.zeros(self.hidden_dim)]
        pooled = np.mean(conv_out, axis=0)
        # LSTM step
        self.h = np.tanh(self.W_lstm @ self.h + pooled)
        rul = float(self.W_out @ self.h)
        return np.array([max(0, rul)])

    def predict_rul(self, vibration_history: List[float], power_history: List[float]) -> float:
        """Predict remaining useful life in minutes."""
        # Pad to input_window
        if len(vibration_history) < self.input_window:
            vib = vibration_history + [0] * (self.input_window - len(vibration_history))
            pwr = power_history + [0] * (self.input_window - len(power_history))
        else:
            vib = vibration_history[-self.input_window:]
            pwr = power_history[-self.input_window:]
        features = np.array(vib + pwr)
        pred = self.forward(features)
        return float(pred[0])


class RLParameterOptimizer(NeuralComponent):
    """Simulates a reinforcement learning agent for optimizing cutting parameters."""
    def __init__(self, name: str, state_dim: int = 5, action_dim: int = 3):
        super().__init__(name, input_dim=state_dim, output_dim=action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.W = np.random.randn(action_dim, state_dim) * 0.01
        self.epsilon = 0.1

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        return self.W @ x  # Q-values

    def act(self, state: np.ndarray) -> Tuple[float, float]:
        """Return (feed_rate_factor, cutting_speed_factor) in [0.5, 1.5]."""
        q = self.forward(state)
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            action_idx = int(np.argmax(q))
        # Map action to parameter adjustments
        factors = {0: (0.7, 0.7), 1: (1.0, 1.0), 2: (1.3, 1.2)}
        return factors[action_idx]


class AnomalyDetectorCustom(AnomalyDetector):
    """Detects unusual wear patterns or machine conditions."""
    def __init__(self, name: str, input_dim: int = 4, hidden_dim: int = 8):
        super().__init__(name, input_dim, hidden_dim)
        self.threshold = 0.5

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Simple autoencoder: reconstruct input
        return x  # identity for simulation

    def anomaly_score(self, x: np.ndarray) -> float:
        # Simulated score based on input variance
        return float(np.std(x))

    def is_anomaly(self, x: np.ndarray) -> bool:
        return self.anomaly_score(x) > self.threshold


# =============================================================================
# 3. Custom Embedding
# =============================================================================

class MachineStateEmbedding(EmbeddingModel):
    """Embedding of machine state (vibration, temperature, wear ratio, power)."""
    def __init__(self, name: str = "machine_embedder", embedding_dim: int = 32):
        super().__init__(name, embedding_dim=embedding_dim)

    def embed(self, obj: COHObject) -> np.ndarray:
        machine = obj.get_attribute("machine")
        if machine is None:
            return np.zeros(self.embedding_dim)
        tool = machine.current_tool
        if tool:
            wear_ratio = 1 - (tool.current_life_min / tool.initial_life_min)
        else:
            wear_ratio = 0
        features = np.array([
            machine.vibration_level,
            machine.temperature / 100.0,
            wear_ratio,
            machine.spindle_power / machine.spindle_power_max
        ])
        proj = np.random.randn(self.embedding_dim, 4) * 0.1
        emb = proj @ features
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb


# =============================================================================
# 4. Daemons
# =============================================================================

class ToolWearDaemon(ConstraintDaemon):
    """Continuously monitors RUL and triggers pre-emptive tool change."""
    def __init__(self, parent: COHObject, interval: float = 5.0, rul_threshold: float = 30.0):
        super().__init__(parent, interval)
        self.rul_threshold = rul_threshold
        self.history = defaultdict(lambda: deque(maxlen=100))

    def check(self) -> None:
        machine = self.parent.get_attribute("machine")
        if machine is None or not machine.is_running:
            return
        tool = machine.current_tool
        if tool is None:
            return
        # Record vibration and power history
        self.history[machine.machine_id].append((machine.vibration_level, machine.spindle_power))
        vib_hist = [v for v, _ in self.history[machine.machine_id]]
        pwr_hist = [p for _, p in self.history[machine.machine_id]]
        # Predict RUL
        predictor = self.parent.get_neural_component("rul_predictor")
        rul = predictor.predict_rul(vib_hist, pwr_hist)
        print(f"[ToolWearDaemon] Machine {machine.machine_id} predicted RUL: {rul:.1f} min")
        if rul < self.rul_threshold:
            print(f"[ToolWearDaemon] RUL below threshold! Scheduling tool change.")
            self.parent.execute_method("schedule_maintenance", machine.machine_id, tool.tool_id)
            # Actually replace with a new tool (simplified)
            if machine.machine_id == "CNC1":
                new_tool = self.parent.world.tools.get("T002")
                if new_tool and new_tool.current_life_min > 0:
                    machine.current_tool = new_tool
            elif machine.machine_id == "CNC2":
                new_tool = self.parent.world.tools.get("T004")
                if new_tool:
                    machine.current_tool = new_tool


class QualityMonitor(ConstraintDaemon):
    """Monitors part quality, adjusts parameters if drift detected."""
    def __init__(self, parent: COHObject, interval: float = 10.0):
        super().__init__(parent, interval)
        self.defect_rates = []

    def check(self) -> None:
        machine = self.parent.get_attribute("machine")
        if machine is None:
            return
        # Simulate defect rate based on vibration and tool wear
        tool = machine.current_tool
        if tool:
            wear_ratio = 1 - (tool.current_life_min / tool.initial_life_min)
        else:
            wear_ratio = 1
        defect_rate = 0.01 + wear_ratio * 0.2 + machine.vibration_level * 0.1
        self.defect_rates.append(defect_rate)
        if defect_rate > 0.1:
            print(f"[QualityMonitor] High defect rate ({defect_rate:.2f}) on {machine.machine_id}. Adjusting parameters.")
            optimizer = self.parent.get_neural_component("param_optimizer")
            state = np.array([machine.vibration_level, wear_ratio, machine.temperature/100, machine.spindle_power/machine.spindle_power_max, defect_rate])
            feed_factor, speed_factor = optimizer.act(state)
            new_feed = machine.feed_rate_max * feed_factor
            new_speed = machine.cutting_speed_max * speed_factor
            self.parent.execute_method("adjust_parameters", machine.machine_id, feed_rate=new_feed, cutting_speed=new_speed)


# =============================================================================
# 5. Main Predictive Maintenance COH Object
# =============================================================================

class PredictiveMaintenanceSystem(COHObject):
    """
    Predictive maintenance system for CNC machines with tool wear prediction,
    parameter optimization, and constraint-aware scheduling.
    """
    def __init__(self, name: str = "PredMaintenance", machine_id: str = "CNC1", world: Optional[ManufacturingWorld] = None):
        super().__init__(name)
        self.world = world or ManufacturingWorld()
        self.machine = self.world.machines[machine_id]
        self.machine.is_running = True

        # ---- Attributes (A) ----
        self.add_attribute("world", self.world)
        self.add_attribute("machine", self.machine)
        self.add_attribute("vibration_history", deque(maxlen=100))
        self.add_attribute("power_history", deque(maxlen=100))
        self.add_attribute("last_rul", 100.0)
        self.add_attribute("total_downtime_min", 0.0)

        # ---- Neural Components (N) ----
        self.rul_predictor = CNNLSTM_RULPredictor(name="rul_predictor", input_window=20)
        self.add_neural_component("rul_predictor", self.rul_predictor)
        self.param_optimizer = RLParameterOptimizer(name="param_optimizer", state_dim=5, action_dim=3)
        self.add_neural_component("param_optimizer", self.param_optimizer)
        self.anomaly_detector = AnomalyDetectorCustom(name="wear_anomaly", input_dim=4)
        self.add_neural_component("wear_anomaly", self.anomaly_detector)

        # ---- Embedding (E) ----
        embedder = MachineStateEmbedding(name="machine_embedder", embedding_dim=32)
        self.add_neural_component("embedder", embedder, is_embedding_model=True)

        # ---- Methods (M) ----
        self.add_method("run_production", self.run_production)
        self.add_method("schedule_maintenance", self.schedule_maintenance)
        self.add_method("adjust_parameters", self.adjust_parameters)
        self.add_method("emergency_stop", self.emergency_stop)
        self.add_method("daily_report", self.daily_report)

        # ---- Identity Constraints (I) ----
        self.add_identity_constraint({
            'name': 'tool_life_positive',
            'specification': 'machine.current_tool.current_life_min >= 0',
            'severity': 10,
            'category': 'safety'
        })
        self.add_identity_constraint({
            'name': 'spindle_power_limit',
            'specification': 'machine.spindle_power <= machine.spindle_power_max',
            'severity': 9
        })
        self.add_identity_constraint({
            'name': 'vibration_safe',
            'specification': 'machine.vibration_level <= 0.8',
            'severity': 8
        })

        # ---- Trigger Constraints (T) ----
        self.add_trigger_constraint({
            'name': 'vibration_emergency',
            'specification': 'WHEN machine.vibration_level > 0.9 DO emergency_stop(machine.machine_id)',
            'priority': 'CRITICAL'
        })
        self.add_trigger_constraint({
            'name': 'low_rul_trigger',
            'specification': 'WHEN last_rul < 30 DO schedule_maintenance(machine.machine_id)',
            'priority': 'HIGH'
        })

        # ---- Goal Constraints (G) ----
        self.add_goal_constraint({
            'name': 'minimize_downtime',
            'specification': 'MINIMIZE total_downtime_min',
            'priority': 'HIGH'
        })
        self.add_goal_constraint({
            'name': 'maximize_tool_life',
            'specification': 'MAXIMIZE Σ(tool_life_used) / Σ(expected_life)',
            'priority': 'MEDIUM'
        })
        self.add_goal_constraint({
            'name': 'minimize_scrap_rate',
            'specification': 'MINIMIZE defect_rate',
            'priority': 'HIGH'
        })

        # ---- Daemons (D) ----
        tool_wear_daemon = ToolWearDaemon(self, interval=5.0, rul_threshold=30.0)
        self.daemons['tool_wear'] = tool_wear_daemon
        quality_daemon = QualityMonitor(self, interval=10.0)
        self.daemons['quality'] = quality_daemon

        # Register reasoners
        self.constraint_system.register_reasoner("temporal", TemporalReasoner())
        self.constraint_system.register_reasoner("safety", SafetyReasoner())

    def run_production(self, duration_min: float) -> Dict:
        """Run production for a given duration."""
        result = self.world.run_machine(self.machine.machine_id, duration_min)
        # Record histories
        vib_hist = self.get_attribute("vibration_history")
        pwr_hist = self.get_attribute("power_history")
        vib_hist.append(self.machine.vibration_level)
        pwr_hist.append(self.machine.spindle_power)
        self.add_attribute("vibration_history", vib_hist)
        self.add_attribute("power_history", pwr_hist)
        # Update RUL prediction
        predictor = self.get_neural_component("rul_predictor")
        rul = predictor.predict_rul(list(vib_hist), list(pwr_hist))
        self.add_attribute("last_rul", rul)
        # Check for anomalies
        if self.machine.current_tool:
            tool = self.machine.current_tool
            wear_ratio = 1 - (tool.current_life_min / tool.initial_life_min)
            features = np.array([self.machine.vibration_level, wear_ratio, self.machine.temperature/100, self.machine.spindle_power/self.machine.spindle_power_max])
            if self.anomaly_detector.is_anomaly(features):
                print(f"[Anomaly] Unusual wear pattern detected on {self.machine.machine_id}")
        return result

    def schedule_maintenance(self, machine_id: str, tool_id: str = None) -> None:
        """Schedule tool change or machine maintenance."""
        # Stop machine if running
        self.machine.is_running = False
        # Simulate downtime
        downtime = 15.0  # minutes
        self.add_attribute("total_downtime_min", self.get_attribute("total_downtime_min") + downtime)
        self.world.schedule_maintenance(machine_id, tool_id)
        self.machine.is_running = True

    def adjust_parameters(self, machine_id: str, feed_rate: float = None, cutting_speed: float = None) -> None:
        """Adjust cutting parameters."""
        self.world.adjust_parameters(machine_id, feed_rate, cutting_speed)

    def emergency_stop(self, machine_id: str) -> None:
        """Emergency stop."""
        self.world.emergency_stop(machine_id)

    def daily_report(self) -> Dict:
        """Generate daily performance report."""
        tool = self.machine.current_tool
        if tool:
            tool_life_used = tool.initial_life_min - tool.current_life_min
            tool_life_pct = (tool.current_life_min / tool.initial_life_min) * 100
        else:
            tool_life_pct = 0
        report = {
            "machine": self.machine.machine_id,
            "parts_produced": self.world.parts_produced,
            "tool_life_remaining_pct": tool_life_pct,
            "total_downtime_min": self.get_attribute("total_downtime_min"),
            "last_rul_min": self.get_attribute("last_rul"),
            "vibration": self.machine.vibration_level,
            "temperature": self.machine.temperature,
        }
        return report


# =============================================================================
# 6. Simulation
# =============================================================================

def run_simulation(days: int = 5, minutes_per_day: int = 480):
    """Run predictive maintenance simulation for given number of days."""
    world = ManufacturingWorld()
    system = PredictiveMaintenanceSystem("CNC_Predictive", machine_id="CNC1", world=world)

    system.initialize_system()
    system.start_daemons()

    print("=== Predictive Maintenance Simulation for CNC Machining ===")
    for day in range(1, days + 1):
        print(f"\n--- Day {day} ---")
        # Run production in 1-hour intervals
        for hour in range(8):  # 8 hours of production per day
            duration = 60  # minutes
            result = system.execute_method("run_production", duration)
            if "error" in result:
                print(f"Hour {hour+1}: {result['error']}")
                break
            else:
                print(f"Hour {hour+1}: wear remaining {result['wear_remaining']:.1f} min, vibration {result['vibration']:.2f}")
            time.sleep(0.2)  # slow simulation
        # End of day report
        report = system.execute_method("daily_report")
        print(f"\nDay {day} Report:")
        for k, v in report.items():
            print(f"  {k}: {v}")
        # Simulate end-of-day maintenance check
        if report["tool_life_remaining_pct"] < 20:
            print("Tool life low, scheduling maintenance for tomorrow.")
            system.execute_method("schedule_maintenance", "CNC1", "T001")
        time.sleep(1)

    system.stop_daemons()
    print("\n=== Simulation Complete ===")
    final_report = system.execute_method("daily_report")
    print(f"Final report: {final_report}")


if __name__ == "__main__":
    run_simulation(days=3, minutes_per_day=480)