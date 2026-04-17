#!/usr/bin/env python3
"""
AI-Driven Supply Chain Optimizer - COH Implementation using GISMOL Toolkit

This example models a global supply chain as a Constrained Object Hierarchy (COH).
It forecasts demand, manages inventory, routes shipments, and adapts to disruptions.
"""

import time
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import deque, defaultdict
from dataclasses import dataclass, field

# Import GISMOL components
from gismol.core import COHObject, COHRepository
from gismol.core.daemons import ConstraintDaemon
from gismol.neural import NeuralComponent, Regressor, AnomalyDetector
from gismol.neural.embeddings import EmbeddingModel
from gismol.reasoners import BaseReasoner, TemporalReasoner


# =============================================================================
# 1. Simulated Environment (Supply Chain World)
# =============================================================================

@dataclass
class Product:
    sku: str
    name: str
    weight_kg: float
    base_price: float
    safety_stock: int
    reorder_point: int
    carrying_cost_rate: float = 0.01  # per day as fraction of value

@dataclass
class Supplier:
    name: str
    lead_time_days: int
    reliability: float  # 0-1, probability of on-time delivery
    location: Tuple[float, float]  # lat/lon

@dataclass
class Warehouse:
    name: str
    location: Tuple[float, float]
    capacity: int  # max units
    inventory: Dict[str, int] = field(default_factory=dict)

@dataclass
class TransportRoute:
    origin: str
    destination: str
    cost_per_km: float
    distance_km: float
    capacity: int
    disruption_risk: float  # 0-1

class SupplyChainWorld:
    """Simulated environment with products, suppliers, warehouses, and transport."""
    def __init__(self):
        # Products
        self.products = {
            "P001": Product("P001", "Widget A", 0.5, 10.0, safety_stock=50, reorder_point=100, carrying_cost_rate=0.01),
            "P002": Product("P002", "Gadget B", 1.2, 25.0, safety_stock=30, reorder_point=80, carrying_cost_rate=0.015),
            "P003": Product("P003", "Component C", 0.3, 5.0, safety_stock=200, reorder_point=500, carrying_cost_rate=0.005),
        }
        # Suppliers
        self.suppliers = {
            "SupA": Supplier("SupA", lead_time_days=5, reliability=0.95, location=(34.05, -118.24)),
            "SupB": Supplier("SupB", lead_time_days=10, reliability=0.85, location=(40.71, -74.01)),
            "SupC": Supplier("SupC", lead_time_days=3, reliability=0.98, location=(51.51, -0.13)),
        }
        # Warehouses
        self.warehouses = {
            "WH_NY": Warehouse("WH_NY", location=(40.71, -74.01), capacity=10000),
            "WH_LA": Warehouse("WH_LA", location=(34.05, -118.24), capacity=8000),
            "WH_LON": Warehouse("WH_LON", location=(51.51, -0.13), capacity=5000),
        }
        for wh in self.warehouses.values():
            for sku in self.products:
                wh.inventory[sku] = random.randint(100, 500)
        # Transport routes
        self.routes = [
            TransportRoute("WH_NY", "WH_LA", cost_per_km=0.5, distance_km=4500, capacity=2000, disruption_risk=0.1),
            TransportRoute("WH_LA", "WH_NY", cost_per_km=0.5, distance_km=4500, capacity=2000, disruption_risk=0.1),
            TransportRoute("WH_NY", "WH_LON", cost_per_km=0.8, distance_km=5500, capacity=1500, disruption_risk=0.15),
            TransportRoute("WH_LON", "WH_NY", cost_per_km=0.8, distance_km=5500, capacity=1500, disruption_risk=0.15),
            TransportRoute("WH_LA", "WH_LON", cost_per_km=0.7, distance_km=9000, capacity=1000, disruption_risk=0.2),
        ]
        self.disruption_active = False
        self.disruption_type = None
        self.current_day = 0
        self.demand_history = defaultdict(list)  # sku -> list of daily demand

    def get_demand(self, sku: str, day: int) -> int:
        """Simulate random demand with seasonal pattern."""
        # Base demand
        base = 100 if sku == "P001" else 60 if sku == "P002" else 300
        # Seasonality (sinusoidal)
        seasonal = 20 * math.sin(2 * math.pi * day / 30)
        # Random noise
        noise = random.gauss(0, 15)
        demand = max(0, int(base + seasonal + noise))
        self.demand_history[sku].append(demand)
        return demand

    def get_supplier_lead_time(self, supplier_name: str) -> int:
        """Return actual lead time (may be longer if disruption or low reliability)."""
        sup = self.suppliers[supplier_name]
        base = sup.lead_time_days
        if self.disruption_active and self.disruption_type == "supplier_delay":
            base += random.randint(5, 15)
        if random.random() > sup.reliability:
            base += random.randint(2, 7)
        return base

    def get_route_cost(self, origin: str, destination: str, quantity: int) -> float:
        """Compute transport cost; may increase if disruption."""
        for route in self.routes:
            if route.origin == origin and route.destination == destination:
                cost = route.cost_per_km * route.distance_km * (quantity / route.capacity)
                if self.disruption_active and self.disruption_type == "port_strike":
                    cost *= 1.5
                return cost
        return float('inf')


# =============================================================================
# 2. Neural Components
# =============================================================================

class TransformerDemandPredictor(NeuralComponent):
    """Simulates a transformer model for multivariate time series demand forecasting."""
    def __init__(self, name: str, input_window: int = 30, hidden_dim: int = 64, output_horizon: int = 7):
        super().__init__(name, input_dim=input_window * 3, output_dim=output_horizon)  # 3 products
        self.input_window = input_window
        self.hidden_dim = hidden_dim
        self.output_horizon = output_horizon
        # Simulated weights
        self.W1 = np.random.randn(hidden_dim, self.input_dim) * 0.01
        self.W2 = np.random.randn(self.output_dim, hidden_dim) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        h = np.tanh(self.W1 @ x)
        out = self.W2 @ h
        return out

    def predict(self, history: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """Predict demand for next output_horizon days for each SKU."""
        # Build feature vector: concatenate last input_window days of each product
        features = []
        for sku in ["P001", "P002", "P003"]:
            hist = history.get(sku, [])[-self.input_window:]
            if len(hist) < self.input_window:
                hist = [0] * (self.input_window - len(hist)) + hist
            features.extend(hist)
        features = np.array(features) / 500.0  # normalize
        pred = self.forward(features)
        # Reshape to (output_horizon, 3)
        pred = pred.reshape(self.output_horizon, 3)
        result = {}
        for i, sku in enumerate(["P001", "P002", "P003"]):
            result[sku] = [max(0, int(pred[t, i] * 500)) for t in range(self.output_horizon)]
        return result


class GraphNeuralRouter(NeuralComponent):
    """Simulates a GNN that selects optimal transport routes based on costs and disruptions."""
    def __init__(self, name: str, node_features: int = 4, edge_features: int = 3):
        super().__init__(name, input_dim=node_features + edge_features, output_dim=1)
        self.W = np.random.randn(1, self.input_dim) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        return self.W @ x  # score

    def rank_routes(self, origin: str, destination: str, quantity: int, world: SupplyChainWorld) -> List[TransportRoute]:
        """Return routes sorted by predicted efficiency."""
        candidates = [r for r in world.routes if r.origin == origin and r.destination == destination]
        scored = []
        for route in candidates:
            features = np.array([
                route.distance_km / 10000,
                route.cost_per_km,
                route.disruption_risk,
                quantity / route.capacity,
                (1 if world.disruption_active else 0)
            ])
            score = float(self.forward(features))
            scored.append((score, route))
        scored.sort(reverse=True)
        return [r for _, r in scored]


# =============================================================================
# 3. Custom Embedding
# =============================================================================

class SupplyChainEmbedding(EmbeddingModel):
    """Embedding of product, region, season, supplier risk."""
    def __init__(self, name: str = "sc_embedder", embedding_dim: int = 128):
        super().__init__(name, embedding_dim=embedding_dim)

    def embed(self, obj: COHObject) -> np.ndarray:
        # Get context from the parent supply chain object
        world = obj.get_attribute("world")
        if world is None:
            return np.zeros(self.embedding_dim)
        # Create a simple feature vector: e.g., average inventory, disruption flag, day
        inv = obj.get_attribute("inventory_levels", {})
        avg_inv = np.mean(list(inv.values())) if inv else 0
        features = np.array([
            avg_inv / 1000,
            1.0 if world.disruption_active else 0.0,
            world.current_day / 365.0,
            len(obj.get_attribute("customer_orders", [])) / 100
        ])
        # Random projection to embedding_dim
        proj = np.random.randn(self.embedding_dim, len(features)) * 0.1
        emb = proj @ features
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb


# =============================================================================
# 4. Daemons
# =============================================================================

class DisruptionMonitor(ConstraintDaemon):
    """Monitors external events and triggers disruptions."""
    def __init__(self, parent: COHObject, interval: float = 5.0):
        super().__init__(parent, interval)
        self.disruption_types = ["port_strike", "supplier_delay", "none"]
        self.probability = 0.05  # 5% chance per check

    def check(self) -> None:
        world = self.parent.get_attribute("world")
        if world is None:
            return
        # Random disruption
        if random.random() < self.probability:
            disruption = random.choice(["port_strike", "supplier_delay"])
            world.disruption_active = True
            world.disruption_type = disruption
            print(f"[DisruptionMonitor] Disruption detected: {disruption}")
            # Trigger rerouting via parent method
            self.parent.execute_method("handle_disruption", disruption)
        else:
            # Clear disruption after some time (simulated)
            if world.disruption_active and random.random() < 0.2:
                world.disruption_active = False
                print("[DisruptionMonitor] Disruption resolved.")


class InventoryHealthDaemon(ConstraintDaemon):
    """Checks for slow-moving stock and suggests promotions."""
    def __init__(self, parent: COHObject, interval: float = 10.0):
        super().__init__(parent, interval)
        self.sku_turnover = defaultdict(list)

    def check(self) -> None:
        inv = self.parent.get_attribute("inventory_levels", {})
        for sku, qty in inv.items():
            # Simple turnover estimate: if inventory > 3*safety_stock and not moving, suggest promotion
            product = self.parent.get_attribute("world").products.get(sku)
            if product and qty > 3 * product.safety_stock:
                print(f"[InventoryHealth] SKU {sku} has excess stock ({qty}), consider promotion.")


# =============================================================================
# 5. Main Supply Chain COH Object
# =============================================================================

class SupplyChainOptimizer(COHObject):
    """
    AI-driven supply chain optimizer with demand forecasting, inventory management,
    dynamic routing, and disruption handling.
    """
    def __init__(self, name: str = "SupplyChainOptimizer", world: Optional[SupplyChainWorld] = None):
        super().__init__(name)
        self.world = world or SupplyChainWorld()

        # ---- Attributes (A) ----
        self.add_attribute("world", self.world)
        self.add_attribute("inventory_levels", {sku: wh.inventory[sku] for wh in self.world.warehouses.values() for sku in wh.inventory})
        # Aggregate inventory per SKU across warehouses for simplicity
        self._aggregate_inventory()
        self.add_attribute("customer_orders", [])  # list of (sku, qty, due_date)
        self.add_attribute("pending_shipments", [])  # list of (sku, qty, origin, dest, status)
        self.add_attribute("total_cost", 0.0)

        # ---- Neural Components (N) ----
        self.demand_predictor = TransformerDemandPredictor(name="demand_predictor", input_window=30, output_horizon=7)
        self.add_neural_component("demand_predictor", self.demand_predictor)
        self.graph_router = GraphNeuralRouter(name="graph_router")
        self.add_neural_component("graph_router", self.graph_router)
        self.anomaly_detector = AnomalyDetector(name="supplier_anomaly", input_dim=5, hidden_dim=10)
        self.add_neural_component("anomaly_detector", self.anomaly_detector)

        # ---- Embedding (E) ----
        embedder = SupplyChainEmbedding(name="sc_embedder", embedding_dim=128)
        self.add_neural_component("embedder", embedder, is_embedding_model=True)

        # ---- Methods (M) ----
        self.add_method("forecast_demand", self.forecast_demand)
        self.add_method("reorder_point", self.reorder_point)
        self.add_method("route_shipment", self.route_shipment)
        self.add_method("handle_disruption", self.handle_disruption)
        self.add_method("allocate_warehouse_space", self.allocate_warehouse_space)
        self.add_method("daily_update", self.daily_update)

        # ---- Identity Constraints (I) ----
        for sku, prod in self.world.products.items():
            self.add_identity_constraint({
                'name': f'safety_stock_{sku}',
                'specification': f'inventory_levels.{sku} >= {prod.safety_stock}',
                'severity': 9,
                'category': 'inventory'
            })
        self.add_identity_constraint({
            'name': 'transport_budget',
            'specification': 'total_cost <= 10000',  # arbitrary daily budget
            'severity': 7
        })
        self.add_identity_constraint({
            'name': 'fulfillment_time',
            'specification': 'order_fulfillment_time <= 48',
            'severity': 8
        })

        # ---- Trigger Constraints (T) ----
        self.add_trigger_constraint({
            'name': 'low_inventory_reorder',
            'specification': 'WHEN inventory_levels.P001 < 100 OR inventory_levels.P002 < 80 OR inventory_levels.P003 < 500 DO place_reorder()',
            'priority': 'HIGH'
        })
        self.add_trigger_constraint({
            'name': 'disruption_reroute',
            'specification': 'WHEN world.disruption_active == True DO handle_disruption(world.disruption_type)',
            'priority': 'CRITICAL'
        })

        # ---- Goal Constraints (G) ----
        self.add_goal_constraint({
            'name': 'minimize_total_cost',
            'specification': 'MINIMIZE total_cost = carrying_cost + transport_cost + stockout_penalty',
            'priority': 'HIGH'
        })

        # ---- Daemons (D) ----
        disruption_daemon = DisruptionMonitor(self, interval=5.0)
        self.daemons['disruption_monitor'] = disruption_daemon
        inventory_daemon = InventoryHealthDaemon(self, interval=10.0)
        self.daemons['inventory_health'] = inventory_daemon

        # Register reasoners
        self.constraint_system.register_reasoner("temporal", TemporalReasoner())

    def _aggregate_inventory(self):
        """Aggregate inventory levels across warehouses."""
        inv = defaultdict(int)
        for wh in self.world.warehouses.values():
            for sku, qty in wh.inventory.items():
                inv[sku] += qty
        self.add_attribute("inventory_levels", dict(inv))

    def forecast_demand(self, horizon_days: int = 7) -> Dict[str, List[int]]:
        """Use neural predictor to forecast demand."""
        pred = self.demand_predictor.predict(self.world.demand_history)
        return pred

    def reorder_point(self, sku: str) -> bool:
        """Check if inventory below reorder point and place order."""
        inv = self.get_attribute("inventory_levels", {})
        prod = self.world.products.get(sku)
        if not prod:
            return False
        if inv.get(sku, 0) < prod.reorder_point:
            # Place order to a random supplier
            supplier = random.choice(list(self.world.suppliers.values()))
            lead_time = self.world.get_supplier_lead_time(supplier.name)
            print(f"[Reorder] Placing order for {sku} from {supplier.name}, lead time {lead_time} days")
            # Simulate order arrival after lead time (simplified: reduce inventory after lead time)
            # In real system, would add to pending orders.
            return True
        return False

    def route_shipment(self, origin: str, destination: str, sku: str, quantity: int) -> float:
        """Use graph neural router to select best route and return cost."""
        routes = self.graph_router.rank_routes(origin, destination, quantity, self.world)
        if not routes:
            print(f"[Routing] No route from {origin} to {destination}")
            return float('inf')
        best_route = routes[0]
        cost = self.world.get_route_cost(origin, destination, quantity)
        print(f"[Routing] Shipment {quantity} of {sku} from {origin} to {destination} via {best_route.origin}->{best_route.destination} cost {cost:.2f}")
        # Deduct inventory from origin warehouse
        for wh in self.world.warehouses.values():
            if wh.name == origin:
                wh.inventory[sku] = max(0, wh.inventory.get(sku, 0) - quantity)
        # Add to destination warehouse (simulate arrival after transit time)
        for wh in self.world.warehouses.values():
            if wh.name == destination:
                wh.inventory[sku] = wh.inventory.get(sku, 0) + quantity
        self._aggregate_inventory()
        # Update total cost
        self.add_attribute("total_cost", self.get_attribute("total_cost") + cost)
        return cost

    def handle_disruption(self, disruption_type: str) -> None:
        """React to disruption by re-routing shipments and adjusting orders."""
        print(f"[Disruption Handler] Handling {disruption_type}")
        if disruption_type == "port_strike":
            # Re-route all pending shipments away from affected ports
            # For simplicity, we just log.
            pass
        elif disruption_type == "supplier_delay":
            # Increase safety stock temporarily
            for sku, prod in self.world.products.items():
                prod.safety_stock = int(prod.safety_stock * 1.5)
            print("[Disruption Handler] Increased safety stock for all products")

    def allocate_warehouse_space(self, sku: str, quantity: int) -> bool:
        """Find a warehouse with capacity and allocate space."""
        for wh in self.world.warehouses.values():
            current_used = sum(wh.inventory.values())
            if current_used + quantity <= wh.capacity:
                wh.inventory[sku] = wh.inventory.get(sku, 0) + quantity
                self._aggregate_inventory()
                return True
        return False

    def daily_update(self, day: int) -> None:
        """Run daily operations: demand generation, forecasting, reordering, routing."""
        self.world.current_day = day
        # Generate customer orders based on demand
        orders = []
        for sku, prod in self.world.products.items():
            demand = self.world.get_demand(sku, day)
            if demand > 0:
                orders.append((sku, demand, day + 2))  # due in 2 days
        self.add_attribute("customer_orders", orders)
        print(f"\n=== Day {day} ===")
        print(f"Orders: {orders}")

        # Forecast future demand
        forecast = self.forecast_demand(7)
        print(f"Demand forecast (next 7 days): {forecast}")

        # Check reorder points for all SKUs
        for sku in self.world.products:
            self.reorder_point(sku)

        # Fulfill orders: route shipments from nearest warehouse with stock
        for sku, qty, due in orders:
            # Find warehouse with sufficient inventory
            best_wh = None
            for wh in self.world.warehouses.values():
                if wh.inventory.get(sku, 0) >= qty:
                    best_wh = wh
                    break
            if best_wh:
                # Route to a distribution center (for simplicity, use a fixed destination)
                dest = "WH_NY" if best_wh.name != "WH_NY" else "WH_LA"
                cost = self.route_shipment(best_wh.name, dest, sku, qty)
                print(f"Fulfilled order {sku} x{qty} from {best_wh.name} to {dest}, cost {cost:.2f}")
            else:
                # Stockout penalty
                stockout_penalty = qty * self.world.products[sku].base_price * 0.5
                self.add_attribute("total_cost", self.get_attribute("total_cost") + stockout_penalty)
                print(f"STOCKOUT: {sku} x{qty}, penalty {stockout_penalty}")

        # Update inventory aggregation
        self._aggregate_inventory()
        inv = self.get_attribute("inventory_levels")
        print(f"Current inventory: {inv}")
        print(f"Total cost so far: {self.get_attribute('total_cost'):.2f}")


# =============================================================================
# 6. Simulation
# =============================================================================

def run_simulation(days: int = 30):
    """Run supply chain simulation for given number of days."""
    world = SupplyChainWorld()
    optimizer = SupplyChainOptimizer("GlobalSupplyChain", world)

    optimizer.initialize_system()
    optimizer.start_daemons()

    print("=== AI-Driven Supply Chain Optimizer Simulation ===")
    for day in range(days):
        optimizer.execute_method("daily_update", day)
        time.sleep(0.5)  # slow down for readability

    optimizer.stop_daemons()
    print("\n=== Simulation Complete ===")
    print(f"Final total cost: {optimizer.get_attribute('total_cost'):.2f}")


if __name__ == "__main__":
    run_simulation(days=30)