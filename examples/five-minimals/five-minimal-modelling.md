# Five Minimal but Interesting Intelligent Systems Modeled with COH Theory

Below are five concise COH models that illustrate the 9‑tuple formalism across different domains. Each model is minimal (only essential components shown) yet captures the core intelligent behavior of the system.

---

## 1. Smart Thermostat (Home Energy Management)

**Domain:** Simple reactive control with learning.

**COHₜₕₑᵣₘₒₛₜₐₜ = (C, A, M, N, E, I, T, G, D)**

| Component | Instantiation |
|-----------|----------------|
| **C** | `TemperatureSensor`, `Heater`, `UserInterface` |
| **A** | `current_temp ∈ ℝ`, `target_temp ∈ ℝ`, `heater_on ∈ {0,1}`, `energy_used ∈ ℝ` |
| **M** | `measure_temp()`, `set_heater(state)`, `adjust_target(delta)` |
| **N** | `TempPredictor` (simple linear regression predicting next hour's temperature) |
| **E** | Embedding of `(current_temp, target_temp, time_of_day)` → ℝ³ (normalized) |
| **I** | `current_temp ≥ 5°C` (freeze protection), `heater_on ∈ {0,1}` |
| **T** | `IF current_temp < target_temp - 0.5 → set_heater(1)`; `IF current_temp > target_temp + 0.5 → set_heater(0)` |
| **G** | Minimize `|current_temp - target_temp| + 0.01 * energy_used` (comfort + efficiency) |
| **D** | `OvershootMonitor` (logs if temperature exceeds target by >2°C for >10 min) |

**Emergent intelligence:** Learns to pre‑heat before occupancy using neural predictor, balancing comfort and energy.

---

## 2. Simple Dialogue Agent (Customer Support Bot)

**Domain:** Natural language understanding and constrained response.

**COHₐgₑₙₜ = (C, A, M, N, E, I, T, G, D)**

| Component | Instantiation |
|-----------|----------------|
| **C** | `IntentRecognizer`, `ResponseGenerator`, `KnowledgeBase` |
| **A** | `user_input: str`, `intent: str`, `confidence: float`, `response: str` |
| **M** | `parse_input(text)`, `retrieve_answer(intent)`, `send_response(text)` |
| **N** | `BERT‑based intent classifier` (fine‑tuned on support tickets) |
| **E** | `SentenceTransformer` embedding of user input → ℝ₃₈₄ |
| **I** | `confidence ≥ 0.6` before answering; `response not contain profanity` |
| **T** | `IF confidence < 0.6 → respond("I don't understand, please rephrase")`; `IF user_input contains "escalate" → transfer_to_human` |
| **G** | Maximize `customer_satisfaction` (predicted by neural net) while minimizing `response_time` |
| **D** | `ProfanityFilterDaemon` (scans every response in real time) |

**Emergent intelligence:** Adapts intent recognition via active learning; escalates gracefully when uncertain.

---

## 3. Autonomous Vacuum Cleaner (Navigation & Coverage)

**Domain:** Planning under uncertainty with resource constraints.

**COHᵥₐcᵤᵤₘ = (C, A, M, N, E, I, T, G, D)**

| Component | Instantiation |
|-----------|----------------|
| **C** | `DriveMotors`, `Battery`, `DirtSensor`, `MapManager` |
| **A** | `position: (x,y)`, `battery_level ∈ [0,100]`, `dirt_density: ℝ²→ℝ`, `coverage_map: bitmask` |
| **M** | `move(dx,dy)`, `clean_cell()`, `return_to_dock()`, `update_map()` |
| **N** | `CoveragePlanner` (Deep Q‑Network predicting next best cell) |
| **E** | Embedding of `(position, battery, local_dirt_map)` → ℝ₆₄ via CNN |
| **I** | `battery_level ≥ 10` (never fully drain); `position within room boundaries` |
| **T** | `IF battery_level < 20 → return_to_dock()`; `IF dirt_density > threshold → clean_cell()` |
| **G** | Maximize `coverage_percentage - 0.1 * time_elapsed - 100 * (battery_depletion_penalty)` |
| **D** | `BatteryGuardian` (continuously predicts remaining runtime and aborts long moves if insufficient) |

**Emergent intelligence:** Learns efficient cleaning paths, avoids obstacles, and autonomously recharges.

---

## 4. Stock Trading Agent (Reinforcement Learning)

**Domain:** Financial decision making with risk constraints.

**COHₜᵣₐdₑ = (C, A, M, N, E, I, T, G, D)**

| Component | Instantiation |
|-----------|----------------|
| **C** | `MarketDataFeed`, `PortfolioManager`, `OrderExecutor` |
| **A** | `price_history: ℝ¹⁰⁰`, `cash: ℝ`, `holdings: ℝ`, `portfolio_value: ℝ`, `risk_score: ℝ` |
| **M** | `update_prices()`, `compute_signal()`, `place_order(side, qty)`, `rebalance()` |
| **N** | `LSTM‑based price predictor` + `PolicyNetwork` (PPO) |
| **E** | Embedding of `(price_history, technical_indicators)` → ℝ₅₁₂ via transformer |
| **I** | `leverage ≤ 2.0`, `single_position ≤ 0.3 * portfolio_value` (diversification) |
| **T** | `IF portfolio_value_drawdown > 0.15 → reduce_all_positions_by(50%)`; `IF risk_score > 0.8 → halt_trading` |
| **G** | Maximize `Sharpe_ratio` (return / volatility) over rolling 30‑day window |
| **D** | `CircuitBreakerDaemon` (if loss > 5% in 1 minute, liquidates and pauses) |

**Emergent intelligence:** Adapts to market regimes, learns risk‑adjusted strategies, and respects hard safety limits.

---

## 5. Adaptive Cruise Control (Semi‑Autonomous Driving)

**Domain:** Real‑time control with safety constraints.

**COHₐcc = (C, A, M, N, E, I, T, G, D)**

| Component | Instantiation |
|-----------|----------------|
| **C** | `Radar`, `SpeedController`, `BrakeActuator`, `LeadVehicleTracker` |
| **A** | `ego_speed: ℝ`, `lead_distance: ℝ`, `lead_speed: ℝ`, `time_gap: ℝ`, `brake_pressure: ℝ` |
| **M** | `set_throttle(%)`, `apply_brake(%)`, `update_lead_info()`, `compute_safe_gap()` |
| **N** | `LeadVehiclePredictor` (LSTM forecasting next 2s of lead vehicle motion) |
| **E** | Embedding of `(ego_speed, lead_distance, lead_speed, road_curvature)` → ℝ₃₂ |
| **I** | `brake_pressure ≥ 0` and `≤ 100`; `lead_distance ≥ 2.5 m` (absolute safety) |
| **T** | `IF time_gap < 1.0s → increase_brake_pressure(linear)`; `IF lead_distance < 5m AND ego_speed > lead_speed → emergency_brake()` |
| **G** | Minimize `(desired_speed - ego_speed)² + 0.1 * jerk² + 1000 * collision_risk` |
| **D** | `SafetyEnforcer` (overrides throttle if predicted deceleration exceeds tire friction limit) |

**Emergent intelligence:** Smoothly follows lead vehicle, anticipates braking events via neural predictor, and always respects hard safety constraints.

---

## Summary Table of COH Components Across the Five Systems

| System | C (Components) | A (Attributes) | M (Methods) | N (Neural) | E (Embedding) | I (Identity) | T (Triggers) | G (Goals) | D (Daemons) |
|--------|----------------|----------------|-------------|------------|---------------|--------------|--------------|-----------|-------------|
| Thermostat | 3 | 4 | 3 | 1 | 3‑dim | 2 | 2 | 2‑term | 1 |
| Dialogue | 3 | 5 | 3 | 1 | 384‑dim | 2 | 2 | 2‑term | 1 |
| Vacuum | 4 | 5 | 4 | 1 | 64‑dim | 2 | 2 | 3‑term | 1 |
| Trading | 3 | 6 | 4 | 2 | 512‑dim | 2 | 2 | 1 (Sharpe) | 1 |
| ACC | 4 | 6 | 4 | 1 | 32‑dim | 2 | 2 | 3‑term | 1 |

Each model satisfies the COH definition (9‑tuple) and exhibits intelligence through hierarchical composition, constraint satisfaction, and adaptive neural components. They are minimal enough to implement but capture essential characteristics of real intelligent systems.