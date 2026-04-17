#!/usr/bin/env python3
"""
Stock Trading Agent - COH Implementation using GISMOL Toolkit

This example models a reinforcement learning trading agent as a Constrained Object Hierarchy (COH).
It learns to trade a single asset, manages portfolio risk, and enforces constraints.
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

# Import GISMOL components
from gismol.core import COHObject
from gismol.core.daemons import ConstraintDaemon
from gismol.neural import NeuralComponent
from gismol.neural.embeddings import EmbeddingModel
from gismol.neural.optimizers import ConstraintAwareOptimizer


# =============================================================================
# 1. Simulated Market Environment
# =============================================================================
class MarketEnvironment:
    """
    Simulated financial market with mean-reverting price dynamics.
    """
    def __init__(self, initial_price: float = 100.0, volatility: float = 0.02,
                 mean_reversion: float = 0.1, long_term_mean: float = 100.0):
        self.price = initial_price
        self.volatility = volatility
        self.mean_reversion = mean_reversion
        self.long_term_mean = long_term_mean
        self.history = [initial_price]

    def step(self, dt: float = 1/252) -> float:
        """
        Simulate one trading day (dt = 1/252 year).
        Geometric mean-reverting process (Ornstein-Uhlenbeck on log price).
        """
        log_price = math.log(self.price)
        drift = self.mean_reversion * (math.log(self.long_term_mean) - log_price)
        diffusion = self.volatility * random.gauss(0, math.sqrt(dt))
        new_log_price = log_price + drift * dt + diffusion
        self.price = math.exp(new_log_price)
        self.history.append(self.price)
        return self.price

    def get_state_vector(self, lookback: int = 100) -> np.ndarray:
        """Return normalized price history for neural input."""
        recent = self.history[-lookback:] if len(self.history) >= lookback else self.history
        # Pad if necessary
        if len(recent) < lookback:
            recent = [recent[0]] * (lookback - len(recent)) + recent
        # Normalize by first value in window
        norm = np.array(recent) / recent[0] - 1.0
        return norm


# =============================================================================
# 2. Neural Component: LSTM-based Policy Network (simulated with simple RNN)
# =============================================================================
class PolicyNetwork(NeuralComponent):
    """
    Neural network that maps state (price history) to action probabilities
    and value estimate. Simulates an LSTM with a simple recurrent structure.
    """
    def __init__(self, name: str, input_dim: int, hidden_dim: int = 64, action_dim: int = 3):
        super().__init__(name, input_dim=input_dim, output_dim=action_dim + 1)  # +1 for value
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        # Simple RNN weights (simulated)
        self.W_xh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_h = np.zeros(hidden_dim)
        self.W_hp = np.random.randn(action_dim, hidden_dim) * 0.01
        self.b_p = np.zeros(action_dim)
        self.W_hv = np.random.randn(1, hidden_dim) * 0.01
        self.b_v = np.zeros(1)
        self.h = np.zeros(hidden_dim)

    def reset_hidden(self):
        self.h = np.zeros(self.hidden_dim)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Returns (action_probabilities, state_value).
        x: input vector of shape (input_dim,)
        """
        x = np.asarray(x).flatten()
        # RNN step
        h_new = np.tanh(self.W_xh @ x + self.W_hh @ self.h + self.b_h)
        self.h = h_new
        # Action logits
        logits = self.W_hp @ h_new + self.b_p
        # Softmax for action probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        # State value
        value = float(self.W_hv @ h_new + self.b_v)
        return probs, value

    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select action (0=sell, 1=hold, 2=buy)."""
        probs, _ = self.forward(state)
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(self.action_dim, p=probs))


# =============================================================================
# 3. Custom Embedding Model: Technical Indicators (simulated)
# =============================================================================
class MarketEmbedding(EmbeddingModel):
    """
    Embedding of recent price history into a low-dimensional vector
    capturing momentum and volatility.
    """
    def __init__(self, name: str = "market_embedder", embedding_dim: int = 16):
        super().__init__(name, embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim

    def embed(self, obj: COHObject) -> np.ndarray:
        """Compute embedding from market history stored in parent object."""
        market = obj.get_attribute("market")
        if market is None:
            return np.zeros(self.embedding_dim)
        prices = np.array(market.history[-50:])  # last 50 days
        if len(prices) < 2:
            return np.zeros(self.embedding_dim)
        # Simple features: returns over different windows
        ret_1d = (prices[-1] / prices[-2] - 1) if len(prices) >= 2 else 0.0
        ret_5d = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else ret_1d
        ret_20d = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else ret_1d
        volatility = np.std(np.diff(np.log(prices[-20:]))) if len(prices) >= 20 else 0.01
        features = np.array([ret_1d, ret_5d, ret_20d, volatility, prices[-1] / 100.0])
        # Project to embedding dimension using random matrix
        proj = np.random.randn(self.embedding_dim, len(features)) * 0.1
        embedding = proj @ features
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding


# =============================================================================
# 4. Custom Daemon: Circuit Breaker
# =============================================================================
class CircuitBreakerDaemon(ConstraintDaemon):
    """
    Monitors portfolio drawdown and halts trading if loss exceeds threshold.
    """
    def __init__(self, parent: COHObject, interval: float = 1.0, loss_threshold: float = 0.05):
        super().__init__(parent, interval)
        self.loss_threshold = loss_threshold  # 5% loss in one minute
        self.last_portfolio_value = None
        self.last_time = None

    def check(self) -> None:
        current_value = self.parent.get_attribute("portfolio_value", 0.0)
        current_time = self.parent.get_attribute("time", 0.0)
        if self.last_portfolio_value is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt < 1.0 / (24 * 60):  # less than one minute? Simulated
                loss = (self.last_portfolio_value - current_value) / self.last_portfolio_value
                if loss > self.loss_threshold:
                    print(f"[CircuitBreaker] Loss {loss*100:.2f}% in short period! Halting trading.")
                    self.parent.add_attribute("trading_halted", True)
                    # Execute liquidation
                    self.parent.execute_method("liquidate")
        self.last_portfolio_value = current_value
        self.last_time = current_time


# =============================================================================
# 5. Main Trading Agent COH Object
# =============================================================================
class TradingAgent(COHObject):
    """
    Reinforcement learning trading agent with risk constraints.
    """
    def __init__(self, name: str = "TradingBot", market: Optional[MarketEnvironment] = None):
        super().__init__(name)
        self.market = market or MarketEnvironment()

        # ---- Attributes (A) ----
        self.add_attribute("cash", 10000.0)            # USD
        self.add_attribute("holdings", 0.0)            # number of shares
        self.add_attribute("portfolio_value", 10000.0) # cash + holdings * price
        self.add_attribute("time", 0.0)                # days
        self.add_attribute("price_history", deque(maxlen=100))
        self.add_attribute("market", self.market)
        self.add_attribute("trading_halted", False)
        self.add_attribute("risk_score", 0.0)          # simulated risk metric

        # ---- Neural Components (N) ----
        # Input dimension: normalized price history of length 100
        self.policy_net = PolicyNetwork(name="policy_net", input_dim=100, hidden_dim=64, action_dim=3)
        self.add_neural_component("policy_net", self.policy_net)

        # ---- Embedding (E) ----
        embedder = MarketEmbedding(name="market_embedder", embedding_dim=16)
        self.add_neural_component("embedder", embedder, is_embedding_model=True)

        # ---- Methods (M) ----
        self.add_method("update_prices", self.update_prices)
        self.add_method("compute_signal", self.compute_signal)
        self.add_method("place_order", self.place_order)
        self.add_method("rebalance", self.rebalance)
        self.add_method("liquidate", self.liquidate)

        # ---- Identity Constraints (I) ----
        self.add_identity_constraint({
            'name': 'leverage_limit',
            'specification': 'holdings * price <= 2.0 * cash',  # max 2x leverage
            'severity': 9,
            'category': 'risk'
        })
        self.add_identity_constraint({
            'name': 'single_position_limit',
            'specification': 'holdings * price <= 0.3 * portfolio_value',
            'severity': 8,
            'category': 'risk'
        })
        self.add_identity_constraint({
            'name': 'non_negative_cash',
            'specification': 'cash >= 0',
            'severity': 10
        })

        # ---- Trigger Constraints (T) ----
        self.add_trigger_constraint({
            'name': 'drawdown_protection',
            'specification': 'WHEN (portfolio_value_drawdown > 0.15) DO reduce_all_positions_by(50)',
            'priority': 'HIGH'
        })
        self.add_trigger_constraint({
            'name': 'risk_halting',
            'specification': 'WHEN risk_score > 0.8 DO halt_trading()',
            'priority': 'HIGH'
        })

        # ---- Goal Constraints (G) ----
        self.add_goal_constraint({
            'name': 'sharpe_ratio',
            'specification': 'MAXIMIZE (mean_return / volatility) over rolling_30d',
            'priority': 'HIGH'
        })

        # ---- Daemons (D) ----
        circuit_breaker = CircuitBreakerDaemon(self, interval=1.0, loss_threshold=0.05)
        self.daemens['circuit_breaker'] = circuit_breaker

    # ---- Method implementations ----
    def update_prices(self) -> None:
        """Advance market and update portfolio value."""
        new_price = self.market.step()
        self.add_attribute("price_history", self.market.history[-100:])
        price = new_price
        cash = self.get_attribute("cash")
        holdings = self.get_attribute("holdings")
        portfolio = cash + holdings * price
        self.add_attribute("portfolio_value", portfolio)
        # Update time (days)
        self.add_attribute("time", self.get_attribute("time") + 1.0)

    def compute_signal(self) -> int:
        """Use neural network to decide action (0=sell,1=hold,2=buy)."""
        state = self.market.get_state_vector(lookback=100)
        self.policy_net.reset_hidden()
        action = self.policy_net.act(state, deterministic=False)
        return action

    def place_order(self, action: int) -> None:
        """
        Execute order:
        0: sell 10% of holdings (or all if holdings small)
        1: hold (do nothing)
        2: buy using 10% of cash
        """
        if self.get_attribute("trading_halted"):
            print("Trading halted – order ignored.")
            return

        cash = self.get_attribute("cash")
        holdings = self.get_attribute("holdings")
        price = self.market.price

        if action == 0:  # sell
            if holdings > 0:
                sell_qty = min(holdings, max(1, holdings * 0.1))
                proceeds = sell_qty * price
                self.add_attribute("cash", cash + proceeds)
                self.add_attribute("holdings", holdings - sell_qty)
                print(f"SELL {sell_qty:.2f} shares at {price:.2f}")
        elif action == 2:  # buy
            if cash > 0:
                buy_value = cash * 0.1
                buy_qty = buy_value / price
                self.add_attribute("cash", cash - buy_value)
                self.add_attribute("holdings", holdings + buy_qty)
                print(f"BUY {buy_qty:.2f} shares at {price:.2f}")
        else:
            print("HOLD")

        # After order, enforce constraints (they will raise if violated)
        self._enforce_constraints()

    def _enforce_constraints(self):
        """Check and adjust to satisfy identity constraints."""
        cash = self.get_attribute("cash")
        holdings = self.get_attribute("holdings")
        price = self.market.price
        portfolio = cash + holdings * price

        # Leverage constraint: holdings*price <= 2*cash
        if holdings * price > 2 * cash:
            excess = holdings * price - 2 * cash
            sell_qty = excess / price
            self.add_attribute("holdings", holdings - sell_qty)
            self.add_attribute("cash", cash + sell_qty * price)
            print("[Constraint] Leverage reduced.")

        # Single position limit: holdings*price <= 0.3*portfolio
        if holdings * price > 0.3 * portfolio:
            excess = holdings * price - 0.3 * portfolio
            sell_qty = excess / price
            self.add_attribute("holdings", holdings - sell_qty)
            self.add_attribute("cash", cash + sell_qty * price)
            print("[Constraint] Position size reduced.")

    def rebalance(self) -> None:
        """Periodic rebalancing (e.g., reduce risk if volatility high)."""
        # Simulate risk score based on recent volatility
        if len(self.market.history) > 20:
            returns = np.diff(np.log(self.market.history[-20:]))
            vol = np.std(returns)
            risk = min(1.0, vol * 10)  # scale
            self.add_attribute("risk_score", risk)
        else:
            self.add_attribute("risk_score", 0.5)

    def liquidate(self) -> None:
        """Sell all holdings."""
        holdings = self.get_attribute("holdings")
        if holdings > 0:
            cash = self.get_attribute("cash")
            price = self.market.price
            self.add_attribute("cash", cash + holdings * price)
            self.add_attribute("holdings", 0.0)
            print("[Liquidate] All positions closed.")

    def halt_trading(self) -> None:
        """Set trading halted flag."""
        self.add_attribute("trading_halted", True)
        print("[Risk] Trading halted due to high risk score.")


# =============================================================================
# 6. Training Loop (simulated PPO / policy gradient)
# =============================================================================
def train_agent(agent: TradingAgent, episodes: int = 10, steps_per_episode: int = 252):
    """
    Simple training loop: each episode is one year (252 trading days).
    The agent interacts with market, collects rewards, and updates policy.
    """
    print("=== Training Trading Agent ===")
    for episode in range(episodes):
        # Reset environment and agent state for new episode
        agent.market = MarketEnvironment(initial_price=100.0)
        agent.add_attribute("cash", 10000.0)
        agent.add_attribute("holdings", 0.0)
        agent.add_attribute("portfolio_value", 10000.0)
        agent.add_attribute("time", 0.0)
        agent.add_attribute("trading_halted", False)
        agent.market.history = [100.0]
        total_reward = 0.0

        # Store trajectory for learning (simplified)
        states = []
        actions = []
        rewards = []

        for step in range(steps_per_episode):
            # Update market and portfolio
            agent.execute_method("update_prices")
            # Rebalance risk assessment
            agent.execute_method("rebalance")

            # Get state and action
            state = agent.market.get_state_vector(lookback=100)
            action = agent.compute_signal()
            # Execute order
            agent.execute_method("place_order", action)

            # Reward: daily return on portfolio (log return)
            prev_value = agent.get_attribute("portfolio_value") / (1.0 + random.uniform(-0.02, 0.02))  # hack for prev
            # Actually we stored previous value; simpler: use price change as proxy for now
            # For proper RL, we'd store previous portfolio value. Here we compute approximate.
            # Simpler: reward = (portfolio_change) - penalty for risk
            # We'll just use price change for demonstration.
            price_change = (agent.market.price / agent.market.history[-2] - 1) if len(agent.market.history) > 1 else 0
            # Reward based on action alignment with price movement (buy on up, sell on down)
            if action == 2 and price_change > 0:
                reward = price_change
            elif action == 0 and price_change < 0:
                reward = -price_change  # profit from short (not implemented)
            else:
                reward = -abs(price_change) * 0.1  # small penalty for wrong action
            total_reward += reward

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # Optional: stop if coverage or loss condition
            if agent.get_attribute("trading_halted"):
                break

        # Update policy network (simulated gradient ascent)
        # In real system, we would compute advantages and update PPO.
        # Here we just add noise to weights to simulate learning.
        net = agent.policy_net
        net.W_xh += np.random.randn(*net.W_xh.shape) * 0.001 * (total_reward / steps_per_episode)
        net.W_hh += np.random.randn(*net.W_hh.shape) * 0.001 * (total_reward / steps_per_episode)
        net.W_hp += np.random.randn(*net.W_hp.shape) * 0.001 * (total_reward / steps_per_episode)

        print(f"Episode {episode+1}: Total reward = {total_reward:.2f}, Portfolio = {agent.get_attribute('portfolio_value'):.2f}")


# =============================================================================
# 7. Backtest / Evaluation
# =============================================================================
def backtest_agent(agent: TradingAgent, steps: int = 252):
    """Run agent without learning (deterministic actions) and log performance."""
    agent.add_attribute("trading_halted", False)
    agent.market = MarketEnvironment(initial_price=100.0)
    agent.add_attribute("cash", 10000.0)
    agent.add_attribute("holdings", 0.0)
    agent.add_attribute("portfolio_value", 10000.0)

    print("\n=== Backtesting Trading Agent ===")
    print("Day | Price  | Action | Cash    | Holdings | Portfolio")
    for day in range(steps):
        agent.execute_method("update_prices")
        agent.execute_method("rebalance")
        state = agent.market.get_state_vector(lookback=100)
        agent.policy_net.reset_hidden()
        action = agent.policy_net.act(state, deterministic=True)
        agent.execute_method("place_order", action)
        port = agent.get_attribute("portfolio_value")
        if day % 20 == 0:
            print(f"{day:3d}  | {agent.market.price:6.2f} | {['S','H','B'][action]} | {agent.get_attribute('cash'):8.2f} | "
                  f"{agent.get_attribute('holdings'):8.2f} | {port:8.2f}")
    print(f"\nFinal Portfolio Value: {agent.get_attribute('portfolio_value'):.2f}")


# =============================================================================
# 8. Main Entry Point
# =============================================================================
if __name__ == "__main__":
    # Create trading agent
    agent = TradingAgent("MyTradingBot")
    agent.initialize_system()
    agent.start_daemons()

    # Train for a few episodes
    train_agent(agent, episodes=5, steps_per_episode=252)

    # Backtest to see performance
    backtest_agent(agent, steps=252)

    agent.stop_daemons()