#!/usr/bin/env python3
"""
Ethical AI for Social Welfare Allocation - COH Implementation using GISMOL Toolkit

This example models a social welfare allocation system as a Constrained Object Hierarchy (COH).
It assesses applicant needs, ranks them with fairness constraints, detects fraud,
provides explainable decisions, and handles appeals.
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
from gismol.neural import NeuralComponent, Classifier, AnomalyDetector
from gismol.neural.embeddings import EmbeddingModel
from gismol.nlp import ConstraintAwareResponseGenerator, IntentRecognizer
from gismol.reasoners import BaseReasoner, SafetyReasoner, GeneralReasoner


# =============================================================================
# 1. Simulated Domain Models
# =============================================================================

@dataclass
class Applicant:
    """An applicant for social welfare."""
    id: str
    name: str
    age: int
    gender: str  # "M", "F", "NB"
    race: str    # demographic group for fairness monitoring
    income: float  # annual household income
    family_size: int
    disability_status: bool
    housing_status: str  # "homeless", "unstable", "stable"
    region: str
    application_date: int  # day since epoch
    priority_score: float = 0.0
    allocated_amount: float = 0.0
    appeal_filed: bool = False
    fraud_risk: float = 0.0

@dataclass
class WelfareProgram:
    """A social welfare program with budget and eligibility rules."""
    name: str
    budget: float
    spent: float
    unit_value: float  # e.g., voucher value per person
    eligibility_rules: List[str]  # simple conditions
    target_population: str  # "general", "disabled", "families", "seniors"

class SocialWorld:
    """Simulated social welfare environment."""
    def __init__(self):
        # Programs
        self.programs = {
            "housing_voucher": WelfareProgram(
                "housing_voucher", budget=500000, spent=0, unit_value=5000,
                eligibility_rules=["housing_status != 'stable'"],
                target_population="general"
            ),
            "food_assistance": WelfareProgram(
                "food_assistance", budget=300000, spent=0, unit_value=300,
                eligibility_rules=["income < 20000"],
                target_population="families"
            ),
            "disability_support": WelfareProgram(
                "disability_support", budget=200000, spent=0, unit_value=2000,
                eligibility_rules=["disability_status == True"],
                target_population="disabled"
            ),
        }
        # Applicants (generated)
        self.applicants: Dict[str, Applicant] = {}
        self._generate_applicants()
        self.current_day = 0
        self.appeals_queue: List[str] = []  # applicant IDs with appeals
        self.decision_log: List[Dict] = []

    def _generate_applicants(self):
        """Generate 20 synthetic applicants."""
        names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                 "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
                 "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
        regions = ["North", "South", "East", "West"]
        for i in range(1, 21):
            name = names[i-1]
            gender = random.choice(["M", "F", "NB"])
            race = random.choice(["A", "B", "C", "D"])  # anonymized groups
            income = random.uniform(0, 50000)
            family_size = random.randint(1, 6)
            disability = random.random() < 0.15
            housing = random.choices(["homeless", "unstable", "stable"], weights=[0.3, 0.4, 0.3])[0]
            region = random.choice(regions)
            self.applicants[f"A{i:03d}"] = Applicant(
                id=f"A{i:03d}", name=name, age=random.randint(18, 80),
                gender=gender, race=race, income=income, family_size=family_size,
                disability_status=disability, housing_status=housing, region=region,
                application_date=random.randint(0, 100)
            )

    def get_eligible_programs(self, applicant: Applicant) -> List[str]:
        """Return programs for which applicant is eligible."""
        eligible = []
        for prog_name, prog in self.programs.items():
            eligible_flag = True
            for rule in prog.eligibility_rules:
                if not eval(rule, {"applicant": applicant, "self": applicant}):
                    eligible_flag = False
                    break
            if eligible_flag:
                eligible.append(prog_name)
        return eligible

    def allocate(self, applicant_id: str, program: str, amount: float) -> bool:
        """Allocate welfare to an applicant if budget remains."""
        prog = self.programs.get(program)
        if prog and prog.spent + amount <= prog.budget:
            prog.spent += amount
            self.applicants[applicant_id].allocated_amount += amount
            self.decision_log.append({
                "applicant_id": applicant_id,
                "program": program,
                "amount": amount,
                "day": self.current_day,
                "fraud_risk": self.applicants[applicant_id].fraud_risk
            })
            return True
        return False


# =============================================================================
# 2. Neural Components
# =============================================================================

class FairnessConstrainedRanker(NeuralComponent):
    """
    Neural ranker with demographic parity penalty.
    Simulates a neural network that learns to rank applicants while minimizing bias.
    """
    def __init__(self, name: str, input_dim: int = 12, hidden_dim: int = 32):
        super().__init__(name, input_dim=input_dim, output_dim=1)
        self.hidden_dim = hidden_dim
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W2 = np.random.randn(1, hidden_dim) * 0.01
        self.demographic_parity_penalty = 0.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        h = np.tanh(self.W1 @ x)
        score = self.W2 @ h
        return score

    def rank(self, applicants: List[Applicant], features: Dict[str, np.ndarray]) -> List[Tuple[Applicant, float]]:
        """Rank applicants by predicted priority score."""
        scored = []
        for app in applicants:
            feat = features.get(app.id, np.random.randn(self.input_dim))
            score = float(self.forward(feat)[0])
            scored.append((app, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def compute_fairness_penalty(self, allocations: Dict[str, float], demographic_attr: str = "race") -> float:
        """Compute demographic parity violation (simplified)."""
        groups = defaultdict(list)
        for app_id, amount in allocations.items():
            # In real system, would look up group membership
            groups["group_" + str(hash(app_id) % 4)].append(amount)
        # Compute variance of mean allocations across groups
        means = [np.mean(g) for g in groups.values() if g]
        if len(means) > 1:
            return float(np.std(means))
        return 0.0


class ExplainabilityModule(NeuralComponent):
    """Generates natural language explanations for allocation decisions."""
    def __init__(self, name: str, hidden_dim: int = 64):
        super().__init__(name, input_dim=hidden_dim, output_dim=1)  # simplified
        self.templates = [
            "Applicant {name} received {amount} because their need score was {score:.2f}.",
            "Based on income ({income:.0f}) and family size ({family}), the allocation is {amount}.",
            "Priority given due to {reason}. Amount: {amount}.",
        ]

    def explain(self, applicant: Applicant, amount: float, score: float, reason: str) -> str:
        template = random.choice(self.templates)
        return template.format(
            name=applicant.name,
            amount=amount,
            score=score,
            income=applicant.income,
            family=applicant.family_size,
            reason=reason
        )


class FraudDetector(AnomalyDetector):
    """Detects fraudulent applications based on anomaly patterns."""
    def __init__(self, name: str, input_dim: int = 10, hidden_dim: int = 16):
        super().__init__(name, input_dim, hidden_dim)
        self.threshold = 0.7

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Simple autoencoder: return reconstruction
        return x  # identity for simulation

    def anomaly_score(self, x: np.ndarray) -> float:
        # Simulate score based on input features
        return float(np.mean(np.abs(x))) * 2  # simplified

    def detect_fraud(self, applicant: Applicant) -> float:
        """Return fraud risk score between 0 and 1."""
        # Create feature vector from applicant data
        features = np.array([
            applicant.income / 100000,
            applicant.family_size / 10,
            float(applicant.disability_status),
            hash(applicant.housing_status) % 10 / 10.0,
            applicant.age / 100,
        ])
        score = self.anomaly_score(features)
        return min(1.0, score)


# =============================================================================
# 3. Custom Embedding
# =============================================================================

class ApplicantEmbedding(EmbeddingModel):
    """Embedding of applicant features + historical allocations."""
    def __init__(self, name: str = "applicant_embedder", embedding_dim: int = 32):
        super().__init__(name, embedding_dim=embedding_dim)

    def embed(self, obj: COHObject) -> np.ndarray:
        # Expecting the parent object to provide an applicant reference
        applicant = obj.get_attribute("current_applicant")
        if applicant is None:
            return np.zeros(self.embedding_dim)
        features = np.array([
            applicant.income / 100000,
            applicant.family_size / 10,
            float(applicant.disability_status),
            hash(applicant.housing_status) % 10 / 10.0,
            applicant.age / 100,
            applicant.priority_score,
            applicant.fraud_risk
        ])
        # Pad or truncate
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        else:
            features = features[:self.embedding_dim]
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        return features


# =============================================================================
# 4. Daemons
# =============================================================================

class FairnessAuditor(ConstraintDaemon):
    """Continuously computes bias metrics, alerts if thresholds exceeded."""
    def __init__(self, parent: COHObject, interval: float = 30.0, disparity_threshold: float = 0.1):
        super().__init__(parent, interval)
        self.disparity_threshold = disparity_threshold
        self.allocation_history = []

    def check(self) -> None:
        system = self.parent
        # Get recent allocations from the world
        world = system.get_attribute("world")
        if world is None:
            return
        # Group allocations by race (simplified: use race attribute)
        groups = defaultdict(list)
        for log in world.decision_log[-50:]:  # last 50 decisions
            app = world.applicants.get(log["applicant_id"])
            if app:
                groups[app.race].append(log["amount"])
        # Compute mean per group
        means = {g: np.mean(vals) if vals else 0 for g, vals in groups.items()}
        if len(means) > 1:
            max_mean = max(means.values())
            min_mean = min(means.values())
            disparity = max_mean - min_mean if max_mean > 0 else 0
            if disparity > self.disparity_threshold:
                print(f"[FairnessAuditor] Disparity detected: {disparity:.3f} (threshold {self.disparity_threshold})")
                # Could trigger re-weighting or alert


class BudgetMonitor(ConstraintDaemon):
    """Tracks spending rate, prevents overshoot."""
    def __init__(self, parent: COHObject, interval: float = 10.0, warning_ratio: float = 0.9):
        super().__init__(parent, interval)
        self.warning_ratio = warning_ratio

    def check(self) -> None:
        world = self.parent.get_attribute("world")
        if world is None:
            return
        for prog_name, prog in world.programs.items():
            spend_ratio = prog.spent / prog.budget if prog.budget > 0 else 0
            if spend_ratio > self.warning_ratio:
                print(f"[BudgetMonitor] Program {prog_name} at {spend_ratio*100:.1f}% of budget!")
                if spend_ratio > 0.98:
                    print(f"[BudgetMonitor] {prog_name} nearly exhausted. Halting new allocations.")
                    self.parent.add_attribute(f"halt_{prog_name}", True)


class AppealTracker(ConstraintDaemon):
    """Logs overturned decisions for continuous learning."""
    def __init__(self, parent: COHObject, interval: float = 60.0):
        super().__init__(parent, interval)
        self.overturned = []

    def check(self) -> None:
        world = self.parent.get_attribute("world")
        if world is None:
            return
        # Check for appeals that were resolved in favor of applicant
        for app_id in world.appeals_queue[:]:
            app = world.applicants.get(app_id)
            if app and app.appeal_filed and app.allocated_amount > 0:
                # Appeal resolved positively
                self.overturned.append({"applicant_id": app_id, "amount": app.allocated_amount})
                world.appeals_queue.remove(app_id)
                print(f"[AppealTracker] Appeal for {app_id} resolved. Learning from decision.")


# =============================================================================
# 5. Main Ethical Welfare Allocation System
# =============================================================================

class EthicalWelfareSystem(COHObject):
    """
    Ethical AI system for social welfare allocation with fairness, explainability,
    fraud detection, and appeals handling.
    """
    def __init__(self, name: str = "WelfareAllocator", world: Optional[SocialWorld] = None):
        super().__init__(name)
        self.world = world or SocialWorld()

        # ---- Attributes (A) ----
        self.add_attribute("world", self.world)
        self.add_attribute("current_applicant", None)
        self.add_attribute("allocation_decisions", [])
        self.add_attribute("fairness_metrics", {})

        # ---- Neural Components (N) ----
        self.ranker = FairnessConstrainedRanker(name="ranker", input_dim=12)
        self.add_neural_component("ranker", self.ranker)
        self.explainer = ExplainabilityModule(name="explainer")
        self.add_neural_component("explainer", self.explainer)
        self.fraud_detector = FraudDetector(name="fraud_detector", input_dim=5)
        self.add_neural_component("fraud_detector", self.fraud_detector)

        # ---- Embedding (E) ----
        embedder = ApplicantEmbedding(name="applicant_embedder", embedding_dim=32)
        self.add_neural_component("embedder", embedder, is_embedding_model=True)

        # ---- Methods (M) ----
        self.add_method("assess_need", self.assess_need)
        self.add_method("compute_priority_score", self.compute_priority_score)
        self.add_method("allocate_resource", self.allocate_resource)
        self.add_method("log_decision", self.log_decision)
        self.add_method("review_appeal", self.review_appeal)
        self.add_method("daily_allocation_cycle", self.daily_allocation_cycle)

        # ---- Identity Constraints (I) ----
        self.add_identity_constraint({
            'name': 'minimum_entitlement',
            'specification': 'allocated_amount >= 0',
            'severity': 9
        })
        self.add_identity_constraint({
            'name': 'no_discrimination',
            'specification': 'fairness_metrics.disparity <= 0.1',
            'severity': 10,
            'category': 'ethics'
        })
        self.add_identity_constraint({
            'name': 'budget_compliance',
            'specification': 'world.programs[program].spent <= world.programs[program].budget',
            'severity': 10
        })
        self.add_identity_constraint({
            'name': 'transparency',
            'specification': 'every_decision_has_explanation == True',
            'severity': 8
        })

        # ---- Trigger Constraints (T) ----
        self.add_trigger_constraint({
            'name': 'fraud_alert',
            'specification': 'WHEN fraud_risk > 0.8 DO flag_for_review(current_applicant)',
            'priority': 'HIGH'
        })
        self.add_trigger_constraint({
            'name': 'appeal_review',
            'specification': 'WHEN appeal_filed == True DO review_appeal(applicant_id)',
            'priority': 'MEDIUM'
        })
        self.add_trigger_constraint({
            'name': 'budget_exhaustion',
            'specification': 'WHEN halt_program == True DO pause_allocations(program)',
            'priority': 'HIGH'
        })

        # ---- Goal Constraints (G) ----
        self.add_goal_constraint({
            'name': 'maximize_welfare_impact',
            'specification': 'MAXIMIZE Σ(need_score * allocation_amount) subject to fairness_constraints',
            'priority': 'HIGH'
        })
        self.add_goal_constraint({
            'name': 'fairness_constraints',
            'specification': 'demographic_parity <= 0.1 AND equal_opportunity >= 0.8',
            'priority': 'CRITICAL'
        })

        # ---- Daemons (D) ----
        fairness_auditor = FairnessAuditor(self, interval=30.0, disparity_threshold=0.1)
        self.daemons['fairness_auditor'] = fairness_auditor
        budget_monitor = BudgetMonitor(self, interval=10.0, warning_ratio=0.9)
        self.daemons['budget_monitor'] = budget_monitor
        appeal_tracker = AppealTracker(self, interval=60.0)
        self.daemons['appeal_tracker'] = appeal_tracker

        # Register reasoners
        self.constraint_system.register_reasoner("ethics", SafetyReasoner())
        self.constraint_system.register_reasoner("general", GeneralReasoner())

    def assess_need(self, applicant: Applicant) -> float:
        """Compute a raw need score based on income, family size, housing, disability."""
        # Basic formula
        income_factor = max(0, (30000 - applicant.income) / 30000)
        family_factor = applicant.family_size / 6
        housing_factor = {"homeless": 1.0, "unstable": 0.6, "stable": 0.2}[applicant.housing_status]
        disability_factor = 0.3 if applicant.disability_status else 0
        need = income_factor * 0.4 + family_factor * 0.2 + housing_factor * 0.3 + disability_factor * 0.1
        return min(1.0, need)

    def compute_priority_score(self, applicant: Applicant) -> float:
        """Use neural ranker to compute priority score, incorporating need and fraud risk."""
        need = self.assess_need(applicant)
        # Build feature vector
        features = np.array([
            need,
            applicant.income / 100000,
            applicant.family_size / 10,
            float(applicant.disability_status),
            hash(applicant.housing_status) % 10 / 10.0,
            applicant.age / 100,
            applicant.fraud_risk,
            # Add some synthetic features
            random.gauss(0.5, 0.2)
        ])
        # Pad to input_dim
        if len(features) < 12:
            features = np.pad(features, (0, 12 - len(features)))
        score = float(self.ranker.forward(features)[0])
        # Adjust for fraud risk: lower score if high fraud
        adjusted = score * (1 - applicant.fraud_risk * 0.5)
        return max(0.0, min(1.0, adjusted))

    def allocate_resource(self, applicant: Applicant, program: str) -> Tuple[bool, float, str]:
        """Attempt to allocate welfare to applicant, return (success, amount, explanation)."""
        # Check if program is halted
        if self.get_attribute(f"halt_{program}", False):
            return False, 0.0, f"Program {program} budget exhausted."
        prog = self.world.programs.get(program)
        if not prog:
            return False, 0.0, "Invalid program."
        # Check eligibility
        if program not in self.world.get_eligible_programs(applicant):
            return False, 0.0, "Applicant not eligible."
        # Compute allocation amount based on need and priority
        need = self.assess_need(applicant)
        priority = self.compute_priority_score(applicant)
        base_amount = prog.unit_value * (0.5 + need * 0.5) * (0.5 + priority * 0.5)
        amount = min(base_amount, prog.budget - prog.spent)
        if amount <= 0:
            return False, 0.0, "Insufficient budget."
        # Perform allocation
        success = self.world.allocate(applicant.id, program, amount)
        if success:
            explanation = self.explainer.explain(applicant, amount, priority, f"need={need:.2f}")
            return True, amount, explanation
        return False, 0.0, "Allocation failed."

    def log_decision(self, applicant: Applicant, program: str, amount: float, explanation: str) -> None:
        """Log the decision for audit and learning."""
        decisions = self.get_attribute("allocation_decisions", [])
        decisions.append({
            "applicant_id": applicant.id,
            "program": program,
            "amount": amount,
            "explanation": explanation,
            "fraud_risk": applicant.fraud_risk,
            "priority_score": applicant.priority_score
        })
        self.add_attribute("allocation_decisions", decisions)

    def review_appeal(self, applicant_id: str) -> None:
        """Review an appeal, potentially overturn the decision."""
        applicant = self.world.applicants.get(applicant_id)
        if not applicant:
            return
        # Simulate review: if original allocation was low relative to need, increase
        need = self.assess_need(applicant)
        if need > 0.7 and applicant.allocated_amount < 2000:
            # Overturn: increase allocation
            additional = 1000
            # Find a program with remaining budget
            for prog_name, prog in self.world.programs.items():
                if prog.spent + additional <= prog.budget and prog_name in self.world.get_eligible_programs(applicant):
                    self.world.allocate(applicant_id, prog_name, additional)
                    print(f"[Appeal] Overturned: {applicant.name} received additional {additional}")
                    break
        applicant.appeal_filed = False
        self.add_attribute("current_applicant", None)

    def daily_allocation_cycle(self) -> None:
        """Run one day's allocation process for all applicants."""
        self.world.current_day += 1
        print(f"\n=== Day {self.world.current_day}: Welfare Allocation ===")

        # First, detect fraud for all applicants
        for app in self.world.applicants.values():
            app.fraud_risk = self.fraud_detector.detect_fraud(app)
            if app.fraud_risk > 0.8:
                print(f"[Fraud] High risk for {app.name} ({app.fraud_risk:.2f}) - flagged for review")

        # Compute priority scores for all applicants
        scored_apps = []
        for app in self.world.applicants.values():
            app.priority_score = self.compute_priority_score(app)
            scored_apps.append((app.priority_score, app))
        scored_apps.sort(reverse=True)

        # Allocate resources to highest priority applicants first
        for _, app in scored_apps:
            if app.allocated_amount > 0:
                continue  # already allocated
            # Determine best program for this applicant
            eligible = self.world.get_eligible_programs(app)
            if not eligible:
                continue
            # Choose program with highest remaining budget relative to need
            best_prog = None
            best_score = -1
            for prog_name in eligible:
                prog = self.world.programs[prog_name]
                remaining_ratio = (prog.budget - prog.spent) / prog.budget
                score = remaining_ratio * app.priority_score
                if score > best_score:
                    best_score = score
                    best_prog = prog_name
            if best_prog:
                success, amount, explanation = self.allocate_resource(app, best_prog)
                if success:
                    self.log_decision(app, best_prog, amount, explanation)
                    print(f"Allocated {amount:.0f} to {app.name} from {best_prog}: {explanation}")
                else:
                    print(f"Could not allocate to {app.name}: {explanation}")

        # Process appeals
        for app_id in self.world.appeals_queue[:]:
            self.review_appeal(app_id)

        # Generate fairness report
        self._report_fairness()

    def _report_fairness(self):
        """Compute and log fairness metrics."""
        groups = defaultdict(list)
        for log in self.world.decision_log[-100:]:
            app = self.world.applicants.get(log["applicant_id"])
            if app:
                groups[app.race].append(log["amount"])
        means = {g: np.mean(vals) if vals else 0 for g, vals in groups.items()}
        if means:
            disparity = max(means.values()) - min(means.values())
            self.add_attribute("fairness_metrics", {"disparity": disparity, "group_means": means})
            print(f"[Fairness] Disparity: {disparity:.3f}")


# =============================================================================
# 6. Simulation
# =============================================================================

def run_simulation(days: int = 10):
    """Run ethical welfare allocation simulation for given number of days."""
    world = SocialWorld()
    system = EthicalWelfareSystem("EthicalWelfareAI", world)

    system.initialize_system()
    system.start_daemons()

    print("=== Ethical AI for Social Welfare Allocation Simulation ===")
    for day in range(days):
        system.execute_method("daily_allocation_cycle")
        time.sleep(1.0)  # slow for readability

    system.stop_daemons()

    print("\n=== Final Summary ===")
    total_allocated = sum(a.allocated_amount for a in world.applicants.values())
    print(f"Total allocated: {total_allocated:.0f}")
    for prog_name, prog in world.programs.items():
        print(f"{prog_name}: spent {prog.spent:.0f} / {prog.budget:.0f}")
    print(f"Appeals resolved: {len(system.get_attribute('allocation_decisions', []))}")


if __name__ == "__main__":
    run_simulation(days=5)