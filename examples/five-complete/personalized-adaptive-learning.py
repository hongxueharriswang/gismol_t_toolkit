#!/usr/bin/env python3
"""
Personalised Adaptive Learning Platform - COH Implementation using GISMOL Toolkit

This example models an adaptive learning system as a Constrained Object Hierarchy (COH).
It tracks student knowledge, recommends personalised activities, assesses progress,
and respects curriculum standards.
"""

import time
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Import GISMOL components
from gismol.core import COHObject, COHRepository
from gismol.core.daemons import ConstraintDaemon
from gismol.neural import NeuralComponent, Classifier, Regressor
from gismol.neural.embeddings import EmbeddingModel
from gismol.reasoners import BaseReasoner, TemporalReasoner, SafetyReasoner


# =============================================================================
# 1. Simulated Domain Models
# =============================================================================

@dataclass
class Topic:
    """A learning topic with prerequisites and standards."""
    name: str
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    prerequisites: List[str]  # list of topic names
    standards: List[str]  # curriculum standards IDs

@dataclass
class LearningActivity:
    """An activity (video, quiz, exercise) with associated topic and difficulty."""
    id: str
    title: str
    topic: str
    difficulty: float
    estimated_duration_min: int
    content_type: str  # "video", "quiz", "exercise", "simulation"
    is_gamified: bool = False

@dataclass
class Student:
    """Student model with knowledge, engagement, and history."""
    id: str
    name: str
    learning_style: str  # "visual", "auditory", "kinesthetic", "reading"
    initial_mastery: Dict[str, float] = field(default_factory=dict)

class LearningWorld:
    """Simulated learning environment with topics, activities, and students."""
    def __init__(self):
        # Define topics (math curriculum)
        self.topics = {
            "arithmetic": Topic("arithmetic", 0.2, [], ["CCSS.Math.3.OA"]),
            "fractions": Topic("fractions", 0.4, ["arithmetic"], ["CCSS.Math.4.NF"]),
            "decimals": Topic("decimals", 0.4, ["arithmetic"], ["CCSS.Math.4.NF"]),
            "algebra_basics": Topic("algebra_basics", 0.6, ["arithmetic"], ["CCSS.Math.6.EE"]),
            "linear_equations": Topic("linear_equations", 0.7, ["algebra_basics"], ["CCSS.Math.8.EE"]),
            "geometry": Topic("geometry", 0.5, ["arithmetic"], ["CCSS.Math.4.G"]),
            "trigonometry": Topic("trigonometry", 0.8, ["geometry"], ["CCSS.Math.HSG"]),
        }
        # Learning activities
        self.activities = [
            LearningActivity("act_001", "Arithmetic Basics Video", "arithmetic", 0.2, 5, "video"),
            LearningActivity("act_002", "Arithmetic Quiz", "arithmetic", 0.3, 10, "quiz"),
            LearningActivity("act_003", "Fractions Introduction", "fractions", 0.4, 8, "video"),
            LearningActivity("act_004", "Fractions Practice", "fractions", 0.5, 15, "exercise"),
            LearningActivity("act_005", "Decimals Explained", "decimals", 0.4, 6, "video"),
            LearningActivity("act_006", "Algebra Basics", "algebra_basics", 0.6, 12, "video"),
            LearningActivity("act_007", "Algebra Drill", "algebra_basics", 0.65, 10, "exercise"),
            LearningActivity("act_008", "Linear Equations", "linear_equations", 0.7, 15, "video"),
            LearningActivity("act_009", "Linear Equations Practice", "linear_equations", 0.75, 20, "exercise"),
            LearningActivity("act_010", "Geometry Shapes", "geometry", 0.5, 8, "video"),
            LearningActivity("act_011", "Geometry Quiz", "geometry", 0.55, 10, "quiz"),
            LearningActivity("act_012", "Trigonometry Basics", "trigonometry", 0.8, 12, "video"),
            LearningActivity("gamified_001", "Math Adventure Game", "arithmetic", 0.25, 20, "simulation", is_gamified=True),
            LearningActivity("gamified_002", "Fraction Defender Game", "fractions", 0.45, 20, "simulation", is_gamified=True),
        ]
        # Students
        self.students = {
            "S1": Student("S1", "Alice", "visual", {"arithmetic": 0.8, "fractions": 0.3, "decimals": 0.2, "algebra_basics": 0.1}),
            "S2": Student("S2", "Bob", "kinesthetic", {"arithmetic": 0.6, "fractions": 0.4}),
            "S3": Student("S3", "Charlie", "reading", {"arithmetic": 0.9, "geometry": 0.2}),
        }
        self.current_student = self.students["S1"]
        self.current_day = 0

    def get_activity_effect(self, activity: LearningActivity, student_mastery: float) -> float:
        """Simulate learning gain: how much mastery increases after activity."""
        # Higher gain when difficulty is slightly above current mastery
        diff = activity.difficulty - student_mastery
        gain = 0.1 * math.exp(-((diff - 0.1) ** 2) / 0.1) + random.uniform(0, 0.05)
        return min(0.3, max(0.01, gain))

    def estimate_engagement(self, student: Student, activity: LearningActivity) -> float:
        """Simulated engagement score based on learning style and activity type."""
        score = 0.5
        if student.learning_style == "visual" and activity.content_type == "video":
            score += 0.3
        elif student.learning_style == "kinesthetic" and activity.content_type in ("exercise", "simulation"):
            score += 0.3
        elif student.learning_style == "reading" and activity.content_type == "quiz":
            score += 0.2
        if activity.is_gamified:
            score += 0.2
        return min(1.0, score)


# =============================================================================
# 2. Neural Components
# =============================================================================

class KnowledgeTracingLSTM(NeuralComponent):
    """Simulates an LSTM that tracks student knowledge over time."""
    def __init__(self, name: str, n_topics: int = 7, hidden_dim: int = 32):
        super().__init__(name, input_dim=n_topics * 2, output_dim=n_topics)
        self.n_topics = n_topics
        self.hidden_dim = hidden_dim
        # Simulated weights
        self.Wxh = np.random.randn(hidden_dim, self.input_dim) * 0.01
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bh = np.zeros(hidden_dim)
        self.Why = np.random.randn(self.output_dim, hidden_dim) * 0.01
        self.by = np.zeros(self.output_dim)
        self.h = np.zeros(hidden_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        self.h = np.tanh(self.Wxh @ x + self.Whh @ self.h + self.bh)
        y = self.Why @ self.h + self.by
        return np.clip(y, 0, 1)

    def update_knowledge(self, topic_mastery: Dict[str, float], activity_topic: str, performance: float) -> Dict[str, float]:
        """Update mastery after an activity."""
        topic_list = list(topic_mastery.keys())
        # Build input: current mastery + performance one-hot
        features = [topic_mastery[t] for t in topic_list] + [performance]
        # Pad to input_dim
        if len(features) < self.input_dim:
            features += [0] * (self.input_dim - len(features))
        out = self.forward(np.array(features))
        new_mastery = {t: float(out[i]) for i, t in enumerate(topic_list)}
        return new_mastery


class EngagementClassifier(Classifier):
    """Predicts student engagement/dropout risk."""
    def __init__(self, name: str, input_dim: int = 10, hidden_dim: int = 16):
        super().__init__(name, input_dim, output_dim=2)  # 0 = low engagement, 1 = high
        self.hidden_dim = hidden_dim
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W2 = np.random.randn(2, hidden_dim) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        h = np.tanh(self.W1 @ x)
        out = self.W2 @ h
        exp_out = np.exp(out - np.max(out))
        return exp_out / np.sum(exp_out)

    def predict_engagement(self, features: np.ndarray) -> float:
        probs = self.forward(features)
        return probs[1]  # probability of high engagement


class ContentEmbedder(EmbeddingModel):
    """Embedding for learning objects mapping to skills."""
    def __init__(self, name: str = "content_embedder", embedding_dim: int = 32, n_topics: int = 7):
        super().__init__(name, embedding_dim=embedding_dim)
        self.n_topics = n_topics
        self.proj = np.random.randn(embedding_dim, n_topics) * 0.1

    def embed(self, obj: COHObject) -> np.ndarray:
        # For a learning activity, embed its topic difficulty vector
        # This is a placeholder; in real system would use activity features
        if hasattr(obj, "activity"):
            act = obj.activity
            # Create one-hot for topic index (simplified)
            vec = np.zeros(self.n_topics)
            # In real system, we'd map topic name to index
            emb = self.proj @ vec
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            return emb
        return np.zeros(self.embedding_dim)


# =============================================================================
# 3. Daemons
# =============================================================================

class DropoutPredictorDaemon(ConstraintDaemon):
    """Continuously evaluates engagement trend and alerts teacher."""
    def __init__(self, parent: COHObject, interval: float = 10.0):
        super().__init__(parent, interval)
        self.engagement_history = deque(maxlen=10)

    def check(self) -> None:
        student = self.parent.get_attribute("current_student")
        if student is None:
            return
        engagement = self.parent.get_attribute("engagement_score", 0.5)
        self.engagement_history.append(engagement)
        if len(self.engagement_history) >= 5:
            trend = self.engagement_history[-1] - self.engagement_history[0]
            if trend < -0.2:
                print(f"[DropoutPredictor] Student {student.name} engagement dropping! Alert teacher.")


class FairnessMonitor(ConstraintDaemon):
    """Checks for bias in recommendations across demographic groups."""
    def __init__(self, parent: COHObject, interval: float = 30.0):
        super().__init__(parent, interval)
        self.recommendation_log = []

    def check(self) -> None:
        # Simplified: log if any student gets significantly harder content than peers
        current_student = self.parent.get_attribute("current_student")
        if current_student:
            recent_activity = self.parent.get_attribute("last_activity")
            if recent_activity:
                self.recommendation_log.append((current_student.id, recent_activity.difficulty))
                # Check if this student's average difficulty is >0.2 above others
                # (simplified)
                pass


# =============================================================================
# 4. Main Adaptive Learning Platform COH Object
# =============================================================================

class AdaptiveLearningPlatform(COHObject):
    """
    Personalised adaptive learning platform with knowledge tracing,
    activity recommendation, and constraint-aware adaptation.
    """
    def __init__(self, name: str = "AdaptiveLearning", world: Optional[LearningWorld] = None):
        super().__init__(name)
        self.world = world or LearningWorld()

        # ---- Attributes (A) ----
        self.add_attribute("world", self.world)
        self.add_attribute("current_student", self.world.current_student)
        self.add_attribute("student_knowledge", self.world.current_student.initial_mastery.copy())
        self.add_attribute("engagement_score", 0.7)
        self.add_attribute("session_duration_min", 0.0)
        self.add_attribute("learning_style", self.world.current_student.learning_style)
        self.add_attribute("last_activity", None)
        self.add_attribute("activity_history", [])

        # ---- Neural Components (N) ----
        n_topics = len(self.world.topics)
        self.knowledge_tracer = KnowledgeTracingLSTM(name="knowledge_tracer", n_topics=n_topics)
        self.add_neural_component("knowledge_tracer", self.knowledge_tracer)
        self.engagement_classifier = EngagementClassifier(name="engagement_clf", input_dim=10)
        self.add_neural_component("engagement_classifier", self.engagement_classifier)
        self.content_embedder = ContentEmbedder(name="content_embedder", n_topics=n_topics)
        self.add_neural_component("embedder", self.content_embedder, is_embedding_model=True)

        # ---- Methods (M) ----
        self.add_method("assess_knowledge", self.assess_knowledge)
        self.add_method("recommend_next_activity", self.recommend_next_activity)
        self.add_method("generate_quiz", self.generate_quiz)
        self.add_method("update_mastery", self.update_mastery)
        self.add_method("flag_struggling_student", self.flag_struggling_student)
        self.add_method("daily_update", self.daily_update)

        # ---- Identity Constraints (I) ----
        for topic in self.world.topics.values():
            for std in topic.standards:
                self.add_identity_constraint({
                    'name': f'standard_{std}',
                    'specification': f'curriculum_coverage.{std} >= 1.0',
                    'severity': 9,
                    'category': 'curriculum'
                })
        self.add_identity_constraint({
            'name': 'privacy',
            'specification': 'student_data_shared == False',
            'severity': 10
        })
        self.add_identity_constraint({
            'name': 'assessment_validity',
            'specification': 'quiz_questions aligned_with_standards',
            'severity': 8
        })

        # ---- Trigger Constraints (T) ----
        self.add_trigger_constraint({
            'name': 'low_engagement_trigger',
            'specification': 'WHEN engagement_score < 0.3 DO recommend_break_or_gamified_content()',
            'priority': 'HIGH'
        })
        self.add_trigger_constraint({
            'name': 'mastery_advance',
            'specification': 'WHEN mastery(topic) > 0.8 DO advance_to_next_topic()',
            'priority': 'MEDIUM'
        })
        self.add_trigger_constraint({
            'name': 'struggle_trigger',
            'specification': 'WHEN student_struggles > 3 DO provide_hint()',
            'priority': 'HIGH'
        })

        # ---- Goal Constraints (G) ----
        self.add_goal_constraint({
            'name': 'learning_gain',
            'specification': 'MAXIMIZE Σ(mastery_gain) / session_duration',
            'priority': 'HIGH'
        })
        self.add_goal_constraint({
            'name': 'frustration_minimization',
            'specification': 'MINIMIZE Σ(time_spent_struggling)',
            'priority': 'MEDIUM'
        })
        self.add_goal_constraint({
            'name': 'curriculum_coverage',
            'specification': 'MAXIMIZE Σ(covered_standards) / total_standards',
            'priority': 'HIGH'
        })

        # ---- Daemons (D) ----
        dropout_daemon = DropoutPredictorDaemon(self, interval=10.0)
        self.daemons['dropout_predictor'] = dropout_daemon
        fairness_daemon = FairnessMonitor(self, interval=30.0)
        self.daemons['fairness_monitor'] = fairness_daemon

        # Register reasoners
        self.constraint_system.register_reasoner("temporal", TemporalReasoner())
        self.constraint_system.register_reasoner("safety", SafetyReasoner())

    def assess_knowledge(self, topic: str) -> float:
        """Simulate a quiz to assess mastery."""
        current = self.get_attribute("student_knowledge", {}).get(topic, 0.0)
        # Simulate assessment: measured mastery may differ slightly
        measured = max(0.0, min(1.0, current + random.gauss(0, 0.05)))
        return measured

    def recommend_next_activity(self) -> Optional[LearningActivity]:
        """Recommend the most suitable activity based on knowledge and engagement."""
        knowledge = self.get_attribute("student_knowledge", {})
        engagement = self.get_attribute("engagement_score", 0.5)
        learning_style = self.get_attribute("learning_style")

        # Find topic with lowest mastery (but not too low)
        candidate_topics = [t for t, m in knowledge.items() if m < 0.8]
        if not candidate_topics:
            return None
        target_topic = min(candidate_topics, key=lambda t: knowledge[t])
        # Filter activities for that topic
        candidates = [a for a in self.world.activities if a.topic == target_topic]
        if not candidates:
            return None
        # Score each activity based on engagement prediction
        def score(act):
            s = 0.5
            # Match learning style
            if learning_style == "visual" and act.content_type == "video":
                s += 0.3
            elif learning_style == "kinesthetic" and act.content_type in ("exercise", "simulation"):
                s += 0.3
            elif learning_style == "reading" and act.content_type == "quiz":
                s += 0.2
            if act.is_gamified:
                s += 0.2
            # Difficulty should be slightly above current mastery
            diff = act.difficulty - knowledge[target_topic]
            if 0.1 < diff < 0.4:
                s += 0.2
            elif diff < -0.2:
                s -= 0.2
            return s
        best = max(candidates, key=score)
        return best

    def generate_quiz(self, topic: str, difficulty: float) -> List[Dict]:
        """Generate a quiz with questions at specified difficulty."""
        num_questions = max(3, min(10, int(5 + difficulty * 5)))
        questions = [{"text": f"Sample question {i+1} on {topic}", "difficulty": difficulty} for i in range(num_questions)]
        return questions

    def update_mastery(self, activity: LearningActivity, performance: float) -> None:
        """Update student's mastery after completing an activity."""
        current = self.get_attribute("student_knowledge", {})
        # Use knowledge tracer to update
        new_mastery = self.knowledge_tracer.update_knowledge(current, activity.topic, performance)
        self.add_attribute("student_knowledge", new_mastery)
        # Also update engagement score based on performance and activity type
        old_eng = self.get_attribute("engagement_score", 0.5)
        delta = (performance - 0.5) * 0.2 + (0.1 if activity.is_gamified else 0)
        new_eng = max(0.0, min(1.0, old_eng + delta))
        self.add_attribute("engagement_score", new_eng)

    def flag_struggling_student(self) -> None:
        """Alert teacher if student is struggling."""
        student = self.get_attribute("current_student")
        print(f"[FLAG] Student {student.name} is struggling. Teacher notified.")

    def provide_hint(self) -> None:
        """Provide a hint for the current activity."""
        print("[HINT] Here's a hint to help you progress.")

    def advance_to_next_topic(self) -> None:
        """Advance to the next recommended topic."""
        knowledge = self.get_attribute("student_knowledge", {})
        mastered_topics = [t for t, m in knowledge.items() if m > 0.8]
        # Find a topic whose prerequisites are all mastered
        for topic, obj in self.world.topics.items():
            if all(p in mastered_topics for p in obj.prerequisites) and topic not in mastered_topics:
                print(f"[ADVANCE] Moving to next topic: {topic}")
                break

    def recommend_break_or_gamified_content(self) -> None:
        """Suggest a break or gamified activity when engagement is low."""
        print("[RECOMMEND] Engagement low. Try a gamified activity or take a short break.")

    def daily_update(self, day: int) -> None:
        """Run daily learning session for the current student."""
        self.world.current_day = day
        student = self.get_attribute("current_student")
        print(f"\n=== Day {day}: Student {student.name} ===")

        # Assess initial knowledge (simulate quiz)
        for topic in list(self.get_attribute("student_knowledge").keys())[:3]:
            measured = self.assess_knowledge(topic)
            print(f"  {topic} mastery (assessed): {measured:.2f}")

        # Recommend activity
        activity = self.recommend_next_activity()
        if activity is None:
            print("No suitable activity found. Student has mastered all topics?")
            return
        print(f"Recommended: {activity.title} (difficulty {activity.difficulty})")

        # Simulate student completing activity
        current_mastery = self.get_attribute("student_knowledge", {}).get(activity.topic, 0.0)
        # Performance depends on difference between difficulty and mastery
        diff = activity.difficulty - current_mastery
        base_perf = 0.7 - diff * 0.5  # harder -> lower performance
        performance = max(0.0, min(1.0, base_perf + random.gauss(0, 0.1)))
        print(f"Student performance: {performance:.2f}")

        # Update mastery and engagement
        self.update_mastery(activity, performance)
        self.add_attribute("last_activity", activity)
        history = self.get_attribute("activity_history", [])
        history.append(activity.id)
        self.add_attribute("activity_history", history)
        self.add_attribute("session_duration_min", self.get_attribute("session_duration_min") + activity.estimated_duration_min)

        # Check for struggling (if performance low multiple times)
        struggle_count = self.get_attribute("struggle_count", 0)
        if performance < 0.4:
            struggle_count += 1
            self.add_attribute("struggle_count", struggle_count)
            if struggle_count >= 3:
                self.execute_method("flag_struggling_student")
        else:
            self.add_attribute("struggle_count", 0)

        # Log current knowledge
        knowledge = self.get_attribute("student_knowledge")
        print(f"Current mastery: {knowledge}")
        print(f"Engagement score: {self.get_attribute('engagement_score'):.2f}")


# =============================================================================
# 5. Simulation
# =============================================================================

def run_simulation(days: int = 5):
    """Run adaptive learning simulation for given number of days."""
    world = LearningWorld()
    platform = AdaptiveLearningPlatform("AdaptiveMath", world)

    platform.initialize_system()
    platform.start_daemons()

    print("=== Personalised Adaptive Learning Platform Simulation ===")
    for day in range(days):
        platform.execute_method("daily_update", day)
        time.sleep(1.0)  # slow for readability

    platform.stop_daemons()
    print("\n=== Simulation Complete ===")
    final_knowledge = platform.get_attribute("student_knowledge")
    print(f"Final student mastery: {final_knowledge}")


if __name__ == "__main__":
    run_simulation(days=5)