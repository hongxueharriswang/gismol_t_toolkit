#!/usr/bin/env python3
"""
Simple Dialogue Agent - COH Implementation using GISMOL Toolkit

This example models a customer support chatbot as a Constrained Object Hierarchy (COH).
It recognizes intents, retrieves answers, enforces constraints, and escalates when needed.
"""

import re
import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple

# Import GISMOL components
from gismol.core import COHObject, COHRepository
from gismol.core.daemons import ConstraintDaemon
from gismol.neural import NeuralComponent
from gismol.neural.embeddings import EmbeddingModel
from gismol.nlp import IntentRecognizer, ResponseValidator, COHDialogueManager
from gismol.reasoners import BaseReasoner


# =============================================================================
# 1. Custom Neural Component: Intent Classifier (simulated BERT-like)
# =============================================================================
class IntentClassifier(NeuralComponent):
    """
    Neural component that classifies user input into one of several intents.
    In a real system this would be a fine‑tuned transformer; here we simulate
    with keyword matching but still provide a neural API.
    """
    def __init__(self, name: str = "intent_classifier", **kwargs):
        super().__init__(name, input_dim=384, output_dim=5, **kwargs)
        # Define intent labels and their keyword patterns
        self.intents = {
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
            "product_info": ["product", "features", "specifications", "tell me about"],
            "pricing": ["price", "cost", "how much", "subscription", "plan"],
            "support": ["help", "issue", "problem", "not working", "error"],
            "goodbye": ["bye", "goodbye", "exit", "quit", "see you"]
        }
        self._confidence_threshold = 0.6

    def forward(self, x: np.ndarray) -> np.ndarray:
        # In a real system, x would be a text embedding; here we ignore it
        # and return a dummy probability distribution.
        return np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    def classify(self, text: str) -> Tuple[str, float]:
        """Return intent name and confidence score."""
        text_lower = text.lower()
        best_intent = "unknown"
        best_score = 0.0
        for intent, keywords in self.intents.items():
            for kw in keywords:
                if kw in text_lower:
                    score = 1.0  # simple match
                    if score > best_score:
                        best_score = score
                        best_intent = intent
        # Apply noise to simulate neural uncertainty
        confidence = min(best_score + random.uniform(-0.1, 0.1), 1.0)
        if confidence < 0:
            confidence = 0.0
        return best_intent, confidence


# =============================================================================
# 2. Custom Embedding Model (for semantic similarity)
# =============================================================================
class DialogueEmbedding(EmbeddingModel):
    """Simple embedding: bag‑of‑words hashed to fixed dimension."""
    def __init__(self, name: str = "dialogue_embedder", dim: int = 64, **kwargs):
        super().__init__(name, embedding_dim=dim, **kwargs)
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        """Hash words into a sparse vector of dimension `dim`."""
        vec = np.zeros(self.dim)
        words = re.findall(r'\w+', text.lower())
        for word in words:
            idx = hash(word) % self.dim
            vec[idx] += 1
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_object(self, obj: COHObject) -> np.ndarray:
        # For a COHObject, use its name and recent conversation history
        history = obj.get_attribute("conversation_history", [])
        recent = " ".join(history[-3:]) if history else obj.name
        return self.embed(recent)


# =============================================================================
# 3. Knowledge Base (simple dictionary)
# =============================================================================
KNOWLEDGE_BASE = {
    "greeting": "Hello! How can I assist you today?",
    "product_info": "Our product offers high performance and reliability. Would you like specific features?",
    "pricing": "Pricing starts at $29/month. For detailed plans, please visit our website.",
    "support": "I'm sorry you're having trouble. Can you describe the issue in more detail?",
    "goodbye": "Thank you for chatting! Have a great day.",
    "unknown": "I'm not sure I understand. Could you please rephrase?"
}


# =============================================================================
# 4. Custom Daemon: Profanity Filter
# =============================================================================
class ProfanityFilterDaemon(ConstraintDaemon):
    """Scans every user input and bot response for profanity."""
    PROFANITY_SET = {"badword", "offensive", "curse"}  # simplified

    def __init__(self, parent: COHObject, interval: float = 0.5):
        super().__init__(parent, interval)
        self.last_input = ""

    def check(self) -> None:
        # Monitor the most recent user input (stored as attribute)
        user_input = self.parent.get_attribute("last_user_input", "")
        if user_input != self.last_input:
            self.last_input = user_input
            if self._contains_profanity(user_input):
                print("[Daemon] Profanity detected in user input – logging incident.")
                # Could trigger a safe response or escalate
                self.parent.add_attribute("profanity_detected", True)

    def _contains_profanity(self, text: str) -> bool:
        text_lower = text.lower()
        return any(bad in text_lower for bad in self.PROFANITY_SET)


# =============================================================================
# 5. Main Chatbot COH Object
# =============================================================================
class SupportBot(COHObject):
    """
    A customer support bot implemented as a COHObject.
    It includes intent classification, response generation, constraints, and daemons.
    """
    def __init__(self, name: str = "SupportBot"):
        super().__init__(name)

        # ---- Attributes (A) ----
        self.add_attribute("last_user_input", "")
        self.add_attribute("last_intent", "unknown")
        self.add_attribute("last_confidence", 0.0)
        self.add_attribute("last_response", "")
        self.add_attribute("conversation_history", [])
        self.add_attribute("profanity_detected", False)

        # ---- Neural Components (N) ----
        self.intent_classifier = IntentClassifier(name="intent_clf")
        self.add_neural_component("intent_classifier", self.intent_classifier)

        # ---- Embedding (E) ----
        embedder = DialogueEmbedding(name="bot_embedder", dim=64)
        self.add_neural_component("embedder", embedder, is_embedding_model=True)

        # ---- Methods (M) ----
        self.add_method("process_input", self.process_input)
        self.add_method("generate_response", self.generate_response)
        self.add_method("send_response", self.send_response)
        self.add_method("escalate_to_human", self.escalate_to_human)

        # ---- Identity Constraints (I) ----
        self.add_identity_constraint({
            'name': 'confidence_threshold',
            'specification': 'last_confidence >= 0.6 or last_intent == "unknown"',
            'severity': 7,
            'category': 'quality'
        })
        self.add_identity_constraint({
            'name': 'no_profanity_in_response',
            'specification': 'profanity_detected == False',
            'severity': 9,
            'category': 'safety'
        })

        # ---- Trigger Constraints (T) ----
        # Low confidence -> ask for rephrase
        self.add_trigger_constraint({
            'name': 'low_confidence_fallback',
            'specification': 'WHEN last_confidence < 0.6 AND last_intent != "goodbye" DO generate_response(rephrase=True)',
            'priority': 'HIGH'
        })
        # Escalate if user says "escalate" or "agent"
        self.add_trigger_constraint({
            'name': 'escalation_trigger',
            'specification': 'WHEN "escalate" in last_user_input or "agent" in last_user_input DO escalate_to_human()',
            'priority': 'HIGH'
        })

        # ---- Goal Constraints (G) ----
        # Maximize customer satisfaction (simulated)
        self.add_goal_constraint({
            'name': 'customer_satisfaction',
            'specification': 'MAXIMIZE predicted_satisfaction SUBJECT TO confidence_threshold',
            'priority': 'HIGH'
        })

        # ---- Daemons (D) ----
        profanity_daemon = ProfanityFilterDaemon(self, interval=0.5)
        self.daemons['profanity_filter'] = profanity_daemon

    # ---- Method implementations ----
    def process_input(self, user_input: str) -> None:
        """Classify intent and store results."""
        self.add_attribute("last_user_input", user_input)
        intent, conf = self.intent_classifier.classify(user_input)
        self.add_attribute("last_intent", intent)
        self.add_attribute("last_confidence", conf)

        # Update conversation history
        history = self.get_attribute("conversation_history", [])
        history.append(f"User: {user_input}")
        if len(history) > 20:
            history = history[-20:]
        self.add_attribute("conversation_history", history)

        # If profanity detected, clear flag after processing
        self.add_attribute("profanity_detected", False)

    def generate_response(self, rephrase: bool = False) -> str:
        """Generate response based on intent and constraints."""
        intent = self.get_attribute("last_intent", "unknown")
        conf = self.get_attribute("last_confidence", 0.0)

        if rephrase or (conf < 0.6 and intent != "goodbye"):
            return "I didn't quite catch that. Could you please rephrase your question?"

        if intent == "goodbye":
            response = KNOWLEDGE_BASE["goodbye"]
        elif intent == "greeting":
            response = KNOWLEDGE_BASE["greeting"]
        elif intent == "product_info":
            response = KNOWLEDGE_BASE["product_info"]
        elif intent == "pricing":
            response = KNOWLEDGE_BASE["pricing"]
        elif intent == "support":
            response = KNOWLEDGE_BASE["support"]
        else:
            response = KNOWLEDGE_BASE["unknown"]

        # Simulate satisfaction prediction (just a random number for demo)
        satisfaction = random.uniform(0.7, 0.95) if intent != "unknown" else 0.4
        self.add_attribute("predicted_satisfaction", satisfaction)

        self.add_attribute("last_response", response)
        return response

    def send_response(self, response: str) -> None:
        """Output the response and log it."""
        print(f"Bot: {response}")
        history = self.get_attribute("conversation_history", [])
        history.append(f"Bot: {response}")
        self.add_attribute("conversation_history", history)

    def escalate_to_human(self) -> None:
        """Transfer the conversation to a human agent."""
        print("[Escalation] Transferring to human agent. Please hold...")
        self.add_attribute("last_response", "Please wait while I connect you to a human agent.")
        # In a real system, this would invoke a handoff API

    # ---- Interactive REPL ----
    def start_chat(self):
        """Run a command‑line chat loop."""
        print("\n=== Customer Support Bot (type 'exit' to quit) ===\n")
        self.start_daemons()
        try:
            while True:
                user_input = input("You: ").strip()
                if user_input.lower() in ("exit", "quit"):
                    self.process_input("goodbye")
                    response = self.generate_response()
                    self.send_response(response)
                    break
                self.process_input(user_input)
                # Trigger constraints are evaluated automatically by daemons,
                # but we manually call response generation for simplicity.
                # In a full COH system, daemons would react to attribute changes.
                # Here we generate response based on current state.
                response = self.generate_response()
                self.send_response(response)
        finally:
            self.stop_daemons()


# =============================================================================
# 6. Main Entry Point
# =============================================================================
if __name__ == "__main__":
    # Create the bot
    bot = SupportBot("CustomerSupportBot")

    # Initialize (validates constraints)
    try:
        bot.initialize_system()
        print("Bot system initialized successfully.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit(1)

    # Start interactive chat
    bot.start_chat()