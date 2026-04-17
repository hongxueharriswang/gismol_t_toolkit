# GISMOL API Documentation

**Version:** 1.0.0  
**Status:** Complete Implementation of COH Theory

GISMOL (General Intelligent System Modelling Language) is a Python implementation of Constrained Object Hierarchies (COH). This document provides complete API reference for all public classes, methods, and functions.

---

## Table of Contents

1. [Core Module (`gismol.core`)](#1-core-module-gismolcore)
   - [COHObject](#cohobject)
   - [COHRelation](#cohrelation)
   - [Constraint](#constraint)
   - [ConstraintSystem](#constraintsystem)
   - [ConstraintDaemon](#constraintdaemon)
   - [COHRepository](#cohrepository)
   - [Exceptions](#exceptions)

2. [Neural Module (`gismol.neural`)](#2-neural-module-gismolneural)
   - [NeuralComponent](#neuralcomponent)
   - [Classifier](#classifier)
   - [Regressor](#regressor)
   - [Generator](#generator)
   - [DialogueGenerator](#dialoguegenerator)
   - [NeuralSchedulingModel](#neuralSchedulingModel)
   - [NeuralResourceMatcher](#neuralresourcematcher)
   - [NeuralDurationPredictor](#neuraldurationpredictor)
   - [AnomalyDetector](#anomalydetector)
   - [EmbeddingModel](#embeddingmodel)
   - [TextEmbedder](#textembedder)
   - [COHObjectEmbedder](#cohobjectembedder)
   - [MultiModalEmbedder](#multimodalembedder)
   - [CodeEmbedder](#codeembedder)
   - [NeuralLayer](#neurallayer)
   - [LinearLayer](#linearlayer)
   - [ConvLayer](#convlayer)
   - [PoolingLayer](#poolinglayer)
   - [NeuralNetwork](#neuralnetwork)
   - [ConstraintAwareOptimizer](#constraintawareoptimizer)
   - [HierarchicalOptimizer](#hierarchicaloptimizer)
   - [AdaptiveLearningRateOptimizer](#adaptivelearningrateoptimizer)

3. [NLP Module (`gismol.nlp`)](#3-nlp-module-gismolnlp)
   - [COHDialogueManager](#cohdialoguemanager)
   - [GISMOLChat](#gismolchat)
   - [ConstraintAwareResponseGenerator](#constraintawareresponsegenerator)
   - [IntentRecognizer](#intentrecognizer)
   - [EntityRelationExtractor](#entityrelationextractor)
   - [RelationsMiner](#relationsminer)
   - [Text2COH](#text2coh)
   - [ConstraintParser](#constraintparser)
   - [TextNormalizer](#textnormalizer)
   - [TextParser](#textparser)
   - [SimilarityCalculator](#similaritycalculator)
   - [ResponseValidator](#responsevalidator)

4. [Reasoners Module (`gismol.reasoners`)](#4-reasoners-module-gismolreasoners)
   - [Reasoner](#reasoner)
   - [BaseReasoner](#basereasoner)
   - [BiologicalReasoner](#biologicalreasoner)
   - [PhysicalReasoner](#physicalreasoner)
   - [GeometricReasoner](#geometricreasoner)
   - [GeneralReasoner](#generalreasoner)
   - [AttributeReasoner](#attributereasoner)
   - [CardinalityReasoner](#cardinalityreasoner)
   - [RelationalReasoner](#relationalreasoner)
   - [CompositionReasoner](#compositionreasoner)
   - [ComponentReasoner](#componentreasoner)
   - [CausalReasoner](#causalreasoner)
   - [ProbabilisticReasoner](#probabilisticreasoner)
   - [TemporalReasoner](#temporalreasoner)
   - [ResourceReasoner](#resourcereasoner)
   - [TriggerReasoner](#triggerreasoner)
   - [GoalReasoner](#goalreasoner)
   - [SafetyReasoner](#safetyreasoner)

5. [Utility Functions and Constants](#5-utility-functions-and-constants)

---

## 1. Core Module (`gismol.core`)

### COHObject

The fundamental entity in the COH framework representing an intelligent object with constraints, neural components, and hierarchical relationships.

**Class:** `COHObject(name: str = None, parent: Optional[COHObject] = None)`

#### Attributes

| Name | Type | Description |
|------|------|-------------|
| `id` | `str` | Unique identifier (UUID) |
| `name` | `str` | Human‑readable name |
| `parent` | `Optional[COHObject]` | Parent in hierarchy |
| `children` | `List[COHObject]` | Child components |
| `attributes` | `Dict[str, Any]` | State variables |
| `methods` | `Dict[str, callable]` | Executable methods |
| `neural_components` | `Dict[str, NeuralComponent]` | Neural models |
| `embedding_model` | `Optional[EmbeddingModel]` | Semantic embedding |
| `identity_constraints` | `List[Constraint]` | Invariant constraints |
| `trigger_constraints` | `List[Constraint]` | ECA rules |
| `goal_constraints` | `List[Constraint]` | Optimisation objectives |
| `constraint_system` | `ConstraintSystem` | Constraint evaluator |
| `daemons` | `Dict[str, ConstraintDaemon]` | Monitoring daemons |
| `relations` | `COHRelation` | Relationship manager |

#### Methods

##### `add_child(child: COHObject) -> None`
Adds a child component. Raises `HierarchyCycleError` if a cycle would be created.

##### `remove_child(child: COHObject) -> None`
Removes a child component.

##### `add_attribute(key: str, value: Any) -> None`
Adds or updates an attribute.

##### `get_attribute(key: str, default: Any = None) -> Any`
Retrieves an attribute value.

##### `add_method(name: str, func: callable) -> None`
Registers an executable method. The function should accept `self` (the COHObject) as first argument.

##### `execute_method(name: str, *args, **kwargs) -> Any`
Executes a method with constraint checking (trigger constraints are evaluated before; identity constraints after). Raises `ConstraintViolation` on failure.

##### `add_identity_constraint(constraint_spec: Dict) -> None`
Adds an identity constraint. Dictionary keys: `name`, `specification`, `severity` (optional), `category` (optional), `priority` (optional).

##### `add_trigger_constraint(constraint_spec: Dict) -> None`
Adds a trigger constraint. Same dictionary format.

##### `add_goal_constraint(constraint_spec: Dict) -> None`
Adds a goal constraint.

##### `add_neural_component(name: str, component: NeuralComponent, is_embedding_model: bool = False) -> None`
Adds a neural component. If `is_embedding_model` is True, sets `embedding_model` to this component.

##### `get_neural_component(name: str) -> Optional[NeuralComponent]`
Retrieves a neural component by name.

##### `get_context() -> Dict[str, Any]`
Builds a context dictionary for constraint evaluation, containing `object`, `name`, `id`, `parent`, `children`, all attributes, and neural components under `neural_*` keys.

##### `semantic_distance(other: COHObject) -> float`
Computes cosine distance between embeddings. Raises `MissingEmbeddingModel` if no embedding model is set.

##### `start_daemons() -> None`
Starts all registered daemons in background threads.

##### `stop_daemons() -> None`
Stops all daemons.

##### `initialize_system() -> None`
Recursively initialises the object and all children, validating all identity constraints. Raises `ConstraintViolation` on failure.

##### `to_dict() -> Dict`
Serialises the object to a dictionary.

---

### COHRelation

Manages relationships between COHObjects.

**Class:** `COHRelation()`

#### Methods

##### `add_relation(source: COHObject, target: COHObject, relation_type: str, name: str = None) -> None`
Adds a relation. `relation_type` is a category (e.g., "contains", "connected_to").

##### `get_relations(relation_type: str = None) -> List[tuple]`
Returns list of `(source, target, name)` tuples, optionally filtered by type.

##### `visualize(format: str = "mermaid") -> str`
Returns a Mermaid diagram string of the relation graph.

---

### Constraint

Represents a constraint with specification and metadata.

**Class:** `Constraint(name: str, specification: str, category: ConstraintCategory = ConstraintCategory.AUTO, severity: int = 5, priority: str = "MEDIUM", parsed_spec: Dict = field(default_factory=dict))`

#### Class Methods

##### `from_dict(data: Dict, category: str = None) -> Constraint`
Creates a constraint from a dictionary. Auto‑detects category if not provided.

#### Instance Methods

##### `to_dict() -> Dict`
Serialises to dictionary.

---

### ConstraintSystem

Manages multiple constraints and dispatches them to appropriate reasoners.

**Class:** `ConstraintSystem()`

#### Methods

##### `add_constraint(constraint: Constraint) -> None`
Adds a constraint.

##### `register_reasoner(category: str, reasoner: Reasoner) -> None`
Registers a reasoner for a constraint category.

##### `validate_single(constraint: Constraint, context: Dict) -> bool`
Evaluates a single constraint using the registered reasoner.

##### `validate_all(context: Dict) -> Dict[str, bool]`
Evaluates all constraints, returns `{constraint_name: bool}`.

##### `validate_all_raise(context: Dict) -> bool`
Evaluates all constraints, raises `ConstraintViolation` on first failure.

---

### ConstraintDaemon

Base class for background monitoring daemons.

**Class:** `ConstraintDaemon(parent: COHObject, interval: float = 0.1)`

#### Methods

##### `check() -> None` (abstract)
Perform monitoring. Must be overridden.

##### `start() -> None`
Starts the daemon thread.

##### `stop() -> None`
Stops the daemon thread.

**Concrete subclasses:**
- `IdentityConstraintDaemon(parent, interval=0.1)`
- `TriggerConstraintDaemon(parent, interval=0.05)`
- `GoalConstraintDaemon(parent, interval=1.0)`

---

### COHRepository

Manages collections of COHObjects and their relationships.

**Class:** `COHRepository()`

#### Methods

##### `add_object(obj: COHObject) -> None`
Adds an object to the repository.

##### `get_object(obj_id: str) -> Optional[COHObject]`
Retrieves an object by ID.

##### `find_by_name(name: str) -> List[COHObject]`
Finds objects by partial name match.

##### `find_by_attribute(key: str, value: Any) -> List[COHObject]`
Finds objects with a matching attribute value.

##### `add_relation(source: COHObject, target: COHObject, relation_type: str, name: str = None) -> None`
Adds a relation to the repository’s relation manager.

##### `set_focus_object(obj_id_or_name: str) -> None`
Sets the focus object for conversational context.

##### `find_semantic_matches(query: str, threshold: float = 0.7) -> List[COHObject]`
Returns objects whose embedding is similar to the query text.

##### `classify_and_extend(text: str) -> Optional[COHObject]`
Creates a new COHObject from text classification (simplified).

##### `integrate_objects(objects: List[COHObject]) -> None`
Adds a list of objects to the repository.

##### `to_dict() -> Dict`
Serialises the repository.

---

### Exceptions

All exceptions inherit from `COHError`.

| Exception | Description |
|-----------|-------------|
| `ConstraintViolation` | Raised when a constraint is violated. Contains `constraint_name`, `specification`, `context`, `severity`. Method `detailed_report()` returns a string; `attempt_autofix()` attempts automatic resolution (overridable). |
| `ResolutionFailure` | Constraint resolution failed. |
| `MissingEmbeddingModel` | Semantic operation requires an embedding model. |
| `PlanningFailed` | Planning or scheduling failed. |
| `HierarchyCycleError` | A cycle was detected in the component DAG. |
| `InvalidConstraintError` | Constraint specification is invalid. |
| `NeuralComponentError` | Base exception for neural component errors. |

---

## 2. Neural Module (`gismol.neural`)

### NeuralComponent

Base class for all neural components. Inherits from `COHObject` and `ABC`.

**Class:** `NeuralComponent(name: str, input_dim: int = None, output_dim: int = None, **kwargs)`

#### Methods

##### `forward(x: np.ndarray) -> np.ndarray` (abstract)
Forward pass. Must be overridden.

##### `train_component(dataset: List[Tuple[np.ndarray, np.ndarray]], epochs: int = 10, learning_rate: float = 0.01) -> None`
Trains the component on `(input, target)` pairs.

##### `on_constraint_update() -> None`
Callback when constraints change. Triggers retraining.

---

### Classifier

Neural network classifier.

**Class:** `Classifier(name: str, input_dim: int, output_dim: int, **kwargs)`

**Additional method:** `predict(x: np.ndarray) -> int` – returns predicted class index.

---

### Regressor

Neural network regressor.

**Class:** `Regressor(name: str, input_dim: int, output_dim: int, hidden_dim: int = 64, **kwargs)`

---

### Generator

Generative neural component.

**Class:** `Generator(name: str, latent_dim: int, output_dim: int, **kwargs)`

**Additional method:** `generate(num_samples: int = 1) -> List[np.ndarray]` – generates samples from latent space.

---

### DialogueGenerator

Subclass of `Generator` for dialogue.

**Additional method:** `generate_response(context: str) -> str`

---

### NeuralSchedulingModel

End‑to‑end neural scheduler (simulated).

**Class:** `NeuralSchedulingModel(task_dim: int, resource_dim: int, hidden_dim: int = 256, **kwargs)`

**Methods:**
- `generate(task_hierarchy: Dict, resource_pool: List, **kwargs) -> List`
- `constraint_aware_generate(task_hierarchy: Dict, resource_pool: List, constraint_system: ConstraintSystem) -> List`

---

### NeuralResourceMatcher

Matches tasks to resources using embeddings.

**Class:** `NeuralResourceMatcher(capability_list: List[str], embedding_dim: int = 128, **kwargs)`

**Methods:**
- `match(task: Dict, resources: List[COHObject]) -> List[COHObject]`
- `train_matcher(dataset: List, epochs: int = 20) -> None`

---

### NeuralDurationPredictor

Bayesian duration predictor.

**Class:** `NeuralDurationPredictor(input_dim: int, hidden_dim: int = 128, **kwargs)`

**Methods:**
- `predict(task: Dict, context: Dict) -> Tuple[float, float]` – returns (mean, uncertainty).
- `calibrate(historical_data: List) -> None`

---

### AnomalyDetector

Autoencoder‑based anomaly detector.

**Class:** `AnomalyDetector(input_dim: int, hidden_dim: int = 64, **kwargs)`

**Methods:**
- `anomaly_score(x: np.ndarray) -> float`
- `is_anomaly(x: np.ndarray) -> bool`

---

### EmbeddingModel

Base class for embedding models.

**Class:** `EmbeddingModel(name: str, embedding_dim: int = 384, **kwargs)`

**Methods:**
- `embed(input_data: Any) -> np.ndarray` (abstract)
- `embed_text(text: str) -> np.ndarray`
- `embed_object(obj: COHObject) -> np.ndarray`

---

### TextEmbedder

Text embedding using random projection (demo). In real use, would wrap SentenceTransformers.

**Class:** `TextEmbedder(name: str = "text_embedder", **kwargs)`

**Additional method:** `embed_with_constraints(text: str, constraints: Dict) -> np.ndarray`

---

### COHObjectEmbedder

Embedding for COHObjects based on attributes and children.

**Class:** `COHObjectEmbedder(name: str = "object_embedder", **kwargs)`

---

### MultiModalEmbedder

Combines text and object embeddings.

**Class:** `MultiModalEmbedder(name: str = "multi_modal", text_dim: int = 384, object_dim: int = 384, **kwargs)`

---

### CodeEmbedder

Embedding for source code (simplified).

**Class:** `CodeEmbedder(name: str = "code_embedder", **kwargs)`

---

### NeuralLayer

Constraint‑aware neural network layer. Subclass of `COHObject`.

**Class:** `NeuralLayer(name: str, **kwargs)`

**Methods:**
- `forward(x: np.ndarray) -> np.ndarray`
- `get_parameter_count() -> int`
- `get_memory_footprint() -> int`

**Concrete subclasses:**
- `LinearLayer(name, in_features, out_features, activation='relu')`
- `ConvLayer(name, in_channels, out_channels, kernel_size)`
- `PoolingLayer(name, pool_size, mode='max')`

---

### NeuralNetwork

Composable neural network as a COHObject.

**Class:** `NeuralNetwork(name: str, **kwargs)`

**Methods:**
- `add_layer(layer: NeuralLayer) -> None`
- `build(input_shape: Tuple[int, ...]) -> None`
- `forward(x: np.ndarray) -> np.ndarray`
- `validate_network() -> bool`

---

### ConstraintAwareOptimizer

Optimizer that adds penalty terms for constraints.

**Class:** `ConstraintAwareOptimizer(name: str, params: List, constraints: List[Callable], penalty_weight: float = 0.1, lr: float = 0.01)`

**Methods:**
- `zero_grad() -> None`
- `step(closure: Optional[Callable] = None) -> float`

---

### HierarchicalOptimizer

Optimizer that respects COH hierarchy.

**Class:** `HierarchicalOptimizer(name: str, params: List, root_object: COHObject, lr: float = 0.01)`

---

### AdaptiveLearningRateOptimizer

Optimizer with adaptive learning rate based on constraint satisfaction.

**Class:** `AdaptiveLearningRateOptimizer(name: str, params: List, constraints: List[Callable], base_lr: float = 0.01)`

---

## 3. NLP Module (`gismol.nlp`)

### COHDialogueManager

Manages dialogue with constraint‑aware response generation.

**Class:** `COHDialogueManager(repository: COHRepository = None)`

**Methods:**
- `respond(user_input: str) -> str` – generates response.
- `update_context(context: Dict) -> None`

---

### GISMOLChat

Advanced chat interface with hierarchical context.

**Class:** `GISMOLChat(repository: COHRepository = None)`

**Additional method:** `set_focus_object(obj_name: str) -> None`

---

### ConstraintAwareResponseGenerator

Generates responses that satisfy constraints.

**Class:** `ConstraintAwareResponseGenerator()`

**Method:** `generate_response(intent: str, context: Dict, objects: List[COHObject]) -> str`

---

### IntentRecognizer

Recognises intent from natural language using patterns.

**Class:** `IntentRecognizer()`

**Method:** `recognize_intent(query: str) -> Dict[str, Any]` – returns `{'intent': str, 'parameters': tuple}`.

---

### EntityRelationExtractor

Extracts entities and relations, creates COHObjects.

**Class:** `EntityRelationExtractor()`

**Method:** `extract_to_coh(text: str, repository: COHRepository) -> List[COHObject]`

---

### RelationsMiner

Mines relations from text.

**Class:** `RelationsMiner()`

**Method:** `mine(text: str, objects: List[COHObject]) -> List[Tuple[COHObject, COHObject, str]]`

---

### Text2COH

Converts text documents to COHObjects.

**Class:** `Text2COH(repository: COHRepository)`

**Method:** `process_knowledge_source(document: str) -> Dict[str, Any]` – returns `{'new_objects': int, 'integrated': int}`.

---

### ConstraintParser

Parses natural language constraints.

**Class:** `ConstraintParser()`

**Methods:**
- `parse(text: str) -> Dict` – returns structured representation.
- `to_coh_constraints(parsed: Dict) -> List[Constraint]`

---

### TextNormalizer

Basic text normalisation.

**Class:** `TextNormalizer()`

**Method:** `normalize(text: str) -> str`

---

### TextParser

Basic text parsing.

**Class:** `TextParser()`

**Method:** `parse(text: str) -> dict` – returns `{'words': List[str], 'length': int, 'sentences': List[str]}`.

---

### SimilarityCalculator

Computes semantic similarity.

**Class:** `SimilarityCalculator(threshold: float = 0.7, embedder: Optional[TextEmbedder] = None)`

**Methods:**
- `text_similarity(text1: str, text2: str) -> float`
- `object_similarity(obj1: COHObject, obj2: COHObject, embedder=None) -> float`
- `is_similar(similarity: float) -> bool`

---

### ResponseValidator

Validates responses against COH constraints.

**Class:** `ResponseValidator()`

**Method:** `validate(response: str, objects: List[COHObject]) -> List[Dict[str, Any]]` – returns list of validation results.

---

## 4. Reasoners Module (`gismol.reasoners`)

### Reasoner

Abstract base class with registry.

**Class Methods:**
- `get_reasoner(reasoner_type: str) -> Type[Reasoner]`

**Instance Methods:**
- `evaluate(constraint: Constraint, context: Dict) -> bool` (abstract)
- `attempt_resolution(constraint: Constraint, context: Dict) -> bool`

---

### BaseReasoner

Fallback reasoner for numeric and boolean comparisons. Supports dot‑notation.

**Class:** `BaseReasoner(reasoner_type="base")`

---

### Domain‑Specific Reasoners

| Class | Type String | Description |
|-------|-------------|-------------|
| `BiologicalReasoner` | `"biological"` | Handles biological constraints (concentration, CFU/ml). |
| `PhysicalReasoner` | `"physical"` | Physics laws with tolerance (e.g., `F=ma with tolerance 0.05`). |
| `GeometricReasoner` | `"geometric"` | Distances, positions, containment. |

---

### General Reasoners

| Class | Type String | Description |
|-------|-------------|-------------|
| `GeneralReasoner` | `"general"` | Logical operators (`and`, `or`, `implies`, `iff`), custom functions. |
| `AttributeReasoner` | `"attribute"` | Direct attribute comparisons. |
| `CardinalityReasoner` | `"cardinality"` | `count(collection) < N` constraints. |
| `RelationalReasoner` | `"relational"` | `transitive()`, `symmetric()`, simple `A related_to B`. |
| `CompositionReasoner` | `"composition"` | Part‑whole relationships (`X has Y`). |
| `ComponentReasoner` | `"component"` | Inter‑component connections. |

---

### Advanced Reasoners

| Class | Type String | Description |
|-------|-------------|-------------|
| `CausalReasoner` | `"causal"` | Cause‑effect with probabilities (`A causes B with p=0.8`). |
| `ProbabilisticReasoner` | `"probabilistic"` | `P(event) < threshold`. |
| `TemporalReasoner` | `"temporal"` | Time‑based: `within Xs after Y`, `before`. |
| `ResourceReasoner` | `"resource"` | CPU, memory, bandwidth constraints. |
| `TriggerReasoner` | `"trigger"` | ECA rules (`WHEN event DO action ENSURE postcondition`). |
| `GoalReasoner` | `"goal"` | `MAXIMIZE ... SUBJECT TO ...`. |
| `SafetyReasoner` | `"safety"` | Redundant validation for critical systems. |

---

## 5. Utility Functions and Constants

### ConstraintCategory Enum

Values: `ATTRIBUTE`, `GEOMETRIC`, `TEMPORAL`, `RESOURCE`, `CARDINALITY`, `COMPOSITION`, `PHYSICAL`, `CAUSAL`, `RELATIONAL`, `PROBABILISTIC`, `IDENTITY`, `TRIGGER`, `GOAL`, `SAFETY`, `AUTO`.

---

### Package Constants

- `__version__ = "1.0.0"`
- `__author__ = "Harris Wang"`

---

## Example Usage Snippets

### Creating and running a thermostat

```python
from gismol import COHObject

thermostat = COHObject("Thermostat")
thermostat.add_attribute("current_temp", 20.0)
thermostat.add_identity_constraint({
    'name': 'freeze',
    'specification': 'current_temp >= 5',
    'severity': 10
})
thermostat.initialize_system()
```

### Adding a neural predictor

```python
from gismol.neural import Regressor
predictor = Regressor("predictor", input_dim=4, output_dim=1)
thermostat.add_neural_component("predictor", predictor)
```

### Adding a trigger

```python
thermostat.add_trigger_constraint({
    'name': 'heat_on',
    'specification': 'WHEN current_temp < target_temp - 0.5 DO set_heater(1)'
})
```

### Using a reasoner

```python
from gismol.reasoners import TemporalReasoner
thermostat.constraint_system.register_reasoner("temporal", TemporalReasoner())
```

### NLP dialogue

```python
from gismol.nlp import COHDialogueManager
dialogue = COHDialogueManager(repo)
response = dialogue.respond("What is the temperature?")
```

---

This concludes the GISMOL API documentation. For more detailed examples, refer to the five complete system implementations (thermostat, dialogue bot, vacuum cleaner, trading agent, adaptive cruise control).