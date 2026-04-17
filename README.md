

# GISMOL – General Intelligent System Modelling Language

**GISMOL** is a Python implementation of **Constrained Object Hierarchies (COH)** – a neuroscience‑grounded, mathematically rigorous framework for building intelligent systems. GISMOL unifies hierarchical composition, adaptive learning, and constraint‑driven behaviour into a single, coherent toolkit. This implementation is based on a tree‑structured architecture supporting multi‑level modularity.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-ready-brightgreen)](docs/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🧠 What is COH?

Constrained Object Hierarchies (COH) models intelligence as a **9‑tuple**:

```
O = (C, A, M, N, E, I, T, G, D)
```

| Symbol | Name                 | Role                                                                 |
|--------|----------------------|----------------------------------------------------------------------|
| **C**  | Components           | Hierarchical parts (sub‑objects)                                     |
| **A**  | Attributes           | State variables                                                      |
| **M**  | Methods              | Executable actions                                                   |
| **N**  | Neural components    | Learnable, adaptive models                                           |
| **E**  | Embedding            | Unified semantic representation                                      |
| **I**  | Identity constraints | Invariants that must always hold                                     |
| **T**  | Trigger constraints  | Event–Condition–Action rules                                         |
| **G**  | Goal constraints     | Optimisation objectives                                              |
| **D**  | Daemons              | Continuous monitoring processes (background threads)                 |

GISMOL is designed to provide ready‑to‑use classes and tools to implement every component of the COH 9‑tuple.

---

## 🚀 Features

- **Complete COH implementation** – All nine components are first‑class citizens.
- **Constraint‑aware reasoning** – Built‑in reasoners for logical, temporal, geometric, probabilistic, and ethical constraints.
- **Neural integration** – Neural components inherit `COHObject` and respect constraints during training.
- **Natural language processing** – Dialogue management, intent recognition, and constraint parsing.
- **Real‑time daemons** – Background monitors that enforce invariants and triggers.
- **Hierarchical object system** – DAG‑based component composition with cycle detection.
- **Semantic embeddings** – Embedding models for objects and text.
- **Explainability** – Every decision can be explained via the `ExplainabilityModule`.

---

## 📦 Installation

### From source (recommended)

```bash
git clone https://github.com/your-username/gismol.git
cd gismol
pip install -e .
```

### Dependencies

- Python 3.8+
- numpy
- (optional) torch, transformers – for advanced neural components

---

## 🏃 Quick Start

```python
from gismol import COHObject

# Create an intelligent thermostat
thermostat = COHObject("SmartThermostat")
thermostat.add_attribute("current_temp", 20.0)
thermostat.add_attribute("target_temp", 21.0)

# Add an identity constraint (invariant)
thermostat.add_identity_constraint({
    'name': 'freeze_protection',
    'specification': 'current_temp >= 5',
    'severity': 10
})

# Add a trigger constraint (ECA rule)
thermostat.add_trigger_constraint({
    'name': 'heat_on',
    'specification': 'WHEN current_temp < target_temp - 0.5 DO set_heater(1)'
})

# Add a method
def set_heater(self, state):
    self.add_attribute("heater_on", state)

thermostat.add_method("set_heater", set_heater)

# Initialise and run
thermostat.initialize_system()
thermostat.start_daemons()
```

---

## 📚 Complete Examples

GISMOL includes full‑fledged implementations across five domains:

| Domain                    | Example                                        |
|---------------------------|------------------------------------------------|
| 🏭 Business               | AI‑Driven Supply Chain Optimizer              |
| 🎓 Education              | Personalised Adaptive Learning Platform       |
| ⚙️ Manufacturing          | Predictive Maintenance for CNC Machining      |
| 🚀 Space Exploration      | Autonomous Rover for Mars Exploration         |
| ⚖️ Social Science         | Ethical AI for Social Welfare Allocation      |

Each example demonstrates the full COH 9‑tuple in a realistic scenario.  
Run them directly:

```bash
python examples/supply_chain.py
python examples/adaptive_learning.py
python examples/predictive_maintenance.py
python examples/mars_rover.py
python examples/welfare_allocation.py
```

---

## 🧩 Module Structure

```
gismol/
├── core/               # COHObject, constraints, daemons, repository
├── neural/             # Neural components, embeddings, optimizers
├── nlp/                # Dialogue, intent recognition, constraint parsing
├── reasoners/          # Logical, temporal, geometric, ethical reasoners
└── examples/           # Complete domain applications
```

---

## 📖 Documentation

- **API Reference** – See [`docs/API.md`](docs/API.md)
- **Tutorial** – [A Developer’s Guide to COH and GISMOL](docs/TUTORIAL.md)
- **COH Theory Paper** – [The 9‑Tuple Formation of Constrained Object Hierarchies](https://arxiv.org/abs/...)

---

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

### Development setup

```bash
git clone https://github.com/your-username/gismol.git
cd gismol
pip install -e ".[dev]"
pre-commit install
```

Run tests:

```bash
pytest tests/
```

---

## 📄 License

GISMOL is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📧 Contact & Acknowledgements

**Author:** Harris Wang  
**Email:** harrisw@athabascau.ca  
**Paper:** Wang, H. (2025). Constrained Object Hierarchies as a unified theoretical model for intelligence and intelligent systems. *Computers*, 14, 478.

This work was supported by Athabasca University, School of Computing and Information Systems.

---

## ⭐ Citation

If you use GISMOL in your research, please cite:

```bibtex
@article{wang2025constrained,
  title={Constrained Object Hierarchies as a unified theoretical model for intelligence and intelligent systems},
  author={Wang, Harris},
  journal={Computers},
  volume={14},
  pages={478},
  year={2025}
}
```
```

---

## `LICENSE`

```text
MIT License

Copyright (c) 2025 Harris Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


