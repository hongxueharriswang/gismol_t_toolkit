
## `CONTRIBUTING.md`

```markdown
# Contributing to GISMOL

Thank you for your interest in contributing to GISMOL! We welcome all forms of contributions: bug reports, documentation improvements, feature requests, and code changes.

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## How to Contribute

### 1. Report Bugs or Request Features

Use the [GitHub Issues](https://github.com/your-username/gismol/issues) page. Please check existing issues before creating a new one.

### 2. Improve Documentation

Documentation is essential. You can help by:
- Fixing typos or unclear explanations in docstrings or README.
- Adding examples or tutorials.
- Translating documentation.

### 3. Submit Code Changes

#### Setup Development Environment

```bash
git clone https://github.com/your-username/gismol.git
cd gismol
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[dev]"
pre-commit install
```

#### Run Tests

```bash
pytest tests/
```

#### Coding Style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use [Black](https://github.com/psf/black) for code formatting.
- Use [isort](https://pycqa.github.io/isort/) for import sorting.
- Add docstrings for all public classes and methods (Google style).

#### Commit Message Guidelines

Use conventional commits:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `style:` formatting (no code change)
- `refactor:` code change that neither fixes a bug nor adds a feature
- `test:` adding or correcting tests
- `chore:` changes to build process or tooling

Example: `feat(core): add method to export COHObject to JSON`

#### Pull Request Process

1. Fork the repository and create a branch from `main`.
2. Make your changes, add tests if applicable.
3. Run `pytest` and `black .` to ensure everything passes.
4. Push your branch and open a Pull Request.
5. Request a review from a maintainer.

## Development Guidelines

### Adding a New Reasoner

1. Create a subclass of `Reasoner` in `gismol/reasoners/advanced.py` or a new file.
2. Register it with a unique `reasoner_type`.
3. Add tests in `tests/test_reasoners.py`.

### Adding a New Neural Component

1. Subclass `NeuralComponent` in `gismol/neural/components.py`.
2. Implement `forward()` and optionally `train_component()`.
3. Add an example usage to `examples/`.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
```