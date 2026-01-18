# Contributing to Open Cure Discovery

First off, thank you for considering contributing to Open Cure Discovery! This project aims to democratize drug discovery, and every contribution helps advance this mission.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Scientific Standards](#scientific-standards)

---

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful, inclusive, and constructive in all interactions.

**Key principles:**
- Be welcoming to newcomers
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community and science

---

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

**When reporting a bug, include:**
- Your operating system and version
- GPU model and CUDA version
- Python version
- Steps to reproduce the issue
- Expected vs actual behavior
- Relevant log output

Use the bug report template when creating an issue.

### Suggesting Features

We welcome feature suggestions! Please:
- Check existing issues/discussions first
- Describe the use case and problem it solves
- Consider the scope (does it fit the project's mission?)
- Be open to discussion about implementation

### Contributing Code

#### Good First Issues

Look for issues labeled `good first issue` - these are suitable for newcomers.

#### Types of Contributions

| Type | Description |
|------|-------------|
| **Core Features** | Docking, ML models, scoring |
| **Data Integration** | New databases, data loaders |
| **UI/UX** | CLI improvements, web dashboard |
| **Documentation** | Guides, tutorials, API docs |
| **Testing** | Unit tests, integration tests |
| **Scientific Validation** | Benchmarks, method validation |
| **Translations** | Internationalization |

### Contributing Scientific Expertise

Not a developer? You can still contribute!
- Review scientific methods
- Suggest disease targets
- Validate results
- Write scientific documentation
- Connect us with research labs

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (for GPU features)
- Git

### Setting Up Your Environment

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/open-cure-discovery.git
cd open-cure-discovery

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode with all extras
pip install -e ".[all]"

# Install pre-commit hooks
pre-commit install

# Verify installation
pytest
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/core/test_docking.py

# Run tests matching a pattern
pytest -k "test_scoring"
```

### Code Quality Tools

```bash
# Run linter
ruff check src/

# Run type checker
mypy src/

# Format code
black src/ tests/
isort src/ tests/

# Run all checks (via pre-commit)
pre-commit run --all-files
```

---

## Pull Request Process

### Before You Start

1. **Create an issue** first for significant changes
2. **Discuss** the approach before investing time
3. **Check** that no one else is working on it

### Creating a Pull Request

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following style guidelines
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run all checks**:
   ```bash
   pytest
   ruff check src/
   mypy src/
   ```
7. **Commit** with clear messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```
8. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
9. **Open a Pull Request** on GitHub

### PR Requirements

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear
- [ ] PR description explains the changes
- [ ] Linked to relevant issue(s)

### Review Process

1. A maintainer will review your PR
2. Address any feedback
3. Once approved, a maintainer will merge

---

## Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications enforced by our tools:

```python
# Good: Descriptive names, type hints, docstrings
def calculate_binding_score(
    molecule: Molecule,
    target: ProteinTarget,
    method: str = "autodock"
) -> float:
    """
    Calculate binding score between a molecule and protein target.

    Args:
        molecule: The ligand molecule to dock.
        target: The protein target structure.
        method: Docking method to use.

    Returns:
        Binding energy in kcal/mol (lower is better).

    Raises:
        DockingError: If docking fails.
    """
    ...
```

### Commit Messages

Format: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance tasks

Examples:
```
feat: add ADMET toxicity predictor
fix: correct VRAM estimation for RTX cards
docs: update installation guide for Windows
test: add unit tests for scoring module
```

### Documentation

- All public functions need docstrings
- Use Google style docstrings
- Include type hints
- Provide usage examples for complex functions

---

## Scientific Standards

This project aims for scientific credibility. Please follow these guidelines:

### Method Documentation

- Cite relevant papers for algorithms
- Document assumptions and limitations
- Explain parameter choices

### Validation

- New methods must include benchmark results
- Compare against established baselines
- Report metrics (AUC, correlation, etc.)
- Run multi-target validation suite: `python examples/multi_target_validation.py`
- Ensure known drugs rank above negative controls

### Data Handling

- Document data sources and licenses
- Maintain data provenance
- Handle missing/invalid data gracefully

### Reproducibility

- Use fixed random seeds where applicable
- Document all dependencies and versions
- Provide scripts to reproduce results

---

## Questions?

- **GitHub Discussions**: For general questions
- **GitHub Issues**: For bugs and features
- **Email**: opencurediscovery@example.com

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Acknowledged in any publications

Thank you for helping accelerate the discovery of new cures!
