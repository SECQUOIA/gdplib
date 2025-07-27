# GitHub Copilot Custom Instructions for GDPlib

## Project Context
GDPlib is an open-source Python library for Generalized Disjunctive Programming (GDP) models. The library provides a collection of GDP model implementations for benchmarking and educational purposes.

## Key Technologies & Dependencies
- **Primary Framework**: Pyomo (optimization modeling in Python)
- **Core Dependencies**: pandas, matplotlib, scipy, pint, openpyxl
- **Python Version**: 3.6+
- **License**: BSD 3-clause

## Coding Standards & Best Practices
- Follow PEP 8 style guidelines (enforced by Black formatter)
- Use type hints where appropriate
- Maintain backward compatibility with existing model interfaces
- All models should expose a `build_model()` function in their `__init__.py`

## Model Development Guidelines
- Each model should be in its own subdirectory under `gdplib/`
- Include a `README.md` file explaining the model
- Use relative imports within modules, absolute imports for `__main__` scripts
- Models should be self-contained with minimal external dependencies

## Testing Approach
- Focus on model construction and basic validation
- Test that models can be built without errors
- Verify model structure and constraints
- Use pytest framework for test organization

## Documentation Style
- Clear, concise docstrings following NumPy/SciPy style
- Include mathematical formulations where relevant
- Provide usage examples in README files
- Reference original papers or sources when applicable

## Import Patterns
```python
# For library modules
from .module import function

# For main scripts
from gdplib.module import function
```

## Common Pyomo Patterns
```python
from pyomo.environ import *
from pyomo.gdp import *

def build_model():
    model = ConcreteModel()
    # Model construction
    return model
```

## Helpful Context
- GDP (Generalized Disjunctive Programming) involves logical relationships and disjunctions
- Models often represent optimization problems in chemical engineering, operations research
- Performance and scalability matter for benchmarking purposes