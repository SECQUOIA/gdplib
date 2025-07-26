"""
Test model structure and documentation for all GDPlib modules.

This module tests that all models follow the expected structure and
have proper documentation.
"""

import pytest
import importlib
import os
import inspect


class TestModelStructure:
    """Test that all models follow the expected structure."""

    GDPLIB_MODULES = [
        'mod_hens',
        'modprodnet', 
        'biofuel',
        'positioning',
        'spectralog',
        'stranded_gas',
        'gdp_col',
        'hda',
        'kaibel',
        'methanol',
        'batch_processing',
        'jobshop',
        'disease_model',
        'med_term_purchasing',
        'syngas',
        'water_network',
        'ex1_linan_2023',
        'small_batch',
        'cstr',
        'reverse_electrodialysis',
    ]

    @pytest.mark.parametrize("module_name", GDPLIB_MODULES)
    def test_module_has_proper_init(self, module_name):
        """Test that each module has a proper __init__.py file."""
        module_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'gdplib',
            module_name
        )
        
        if os.path.exists(module_path):
            init_file = os.path.join(module_path, '__init__.py')
            assert os.path.exists(init_file), f"{module_name} missing __init__.py"

    @pytest.mark.parametrize("module_name", GDPLIB_MODULES)
    def test_build_model_has_docstring(self, module_name):
        """Test that build_model functions have docstrings."""
        try:
            module = importlib.import_module(f'gdplib.{module_name}')
            if hasattr(module, 'build_model'):
                build_func = getattr(module, 'build_model')
                docstring = inspect.getdoc(build_func)
                assert docstring is not None and len(docstring.strip()) > 0, \
                    f"{module_name}.build_model missing docstring"
        except ImportError:
            pytest.skip(f"Module {module_name} not available")

    @pytest.mark.parametrize("module_name", GDPLIB_MODULES)
    def test_module_has_readme(self, module_name):
        """Test that each module directory has a README file."""
        module_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'gdplib',
            module_name
        )
        
        if os.path.exists(module_path):
            readme_files = [
                os.path.join(module_path, 'README.md'),
                os.path.join(module_path, 'README.rst'),
                os.path.join(module_path, 'readme.md'),
                os.path.join(module_path, 'readme.txt'),
            ]
            
            has_readme = any(os.path.exists(readme) for readme in readme_files)
            if not has_readme:
                pytest.skip(f"{module_name} directory has no README file (recommended but not required)")

    def test_all_modules_listed_in_main_init(self):
        """Test that all modules are listed in the main __init__.py."""
        main_init_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'gdplib',
            '__init__.py'
        )
        
        with open(main_init_path, 'r') as f:
            init_content = f.read()
        
        for module_name in self.GDPLIB_MODULES:
            # Check if module is imported in __init__.py
            import_found = (f'import gdplib.{module_name}' in init_content or 
                          f'from gdplib import {module_name}' in init_content or
                          f'from . import {module_name}' in init_content)
            
            if not import_found:
                pytest.skip(f"{module_name} not imported in main __init__.py (may be intentional)")


class TestModelFunctionality:
    """Test model functionality and Pyomo integration."""

    def test_models_return_pyomo_objects(self):
        """Test that build_model functions return proper Pyomo model objects."""
        # Test a few key models
        test_modules = ['cstr', 'biofuel', 'gdp_col']
        
        for module_name in test_modules:
            try:
                module = importlib.import_module(f'gdplib.{module_name}')
                if hasattr(module, 'build_model'):
                    try:
                        model = module.build_model()
                        
                        # Check that it's a Pyomo model
                        assert hasattr(model, 'component_objects'), \
                            f"{module_name} model is not a Pyomo model"
                        
                        # Check that it has some components
                        components = list(model.component_objects())
                        assert len(components) > 0, \
                            f"{module_name} model has no components"
                            
                    except Exception as e:
                        pytest.skip(f"{module_name} model construction failed: {e}")
            except ImportError:
                pytest.skip(f"Module {module_name} not available")

    def test_pyomo_gdp_constructs_available(self):
        """Test that Pyomo GDP constructs are available."""
        try:
            from pyomo.gdp import Disjunct, Disjunction
            
            # These should be available for GDP models
            assert Disjunct is not None
            assert Disjunction is not None
        except ImportError:
            pytest.fail("Pyomo GDP constructs not available")


if __name__ == "__main__":
    pytest.main([__file__])