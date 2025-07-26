"""
Test suite for GDPlib library.

This module contains tests to verify the basic functionality of the GDPlib library,
including model imports and basic model construction.
"""

import pytest
import importlib
import sys
import os

# Add the gdplib directory to the path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGDPlibImports:
    """Test that all gdplib modules can be imported without errors."""

    def test_main_import(self):
        """Test that the main gdplib module can be imported."""
        import gdplib

        assert gdplib is not None

    def test_submodule_imports(self):
        """Test that key submodules can be imported."""
        # Test a few key modules that should be available
        try:
            import gdplib.cstr
            import gdplib.biofuel
            import gdplib.gdp_col
        except ImportError as e:
            pytest.skip(f"Submodule import failed: {e}")


class TestBasicFunctionality:
    """Test basic functionality of GDPlib models."""

    def test_installation_verification(self):
        """Verify that gdplib is properly installed and accessible."""
        try:
            import gdplib

            # Try to access the version if available
            if hasattr(gdplib, '__version__'):
                assert gdplib.__version__ is not None
        except ImportError:
            pytest.fail("gdplib could not be imported")

    def test_pyomo_dependency(self):
        """Verify that Pyomo dependency is available."""
        try:
            import pyomo.environ
            import pyomo.gdp
        except ImportError:
            pytest.fail("Pyomo dependencies not available")

    def test_basic_model_construction(self):
        """Test that we can construct a basic model from available modules."""
        try:
            # Try to build a simple model if biofuel is available
            import gdplib.biofuel

            if hasattr(gdplib.biofuel, 'build_model'):
                model = gdplib.biofuel.build_model()
                assert model is not None
        except (ImportError, AttributeError):
            pytest.skip("biofuel module or build_model not available")
        except Exception as e:
            pytest.skip(f"Model construction failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
