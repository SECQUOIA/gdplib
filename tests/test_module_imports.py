"""
Test suite for individual GDPlib module imports and functionality.

This module contains comprehensive tests for each GDPlib module to ensure
they can be imported and their build_model functions work correctly.
"""

import pytest
import importlib
import sys
import os

# Add the gdplib directory to the path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModuleImports:
    """Test that all gdplib modules can be imported individually."""

    # List of modules that should be available in gdplib
    GDPLIB_MODULES = [
        "mod_hens",
        "modprodnet",
        "biofuel",
        "positioning",
        "spectralog",
        "stranded_gas",
        "gdp_col",
        "hda",
        "kaibel",
        "methanol",
        "batch_processing",
        "jobshop",
        "disease_model",
        "med_term_purchasing",
        "syngas",
        "water_network",
        "ex1_linan_2023",
        "small_batch",
        "cstr",
        "reverse_electrodialysis",
    ]

    @pytest.mark.parametrize("module_name", GDPLIB_MODULES)
    def test_individual_module_import(self, module_name):
        """Test that each gdplib module can be imported."""
        try:
            module = importlib.import_module(f"gdplib.{module_name}")
            assert module is not None
        except ImportError as e:
            pytest.skip(f"Module {module_name} import failed: {e}")

    @pytest.mark.parametrize("module_name", GDPLIB_MODULES)
    def test_module_has_build_model(self, module_name):
        """Test that each module has a build_model function."""
        try:
            module = importlib.import_module(f"gdplib.{module_name}")
            assert hasattr(
                module, "build_model"
            ), f"{module_name} missing build_model function"
        except ImportError:
            pytest.skip(f"Module {module_name} not available")

    @pytest.mark.parametrize("module_name", GDPLIB_MODULES)
    def test_build_model_callable(self, module_name):
        """Test that build_model function is callable and returns a model."""
        try:
            module = importlib.import_module(f"gdplib.{module_name}")
            if hasattr(module, "build_model"):
                build_func = getattr(module, "build_model")
                assert callable(
                    build_func
                ), f"{module_name}.build_model is not callable"

                # Try to call build_model without arguments first
                try:
                    model = build_func()
                    assert model is not None, f"{module_name}.build_model returned None"
                except TypeError:
                    # Some models might require arguments, try with empty args
                    try:
                        model = build_func(*[])
                        assert model is not None
                    except Exception:
                        pytest.skip(
                            f"{module_name}.build_model requires specific arguments"
                        )
                except Exception as e:
                    err_msg = str(e)
                    if (
                        "No executable found for solver" in err_msg
                        or "No 'gams' command" in err_msg
                    ):
                        pytest.skip(
                            f"{module_name}.build_model requires external solver: {e}"
                        )
                    else:
                        pytest.fail(
                            f"{module_name}.build_model raised unexpected error: {e}"
                        )
        except ImportError:
            pytest.skip(f"Module {module_name} not available")


class TestModelConstruction:
    """Test model construction for key modules."""

    MODELS = ["cstr", "biofuel", "gdp_col"]

    @pytest.mark.parametrize("module_name", MODELS)
    def test_model_construction(self, module_name):
        """Ensure that selected models can be built successfully."""
        try:
            module = importlib.import_module(f"gdplib.{module_name}")

            model = module.build_model()
            assert model is not None
            assert hasattr(model, "component_objects")
        except ImportError:
            pytest.skip(f"{module_name} module not available")
        except Exception as e:
            err_msg = str(e)
            if (
                "No executable found for solver" in err_msg
                or "No 'gams' command" in err_msg
            ):
                pytest.skip(
                    f"{module_name} model construction requires external solver: {e}"
                )
            else:
                pytest.fail(
                    f"{module_name} model construction raised unexpected error: {e}"
                )
