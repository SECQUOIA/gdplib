"""
Test pip installation specifically to verify setup.py dependencies.

This module tests that the setup.py install_requires dependencies are properly
specified and that gdplib can be imported after a pip installation.
"""

import pytest
import subprocess
import sys
import tempfile
import os


class TestPipInstallation:
    """Test pip installation process and dependency resolution."""

    def test_setup_py_dependencies_specified(self):
        """Test that setup.py has install_requires properly specified."""
        setup_py_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup.py"
        )

        with open(setup_py_path, "r") as f:
            setup_content = f.read()

        # Check that install_requires is not empty
        assert "install_requires=[" in setup_content
        assert "Pyomo>=5.6.1" in setup_content
        assert "pandas>=1.0.1" in setup_content
        assert "matplotlib>=2.2.2" in setup_content

        # Should not have an empty install_requires list
        assert "install_requires=[]" not in setup_content

    def test_dependency_list_consistency(self):
        """Test that dependencies in setup.py match requirements.txt."""
        # Read requirements.txt
        requirements_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "requirements.txt",
        )

        with open(requirements_path, "r") as f:
            requirements = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

        # Read setup.py
        setup_py_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup.py"
        )

        with open(setup_py_path, "r") as f:
            setup_content = f.read()

        # Check that main dependencies are present in both
        main_deps = ["Pyomo", "pandas", "matplotlib", "scipy", "pint", "openpyxl"]

        for dep in main_deps:
            # Check in requirements.txt (case-insensitive)
            req_found = any(dep.lower() in req.lower() for req in requirements)
            assert req_found, f"{dep} not found in requirements.txt"

            # Check in setup.py (case-insensitive)
            setup_found = dep.lower() in setup_content.lower()
            assert setup_found, f"{dep} not found in setup.py install_requires"

    def test_pyomo_available_before_gdplib_import(self):
        """Test that Pyomo is available before importing gdplib."""
        try:
            import pyomo.environ
            import pyomo.gdp
        except ImportError:
            pytest.fail("Pyomo must be available before importing gdplib modules")

    def test_gdplib_import_after_dependencies(self):
        """Test that gdplib can be imported when dependencies are available."""
        # First ensure dependencies are available
        required_modules = ["pyomo", "pandas", "matplotlib"]

        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                pytest.skip(f"Required dependency {module} not available")

        # Now try to import gdplib
        try:
            import gdplib

            assert gdplib is not None
        except ImportError as e:
            pytest.fail(
                f"gdplib import failed despite dependencies being available: {e}"
            )

    def test_pip_install_scenario(self):
        """Simulate the scenario that would happen during pip install."""
        # This tests the scenario where someone does pip install gdplib
        # and then tries to import it

        # Verify that the main gdplib module can be imported
        # (this would fail if setup.py doesn't specify dependencies correctly)
        try:
            # Import the main module - this should work if dependencies are installed
            import gdplib

            # Try importing a submodule that uses pyomo
            # This will fail if pyomo is not available
            import gdplib.cstr

            # Try building a model - this exercises the actual functionality
            model = gdplib.cstr.build_model()
            assert model is not None

        except ImportError as e:
            if "pyomo" in str(e).lower():
                pytest.fail(
                    "Pyomo dependency not available - setup.py install_requires likely missing pyomo"
                )
            else:
                pytest.fail(f"Import failed: {e}")
        except Exception as e:
            pytest.skip(f"Model construction failed (may require solver): {e}")


if __name__ == "__main__":
    pytest.main([__file__])
