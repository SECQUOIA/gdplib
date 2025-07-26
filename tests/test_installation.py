"""
Test installation and pip functionality for GDPlib.

This module specifically tests the pip installation process and verifies
that all dependencies are correctly specified and installed.
"""

import pytest
import subprocess
import sys
import pkg_resources


class TestInstallation:
    """Test pip installation functionality."""

    def test_pip_installation_dependencies(self):
        """Test that all required dependencies from requirements.txt are installed."""
        required_packages = [
            "pyomo",
            "pandas",
            "matplotlib",
            "scipy",
            "pint",
            "openpyxl",
        ]

        missing_packages = []
        for package in required_packages:
            try:
                pkg_resources.get_distribution(package)
            except pkg_resources.DistributionNotFound:
                missing_packages.append(package)

        if missing_packages:
            pytest.fail(f"Missing required packages: {missing_packages}")

    def test_gdplib_installation(self):
        """Test that gdplib itself is properly installed."""
        try:
            import gdplib

            # Verify it's installed as a package
            pkg_resources.get_distribution("gdplib")
        except pkg_resources.DistributionNotFound:
            pytest.skip("gdplib not installed as package (development mode)")
        except ImportError:
            pytest.fail("gdplib package cannot be imported")

    def test_requirements_txt_validity(self):
        """Test that requirements.txt contains valid package specifications."""
        import os

        requirements_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "requirements.txt",
        )

        if not os.path.exists(requirements_path):
            pytest.skip("requirements.txt not found")

        with open(requirements_path, "r") as f:
            requirements = f.read().strip().split("\n")

        # Filter out empty lines and comments
        requirements = [
            req.strip()
            for req in requirements
            if req.strip() and not req.strip().startswith("#")
        ]

        assert len(requirements) > 0, "requirements.txt is empty"

        # Check that each requirement has a valid format
        for req in requirements:
            assert (
                ">=" in req or "==" in req or ">" in req or "<" in req
            ), f"Invalid requirement format: {req}"


if __name__ == "__main__":
    pytest.main([__file__])
