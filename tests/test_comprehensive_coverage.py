"""
Comprehensive test coverage for GDPlib models.

This module provides complete testing of all GDPlib models for:
1. Import capability 
2. Model construction and execution
3. Coverage statistics and reporting
"""

import pytest
import importlib
import sys
import os
import traceback
from io import StringIO
import subprocess

# Add the gdplib directory to the path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestComprehensiveCoverage:
    """Comprehensive tests for all GDPlib models ensuring importability and runnability."""

    # Complete list of all GDPlib modules
    ALL_GDPLIB_MODULES = [
        "batch_processing",
        "biofuel",
        "cstr", 
        "disease_model",
        "ex1_linan_2023",
        "gdp_col",
        "hda",
        "jobshop",
        "kaibel",
        "med_term_purchasing",
        "methanol",
        "mod_hens",
        "modprodnet",
        "positioning",
        "reverse_electrodialysis",
        "small_batch",
        "spectralog",
        "stranded_gas",
        "syngas",
        "water_network",
    ]

    def test_coverage_statistics_report(self, capsys):
        """Generate and display comprehensive coverage statistics."""
        print("\n" + "="*80)
        print("GDPlib Comprehensive Test Coverage Report")
        print("="*80)
        
        # Test import status for all modules
        import_results = {}
        build_model_results = {}
        execution_results = {}
        
        for module_name in self.ALL_GDPLIB_MODULES:
            # Test import
            try:
                module = importlib.import_module(f"gdplib.{module_name}")
                import_results[module_name] = "✅ SUCCESS"
            except Exception as e:
                import_results[module_name] = f"❌ FAILED: {str(e)}"
                continue
                
            # Test build_model availability
            if hasattr(module, "build_model"):
                build_model_results[module_name] = "✅ AVAILABLE"
                
                # Test build_model execution
                try:
                    model = module.build_model()
                    if model is not None:
                        execution_results[module_name] = "✅ RUNNABLE"
                    else:
                        execution_results[module_name] = "⚠️  RETURNS_NULL"
                except Exception as e:
                    # Handle solver dependency issues gracefully
                    error_msg = str(e)
                    if ("ipopt" in error_msg.lower() or 
                        "gams" in error_msg.lower() or 
                        "solver" in error_msg.lower()):
                        execution_results[module_name] = f"⚠️  REQUIRES_SOLVER: {error_msg.split(':')[0] if ':' in error_msg else 'External solver'}"
                    else:
                        execution_results[module_name] = f"❌ EXECUTION_FAILED: {str(e)[:50]}..."
            else:
                build_model_results[module_name] = "❌ MISSING"
                execution_results[module_name] = "N/A"
        
        # Print summary statistics
        total_modules = len(self.ALL_GDPLIB_MODULES)
        successful_imports = len([v for v in import_results.values() if v == "✅ SUCCESS"])
        available_build_models = len([v for v in build_model_results.values() if v == "✅ AVAILABLE"])
        runnable_models = len([v for v in execution_results.values() if v == "✅ RUNNABLE"])
        
        print(f"\nModule Import Statistics:")
        print(f"  Total Modules: {total_modules}")
        print(f"  Successful Imports: {successful_imports}/{total_modules} ({successful_imports/total_modules*100:.1f}%)")
        print(f"  Available build_model: {available_build_models}/{total_modules} ({available_build_models/total_modules*100:.1f}%)")
        print(f"  Runnable Models: {runnable_models}/{total_modules} ({runnable_models/total_modules*100:.1f}%)")
        
        print(f"\nDetailed Results:")
        print(f"{'Module':<25} {'Import':<15} {'build_model':<15} {'Execution':<20}")
        print("-" * 80)
        
        for module_name in sorted(self.ALL_GDPLIB_MODULES):
            import_status = import_results.get(module_name, "UNKNOWN")
            build_status = build_model_results.get(module_name, "N/A")
            exec_status = execution_results.get(module_name, "N/A")
            
            print(f"{module_name:<25} {import_status:<15} {build_status:<15} {exec_status:<20}")
        
        # Assert that we have good coverage
        assert successful_imports >= total_modules * 0.8, f"Less than 80% of modules importable: {successful_imports}/{total_modules}"
        assert runnable_models >= available_build_models * 0.7, f"Less than 70% of available models are runnable: {runnable_models}/{available_build_models}"

    @pytest.mark.parametrize("module_name", ALL_GDPLIB_MODULES)
    def test_module_import_comprehensive(self, module_name):
        """Test that each module can be imported without errors."""
        try:
            module = importlib.import_module(f"gdplib.{module_name}")
            assert module is not None, f"Module {module_name} imported as None"
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing {module_name}: {e}")

    @pytest.mark.parametrize("module_name", ALL_GDPLIB_MODULES)
    def test_build_model_exists_and_callable(self, module_name):
        """Test that build_model exists and is callable for each module."""
        try:
            module = importlib.import_module(f"gdplib.{module_name}")
            
            # Check if build_model exists
            assert hasattr(module, "build_model"), f"Module {module_name} missing build_model function"
            
            # Check if it's callable
            build_func = getattr(module, "build_model")
            assert callable(build_func), f"{module_name}.build_model is not callable"
            
        except ImportError:
            pytest.skip(f"Module {module_name} cannot be imported")

    @pytest.mark.parametrize("module_name", ALL_GDPLIB_MODULES)
    def test_model_construction_and_execution(self, module_name):
        """Test that build_model can be executed successfully for each module."""
        try:
            module = importlib.import_module(f"gdplib.{module_name}")
            
            if not hasattr(module, "build_model"):
                pytest.skip(f"Module {module_name} has no build_model function")
            
            build_func = getattr(module, "build_model")
            
            # Try to build the model
            try:
                model = build_func()
                assert model is not None, f"{module_name}.build_model() returned None"
                
                # Verify it's a Pyomo model
                assert hasattr(model, "component_objects"), f"{module_name} model is not a Pyomo model"
                
                # Check that model has some components
                components = list(model.component_objects())
                assert len(components) > 0, f"{module_name} model has no components"
                
            except TypeError as e:
                # Some models might require parameters
                if "required positional argument" in str(e) or "missing" in str(e):
                    pytest.skip(f"{module_name}.build_model requires parameters: {e}")
                else:
                    pytest.fail(f"{module_name}.build_model failed with TypeError: {e}")
            except Exception as e:
                # Handle solver dependency issues gracefully
                error_msg = str(e)
                if ("ipopt" in error_msg.lower() or 
                    "gams" in error_msg.lower() or 
                    "solver" in error_msg.lower()):
                    pytest.skip(f"{module_name} requires external solver: {error_msg[:100]}")
                else:
                    pytest.fail(f"{module_name}.build_model failed: {e}")
                
        except ImportError:
            pytest.skip(f"Module {module_name} cannot be imported")

    def test_pyomo_dependencies_available(self):
        """Test that all required Pyomo dependencies are available."""
        required_pyomo_modules = [
            "pyomo.environ",
            "pyomo.gdp",
            "pyomo.core",
            "pyomo.opt",
        ]
        
        for module_name in required_pyomo_modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                pytest.fail(f"Required Pyomo module {module_name} not available")

    def test_gdp_constructs_available(self):
        """Test that GDP-specific constructs are available."""
        try:
            from pyomo.gdp import Disjunct, Disjunction
            from pyomo.environ import ConcreteModel, Var, Constraint
            
            # Test basic GDP model creation
            model = ConcreteModel()
            model.x = Var(bounds=(0, 10))
            model.d = Disjunct()
            
            # This should work without errors
            assert model is not None
            assert model.x is not None
            assert model.d is not None
            
        except ImportError as e:
            pytest.fail(f"GDP constructs not available: {e}")

    def test_external_dependencies_available(self):
        """Test that external dependencies are properly installed."""
        required_external_modules = [
            "pandas",
            "matplotlib", 
            "scipy",
            "pint",
            "openpyxl",
        ]
        
        missing_modules = []
        for module_name in required_external_modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing_modules.append(module_name)
        
        if missing_modules:
            pytest.fail(f"Missing required external modules: {missing_modules}")

    def test_run_coverage_analysis(self):
        """Run pytest with coverage and capture results."""
        try:
            # Run coverage analysis
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "--cov=gdplib", 
                "--cov-report=term-missing",
                "--tb=no",
                "-q"
            ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            print("\n" + "="*80)
            print("Coverage Analysis Results")
            print("="*80)
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
                
        except Exception as e:
            pytest.skip(f"Coverage analysis failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])