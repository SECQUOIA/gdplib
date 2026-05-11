from build_model_test_utils import (
    get_required_build_model_parameters,
    is_missing_external_solver_error,
)


def test_required_build_model_parameters_detects_required_arguments():
    def build_model(case, *, formulation):
        return None

    assert get_required_build_model_parameters(build_model) == ["case", "formulation"]


def test_required_build_model_parameters_ignores_optional_and_variadic_arguments():
    def build_model(case=None, *args, **kwargs):
        return None

    assert get_required_build_model_parameters(build_model) == []


def test_required_build_model_parameters_does_not_infer_from_type_error_text():
    def build_model():
        raise TypeError("missing internal data")

    assert get_required_build_model_parameters(build_model) == []


def test_missing_external_solver_error_detection():
    assert is_missing_external_solver_error(
        RuntimeError("No executable found for solver 'ipopt'")
    )
    assert is_missing_external_solver_error(
        RuntimeError("No 'gams' command found on system PATH")
    )
    assert not is_missing_external_solver_error(TypeError("missing internal data"))
