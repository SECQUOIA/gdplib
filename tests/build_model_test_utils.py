import inspect


def get_required_build_model_parameters(build_func):
    """Return required parameters that prevent zero-argument construction."""
    try:
        signature = inspect.signature(build_func)
    except (TypeError, ValueError):
        return []

    required_params = []
    for name, param in signature.parameters.items():
        if param.default is not inspect.Parameter.empty:
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            required_params.append(name)
    return required_params


def is_missing_external_solver_error(error):
    """Return whether an exception is caused by an unavailable optional solver."""
    error_msg = str(error).lower()
    return any(
        pattern in error_msg
        for pattern in (
            "no executable found for solver",
            "no 'gams' command",
            "could not locate the 'ipopt' executable",
            "ipopt executable",
        )
    )
