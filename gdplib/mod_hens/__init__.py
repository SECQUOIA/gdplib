from .conventional import build_conventional as _conv
from .modular_discrete import (
    build_modular_option as _disc_opt,
    build_require_modular as _disc_mod,
)
from .modular_discrete_single_module import build_single_module as _disc_sing
from .modular_integer import (
    build_modular_option as _int_opt,
    build_require_modular as _int_mod,
    build_single_module as _int_sing,
)

_CASE_BUILDERS = {
    "conventional": _conv,
    "single_module_integer": _int_sing,
    "multiple_module_integer": _int_mod,
    "require_modular_integer": _int_mod,
    "mixed_integer": _int_opt,
    "modular_option_integer": _int_opt,
    "single_module_discrete": _disc_sing,
    "multiple_module_discrete": _disc_mod,
    "require_modular_discrete": _disc_mod,
    "mixed_discrete": _disc_opt,
    "modular_option_discrete": _disc_opt,
}


def build_model(case="conventional", cafaro_approx=True, num_stages=4):
    """Build a heat exchanger network synthesis (HENS) model.

    Args:
        case: Model variant (conventional, single_module_integer, etc.).
            The legacy require_modular_* and modular_option_* case names are
            accepted as aliases.
        cafaro_approx: Whether to use Cafaro approximation
        num_stages: Number of stages in the heat exchanger network

    Returns:
        Pyomo model object
    """
    try:
        builder = _CASE_BUILDERS[case]
    except KeyError:
        valid_cases = ", ".join(sorted(_CASE_BUILDERS))
        raise ValueError(
            f"Invalid mod_hens case {case!r}. Expected one of: {valid_cases}"
        ) from None

    return builder(cafaro_approx, num_stages)


__all__ = ["build_model"]
