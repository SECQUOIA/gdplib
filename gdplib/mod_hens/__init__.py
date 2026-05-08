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


def build_model(case="conventional", cafaro_approx=True, num_stages=4):
    """Build a heat exchanger network synthesis (HENS) model.

    Args:
        case: Model variant (conventional, single_module_integer, etc.)
        cafaro_approx: Whether to use Cafaro approximation
        num_stages: Number of stages in the heat exchanger network

    Returns:
        Pyomo model object
    """
    # TODO: we might need to come up with better names for these cases.
    if case == "conventional":
        return _conv(cafaro_approx, num_stages)
    elif case == "single_module_integer":
        return _int_sing(cafaro_approx, num_stages)
    elif case == "require_modular_integer":
        return _int_mod(cafaro_approx, num_stages)
    elif case == "modular_option_integer":
        return _int_opt(cafaro_approx, num_stages)
    elif case == "single_module_discrete":
        return _disc_sing(cafaro_approx, num_stages)
    elif case == "require_modular_discrete":
        return _disc_mod(cafaro_approx, num_stages)
    elif case == "modular_option_discrete":
        return _disc_opt(cafaro_approx, num_stages)


__all__ = ["build_model"]
