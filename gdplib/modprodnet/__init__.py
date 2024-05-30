from .model import build_model as _capacity_expansion
from .distributed import build_modular_model as build_distributed_model
from .quarter_distributed import build_modular_model as build_quarter_distributed_model


def build_model(case="Growth"):
    if case in ["Growth", "Dip", "Decay"]:
        return _capacity_expansion(case)
    elif case == "Distributed":
        return build_distributed_model()
    elif case == "QuarterDistributed":
        return build_quarter_distributed_model()
    else:
        raise ValueError("Invalid case: {}".format(case))


__all__ = ['build_model']
