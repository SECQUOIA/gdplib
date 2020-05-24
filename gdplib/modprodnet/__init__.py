from functools import partial

from .model import build_model as _capacity_expansion
from .distributed import build_modular_model as build_distributed_model
from .quarter_distributed import build_modular_model as build_quarter_distributed_model

build_cap_expand_growth = partial(_capacity_expansion, case="Growth")
build_cap_expand_dip = partial(_capacity_expansion, case="Dip")
build_cap_expand_decay = partial(_capacity_expansion, case="Decay")

__all__ = ['build_cap_expand_growth', 'build_cap_expand_dip', 'build_cap_expand_decay', 'build_distributed_model',
           'build_quarter_distributed_model']
