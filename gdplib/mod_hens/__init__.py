from functools import partial

from .conventional import build_conventional as _conv
from .modular_discrete import build_modular_option as _disc_opt, build_require_modular as _disc_mod
from .modular_discrete_single_module import build_single_module as _disc_sing
from .modular_integer import (
    build_modular_option as _int_opt, build_require_modular as _int_mod,
    build_single_module as _int_sing, )

# These are the functions that we want to expose as public
build_conventional = partial(_conv, cafaro_approx=True, num_stages=4)
build_integer_single_module = partial(_int_sing, cafaro_approx=True, num_stages=4)
build_integer_require_modular = partial(_int_mod, cafaro_approx=True, num_stages=4)
build_integer_modular_option = partial(_int_opt, cafaro_approx=True, num_stages=4)
build_discrete_single_module = partial(_disc_sing, cafaro_approx=True, num_stages=4)
build_discrete_require_modular = partial(_disc_mod, cafaro_approx=True, num_stages=4)
build_discrete_modular_option = partial(_disc_opt, cafaro_approx=True, num_stages=4)

__all__ = ['build_conventional', 'build_integer_single_module', 'build_integer_require_modular',
           'build_integer_modular_option', 'build_discrete_single_module', 'build_discrete_require_modular',
           'build_discrete_modular_option']
