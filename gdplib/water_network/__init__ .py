# from .wn_pwl import build_model_Piecewise as _pwl
# from .wn_q import build_model_Quadratic as _q

# from .wn_minlp import build_model_MINLP as _original
from gdplib.water_network.wnd import build_model

# def build_model(case="quadratic"):
#     if case == "quadratic":
#         return _q()
#     elif case == "piecewise":
#         return _pwl()
#     else:
#         raise ValueError(f"Invalid case: {case}")


__all__ = ['build_model']
# if __name__ == "__main__":
#     build_model
