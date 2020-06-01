from pyomo.environ import Suffix

from .column import build_column


def build_model():
    m = build_column(min_trays=8, max_trays=17, xD=0.95, xB=0.95)
    # Fix feed conditions
    m.feed['benzene'].fix(50)
    m.feed['toluene'].fix(50)
    m.T_feed.fix(368)
    m.feed_vap_frac.fix(0.40395)
    # Initial values of reflux and reboil ratios
    m.reflux_ratio.set_value(1.4)
    m.reboil_ratio.set_value(1.3)
    # Fix to be total condenser
    m.partial_cond.deactivate()
    m.total_cond.indicator_var.fix(1)
    # Give initial values of the tray existence/absence
    for t in m.conditional_trays:
        m.tray[t].indicator_var.set_value(1)
        m.no_tray[t].indicator_var.set_value(0)

    m.BigM = Suffix(direction=Suffix.LOCAL)
    m.BigM[None] = 100


__all__ = ['build_model']
