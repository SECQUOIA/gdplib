"""Distillation column model for 2018 PSE conference"""

from __future__ import division

from math import fabs

from pyomo.environ import SolverFactory, Suffix, value
from pyomo.util.infeasible import log_infeasible_constraints

from gdplib.gdp_col.column import build_column
from gdplib.gdp_col.initialize import initialize


def main():
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
    # Run custom initialization routine
    initialize(m)

    m.BigM = Suffix(direction=Suffix.LOCAL)
    m.BigM[None] = 100

    SolverFactory('gdpopt').solve(
        m, tee=True, init_strategy='fix_disjuncts',
        mip_solver='glpk')
    log_infeasible_constraints(m, tol=1E-3)
    display_column(m)
    return m


def display_column(m):
    print('Objective: %s' % value(m.obj))
    print('Qc: {: >3.0f}kW  DB: {: >3.0f} DT: {: >3.0f} dis: {: >3.0f}'
          .format(value(m.Qc * 1E3),
                  value(m.D['benzene']),
                  value(m.D['toluene']),
                  value(m.dis)))
    for t in reversed(list(m.trays)):
        print('T{: >2.0f}-{:1.0g} T: {: >3.0f} '
              'F: {: >4.0f} '
              'L: {: >4.0f} V: {: >4.0f} '
              'xB: {: >3.0f} xT: {: >3.0f} yB: {: >3.0f} yT: {: >3.0f}'
              .format(t,
                      fabs(value(m.tray[t].indicator_var))
                      if t in m.conditional_trays else 1,
                      value(m.T[t]) - 273.15,
                      value(sum(m.feed[c] for c in m.comps))
                      if t == m.feed_tray else 0,
                      value(m.liq[t]),
                      value(m.vap[t]),
                      value(m.x['benzene', t]) * 100,
                      value(m.x['toluene', t]) * 100,
                      value(m.y['benzene', t]) * 100,
                      value(m.y['toluene', t]) * 100
                      ))
    print('Qb: {: >3.0f}kW  BB: {: > 3.0f} BT: {: >3.0f} bot: {: >3.0f}'
          .format(value(m.Qb * 1E3),
                  value(m.B['benzene']),
                  value(m.B['toluene']),
                  value(m.bot)))
    for t in reversed(list(m.trays)):
        print('T{: >2.0f}-{:1.0g} '
              'FB: {: >3.0f} FT: {: >3.0f} '
              'LB: {: >4.0f} LT: {: >4.0f} VB: {: >4.0f} VT: {: >4.0f}'
              .format(t,
                      fabs(value(m.tray[t].indicator_var))
                      if t in m.conditional_trays else 1,
                      value(m.feed['benzene']) if t == m.feed_tray else 0,
                      value(m.feed['toluene']) if t == m.feed_tray else 0,
                      value(m.L['benzene', t]),
                      value(m.L['toluene', t]),
                      value(m.V['benzene', t]),
                      value(m.V['toluene', t])
                      ))
    print('RF: {: >3.2f} RB: {: >3.2f}'
          .format(value(m.reflux_frac / (1 - m.reflux_frac)),
                  value(m.boilup_frac / (1 - m.boilup_frac))))


if __name__ == "__main__":
    m = main()
