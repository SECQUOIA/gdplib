""" Side feed flash """

from __future__ import division

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    exp,
    minimize,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    SolverFactory,
    value,
    Var,
)


def calc_side_feed_flash(m):
    msf = ConcreteModel('SIDE FEED FLASH')

    msf.nc = RangeSet(1, m.c, doc='Number of components')

    m.xfi = {}
    for nc in msf.nc:
        m.xfi[nc] = 1 / m.c

    msf.Tf = Param(doc='Side feed temperature in K', initialize=m.Tf0)
    msf.xf = Var(
        msf.nc,
        doc='Side feed liquid composition',
        domain=NonNegativeReals,
        bounds=(0, 1),
        initialize=m.xfi,
    )
    msf.yf = Var(
        msf.nc,
        doc='Side feed vapor composition',
        domain=NonNegativeReals,
        bounds=(0, 1),
        initialize=m.xfi,
    )
    msf.Keqf = Var(
        msf.nc,
        doc='Vapor-liquid equilibrium constant',
        domain=NonNegativeReals,
        bounds=(0, 10),
        initialize=0,
    )
    msf.Pvf = Var(
        msf.nc,
        doc='Side feed vapor pressure in bar',
        domain=NonNegativeReals,
        bounds=(0, 10),
        initialize=0,
    )
    msf.q = Var(
        doc='Vapor fraction', bounds=(0, 1), domain=NonNegativeReals, initialize=0
    )

    @msf.Constraint(doc="Vapor fraction")
    def _algq(msf):
        return (
            sum(
                m.xfi[nc] * (1 - msf.Keqf[nc]) / (1 + msf.q * (msf.Keqf[nc] - 1))
                for nc in msf.nc
            )
            == 0
        )

    @msf.Constraint(msf.nc, doc="Side feed liquid composition")
    def _algx(msf, nc):
        return msf.xf[nc] * (1 + msf.q * (msf.Keqf[nc] - 1)) == m.xfi[nc]

    @msf.Constraint(msf.nc, doc="Side feed vapor composition")
    def _algy(msf, nc):
        return msf.yf[nc] == msf.xf[nc] * msf.Keqf[nc]

    @msf.Constraint(msf.nc, doc="Vapor-liquid equilibrium constant")
    def _algKeq(msf, nc):
        return msf.Keqf[nc] * m.Pf == msf.Pvf[nc]

    @msf.Constraint(msf.nc, doc="Side feed vapor pressure")
    def _algPvf(msf, nc):
        return msf.Pvf[nc] == m.prop[nc, 'PC'] * exp(
            m.prop[nc, 'TC']
            / msf.Tf
            * (
                m.prop[nc, 'vpA'] * (1 - msf.Tf / m.prop[nc, 'TC'])
                + m.prop[nc, 'vpB'] * (1 - msf.Tf / m.prop[nc, 'TC']) ** 1.5
                + m.prop[nc, 'vpC'] * (1 - msf.Tf / m.prop[nc, 'TC']) ** 3
                + m.prop[nc, 'vpD'] * (1 - msf.Tf / m.prop[nc, 'TC']) ** 6
            )
        )

    msf.OBJ = Objective(expr=1, sense=minimize)

    ####
    SolverFactory('ipopt').solve(msf, tee=False)

    m.yfi = {}
    for nc in msf.nc:
        m.yfi[nc] = value(msf.yf[nc])

    m.q_init = value(msf.q)

    return m
