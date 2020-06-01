from __future__ import division

from pyomo.environ import (ConcreteModel, NonNegativeReals, Set, SolverFactory,
                           Var, log, sqrt)


def calculate_Fenske(xD, xB):
    m = ConcreteModel()
    min_T, max_T = 300, 400
    m.comps = Set(initialize=['benzene', 'toluene'])
    m.trays = Set(initialize=['condenser', 'reboiler'])
    m.Kc = Var(
        m.comps, m.trays, doc='Phase equilibrium constant',
        domain=NonNegativeReals, initialize=1, bounds=(0, 1000))
    m.T = Var(m.trays, doc='Temperature [K]',
              domain=NonNegativeReals,
              bounds=(min_T, max_T))
    m.T['condenser'].fix(82 + 273.15)
    m.T['reboiler'].fix(108 + 273.15)
    m.P = Var(doc='Pressure [bar]',
              bounds=(0, 5))
    m.P.fix(1.01)
    m.T_ref = 298.15
    m.gamma = Var(
        m.comps, m.trays,
        doc='liquid activity coefficent of component on tray',
        domain=NonNegativeReals, bounds=(0, 10), initialize=1)
    m.Pvap = Var(
        m.comps, m.trays,
        doc='pure component vapor pressure of component on tray in bar',
        domain=NonNegativeReals, bounds=(1E-3, 5), initialize=0.4)
    m.Pvap_X = Var(
        m.comps, m.trays,
        doc='Related to fraction of critical temperature (1 - T/Tc)',
        bounds=(0.25, 0.5), initialize=0.4)

    m.pvap_const = {
        'benzene': {'A': -6.98273, 'B': 1.33213, 'C': -2.62863,
                    'D': -3.33399, 'Tc': 562.2, 'Pc': 48.9},
        'toluene': {'A': -7.28607, 'B': 1.38091, 'C': -2.83433,
                    'D': -2.79168, 'Tc': 591.8, 'Pc': 41.0}}

    @m.Constraint(m.comps, m.trays)
    def phase_equil_const(_, c, t):
        return m.Kc[c, t] * m.P == (
            m.gamma[c, t] * m.Pvap[c, t])

    @m.Constraint(m.comps, m.trays)
    def Pvap_relation(_, c, t):
        k = m.pvap_const[c]
        x = m.Pvap_X[c, t]
        return (log(m.Pvap[c, t]) - log(k['Pc'])) * (1 - x) == (
            k['A'] * x +
            k['B'] * x ** 1.5 +
            k['C'] * x ** 3 +
            k['D'] * x ** 6)

    @m.Constraint(m.comps, m.trays)
    def Pvap_X_defn(_, c, t):
        k = m.pvap_const[c]
        return m.Pvap_X[c, t] == 1 - m.T[t] / k['Tc']

    @m.Constraint(m.comps, m.trays)
    def gamma_calc(_, c, t):
        return m.gamma[c, t] == 1

    m.relative_volatility = Var(m.trays, domain=NonNegativeReals)

    @m.Constraint(m.trays)
    def relative_volatility_calc(_, t):
        return m.Kc['benzene', t] == (
            m.Kc['toluene', t] * m.relative_volatility[t])

    @m.Expression()
    def fenske(_):
        return log((xD / (1 - xD)) * (xB / (1 - xB))) / (
            log(sqrt(m.relative_volatility['condenser'] *
                     m.relative_volatility['reboiler'])))

    SolverFactory('ipopt').solve(m, tee=True)
    from pyomo.util.infeasible import log_infeasible_constraints
    log_infeasible_constraints(m, tol=1E-3)
    m.fenske.display()


if __name__ == '__main__':
    m = calculate_Fenske(0.95, 0.95)
