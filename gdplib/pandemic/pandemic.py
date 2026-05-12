"""
pandemic.py

GDP model for optimal disease control via an event-constrained SEIR formulation.

The model minimizes the total quarantine intervention over a time horizon [0, 200] days
subject to SEIR epidemic dynamics and an event constraint requiring the fraction of
infected individuals i(t) to remain below a healthcare capacity threshold i_max = 0.02
for at least alpha = 0.9 of the time horizon. SEIR dynamics are discretized via
backward finite differences, and disjunctions encode whether the capacity constraint
is satisfied at each time point. The event probability is approximated by trapezoidal
integration of the disjunct indicator variables over the discretized time domain.

References
----------
[1] Ovalle, D., Mazzadi, S., Laird, C. D., Grossmann, I. E., & Pulsipher, J. L. (2025).
    Event constrained programming. Computers & Chemical Engineering, 199, 109145.
    https://doi.org/10.1016/j.compchemeng.2025.109145
"""

import numpy as np
from scipy.integrate import odeint
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    minimize,
    Constraint,
    TransformationFactory,
    SolverFactory,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.dae import ContinuousSet, DerivativeVar, Integral


def _seir_ode(y, t, gamma, beta, xi, u0):
    s, e, i, r = y
    return [
        -(1 - u0) * beta * s * i,
        (1 - u0) * beta * s * i - xi * e,
        xi * e - gamma * i,
        gamma * i,
    ]


def build_model(num_times=101):
    """
    Build the GDP model for optimal disease control using an event-constrained SEIR model.

    The model minimizes the total quarantine control effort u(t) over [0, 200] days
    subject to SEIR epidemic dynamics and an event constraint that requires the fraction
    of infected individuals i(t) to stay below the healthcare capacity threshold
    i_max = 0.02 for at least alpha = 0.9 of the time horizon. The event probability is
    approximated via trapezoidal integration of the disjunct indicator variables over
    the discretized time domain. Variables are initialized from an uncontrolled SEIR
    trajectory (u=0.5) computed via numerical integration.

    Parameters
    ----------
    num_times : int, optional
        Number of equidistant time points for the linspace discretization of [0, 200].
        Additional fine-grained points near t=0 are appended to capture early-time
        dynamics. Default is 101.

    Returns
    -------
    m : pyomo.ConcreteModel
        Pyomo GDP model of the event-constrained disease control problem,
        ready for GDP transformation and solving.

    References
    ----------
    [1] Ovalle, D., Mazzadi, S., Laird, C. D., Grossmann, I. E., & Pulsipher, J. L. (2025).
        Event constrained programming. Computers & Chemical Engineering, 199, 109145.
        https://doi.org/10.1016/j.compchemeng.2025.109145
    """
    # SEIR epidemiological parameters
    beta = 0.727  # infection rate (rho)
    gamma = 0.303  # recovery rate (eta)
    xi = 0.3  # incubation rate (zeta)
    N = 1e5

    # Constraint threshold, time horizon, and event probability level
    i_max = 0.02
    tol = 1e-9
    t0 = 0
    tf = 200
    alpha = 0.9
    u0 = 0.5

    # Extra points near t=0 resolve sharp early-time dynamics in the SEIR trajectory
    extra_ts = [0.001, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08, 0.2, 0.4, 0.8]
    ts = np.sort(np.append(np.linspace(t0, tf, num_times), extra_ts))

    # Initial conditions (fractional populations)
    y0 = [1 - 1 / N, 1 / N, 0, 0]

    # Warm-start initialization from a fixed-control (u=0.5) SEIR trajectory
    seir = odeint(_seir_ode, y0, ts, args=(gamma, beta, xi, u0))
    s_init = dict(zip(ts, seir[:, 0]))
    e_init = dict(zip(ts, seir[:, 1]))
    i_init = dict(zip(ts, seir[:, 2]))
    r_init = dict(zip(ts, seir[:, 3]))

    m = ConcreteModel()

    m.t = ContinuousSet(initialize=ts)
    m.s = Var(m.t, bounds=(0, 1), initialize=s_init)
    m.e = Var(m.t, bounds=(0, 1), initialize=e_init)
    m.i = Var(m.t, bounds=(0, 1), initialize=i_init)
    m.r = Var(m.t, bounds=(0, 1), initialize=r_init)
    m.u = Var(m.t, bounds=(0, 0.8), initialize=u0)

    m.dsdt = DerivativeVar(m.s, wrt=m.t)
    m.dedt = DerivativeVar(m.e, wrt=m.t)
    m.didt = DerivativeVar(m.i, wrt=m.t)
    m.drdt = DerivativeVar(m.r, wrt=m.t)

    @m.ConstraintList()
    def inits(m):
        yield m.s[0] == y0[0]
        yield m.e[0] == y0[1]
        yield m.i[0] == y0[2]
        yield m.r[0] == y0[3]

    @m.Constraint(m.t)
    def ode1(m, t):
        if t == 0:
            return Constraint.Skip
        return m.dsdt[t] == -(1 - m.u[t]) * beta * m.s[t] * m.i[t]

    @m.Constraint(m.t)
    def ode2(m, t):
        if t == 0:
            return Constraint.Skip
        return m.dedt[t] == (1 - m.u[t]) * beta * m.s[t] * m.i[t] - xi * m.e[t]

    @m.Constraint(m.t)
    def ode3(m, t):
        if t == 0:
            return Constraint.Skip
        return m.didt[t] == xi * m.e[t] - gamma * m.i[t]

    @m.Constraint(m.t)
    def ode4(m, t):
        if t == 0:
            return Constraint.Skip
        return m.drdt[t] == gamma * m.i[t]

    def _intu(m, t):
        return m.u[t]

    m.intu = Integral(m.t, wrt=m.t, rule=_intu)
    m.obj = Objective(sense=minimize, expr=m.intu)

    discretizer = TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m, wrt=m.t, scheme='BACKWARD', nfe=len(ts) - 1)

    # Disjunctions: capacity constraint satisfied or violated at each time point
    def build_satisfy_constraints(disjunct, t):
        m = disjunct.model()

        @disjunct.Constraint()
        def infected_upper(disjunct):
            return m.i[t] <= i_max

    def build_not_satisfy_constraints(disjunct, t):
        m = disjunct.model()

        @disjunct.Constraint()
        def infected_upper(disjunct):
            return m.i[t] >= i_max + tol

    m.disjunct_satisfy = Disjunct(m.t, rule=build_satisfy_constraints)
    m.disjunct_not_satisfy = Disjunct(m.t, rule=build_not_satisfy_constraints)

    @m.Disjunction(m.t)
    def constrs_satisfy_or_not(m, t):
        return [m.disjunct_satisfy[t], m.disjunct_not_satisfy[t]]

    # Event constraint: fraction of time with i(t) <= i_max must be >= alpha
    m.event_constr = Constraint(
        expr=1
        / tf
        * sum(
            (
                m.disjunct_satisfy[m.t.at(i - 1)].indicator_var.get_associated_binary()
                + m.disjunct_satisfy[m.t.at(i)].indicator_var.get_associated_binary()
            )
            * (m.t.at(i) - m.t.at(i - 1))
            / 2
            for i in range(2, len(m.t) + 1)
        )
        >= alpha
    )

    return m


if __name__ == '__main__':
    m = build_model()
    TransformationFactory('gdp.bigm').apply_to(m)
    opt = SolverFactory('baron')
    results = opt.solve(m, tee=True)
    print(results)
