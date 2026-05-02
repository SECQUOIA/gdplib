"""
grid.py

GDP model for event-constrained optimal power flow on the IEEE 14-bus network.

The model minimizes the total capacity expansion (slack) needed for generators and
transmission lines to satisfy power balance constraints across a set of sampled load
scenarios. An event constraint enforces that, in at least 90% of scenarios (alpha=0.9),
at least a minimum number of generator and line capacity limits are simultaneously
satisfied. The GDP formulation uses disjunctions to model whether each capacity
constraint is satisfied or violated in each scenario, and a logical equivalence to
define the satisfaction event per scenario using ATLEAST logic.

References
----------
[1] Ovalle, D., Mazzadi, S., Laird, C. D., Grossmann, I. E., & Pulsipher, J. L. (2025).
    Event constrained programming. Computers & Chemical Engineering, 199, 109145.
    https://doi.org/10.1016/j.compchemeng.2025.109145
"""

import numpy as np
from math import ceil
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Var,
    Objective,
    minimize,
    Constraint,
    BooleanVar,
    LogicalConstraint,
    Param,
    TransformationFactory,
    SolverFactory,
    atleast,
    land,
    equivalent,
)
from pyomo.gdp import Disjunct, Disjunction


def build_model(active_gens=4, active_lines=20, num_samples=500):
    """
    Build the GDP model for event-constrained optimal power flow on the IEEE 14-bus network.

    The model minimizes the total capacity slack for generators and transmission lines
    across a set of Monte Carlo load scenarios drawn from a multivariate normal
    distribution. An event constraint enforces that, with probability at least alpha=0.9
    (approximated via Sample Average Approximation), at least active_gens generators and
    active_lines transmission lines simultaneously satisfy their capacity bounds in a
    given scenario. The event logic uses the ATLEAST operator from Generalized Disjunctive
    Programming, which is less conservative than a joint chance constraint requiring all
    equipment to be within limits simultaneously.

    Parameters
    ----------
    active_gens : int, optional
        Minimum number of generators whose capacity constraints must be satisfied
        in a scenario for the event to hold. Default is 4.
    active_lines : int, optional
        Minimum number of transmission lines whose capacity constraints must be
        satisfied in a scenario for the event to hold. Default is 20.
    num_samples : int, optional
        Number of Monte Carlo samples for the uncertain nodal load demands. Default is 500.

    Returns
    -------
    m : pyomo.ConcreteModel
        Pyomo GDP model of the event-constrained optimal power flow problem,
        ready for GDP transformation and solving.

    References
    ----------
    [1] Ovalle, D., Mazzadi, S., Laird, C. D., Grossmann, I. E., & Pulsipher, J. L. (2025).
        Event constrained programming. Computers & Chemical Engineering, 199, 109145.
        https://doi.org/10.1016/j.compchemeng.2025.109145
    """
    if not (0 <= active_gens <= 5):
        raise ValueError(f"active_gens must be between 0 and 5, got {active_gens}")
    if not (0 <= active_lines <= 20):
        raise ValueError(f"active_lines must be between 0 and 20, got {active_lines}")

    # Nominal loads and covariance for uncertain demand parameters
    theta_nom = np.array([87.3, 50.0, 25.0, 28.8, 50.0, 25.0, 0, 0, 0, 0, 0])
    covar = np.identity(len(theta_nom)) * 1200.0
    covar[covar == 0] = 240.0

    # Network parameters
    line_cap = 50.0
    gen_cap = [332.0, 140.0, 100.0, 100.0, 100.0]
    num_lines = 20
    num_gens = 5

    # Sample uncertain load scenarios
    np.random.seed(42)
    thetas = np.random.multivariate_normal(theta_nom, covar, num_samples)
    thetas[thetas <= 0.0] = 0.0

    m = ConcreteModel()

    # Index sets
    m.K = RangeSet(0, num_samples - 1)
    m.G = RangeSet(1, num_gens)
    m.L = RangeSet(1, num_lines)

    # Variables
    m.z_line = Var(m.L, m.K, bounds=(-150, 150))
    m.z_gen = Var(m.G, m.K, bounds=(0, 632))
    m.d_line = Var(m.L, bounds=(0, 100))
    m.d_gen = Var(m.G, bounds=(0, 300))

    # Objective: minimize total capacity slack
    m.obj = Objective(
        sense=minimize,
        expr=sum(m.d_gen[g] for g in m.G) + sum(m.d_line[l] for l in m.L),
    )

    # Power balance constraints (IEEE 14-bus topology)
    @m.Constraint(m.K)
    def h1(m, k):
        return m.z_gen[1, k] - m.z_line[1, k] - m.z_line[6, k] == 0

    @m.Constraint(m.K)
    def h2(m, k):
        return (
            m.z_gen[2, k]
            + m.z_line[1, k]
            - sum(m.z_line[i, k] for i in [2, 4, 5])
            - thetas[k, 0]
            == 0
        )

    @m.Constraint(m.K)
    def h3(m, k):
        return m.z_gen[3, k] + m.z_line[2, k] - m.z_line[3, k] - thetas[k, 1] == 0

    @m.Constraint(m.K)
    def h4(m, k):
        return (
            sum(m.z_line[i, k] for i in [3, 4, 8])
            - sum(m.z_line[i, k] for i in [7, 11])
            - thetas[k, 2]
            == 0
        )

    @m.Constraint(m.K)
    def h5(m, k):
        return (
            sum(m.z_line[i, k] for i in [5, 6, 7, 12]) - thetas[k, 3] == 0
        )

    @m.Constraint(m.K)
    def h6(m, k):
        return (
            m.z_gen[4, k]
            + sum(m.z_line[i, k] for i in [16, 18])
            - sum(m.z_line[i, k] for i in [12, 19])
            - thetas[k, 4]
            == 0
        )

    @m.Constraint(m.K)
    def h7(m, k):
        return m.z_line[9, k] - sum(m.z_line[i, k] for i in [8, 10]) == 0

    @m.Constraint(m.K)
    def h8(m, k):
        return m.z_gen[5, k] - m.z_line[9, k] == 0

    @m.Constraint(m.K)
    def h9(m, k):
        return (
            sum(m.z_line[i, k] for i in [10, 11])
            - sum(m.z_line[i, k] for i in [13, 14])
            - thetas[k, 5]
            == 0
        )

    @m.Constraint(m.K)
    def h10(m, k):
        return sum(m.z_line[i, k] for i in [13, 20]) - thetas[k, 6] == 0

    @m.Constraint(m.K)
    def h11(m, k):
        return m.z_line[19, k] - m.z_line[20, k] - thetas[k, 7] == 0

    @m.Constraint(m.K)
    def h12(m, k):
        return m.z_line[17, k] - m.z_line[18, k] - thetas[k, 8] == 0

    @m.Constraint(m.K)
    def h13(m, k):
        return (
            m.z_line[15, k]
            - sum(m.z_line[i, k] for i in [16, 17])
            - thetas[k, 9]
            == 0
        )

    @m.Constraint(m.K)
    def h14(m, k):
        return m.z_line[14, k] - m.z_line[15, k] - thetas[k, 10] == 0

    # Generator disjunctions: capacity constraint satisfied or violated per scenario
    def build_satisfy_gen_constraints(disjunct, g, k):
        m = disjunct.model()

        @disjunct.Constraint()
        def gen_upper(disjunct):
            return m.z_gen[g, k] - gen_cap[g - 1] - m.d_gen[g] <= 0

    def build_not_satisfy_gen_constraints(disjunct, g, k):
        m = disjunct.model()

        @disjunct.Constraint()
        def gen_upper(disjunct):
            return m.z_gen[g, k] - gen_cap[g - 1] - m.d_gen[g] >= 0

    m.gen_disjunct1 = Disjunct(m.G, m.K, rule=build_satisfy_gen_constraints)
    m.gen_disjunct2 = Disjunct(m.G, m.K, rule=build_not_satisfy_gen_constraints)

    @m.Disjunction(m.G, m.K)
    def gen_constrs_satisfy_or_not(m, g, k):
        return [m.gen_disjunct1[g, k], m.gen_disjunct2[g, k]]

    # Line disjunctions: both bounds satisfied, lower violated, or upper violated
    def build_satisfy_line_constraints(disjunct, l, k):
        m = disjunct.model()

        @disjunct.Constraint()
        def line_lower(disjunct):
            return -m.z_line[l, k] - line_cap - m.d_line[l] <= 0

        @disjunct.Constraint()
        def line_upper(disjunct):
            return m.z_line[l, k] - line_cap - m.d_line[l] <= 0

    def build_not_satisfy_lower_line_constraints(disjunct, l, k):
        m = disjunct.model()

        @disjunct.Constraint()
        def line_lower(disjunct):
            return -m.z_line[l, k] - line_cap - m.d_line[l] >= 0

    def build_not_satisfy_upper_line_constraints(disjunct, l, k):
        m = disjunct.model()

        @disjunct.Constraint()
        def line_upper(disjunct):
            return m.z_line[l, k] - line_cap - m.d_line[l] >= 0

    m.line_disjunct1 = Disjunct(m.L, m.K, rule=build_satisfy_line_constraints)
    m.line_disjunct2 = Disjunct(
        m.L, m.K, rule=build_not_satisfy_lower_line_constraints
    )
    m.line_disjunct3 = Disjunct(
        m.L, m.K, rule=build_not_satisfy_upper_line_constraints
    )

    @m.Disjunction(m.L, m.K)
    def line_constrs_satisfy_or_not(m, l, k):
        return [m.line_disjunct1[l, k], m.line_disjunct2[l, k], m.line_disjunct3[l, k]]

    # Event variable: True if scenario k jointly satisfies enough constraints
    m.w = BooleanVar(m.K)

    @m.LogicalConstraint(m.K)
    def event_logic(m, k):
        return equivalent(
            land(
                atleast(active_gens, *m.gen_disjunct1[:, k].indicator_var),
                atleast(active_lines, *m.line_disjunct1[:, k].indicator_var),
            ),
            m.w[k],
        )

    # Chance constraint: event must hold in at least 90% of scenarios
    alpha = 0.9
    m.min_constrs = Param(mutable=True, initialize=ceil(alpha * num_samples))
    m.event_constr = LogicalConstraint(expr=atleast(m.min_constrs, m.w))

    return m


if __name__ == '__main__':
    m = build_model()
    TransformationFactory('core.logical_to_linear').apply_to(m)
    TransformationFactory('gdp.bigm').apply_to(m)
    opt = SolverFactory('gurobi_direct')
    results = opt.solve(m, tee=True)
    print(results)
