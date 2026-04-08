"""
Multiperiod Blending Problem (MPBP)
-------------------------------------

GDP formulation of the multiperiod blending problem. The network contains supply
tanks (S), blending tanks (B), and demand tanks (D) connected by arcs over a
discrete time horizon T. Each blending tank operates in either charging (inflow)
or discharging (outflow) mode per time period, determined by GDP disjunctions.

A source-tracking (redundant) formulation decomposes flows and inventories by
origin, enabling valid linear inequalities for bilinear composition terms.

The objective maximizes net profit: demand revenue minus supply cost, variable
arc costs, and fixed arc activation costs over all time periods.

References:
    > Lotero, I., Trespalacios, F., Grossmann, I. E., Papageorgiou, D. J., & Cheon, M.-S. (2016). An MILP-MINLP decomposition method for the global optimization of a source based model of the multiperiod blending problem. Computers & Chemical Engineering, 87, 13–35. https://doi.org/10.1016/j.compchemeng.2015.12.017
    > Ovalle, D., Bhatia, A., Laird, C. D., & Grossmann, I. E. (2026). A logic-based decomposition for the global optimization of the multiperiod blending problem using symmetry-breaking cuts. Industrial & Engineering Chemistry Research, 65(7), 3981–3998. https://doi.org/10.1021/acs.iecr.5c02853

Command-line usage:
    python multiperiod_blending.py [--instance INSTANCE] [--solver SOLVER]

    Options:
        --instance  Path to the JSON instance file.
                    (default: instances_json/mpbp_6.json)
        --solver    Name of the solver to use, e.g. gurobi, cplex, glpk.
                    (default: gurobi)

    Examples:
        python multiperiod_blending.py
        python multiperiod_blending.py --solver cplex
        python multiperiod_blending.py --instance instances_json/mpbp_3.json --solver cplex
"""

import ast
import json
import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction


def convert_json_to_data(obj) -> dict:
    """
    Recursively convert a JSON-deserialized object back to the original Python structure.

    JSON serialization converts tuple keys to strings and tuples to lists.
    This function reverses those transformations for use with Pyomo sets and parameters.

    Parameters
    ----------
    obj : dict, list, or scalar
        A JSON-deserialized Python object (as returned by ``json.load``).

    Returns
    -------
    dict, tuple, or scalar
        The converted object with tuple keys and tuple values restored.
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Try to convert string keys that look like tuples back to tuple keys
            if isinstance(key, str) and key.startswith("(") and key.endswith(")"):
                try:
                    parsed_key = ast.literal_eval(key)
                    if isinstance(parsed_key, tuple):
                        new_key = parsed_key
                    else:
                        new_key = key
                except (ValueError, SyntaxError):
                    new_key = key
            else:
                new_key = key
            result[new_key] = convert_json_to_data(value)
        return result
    elif isinstance(obj, list):
        # Convert list back to tuple (assuming all tuples were converted to lists)
        return tuple(convert_json_to_data(item) for item in obj)
    else:
        return obj


def build_model(data: dict = None):
    """
    Build the multiperiod blending problem (MPBP) as a Pyomo GDP model.

    Parameters
    ----------
    data : dict
        Problem data dictionary loaded from a JSON instance file via
        ``convert_json_to_data``.

    Returns
    -------
    m : pyo.ConcreteModel
        Pyomo GDP model ready for transformation (e.g., ``gdp.bigm``) and solution.
    """

    if data is None:
        import os

        default_path = os.path.join(
            os.path.dirname(__file__),
            "instances_json",
            "mpbp_6.json",
        )
        with open(default_path, "r") as f:
            data = convert_json_to_data(json.load(f))

    # PYOMO MODEL
    m = pyo.ConcreteModel()

    # SETS
    m.S = pyo.Set(initialize=data["S"])  # Set of supply tanks
    m.B = pyo.Set(initialize=data["B"])  # Set of blending tanks
    m.D = pyo.Set(initialize=data["D"])  # Set of demand tanks
    m.N = pyo.Set(initialize=data["N"])  # Set of tanks
    m.Q = pyo.Set(initialize=data["Q"])  # Set of substances
    m.T = pyo.Set(initialize=data["T"])  # Set of discrete time periods
    m.A = pyo.Set(initialize=data["A"])  # Set of existing arcs

    m.R = pyo.Set(initialize=data["R"])  # Set of sources
    m.B_hat = pyo.Set(
        initialize=data["B_hat"]
    )  # Set of blending tanks with initial composition

    # Special node sets
    m.NB = pyo.Set(initialize=data["NB"])
    m.BN = pyo.Set(initialize=data["BN"])
    m.SD = pyo.Set(initialize=data["SD"])
    m.BD = pyo.Set(initialize=data["BD"])

    # PARAMETERS
    # Arc dependencies
    m.Nin = pyo.Param(m.N, initialize=data["Nin"], within=pyo.Any)
    m.Nout = pyo.Param(m.N, initialize=data["Nout"], within=pyo.Any)

    # Initial inventories
    m.I0 = pyo.Param(m.N, initialize=data["I0"], within=pyo.NonNegativeReals)

    # Initial composition
    m.C0 = pyo.Param(m.Q, m.B, initialize=data["C0"], within=pyo.NonNegativeReals)
    m.C0_hat = pyo.Param(
        m.Q, m.R, initialize=data["C0_hat"], within=pyo.NonNegativeReals
    )

    # Inventory bounds
    m.I_bounds = pyo.Param(m.N, initialize=data["I_bounds"], within=pyo.Any)

    # Flow bounds
    m.F_bounds = pyo.Param(m.A, initialize=data["F_bounds"], within=pyo.Any)
    m.Fmax = pyo.Param(initialize=data["Fmax"], within=pyo.NonNegativeReals)

    # Demanded flow bounds
    m.FD_bounds = pyo.Param(m.D, m.T, initialize=data["FD_bounds"], within=pyo.Any)

    # Demanded composition bounds
    m.CD_bounds = pyo.Param(m.Q, m.D, initialize=data["CD_bounds"], within=pyo.Any)

    # Composition bounds
    m.C_bounds = pyo.Param(m.Q, initialize=data["C_bounds"], within=pyo.Any)

    # Supply conditions
    m.CIN = pyo.Param(m.Q, m.S, initialize=data["CIN"], within=pyo.NonNegativeReals)
    m.FIN = pyo.Param(m.S, m.T, initialize=data["FIN"], within=pyo.NonNegativeReals)

    # Economic parameters
    m.betaT_s = pyo.Param(m.S, initialize=data["betaT_s"], within=pyo.Reals)
    m.betaT_d = pyo.Param(m.D, initialize=data["betaT_d"], within=pyo.Reals)
    m.alphaN = pyo.Param(m.A, initialize=data["alphaN"], within=pyo.Reals)
    m.betaN = pyo.Param(m.A, initialize=data["betaN"], within=pyo.Reals)

    # CONTINUOUS VARIABLES
    m.F = pyo.Var(m.A, m.T, within=pyo.NonNegativeReals, bounds=(0, m.Fmax))
    m.FD = pyo.Var(
        m.D, m.T, within=pyo.NonNegativeReals, bounds=lambda _, d, t: m.FD_bounds[d, t]
    )
    m.I = pyo.Var(
        m.N, m.T, within=pyo.NonNegativeReals, bounds=lambda _, n, t: m.I_bounds[n]
    )
    m.C = pyo.Var(
        m.Q,
        m.B,
        m.T,
        within=pyo.NonNegativeReals,
        bounds=lambda _, q, b, t: m.C_bounds[q],
    )

    m.F_til = pyo.Var(m.R, m.A, m.T, within=pyo.NonNegativeReals, bounds=(0, m.Fmax))
    m.I_til = pyo.Var(
        m.R,
        m.B,
        m.T,
        within=pyo.NonNegativeReals,
        bounds=lambda _, r, b, t: m.I_bounds[b],
    )

    # CONSTRAINTS

    # Supply inventory balance
    @m.Constraint(m.S, m.T)
    def supply_bal(m, s, t):
        """Inventory balance for supply tank s at time t."""
        if t == 1:
            return m.I[s, t] == m.I0[s] + m.FIN[s, t] - sum(
                m.F[s, n, t] for n in m.Nout[s]
            )
        else:
            return m.I[s, t] == m.I[s, t - 1] + m.FIN[s, t] - sum(
                m.F[s, n, t] for n in m.Nout[s]
            )

    # Demand inventory balance
    @m.Constraint(m.D, m.T)
    def demand_bal(m, d, t):
        """Inventory balance for demand tank d at time t. FD[d, t] is the withdrawn outflow."""
        if t == 1:
            return (
                m.I[d, t] == m.I0[d] + sum(m.F[n, d, t] for n in m.Nin[d]) - m.FD[d, t]
            )
        else:
            return (
                m.I[d, t]
                == m.I[d, t - 1] + sum(m.F[n, d, t] for n in m.Nin[d]) - m.FD[d, t]
            )

    # Redundant flow calculation
    @m.Constraint(m.A, m.T)
    def ftil_calc(m, nin, nout, t):
        """Total arc flow equals the sum of source-tracked flows F_til over all sources r."""
        return m.F[nin, nout, t] == sum(m.F_til[r, nin, nout, t] for r in m.R)

    # Redundant inventory calculation
    @m.Constraint(m.B, m.T)
    def itil_calc(m, b, t):
        """Total blending tank inventory equals the sum of source-tracked inventories I_til."""
        return m.I[b, t] == sum(m.I_til[r, b, t] for r in m.R)

    # Flow activation disjunctions

    # Flow NB
    def build_nb_activation_flow_equations(disjunct, nin, nout, t):
        """Active disjunct for NB arc (nin -> nout): enforces arc flow bounds."""
        m = disjunct.model()

        # Flow bounds
        @disjunct.Constraint()
        def active_nb_flow_bound_L(disjunct):
            return m.F_bounds[nin, nout][0] <= m.F[nin, nout, t]

        @disjunct.Constraint()
        def active_nb_flow_bound_U(disjunct):
            return m.F[nin, nout, t] <= m.F_bounds[nin, nout][1]

    def build_nb_deactivation_flow_equations(disjunct, nin, nout, t):
        """Inactive disjunct for NB arc (nin -> nout): sets flow to zero."""
        m = disjunct.model()

        # Flow deactivation
        @disjunct.Constraint()
        def deactivate_nb_flow(disjunct):
            return m.F[nin, nout, t] == 0

    # Create disjunction
    m.X_nb = Disjunct(m.NB, m.T, rule=build_nb_activation_flow_equations)
    m.X_nb_not = Disjunct(m.NB, m.T, rule=build_nb_deactivation_flow_equations)

    @m.Disjunction(m.NB, m.T)
    def X_nb_is_active_or_not(m, nin, nout, t):
        """Disjunction for each NB arc: flow is either active or zero at time t."""
        return [m.X_nb[nin, nout, t], m.X_nb_not[nin, nout, t]]

    # Flow SD
    def build_sd_activation_flow_equations(disjunct, nin, nout, t):
        """Active disjunct for SD arc (nin -> nout): enforces flow bounds and composition specs."""
        m = disjunct.model()

        # Flow bounds
        @disjunct.Constraint()
        def active_sd_flow_bound_L(disjunct):
            return m.F_bounds[nin, nout][0] <= m.F[nin, nout, t]

        @disjunct.Constraint()
        def active_sd_flow_bound_U(disjunct):
            return m.F[nin, nout, t] <= m.F_bounds[nin, nout][1]

        # Specification check
        @disjunct.Constraint(m.Q)
        def active_sd_spec_bound_L(disjunct, q):
            if m.CD_bounds[q, nout][0] <= m.CIN[q, nin]:
                return pyo.Constraint.Feasible
            else:
                return m.F[nin, nout, t] == 0

        @disjunct.Constraint(m.Q)
        def active_sd_spec_bound_U(disjunct, q):
            if m.CIN[q, nin] <= m.CD_bounds[q, nout][1]:
                return pyo.Constraint.Feasible
            else:
                return m.F[nin, nout, t] == 0

    def build_sd_deactivation_flow_equations(disjunct, nin, nout, t):
        """Inactive disjunct for SD arc (nin -> nout): sets flow to zero."""
        m = disjunct.model()

        # Flow deactivation
        @disjunct.Constraint()
        def deactivate_sd_flow(disjunct):
            return m.F[nin, nout, t] == 0

    # Create disjunction
    m.X_sd = Disjunct(m.SD, m.T, rule=build_sd_activation_flow_equations)
    m.X_sd_not = Disjunct(m.SD, m.T, rule=build_sd_deactivation_flow_equations)

    @m.Disjunction(m.SD, m.T)
    def X_sd_is_active_or_not(m, nin, nout, t):
        """Disjunction for each SD arc: flow is either active (with spec checks) or zero at time t."""
        return [m.X_sd[nin, nout, t], m.X_sd_not[nin, nout, t]]

    # Flow BD
    def build_bd_activation_flow_equations(disjunct, nin, nout, t):
        """Active disjunct for BD arc (nin -> nout): enforces flow bounds and redundant bilinear bounds."""
        m = disjunct.model()

        # Flow bounds
        @disjunct.Constraint()
        def active_bd_flow_bound_L(disjunct):
            return m.F_bounds[nin, nout][0] <= m.F[nin, nout, t]

        @disjunct.Constraint()
        def active_bd_flow_bound_U(disjunct):
            return m.F[nin, nout, t] <= m.F_bounds[nin, nout][1]

        # Bilinear FC redundant bounds
        @disjunct.Constraint(m.Q)
        def active_rc_bilinear_fc_bound_L(disjunct, q):
            return m.CD_bounds[q, nout][0] * m.F[nin, nout, t] <= sum(
                m.F_til[r, nin, nout, t] * m.C0_hat[q, r] for r in m.R
            )

        @disjunct.Constraint(m.Q)
        def active_rc_bilinear_fc_bound_U(disjunct, q):
            return (
                sum(m.F_til[r, nin, nout, t] * m.C0_hat[q, r] for r in m.R)
                <= m.CD_bounds[q, nout][1] * m.F[nin, nout, t]
            )

        if t > 1:
            # Specification check
            @disjunct.Constraint(m.Q)
            def active_bd_spec_bound_L(disjunct, q):
                return m.CD_bounds[q, nout][0] <= m.C[q, nin, t - 1]

            @disjunct.Constraint(m.Q)
            def active_bd_spec_bound_U(disjunct, q):
                return m.C[q, nin, t - 1] <= m.CD_bounds[q, nout][1]

            # Bilinear IC redundant bounds
            @disjunct.Constraint(m.Q)
            def active_rc_bilinear_ic_bound_L(disjunct, q):
                return m.CD_bounds[q, nout][0] * m.I[nin, t - 1] <= sum(
                    m.I_til[r, nin, t - 1] * m.C0_hat[q, r] for r in m.R
                )

            @disjunct.Constraint(m.Q)
            def active_rc_bilinear_ic_bound_U(disjunct, q):
                return (
                    sum(m.I_til[r, nin, t - 1] * m.C0_hat[q, r] for r in m.R)
                    <= m.CD_bounds[q, nout][1] * m.I[nin, t - 1]
                )

    def build_bd_deactivation_flow_equations(disjunct, nin, nout, t):
        """Inactive disjunct for BD arc (nin -> nout): sets flow to zero."""
        m = disjunct.model()

        # Flow deactivation
        @disjunct.Constraint()
        def deactivate_bd_flow(disjunct):
            return m.F[nin, nout, t] == 0

    # Create disjunction
    m.X_bd = Disjunct(m.BD, m.T, rule=build_bd_activation_flow_equations)
    m.X_bd_not = Disjunct(m.BD, m.T, rule=build_bd_deactivation_flow_equations)

    @m.Disjunction(m.BD, m.T)
    def X_bd_is_active_or_not(m, nin, nout, t):
        """Disjunction for each BD arc: flow is either active (with redundant bounds) or zero at time t."""
        return [m.X_bd[nin, nout, t], m.X_bd_not[nin, nout, t]]

    # Tank mode disjunctions
    def build_YB_charging_equations(disjunct, b, t):
        """Charging disjunct for blending tank b at time t."""
        m = disjunct.model()

        # Inventory mass balance
        @disjunct.Constraint()
        def YB_charging_mass_balance(disjunct):
            if t == 1:
                return m.I[b, t] == m.I0[b] + sum(m.F[n, b, t] for n in m.Nin[b])
            else:
                return m.I[b, t] == m.I[b, t - 1] + sum(m.F[n, b, t] for n in m.Nin[b])

        # Inventory bilinear balance
        @disjunct.Constraint(m.Q)
        def YB_charging_bilinear_balance(disjunct, q):
            if t == 1:
                return m.I[b, t] * m.C[q, b, t] == m.I0[b] * m.C0[q, b] + sum(
                    m.F[s, b, t] * m.CIN[q, s] for s in m.S if (s, b) in m.A
                ) + sum(m.F[bb, b, t] * m.C0[q, bb] for bb in m.B if (bb, b) in m.A)
            else:
                return m.I[b, t] * m.C[q, b, t] == m.I[b, t - 1] * m.C[
                    q, b, t - 1
                ] + sum(m.F[s, b, t] * m.CIN[q, s] for s in m.S if (s, b) in m.A) + sum(
                    m.F[bb, b, t] * m.C[q, bb, t - 1] for bb in m.B if (bb, b) in m.A
                )

        # Redundant balance
        @disjunct.Constraint(m.R)
        def YB_charging_redundant_balance(disjunct, r):
            if t == 1:
                return m.I_til[r, b, t] == m.I0[b] + sum(
                    m.F_til[r, n, b, t] for n in m.Nin[b]
                )
            else:
                return m.I_til[r, b, t] == m.I_til[r, b, t - 1] + sum(
                    m.F_til[r, n, b, t] for n in m.Nin[b]
                )

    def build_YB_discharging_equations(disjunct, b, t):
        """Discharging disjunct for blending tank b at time t."""
        m = disjunct.model()

        # Inventory mass balance
        @disjunct.Constraint()
        def YB_discharging_mass_balance(disjunct):
            if t == 1:
                return m.I[b, t] == m.I0[b] - sum(m.F[b, n, t] for n in m.Nout[b])
            else:
                return m.I[b, t] == m.I[b, t - 1] - sum(m.F[b, n, t] for n in m.Nout[b])

        if t > 1:
            # Specification transition
            @disjunct.Constraint(m.Q)
            def YB_discharging_spec_trans(disjunct, q):
                return m.C[q, b, t] == m.C[q, b, t - 1]

        # Redundant balance
        @disjunct.Constraint(m.R)
        def YB_discharging_redundant_balance(disjunct, r):
            if t == 1:
                return m.I_til[r, b, t] == m.I0[b] - sum(
                    m.F_til[r, b, n, t] for n in m.Nout[b]
                )
            else:
                return m.I_til[r, b, t] == m.I_til[r, b, t - 1] - sum(
                    m.F_til[r, b, n, t] for n in m.Nout[b]
                )

    # Create disjunction
    m.YB = Disjunct(m.B, m.T, rule=build_YB_charging_equations)
    m.YB_not = Disjunct(m.B, m.T, rule=build_YB_discharging_equations)

    @m.Disjunction(m.B, m.T)
    def YB_is_charging_or_discharging(m, b, t):
        """Disjunction for each blending tank: charging or discharging at time t."""
        return [m.YB[b, t], m.YB_not[b, t]]

    # Logic implications
    @m.LogicalConstraint(m.NB, m.T)
    def charging_logic_implication(m, nin, nout, t):
        """If NB arc flow into blending tank nout is active, then nout must be charging at time t."""
        return m.X_nb[nin, nout, t].indicator_var.implies(m.YB[nout, t].indicator_var)

    @m.LogicalConstraint(m.BN, m.T)
    def discharging_logic_implication(m, nin, nout, t):
        """If any outflow arc from blending tank nin is active, then nin must be discharging at time t."""
        if nout in m.D:
            return m.X_bd[nin, nout, t].indicator_var.implies(
                m.YB_not[nin, t].indicator_var
            )
        elif nout in m.B:
            return m.X_nb[nin, nout, t].indicator_var.implies(
                m.YB_not[nin, t].indicator_var
            )
        else:
            return pyo.LogicalConstraint.Skip

    # F_til fixing
    @m.Constraint(m.R, m.A, m.T)
    def ftil_sn_fix(m, r, nin, nout, t):
        """For supply arcs, fix F_til[r, nin, nout, t] == F[nin, nout, t] when r == nin."""
        if nin in m.S and r == nin:
            return m.F_til[r, nin, nout, t] == m.F[nin, nout, t]
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.R, m.A, m.T)
    def ftil_bn_fix(m, r, nin, nout, t):
        """For blending-origin arcs, fix F_til[r, nin, nout, t] == F[nin, nout, t] when r == nin."""
        if nin in m.B and r == nin:
            return m.F_til[r, nin, nout, t] == m.F[nin, nout, t]
        else:
            return pyo.Constraint.Skip

    # OBJECTIVE
    @m.Objective(sense=pyo.maximize)
    def obj(m):
        """Maximize net profit: demand revenue - supply cost - variable arc costs - fixed activation costs."""
        return sum(
            sum(m.betaT_d[d] * m.F[n, d, t] for d in m.D for n in m.Nin[d])
            - sum(m.betaT_s[s] * m.F[s, n, t] for s in m.S for n in m.Nout[s])
            - sum(m.betaN[nin, nout] * m.F[nin, nout, t] for (nin, nout) in m.A)
            - (
                sum(
                    m.alphaN[nin, nout]
                    * m.X_nb[nin, nout, t].indicator_var.get_associated_binary()
                    for (nin, nout) in m.NB
                )
                + sum(
                    m.alphaN[nin, nout]
                    * m.X_sd[nin, nout, t].indicator_var.get_associated_binary()
                    for (nin, nout) in m.SD
                )
                + sum(
                    m.alphaN[nin, nout]
                    * m.X_bd[nin, nout, t].indicator_var.get_associated_binary()
                    for (nin, nout) in m.BD
                )
            )
            for t in m.T
        )  # m.alphaN[nin, nout] * m.X[nin, nout, t]

    return m


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Solve the multiperiod blending problem."
    )
    parser.add_argument(
        "--instance",
        default="instances_json/mpbp_6.json",
        help="Path to the JSON instance file (default: instances_json/mpbp_6.json)",
    )
    parser.add_argument(
        "--solver",
        default="gurobi",
        help="Name of the solver to use (default: gurobi)",
    )
    args = parser.parse_args()

    with open(args.instance, "r") as f:
        json_obj = json.load(f)
    d = convert_json_to_data(json_obj)

    m = build_model(d)
    pyo.TransformationFactory("core.logical_to_linear").apply_to(m)
    pyo.TransformationFactory("gdp.bigm").apply_to(m)

    opt = pyo.SolverFactory(args.solver)
    if not opt.available():
        raise RuntimeError(
            f"Solver '{args.solver}' is not available. "
            "Please install it or pass an available solver name via --solver."
        )
    status = opt.solve(m, tee=True)
    pyo.assert_optimal_termination(status)
