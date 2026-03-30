import ast
import json
import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction


def convert_json_to_data(obj) -> dict:
    """
    Recursively convert JSON-like structure back to original Python structure:
    - List values become tuples
    - String keys that represent tuples become tuple keys
    - Handles nested dictionaries and lists
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
    
def build_model(data):
    
    # PYOMO MODEL
    m = pyo.ConcreteModel()

    # SETS
    m.S = pyo.Set(initialize=data['S'])   # Set of supply tanks
    m.B = pyo.Set(initialize=data['B']) # Set of blending tanks
    m.D = pyo.Set(initialize=data['D']) # Set of demand tanks
    m.N = pyo.Set(initialize=data['N'])  # Set of tanks
    m.Q = pyo.Set(initialize=data['Q'])  # Set of substances
    m.T = pyo.Set(initialize=data['T'])  # Set of discrete time periods
    m.A = pyo.Set(initialize=data['A'])  # Set of existing arcs

    m.R = pyo.Set(initialize=data['R'])  # Set of sources
    m.B_hat = pyo.Set(initialize=data['B_hat']) # Set of blending tanks with initial composition

    # Special node sets
    m.NB = pyo.Set(initialize=data['NB'])
    m.BN = pyo.Set(initialize=data['BN'])
    m.SD = pyo.Set(initialize=data['SD'])
    m.BD = pyo.Set(initialize=data['BD'])

    # PARAMETERS
    # Arc dependencies
    m.Nin = pyo.Param(m.N, initialize=data['Nin'], within=pyo.Any)
    m.Nout = pyo.Param(m.N, initialize=data['Nout'], within=pyo.Any)

    # Initial inventories
    m.I0 = pyo.Param(m.N, initialize=data['I0'], within=pyo.NonNegativeReals)

    # Initial composition
    m.C0 = pyo.Param(m.Q, m.B, initialize=data['C0'], within=pyo.NonNegativeReals)
    m.C0_hat = pyo.Param(m.Q, m.R, initialize=data['C0_hat'], within=pyo.NonNegativeReals)

    # Inventory bounds
    m.I_bounds = pyo.Param(m.N, initialize=data['I_bounds'], within=pyo.Any)

    # Flow bounds
    m.F_bounds = pyo.Param(m.A, initialize=data['F_bounds'], within=pyo.Any)
    m.Fmax = pyo.Param(initialize=data['Fmax'], within=pyo.NonNegativeReals)

    # Demanded flow bounds
    m.FD_bounds = pyo.Param(m.D, m.T, initialize=data['FD_bounds'], within=pyo.Any)

    # Demanded composition bounds
    m.CD_bounds = pyo.Param(m.Q, m.D, initialize=data['CD_bounds'], within=pyo.Any)

    # Composition bounds
    m.C_bounds = pyo.Param(m.Q, initialize=data['C_bounds'], within=pyo.Any)

    # Supply conditions
    m.CIN = pyo.Param(m.Q, m.S, initialize=data['CIN'], within=pyo.NonNegativeReals)
    m.FIN = pyo.Param(m.S, m.T, initialize=data['FIN'], within=pyo.NonNegativeReals)

    # Economic parameters
    m.betaT_s = pyo.Param(m.S, initialize=data['betaT_s'], within=pyo.Reals)
    m.betaT_d = pyo.Param(m.D, initialize=data['betaT_d'], within=pyo.Reals)
    m.alphaN = pyo.Param(m.A, initialize=data['alphaN'], within=pyo.Reals)
    m.betaN = pyo.Param(m.A, initialize=data['betaN'], within=pyo.Reals)

    # CONTINUOUS VARIABLES
    m.F = pyo.Var(m.A, m.T, within=pyo.NonNegativeReals, bounds=(0, m.Fmax))
    m.FD = pyo.Var(m.D, m.T, within=pyo.NonNegativeReals, bounds=lambda _, d, t: m.FD_bounds[d,t])
    m.I = pyo.Var(m.N, m.T, within=pyo.NonNegativeReals, bounds=lambda _, n, t: m.I_bounds[n])
    m.C = pyo.Var(m.Q, m.B, m.T, within=pyo.NonNegativeReals, bounds=lambda _, q, b, t: m.C_bounds[q])

    m.F_til = pyo.Var(m.R, m.A, m.T, within=pyo.NonNegativeReals, bounds=(0, m.Fmax))
    m.I_til = pyo.Var(m.R, m.B, m.T, within=pyo.NonNegativeReals, bounds=lambda _, r, b, t: m.I_bounds[b])

    # CONSTRAINTS

    # Supply inventory balance
    @m.Constraint(m.S, m.T)
    def supply_bal(m, s, t):
        if t == 1:
            return m.I[s, t] == m.I0[s] + m.FIN[s, t] - sum(m.F[s, n, t] for n in m.Nout[s])
        else:
            return m.I[s, t] == m.I[s, t-1] + m.FIN[s, t] - sum(m.F[s, n, t] for n in m.Nout[s])

    # Demand inventory balance
    @m.Constraint(m.D, m.T)
    def demand_bal(m, d, t):
        if t == 1:
            return m.I[d, t] == m.I0[d] + sum(m.F[n, d, t] for n in m.Nin[d]) - m.FD[d, t]
        else:
            return m.I[d, t] == m.I[d, t-1] + sum(m.F[n, d, t] for n in m.Nin[d]) - m.FD[d, t]

    # Redundant flow calculation
    @m.Constraint(m.A, m.T)
    def ftil_calc(m, nin, nout, t):
        return m.F[nin, nout, t] == sum(m.F_til[r, nin, nout, t] for r in m.R)

    # Redundant inventory calculation
    @m.Constraint(m.B, m.T)
    def itil_calc(m, b, t):
        return m.I[b, t] == sum(m.I_til[r, b, t] for r in m.R)

    # Flow activation disjunctions

    # Flow NB
    def build_nb_activation_flow_equations(disjunct, nin, nout, t):
        m = disjunct.model()
            
        # Flow bounds
        @disjunct.Constraint()
        def active_nb_flow_bound_L(disjunct):
            return m.F_bounds[nin,nout][0] <= m.F[nin, nout, t]

        @disjunct.Constraint()
        def active_nb_flow_bound_U(disjunct):
            return m.F[nin, nout, t] <= m.F_bounds[nin,nout][1]

    def build_nb_deactivation_flow_equations(disjunct, nin, nout, t):
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
        return [m.X_nb[nin, nout, t], m.X_nb_not[nin, nout, t]]
        #return Disjunction.Skip

    # Flow SD
    def build_sd_activation_flow_equations(disjunct, nin, nout, t):
        m = disjunct.model()
            
        # Flow bounds
        @disjunct.Constraint()
        def active_sd_flow_bound_L(disjunct):
            return m.F_bounds[nin,nout][0] <= m.F[nin, nout, t]

        @disjunct.Constraint()
        def active_sd_flow_bound_U(disjunct):
            return m.F[nin, nout, t] <= m.F_bounds[nin,nout][1]

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
        return [m.X_sd[nin, nout, t], m.X_sd_not[nin, nout, t]]

    # Flow BD
    def build_bd_activation_flow_equations(disjunct, nin, nout, t):
        m = disjunct.model()
            
        # Flow bounds
        @disjunct.Constraint()
        def active_bd_flow_bound_L(disjunct):
            return m.F_bounds[nin,nout][0] <= m.F[nin, nout, t]

        @disjunct.Constraint()
        def active_bd_flow_bound_U(disjunct):
            return m.F[nin, nout, t] <= m.F_bounds[nin,nout][1]

        # Bilinear FC redundant bounds
        @disjunct.Constraint(m.Q)
        def active_rc_bilinear_fc_bound_L(disjunct, q):
            return m.CD_bounds[q, nout][0]*m.F[nin, nout, t] <= sum(m.F_til[r, nin, nout, t]*m.C0_hat[q,r] for r in m.R)

        @disjunct.Constraint(m.Q)
        def active_rc_bilinear_fc_bound_U(disjunct, q):
            return sum(m.F_til[r, nin, nout, t]*m.C0_hat[q,r] for r in m.R) <=  m.CD_bounds[q, nout][1]*m.F[nin, nout, t]

        if t > 1:
            # Specification check
            @disjunct.Constraint(m.Q)
            def active_bd_spec_bound_L(disjunct, q):
                return m.CD_bounds[q, nout][0] <= m.C[q, nin, t-1]

            @disjunct.Constraint(m.Q)
            def active_bd_spec_bound_U(disjunct, q):
                return m.C[q, nin, t-1] <= m.CD_bounds[q, nout][1]

            # Bilinear IC redundant bounds
            @disjunct.Constraint(m.Q)
            def active_rc_bilinear_ic_bound_L(disjunct, q):
                return m.CD_bounds[q, nout][0]*m.I[nin, t-1] <= sum(m.I_til[r, nin, t-1]*m.C0_hat[q,r] for r in m.R)

            @disjunct.Constraint(m.Q)
            def active_rc_bilinear_ic_bound_U(disjunct, q):
                return sum(m.I_til[r, nin, t-1]*m.C0_hat[q,r] for r in m.R) <= m.CD_bounds[q, nout][1]*m.I[nin, t-1] 

    def build_bd_deactivation_flow_equations(disjunct, nin, nout, t):
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
        return [m.X_bd[nin, nout, t], m.X_bd_not[nin, nout, t]]

    # Tank mode disjunctions 
    def build_YB_charging_equations(disjunct, b, t):
        m = disjunct.model()
        
        # Inventory mass balance
        @disjunct.Constraint()
        def YB_charging_mass_balance(disjunct):
            if t == 1:
                return m.I[b, t] == m.I0[b] + sum(m.F[n, b, t] for n in m.Nin[b])
            else:
                return m.I[b, t] == m.I[b, t-1] + sum(m.F[n, b, t] for n in m.Nin[b])

        # Inventory bilinear balance
        @disjunct.Constraint(m.Q)
        def YB_charging_bilinear_balance(disjunct, q):
            if t == 1:
                return m.I[b, t]*m.C[q, b, t] == m.I0[b]*m.C0[q, b] + sum(m.F[s, b, t]*m.CIN[q, s] for s in m.S if (s,b) in m.A) +  sum(m.F[bb, b, t]*m.C0[q, bb] for bb in m.B if (bb,b) in m.A)
            else:
                return m.I[b, t]*m.C[q, b, t] == m.I[b, t-1]*m.C[q, b, t-1] + sum(m.F[s, b, t]*m.CIN[q, s] for s in m.S if (s,b) in m.A) +  sum(m.F[bb, b, t]*m.C[q, bb, t-1] for bb in m.B if (bb,b) in m.A)

        
        # Redundant balance
        @disjunct.Constraint(m.R)
        def YB_charging_redundant_balance(disjunct, r):
            if t == 1:
                return m.I_til[r, b, t] == m.I0[b] + sum(m.F_til[r, n, b, t] for n in m.Nin[b])
            else:
                return m.I_til[r, b, t] == m.I_til[r, b, t-1] + sum(m.F_til[r, n, b, t] for n in m.Nin[b])


    def build_YB_discharging_equations(disjunct, b, t):
        m = disjunct.model()

        # Inventory mass balance
        @disjunct.Constraint()
        def YB_discharging_mass_balance(disjunct):
            if t == 1:
                return m.I[b, t] == m.I0[b] - sum(m.F[b, n, t] for n in m.Nout[b])
            else:
                return m.I[b, t] == m.I[b, t-1] - sum(m.F[b, n, t] for n in m.Nout[b])

        if t > 1:
            # Specification transition
            @disjunct.Constraint(m.Q)
            def YB_discharging_spec_trans(disjunct, q):
                return m.C[q, b, t] == m.C[q, b, t-1] 

        # Redundant balance
        @disjunct.Constraint(m.R)
        def YB_discharging_redundant_balance(disjunct, r):
            if t == 1:
                return m.I_til[r, b, t] == m.I0[b] - sum(m.F_til[r, b, n, t] for n in m.Nout[b])
            else:
                return m.I_til[r, b, t] == m.I_til[r, b, t-1] - sum(m.F_til[r, b, n, t] for n in m.Nout[b])

    # Create disjunction 
    m.YB = Disjunct(m.B, m.T, rule= build_YB_charging_equations)
    m.YB_not = Disjunct(m.B, m.T, rule= build_YB_discharging_equations)

    @m.Disjunction(m.B, m.T)
    def YB_is_charging_or_discharging(m, b, t):
        return [m.YB[b, t], m.YB_not[b, t]]

    # Logic implications
    @m.LogicalConstraint(m.NB, m.T)
    def charging_logic_implication(m, nin, nout, t):
        return m.X_nb[nin, nout, t].indicator_var.implies(m.YB[nout, t].indicator_var)

    @m.LogicalConstraint(m.BN, m.T)
    def discharging_logic_implication(m, nin, nout, t):
        if nout in m.D:
            return m.X_bd[nin, nout, t].indicator_var.implies(m.YB_not[nin, t].indicator_var)
        elif nout in m.B:
            return m.X_nb[nin, nout, t].indicator_var.implies(m.YB_not[nin, t].indicator_var)  # TODO Not in formulation
        else:
            return pyo.LogicalConstraint.Skip

    # F_til fixing
    @m.Constraint(m.R, m.A, m.T)
    def ftil_sn_fix(m, r, nin, nout, t):
        if nin in m.S and r == nin:
            return m.F_til[r, nin, nout, t] == m.F[nin, nout, t]
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.R, m.A, m.T) # TODO: Is it forall t?
    def ftil_bn_fix(m, r, nin, nout, t):
        if nin in m.B and r == nin:
            return m.F_til[r, nin, nout, t] == m.F[nin, nout, t]
        else:
            return pyo.Constraint.Skip

    # OBJECTIVE
    @m.Objective(sense=pyo.maximize)
    def obj(m):
        return sum(sum(m.betaT_d[d] * m.F[n, d, t] for d in m.D for n in m.Nin[d]) - sum(m.betaT_s[s] * m.F[s, n, t] for s in m.S for n in m.Nout[s]) - sum(m.betaN[nin, nout] * m.F[nin, nout,  t] for (nin, nout) in m.A) - (sum(m.alphaN[nin, nout] * m.X_nb[nin, nout, t].indicator_var.get_associated_binary() for (nin, nout) in m.NB) + sum(m.alphaN[nin, nout] * m.X_sd[nin, nout, t].indicator_var.get_associated_binary() for (nin, nout) in m.SD) + sum(m.alphaN[nin, nout] * m.X_bd[nin, nout, t].indicator_var.get_associated_binary() for (nin, nout) in m.BD)) for t in m.T) # m.alphaN[nin, nout] * m.X[nin, nout, t]
   
    return m 


if __name__ == "__main__":
    # Opening instance
    with open('instances_json/mpbp_6.json', 'r') as f:
        json_obj = json.load(f)
    d = convert_json_to_data(json_obj)

    m = build_model(d)    # building model
    pyo.TransformationFactory('core.logical_to_linear').apply_to(m)
    pyo.TransformationFactory('gdp.bigm').apply_to(m)
    
    # Solving with gurobi. If gurobi unavailable - can use any MIQCP/MINLP solver of choice
    opt = pyo.SolverFactory('gurobi')
    status = opt.solve(m, tee=True)
    pyo.assert_optimal_termination(status)
