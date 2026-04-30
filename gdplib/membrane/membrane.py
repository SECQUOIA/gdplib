"""
membrane.py

GDP superstructure model for optimal design of a multistage nanofiltration membrane
cascade for Lithium-Cobalt separation from battery recycling streams.

The model decides the optimal positions for feed injection, diafiltrate injection, and
inter-stage reflux streams across K cascade stages, each discretized into N membrane
elements. The objective is to maximize Cobalt recovery subject to a minimum Lithium
recovery constraint.

References
----------
[1] Ovalle, D., Tran, N., Laird, C. D., & Grossmann, I. E. (2024). Optimal Membrane Cascade
    Design for Critical Mineral Recovery Through Logic-based Superstructure Optimization.
    Systems and Control Transactions, 3, 853-859. https://doi.org/10.69997/sct.127917
"""

from pyomo.environ import ConcreteModel, Constraint, Var, Block, Set, Param, Objective, SolverFactory, TransformationFactory, maximize, BooleanVar, LogicalConstraint, exactly
from pyomo.gdp import Disjunct, Disjunction

def create_membrane_element():
    #                                   ---------------
    #                                   |             |
    # feed + diaf + refl-->             |             |
    #                       --> rin --> |             |  --> rout
    #               fin -->             |             |
    #                                   |             |
    #                                   ---------------
    #                                   |             |
    #                       --> pin --> |             |  --> pout
    #                                   |             |
    #                                   ---------------

    m = ConcreteModel()
    m.STREAMS = ['fin', 'feed', 'diaf', 'refl', 'rin', 'rout', 'pin', 'pout'] # f-feed, s-side, r-retentate, p-permeate]
    m.COMPONENTS = ['Co', 'Li']

    # key variables
    m.F = Var(m.STREAMS, initialize=10, bounds=(0,130)) # volumetric flow        
    m.x = Var(m.STREAMS, m.COMPONENTS, initialize=10, bounds=(0,130)) # concentration        
    m.J = Var(initialize=1) # solvent flux (usually specified with .fix)
    m.L = Var(bounds=(0,1000), initialize=100) # length of the membrane (usually specified with .fix)         
    m.S = Var(m.COMPONENTS, initialize=1) # seiving coefficient (usually specified with .fix)
    m.w = Var(initialize=1) 

    # inlet mixer balances
    @m.Constraint()
    def inlet_mixer_overall_balance(m):
        return m.F['fin'] + m.F['feed'] + m.F['diaf'] + m.F['refl'] - m.F['rin'] == 0

    @m.Constraint(m.COMPONENTS)
    def inlet_mixer_comp_balance(m,c):
        return m.F['fin']*m.x['fin',c] + m.F['feed']*m.x['feed',c] + m.F['diaf']*m.x['diaf',c] + m.F['refl']*m.x['refl',c] - m.F['rin']*m.x['rin',c] == 0

    # membrane balances
    @m.Constraint()
    def membrane_overall_balance(m):
        return m.F['rin'] + m.F['pin'] - m.F['rout'] - m.F['pout'] == 0

    @m.Constraint(m.COMPONENTS)
    def membrane_component_balances(m,c):
        return m.F['rin']*m.x['rin',c] + m.F['pin']*m.x['pin',c] - m.F['rout']*m.x['rout',c] - m.F['pout']*m.x['pout',c] == 0

    @m.Constraint()
    def permeate_overall_balance(m):
        return m.F['pin'] + m.J*m.L*m.w - m.F['pout'] == 0

    # membrane seiving coef constraints
    @m.Constraint(m.COMPONENTS)
    def seiving(m,c):
        return m.x['rout',c] == (m.F['rout']/m.F['rin'])**(m.S[c]-1) *  m.x['rin',c]

    return m

def create_discretized_membrane(N=10):
    m = ConcreteModel()
    m.L = Var(bounds=(0,None), initialize=1.0)
    m.N_length = N
    m.N = Set(initialize=range(m.N_length), ordered=True)

    @m.Block(m.N)
    def elements(b, d):
        e = create_membrane_element()
        e.J.fix(0.10000)
        e.w.fix(1.5)
        e.S['Li'].fix(1.3000)
        e.S['Co'].fix(0.50000)
        return e

    m.COMPONENTS = m.elements[0].COMPONENTS

    @m.Constraint(m.N)
    def link_length(m, d):
        return m.elements[d].L == 1.0/N * m.L
    
    @m.Constraint(m.N)
    def link_retentate_flow(m, d):
        if d == m.N.first():
            return Constraint.Skip
        return m.elements[m.N.prev(d)].F['rout'] == m.elements[d].F['fin']

    @m.Constraint(m.N, m.COMPONENTS)
    def link_retentate_conc(m, d, c):
        if d == m.N.first():
            return Constraint.Skip
        return m.elements[m.N.prev(d)].x['rout', c] == m.elements[d].x['fin', c]

    @m.Constraint(m.N)
    def link_permeate_flow(m, d):
        if d == m.N.first():
            return Constraint.Skip
        return m.elements[m.N.prev(d)].F['pout'] == m.elements[d].F['pin']

    @m.Constraint(m.N, m.COMPONENTS)
    def link_permeate_conc(m, d, c):
        if d == m.N.first():
            return Constraint.Skip
        return m.elements[m.N.prev(d)].x['pout', c] == m.elements[d].x['pin', c]
    
    # Define global in and outs of the membrane
    m.COMPONENTS = m.elements[m.N.first()].COMPONENTS
    m.STREAMS = ['fin', 'rout', 'pin', 'pout']
    m.F = Var(m.STREAMS, initialize=10, bounds=(0,None))
    m.x = Var(m.STREAMS, m.COMPONENTS, initialize=10, bounds=(0,None))

    @m.Constraint()
    def feed_retentate_flow(m):
        return m.F['fin'] == m.elements[m.N.first()].F['fin']

    @m.Constraint(m.COMPONENTS)
    def feed_retentate_concentration(m,c):
        return m.x['fin', c] == m.elements[m.N.first()].x['fin',c]

    @m.Constraint()
    def feed_permeate_flow(m):
        return m.F['pin'] == m.elements[m.N.first()].F['pin']

    @m.Constraint(m.COMPONENTS)
    def feed_permeate_concentration(m,c):
        return m.x['pin', c] == m.elements[m.N.first()].x['pin',c]
    
    @m.Constraint()
    def outlet_retentate_flow(m):
        return m.F['rout'] == m.elements[m.N.last()].F['rout']

    @m.Constraint(m.COMPONENTS)
    def outlet_retentate_concentration(m,c):
        return m.x['rout', c] == m.elements[m.N.last()].x['rout',c]

    @m.Constraint()
    def outlet_permeate_flow(m):
        return m.F['pout'] == m.elements[m.N.last()].F['pout']

    @m.Constraint(m.COMPONENTS)
    def outlet_permeate_concentration(m,c):
        return m.x['pout', c] == m.elements[m.N.last()].x['pout',c]
    
    return m

def create_discretized_stages(K=3, N=10, isotropic=True):
    m = ConcreteModel()
    m.K = Set(initialize=range(K), ordered=True)
    m.N_length = N
    m.N = Set(initialize=range(m.N_length), ordered=True)
    m.case = 'isotropic' if isotropic else 'anisotropic'

    @m.Block(m.K)
    def stages(b, k):
        m = b.parent_block()
        s = create_discretized_membrane(m.N_length)
        return s
    
    # Define global in and outs of the membrane
    m.COMPONENTS = m.stages[m.K.first()].COMPONENTS
    m.STREAMS = ['pout']
    m.F = Var(m.STREAMS, initialize=10, bounds=(0,None))
    m.x = Var(m.STREAMS, m.COMPONENTS, initialize=10, bounds=(0,None))
    
    @m.Constraint(m.K)
    def link_pout_with_rin_F(m, k):
        if k == m.K.first():
            return Constraint.Skip
        return m.stages[m.K.prev(k)].F['pout'] == m.stages[k].F['fin']
    
    @m.Constraint(m.K, m.COMPONENTS)
    def link_pout_with_rin_x(m, k, c):
        if k == m.K.first():
            return Constraint.Skip
        return m.stages[m.K.prev(k)].x['pout',c] == m.stages[k].x['fin', c]

    # Permeate never comes into a stage
    for k in m.K:
        m.stages[k].F['pin'].fix(0.0)
        m.stages[k].x['pin','Li'].fix(0.0)
        m.stages[k].x['pin','Co'].fix(0.0)

    # No feed coming into the first stage
    m.stages[m.K.first()].elements[m.N.first()].F['fin'].fix(0.0)
    m.stages[m.K.first()].elements[m.N.first()].x['fin','Li'].fix(0.0)
    m.stages[m.K.first()].elements[m.N.first()].x['fin','Co'].fix(0.0)

    # No recyle coming into the last stage
    for n in m.N:
        m.stages[m.K.last()].elements[n].F['refl'].fix(0.0)
        m.stages[m.K.last()].elements[n].x['refl','Li'].fix(0.0)
        m.stages[m.K.last()].elements[n].x['refl','Co'].fix(0.0)
    
    @m.Constraint(m.K)
    def isotropic_cascade(m, k):
        if k == m.K.last():
            return Constraint.Skip
        if m.case == 'isotropic':
            return m.stages[k].L - m.stages[m.K.next(k)].L == 0
        else:
            return m.stages[k].L - m.stages[m.K.next(k)].L >= 0

    @m.Constraint()
    def last_pout_F(m):
        return m.stages[m.K.last()].F['pout'] == m.F['pout']
    
    @m.Constraint()
    def last_pout_x(m):
        return m.stages[m.K.last()].x['pout','Li'] == m.x['pout','Li']
    
    return m

def build_model(K=3, N=10, isotropic=True):
    """
    Build the GDP superstructure model for a multistage nanofiltration membrane cascade.

    The model decides the optimal positions for feed injection, diafiltrate injection,
    and inter-stage reflux streams in a K-stage cascade to maximize Cobalt recovery
    while meeting a minimum Lithium recovery constraint. Each stage is discretized into
    N membrane elements linked in series on both the retentate and permeate sides.

    Disjunctions select whether feed, diafiltrate, or reflux is installed at each
    (stage, element) location. Logical constraints enforce that exactly one feed and
    one diafiltrate are placed across the entire cascade, and exactly one reflux per
    intermediate stage.

    Parameters
    ----------
    K : int, optional
        Number of cascade stages. Default is 3.
    N : int, optional
        Number of discretization elements per stage. Default is 10.
    isotropic : bool, optional
        If True, all stages are constrained to equal membrane length (isotropic cascade).
        If False, stages may have decreasing lengths (anisotropic cascade). Default is True.

    Returns
    -------
    m : pyomo.ConcreteModel
        Pyomo GDP model of the membrane cascade superstructure, ready for GDP
        transformation and solving.

    References
    ----------
    [1] Ovalle, D., Tran, N., Laird, C. D., & Grossmann, I. E. (2024). Optimal Membrane
        Cascade Design for Critical Mineral Recovery Through Logic-based Superstructure
        Optimization. Systems and Control Transactions, 3, 853-859.
        https://doi.org/10.69997/sct.127917
    """
    m = create_discretized_stages(K, N, isotropic=isotropic)
    
    m.F_feed = Param(initialize=100.0)
    m.x_feed = Param(m.COMPONENTS, initialize={'Li':1.7, 'Co':17.0})
    m.F_diaf = Param(initialize=30.0)
    m.x_diaf = Param(m.COMPONENTS, initialize={'Li':0.1, 'Co':0.2})

    # Feed position
    def install_feed(disjunct, k, n):
        m = disjunct.model()

        @disjunct.Constraint()
        def install_feed_flow(disjunct):
            return m.stages[k].elements[n].F['feed'] == m.F_feed
       
        @disjunct.Constraint(m.COMPONENTS)
        def install_feed_comp(disjunct, c):
            return m.stages[k].elements[n].x['feed', c] == m.x_feed[c]
        
    def not_install_feed(disjunct, k, n):
        m = disjunct.model()

        @disjunct.Constraint()
        def not_install_feed_flow(disjunct):
            return m.stages[k].elements[n].F['feed'] == 0
       
        @disjunct.Constraint(m.COMPONENTS)
        def not_install_feed_comp(disjunct, c):
            return m.stages[k].elements[n].x['feed', c] == 0
        
    m.install_feed_disjunct = Disjunct(m.K, m.N, rule=install_feed)
    m.not_install_feed_disjunct = Disjunct(m.K, m.N, rule=not_install_feed)

    @m.Disjunction(m.K, m.N)
    def install_feed(m, k, n):
        return [m.install_feed_disjunct[k, n], m.not_install_feed_disjunct[k,n]]
    
    # Diafiltrate position
    def install_diafiltrate(disjunct, k, n):
        m = disjunct.model()

        @disjunct.Constraint()
        def install_diafiltrate_flow(disjunct):
            return m.stages[k].elements[n].F['diaf'] == m.F_diaf
       
        @disjunct.Constraint(m.COMPONENTS)
        def install_diafiltrate_comp(disjunct, c):
            return m.stages[k].elements[n].x['diaf', c] == m.x_diaf[c]
        
    def not_install_diafiltrate(disjunct, k, n):
        m = disjunct.model()

        @disjunct.Constraint()
        def not_install_diafiltrate_flow(disjunct):
            return m.stages[k].elements[n].F['diaf'] == 0
       
        @disjunct.Constraint(m.COMPONENTS)
        def not_install_diafiltrate_comp(disjunct, c):
            return m.stages[k].elements[n].x['diaf', c] == 0
        
    m.install_diafiltrate_disjunct = Disjunct(m.K, m.N, rule=install_diafiltrate)
    m.not_install_diafiltrate_disjunct = Disjunct(m.K, m.N, rule=not_install_diafiltrate)

    @m.Disjunction(m.K, m.N)
    def install_diafiltrate(m, k, n):
        return [m.install_diafiltrate_disjunct[k, n], m.not_install_diafiltrate_disjunct[k,n]]
    
    # Reflux position
    K_minus_last = list(m.K)
    K_minus_last.pop(-1)
    
    def install_reflux(disjunct, k, n):
        m = disjunct.model()

        @disjunct.Constraint()
        def install_reflux_flow(disjunct):
            return m.stages[k].elements[n].F['refl'] == m.stages[m.K.next(k)].elements[m.N.last()].F['rout']
       
        @disjunct.Constraint(m.COMPONENTS)
        def install_reflux_comp(disjunct, c):
            return m.stages[k].elements[n].x['refl', c] == m.stages[m.K.next(k)].elements[m.N.last()].x['rout',c]
        
    def not_install_reflux(disjunct, k, n):
        m = disjunct.model()

        @disjunct.Constraint()
        def not_install_reflux_flow(disjunct):
            return m.stages[k].elements[n].F['refl'] == 0
       
        @disjunct.Constraint(m.COMPONENTS)
        def not_install_reflux_comp(disjunct, c):
            return m.stages[k].elements[n].x['refl', c] == 0
        
        
    m.install_reflux_disjunct = Disjunct(K_minus_last, m.N, rule=install_reflux)
    m.not_install_reflux_disjunct = Disjunct(K_minus_last, m.N, rule=not_install_reflux)

    @m.Disjunction(K_minus_last, m.N)
    def install_reflux(m, k, n):
        return [m.install_reflux_disjunct[k, n], m.not_install_reflux_disjunct[k,n]]
    

    @m.LogicalConstraint()
    def one_feed(m):
        return exactly(1, [m.install_feed_disjunct[k, n].indicator_var for k in m.K for n in m.N])
    
    @m.LogicalConstraint()
    def one_diafiltrate(m):
        return exactly(1, [m.install_diafiltrate_disjunct[k, n].indicator_var for k in m.K for n in m.N])
    
    @m.LogicalConstraint(K_minus_last)
    def one_reflux_per_stage(m, k):
        return exactly(1, [m.install_reflux_disjunct[k, n].indicator_var for n in m.N])
    

    #___________________________________________________________________________________________
    # Optimize for Lithium recovery
    m.R_Li = Param(initialize=0.90, mutable=True)
    
    # Li recovery for one membrane
    @m.Constraint()
    def Li_recovery(m):
        return m.stages[m.K.last()].elements[m.N.last()].F['pout']*m.stages[m.K.last()].elements[m.N.last()].x['pout', 'Li'] >= m.R_Li * (m.F_feed * m.x_feed['Li'] + m.F_diaf * m.x_diaf['Li'])
     
    m.obj = Objective( expr = m.stages[m.K.first()].elements[m.N.last()].F['rout']*m.stages[m.K.first()].elements[m.N.last()].x['rout','Co'], sense=maximize)
    return m


if __name__ == '__main__':
    m = build_model(K=3, N=3, isotropic=True)

    TransformationFactory('core.logical_to_linear').apply_to(m)
    transformation_string = 'gdp.' + 'bigm'
    TransformationFactory(transformation_string).apply_to(m)


    opt = SolverFactory('scip')
    results = opt.solve(m, tee=True)

    print(m.stages[m.K.first()].L())
    print(m.obj())

    print(results)