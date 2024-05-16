"""Distillation column model for 2018 PSE conference"""

from __future__ import division

from pyomo.environ import (
    Block, ConcreteModel, Constraint, log, minimize, NonNegativeReals, Objective, RangeSet, Set, Var, )
from pyomo.gdp import Disjunct, Disjunction


def build_column(min_trays, max_trays, xD, xB):
    """
    Build a Pyomo model of a distillation column for separation of benzene and toluene.

    Parameters
    ----------
    min_trays : int
        Minimum number of trays in the column
    max_trays : int
        Maximum number of trays in the column
    xD : float
        Distillate(benzene) purity
    xB : float
        Bottoms(toluene) purity

    Returns
    -------
    Pyomo.ConcreteModel
        A Pyomo model of the distillation column for separation of benzene and toluene.
    """
    m = ConcreteModel('benzene-toluene column')
    m.comps = Set(initialize=['benzene', 'toluene'], doc='Set of components')
    min_T, max_T = 300, 400 # Define temperature bounds [K]
    max_flow = 500 # maximum flow rate [mol/s]
    m.T_feed = Var(
        doc='Feed temperature [K]', domain=NonNegativeReals,
        bounds=(min_T, max_T), initialize=368)
    m.feed_vap_frac = Var(
        doc='Vapor fraction of feed',
        initialize=0, bounds=(0, 1))
    m.feed = Var(
        m.comps, doc='Total component feed flow [mol/s]', initialize=50)

    m.condens_tray = max_trays
    m.feed_tray = int(round(max_trays / 2))
    m.reboil_tray = 1
    m.distillate_purity = xD
    m.bottoms_purity = xB

    m.trays = RangeSet(max_trays, doc='Set of potential trays')
    m.conditional_trays = Set(
        initialize=m.trays - [m.condens_tray, m.feed_tray, m.reboil_tray],
        doc="Trays that may be turned on and off.")
    m.tray = Disjunct(m.conditional_trays, doc='Disjunct for tray existence')
    m.no_tray = Disjunct(m.conditional_trays, doc='Disjunct for tray absence')

    @m.Disjunction(m.conditional_trays, doc='Tray exists or does not')
    def tray_no_tray(b, t):
        """
        Disjunction for tray existence or absence.

        Parameters
        ----------
        b : Pyomo.Disjunct
            Pyomo disjunct representing the existence or absence of a tray in the distillation column model.
        t : int
            Index of tray in the distillation column model. Tray numbering ascends from the reboiler at the bottom (tray 1) to the condenser at the top (tray max_trays)

        Returns
        -------
        List of Disjuncts
            List of disjuncts representing the existence or absence of a tray in the distillation column model.
        """
        return [b.tray[t], b.no_tray[t]]
    m.minimum_num_trays = Constraint(
        expr=sum(m.tray[t].binary_indicator_var
                 for t in m.conditional_trays) + 1  # for feed tray
        >= min_trays, doc='Minimum number of trays')

    m.x = Var(m.comps, m.trays, doc='Liquid mole fraction',
              bounds=(0, 1), domain=NonNegativeReals, initialize=0.5)
    m.y = Var(m.comps, m.trays, doc='Vapor mole fraction',
              bounds=(0, 1), domain=NonNegativeReals, initialize=0.5)
    m.L = Var(m.comps, m.trays,
              doc='component liquid flows from tray in mol/s',
              domain=NonNegativeReals, bounds=(0, max_flow),
              initialize=50)
    m.V = Var(m.comps, m.trays,
              doc='component vapor flows from tray in mol/s',
              domain=NonNegativeReals, bounds=(0, max_flow),
              initialize=50)
    m.liq = Var(m.trays, domain=NonNegativeReals,
                doc='liquid flows from tray in mol/s', initialize=100,
                bounds=(0, max_flow))
    m.vap = Var(m.trays, domain=NonNegativeReals,
                doc='vapor flows from tray in mol/s', initialize=100,
                bounds=(0, max_flow))
    m.B = Var(m.comps, domain=NonNegativeReals,
              doc='bottoms component flows in mol/s',
              bounds=(0, max_flow), initialize=50)
    m.D = Var(m.comps, domain=NonNegativeReals,
              doc='distillate component flows in mol/s',
              bounds=(0, max_flow), initialize=50)
    m.bot = Var(domain=NonNegativeReals, initialize=50, bounds=(0, 100),
                doc='bottoms flow in mol/s')
    m.dis = Var(domain=NonNegativeReals, initialize=50,
                doc='distillate flow in mol/s', bounds=(0, 100))
    m.reflux_ratio = Var(domain=NonNegativeReals, bounds=(0.5, 4),
                         doc='reflux ratio', initialize=0.8329)
    m.reboil_ratio = Var(domain=NonNegativeReals, bounds=(0.5, 4),
                         doc='reboil ratio', initialize=0.9527)
    m.reflux_frac = Var(domain=NonNegativeReals, bounds=(0, 1 - 1E-6),
                        doc='reflux fractions')
    m.boilup_frac = Var(domain=NonNegativeReals, bounds=(0, 1 - 1E-6),
                        doc='boilup fraction')

    m.partial_cond = Disjunct()
    m.total_cond = Disjunct()
    m.condenser_choice = Disjunction(expr=[m.partial_cond, m.total_cond])

    for t in m.conditional_trays:
        _build_conditional_tray_mass_balance(m, t, m.tray[t], m.no_tray[t])
    _build_feed_tray_mass_balance(m)
    _build_condenser_mass_balance(m)
    _build_reboiler_mass_balance(m)

    @m.Constraint(m.comps,
                  doc="Bottoms flow is equal to liquid leaving reboiler.")
    def bottoms_mass_balance(m, c):
        """
        Constraint that the bottoms flow is equal to the liquid leaving the reboiler.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo model of the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the bottoms flow is equal to the liquid leaving the reboiler.
        """
        return m.B[c] == m.L[c, m.reboil_tray]

    @m.Constraint()
    def boilup_frac_defn(m):
        """
        Boilup fraction is the ratio between the bottoms flow and the liquid leaving the reboiler.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo model of the distillation column.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the boilup fraction is the ratio between the bottoms flow and the liquid leaving the reboiler.
        """
        return m.bot == (1 - m.boilup_frac) * m.liq[m.reboil_tray + 1]

    @m.Constraint()
    def reflux_frac_defn(m):
        """
        Reflux fraction is the ratio between the distillate flow and the difference in vapor flow in the condenser tray.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo model of the distillation column.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the reflux fraction is the ratio between the distillate flow and the difference in vapor flow in the condenser tray.
        """
        return m.dis == (1 - m.reflux_frac) * (
            m.vap[m.condens_tray - 1] - m.vap[m.condens_tray])

    @m.Constraint(m.trays)
    def liquid_sum(m, t):
        """
        Total liquid flow on each tray is the sum of all component liquid flows on the tray.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo model of the distillation column.
        t : int
            Index of tray in the distillation column model.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the total liquid flow on each tray is the sum of all component liquid flows on the tray.
        """
        return sum(m.L[c, t] for c in m.comps) == m.liq[t]

    @m.Constraint(m.trays)
    def vapor_sum(m, t):
        """
        Total vapor flow on each tray is the sum of all component vapor flows on the tray.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo model of the distillation column.
        t : int
            Index of tray in the distillation column model.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the total vapor flow on each tray is the sum of all component vapor flows on the tray.
        """
        return sum(m.V[c, t] for c in m.comps) == m.vap[t]

    m.bottoms_sum = Constraint(
        expr=sum(m.B[c] for c in m.comps) == m.bot, doc="Total bottoms flow is the sum of all component flows at the bottom.")
    m.distil_sum = Constraint(
        expr=sum(m.D[c] for c in m.comps) == m.dis, doc="Total distillate flow is the sum of all component flows at the top.")

    """Phase Equilibrium relations"""
    m.Kc = Var(
        m.comps, m.trays, doc='Phase equilibrium constant',
        domain=NonNegativeReals, initialize=1, bounds=(0, 1000))
    m.T = Var(m.trays, doc='Temperature [K]',
              domain=NonNegativeReals,
              bounds=(min_T, max_T))

    @m.Constraint(m.trays)
    def monotonoic_temperature(_, t):
        """
        Temperature of tray t is greater than or equal to temperature of tray t+1. The temperature decreases as the trays ascend.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            An unused placeholder parameter required by Pyomo's constraint interface, representing each potential tray in the distillation column where the temperature constraint is applied.
        t : int
            Index of tray in the distillation column model.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the temperature of tray t is greater than or equal to temperature of tray t+1. If t is the condenser tray, the constraint is skipped.
        """
        return m.T[t] >= m.T[t + 1] if t < max_trays else Constraint.Skip

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

    for t in m.conditional_trays:
        _build_tray_phase_equilibrium(m, t, m.tray[t])
    m.feed_tray_phase_eq = Block()
    m.reboiler_phase_eq = Block()
    m.condenser_phase_eq = Block()
    _build_tray_phase_equilibrium(m, m.feed_tray, m.feed_tray_phase_eq)
    _build_tray_phase_equilibrium(m, m.reboil_tray, m.reboiler_phase_eq)
    _build_tray_phase_equilibrium(m, m.condens_tray, m.condenser_phase_eq)

    m.H_L = Var(
        m.comps, m.trays, bounds=(0.1, 16),
        doc='Liquid molar enthalpy of component in tray (kJ/mol)')
    m.H_V = Var(
        m.comps, m.trays, bounds=(30, 16 + 40),
        doc='Vapor molar enthalpy of component in tray (kJ/mol)')
    m.H_L_spec_feed = Var(
        m.comps, doc='Component liquid molar enthalpy in feed [kJ/mol]',
        initialize=0, bounds=(0.1, 16))
    m.H_V_spec_feed = Var(
        m.comps, doc='Component vapor molar enthalpy in feed [kJ/mol]',
        initialize=0, bounds=(30, 16 + 40))
    m.Qb = Var(domain=NonNegativeReals, doc='reboiler duty (MJ/s)',
               initialize=1, bounds=(0, 8))
    m.Qc = Var(domain=NonNegativeReals, doc='condenser duty (MJ/s)',
               initialize=1, bounds=(0, 8))

    m.vap_Cp_const = {
        'benzene': {'A': -3.392E1, 'B': 4.739E-1, 'C': -3.017E-4,
                    'D': 7.130E-8, 'E': 0},
        'toluene': {'A': -2.435E1, 'B': 5.125E-1, 'C': -2.765E-4,
                    'D': 4.911E-8, 'E': 0}}
    m.liq_Cp_const = {
        'benzene': {'A': 1.29E5, 'B': -1.7E2, 'C': 6.48E-1,
                    'D': 0, 'E': 0},
        'toluene': {'A': 1.40E5, 'B': -1.52E2, 'C': 6.95E-1,
                    'D': 0, 'E': 0}}
    m.dH_vap = {'benzene': 33.770E3, 'toluene': 38.262E3}  # J/mol

    _build_column_heat_relations(m)

    @m.Constraint()
    def distillate_req(m):
        """
        Flow of benzene in the distillate meets the specified purity requirement.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo model of the distillation column.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the flow of benzene in the distillate meets the specified purity requirement. The flow of benzene in the distillate is greater than or equal to the distillate purity times the total distillate flow.
        """
        return m.D['benzene'] >= m.distillate_purity * m.dis

    @m.Constraint()
    def bottoms_req(m):
        """
        Flow of toluene in the bottoms meets the specified purity requirement.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo model of the distillation column.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the flow of toluene in the bottoms meets the specified purity requirement. The flow of toluene in the bottoms is greater than or equal to the bottoms purity times the total bottoms flow.
        """
        return m.B['toluene'] >= m.bottoms_purity * m.bot

    # Define the objective function as the sum of reboiler and condenser duty plus an indicator for tray activation
    # The objective is to minimize the sum of condenser and reboiler duties, Qc and Qb, multiplied by 1E3 to convert units,
    # and also the number of activated trays, which is obtained by summing up the indicator variables for the trays by 1E3 [$/No. of Trays].
    # m.obj = Objective(expr=(m.Qc + m.Qb) * 1E-3, sense=minimize)
    m.obj = Objective( expr=(m.Qc + m.Qb) * 1E3 + 1E3 * (
        sum(m.tray[t].binary_indicator_var for t in m.conditional_trays) + 1),
                       sense=minimize)
    # m.obj = Objective(
    #     expr=sum(m.tray[t].indicator_var for t in m.conditional_trays) + 1)

    @m.Constraint()
    def reflux_ratio_calc(m):
        """
        Reflux ratio is the ratio between the distillate flow and the difference in vapor flow in the condenser tray.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo model of the distillation column..

        Returns
        -------
        Pyomo.Constraint
            Constraint that the reflux ratio is the ratio between the distillate flow and the difference in vapor flow in the condenser tray.
        """
        return m.reflux_frac * (m.reflux_ratio + 1) == m.reflux_ratio

    @m.Constraint()
    def reboil_ratio_calc(m):
        """
        Reboil ratio is the ratio between the bottoms flow and the liquid leaving the reboiler.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo model of the distillation column.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the reboil ratio is the ratio between the bottoms flow and the liquid leaving the reboiler.
        """
        return m.boilup_frac * (m.reboil_ratio + 1) == m.reboil_ratio

    @m.Constraint(m.conditional_trays)
    def tray_ordering(m, t):
        """
        Trays close to the feed should be activated first.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo model of the distillation column.
        t : int
            Index of tray in the distillation column model.

        Returns
        -------
        Pyomo.Constraint
            Constraint that trays close to the feed should be activated first.
        """
        if t + 1 < m.condens_tray and t > m.feed_tray:
            return m.tray[t].binary_indicator_var >= \
                m.tray[t + 1].binary_indicator_var
        elif t > m.reboil_tray and t + 1 < m.feed_tray:
            return m.tray[t + 1].binary_indicator_var >= \
                m.tray[t].binary_indicator_var
        else:
            return Constraint.NoConstraint

    return m


def _build_conditional_tray_mass_balance(m, t, tray, no_tray):
    """
    Builds the constraints for mass balance, liquid and vapor composition for a given tray (t) in the distillation column.

    Parameters
    ----------
    m : Pyomo.ConcreteModel
        Pyomo model of the distillation column.
    t : int
        Index of tray in the distillation column model.
    tray : Pyomo.Disjunct
        Disjunct representing the existence of the tray.
    no_tray : Pyomo.Disjunct
        Disjunct representing the absence of the tray.

    Returns
    -------
    None
        None, but the mass balance constraints for the conditional tray are added to the Pyomo model.
    """
    @tray.Constraint(m.comps)
    def mass_balance(_, c):
        """
        Mass balance on each component on a tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            An unused placeholder, typically used to represent the model instance when defining constraints within a method, but not utilized in this specific constraint function. This placeholder denotes the conditional trays in the distillation column model.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the mass balance on each component on a tray is equal to the sum of the feed in, vapor from the tray, liquid from the tray above, liquid to the tray below, and vapor from the tray below.
        """
        return (
            # Feed in if feed tray
            (m.feed[c] if t == m.feed_tray else 0)
            # Vapor from tray t
            - m.V[c, t]
            # Loss to distillate if condenser
            - (m.D[c] if t == m.condens_tray else 0)
            # Liquid from tray above if not condenser
            + (m.L[c, t + 1] if t < m.condens_tray else 0)
            # Loss to bottoms if reboiler
            - (m.B[c] if t == m.reboil_tray else 0)
            # Liquid to tray below if not reboiler
            - (m.L[c, t] if t > m.reboil_tray else 0)
            # Vapor from tray below if not reboiler
            + (m.V[c, t - 1] if t > m.reboil_tray else 0) == 0)

    @tray.Constraint(m.comps)
    def tray_liquid_composition(_, c):
        """
        Liquid composition constraint for the tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            An unused placeholder denoting the conditional trays in the distillation column model.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the liquid flow rate for each component is equal to the liquid flow rate on the tray times the liquid composition on the tray.
        """
        return m.L[c, t] == m.liq[t] * m.x[c, t]

    @tray.Constraint(m.comps)
    def tray_vapor_compositions(_, c):
        """
        Vapor composition constraint for the tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            An unused placeholder denoting the conditional trays in the distillation column model.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the vapor flow rate for each component is equal to the vapor flow rate on the tray times the vapor composition on the tray.
        """
        return m.V[c, t] == m.vap[t] * m.y[c, t]

    @no_tray.Constraint(m.comps)
    def liq_comp_pass_through(_, c):
        """
        Liquid composition constraint for the case when the tray does not exist.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            An unused placeholder denoting the conditional trays in the distillation column model.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the liquid composition is equal to the liquid composition on the tray above, when the tray is not present.
        """
        return m.x[c, t] == m.x[c, t + 1]

    @no_tray.Constraint(m.comps)
    def liq_flow_pass_through(_, c):
        """
        Liquid flow rate constraint for the case when the tray does not exist.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            An unused placeholder denoting the conditional trays in the distillation column model.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the liquid flow rate is equal to the liquid flow rate on the tray above, when the tray is not present.
        """
        return m.L[c, t] == m.L[c, t + 1]

    @no_tray.Constraint(m.comps)
    def vap_comp_pass_through(_, c):
        """
        Vapor composition constraint for the case when the tray does not exist.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            An unused placeholder denoting the conditional trays in the distillation column model.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the vapor composition is equal to the vapor composition on the tray below, when the tray is not present.
        """
        return m.y[c, t] == m.y[c, t - 1]

    @no_tray.Constraint(m.comps)
    def vap_flow_pass_through(_, c):
        """
        Vapor flow rate constraint for the case when the tray does not exist.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            An unused placeholder denoting the conditional trays in the distillation column model.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the vapor flow rate is equal to the vapor flow rate on the tray below, when the tray is not present.
        """
        return m.V[c, t] == m.V[c, t - 1]


def _build_feed_tray_mass_balance(m):
    """
    Constructs the mass balance and composition constraints for the feed tray in the distillation column.

    Parameters
    ----------
    m : Pyomo.ConcreteModel
        Pyomo model of the distillation column.

    Returns
    -------
    None
        None, but the mass balance constraints for the feed tray are added to the Pyomo model.
    """
    t = m.feed_tray

    @m.Constraint(m.comps)
    def feed_mass_balance(_, c):
        """
        Mass balance on each component on a tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder for the model instance, required for defining constraints within the Pyomo framework. It specifies the context in which the feed tray conditions are applied.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the mass balance on each component on a tray is equal to the sum of the feed in, vapor from the tray, liquid from the tray above, liquid to the tray below, and vapor from the tray below.
        """
        return (
            m.feed[c]        # Feed in
            - m.V[c, t]      # Vapor from tray t
            + m.L[c, t + 1]  # Liquid from tray above
            - m.L[c, t]      # Liquid to tray below
            + m.V[c, t - 1]  # Vapor from tray below
            == 0)

    @m.Constraint(m.comps)
    def feed_tray_liquid_composition(_, c):
        """
        Liquid composition constraint for the feed tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder for the model instance, required for defining constraints within the Pyomo framework. It specifies the context in which the feed tray conditions are applied.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the liquid flow rate for each component is equal to the liquid flow rate on the tray times the liquid composition on the tray.
        """
        return m.L[c, t] == m.liq[t] * m.x[c, t]

    @m.Constraint(m.comps)
    def feed_tray_vapor_composition(_, c):
        """
        Vapor composition on each component on a tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder for the model instance, required for defining constraints within the Pyomo framework. It specifies the context in which the feed tray conditions are applied.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the vapor flow rate for each component is equal to the vapor flow rate on the tray times the vapor composition on the tray.
        """
        return m.V[c, t] == m.vap[t] * m.y[c, t]


def _build_condenser_mass_balance(m):
    """
    Constructs the mass balance equations for the condenser tray in a distillation column.

    Parameters
    ----------
    m : Pyomo.ConcreteModel
        Pyomo model of the distillation column..

    Returns
    -------
    None
        None, but the mass balance constraints for the condenser are added to the Pyomo model.
    """
    t = m.condens_tray

    @m.Constraint(m.comps)
    def condenser_mass_balance(_, c):
        """
        Mass balance for each component in the condenser tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the mass balance and composition constraints to the condenser tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the mass balance for each component in the condenser tray is equal to the sum of the vapor from the tray, loss to distillate, liquid to the tray below, and vapor from the tray below.
        """
        return (
            - m.V[c, t]      # Vapor from tray t
            - m.D[c]         # Loss to distillate
            - m.L[c, t]      # Liquid to tray below
            + m.V[c, t - 1]  # Vapor from tray below
            == 0)

    @m.partial_cond.Constraint(m.comps)
    def condenser_liquid_composition(_, c):
        """
        Liquid composition constraint for the condenser tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the mass balance and composition constraints to the condenser tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the liquid flow rate for each component is equal to the liquid flow rate on the tray times the liquid composition on the tray.
        """
        return m.L[c, t] == m.liq[t] * m.x[c, t]

    @m.partial_cond.Constraint(m.comps)
    def condenser_vapor_composition(_, c):
        """
        Vapor composition constraint for the condenser tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the mass balance and composition constraints to the condenser tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the vapor flow rate for each component is equal to the vapor flow rate on the tray times the vapor composition on the tray.
        """
        return m.V[c, t] == m.vap[t] * m.y[c, t]

    @m.total_cond.Constraint(m.comps)
    def no_vapor_flow(_, c):
        """
        No vapor flow for each component in the case of total condensation.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the mass balance and composition constraints to the condenser tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that there is no vapor flow for each component in the case of total condensation.
        """
        return m.V[c, t] == 0

    @m.total_cond.Constraint()
    def no_total_vapor_flow(_):
        """
        No total vapor flow for the condenser tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the mass balance and composition constraints to the condenser tray in the distillation column.

        Returns
        -------
        Pyomo.Constraint
            Constraint that there is no total vapor flow for the condenser tray.
        """
        return m.vap[t] == 0

    @m.total_cond.Constraint(m.comps)
    def liquid_fraction_pass_through(_, c):
        """
        Liquid fraction pass-through for each component in the case of total condensation.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the mass balance and composition constraints to the condenser tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the liquid fraction is equal to the vapor fraction on the tray below in the case of total condensation.
        """
        return m.x[c, t] == m.y[c, t - 1]

    @m.Constraint(m.comps)
    def condenser_distillate_composition(_, c):
        """
        Distillate composition constraint for the condenser tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the mass balance and composition constraints to the condenser tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the distillate flow rate for each component is equal to the distillate flow rate times the distillate composition.
        """
        return m.D[c] == m.dis * m.x[c, t]


def _build_reboiler_mass_balance(m):
    """
    Constructs the mass balance equations for the reboiler tray in a distillation column.

    Parameters
    ----------
    m : Pyomo.ConcreteModel
        Pyomo model of the distillation column..

    Returns
    -------
    None
        None, but the mass balance constraints for the reboiler are added to the Pyomo model.
    """
    t = m.reboil_tray

    @m.Constraint(m.comps)
    def reboiler_mass_balance(_, c):
        """
        Mass balance for each component in the reboiler tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the mass balance and composition constraints to the reboiler tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the mass balance for each component in the reboiler tray is equal to the sum of the vapor from the tray, liquid from the tray above, and loss to bottoms.
        """
        t = m.reboil_tray
        return (
            - m.V[c, t]      # Vapor from tray t
            + m.L[c, t + 1]  # Liquid from tray above
            - m.B[c]         # Loss to bottoms
            == 0)

    @m.Constraint(m.comps)
    def reboiler_liquid_composition(_, c):
        """
        Liquid composition constraint for the reboiler tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the mass balance and composition constraints to the reboiler tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the liquid flow rate for each component is equal to the liquid flow rate on the tray times the liquid composition on the tray.
        """
        return m.L[c, t] == m.liq[t] * m.x[c, t]

    @m.Constraint(m.comps)
    def reboiler_vapor_composition(_, c):
        """
        Vapor composition constraint for the reboiler tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the mass balance and composition constraints to the reboiler tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the vapor flow rate for each component is equal to the vapor flow rate on the tray times the vapor composition on the tray.
        """
        return m.V[c, t] == m.vap[t] * m.y[c, t]


def _build_tray_phase_equilibrium(m, t, tray):
    """_summary_

    Parameters
    ----------
    m : Pyomo.ConcreteModel
        Pyomo model of the distillation column.
    t : int
        Index of tray in the distillation column model.
    tray : Pyomo.Disjunct
        Disjunct representing the existence of the tray.

    Returns
    -------
    None
        None, but the phase equilibrium constraints for the tray are added to the Pyomo model.
    """
    @tray.Constraint(m.comps)
    def raoults_law(_, c):
        """
        Raoult's law for each component in a tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the phase equilibrium constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            The Raoult's law for each component in trays is calculated as the product of the liquid mole fraction and the phase equilibrium constant. The product is equal to the vapor mole fraction.
        """
        return m.y[c, t] == m.x[c, t] * m.Kc[c, t]

    @tray.Constraint(m.comps)
    def phase_equil_const(_, c):
        """
        Phase equilibrium constraint for each component in a tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the phase equilibrium constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            The phase equilibrium constant for each component in a tray multiplied with the pressure is equal to the product of the activity coefficient and the pure component vapor pressure.
        """
        return m.Kc[c, t] * m.P == (
            m.gamma[c, t] * m.Pvap[c, t])

    @tray.Constraint(m.comps)
    def Pvap_relation(_, c):
        """
        Antoine's equation for the vapor pressure of each component in a tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the phase equilibrium constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Antoine's equation for the vapor pressure of each component in a tray is calculated as the logarithm of the vapor pressure minus the logarithm of the critical pressure times one minus the fraction of critical temperature. The equation is equal to the sum of the Antoine coefficients times the fraction of critical temperature raised to different powers.
        """
        k = m.pvap_const[c]
        x = m.Pvap_X[c, t]
        return (log(m.Pvap[c, t]) - log(k['Pc'])) * (1 - x) == (
            k['A'] * x +
            k['B'] * x ** 1.5 +
            k['C'] * x ** 3 +
            k['D'] * x ** 6)

    @tray.Constraint(m.comps)
    def Pvap_X_defn(_, c):
        """
        Defines the relationship between the one minus the reduced temperature variable (Pvap_X) for each component in a tray, and the actual temperature of the tray, normalized by the critical temperature of the component (Tc).

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the phase equilibrium constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            The relationship between the one minus the reduced temperature variable (Pvap_X) for each component in a tray, and the actual temperature of the tray, normalized by the critical temperature of the component (Tc).
        """
        k = m.pvap_const[c]
        return m.Pvap_X[c, t] == 1 - m.T[t] / k['Tc']

    @tray.Constraint(m.comps)
    def gamma_calc(_, c):
        """
        Calculates the activity coefficient for each component in a tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the phase equilibrium constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            The activity coefficient for each component in a tray is calculated as 1.
        """
        return m.gamma[c, t] == 1


def _build_column_heat_relations(m):
    """
    Constructs the enthalpy relations for both liquid and vapor phases in each tray of a distillation column.

    Parameters
    ----------
    m : Pyomo.ConcreteModel
        Pyomo model of the distillation column..

    Returns
    -------
    None
        None, but the energy balance constraints for the distillation column are added to the Pyomo model.
    """
    @m.Expression(m.trays, m.comps)
    def liq_enthalpy_expr(_, t, c):
        """
        Liquid phase enthalpy based on the heat capacity coefficients and the temperature difference from a reference temperature [kJ/mol].

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the distillation column.
        t : int
            Index of tray in the distillation column model.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Expression
            Enthalpy based on the heat capacity coefficients and the temperature difference from a reference temperature [kJ/mol].
        """
        k = m.liq_Cp_const[c]
        return (
            k['A'] * (m.T[t] - m.T_ref) +
            k['B'] * (m.T[t] ** 2 - m.T_ref ** 2) / 2 +
            k['C'] * (m.T[t] ** 3 - m.T_ref ** 3) / 3 +
            k['D'] * (m.T[t] ** 4 - m.T_ref ** 4) / 4 +
            k['E'] * (m.T[t] ** 5 - m.T_ref ** 5) / 5) * 1E-6 # Convert [J/mol] to [MJ/mol]

    @m.Expression(m.trays, m.comps)
    def vap_enthalpy_expr(_, t, c):
        """
        Vapor phase enthalpy based on the heat capacity coefficients and the temperature difference from a reference temperature [kJ/mol].

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the distillation column.
        t : int
            Index of tray in the distillation column model.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Expression
            Enthalpy based on the heat capacity coefficients and the temperature difference from a reference temperature [kJ/mol].
        """
        k = m.vap_Cp_const[c]
        return (
            m.dH_vap[c] +
            k['A'] * (m.T[t] - m.T_ref) +
            k['B'] * (m.T[t] ** 2 - m.T_ref ** 2) / 2 +
            k['C'] * (m.T[t] ** 3 - m.T_ref ** 3) / 3 +
            k['D'] * (m.T[t] ** 4 - m.T_ref ** 4) / 4 +
            k['E'] * (m.T[t] ** 5 - m.T_ref ** 5) / 5) * 1E-3 # Convert [J/mol] to [kJ/mol]

    for t in m.conditional_trays:
        _build_conditional_tray_energy_balance(m, t, m.tray[t], m.no_tray[t])
    _build_feed_tray_energy_balance(m)
    _build_condenser_energy_balance(m)
    _build_reboiler_energy_balance(m)


def _build_conditional_tray_energy_balance(m, t, tray, no_tray):
    """
    Constructs the energy balance constraints for a specific tray in a distillation column, considering both active and inactive (pass-through) scenarios.

    Parameters
    ----------
    m : Pyomo.ConcreteModel
        Pyomo model of the distillation column.
    t : int
        Index of tray in the distillation column model.
    tray : Pyomo.Disjunct
        Disjunct representing the existence of the tray.
    no_tray : Pyomo.Disjunct
        Disjunct representing the absence of the tray.

    Returns
    -------
    None
        None, but the energy balance constraints for the conditional tray are added to the Pyomo model.
    """
    @tray.Constraint()
    def energy_balance(_):
        """_summary_

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the trays in the distillation column.

        Returns
        -------
        Pyomo.Constraint
            _description_
        """
        return sum(
            m.L[c, t + 1] * m.H_L[c, t + 1]  # heat of liquid from tray above
            - m.L[c, t] * m.H_L[c, t]  # heat of liquid to tray below
            + m.V[c, t - 1] * m.H_V[c, t - 1]  # heat of vapor from tray below
            - m.V[c, t] * m.H_V[c, t]  # heat of vapor to tray above
            for c in m.comps) * 1E-3 == 0

    @tray.Constraint(m.comps)
    def liq_enthalpy_calc(_, c):
        """
        Liquid enthalpy calculation for each component in a tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the liquid enthalpy for each component is equal to the liquid enthalpy expression.
        """
        return m.H_L[c, t] == m.liq_enthalpy_expr[t, c]

    @tray.Constraint(m.comps)
    def vap_enthalpy_calc(_, c):
        """
        Vapor enthalpy calculation for each component in a tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the vapor enthalpy for each component is equal to the vapor enthalpy expression.
        """
        return m.H_V[c, t] == m.vap_enthalpy_expr[t, c]

    @no_tray.Constraint(m.comps)
    def liq_enthalpy_pass_through(_, c):
        """
        Liquid enthalpy pass-through for each component in the case of no tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the liquid enthalpy is equal to the liquid enthalpy on the tray below, when the tray is not present.
        """
        return m.H_L[c, t] == m.H_L[c, t + 1]

    @no_tray.Constraint(m.comps)
    def vap_enthalpy_pass_through(_, c):
        """
        Vapor enthalpy pass-through for each component in the case of no tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the vapor enthalpy is equal to the vapor enthalpy on the tray above, when the tray is not present.
        """
        return m.H_V[c, t] == m.H_V[c, t - 1]


def _build_feed_tray_energy_balance(m):
    """
    Energy balance for the feed tray.

    Parameters
    ----------
    m : Pyomo.ConcreteModel
        Pyomo model of the distillation column..

    Returns
    -------
    None
        None, but adds constraints, which are energy balances for the feed tray, to the Pyomo model of the distillation column
    """
    t = m.feed_tray

    @m.Constraint()
    def feed_tray_energy_balance(_):
        """
        Energy balance for the feed tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the feed tray in the distillation column.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the sum of the heat of the feed and the heat of the liquid and vapor streams is equal to zero.
        """
        return (
            sum(m.feed[c] * (
                m.H_L_spec_feed[c] * (1 - m.feed_vap_frac) +
                m.H_V_spec_feed[c] * m.feed_vap_frac)
                for c in m.comps) +
            sum(
                # Heat of liquid from tray above
                m.L[c, t + 1] * m.H_L[c, t + 1]
                # heat of liquid to tray below
                - m.L[c, t] * m.H_L[c, t]
                # heat of vapor from tray below
                + m.V[c, t - 1] * m.H_V[c, t - 1]
                # heat of vapor to tray above
                - m.V[c, t] * m.H_V[c, t]
                for c in m.comps)) * 1E-3 == 0

    @m.Constraint(m.comps)
    def feed_tray_liq_enthalpy_calc(_, c):
        """
        Liquid enthalpy calculation for the feed tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the feed tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the liquid enthalpy is equal to the liquid enthalpy expression.
        """
        return m.H_L[c, t] == m.liq_enthalpy_expr[t, c]

    @m.Constraint(m.comps)
    def feed_tray_vap_enthalpy_calc(_, c):
        """
        Vapor enthalpy calculation for the feed tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the feed tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the vapor enthalpy is equal to the vapor enthalpy expression.
        """
        return m.H_V[c, t] == m.vap_enthalpy_expr[t, c]

    @m.Expression(m.comps)
    def feed_liq_enthalpy_expr(_, c):
        """
        Liquid enthalpy expression for the feed tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the feed tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Expression
            Liquid enthalpy expression for the feed tray.
        """
        k = m.liq_Cp_const[c]
        return (
            k['A'] * (m.T_feed - m.T_ref) +
            k['B'] * (m.T_feed ** 2 - m.T_ref ** 2) / 2 +
            k['C'] * (m.T_feed ** 3 - m.T_ref ** 3) / 3 +
            k['D'] * (m.T_feed ** 4 - m.T_ref ** 4) / 4 +
            k['E'] * (m.T_feed ** 5 - m.T_ref ** 5) / 5) * 1E-6 # Convert the result from [J/mol] to [MJ/mol]

    @m.Constraint(m.comps)
    def feed_liq_enthalpy_calc(_, c):
        """_summary_

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the feed tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            _description_
        """
        return m.H_L_spec_feed[c] == m.feed_liq_enthalpy_expr[c]

    @m.Expression(m.comps)
    def feed_vap_enthalpy_expr(_, c):
        """
        Vapor enthalpy expression for the feed tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the feed tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Expression
            Vapor enthalpy expression for the feed tray.
        """
        k = m.vap_Cp_const[c]
        return (
            m.dH_vap[c] +
            k['A'] * (m.T_feed - m.T_ref) +
            k['B'] * (m.T_feed ** 2 - m.T_ref ** 2) / 2 +
            k['C'] * (m.T_feed ** 3 - m.T_ref ** 3) / 3 +
            k['D'] * (m.T_feed ** 4 - m.T_ref ** 4) / 4 +
            k['E'] * (m.T_feed ** 5 - m.T_ref ** 5) / 5) * 1E-3 # Convert the result from [J/mol] to [kJ/mol]

    @m.Constraint(m.comps)
    def feed_vap_enthalpy_calc(_, c):
        """
        Vapor enthalpy calculation for the feed tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the feed tray in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the vapor enthalpy is equal to the vapor enthalpy expression.
        """
        return m.H_V_spec_feed[c] == m.feed_vap_enthalpy_expr[c]


def _build_condenser_energy_balance(m):
    """
    Energy balance for the condenser.

    Parameters
    ----------
    m : Pyomo.ConcreteModel
        Pyomo model of the distillation column.

    Returns
    -------
    None
        None, but adds constraints, which are energy balances for the condenser, to the Pyomo model of the distillation column
    """
    t = m.condens_tray

    @m.partial_cond.Constraint()
    def partial_condenser_energy_balance(_):
        """
        Energy balance for the partial condenser.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the condenser in the distillation column.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the sum of the heat of the liquid distillate, the heat of the liquid to the tray below, the heat of the vapor from the tray below, and the heat of the vapor from the partial condenser is equal to zero.
        """
        return -m.Qc + sum(
            - m.D[c] * m.H_L[c, t]  # heat of liquid distillate
            - m.L[c, t] * m.H_L[c, t]  # heat of liquid to tray below
            + m.V[c, t - 1] * m.H_V[c, t - 1]  # heat of vapor from tray below
            - m.V[c, t] * m.H_V[c, t]  # heat of vapor from partial condenser
            for c in m.comps) * 1E-3 == 0 # Convert the result from [kJ/mol] to [MJ/mol]

    @m.total_cond.Constraint()
    def total_condenser_energy_balance(_):
        """
        Energy balance for the total condenser.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the condenser in the distillation column.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the sum of the heat of the liquid distillate, the heat of the liquid to the tray below, and the heat of the vapor from the tray below is equal to zero.
        """
        return -m.Qc + sum(
            - m.D[c] * m.H_L[c, t]  # heat of liquid distillate
            - m.L[c, t] * m.H_L[c, t]  # heat of liquid to tray below
            + m.V[c, t - 1] * m.H_V[c, t - 1]  # heat of vapor from tray below
            for c in m.comps) * 1E-3 == 0

    @m.Constraint(m.comps)
    def condenser_liq_enthalpy_calc(_, c):
        """
        Liquid enthalpy calculation for each component in the condenser.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the condenser in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the liquid enthalpy for each component is equal to the liquid enthalpy expression.
        """
        return m.H_L[c, t] == m.liq_enthalpy_expr[t, c]

    @m.partial_cond.Constraint(m.comps)
    def vap_enthalpy_calc(_, c):
        """
        Vapor enthalpy calculation for each component in the condenser.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the condenser in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the vapor enthalpy for each component is equal to the vapor enthalpy expression.
        """
        return m.H_V[c, t] == m.vap_enthalpy_expr[t, c]


def _build_reboiler_energy_balance(m):
    """
    Energy balance for the reboiler.

    Parameters
    ----------
    m : Pyomo.ConcreteModel
        Pyomo model of the distillation column.

    Returns
    -------
    None
        None, but adds constraints, which are energy balances for the reboiler, to the Pyomo model of the distillation column
    """
    t = m.reboil_tray

    @m.Constraint()
    def reboiler_energy_balance(_):
        """
        Energy balance for the reboiler.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the reboiler in the distillation column.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the sum of the heat of the liquid bottoms, the heat of the liquid from the tray above, the heat of the vapor to the tray above, and the heat of the vapor from the reboiler is equal to zero.
        """
        return m.Qb + sum(
            m.L[c, t + 1] * m.H_L[c, t + 1]  # Heat of liquid from tray above
            - m.B[c] * m.H_L[c, t]  # heat of liquid bottoms if reboiler
            - m.V[c, t] * m.H_V[c, t]  # heat of vapor to tray above
            for c in m.comps) * 1E-3 == 0

    @m.Constraint(m.comps)
    def reboiler_liq_enthalpy_calc(_, c):
        """
        Liquid enthalpy calculation for each component in the reboiler.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the reboiler in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the liquid enthalpy for each component is equal to the liquid enthalpy expression.
        """
        return m.H_L[c, t] == m.liq_enthalpy_expr[t, c]

    @m.Constraint(m.comps)
    def reboiler_vap_enthalpy_calc(_, c):
        """
        Vapor enthalpy calculation for each component in the reboiler.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the energy balance constraints to the reboiler in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.

        Returns
        -------
        Pyomo.Constraint
            Constraint that the vapor enthalpy for each component is equal to the vapor enthalpy expression.
        """
        return m.H_V[c, t] == m.vap_enthalpy_expr[t, c]
