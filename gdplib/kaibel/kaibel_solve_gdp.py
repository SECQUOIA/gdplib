""" Kaibel Column model: GDP formulation.

The solution requires the specification of certain parameters, such as the number trays, feed location, etc., and an initialization procedure, which consists of the next three steps:
(i) a preliminary design of the separation considering a sequence of indirect continuous distillation columns (CDCs) to obtain the minimum number of stages with Fenske Equation in the function initialize_kaibel in kaibel_init.py 
(ii) flash calculation for the feed with the function calc_side_feed_flash in kaibel_side_flash.py
(iii) calculation of variable bounds by solving the NLP problem.

After the initialization, the GDP model is built.
"""

from math import copysign

from pyomo.environ import (
    Constraint,
    exp,
    minimize,
    NonNegativeReals,
    Objective,
    RangeSet,
    Set,
    Var,
)
from pyomo.gdp import Disjunct

from gdplib.kaibel.kaibel_init import initialize_kaibel

from gdplib.kaibel.kaibel_side_flash import calc_side_feed_flash

# from .kaibel_side_flash import calc_side_feed_flash


def build_model():
    """
    Build the GDP Kaibel Column model.
    It combines the initialization of the model and the flash calculation for the side feed before the GDP formulation.

    Returns
    -------
    ConcreteModel
        The constructed GDP Kaibel Column model.
    """

    # Calculation of the theoretical minimum number of trays (Knmin) and initial temperature values (TB0, Tf0, TD0).
    m = initialize_kaibel()

    # Side feed init. Returns side feed vapor composition yfi and vapor fraction q_init
    m = calc_side_feed_flash(m)

    m.name = "GDP Kaibel Column"

    #### Calculated initial values
    m.Treb = m.TB0 + 5  # Reboiler temperature [K]
    m.Tbot = m.TB0  # Bottom-most tray temperature [K]
    m.Ttop = m.TD0  # Top-most tray temperature [K]
    m.Tcon = m.TD0 - 5  # Condenser temperature [K]

    m.dv0 = {}  # Initial vapor distributor value
    m.dl0 = {}  # Initial liquid distributor value
    m.dv0[2] = 0.516
    m.dv0[3] = 1 - m.dv0[2]
    m.dl0[2] = 0.36
    m.dl0[3] = 1 - m.dl0[2]

    #### Calculated upper and lower bounds
    m.min_tray = m.Knmin * 0.8  # Lower bound on number of trays
    m.Tlo = m.Tcon - 20  # Temperature lower bound
    m.Tup = m.Treb + 20  # Temperature upper bound

    m.flow_max = 1e3  # Flowrates upper bound [mol/s]
    m.Qmax = 60  # Heat loads upper bound [J/s]

    #### Column tray details
    m.num_trays = m.np  # Trays per section. np = 25
    m.min_num_trays = 10  # Minimum number of trays per section
    m.num_total = m.np * 3  # Total number of trays
    m.feed_tray = 12  # Side feed tray
    m.sideout1_tray = 8  # Side outlet 1 tray
    m.sideout2_tray = 17  # Side outlet 2 tray
    m.reb_tray = 1  # Reboiler tray. Dividing wall starting tray
    m.con_tray = m.num_trays  # Condenser tray. Dividing wall ending tray

    # ------------------------------------------------------------------

    #                          Beginning of model

    # ------------------------------------------------------------------

    ## Sets
    m.section = RangeSet(
        4, doc="Column sections:1=top, 2=feed side, 3=prod side, 4=bot"
    )
    m.section_main = Set(initialize=[1, 4], doc="Main sections of the column")

    m.tray = RangeSet(m.np, doc="Potential trays in each section")
    m.tray_total = RangeSet(m.num_total, doc="Total trays in the column")
    m.tray_below_feed = RangeSet(m.feed_tray, doc="Trays below feed")
    m.tray_below_so1 = RangeSet(m.sideout1_tray, doc="Trays below side outlet 1")
    m.tray_below_so2 = RangeSet(m.sideout2_tray, doc="Trays below side outlet 2")

    m.comp = RangeSet(4, doc="Components")
    m.dw = RangeSet(2, 3, doc="Dividing wall sections")
    m.cplv = RangeSet(2, doc="Heat capacity: 1=liquid, 2=vapor")
    m.so = RangeSet(2, doc="Side product outlets")
    m.bounds = RangeSet(2, doc="Number of boundary condition values")

    m.candidate_trays_main = Set(
        initialize=m.tray - [m.con_tray, m.reb_tray],
        doc="Candidate trays for top and \
                                 bottom sections 1 and 4",
    )
    m.candidate_trays_feed = Set(
        initialize=m.tray - [m.con_tray, m.feed_tray, m.reb_tray],
        doc="Candidate trays for feed section 2",
    )
    m.candidate_trays_product = Set(
        initialize=m.tray - [m.con_tray, m.sideout1_tray, m.sideout2_tray, m.reb_tray],
        doc="Candidate trays for product section 3",
    )

    ## Calculation of initial values
    m.dHvap = {}  # Heat of vaporization [J/mol]

    m.P0 = {}  # Initial pressure [bar]
    m.T0 = {}  # Initial temperature [K]
    m.L0 = {}  # Initial individual liquid flowrate [mol/s]
    m.V0 = {}  # Initial individual vapor flowrate [mol/s]
    m.Vtotal0 = {}  # Initial total vapor flowrate [mol/s]
    m.Ltotal0 = {}  # Initial liquid flowrate [mol/s]
    m.x0 = {}  # Initial liquid composition
    m.y0 = {}  # Initial vapor composition
    m.actv0 = {}  # Initial activity coefficients
    m.cpdT0 = {}  # Initial heat capacity for liquid and vapor phases [J/mol/K]
    m.hl0 = {}  # Initial liquid enthalpy [J/mol]
    m.hv0 = {}  # Initial vapor enthalpy [J/mol]
    m.Pi = m.Preb  # Initial given pressure value [bar]
    m.Ti = {}  # Initial known temperature values [K]

    ## Initial values for pressure, temperature, flowrates, composition, and enthalpy
    for sec in m.section:
        for n_tray in m.tray:
            m.P0[sec, n_tray] = m.Pi

    for sec in m.section:
        for n_tray in m.tray:
            for comp in m.comp:
                m.L0[sec, n_tray, comp] = m.Li
                m.V0[sec, n_tray, comp] = m.Vi

    for sec in m.section:
        for n_tray in m.tray:
            m.Ltotal0[sec, n_tray] = sum(m.L0[sec, n_tray, comp] for comp in m.comp)
            m.Vtotal0[sec, n_tray] = sum(m.V0[sec, n_tray, comp] for comp in m.comp)

    for n_tray in m.tray_total:
        if n_tray == m.reb_tray:
            m.Ti[n_tray] = m.Treb
        elif n_tray == m.num_total:
            m.Ti[n_tray] = m.Tcon
        else:
            m.Ti[n_tray] = m.Tbot + (m.Ttop - m.Tbot) * (n_tray - 2) / (m.num_total - 3)

    for n_tray in m.tray_total:
        if n_tray <= m.num_trays:
            m.T0[1, n_tray] = m.Ti[n_tray]
        elif n_tray >= m.num_trays and n_tray <= m.num_trays * 2:
            m.T0[2, n_tray - m.num_trays] = m.Ti[n_tray]
            m.T0[3, n_tray - m.num_trays] = m.Ti[n_tray]
        elif n_tray >= m.num_trays * 2:
            m.T0[4, n_tray - m.num_trays * 2] = m.Ti[n_tray]

    ## Initial vapor and liquid composition of the feed and activity coefficients
    for sec in m.section:
        for n_tray in m.tray:
            for comp in m.comp:
                m.x0[sec, n_tray, comp] = m.xfi[comp]
                m.actv0[sec, n_tray, comp] = 1
                m.y0[sec, n_tray, comp] = m.xfi[comp]

    ## Assigns the enthalpy boundary values, heat capacity, heat of vaporization calculation, temperature bounds, and light and heavy key components.
    hlb = {}  # Liquid enthalpy [J/mol]
    hvb = {}  # Vapor enthalpy  [J/mol]
    cpb = {}  # Heact capacity [J/mol/K]
    dHvapb = {}  # Heat of vaporization [J/mol]
    Tbounds = {}  # Temperature bounds [K]
    kc = {}  # Light and heavy key components
    Tbounds[1] = m.Tcon  # Condenser temperature [K]
    Tbounds[2] = m.Treb  # Reboiler temperature [K]
    kc[1] = m.lc
    kc[2] = m.hc

    ## Heat of vaporization calculation for each component in the feed.
    for comp in m.comp:
        dHvapb[comp] = -(
            m.Rgas
            * m.prop[comp, 'TC']
            * (
                m.prop[comp, 'vpA'] * (1 - m.Tref / m.prop[comp, 'TC'])
                + m.prop[comp, 'vpB'] * (1 - m.Tref / m.prop[comp, 'TC']) ** 1.5
                + m.prop[comp, 'vpC'] * (1 - m.Tref / m.prop[comp, 'TC']) ** 3
                + m.prop[comp, 'vpD'] * (1 - m.Tref / m.prop[comp, 'TC']) ** 6
            )
            + m.Rgas
            * m.Tref
            * (
                m.prop[comp, 'vpA']
                + 1.5 * m.prop[comp, 'vpB'] * (1 - m.Tref / m.prop[comp, 'TC']) ** 0.5
                + 3 * m.prop[comp, 'vpC'] * (1 - m.Tref / m.prop[comp, 'TC']) ** 2
                + 6 * m.prop[comp, 'vpD'] * (1 - m.Tref / m.prop[comp, 'TC']) ** 5
            )
        )

    ## Boundary values for heat capacity and enthalpy of liquid and vapor phases for light and heavy key components in the feed.
    for b in m.bounds:
        for cp in m.cplv:
            cpb[b, cp] = m.cpc[cp] * (
                (Tbounds[b] - m.Tref) * m.prop[kc[b], 'cpA', cp]
                + (Tbounds[b] ** 2 - m.Tref**2)
                * m.prop[kc[b], 'cpB', cp]
                * m.cpc2['A', cp]
                / 2
                + (Tbounds[b] ** 3 - m.Tref**3)
                * m.prop[kc[b], 'cpC', cp]
                * m.cpc2['B', cp]
                / 3
                + (Tbounds[b] ** 4 - m.Tref**4) * m.prop[kc[b], 'cpD', cp] / 4
            )
        hlb[b] = cpb[b, 1]
        hvb[b] = cpb[b, 2] + dHvapb[b]

    m.hllo = (
        (1 - copysign(0.2, hlb[1])) * hlb[1] / m.Hscale
    )  # Liquid enthalpy lower bound
    m.hlup = (
        (1 + copysign(0.2, hlb[2])) * hlb[2] / m.Hscale
    )  # Liquid enthalpy upper bound
    m.hvlo = (
        (1 - copysign(0.2, hvb[1])) * hvb[1] / m.Hscale
    )  # Vapor enthalpy lower bound
    m.hvup = (
        (1 + copysign(0.2, hvb[2])) * hvb[2] / m.Hscale
    )  # Vapor enthalpy upper bound
    # copysign is a function that returns the first argument with the sign of the second argument

    ## Heat of vaporization for each component in the feed scaled by Hscale
    for comp in m.comp:
        m.dHvap[comp] = dHvapb[comp] / m.Hscale

    ## Heat capacity calculation for liquid and vapor phases using Ruczika-D method for each component in the feed, section, and tray
    for sec in m.section:
        for n_tray in m.tray:
            for comp in m.comp:
                for cp in m.cplv:
                    m.cpdT0[sec, n_tray, comp, cp] = (
                        m.cpc[cp]
                        * (
                            (m.T0[sec, n_tray] - m.Tref) * m.prop[comp, 'cpA', cp]
                            + (m.T0[sec, n_tray] ** 2 - m.Tref**2)
                            * m.prop[comp, 'cpB', cp]
                            * m.cpc2['A', cp]
                            / 2
                            + (m.T0[sec, n_tray] ** 3 - m.Tref**3)
                            * m.prop[comp, 'cpC', cp]
                            * m.cpc2['B', cp]
                            / 3
                            + (m.T0[sec, n_tray] ** 4 - m.Tref**4)
                            * m.prop[comp, 'cpD', cp]
                            / 4
                        )
                        / m.Hscale
                    )

    ## Liquid and vapor enthalpy calculation using Ruczika-D method for each component in the feed, section, and tray
    for sec in m.section:
        for n_tray in m.tray:
            for comp in m.comp:
                m.hl0[sec, n_tray, comp] = m.cpdT0[sec, n_tray, comp, 1]
                m.hv0[sec, n_tray, comp] = m.cpdT0[sec, n_tray, comp, 2] + m.dHvap[comp]

    #### Side feed
    m.cpdTf = {}  # Heat capacity for side feed [J/mol K]
    m.hlf = {}  # Liquid enthalpy for side feed [J/mol]
    m.hvf = {}  # Vapor enthalpy for side feed [J/mol]
    m.F0 = {}  # Side feed flowrate per component [mol/s]

    ## Heat capacity in liquid and vapor phases for side feed for each component using Ruczika-D method
    for comp in m.comp:
        for cp in m.cplv:
            m.cpdTf[comp, cp] = (
                m.cpc[cp]
                * (
                    (m.Tf - m.Tref) * m.prop[comp, 'cpA', cp]
                    + (m.Tf**2 - m.Tref**2)
                    * m.prop[comp, 'cpB', cp]
                    * m.cpc2['A', cp]
                    / 2
                    + (m.Tf**3 - m.Tref**3)
                    * m.prop[comp, 'cpC', cp]
                    * m.cpc2['B', cp]
                    / 3
                    + (m.Tf**4 - m.Tref**4) * m.prop[comp, 'cpD', cp] / 4
                )
                / m.Hscale
            )

    ## Side feed flowrate and liquid and vapor enthalpy calculation using Ruczika-D method for each component in the feed
    for comp in m.comp:
        m.F0[comp] = (
            m.xfi[comp] * m.Fi
        )  # Side feed flowrate per component computed from the feed composition and flowrate Fi
        m.hlf[comp] = m.cpdTf[
            comp, 1
        ]  # Liquid enthalpy for side feed computed from the heat capacity for side feed and liquid phase
        m.hvf[comp] = (
            m.cpdTf[comp, 2] + m.dHvap[comp]
        )  # Vapor enthalpy for side feed computed from the heat capacity for side feed and vapor phase and heat of vaporization

    m.P = Var(
        m.section,
        m.tray,
        doc="Pressure at each potential tray in bars",
        domain=NonNegativeReals,
        bounds=(m.Pcon, m.Preb),
        initialize=m.P0,
    )
    m.T = Var(
        m.section,
        m.tray,
        doc="Temperature at each potential tray [K]",
        domain=NonNegativeReals,
        bounds=(m.Tlo, m.Tup),
        initialize=m.T0,
    )

    m.x = Var(
        m.section,
        m.tray,
        m.comp,
        doc="Liquid composition",
        domain=NonNegativeReals,
        bounds=(0, 1),
        initialize=m.x0,
    )
    m.y = Var(
        m.section,
        m.tray,
        m.comp,
        doc="Vapor composition",
        domain=NonNegativeReals,
        bounds=(0, 1),
        initialize=m.y0,
    )

    m.dl = Var(
        m.dw,
        doc="Liquid distributor in the dividing wall sections",
        bounds=(0.2, 0.8),
        initialize=m.dl0,
    )
    m.dv = Var(
        m.dw,
        doc="Vapor distributor in the dividing wall sections",
        bounds=(0, 1),
        domain=NonNegativeReals,
        initialize=m.dv0,
    )

    m.V = Var(
        m.section,
        m.tray,
        m.comp,
        doc="Vapor flowrate [mol/s]",
        domain=NonNegativeReals,
        bounds=(0, m.flow_max),
        initialize=m.V0,
    )
    m.L = Var(
        m.section,
        m.tray,
        m.comp,
        doc="Liquid flowrate [mol/s]",
        domain=NonNegativeReals,
        bounds=(0, m.flow_max),
        initialize=m.L0,
    )
    m.Vtotal = Var(
        m.section,
        m.tray,
        doc="Total vapor flowrate [mol/s]",
        domain=NonNegativeReals,
        bounds=(0, m.flow_max),
        initialize=m.Vtotal0,
    )
    m.Ltotal = Var(
        m.section,
        m.tray,
        doc="Total liquid flowrate [mol/s]",
        domain=NonNegativeReals,
        bounds=(0, m.flow_max),
        initialize=m.Ltotal0,
    )

    m.D = Var(
        m.comp,
        doc="Distillate flowrate [mol/s]",
        domain=NonNegativeReals,
        bounds=(0, m.flow_max),
        initialize=m.Ddes,
    )
    m.B = Var(
        m.comp,
        doc="Bottoms flowrate [mol/s]",
        domain=NonNegativeReals,
        bounds=(0, m.flow_max),
        initialize=m.Bdes,
    )
    m.S = Var(
        m.so,
        m.comp,
        doc="Product 2 and 3 flowrates [mol/s]",
        domain=NonNegativeReals,
        bounds=(0, m.flow_max),
        initialize=m.Sdes,
    )
    m.Dtotal = Var(
        doc="Distillate flowrate [mol/s]",
        domain=NonNegativeReals,
        bounds=(0, m.flow_max),
        initialize=m.Ddes,
    )
    m.Btotal = Var(
        doc="Bottoms flowrate [mol/s]",
        domain=NonNegativeReals,
        bounds=(0, m.flow_max),
        initialize=m.Bdes,
    )
    m.Stotal = Var(
        m.so,
        doc="Total product 2 and 3 side flowrate [mol/s]",
        domain=NonNegativeReals,
        bounds=(0, m.flow_max),
        initialize=m.Sdes,
    )

    m.hl = Var(
        m.section,
        m.tray,
        m.comp,
        doc='Liquid enthalpy [J/mol]',
        bounds=(m.hllo, m.hlup),
        initialize=m.hl0,
    )
    m.hv = Var(
        m.section,
        m.tray,
        m.comp,
        doc='Vapor enthalpy [J/mol]',
        bounds=(m.hvlo, m.hvup),
        initialize=m.hv0,
    )
    m.Qreb = Var(
        doc="Reboiler heat duty [J/s]",
        domain=NonNegativeReals,
        bounds=(0, m.Qmax),
        initialize=1,
    )
    m.Qcon = Var(
        doc="Condenser heat duty [J/s]",
        domain=NonNegativeReals,
        bounds=(0, m.Qmax),
        initialize=1,
    )

    m.rr = Var(
        doc="Internal reflux ratio in the column",
        domain=NonNegativeReals,
        bounds=(0.7, 1),
        initialize=m.rr0,
    )
    m.bu = Var(
        doc="Boilup rate in the reboiler",
        domain=NonNegativeReals,
        bounds=(0.7, 1),
        initialize=m.bu0,
    )

    m.F = Var(
        m.comp,
        doc="Side feed flowrate [mol/s]",
        domain=NonNegativeReals,
        bounds=(0, 50),
        initialize=m.F0,
    )
    m.q = Var(
        doc="Vapor fraction in side feed",
        domain=NonNegativeReals,
        bounds=(0, 1),
        initialize=1,
    )

    m.actv = Var(
        m.section,
        m.tray,
        m.comp,
        doc="Liquid activity coefficient",
        domain=NonNegativeReals,
        bounds=(0, 10),
        initialize=m.actv0,
    )

    m.errx = Var(
        m.section,
        m.tray,
        doc="Error in liquid composition [mol/mol]",
        bounds=(-1e-3, 1e-3),
        initialize=0,
    )
    m.erry = Var(
        m.section,
        m.tray,
        doc="Error in vapor composition [mol/mol]",
        bounds=(-1e-3, 1e-3),
        initialize=0,
    )
    m.slack = Var(
        m.section,
        m.tray,
        m.comp,
        doc="Slack variable",
        bounds=(-1e-8, 1e-8),
        initialize=0,
    )

    m.tray_exists = Disjunct(
        m.section,
        m.tray,
        doc="Disjunct that enforce the existence of each tray",
        rule=_build_tray_equations,
    )
    m.tray_absent = Disjunct(
        m.section,
        m.tray,
        doc="Disjunct that enforce the absence of each tray",
        rule=_build_pass_through_eqns,
    )

    @m.Disjunction(
        m.section, m.tray, doc="Disjunction between whether each tray exists or not"
    )
    def tray_exists_or_not(m, sec, n_tray):
        """
        Disjunction between whether each tray exists or not.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The Pyomo model object.
        sec : int
            The section index.
        n_tray : int
            The tray index.

        Returns
        -------
        Disjunction
            The disjunction between whether each tray exists or not.
        """
        return [m.tray_exists[sec, n_tray], m.tray_absent[sec, n_tray]]

    @m.Constraint(m.section_main)
    def minimum_trays_main(m, sec):
        """
        Constraint that ensures the minimum number of trays in the main section.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The model object for the GDP Kaibel Column.
        sec : Set
            The section index.

        Returns
        -------
        Constraint
            A constraint expression that enforces the minimum number of trays in the main section to be greater than or equal to the minimum number of trays.
        """
        return (
            sum(
                m.tray_exists[sec, n_tray].binary_indicator_var
                for n_tray in m.candidate_trays_main
            )
            + 1
            >= m.min_num_trays
        )

    @m.Constraint()
    def minimum_trays_feed(m):
        """
        Constraint function that ensures the minimum number of trays in the feed section is met.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The Pyomo model object.

        Returns
        -------
        Constraint
            The constraint expression that enforces the minimum number of trays is greater than or equal to the minimum number of trays.
        """
        return (
            sum(
                m.tray_exists[2, n_tray].binary_indicator_var
                for n_tray in m.candidate_trays_feed
            )
            + 1
            >= m.min_num_trays
        )

    # TOCHECK: pyomo.GDP Syntax

    @m.Constraint()
    def minimum_trays_product(m):
        """
        Constraint function to calculate the minimum number of trays in the product section.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The constraint expression that enforces the minimum number of trays is greater than or equal to the minimum number of trays.
        """
        return (
            sum(
                m.tray_exists[3, n_tray].binary_indicator_var
                for n_tray in m.candidate_trays_product
            )
            + 1
            >= m.min_num_trays
        )

    ## Fixed trays
    enforce_tray_exists(m, 1, 1)  # reboiler
    enforce_tray_exists(m, 1, m.num_trays)  # vapor distributor
    enforce_tray_exists(m, 2, 1)  # dividing wall starting tray
    enforce_tray_exists(m, 2, m.feed_tray)  # feed tray
    enforce_tray_exists(m, 2, m.num_trays)  # dividing wall ending tray
    enforce_tray_exists(m, 3, 1)  # dividing wall starting tray
    enforce_tray_exists(m, 3, m.sideout1_tray)  # side outlet 1 for product 3
    enforce_tray_exists(m, 3, m.sideout2_tray)  # side outlet 2 for product 2
    enforce_tray_exists(m, 3, m.num_trays)  # dividing wall ending tray
    enforce_tray_exists(m, 4, 1)  # liquid distributor
    enforce_tray_exists(m, 4, m.num_trays)  # condenser

    #### Global constraints
    @m.Constraint(
        m.dw, m.tray, doc="Monotonic temperature in the dividing wall sections"
    )
    def monotonic_temperature(m, sec, n_tray):
        """This function returns a constraint object representing the monotonic temperature constraint.

        The monotonic temperature constraint ensures that the temperature on each tray in the distillation column
        is less than or equal to the temperature on the top tray.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The Pyomo model object.
        sec : Set
            The set of sections in the dividing wall sections.
        n_tray : Set
            The set of trays in the distillation column.

        Returns
        -------
        Constraint
            The monotonic temperature constraint specifying that the temperature on each tray is less than or equal to the temperature on the top tray of section 1 which is the condenser.
        """
        return m.T[sec, n_tray] <= m.T[1, m.num_trays]

    @m.Constraint(doc="Liquid distributor")
    def liquid_distributor(m):
        """Defines the liquid distributor constraint.

        This constraint ensures that the sum of the liquid distributors in all sections is equal to 1.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The liquid distributor constraint that enforces the sum of the liquid flow rates in all sections is equal to 1.

        """
        return sum(m.dl[sec] for sec in m.dw) - 1 == 0

    @m.Constraint(doc="Vapor distributor")
    def vapor_distributor(m):
        """
        Add a constraint to ensure that the sum of the vapor distributors is equal to 1.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The Pyomo model object.

        Returns
        -------
        Constraint
            The vapor distributor constraint.
        """
        return sum(m.dv[sec] for sec in m.dw) - 1 == 0

    @m.Constraint(doc="Reboiler composition specification")
    def heavy_product(m):
        """
        Reboiler composition specification for the heavy component in the feed.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The constraint that enforces the reboiler composition is greater than or equal to the specified composition xspechc final liquid composition for butanol, the heavy component in the feed.
        """
        return m.x[1, m.reb_tray, m.hc] >= m.xspec_hc

    @m.Constraint(doc="Condenser composition specification")
    def light_product(m):
        """
        Condenser composition specification for the light component in the feed.

        Parameters
        ----------
        m : Model
            The optimization model.

        Returns
        -------
        Constraint
            The constraint that enforces the condenser composition is greater than or equal to the specified final liquid composition for ethanol, xspeclc , the light component in the feed.
        """
        return m.x[4, m.con_tray, m.lc] >= m.xspec_lc

    @m.Constraint(doc="Side outlet 1 final liquid composition")
    def intermediate1_product(m):
        '''
        This constraint ensures that the intermediate 1 final liquid composition is greater than or equal to the specified composition xspec_inter1, which is the final liquid composition for ethanol.

        Parameters
        ----------
        m : Pyomo ConcreteModel.
            The optimization model.

        Returns
        -------
        Constraint
            The constraint that enforces the intermediate 1 final liquid composition is greater than or equal to the specified composition xspec_inter1, which is the final liquid composition for ethanol.
        '''
        return m.x[3, m.sideout1_tray, 3] >= m.xspec_inter3

    @m.Constraint(doc="Side outlet 2 final liquid composition")
    def intermediate2_product(m):
        """
        This constraint ensures that the intermediate 2 final liquid composition is greater than or equal to the specified composition xspec_inter2, which is the final liquid composition for butanol.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The constraint that enforces the intermediate 2 final liquid composition is greater than or equal to the specified composition xspec_inter2, which is the final liquid composition for butanol.
        """
        return m.x[3, m.sideout2_tray, 2] >= m.xspec_inter2

    @m.Constraint(doc="Reboiler flowrate")
    def _heavy_product_flow(m):
        """
        Reboiler flowrate constraint that ensures the reboiler flowrate is greater than or equal to the specified flowrate Bdes, which is the flowrate of butanol.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The constraint that enforces the reboiler flowrate is greater than or equal to the specified flowrate Bdes, which is the flowrate of butanol.
        """
        return m.Btotal >= m.Bdes

    @m.Constraint(doc="Condenser flowrate")
    def _light_product_flow(m):
        """
        Condenser flowrate constraint that ensures the condenser flowrate is greater than or equal to the specified flowrate Ddes, which is the flowrate of ethanol.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The constraint that enforces the condenser flowrate is greater than or equal to the specified flowrate Ddes, which is the flowrate of ethanol.
        """
        return m.Dtotal >= m.Ddes

    @m.Constraint(m.so, doc="Intermediate flowrate")
    def _intermediate_product_flow(m, so):
        """
        Intermediate flowrate constraint that ensures the intermediate flowrate is greater than or equal to the specified flowrate Sdes, which is the flowrate of the intermediate side product 2 and 3.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.
        so : int
            The side product outlet index.

        Returns
        -------
        Constraint
            The constraint that enforces the intermediate flowrate is greater than or equal to the specified flowrate Sdes, which is the flowrate of the intermediate side product 2 and 3.
        """
        return m.Stotal[so] >= m.Sdes

    @m.Constraint(doc="Internal boilup ratio, V/L")
    def _internal_boilup_ratio(m):
        """
        Internal boilup ratio constraint that ensures the internal boilup ratio is equal to the boilup rate times the liquid flowrate on the reboiler tray is equal to the vapor flowrate on the tray above the reboiler tray.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The constraint that enforces the boilup rate times the liquid flowrate on the reboiler tray is equal to the vapor flowrate on the tray above the reboiler tray.
        """
        return m.bu * m.Ltotal[1, m.reb_tray + 1] == m.Vtotal[1, m.reb_tray]

    @m.Constraint(doc="Internal reflux ratio, L/V")
    def internal_reflux_ratio(m):
        """
        Internal reflux ratio constraint that ensures the internal reflux ratio is equal to the reflux rate times the vapor flowrate on the tray above the condenser tray is equal to the liquid flowrate on the condenser tray.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The constraint that enforces the reflux rate times the vapor flowrate on the tray above the condenser tray is equal to the liquid flowrate on the condenser tray.
        """
        return m.rr * m.Vtotal[4, m.con_tray - 1] == m.Ltotal[4, m.con_tray]

    @m.Constraint(doc="External boilup ratio relation with bottoms")
    def _external_boilup_ratio(m):
        """
        External boilup ratio constraint that ensures the external boilup ratio times the liquid flowrate on the reboiler tray is equal to the bottoms flowrate.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The constraint that enforces the external boilup ratio times the liquid flowrate on the reboiler tray is equal to the bottoms flowrate.
        """
        return m.Btotal == (1 - m.bu) * m.Ltotal[1, m.reb_tray + 1]

    @m.Constraint(doc="External reflux ratio relation with distillate")
    def _external_reflux_ratio(m):
        """
        External reflux ratio constraint that ensures the external reflux ratio times the vapor flowrate on the tray above the condenser tray is equal to the distillate flowrate.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The constraint that enforces the external reflux ratio times the vapor flowrate on the tray above the condenser tray is equal to the distillate flowrate.
        """
        return m.Dtotal == (1 - m.rr) * m.Vtotal[4, m.con_tray - 1]

    @m.Constraint(m.section, m.tray, doc="Total vapor flowrate")
    def _total_vapor_flowrate(m, sec, n_tray):
        """
        Constraint that ensures the total vapor flowrate is equal to the sum of the vapor flowrates of each component on each tray.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.
        sec : int
            The section index.
        n_tray : int
            The tray index.

        Returns
        -------
        Constraint
            The constraint that enforces the total vapor flowrate is equal to the sum of the vapor flowrates of each component on each tray on each section.
        """
        return sum(m.V[sec, n_tray, comp] for comp in m.comp) == m.Vtotal[sec, n_tray]

    @m.Constraint(m.section, m.tray, doc="Total liquid flowrate")
    def _total_liquid_flowrate(m, sec, n_tray):
        """
        Constraint that ensures the total liquid flowrate is equal to the sum of the liquid flowrates of each component on each tray on each section.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.
        sec : int
            The section index.
        n_tray : int
            The tray index.

        Returns
        -------
        Constraint
            The constraint that enforces the total liquid flowrate is equal to the sum of the liquid flowrates of each component on each tray on each section.
        """
        return sum(m.L[sec, n_tray, comp] for comp in m.comp) == m.Ltotal[sec, n_tray]

    @m.Constraint(m.comp, doc="Bottoms and liquid relation")
    def bottoms_equality(m, comp):
        """
        Constraint that ensures the bottoms flowrate is equal to the liquid flowrate of each component on the reboiler tray.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint that enforces the bottoms flowrate is equal to the liquid flowrate of each component on the reboiler tray.
        """
        return m.B[comp] == m.L[1, m.reb_tray, comp]

    @m.Constraint(m.comp)
    def condenser_total(m, comp):
        """
        Constraint that ensures the distillate flowrate in the condenser is null.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.
        comp : int
            The component index.
        Returns
        -------
        Constraint
            The constraint that enforces the distillate flowrate is equal to zero in the condenser.
        """
        return m.V[4, m.con_tray, comp] == 0

    @m.Constraint()
    def total_bottoms_product(m):
        """
        Constraint that ensures the total bottoms flowrate is equal to the sum of the bottoms flowrates of each component.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The constraint that enforces the total bottoms flowrate is equal to the sum of the bottoms flowrates of each component.
        """
        return sum(m.B[comp] for comp in m.comp) == m.Btotal

    @m.Constraint()
    def total_distillate_product(m):
        """
        Constraint that ensures the total distillate flowrate is equal to the sum of the distillate flowrates of each component.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The constraint that enforces the total distillate flowrate is equal to the sum of the distillate flowrates of each component.
        """
        return sum(m.D[comp] for comp in m.comp) == m.Dtotal

    @m.Constraint(m.so)
    def total_side_product(m, so):
        """
        Constraint that ensures the total side product flowrate is equal to the sum of the side product flowrates of each component.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.
        so : int
            The side product index, 2 or 3 for the intermediate side products.

        Returns
        -------
        Constraint
            The constraint that enforces the total side product flowrate is equal to the sum of the side product flowrates of each component.
        """
        return sum(m.S[so, comp] for comp in m.comp) == m.Stotal[so]

    # Considers the number of existent trays and operating costs (condenser and reboiler heat duties) in the column. To ensure equal weights to the capital and operating costs, the number of existent trays is multiplied by a weight coefficient of 1000.

    m.obj = Objective(
        expr=(m.Qcon + m.Qreb) * m.Hscale
        + 1e3
        * (
            sum(
                sum(
                    m.tray_exists[sec, n_tray].binary_indicator_var
                    for n_tray in m.candidate_trays_main
                )
                for sec in m.section_main
            )
            + sum(
                m.tray_exists[2, n_tray].binary_indicator_var
                for n_tray in m.candidate_trays_feed
            )
            + sum(
                m.tray_exists[3, n_tray].binary_indicator_var
                for n_tray in m.candidate_trays_product
            )
            + 1
        ),
        sense=minimize,
        doc="Objective function to minimize the operating costs and number of existent trays in the column",
    )

    @m.Constraint(
        m.section_main, m.candidate_trays_main, doc="Logic proposition for main section"
    )
    def _logic_proposition_main(m, sec, n_tray):
        """
        Apply a logic proposition constraint to the main section and candidate trays to specify the order of trays in the column is from bottom to top provided the condition is met.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.
        sec : int
            The section index.
        n_tray : int
            The tray index.

        Returns
        -------
        Constraint or NoConstraint
            The constraint expression or NoConstraint if the condition is not met.
        """

        if n_tray > m.reb_tray and (n_tray + 1) < m.num_trays:
            return (
                m.tray_exists[sec, n_tray].binary_indicator_var
                <= m.tray_exists[sec, n_tray + 1].binary_indicator_var
            )
        else:
            return Constraint.NoConstraint
        # TOCHECK: Update the logic proposition constraint for the main section with the new pyomo.gdp syntax

    @m.Constraint(m.candidate_trays_feed)
    def _logic_proposition_feed(m, n_tray):
        """
        Apply a logic proposition constraint to the feed section and candidate trays to specify the order of trays in the column is from bottom to top provided the condition is met.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.
        n_tray : int
        The tray index.

        Returns
        -------
        Constraint or NoConstraint
            The constraint expression or NoConstraint if the condition is not met.
        """
        if n_tray > m.reb_tray and (n_tray + 1) < m.feed_tray:
            return (
                m.tray_exists[2, n_tray].binary_indicator_var
                <= m.tray_exists[2, n_tray + 1].binary_indicator_var
            )
        elif n_tray > m.feed_tray and (n_tray + 1) < m.con_tray:
            return (
                m.tray_exists[2, n_tray + 1].binary_indicator_var
                <= m.tray_exists[2, n_tray].binary_indicator_var
            )
        else:
            return Constraint.NoConstraint
        # TODO: Update the logic proposition constraint for the feed section with the new pyomo.gdp syntax

    @m.Constraint(m.candidate_trays_product)
    def _logic_proposition_section3(m, n_tray):
        """
        Apply a logic proposition constraint to the product section and candidate trays to specify the order of trays in the column is from bottom to top provided the condition is met.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.
        n_tray : int
            The tray index.

        Returns
        -------
        Constraint or NoConstraint
            The constraint expression or NoConstraint if the condition is not met.
        """
        if n_tray > 1 and (n_tray + 1) < m.num_trays:
            return (
                m.tray_exists[3, n_tray].binary_indicator_var
                <= m.tray_exists[3, n_tray + 1].binary_indicator_var
            )
        else:
            return Constraint.NoConstraint
        # TODO: Update the logic proposition constraint for the product section with the new pyomo.gdp syntax

    @m.Constraint(m.tray)
    def equality_feed_product_side(m, n_tray):
        """
        Constraint that enforces the equality of the binary indicator variables for the feed and product side trays.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.
        n_tray : int
            The tray index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the equality of the binary indicator variables for the feed and product side trays.
        """
        return (
            m.tray_exists[2, n_tray].binary_indicator_var
            == m.tray_exists[3, n_tray].binary_indicator_var
        )

    # TODO: Update the equality constraint for the feed and product side trays with the new pyomo.gdp syntax

    @m.Constraint()
    def _existent_minimum_numbertrays(m):
        """
        Constraint that enforces the minimum number of trays in the column.

        Parameters
        ----------
        m : Pyomo ConcreteModel
            The optimization model.

        Returns
        -------
        Constraint
            The constraint expression that enforces the minimum number of trays in each section and each tray to be greater than or equal to the minimum number of trays.
        """
        return sum(
            sum(m.tray_exists[sec, n_tray].binary_indicator_var for n_tray in m.tray)
            for sec in m.section
        ) - sum(
            m.tray_exists[3, n_tray].binary_indicator_var for n_tray in m.tray
        ) >= int(
            m.min_tray
        )

    return m


def enforce_tray_exists(m, sec, n_tray):
    """
    Enforce the existence of a tray in the column.

    Parameters
    ----------
    m : Pyomo ConcreteModel
        The optimization model.
    sec : int
        The section index.
    n_tray : int
        The tray index.
    """
    m.tray_exists[sec, n_tray].indicator_var.fix(True)
    m.tray_absent[sec, n_tray].deactivate()


def _build_tray_equations(m, sec, n_tray):
    """
    Build the equations for the tray in the column as a function of the section when the tray exists.
    Points to the appropriate function to build the equations for the section in the column.

    Parameters
    ----------
    m : Pyomo ConcreteModel
        The optimization model.
    sec : int
        The section index.
    n_tray : int
        The tray index.

    Returns
    -------
    None
    """
    build_function = {
        1: _build_bottom_equations,
        2: _build_feed_side_equations,
        3: _build_product_side_equations,
        4: _build_top_equations,
    }
    build_function[sec](m, n_tray)


def _build_bottom_equations(disj, n_tray):
    """
    Build the equations for the bottom section in the column.

    Parameters
    ----------
    disj : Disjunct
        The disjunct object for the bottom section in the column.
    n_tray : int
        The tray index.

    Returns
    -------
    None
    """
    m = disj.model()

    @disj.Constraint(m.comp, doc="Bottom section 1 mass per component balances")
    def _bottom_mass_percomponent_balances(disj, comp):
        """
        Mass per component balances for the bottom section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the bottom section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the mass balance per component in the bottom section of the column.
        """
        return (m.L[1, n_tray + 1, comp] if n_tray < m.num_trays else 0) + (
            m.L[2, 1, comp] if n_tray == m.num_trays else 0
        ) + (m.L[3, 1, comp] if n_tray == m.num_trays else 0) - (
            m.L[1, n_tray, comp] if n_tray > m.reb_tray else 0
        ) + (
            m.V[1, n_tray - 1, comp] if n_tray > m.reb_tray else 0
        ) - (
            m.V[1, n_tray, comp] * m.dv[2] if n_tray == m.num_trays else 0
        ) - (
            m.V[1, n_tray, comp] * m.dv[3] if n_tray == m.num_trays else 0
        ) - (
            m.V[1, n_tray, comp] if n_tray < m.num_trays else 0
        ) - (
            m.B[comp] if n_tray == m.reb_tray else 0
        ) == m.slack[
            1, n_tray, comp
        ]

    @disj.Constraint(doc="Bottom section 1 energy balances")
    def _bottom_energy_balances(disj):
        """
        Energy balances for the bottom section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the bottom section in the column.

        Returns
        -------
        Constraint
            The constraint expression that enforces the energy balance for the bottom section in the column.
        """
        return (
            sum(
                (
                    m.L[1, n_tray + 1, comp] * m.hl[1, n_tray + 1, comp]
                    if n_tray < m.num_trays
                    else 0
                )
                + (m.L[2, 1, comp] * m.hl[2, 1, comp] if n_tray == m.num_trays else 0)
                + (m.L[3, 1, comp] * m.hl[3, 1, comp] if n_tray == m.num_trays else 0)
                - (
                    m.L[1, n_tray, comp] * m.hl[1, n_tray, comp]
                    if n_tray > m.reb_tray
                    else 0
                )
                + (
                    m.V[1, n_tray - 1, comp] * m.hv[1, n_tray - 1, comp]
                    if n_tray > m.reb_tray
                    else 0
                )
                - (
                    m.V[1, n_tray, comp] * m.dv[2] * m.hv[1, n_tray, comp]
                    if n_tray == m.num_trays
                    else 0
                )
                - (
                    m.V[1, n_tray, comp] * m.dv[3] * m.hv[1, n_tray, comp]
                    if n_tray == m.num_trays
                    else 0
                )
                - (
                    m.V[1, n_tray, comp] * m.hv[1, n_tray, comp]
                    if n_tray < m.num_trays
                    else 0
                )
                - (m.B[comp] * m.hl[1, n_tray, comp] if n_tray == m.reb_tray else 0)
                for comp in m.comp
            )
            * m.Qscale
            + (m.Qreb if n_tray == m.reb_tray else 0)
            == 0
        )

    @disj.Constraint(m.comp, doc="Bottom section 1 liquid flowrate per component")
    def _bottom_liquid_percomponent(disj, comp):
        """
        Liquid flowrate per component in the bottom section of the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the bottom section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid flowrate per component in the bottom section of the column.
        """
        return m.L[1, n_tray, comp] == m.Ltotal[1, n_tray] * m.x[1, n_tray, comp]

    @disj.Constraint(m.comp, doc="Bottom section 1 vapor flowrate per component")
    def _bottom_vapor_percomponent(disj, comp):
        """
        Vapor flowrate per component in the bottom section of the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the bottom section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor flowrate per component in the bottom section of the column.
        """
        return m.V[1, n_tray, comp] == m.Vtotal[1, n_tray] * m.y[1, n_tray, comp]

    @disj.Constraint(doc="Bottom section 1 liquid composition equilibrium summation")
    def bottom_liquid_composition_summation(disj):
        """
        Liquid composition equilibrium summation for the bottom section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the bottom section in the column.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid composition equilibrium summation for the bottom section in the column.
            It ensures the sum of the liquid compositions is equal to 1 plus the error in the liquid composition.
        """
        return sum(m.x[1, n_tray, comp] for comp in m.comp) - 1 == m.errx[1, n_tray]

    @disj.Constraint(doc="Bottom section 1 vapor composition equilibrium summation")
    def bottom_vapor_composition_summation(disj):
        """
        Vapor composition equilibrium summation for the bottom section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the bottom section in the column.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor composition equilibrium summation for the bottom section in the column.
            It ensures the sum of the vapor compositions is equal to 1 plus the error in the vapor composition.
        """
        return sum(m.y[1, n_tray, comp] for comp in m.comp) - 1 == m.erry[1, n_tray]

    @disj.Constraint(m.comp, doc="Bottom section 1 vapor composition")
    def bottom_vapor_composition(disj, comp):
        """
        Vapor composition for the bottom section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the bottom section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor composition for the bottom section in the column.
            The equation is derived from the vapor-liquid equilibrium relationship.
        """
        return (
            m.y[1, n_tray, comp]
            == m.x[1, n_tray, comp]
            * (
                m.actv[1, n_tray, comp]
                * (
                    m.prop[comp, 'PC']
                    * exp(
                        m.prop[comp, 'TC']
                        / m.T[1, n_tray]
                        * (
                            m.prop[comp, 'vpA']
                            * (1 - m.T[1, n_tray] / m.prop[comp, 'TC'])
                            + m.prop[comp, 'vpB']
                            * (1 - m.T[1, n_tray] / m.prop[comp, 'TC']) ** 1.5
                            + m.prop[comp, 'vpC']
                            * (1 - m.T[1, n_tray] / m.prop[comp, 'TC']) ** 3
                            + m.prop[comp, 'vpD']
                            * (1 - m.T[1, n_tray] / m.prop[comp, 'TC']) ** 6
                        )
                    )
                )
            )
            / m.P[1, n_tray]
        )

    @disj.Constraint(m.comp, doc="Bottom section 1 liquid enthalpy")
    def bottom_liquid_enthalpy(disj, comp):
        """
        Liquid enthalpy for the bottom section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the bottom section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid enthalpy for the bottom section in the column.
        """
        return m.hl[1, n_tray, comp] == (
            m.cpc[1]
            * (
                (m.T[1, n_tray] - m.Tref) * m.prop[comp, 'cpA', 1]
                + (m.T[1, n_tray] ** 2 - m.Tref**2)
                * m.prop[comp, 'cpB', 1]
                * m.cpc2['A', 1]
                / 2
                + (m.T[1, n_tray] ** 3 - m.Tref**3)
                * m.prop[comp, 'cpC', 1]
                * m.cpc2['B', 1]
                / 3
                + (m.T[1, n_tray] ** 4 - m.Tref**4) * m.prop[comp, 'cpD', 1] / 4
            )
            / m.Hscale
        )

    @disj.Constraint(m.comp, doc="Bottom section 1 vapor enthalpy")
    def bottom_vapor_enthalpy(disj, comp):
        """
        Vapor enthalpy for the bottom section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the bottom section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor enthalpy for the bottom section in the column.
        """
        return m.hv[1, n_tray, comp] == (
            m.cpc[2]
            * (
                (m.T[1, n_tray] - m.Tref) * m.prop[comp, 'cpA', 2]
                + (m.T[1, n_tray] ** 2 - m.Tref**2)
                * m.prop[comp, 'cpB', 2]
                * m.cpc2['A', 2]
                / 2
                + (m.T[1, n_tray] ** 3 - m.Tref**3)
                * m.prop[comp, 'cpC', 2]
                * m.cpc2['B', 2]
                / 3
                + (m.T[1, n_tray] ** 4 - m.Tref**4) * m.prop[comp, 'cpD', 2] / 4
            )
            / m.Hscale
            + m.dHvap[comp]
        )

    @disj.Constraint(m.comp, doc="Bottom section 1 liquid activity coefficient")
    def bottom_activity_coefficient(disj, comp):
        """
        Liquid activity coefficient for the bottom section in the column equal to 1.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the bottom section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid activity coefficient for the bottom section for each component and tray in the column to be equal to 1.
        """
        return m.actv[1, n_tray, comp] == 1


def _build_feed_side_equations(disj, n_tray):
    """
    Build the equations for the feed side section in the column.

    Parameters
    ----------
    disj : Disjunct
        The disjunct object for the feed side section in the column.
    n_tray : int
        The tray index.

    Returns
    -------
    None
    """
    m = disj.model()

    @disj.Constraint(m.comp, doc="Feed section 2 mass per component balances")
    def _feedside_masspercomponent_balances(disj, comp):
        """
        Mass per component balances for the feed side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the feed side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the mass balance per component in the feed side section of the column.
        """
        return (m.L[2, n_tray + 1, comp] if n_tray < m.num_trays else 0) + (
            m.L[4, 1, comp] * m.dl[2] if n_tray == m.num_trays else 0
        ) - m.L[2, n_tray, comp] + (
            m.V[1, m.num_trays, comp] * m.dv[2] if n_tray == 1 else 0
        ) + (
            m.V[2, n_tray - 1, comp] if n_tray > 1 else 0
        ) - m.V[
            2, n_tray, comp
        ] + (
            m.F[comp] if n_tray == m.feed_tray else 0
        ) == 0

    @disj.Constraint(doc="Feed section 2 energy balances")
    def _feedside_energy_balances(disj):
        """
        Energy balances for the feed side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the feed side section in the column.

        Returns
        -------
        Constraint
            The constraint expression that enforces the energy balance for the feed side section in the column.
        """
        return (
            sum(
                (
                    m.L[2, n_tray + 1, comp] * m.hl[2, n_tray + 1, comp]
                    if n_tray < m.num_trays
                    else 0
                )
                + (
                    m.L[4, 1, comp] * m.dl[2] * m.hl[4, 1, comp]
                    if n_tray == m.num_trays
                    else 0
                )
                - m.L[2, n_tray, comp] * m.hl[2, n_tray, comp]
                + (
                    m.V[1, m.num_trays, comp] * m.dv[2] * m.hv[1, m.num_trays, comp]
                    if n_tray == 1
                    else 0
                )
                + (
                    m.V[2, n_tray - 1, comp] * m.hv[2, n_tray - 1, comp]
                    if n_tray > 1
                    else 0
                )
                - m.V[2, n_tray, comp] * m.hv[2, n_tray, comp]
                for comp in m.comp
            )
            * m.Qscale
            + sum(
                (
                    m.F[comp] * (m.hlf[comp] * (1 - m.q) + m.hvf[comp] * m.q)
                    if n_tray == m.feed_tray
                    else 0
                )
                for comp in m.comp
            )
            * m.Qscale
            == 0
        )

    @disj.Constraint(m.comp, doc="Feed section 2 liquid flowrate per component")
    def _feedside_liquid_percomponent(disj, comp):
        """
        Liquid flowrate per component in the feed side section of the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the feed side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid flowrate per component in the feed side section of the column is equal to the total liquid flowrate times the liquid composition.
        """
        return m.L[2, n_tray, comp] == m.Ltotal[2, n_tray] * m.x[2, n_tray, comp]

    @disj.Constraint(m.comp, doc="Feed section 2 vapor flowrate per component")
    def _feedside_vapor_percomponent(disj, comp):
        """
        Vapor flowrate per component in the feed side section of the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the feed side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor flowrate per component in the feed side section of the column is equal to the total vapor flowrate times the vapor composition.
        """
        return m.V[2, n_tray, comp] == m.Vtotal[2, n_tray] * m.y[2, n_tray, comp]

    @disj.Constraint(doc="Feed section 2 liquid composition equilibrium summation")
    def feedside_liquid_composition_summation(disj):
        """
        Liquid composition equilibrium summation for the feed side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the feed side section in the column.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid composition equilibrium summation for the feed side section in the column.
            It ensures the sum of the liquid compositions is equal to 1 plus the error in the liquid composition.
        """
        return sum(m.x[2, n_tray, comp] for comp in m.comp) - 1 == m.errx[2, n_tray]

    @disj.Constraint(doc="Feed section 2 vapor composition equilibrium summation")
    def feedside_vapor_composition_summation(disj):
        """
        Vapor composition equilibrium summation for the feed side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the feed side section in the column.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor composition equilibrium summation for the feed side section in the column.
            It ensures the sum of the vapor compositions is equal to 1 plus the error in the vapor composition.
        """
        return sum(m.y[2, n_tray, comp] for comp in m.comp) - 1 == m.erry[2, n_tray]

    @disj.Constraint(m.comp, doc="Feed section 2 vapor composition")
    def feedside_vapor_composition(disj, comp):
        """
        Vapor composition for the feed side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the feed side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor composition for the feed side section in the column.
            The equation is derived from the vapor-liquid equilibrium relationship.
        """
        return (
            m.y[2, n_tray, comp]
            == m.x[2, n_tray, comp]
            * (
                m.actv[2, n_tray, comp]
                * (
                    m.prop[comp, 'PC']
                    * exp(
                        m.prop[comp, 'TC']
                        / m.T[2, n_tray]
                        * (
                            m.prop[comp, 'vpA']
                            * (1 - m.T[2, n_tray] / m.prop[comp, 'TC'])
                            + m.prop[comp, 'vpB']
                            * (1 - m.T[2, n_tray] / m.prop[comp, 'TC']) ** 1.5
                            + m.prop[comp, 'vpC']
                            * (1 - m.T[2, n_tray] / m.prop[comp, 'TC']) ** 3
                            + m.prop[comp, 'vpD']
                            * (1 - m.T[2, n_tray] / m.prop[comp, 'TC']) ** 6
                        )
                    )
                )
            )
            / m.P[2, n_tray]
        )

    @disj.Constraint(m.comp, doc="Feed section 2 liquid enthalpy")
    def feedside_liquid_enthalpy(disj, comp):
        """
        Liquid enthalpy for the feed side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the feed side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid enthalpy for the feed side section in the column.
        """
        return m.hl[2, n_tray, comp] == (
            m.cpc[1]
            * (
                (m.T[2, n_tray] - m.Tref) * m.prop[comp, 'cpA', 1]
                + (m.T[2, n_tray] ** 2 - m.Tref**2)
                * m.prop[comp, 'cpB', 1]
                * m.cpc2['A', 1]
                / 2
                + (m.T[2, n_tray] ** 3 - m.Tref**3)
                * m.prop[comp, 'cpC', 1]
                * m.cpc2['B', 1]
                / 3
                + (m.T[2, n_tray] ** 4 - m.Tref**4) * m.prop[comp, 'cpD', 1] / 4
            )
            / m.Hscale
        )

    @disj.Constraint(m.comp, doc="Feed section 2 vapor enthalpy")
    def feedside_vapor_enthalpy(disj, comp):
        """
        Vapor enthalpy for the feed side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the feed side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor enthalpy for the feed side section in the column.
        """
        return m.hv[2, n_tray, comp] == (
            m.cpc[2]
            * (
                (m.T[2, n_tray] - m.Tref) * m.prop[comp, 'cpA', 2]
                + (m.T[2, n_tray] ** 2 - m.Tref**2)
                * m.prop[comp, 'cpB', 2]
                * m.cpc2['A', 2]
                / 2
                + (m.T[2, n_tray] ** 3 - m.Tref**3)
                * m.prop[comp, 'cpC', 2]
                * m.cpc2['B', 2]
                / 3
                + (m.T[2, n_tray] ** 4 - m.Tref**4) * m.prop[comp, 'cpD', 2] / 4
            )
            / m.Hscale
            + m.dHvap[comp]
        )

    @disj.Constraint(m.comp, doc="Feed section 2 liquid activity coefficient")
    def feedside_activity_coefficient(disj, comp):
        """
        Liquid activity coefficient for the feed side section in the column equal to 1.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the feed side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid activity coefficient for the feed side section for each component and tray in the column to be equal to 1.
            This is an assumption for the feed side section, since the feed is assumed to be ideal.
        """
        return m.actv[2, n_tray, comp] == 1


def _build_product_side_equations(disj, n_tray):
    """
    Build the equations for the product side section in the column.

    Parameters
    ----------
    disj : Disjunct
        The disjunct object for the product side section in the column.
    n_tray : int
        The tray index.

    Returns
    -------
    None
    """
    m = disj.model()

    @disj.Constraint(m.comp, doc="Product section 3 mass per component balances")
    def _productside_masspercomponent_balances(disj, comp):
        """
        Mass per component balances for the product side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the product side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the mass balance per component in the product side section of the column.
        """
        return (m.L[3, n_tray + 1, comp] if n_tray < m.num_trays else 0) + (
            m.L[4, 1, comp] * m.dl[3] if n_tray == m.num_trays else 0
        ) - m.L[3, n_tray, comp] + (
            m.V[1, m.num_trays, comp] * m.dv[3] if n_tray == 1 else 0
        ) + (
            m.V[3, n_tray - 1, comp] if n_tray > 1 else 0
        ) - m.V[
            3, n_tray, comp
        ] - (
            m.S[1, comp] if n_tray == m.sideout1_tray else 0
        ) - (
            m.S[2, comp] if n_tray == m.sideout2_tray else 0
        ) == 0

    @disj.Constraint(doc="Product section 3 energy balances")
    def _productside_energy_balances(disj):
        """
        Energy balances for the product side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the product side section in the column.

        Returns
        -------
        Constraint
            The constraint expression that enforces the energy balance for the product side section in the column.
        """
        return (
            sum(
                (
                    m.L[3, n_tray + 1, comp] * m.hl[3, n_tray + 1, comp]
                    if n_tray < m.num_trays
                    else 0
                )
                + (
                    m.L[4, 1, comp] * m.dl[3] * m.hl[4, 1, comp]
                    if n_tray == m.num_trays
                    else 0
                )
                - m.L[3, n_tray, comp] * m.hl[3, n_tray, comp]
                + (
                    m.V[1, m.num_trays, comp] * m.dv[3] * m.hv[1, m.num_trays, comp]
                    if n_tray == 1
                    else 0
                )
                + (
                    m.V[3, n_tray - 1, comp] * m.hv[3, n_tray - 1, comp]
                    if n_tray > 1
                    else 0
                )
                - m.V[3, n_tray, comp] * m.hv[3, n_tray, comp]
                - (
                    m.S[1, comp] * m.hl[3, n_tray, comp]
                    if n_tray == m.sideout1_tray
                    else 0
                )
                - (
                    m.S[2, comp] * m.hl[3, n_tray, comp]
                    if n_tray == m.sideout2_tray
                    else 0
                )
                for comp in m.comp
            )
            * m.Qscale
            == 0
        )

    @disj.Constraint(m.comp, doc="Product section 3 liquid flowrate per component")
    def _productside_liquid_percomponent(disj, comp):
        """
        Liquid flowrate per component in the product side section of the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the product side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid flowrate per component in the product side section of the column is equal to the total liquid flowrate times the liquid composition.
        """
        return m.L[3, n_tray, comp] == m.Ltotal[3, n_tray] * m.x[3, n_tray, comp]

    @disj.Constraint(m.comp, doc="Product section 3 vapor flowrate per component")
    def _productside_vapor_percomponent(disj, comp):
        """
        Vapor flowrate per component in the product side section of the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the product side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor flowrate per component in the product side section of the column is equal to the total vapor flowrate times the vapor composition.
        """
        return m.V[3, n_tray, comp] == m.Vtotal[3, n_tray] * m.y[3, n_tray, comp]

    @disj.Constraint(doc="Product section 3 liquid composition equilibrium summation")
    def productside_liquid_composition_summation(disj):
        """
        Liquid composition equilibrium summation for the product side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the product side section in the column.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid composition equilibrium summation for the product side section in the column.
            It ensures the sum of the liquid compositions is equal to 1 plus the error in the liquid composition.
        """
        return sum(m.x[3, n_tray, comp] for comp in m.comp) - 1 == m.errx[3, n_tray]

    @disj.Constraint(doc="Product section 3 vapor composition equilibrium summation")
    def productside_vapor_composition_summation(disj):
        """
        Vapor composition equilibrium summation for the product side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the product side section in the column.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor composition equilibrium summation for the product side section in the column.
            It ensures the sum of the vapor compositions is equal to 1 plus the error in the vapor composition.
        """
        return sum(m.y[3, n_tray, comp] for comp in m.comp) - 1 == m.erry[3, n_tray]

    @disj.Constraint(m.comp, doc="Product section 3 vapor composition")
    def productside_vapor_composition(disj, comp):
        """
        Vapor composition for the product side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the product side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor composition for the product side section in the column.
            The equation is derived from the vapor-liquid equilibrium relationship.
        """
        return (
            m.y[3, n_tray, comp]
            == m.x[3, n_tray, comp]
            * (
                m.actv[3, n_tray, comp]
                * (
                    m.prop[comp, 'PC']
                    * exp(
                        m.prop[comp, 'TC']
                        / m.T[3, n_tray]
                        * (
                            m.prop[comp, 'vpA']
                            * (1 - m.T[3, n_tray] / m.prop[comp, 'TC'])
                            + m.prop[comp, 'vpB']
                            * (1 - m.T[3, n_tray] / m.prop[comp, 'TC']) ** 1.5
                            + m.prop[comp, 'vpC']
                            * (1 - m.T[3, n_tray] / m.prop[comp, 'TC']) ** 3
                            + m.prop[comp, 'vpD']
                            * (1 - m.T[3, n_tray] / m.prop[comp, 'TC']) ** 6
                        )
                    )
                )
            )
            / m.P[3, n_tray]
        )

    @disj.Constraint(m.comp, doc="Product section 3 liquid enthalpy")
    def productside_liquid_enthalpy(disj, comp):
        """
        Liquid enthalpy for the product side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the product side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid enthalpy for the product side section in the column.
        """
        return m.hl[3, n_tray, comp] == (
            m.cpc[1]
            * (
                (m.T[3, n_tray] - m.Tref) * m.prop[comp, 'cpA', 1]
                + (m.T[3, n_tray] ** 2 - m.Tref**2)
                * m.prop[comp, 'cpB', 1]
                * m.cpc2['A', 1]
                / 2
                + (m.T[3, n_tray] ** 3 - m.Tref**3)
                * m.prop[comp, 'cpC', 1]
                * m.cpc2['B', 1]
                / 3
                + (m.T[3, n_tray] ** 4 - m.Tref**4) * m.prop[comp, 'cpD', 1] / 4
            )
            / m.Hscale
        )

    @disj.Constraint(m.comp, doc="Product section 3 vapor enthalpy")
    def productside_vapor_enthalpy(disj, comp):
        """
        Vapor enthalpy for the product side section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the product side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor enthalpy for the product side section in the column.
        """
        return m.hv[3, n_tray, comp] == (
            m.cpc[2]
            * (
                (m.T[3, n_tray] - m.Tref) * m.prop[comp, 'cpA', 2]
                + (m.T[3, n_tray] ** 2 - m.Tref**2)
                * m.prop[comp, 'cpB', 2]
                * m.cpc2['A', 2]
                / 2
                + (m.T[3, n_tray] ** 3 - m.Tref**3)
                * m.prop[comp, 'cpC', 2]
                * m.cpc2['B', 2]
                / 3
                + (m.T[3, n_tray] ** 4 - m.Tref**4) * m.prop[comp, 'cpD', 2] / 4
            )
            / m.Hscale
            + m.dHvap[comp]
        )

    @disj.Constraint(m.comp, doc="Product section 3 liquid activity coefficient")
    def productside_activity_coefficient(disj, comp):
        """
        Liquid activity coefficient for the product side section in the column equal to 1.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the product side section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid activity coefficient for the product side section for each component and tray in the column to be equal to 1.
            This is an assumption for the product side section, since the product is assumed to be ideal.
        """
        return m.actv[3, n_tray, comp] == 1


def _build_top_equations(disj, n_tray):
    """
    Build the equations for the top section in the column.

    Parameters
    ----------
    disj : Disjunct
        The disjunct object for the top section in the column.
    n_tray : int
        The tray index.

    Returns
    -------
    None
    """
    m = disj.model()

    @disj.Constraint(m.comp, doc="Top section 4 mass per component balances")
    def _top_mass_percomponent_balances(disj, comp):
        """
        Mass per component balances for the top section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the top section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the mass balance per component in the top section of the column.
        """
        return (m.L[4, n_tray + 1, comp] if n_tray < m.con_tray else 0) - (
            m.L[4, n_tray, comp] * m.dl[2] if n_tray == 1 else 0
        ) - (m.L[4, n_tray, comp] * m.dl[3] if n_tray == 1 else 0) - (
            m.L[4, n_tray, comp] if n_tray > 1 else 0
        ) + (
            m.V[2, m.num_trays, comp] if n_tray == 1 else 0
        ) + (
            m.V[3, m.num_trays, comp] if n_tray == 1 else 0
        ) + (
            m.V[4, n_tray - 1, comp] if n_tray > 1 else 0
        ) - (
            m.V[4, n_tray, comp] if n_tray < m.con_tray else 0
        ) - (
            m.D[comp] if n_tray == m.con_tray else 0
        ) == 0

    @disj.Constraint(doc="Top scetion 4 energy balances")
    def _top_energy_balances(disj):
        """
        Energy balances for the top section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the top section in the column.

        Returns
        -------
        Constraint
            The constraint expression that enforces the energy balance for the top section in the column.
        """
        return (
            sum(
                (
                    m.L[4, n_tray + 1, comp] * m.hl[4, n_tray + 1, comp]
                    if n_tray < m.con_tray
                    else 0
                )
                - (
                    m.L[4, n_tray, comp] * m.dl[2] * m.hl[4, n_tray, comp]
                    if n_tray == 1
                    else 0
                )
                - (
                    m.L[4, n_tray, comp] * m.dl[3] * m.hl[4, n_tray, comp]
                    if n_tray == 1
                    else 0
                )
                - (m.L[4, n_tray, comp] * m.hl[4, n_tray, comp] if n_tray > 1 else 0)
                + (
                    m.V[2, m.num_trays, comp] * m.hv[2, m.num_trays, comp]
                    if n_tray == 1
                    else 0
                )
                + (
                    m.V[3, m.num_trays, comp] * m.hv[3, m.num_trays, comp]
                    if n_tray == 1
                    else 0
                )
                + (
                    m.V[4, n_tray - 1, comp] * m.hv[4, n_tray - 1, comp]
                    if n_tray > 1
                    else 0
                )
                - (
                    m.V[4, n_tray, comp] * m.hv[4, n_tray, comp]
                    if n_tray < m.con_tray
                    else 0
                )
                - (m.D[comp] * m.hl[4, n_tray, comp] if n_tray == m.con_tray else 0)
                for comp in m.comp
            )
            * m.Qscale
            - (m.Qcon if n_tray == m.con_tray else 0)
            == 0
        )

    @disj.Constraint(m.comp, doc="Top section 4 liquid flowrate per component")
    def _top_liquid_percomponent(disj, comp):
        """
        Liquid flowrate per component in the top section of the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the top section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid flowrate per component in the top section of the column is equal to the total liquid flowrate times the liquid composition.
        """
        return m.L[4, n_tray, comp] == m.Ltotal[4, n_tray] * m.x[4, n_tray, comp]

    @disj.Constraint(m.comp, doc="Top section 4 vapor flowrate per component")
    def _top_vapor_percomponent(disj, comp):
        """
        Vapor flowrate per component in the top section of the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the top section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor flowrate per component in the top section of the column is equal to the total vapor flowrate times the vapor composition.
        """
        return m.V[4, n_tray, comp] == m.Vtotal[4, n_tray] * m.y[4, n_tray, comp]

    @disj.Constraint(doc="Top section 4 liquid composition equilibrium summation")
    def top_liquid_composition_summation(disj):
        """
        Liquid composition equilibrium summation for the top section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the top section in the column.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid composition equilibrium summation for the top section in the column.
            It ensures the sum of the liquid compositions is equal to 1 plus the error in the liquid composition.
        """
        return sum(m.x[4, n_tray, comp] for comp in m.comp) - 1 == m.errx[4, n_tray]

    @disj.Constraint(doc="Top section 4 vapor composition equilibrium summation")
    def top_vapor_composition_summation(disj):
        """
        Vapor composition equilibrium summation for the top section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the top section in the column.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor composition equilibrium summation for the top section in the column.
            It ensures the sum of the vapor compositions is equal to 1 plus the error in the vapor composition.
        """
        return sum(m.y[4, n_tray, comp] for comp in m.comp) - 1 == m.erry[4, n_tray]

    @disj.Constraint(m.comp, doc="Top scetion 4 vapor composition")
    def top_vapor_composition(disj, comp):
        """
        Vapor composition for the top section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the top section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor composition for the top section in the column.
            The equation is derived from the vapor-liquid equilibrium relationship.
        """
        return (
            m.y[4, n_tray, comp]
            == m.x[4, n_tray, comp]
            * (
                m.actv[4, n_tray, comp]
                * (
                    m.prop[comp, 'PC']
                    * exp(
                        m.prop[comp, 'TC']
                        / m.T[4, n_tray]
                        * (
                            m.prop[comp, 'vpA']
                            * (1 - m.T[4, n_tray] / m.prop[comp, 'TC'])
                            + m.prop[comp, 'vpB']
                            * (1 - m.T[4, n_tray] / m.prop[comp, 'TC']) ** 1.5
                            + m.prop[comp, 'vpC']
                            * (1 - m.T[4, n_tray] / m.prop[comp, 'TC']) ** 3
                            + m.prop[comp, 'vpD']
                            * (1 - m.T[4, n_tray] / m.prop[comp, 'TC']) ** 6
                        )
                    )
                )
            )
            / m.P[4, n_tray]
        )

    @disj.Constraint(m.comp, doc="Top section 4 liquid enthalpy")
    def top_liquid_enthalpy(disj, comp):
        """
        Liquid enthalpy for the top section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the top section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid enthalpy for the top section in the column.
        """
        return m.hl[4, n_tray, comp] == (
            m.cpc[1]
            * (
                (m.T[4, n_tray] - m.Tref) * m.prop[comp, 'cpA', 1]
                + (m.T[4, n_tray] ** 2 - m.Tref**2)
                * m.prop[comp, 'cpB', 1]
                * m.cpc2['A', 1]
                / 2
                + (m.T[4, n_tray] ** 3 - m.Tref**3)
                * m.prop[comp, 'cpC', 1]
                * m.cpc2['B', 1]
                / 3
                + (m.T[4, n_tray] ** 4 - m.Tref**4) * m.prop[comp, 'cpD', 1] / 4
            )
            / m.Hscale
        )

    @disj.Constraint(m.comp, doc="Top section 4 vapor enthalpy")
    def top_vapor_enthalpy(disj, comp):
        """
        Vapor enthalpy for the top section in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the top section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor enthalpy for the top section in the column.
        """
        return m.hv[4, n_tray, comp] == (
            m.cpc[2]
            * (
                (m.T[4, n_tray] - m.Tref) * m.prop[comp, 'cpA', 2]
                + (m.T[4, n_tray] ** 2 - m.Tref**2)
                * m.prop[comp, 'cpB', 2]
                * m.cpc2['A', 2]
                / 2
                + (m.T[4, n_tray] ** 3 - m.Tref**3)
                * m.prop[comp, 'cpC', 2]
                * m.cpc2['B', 2]
                / 3
                + (m.T[4, n_tray] ** 4 - m.Tref**4) * m.prop[comp, 'cpD', 2] / 4
            )
            / m.Hscale
            + m.dHvap[comp]
        )

    @disj.Constraint(m.comp, doc="Top section 4 liquid activity coefficient")
    def top_activity_coefficient(disj, comp):
        """
        Liquid activity coefficient for the top section in the column equal to 1.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the top section in the column.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid activity coefficient for the top section for each component and tray in the column to be equal to 1.
            This is an assumption for the top section, since the product is assumed to be ideal.
        """
        return m.actv[4, n_tray, comp] == 1


def _build_pass_through_eqns(disj, sec, n_tray):
    """
    Build the equations for the pass through section in the column when a given tray in the disjunct is not active if it is the first or last tray.

    Parameters
    ----------
    disj : Disjunct
        The disjunct object for the pass through section in the column.
    sec : int
        The section index.
    n_tray : int
        The tray index.

    Returns
    -------
    None
    """
    m = disj.model()

    # If the tray is the first or last tray, then the liquid and vapor flowrates, compositions, enthalpies, and temperature are passed through.
    if n_tray == 1 or n_tray == m.num_trays:
        return

    @disj.Constraint(m.comp, doc="Pass through liquid flowrate")
    def pass_through_liquid_flowrate(disj, comp):
        """
        Pass through liquid flowrate for the given tray in the column.
        The constraint enforces the liquid flowrate for the given tray is equal to the liquid flowrate for the tray above it.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the pass through when the tray is inactive.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid flowrate for the given tray is equal to the liquid flowrate for the tray above it.
        """
        return m.L[sec, n_tray, comp] == m.L[sec, n_tray + 1, comp]

    @disj.Constraint(m.comp, doc="Pass through vapor flowrate")
    def pass_through_vapor_flowrate(disj, comp):
        """
        Pass through vapor flowrate for the given tray in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the pass through when the tray is inactive.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor flowrate for the given tray is equal to the vapor flowrate for the tray below it.
        """
        return m.V[sec, n_tray, comp] == m.V[sec, n_tray - 1, comp]

    @disj.Constraint(m.comp, doc="Pass through liquid composition")
    def pass_through_liquid_composition(disj, comp):
        """
        Pass through liquid composition for the given tray in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the pass through when the tray is inactive.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid composition for the given tray is equal to the liquid composition for the tray above it.
        """
        return m.x[sec, n_tray, comp] == m.x[sec, n_tray + 1, comp]

    @disj.Constraint(m.comp, doc="Pass through vapor composition")
    def pass_through_vapor_composition(disj, comp):
        """
        Pass through vapor composition for the given tray in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the pass through when the tray is inactive.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor composition for the given tray is equal to the vapor composition for the tray below it.
        """
        return m.y[sec, n_tray, comp] == m.y[sec, n_tray + 1, comp]

    @disj.Constraint(m.comp, doc="Pass through liquid enthalpy")
    def pass_through_liquid_enthalpy(disj, comp):
        """
        Pass through liquid enthalpy for the given tray in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the pass through when the tray is inactive.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the liquid enthalpy for the given tray is equal to the liquid enthalpy for the tray above it.
        """
        return m.hl[sec, n_tray, comp] == m.hl[sec, n_tray + 1, comp]

    @disj.Constraint(m.comp, doc="Pass through vapor enthalpy")
    def pass_through_vapor_enthalpy(disj, comp):
        """
        Pass through vapor enthalpy for the given tray in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the pass through when the tray is inactive.
        comp : int
            The component index.

        Returns
        -------
        Constraint
            The constraint expression that enforces the vapor enthalpy for the given tray is equal to the vapor enthalpy for the tray below it.
        """
        return m.hv[sec, n_tray, comp] == m.hv[sec, n_tray - 1, comp]

    @disj.Constraint(doc="Pass through temperature")
    def pass_through_temperature(disj):
        """
        Pass through temperature for the given tray in the column.

        Parameters
        ----------
        disj : Disjunct
            The disjunct object for the pass through when the tray is inactive.

        Returns
        -------
        Constraint
            The constraint expression that enforces the temperature for the given tray is equal to the temperature for the tray below it.
        """
        return m.T[sec, n_tray] == m.T[sec, n_tray - 1]


if __name__ == "__main__":
    model = build_model()
