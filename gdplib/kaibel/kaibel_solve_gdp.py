""" Kaibel Column model: GDP formulation """


from __future__ import division

from math import copysign

from pyomo.environ import (Constraint, exp, minimize, NonNegativeReals, Objective, RangeSet, Set, Var)
from pyomo.gdp import Disjunct

from gdplib.kaibel.kaibel_init import initialize_kaibel

from gdplib.kaibel.kaibel_side_flash import calc_side_feed_flash


def build_model():

    m = initialize_kaibel()


    # Side feed init
    m = calc_side_feed_flash(m)


    m.name = "GDP Kaibel Column"

    #### Calculated initial values
    m.Treb = m.TB0 + 5          # Reboiler temperature in K
    m.Tbot = m.TB0              # Bottom-most tray temperature in K
    m.Ttop = m.TD0              # Top-most tray temperature in K
    m.Tcon = m.TD0 - 5          # Condenser temperature in K

    m.dv0 = {}                  # Initial vapor distributor value
    m.dl0 = {}                  # Initial liquid distributor value
    m.dv0[2] = 0.516
    m.dv0[3] = 1 - m.dv0[2]
    m.dl0[2] = 0.36
    m.dl0[3] = 1 - m.dl0[2]


    #### Calculated upper and lower bounds
    m.min_tray = m.Knmin * 0.8  # Lower bound on number of trays
    m.Tlo = m.Tcon - 20         # Temperature lower bound
    m.Tup = m.Treb + 20         # Temperature upper bound
    
    m.flow_max = 1e3            # Flowrates upper bound
    m.Qmax = 60                 # Heat loads upper bound


    #### Column tray details
    m.num_trays = m.np          # Trays per section
    m.min_num_trays = 10        # Minimum number of trays per section
    m.num_total = m.np * 3      # Total number of trays 
    m.feed_tray = 12            # Side feed tray
    m.sideout1_tray = 8         # Side outlet 1 tray
    m.sideout2_tray = 17        # Side outlet 2 tray
    m.reb_tray = 1              # Reboiler tray
    m.con_tray = m.num_trays    # Condenser tray



    # ------------------------------------------------------------------
    
    #                          Beginning of model
    
    # ------------------------------------------------------------------


    ## Sets
    m.section = RangeSet(4,
                         doc="Column sections:1=top, 2=feed side, 3=prod side, 4=bot")
    m.section_main = Set(initialize=[1, 4])

    m.tray = RangeSet(m.np,
                      doc="Potential trays in each section")
    m.tray_total = RangeSet(m.num_total,
                            doc="Total trays in the column")
    m.tray_below_feed = RangeSet(m.feed_tray,
                                 doc="Trays below feed")
    m.tray_below_so1 = RangeSet(m.sideout1_tray,
                                doc="Trays below side outlet 1")
    m.tray_below_so2 = RangeSet(m.sideout2_tray,
                                doc="Trays below side outlet 1")

    m.comp = RangeSet(4,
                      doc="Components")
    m.dw = RangeSet(2, 3,
                    doc="Dividing wall sections")
    m.cplv = RangeSet(2,
                      doc="Heat capacity: 1=liquid, 2=vapor")
    m.so = RangeSet(2,
                    doc="Side product outlets")
    m.bounds = RangeSet(2,
                        doc="Number of boundary condition values")
    
    m.candidate_trays_main = Set(initialize=m.tray
                                 - [m.con_tray, m.reb_tray],
                                 doc="Candidate trays for top and \
                                 bottom sections 1 and 4")
    m.candidate_trays_feed = Set(initialize=m.tray
                                 - [m.con_tray, m.feed_tray, m.reb_tray],
                                 doc="Candidate trays for feed section 2")
    m.candidate_trays_product = Set(initialize=m.tray
                                    - [m.con_tray, m.sideout1_tray,
                                       m.sideout2_tray, m.reb_tray],
                                    doc="Candidate trays for product section 3")

    
    ## Calculation of initial values
    m.dHvap = {}                # Heat of vaporization

    m.P0 = {}                   # Initial pressure 
    m.T0 = {}                   # Initial temperature
    m.L0 = {}                   # Initial individual liquid flowrate in mol/s 
    m.V0 = {}                   # Initial individual vapor flowrate 
    m.Vtotal0 = {}              # Initial total vapor flowrate in mol/s
    m.Ltotal0 = {}              # Initial liquid flowrate in mol/s
    m.x0 = {}                   # Initial liquid composition
    m.y0 = {}                   # Initial vapor composition
    m.actv0 = {}                # Initial activity coefficients
    m.cpdT0 = {}                # Initial heat capacity for liquid and vapor phases
    m.hl0 = {}                  # Initial liquid enthalpy in J/mol
    m.hv0 = {}                  # Initial vapor enthalpy in J/mol
    m.Pi = m.Preb               # Initial given pressure value
    m.Ti = {}                   # Initial known temperature values

    
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
            m.Ltotal0[sec, n_tray] = sum(
                m.L0[sec, n_tray, comp] for comp in m.comp)
            m.Vtotal0[sec, n_tray] = sum(
                m.V0[sec, n_tray, comp] for comp in m.comp)

        
    for n_tray in m.tray_total:
        if n_tray == m.reb_tray:
            m.Ti[n_tray] = m.Treb
        elif n_tray == m.num_total:
            m.Ti[n_tray] = m.Tcon
        else:
            m.Ti[n_tray] = (
                m.Tbot
                + (m.Ttop - m.Tbot) * \
                (n_tray - 2) / (m.num_total - 3)
            )

            
    for n_tray in m.tray_total:
        if n_tray <= m.num_trays:
            m.T0[1, n_tray] = m.Ti[n_tray]
        elif n_tray >= m.num_trays and n_tray <= m.num_trays * 2:
            m.T0[2, n_tray - m.num_trays] = m.Ti[n_tray]
            m.T0[3, n_tray - m.num_trays] = m.Ti[n_tray]
        elif n_tray >= m.num_trays * 2:
            m.T0[4, n_tray - m.num_trays*2] = m.Ti[n_tray]

    for sec in m.section:
        for n_tray in m.tray:
            for comp in m.comp:
                m.x0[sec, n_tray, comp] = m.xfi[comp]
                m.actv0[sec, n_tray, comp] = 1
                m.y0[sec, n_tray, comp] = m.xfi[comp]
                
    
    ## Enthalpy boundary values
    hlb = {}                    # Liquid enthalpy 
    hvb = {}                    # Vapor enthalpy 
    cpb = {}                    # Heact capacity 
    dHvapb = {}                 # Heat of vaporization
    Tbounds = {}                # Temperature bounds 
    kc = {}                     # Light and heavy key components
    Tbounds[1] = m.Tcon
    Tbounds[2] = m.Treb
    kc[1] = m.lc
    kc[2] = m.hc

    
    for comp in m.comp:
        dHvapb[comp] = -(
            m.Rgas * m.prop[comp, 'TC'] * (
                m.prop[comp, 'vpA'] * \
                (1 - m.Tref / m.prop[comp, 'TC'])
                + m.prop[comp, 'vpB'] * \
                (1 - m.Tref / m.prop[comp, 'TC'])**1.5
                + m.prop[comp, 'vpC'] * \
                (1 - m.Tref / m.prop[comp, 'TC'])**3
                + m.prop[comp, 'vpD'] * \
                (1 - m.Tref / m.prop[comp, 'TC'])**6)
            + m.Rgas * m.Tref * (
                m.prop[comp, 'vpA']
                + 1.5 * m.prop[comp, 'vpB'] * \
                (1 - m.Tref / m.prop[comp, 'TC'])**0.5
                + 3 * m.prop[comp, 'vpC'] * \
                (1 - m.Tref / m.prop[comp, 'TC'])**2
                + 6 * m.prop[comp, 'vpD'] * \
                (1 - m.Tref / m.prop[comp, 'TC'])**5
            )
        )

        
    for b in m.bounds:
        for cp in m.cplv:
            cpb[b, cp] = m.cpc[cp] * (
                (Tbounds[b] - m.Tref) * \
                m.prop[kc[b], 'cpA', cp]
                + (Tbounds[b]**2 - m.Tref**2) * \
                m.prop[kc[b], 'cpB', cp] * \
                m.cpc2['A', cp] / 2
                + (Tbounds[b]**3 - m.Tref**3) * \
                m.prop[kc[b], 'cpC', cp] * \
                m.cpc2['B', cp] / 3
                + (Tbounds[b]**4 - m.Tref**4) * \
                m.prop[kc[b], 'cpD', cp] / 4
            )
        hlb[b] = (
            cpb[b, 1]
        )
        hvb[b] = (
            cpb[b, 2]
            + dHvapb[b]
        )

    m.hllo = (1 - copysign(0.2, hlb[1])) * hlb[1] / m.Hscale
    m.hlup = (1 + copysign(0.2, hlb[2])) * hlb[2] / m.Hscale
    m.hvlo = (1 - copysign(0.2, hvb[1])) * hvb[1] / m.Hscale
    m.hvup = (1 + copysign(0.2, hvb[2])) * hvb[2] / m.Hscale

    
    for comp in m.comp:
        m.dHvap[comp] = dHvapb[comp] / m.Hscale

        
    for sec in m.section:
        for n_tray in m.tray:
            for comp in m.comp:
                for cp in m.cplv:
                    m.cpdT0[sec, n_tray, comp, cp] = (
                        m.cpc[cp] * (
                            (m.T0[sec, n_tray] - m.Tref) * \
                            m.prop[comp, 'cpA', cp]
                            + (m.T0[sec, n_tray]**2 - m.Tref**2) * \
                            m.prop[comp, 'cpB', cp] * \
                            m.cpc2['A', cp] / 2
                            + (m.T0[sec, n_tray]**3 - m.Tref**3) * \
                            m.prop[comp, 'cpC', cp] * \
                            m.cpc2['B', cp] / 3
                            + (m.T0[sec, n_tray]**4 - m.Tref**4) * \
                            m.prop[comp, 'cpD', cp] / 4
                        ) / m.Hscale
                    )


    for sec in m.section:
        for n_tray in m.tray:
            for comp in m.comp:
                m.hl0[sec, n_tray, comp] = (
                    m.cpdT0[sec, n_tray, comp, 1]
                ) 
                m.hv0[sec, n_tray, comp] = (
                    m.cpdT0[sec, n_tray, comp, 2]
                    + m.dHvap[comp]
                ) 

    #### Side feed
    m.cpdTf = {}                # Heat capacity for side feed J/mol K
    m.hlf = {}                  # Liquid enthalpy for side feed in J/mol
    m.hvf = {}                  # Vapor enthalpy for side feed in J/mol
    m.F0 = {}                   # Side feed flowrate per component in mol/s

    for comp in m.comp:
        for cp in m.cplv:
            m.cpdTf[comp, cp] = (
                m.cpc[cp]*(
                    (m.Tf - m.Tref) * \
                    m.prop[comp, 'cpA', cp]
                    + (m.Tf**2 - m.Tref**2) * \
                    m.prop[comp, 'cpB', cp] * \
                    m.cpc2['A', cp] / 2
                    + (m.Tf**3 - m.Tref**3) * \
                    m.prop[comp, 'cpC', cp] * \
                    m.cpc2['B', cp] / 3
                    + (m.Tf**4 - m.Tref**4) * \
                    m.prop[comp, 'cpD', cp] / 4
                ) / m.Hscale
            )
            
    for comp in m.comp:
        m.F0[comp] = m.xfi[comp] * m.Fi
        m.hlf[comp] = (
            m.cpdTf[comp, 1]
        )
        m.hvf[comp] = (
            m.cpdTf[comp, 2]
            + m.dHvap[comp]
        )

    m.P = Var(m.section, m.tray,
              doc="Pressure at each potential tray in bars",
              domain=NonNegativeReals,
              bounds=(m.Pcon, m.Preb),
              initialize=m.P0)
    m.T = Var(m.section, m.tray,
              doc="Temperature at each potential tray in K",
              domain=NonNegativeReals,
              bounds=(m.Tlo, m.Tup),
              initialize=m.T0)

    m.x = Var(m.section, m.tray, m.comp,
              doc="Liquid composition",
              domain=NonNegativeReals,
              bounds=(0, 1),
              initialize=m.x0)
    m.y = Var(m.section, m.tray, m.comp,
              doc="Vapor composition",
              domain=NonNegativeReals,
              bounds=(0, 1),
              initialize=m.y0)

    m.dl = Var(m.dw,
               doc="Liquid distributor",
               bounds=(0.2, 0.8),
               initialize=m.dl0)
    m.dv = Var(m.dw,
               doc="Vapor distributor",
               bounds=(0, 1),
               domain=NonNegativeReals,
               initialize=m.dv0)

    m.V = Var(m.section, m.tray, m.comp,
              doc="Vapor flowrate in mol/s",
              domain=NonNegativeReals,
              bounds=(0, m.flow_max),
              initialize=m.V0)
    m.L = Var(m.section, m.tray, m.comp,
              doc="Liquid flowrate in mol/s",
              domain=NonNegativeReals,
              bounds=(0, m.flow_max),
              initialize=m.L0)
    m.Vtotal = Var(m.section, m.tray,
                   doc="Total vapor flowrate in mol/s",
                   domain=NonNegativeReals,
                   bounds=(0, m.flow_max),
                   initialize=m.Vtotal0)
    m.Ltotal = Var(m.section, m.tray,
                   doc="Total liquid flowrate in mol/s",
                   domain=NonNegativeReals,
                   bounds=(0, m.flow_max),
                   initialize=m.Ltotal0)

    m.D = Var(m.comp,
              doc="Distillate flowrate in mol/s",
              domain=NonNegativeReals,
              bounds=(0, m.flow_max),
              initialize=m.Ddes)
    m.B = Var(m.comp,
              doc="Bottoms flowrate in mol/s",
              domain=NonNegativeReals,
              bounds=(0, m.flow_max),
              initialize=m.Bdes)
    m.S = Var(m.so, m.comp,
              doc="Product 2 and 3 flowrates in mol/s",
              domain=NonNegativeReals,
              bounds=(0, m.flow_max),
              initialize=m.Sdes)
    m.Dtotal = Var(doc="Distillate flowrate in mol/s",
                   domain=NonNegativeReals,
                   bounds=(0, m.flow_max),
                   initialize=m.Ddes)
    m.Btotal = Var(doc="Bottoms flowrate in mol/s",
                   domain=NonNegativeReals,
                   bounds=(0, m.flow_max),
                   initialize=m.Bdes)
    m.Stotal = Var(m.so,
                   doc="Total product 2 and 3 side flowrate in mol/s",
                   domain=NonNegativeReals,
                   bounds=(0, m.flow_max),
                   initialize=m.Sdes)

    m.hl = Var(m.section, m.tray, m.comp,
               doc='Liquid enthalpy in J/mol',
               bounds=(m.hllo, m.hlup),
               initialize=m.hl0)
    m.hv = Var(m.section, m.tray, m.comp,
               doc='Vapor enthalpy in J/mol',
               bounds=(m.hvlo, m.hvup),
               initialize=m.hv0)
    m.Qreb = Var(doc="Reboiler heat duty in J/s",
                 domain=NonNegativeReals,
                 bounds=(0, m.Qmax),
                 initialize=1)
    m.Qcon = Var(doc="Condenser heat duty in J/s",
                 domain=NonNegativeReals,
                 bounds=(0, m.Qmax),
                 initialize=1)

    m.rr = Var(doc="Internal reflux ratio",
               domain=NonNegativeReals,
               bounds=(0.7, 1),
               initialize=m.rr0)
    m.bu = Var(doc="Boilup rate",
               domain=NonNegativeReals,
               bounds=(0.7, 1),
               initialize=m.bu0)

    m.F = Var(m.comp,
              doc="Side feed flowrate in mol/s",
              domain=NonNegativeReals,
              bounds=(0, 50),
              initialize=m.F0)
    m.q = Var(doc="Vapor fraction in side feed",
              domain=NonNegativeReals,
              bounds=(0, 1),
              initialize=1)

    m.actv = Var(m.section, m.tray, m.comp,
                 doc="Liquid activity coefficient",
                 domain=NonNegativeReals,
                 bounds=(0, 10),
                 initialize=m.actv0)

    m.errx = Var(m.section, m.tray,
                 bounds=(-1e-3, 1e-3),
                 initialize=0)
    m.erry = Var(m.section, m.tray,
                 bounds=(-1e-3, 1e-3), initialize=0)
    m.slack = Var(m.section, m.tray, m.comp,
                  doc="Slack variable",
                  bounds=(-1e-8, 1e-8),
                  initialize=0)

    m.tray_exists = Disjunct(m.section, m.tray,
                             rule=_build_tray_equations)
    m.tray_absent = Disjunct(m.section, m.tray,
                             rule=_build_pass_through_eqns)


    @m.Disjunction(m.section, m.tray,
                   doc="Disjunction between whether each tray exists or not")
    def tray_exists_or_not(m, sec, n_tray):
        return [m.tray_exists[sec, n_tray], m.tray_absent[sec, n_tray]]

    @m.Constraint(m.section_main)
    def minimum_trays_main(m, sec):
        return sum(m.tray_exists[sec, n_tray].binary_indicator_var
                   for n_tray in m.candidate_trays_main) + 1  >= m.min_num_trays

    @m.Constraint()
    def minimum_trays_feed(m):
        return sum(m.tray_exists[2, n_tray].binary_indicator_var
                   for n_tray in m.candidate_trays_feed) + 1  >= m.min_num_trays

    @m.Constraint()
    def minimum_trays_product(m):
        return sum(m.tray_exists[3, n_tray].binary_indicator_var
                   for n_tray in m.candidate_trays_product) + 1  >= m.min_num_trays

    
    ## Fixed trays
    enforce_tray_exists(m, 1, 1)              # reboiler
    enforce_tray_exists(m, 1, m.num_trays)    # vapor distributor
    enforce_tray_exists(m, 2, 1)              # dividing wall starting tray
    enforce_tray_exists(m, 2, m.feed_tray)    # feed tray
    enforce_tray_exists(m, 2, m.num_trays)    # dividing wall ending tray
    enforce_tray_exists(m, 3, 1)              # dividing wall starting tray
    enforce_tray_exists(m, 3, m.sideout1_tray)# side outlet 1 for product 3
    enforce_tray_exists(m, 3, m.sideout2_tray)# side outlet 2 for product 2
    enforce_tray_exists(m, 3, m.num_trays)    # dividing wall ending tray
    enforce_tray_exists(m, 4, 1)              # liquid distributor
    enforce_tray_exists(m, 4, m.num_trays)    # condenser




    #### Global constraints
    @m.Constraint(m.dw, m.tray, doc="Monotonic temperature")
    def monotonic_temperature(m, sec, n_tray):
        return m.T[sec, n_tray] <= m.T[1, m.num_trays]

    
    @m.Constraint(doc="Liquid distributor")
    def liquid_distributor(m):
        return sum(m.dl[sec] for sec in m.dw) - 1 == 0


    @m.Constraint(doc="Vapor distributor")
    def vapor_distributor(m):
        return sum(m.dv[sec] for sec in m.dw) - 1 == 0


    @m.Constraint(doc="Reboiler composition specification")
    def heavy_product(m):
        return m.x[1, m.reb_tray, m.hc] >= m.xspec_hc


    @m.Constraint(doc="Condenser composition specification")
    def light_product(m):
        return m.x[4, m.con_tray, m.lc] >= m.xspec_lc

    @m.Constraint(doc="Side outlet 1 final liquid composition")
    def intermediate1_product(m):
        return m.x[3, m.sideout1_tray, 3] >= m.xspec_inter3


    @m.Constraint(doc="Side outlet 2 final liquid composition")
    def intermediate2_product(m):
        return m.x[3, m.sideout2_tray, 2] >= m.xspec_inter2

    
    @m.Constraint(doc="Reboiler flowrate")
    def _heavy_product_flow(m):
        return m.Btotal >= m.Bdes 

    
    @m.Constraint(doc="Condenser flowrate")
    def _light_product_flow(m):
        return m.Dtotal >= m.Ddes 

    
    @m.Constraint(m.so, doc="Intermediate flowrate")
    def _intermediate_product_flow(m, so):
        return m.Stotal[so] >= m.Sdes 

    
    @m.Constraint(doc="Internal boilup ratio, V/L")
    def _internal_boilup_ratio(m):
        return m.bu * m.Ltotal[1, m.reb_tray + 1] == m.Vtotal[1, m.reb_tray]


    @m.Constraint(doc="Internal reflux ratio, L/V")
    def internal_reflux_ratio(m):
        return m.rr * m.Vtotal[4, m.con_tray - 1] == m.Ltotal[4, m.con_tray]


    @m.Constraint(doc="External boilup ratio relation with bottoms")
    def _external_boilup_ratio(m):
        return m.Btotal == (1 - m.bu) * m.Ltotal[1, m.reb_tray + 1] 


    @m.Constraint(doc="External reflux ratio relation with distillate")
    def _external_reflux_ratio(m):
        return m.Dtotal == (1 - m.rr) * m.Vtotal[4, m.con_tray - 1] 

    
    @m.Constraint(m.section, m.tray, doc="Total vapor flowrate")
    def _total_vapor_flowrate(m, sec, n_tray):
        return sum(m.V[sec, n_tray, comp] for comp in m.comp) ==  m.Vtotal[sec, n_tray] 


    @m.Constraint(m.section, m.tray, doc="Total liquid flowrate")
    def _total_liquid_flowrate(m, sec, n_tray):
        return sum(m.L[sec, n_tray, comp] for comp in m.comp) == m.Ltotal[sec, n_tray] 


    @m.Constraint(m.comp, doc="Bottoms and liquid relation")
    def bottoms_equality(m, comp):
        return m.B[comp] == m.L[1, m.reb_tray, comp]


    @m.Constraint(m.comp)
    def condenser_total(m, comp):
        return m.V[4, m.con_tray, comp] ==  0

    
    @m.Constraint()
    def total_bottoms_product(m):
        return sum(m.B[comp] for comp in m.comp) == m.Btotal

    
    @m.Constraint()
    def total_distillate_product(m):
        return sum(m.D[comp] for comp in m.comp) == m.Dtotal


    @m.Constraint(m.so)
    def total_side_product(m, so):
        return sum(m.S[so, comp] for comp in m.comp) == m.Stotal[so]



    
    m.obj = Objective(
        expr= (m.Qcon + m.Qreb) * m.Hscale + 1e3 * (
            sum(
                sum(m.tray_exists[sec, n_tray].binary_indicator_var
                     for n_tray in m.candidate_trays_main)
                for sec in m.section_main)
            + sum(m.tray_exists[2, n_tray].binary_indicator_var
                    for n_tray in m.candidate_trays_feed)
            + sum(m.tray_exists[3, n_tray].binary_indicator_var
                    for n_tray in m.candidate_trays_product)
            + 1),
        sense=minimize)





    @m.Constraint(m.section_main, m.candidate_trays_main)
    def _logic_proposition_main(m, sec, n_tray):
        if n_tray > m.reb_tray and (n_tray + 1) < m.num_trays:
            return m.tray_exists[sec, n_tray].binary_indicator_var <= m.tray_exists[sec, n_tray + 1].binary_indicator_var
        else:
            return Constraint.NoConstraint


    @m.Constraint(m.candidate_trays_feed)
    def _logic_proposition_feed(m, n_tray):
        if n_tray > m.reb_tray and (n_tray + 1) < m.feed_tray:
            return m.tray_exists[2, n_tray].binary_indicator_var <= m.tray_exists[2, n_tray + 1].binary_indicator_var
        elif n_tray > m.feed_tray and (n_tray + 1) < m.con_tray:
            return m.tray_exists[2, n_tray + 1].binary_indicator_var <= m.tray_exists[2, n_tray].binary_indicator_var
        else:
            return Constraint.NoConstraint


    @m.Constraint(m.candidate_trays_product)
    def _logic_proposition_section3(m, n_tray):
        if n_tray > 1 and (n_tray + 1) < m.num_trays:
            return m.tray_exists[3, n_tray].binary_indicator_var <= m.tray_exists[3, n_tray + 1].binary_indicator_var
        else:
            return Constraint.NoConstraint

    
    @m.Constraint(m.tray)
    def equality_feed_product_side(m, n_tray):
            return m.tray_exists[2, n_tray].binary_indicator_var == m.tray_exists[3, n_tray].binary_indicator_var


    @m.Constraint()
    def _existent_minimum_numbertrays(m):
        return sum(
            sum(m.tray_exists[sec, n_tray].binary_indicator_var
                for n_tray in m.tray) for sec in m.section) - sum(m.tray_exists[3, n_tray].binary_indicator_var for n_tray in m.tray)  >= int(m.min_tray)


    return m



def enforce_tray_exists(m, sec, n_tray):
    m.tray_exists[sec, n_tray].indicator_var.fix(True)
    m.tray_absent[sec, n_tray].deactivate()


def _build_tray_equations(m, sec, n_tray):
    build_function = {
        1: _build_bottom_equations,
        2: _build_feed_side_equations,
        3: _build_product_side_equations,
        4: _build_top_equations
    }
    build_function[sec](m, n_tray)



def _build_bottom_equations(disj, n_tray):
    m = disj.model()

    @disj.Constraint(m.comp,
                     doc="Bottom section 1 mass per component balances")
    def _bottom_mass_percomponent_balances(disj, comp):
        return (
            (m.L[1, n_tray + 1, comp]
             if n_tray < m.num_trays else 0)
            + (m.L[2, 1, comp]
               if n_tray == m.num_trays else 0)
            + (m.L[3, 1, comp]
               if n_tray == m.num_trays else 0)
            - (m.L[1, n_tray, comp]
               if n_tray > m.reb_tray else 0)
            + (m.V[1, n_tray - 1, comp]
               if n_tray > m.reb_tray else 0)
            - (m.V[1, n_tray, comp] * m.dv[2]
               if n_tray == m.num_trays else 0)
            - (m.V[1, n_tray, comp] * m.dv[3]
               if n_tray == m.num_trays else 0)
            - (m.V[1, n_tray, comp]
               if n_tray < m.num_trays else 0)
            - (m.B[comp]
               if n_tray == m.reb_tray else 0)
            == m.slack[1, n_tray, comp]
        )
    
    @disj.Constraint(doc="Bottom section 1 energy balances")
    def _bottom_energy_balances(disj):
        return (
            sum(
                (m.L[1, n_tray + 1, comp] * m.hl[1, n_tray + 1, comp]
                 if n_tray < m.num_trays else 0)
                + (m.L[2, 1, comp] * m.hl[2, 1, comp]
                   if n_tray == m.num_trays else 0)
                + (m.L[3, 1, comp] * m.hl[3, 1, comp]
                   if n_tray == m.num_trays else 0)
                - (m.L[1, n_tray, comp] * m.hl[1, n_tray, comp]
                   if n_tray > m.reb_tray else 0)
                + (m.V[1, n_tray - 1, comp] * m.hv[1, n_tray - 1, comp]
                   if n_tray > m.reb_tray else 0)
                - (m.V[1, n_tray, comp] * m.dv[2] * m.hv[1, n_tray, comp]
                   if n_tray == m.num_trays else 0)
                - (m.V[1, n_tray, comp] * m.dv[3] * m.hv[1, n_tray, comp]
                   if n_tray == m.num_trays else 0)
                - (m.V[1, n_tray, comp] * m.hv[1, n_tray, comp]
                   if n_tray < m.num_trays else 0)
                - (m.B[comp] * m.hl[1, n_tray, comp]
                   if n_tray == m.reb_tray else 0)
                for comp in m.comp) * m.Qscale
            + (m.Qreb if n_tray == m.reb_tray else 0) 
            ==0
        )


    @disj.Constraint(m.comp,
                     doc="Bottom section 1 liquid flowrate per component")
    def _bottom_liquid_percomponent(disj, comp):
        return m.L[1, n_tray, comp] == m.Ltotal[1, n_tray] * m.x[1, n_tray, comp]


    @disj.Constraint(m.comp,
                     doc="Bottom section 1 vapor flowrate per component")
    def _bottom_vapor_percomponent(disj, comp):
        return m.V[1, n_tray, comp]  == m.Vtotal[1, n_tray] * m.y[1, n_tray, comp]


    @disj.Constraint(doc="Bottom section 1 liquid composition equilibrium summation")
    def bottom_liquid_composition_summation(disj):
        return sum(m.x[1, n_tray, comp] for comp in m.comp) - 1 == m.errx[1, n_tray]


    @disj.Constraint(doc="Bottom section 1 vapor composition equilibrium summation")
    def bottom_vapor_composition_summation(disj):
        return sum(m.y[1, n_tray, comp] for comp in m.comp) - 1 == m.erry[1, n_tray]


    @disj.Constraint(m.comp,
                     doc="Bottom section 1 vapor composition")
    def bottom_vapor_composition(disj, comp):
        return m.y[1, n_tray, comp] == m.x[1, n_tray, comp] * (
            m.actv[1, n_tray, comp] * (
                m.prop[comp, 'PC'] * exp(
                    m.prop[comp, 'TC'] / m.T[1, n_tray] * (
                        m.prop[comp, 'vpA'] * \
                        (1 - m.T[1, n_tray] / m.prop[comp, 'TC'])
                        + m.prop[comp, 'vpB'] * \
                        (1 - m.T[1, n_tray]/m.prop[comp, 'TC'])**1.5
                        + m.prop[comp, 'vpC'] * \
                        (1 - m.T[1, n_tray]/m.prop[comp, 'TC'])**3
                        + m.prop[comp, 'vpD'] * \
                        (1 - m.T[1, n_tray]/m.prop[comp, 'TC'])**6
                    )
                )
            )
        ) / m.P[1, n_tray]


    
    @disj.Constraint(m.comp,
                     doc="Bottom section 1 liquid enthalpy")
    def bottom_liquid_enthalpy(disj, comp):
        return m.hl[1, n_tray, comp] == (
            m.cpc[1] * (
                (m.T[1, n_tray] - m.Tref) * \
                m.prop[comp, 'cpA', 1]
                + (m.T[1, n_tray]**2 - m.Tref**2) * \
                m.prop[comp, 'cpB', 1] * m.cpc2['A', 1] / 2
                + (m.T[1, n_tray]**3 - m.Tref**3) * \
                m.prop[comp, 'cpC', 1] * m.cpc2['B', 1] / 3
                + (m.T[1, n_tray]**4 - m.Tref**4) * \
                m.prop[comp, 'cpD', 1] / 4
            ) / m.Hscale
        ) 


    @disj.Constraint(m.comp,
                     doc="Bottom section 1 vapor enthalpy")
    def bottom_vapor_enthalpy(disj, comp):
        return m.hv[1, n_tray, comp] == (
            m.cpc[2] * (
                (m.T[1, n_tray] - m.Tref) * \
                m.prop[comp, 'cpA', 2]
                + (m.T[1, n_tray]**2 - m.Tref**2) * \
                m.prop[comp, 'cpB', 2] * m.cpc2['A', 2] / 2
                + (m.T[1, n_tray]**3 - m.Tref**3) * \
                m.prop[comp, 'cpC', 2] * m.cpc2['B', 2] / 3
                + (m.T[1, n_tray]**4 - m.Tref**4) * \
                m.prop[comp, 'cpD', 2] / 4
            ) / m.Hscale
            + m.dHvap[comp]
        )  

    @disj.Constraint(m.comp,
                     doc="Bottom section 1 liquid activity coefficient")
    def bottom_activity_coefficient(disj, comp):
        return m.actv[1, n_tray, comp] == 1 


    

    
def _build_feed_side_equations(disj, n_tray):
    m = disj.model()

    @disj.Constraint(m.comp,
                     doc="Feed section 2 mass per component balances")
    def _feedside_masspercomponent_balances(disj, comp):
        return ( 
            (m.L[2, n_tray + 1, comp]
             if n_tray < m.num_trays else 0)
            + (m.L[4, 1, comp] * m.dl[2]
               if n_tray == m.num_trays else 0)
            - m.L[2, n_tray, comp] 
            + (m.V[1, m.num_trays, comp] * m.dv[2]
               if n_tray == 1 else 0)
            + (m.V[2, n_tray - 1, comp]
               if n_tray > 1 else 0)
            - m.V[2, n_tray, comp] 
            + (m.F[comp]
               if n_tray == m.feed_tray else 0)
            == 0
        )


    @disj.Constraint(doc="Feed section 2 energy balances")
    def _feedside_energy_balances(disj):
        return (
            sum(
                (m.L[2, n_tray + 1, comp] * m.hl[2, n_tray + 1, comp]
                 if n_tray < m.num_trays else 0)
                + (m.L[4, 1, comp] * m.dl[2] * m.hl[4, 1, comp]
                   if  n_tray == m.num_trays else 0)
                - m.L[2, n_tray, comp] * m.hl[2, n_tray, comp]
                + (m.V[1, m.num_trays, comp] * m.dv[2] * m.hv[1, m.num_trays, comp]
                   if n_tray == 1 else 0)
                + (m.V[2, n_tray - 1, comp] * m.hv[2, n_tray - 1, comp]
                   if n_tray > 1 else 0)
                - m.V[2, n_tray, comp] * m.hv[2, n_tray, comp]
                for comp in m.comp) * m.Qscale
            + sum(
                (m.F[comp] * (m.hlf[comp] * (1 - m.q) + m.hvf[comp] * m.q)
                 if n_tray == m.feed_tray else 0)
                for comp in m.comp) * m.Qscale
            ==0
        )


    @disj.Constraint(m.comp,
                     doc="Feed section 2 liquid flowrate per component")
    def _feedside_liquid_percomponent(disj, comp):
        return m.L[2, n_tray, comp] == m.Ltotal[2, n_tray] * m.x[2, n_tray, comp]


    @disj.Constraint(m.comp,
                     doc="Feed section 2 vapor flowrate per component")
    def _feedside_vapor_percomponent(disj, comp):
        return m.V[2, n_tray, comp]  == m.Vtotal[2, n_tray] * m.y[2, n_tray, comp]


    @disj.Constraint(doc="Feed section 2 liquid composition equilibrium summation")
    def feedside_liquid_composition_summation(disj):
        return sum(m.x[2, n_tray, comp] for comp in m.comp) - 1 == m.errx[2, n_tray]


    @disj.Constraint(doc="Feed section 2 vapor composition equilibrium summation")
    def feedside_vapor_composition_summation(disj):
        return sum(m.y[2, n_tray, comp] for comp in m.comp) - 1 == m.erry[2, n_tray]


    @disj.Constraint(m.comp,
                     doc="Feed section 2 vapor composition")
    def feedside_vapor_composition(disj, comp):
        return m.y[2, n_tray, comp] == m.x[2, n_tray, comp] * (
            m.actv[2, n_tray, comp] * (
                m.prop[comp, 'PC'] * exp(
                    m.prop[comp, 'TC'] / m.T[2, n_tray] * (
                        m.prop[comp, 'vpA'] * \
                        (1 - m.T[2, n_tray] / m.prop[comp, 'TC'])
                        + m.prop[comp, 'vpB'] * \
                        (1 - m.T[2, n_tray] / m.prop[comp, 'TC'])**1.5
                        + m.prop[comp, 'vpC'] * \
                        (1 - m.T[2, n_tray] / m.prop[comp, 'TC'])**3
                        + m.prop[comp, 'vpD'] * \
                        (1 - m.T[2, n_tray] / m.prop[comp, 'TC'])**6
                    )
                )
            )
        ) / m.P[2, n_tray]



    @disj.Constraint(m.comp,
                     doc="Feed section 2 liquid enthalpy")
    def feedside_liquid_enthalpy(disj, comp):
        return m.hl[2, n_tray, comp] == (
            m.cpc[1] * (
                (m.T[2, n_tray] - m.Tref) * \
                m.prop[comp, 'cpA', 1]
                + (m.T[2, n_tray]**2 - m.Tref**2) * \
                m.prop[comp, 'cpB', 1] * m.cpc2['A', 1] / 2
                + (m.T[2, n_tray]**3 - m.Tref**3) * \
                m.prop[comp, 'cpC', 1] * m.cpc2['B', 1] / 3
                + (m.T[2, n_tray]**4 - m.Tref**4) * \
                m.prop[comp, 'cpD', 1] / 4
            ) / m.Hscale
        ) 


    @disj.Constraint(m.comp,
                     doc="Feed section 2 vapor enthalpy")
    def feedside_vapor_enthalpy(disj, comp):
        return m.hv[2, n_tray, comp] == (
            m.cpc[2] * (
                (m.T[2, n_tray] - m.Tref) * \
                m.prop[comp, 'cpA', 2]
                + (m.T[2, n_tray]**2 - m.Tref**2) * \
                m.prop[comp, 'cpB', 2] * m.cpc2['A', 2] / 2
                + (m.T[2, n_tray]**3 - m.Tref**3) * \
                m.prop[comp, 'cpC', 2] * m.cpc2['B', 2] / 3
                + (m.T[2, n_tray]**4 - m.Tref**4) * \
                m.prop[comp, 'cpD', 2] / 4
            ) / m.Hscale
            + m.dHvap[comp]
        )  

    
    @disj.Constraint(m.comp,
                     doc="Feed section 2 liquid activity coefficient")
    def feedside_activity_coefficient(disj, comp):
        return m.actv[2, n_tray, comp] == 1 




    
def _build_product_side_equations(disj, n_tray):
    m = disj.model()

    @disj.Constraint(m.comp,
                     doc="Product section 3 mass per component balances")
    def _productside_masspercomponent_balances(disj, comp):
        return ( 
            (m.L[3, n_tray + 1, comp]
             if n_tray < m.num_trays else 0)
            + (m.L[4, 1, comp] * m.dl[3]
               if n_tray == m.num_trays else 0)
            - m.L[3, n_tray, comp] 
            + (m.V[1, m.num_trays, comp] * m.dv[3]
               if n_tray == 1 else 0)
            + (m.V[3, n_tray - 1, comp]
               if n_tray > 1 else 0)
            - m.V[3, n_tray, comp]
            - (m.S[1, comp]
               if n_tray == m.sideout1_tray else 0)
            - (m.S[2, comp]
               if n_tray == m.sideout2_tray else 0)
            ==0
        )

 
    @disj.Constraint(doc="Product section 3 energy balances")
    def _productside_energy_balances(disj):
        return (
            sum(
                (m.L[3, n_tray + 1, comp] * m.hl[3, n_tray + 1, comp]
                 if n_tray < m.num_trays else 0)
                + (m.L[4, 1, comp] * m.dl[3] * m.hl[4, 1, comp]
                   if n_tray == m.num_trays else 0)
                - m.L[3, n_tray, comp] * m.hl[3, n_tray, comp]
                + (m.V[1, m.num_trays, comp] * m.dv[3] * m.hv[1, m.num_trays, comp]
                   if n_tray == 1 else 0)
                + (m.V[3, n_tray - 1, comp] * m.hv[3, n_tray - 1, comp]
                   if n_tray > 1 else 0)
                - m.V[3, n_tray, comp] * m.hv[3, n_tray, comp]
                - (m.S[1, comp] * m.hl[3, n_tray, comp]
                   if n_tray == m.sideout1_tray else 0)
                - (m.S[2, comp] * m.hl[3, n_tray, comp]
                   if n_tray == m.sideout2_tray else 0)
                for comp in m.comp) * m.Qscale
            ==0
        )
    

    @disj.Constraint(m.comp,
                     doc="Product section 3 liquid flowrate per component")
    def _productside_liquid_percomponent(disj, comp):
        return m.L[3, n_tray, comp] == m.Ltotal[3, n_tray] * m.x[3, n_tray, comp]


    @disj.Constraint(m.comp,
                     doc="Product section 3 vapor flowrate per component")
    def _productside_vapor_percomponent(disj, comp):
        return m.V[3, n_tray, comp]  == m.Vtotal[3, n_tray] * m.y[3, n_tray, comp]


    @disj.Constraint(doc="Product section 3 liquid composition equilibrium summation")
    def productside_liquid_composition_summation(disj):
        return sum(m.x[3, n_tray, comp] for comp in m.comp) - 1 == m.errx[3, n_tray]


    @disj.Constraint(doc="Product section 3 vapor composition equilibrium summation")
    def productside_vapor_composition_summation(disj):
        return sum(m.y[3, n_tray, comp] for comp in m.comp) - 1 == m.erry[3, n_tray]


    @disj.Constraint(m.comp,
                     doc="Product section 3 vapor composition")
    def productside_vapor_composition(disj, comp):
        return m.y[3, n_tray, comp] == m.x[3, n_tray, comp] * (
            m.actv[3, n_tray, comp] * (
                m.prop[comp, 'PC'] * exp(
                    m.prop[comp, 'TC'] / m.T[3, n_tray] * (
                        m.prop[comp, 'vpA'] * \
                        (1 - m.T[3, n_tray]/m.prop[comp, 'TC'])
                        + m.prop[comp, 'vpB'] * \
                        (1 - m.T[3, n_tray]/m.prop[comp, 'TC'])**1.5
                        + m.prop[comp, 'vpC'] * \
                        (1 - m.T[3, n_tray]/m.prop[comp, 'TC'])**3
                        + m.prop[comp, 'vpD'] * \
                        (1 - m.T[3, n_tray]/m.prop[comp, 'TC'])**6
                    )
                )
            )
        ) / m.P[3, n_tray]



    @disj.Constraint(m.comp,
                     doc="Product section 3 liquid enthalpy")
    def productside_liquid_enthalpy(disj, comp):
        return m.hl[3, n_tray, comp] == (
            m.cpc[1] * (
                (m.T[3, n_tray] - m.Tref) * \
                m.prop[comp, 'cpA', 1]
                + (m.T[3, n_tray]**2 - m.Tref**2) * \
                m.prop[comp, 'cpB', 1] * m.cpc2['A', 1] / 2
                + (m.T[3, n_tray]**3 - m.Tref**3) * \
                m.prop[comp, 'cpC', 1] * m.cpc2['B', 1] / 3
                + (m.T[3, n_tray]**4 - m.Tref**4) * \
                m.prop[comp, 'cpD', 1] / 4
            ) / m.Hscale
        ) 


    @disj.Constraint(m.comp,
                     doc="Product section 3 vapor enthalpy")
    def productside_vapor_enthalpy(disj, comp):
        return m.hv[3, n_tray, comp] == (
            m.cpc[2] * (
                (m.T[3, n_tray] - m.Tref) * \
                m.prop[comp, 'cpA', 2]
                + (m.T[3, n_tray]**2 - m.Tref**2) * \
                m.prop[comp, 'cpB', 2] * m.cpc2['A', 2] / 2
                + (m.T[3, n_tray]**3 - m.Tref**3) * \
                m.prop[comp, 'cpC', 2] * m.cpc2['B', 2] / 3
                + (m.T[3, n_tray]**4 - m.Tref**4) * \
                m.prop[comp, 'cpD', 2] / 4
            ) / m.Hscale
            + m.dHvap[comp]
        )  


    @disj.Constraint(m.comp,
                     doc="Product section 3 liquid activity coefficient")
    def productside_activity_coefficient(disj, comp):
        return m.actv[3, n_tray, comp] == 1 



    

def _build_top_equations(disj, n_tray):
    m = disj.model()

    @disj.Constraint(m.comp,
                     doc="Top section 4 mass per component balances")
    def _top_mass_percomponent_balances(disj, comp):
        return (
            (m.L[4, n_tray + 1, comp]
             if n_tray < m.con_tray else 0)
            - (m.L[4, n_tray, comp] * m.dl[2]
               if n_tray == 1 else 0) 
            - (m.L[4, n_tray, comp] * m.dl[3]
               if n_tray == 1 else 0)
            - (m.L[4, n_tray, comp]
               if n_tray > 1 else 0)
            + (m.V[2, m.num_trays, comp]
               if n_tray == 1 else 0)
            + (m.V[3, m.num_trays, comp]
               if n_tray == 1 else 0)
            + (m.V[4, n_tray - 1, comp]
               if n_tray > 1 else 0)
            - (m.V[4, n_tray, comp]
               if n_tray < m.con_tray else 0)
            - (m.D[comp]
               if n_tray == m.con_tray else 0)
            ==0
        )


    @disj.Constraint(doc="Top scetion 4 energy balances")
    def _top_energy_balances(disj):
        return (
            sum(
                (m.L[4, n_tray + 1, comp] * m.hl[4, n_tray + 1, comp]
                 if n_tray < m.con_tray else 0)
                - (m.L[4, n_tray, comp] * m.dl[2] * m.hl[4, n_tray, comp]
                   if n_tray == 1 else 0) 
                - (m.L[4, n_tray, comp] * m.dl[3] * m.hl[4, n_tray, comp]
                   if n_tray == 1 else 0)
                - (m.L[4, n_tray, comp] * m.hl[4, n_tray, comp]
                   if n_tray > 1 else 0)
                + (m.V[2, m.num_trays, comp] * m.hv[2, m.num_trays, comp]
                   if n_tray == 1 else 0)
                + (m.V[3, m.num_trays, comp] * m.hv[3, m.num_trays, comp]
                   if n_tray == 1 else 0)
                + (m.V[4, n_tray - 1, comp] * m.hv[4, n_tray - 1, comp]
                   if n_tray > 1 else 0)
                - (m.V[4, n_tray, comp] * m.hv[4, n_tray, comp]
                   if n_tray < m.con_tray else 0)
                - (m.D[comp] * m.hl[4, n_tray, comp]
                   if n_tray == m.con_tray else 0)
                for comp in m.comp)  * m.Qscale
            - (m.Qcon if n_tray == m.con_tray else 0)
            ==0
        )


    @disj.Constraint(m.comp,
                     doc="Top section 4 liquid flowrate per component")
    def _top_liquid_percomponent(disj, comp):
        return m.L[4, n_tray, comp] == m.Ltotal[4, n_tray] * m.x[4, n_tray, comp]


    @disj.Constraint(m.comp,
                     doc="Top section 4 vapor flowrate per component")
    def _top_vapor_percomponent(disj, comp):
        return m.V[4, n_tray, comp]  == m.Vtotal[4, n_tray] * m.y[4, n_tray, comp]


    @disj.Constraint(doc="Top section 4 liquid composition equilibrium summation")
    def top_liquid_composition_summation(disj):
        return sum(m.x[4, n_tray, comp] for comp in m.comp) - 1 == m.errx[4, n_tray]


    @disj.Constraint(doc="Top section 4 vapor composition equilibrium summation")
    def top_vapor_composition_summation(disj):
        return sum(m.y[4, n_tray, comp] for comp in m.comp) - 1 == m.erry[4, n_tray]


    @disj.Constraint(m.comp,
                     doc="Top scetion 4 vapor composition")
    def top_vapor_composition(disj, comp):
        return m.y[4, n_tray, comp] == m.x[4, n_tray, comp] * (
            m.actv[4, n_tray, comp] * (
                m.prop[comp, 'PC'] * exp(
                    m.prop[comp, 'TC'] / m.T[4, n_tray] * (
                        m.prop[comp, 'vpA'] * \
                        (1 - m.T[4, n_tray]/m.prop[comp, 'TC'])
                        + m.prop[comp, 'vpB'] * \
                        (1 - m.T[4, n_tray]/m.prop[comp, 'TC'])**1.5
                        + m.prop[comp, 'vpC'] * \
                        (1 - m.T[4, n_tray]/m.prop[comp, 'TC'])**3
                        + m.prop[comp, 'vpD'] * \
                        (1 - m.T[4, n_tray]/m.prop[comp, 'TC'])**6
                    )
                )
            )
        ) / m.P[4, n_tray]



    @disj.Constraint(m.comp,
                     doc="Top section 4 liquid enthalpy")
    def top_liquid_enthalpy(disj, comp):
        return m.hl[4, n_tray, comp] == (
            m.cpc[1] * (
                (m.T[4, n_tray] - m.Tref) * \
                m.prop[comp, 'cpA', 1]
                + (m.T[4, n_tray]**2 - m.Tref**2) * \
                m.prop[comp, 'cpB', 1] * m.cpc2['A', 1] / 2
                + (m.T[4, n_tray]**3 - m.Tref**3) * \
                m.prop[comp, 'cpC', 1] * m.cpc2['B', 1] / 3
                + (m.T[4, n_tray]**4 - m.Tref**4) * \
                m.prop[comp, 'cpD', 1] / 4
            ) / m.Hscale
        ) 


    @disj.Constraint(m.comp,
                     doc="Top section 4 vapor enthalpy")
    def top_vapor_enthalpy(disj, comp):
        return m.hv[4, n_tray, comp] == (
            m.cpc[2] * (
                (m.T[4, n_tray] - m.Tref) * \
                m.prop[comp, 'cpA', 2]
                + (m.T[4, n_tray]**2 - m.Tref**2) * \
                m.prop[comp, 'cpB', 2] * m.cpc2['A', 2] / 2
                + (m.T[4, n_tray]**3 - m.Tref**3) * \
                m.prop[comp, 'cpC', 2] * m.cpc2['B', 2] / 3
                + (m.T[4, n_tray]**4 - m.Tref**4) * \
                m.prop[comp, 'cpD', 2] / 4
            ) / m.Hscale
            + m.dHvap[comp]
        )  

    
    @disj.Constraint(m.comp,
                     doc="Top section 4 liquid activity coefficient")
    def top_activity_coefficient(disj, comp):
        return m.actv[4, n_tray, comp] == 1 



    
def _build_pass_through_eqns(disj, sec, n_tray):
    m = disj.model()

    if n_tray == 1 or n_tray == m.num_trays:
        return 
        
    @disj.Constraint(m.comp,
                     doc="Pass through liquid flowrate")
    def pass_through_liquid_flowrate(disj, comp):
        return m.L[sec, n_tray, comp] == m.L[sec, n_tray + 1, comp]


    @disj.Constraint(m.comp,
                     doc="Pass through vapor flowrate")
    def pass_through_vapor_flowrate(disj, comp):
        return m.V[sec, n_tray, comp] == m.V[sec, n_tray - 1, comp]


    @disj.Constraint(m.comp,
                     doc="Pass through liquid composition")
    def pass_through_liquid_composition(disj, comp):
        return m.x[sec, n_tray, comp] == m.x[sec, n_tray + 1, comp]


    @disj.Constraint(m.comp,
                     doc="Pass through vapor composition")
    def pass_through_vapor_composition(disj, comp):
        return m.y[sec, n_tray, comp] == m.y[sec, n_tray + 1, comp]


    @disj.Constraint(m.comp,
                     doc="Pass through liquid enthalpy")
    def pass_through_liquid_enthalpy(disj, comp):
        return m.hl[sec, n_tray, comp] == m.hl[sec, n_tray + 1, comp]


    @disj.Constraint(m.comp,
                     doc="Pass through vapor enthalpy")
    def pass_through_vapor_enthalpy(disj, comp):
        return m.hv[sec, n_tray, comp] == m.hv[sec, n_tray - 1, comp]


    @disj.Constraint(doc="Pass through temperature")
    def pass_through_temperature(disj):
        return m.T[sec, n_tray] == m.T[sec, n_tray - 1]
 
    
    

if __name__ == "__main__":
    model = build_model()

