"""Properties of the system"""

from pyomo.environ import ConcreteModel


def get_model_with_properties():
    """
    Attach properties to the model and return the updated model.
    The properties are the physical properties of the components in the system and constants for the calculation of liquid heat capacity.
    It also includes the known initial values and scaling factors for the system, such as the number of trays, components, and flowrates.
    Specifications for the product and the column are also included.
    Case Study: methanol (1), ethanol (2), propanol (3), and butanol (4)

    Returns:
        Pyomo ConcreteModel: The Pyomo model object with attached properties.
    """

    m = ConcreteModel("Properties of the system")

    # ------------------------------------------------------------------
    #                              Data
    # ------------------------------------------------------------------

    m.np = 25  # Number of possible trays. Upper bound for each section.
    m.c = 4  # Number of components. Methanol (1), ethanol (2), propanol (3), and butanol (4)
    m.lc = 1  # Light component, methanol
    m.hc = 4  # Heavy component, butanol

    #### Constant parameters
    m.Rgas = 8.314  # Ideal gas constant [J/mol/K]
    m.Tref = 298.15  # Reference temperature [K]

    #### Product specifications
    m.xspec_lc = 0.99  # Final liquid composition for methanol (1)
    m.xspec_hc = 0.99  # Fnal liquid composition for butanol (4)
    m.xspec_inter2 = 0.99  # Final liquid composition for ethanol (2)
    m.xspec_inter3 = 0.99  # Final liquid composition for propanol (3)
    m.Ddes = 50  # Final flowrate in distillate [mol/s]
    m.Bdes = 50  # Final flowrate in bottoms [mol/s]
    m.Sdes = 50  # Final flowrate in side product streams [mol/s]

    # #### Known initial values
    m.Fi = m.Ddes + m.Bdes + 2 * m.Sdes  # Side feed flowrate [mol/s]
    m.Vi = 400  # Initial value for vapor flowrate [mol/s]
    m.Li = 400  # Initial value for liquid flowrate [mol/s]

    m.Tf = 358  # Side feed temperature [K]

    m.Preb = 1.2  # Reboiler pressure [bar]
    m.Pbot = 1.12  # Bottom-most tray pressure [bar]
    m.Ptop = 1.08  # Top-most tray pressure [bar]
    m.Pcon = 1.05  # Condenser pressure [bar]
    m.Pf = 1.02  # Column pressure [bar]

    m.rr0 = 0.893  # Internal reflux ratio initial value
    m.bu0 = 0.871  # Internal reflux ratio initial value

    #### Scaling factors
    m.Hscale = 1e3
    m.Qscale = 1e-3

    #### Constants for the calculation of liquid heat capacity
    m.cpc = {}  # Constant 1 for liquid heat capacity
    m.cpc2 = {}  # Constant 2 for liquid heat capacity
    m.cpc[1] = m.Rgas
    m.cpc[2] = 1
    m.cpc2['A', 1] = 1 / 100
    m.cpc2['B', 1] = 1 / 1e4
    m.cpc2['A', 2] = 1
    m.cpc2['B', 2] = 1

    # ------------------------------------------------------------------
    #                          Physical Properties
    #
    # Notation:
    # MW ........................ molecular weight [g/mol]
    # TB ........................ boiling point temperature [K]
    # TC ........................ critical temperature [K]
    # PC ........................ critical pressure [bar]
    # w  ........................ acentric factor
    # lden ...................... liquid density [g/m3],
    # dHvap ..................... heat of vaporization [J/mol].
    # vpA, vpB, vpC, and vpD .... vapor pressure constants
    # cpA, cpB, cpC, and cpD .... heat capacity constants [J/mol]:
    #                             1 for liq and 2 for vapor phase
    #
    # Reference A: R.C. Reid, J.M. Prausnitz and B.E. Poling,
    # "The Properties of gases and liquids", 1987 and 2004 Eds.
    #
    # ------------------------------------------------------------------

    m.prop = {}  # Properties of components:
    cpL = {}  # Ruczika-D method for liquid heat capacity calculation
    # (Reference A, page 6.20)
    sumA = {}
    sumB = {}
    sumC = {}
    cpL['a', 'C(H3)(C)'] = 4.19845
    cpL['b', 'C(H3)(C)'] = -0.312709
    cpL['c', 'C(H3)(C)'] = 0.178609
    cpL['a', 'C(H2)(C2)'] = 2.7345
    cpL['b', 'C(H2)(C2)'] = 0.122732
    cpL['c', 'C(H2)(C2)'] = -0.123482
    cpL['a', 'C(H2)(C)(O)'] = 0.517007
    cpL['b', 'C(H2)(C)(O)'] = 1.26631
    cpL['c', 'C(H2)(C)(O)'] = -0.0939713
    cpL['a', 'O(H)(C)'] = 16.1555
    cpL['b', 'O(H)(C)'] = -11.938
    cpL['c', 'O(H)(C)'] = 2.85117
    cpL['a', 'C(H3)(O)'] = 3.70344
    cpL['b', 'C(H3)(O)'] = -1.12884
    cpL['c', 'C(H3)(O)'] = 0.51239
    sumA[1] = cpL['a', 'C(H3)(O)'] + cpL['a', 'O(H)(C)']
    sumB[1] = cpL['b', 'C(H3)(O)'] + cpL['b', 'O(H)(C)']
    sumC[1] = cpL['c', 'C(H3)(O)'] + cpL['c', 'O(H)(C)']
    sumA[2] = cpL['a', 'C(H3)(C)'] + cpL['a', 'C(H2)(C)(O)'] + cpL['a', 'O(H)(C)']
    sumB[2] = cpL['b', 'C(H3)(C)'] + cpL['b', 'C(H2)(C)(O)'] + cpL['b', 'O(H)(C)']
    sumC[2] = cpL['c', 'C(H3)(C)'] + cpL['c', 'C(H2)(C)(O)'] + cpL['c', 'O(H)(C)']
    sumA[3] = (
        cpL['a', 'C(H3)(C)']
        + cpL['a', 'C(H2)(C2)']
        + cpL['a', 'C(H2)(C)(O)']
        + cpL['a', 'O(H)(C)']
    )
    sumB[3] = (
        cpL['b', 'C(H3)(C)']
        + cpL['b', 'C(H2)(C2)']
        + cpL['b', 'C(H2)(C)(O)']
        + cpL['b', 'O(H)(C)']
    )
    sumC[3] = (
        cpL['c', 'C(H3)(C)']
        + cpL['c', 'C(H2)(C2)']
        + cpL['c', 'C(H2)(C)(O)']
        + cpL['c', 'O(H)(C)']
    )
    sumA[4] = (
        cpL['a', 'C(H3)(C)']
        + 2 * cpL['a', 'C(H2)(C2)']
        + cpL['a', 'C(H2)(C)(O)']
        + cpL['a', 'O(H)(C)']
    )
    sumB[4] = (
        cpL['b', 'C(H3)(C)']
        + 2 * cpL['b', 'C(H2)(C2)']
        + cpL['b', 'C(H2)(C)(O)']
        + cpL['b', 'O(H)(C)']
    )
    sumC[4] = (
        cpL['c', 'C(H3)(C)']
        + 2 * cpL['c', 'C(H2)(C2)']
        + cpL['c', 'C(H2)(C)(O)']
        + cpL['c', 'O(H)(C)']
    )

    ## Methanol: component 1
    m.prop[1, 'MW'] = 32.042
    m.prop[1, 'TB'] = 337.7
    m.prop[1, 'TC'] = 512.6
    m.prop[1, 'PC'] = 80.9
    m.prop[1, 'w'] = 0.556
    m.prop[1, 'lden'] = 792e3
    m.prop[1, 'dHvap'] = 38.376e3
    m.prop[1, 'vpA'] = -8.54796
    m.prop[1, 'vpB'] = 0.76982
    m.prop[1, 'vpC'] = -3.10850
    m.prop[1, 'vpD'] = 1.54481
    m.prop[1, 'cpA', 1] = sumA[1]
    m.prop[1, 'cpB', 1] = sumB[1]
    m.prop[1, 'cpC', 1] = sumC[1]
    m.prop[1, 'cpD', 1] = 0
    m.prop[1, 'cpA', 2] = 2.115e1
    m.prop[1, 'cpB', 2] = 7.092e-2
    m.prop[1, 'cpC', 2] = 2.587e-5
    m.prop[1, 'cpD', 2] = -2.852e-8

    ## Ethanol: component 2
    m.prop[2, 'MW'] = 46.069
    m.prop[2, 'TB'] = 351.4
    m.prop[2, 'TC'] = 513.9
    m.prop[2, 'PC'] = 61.4
    m.prop[2, 'w'] = 0.644
    m.prop[2, 'lden'] = 789.3e3
    m.prop[2, 'dHvap'] = 42.698e3
    m.prop[2, 'vpA'] = -8.51838
    m.prop[2, 'vpB'] = 0.34163
    m.prop[2, 'vpC'] = -5.73683
    m.prop[2, 'vpD'] = 8.32581
    m.prop[2, 'cpA', 1] = sumA[2]
    m.prop[2, 'cpB', 1] = sumB[2]
    m.prop[2, 'cpC', 1] = sumC[2]
    m.prop[2, 'cpD', 1] = 0
    m.prop[2, 'cpA', 2] = 9.014
    m.prop[2, 'cpB', 2] = 2.141e-1
    m.prop[2, 'cpC', 2] = -8.390e-5
    m.prop[2, 'cpD', 2] = 1.373e-9

    ## Propanol: component 3
    m.prop[3, 'MW'] = 60.096
    m.prop[3, 'TB'] = 370.3
    m.prop[3, 'TC'] = 536.8
    m.prop[3, 'PC'] = 51.7
    m.prop[3, 'w'] = 0.623
    m.prop[3, 'lden'] = 804e3
    m.prop[3, 'dHvap'] = 47.763e3
    m.prop[3, 'vpA'] = -8.05594
    m.prop[3, 'vpB'] = 4.25183e-2
    m.prop[3, 'vpC'] = -7.51296
    m.prop[3, 'vpD'] = 6.89004
    m.prop[3, 'cpA', 1] = sumA[3]
    m.prop[3, 'cpB', 1] = sumB[3]
    m.prop[3, 'cpC', 1] = sumC[3]
    m.prop[3, 'cpD', 1] = 0
    m.prop[3, 'cpA', 2] = 2.47
    m.prop[3, 'cpB', 2] = 3.325e-1
    m.prop[3, 'cpC', 2] = -1.855e-4
    m.prop[3, 'cpD', 2] = 4.296e-8

    ## Butanol: component 4
    m.prop[4, 'MW'] = 74.123
    m.prop[4, 'TB'] = 390.9
    m.prop[4, 'TC'] = 563.1
    m.prop[4, 'PC'] = 44.2
    m.prop[4, 'w'] = 0.593
    m.prop[4, 'lden'] = 810e3
    m.prop[4, 'dHvap'] = 52.607e3
    m.prop[4, 'vpA'] = -8.00756
    m.prop[4, 'vpB'] = 0.53783
    m.prop[4, 'vpC'] = -9.34240
    m.prop[4, 'vpD'] = 6.68692
    m.prop[4, 'cpA', 1] = sumA[4]
    m.prop[4, 'cpB', 1] = sumB[4]
    m.prop[4, 'cpC', 1] = sumC[4]
    m.prop[4, 'cpD', 1] = 0
    m.prop[4, 'cpA', 2] = 3.266
    m.prop[4, 'cpB', 2] = 4.18e-1
    m.prop[4, 'cpC', 2] = -2.242e-4
    m.prop[4, 'cpD', 2] = 4.685e-8

    return m
