"""
HDA_GDP_gdpopt.py
This model describes the profit maximization of a Hydrodealkylation of Toluene process, first presented in Reference [1], and later implemented as a GDP in Reference [2]. The MINLP formulation of this problem is available in GAMS, Reference [3].

The chemical plant performed the hydro-dealkylation of toluene into benzene and methane. The flowsheet model was used to make decisions on choosing between alternative process units at various stages of the process. The resulting model is GDP model. The disjunctions in the model include:
    1. Inlet purify selection at feed
    2. Reactor operation mode selection (adiabatic / isothermal)
    3. Vapor recovery methane purge / recycle with membrane
    4. Vapor recovery hydrogen recycle
    5. Liquid separation system methane stabilizing via column or flash drum
    6. Liquid separation system toluene recovery via column or flash drum

The model enforces constraints to ensure that the mass and energy balances are satisfied, the purity of the products is within the required limits, the recovery specification are met, and the temperature and pressure conditions in the process units are maintained within the operational limits.

The objective of the model is to maximize the profit by determining the optimal process configuration and operating conditions. The decision variables include the number of trays in the absorber and distillation column, the reflux ratio, the pressure in the distillation column, the temperature and pressure in the flash drums, the heating requirement in the furnace, the electricity requirement in the compressor, the heat exchange in the coolers and heaters, the surface area in the membrane separators, the temperature and pressure in the mixers, the temperature and pressure in the reactors, and the volume and rate constant in the reactors.

References:
    [1] James M Douglas (1988). Conceptual Design of Chemical Processes, McGraw-Hill. ISBN-13: 978-0070177628
    [2] G.R. Kocis, and I.E. Grossmann (1989). Computational Experience with DICOPT Solving MINLP Problems in Process Synthesis. Computers and Chemical Engineering 13, 3, 307-315. https://doi.org/10.1016/0098-1354(89)85008-2
    [3] GAMS Development Corporation (2023). Hydrodealkylation Process. Available at: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_hda.html
"""

import math
import os
import pandas as pd

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.util.infeasible import log_infeasible_constraints


def HDA_model():
    """
    Builds the Hydrodealkylation of Toluene process model.

    Parameters
    ----------
    alpha : float
        compressor coefficient
    compeff : float
        compressor efficiency
    cp_cv_ratio : float
        ratio of cp to cv
    abseff : float
        absorber tray efficiency
    disteff : float
        column tray efficiency
    uflow : float
        upper bound - flow logicals
    upress : float
        upper bound - pressure logicals
    utemp : float
        upper bound - temperature logicals
    costelec : float
        electricity cost
    costqc : float
        cooling cost
    costqh : float
        heating cost
    costfuel : float
        fuel cost furnace
    furnpdrop : float
        pressure drop of furnace
    heatvap : float
        heat of vaporization [kJ/kg-mol]
    cppure : float
        pure component heat capacities [kJ/kg-mol-K]
    gcomp : float
        guess composition values [mol/mol]
    cp : float
        heat capacities [kJ/kg-mol-K]
    anta : float
        antoine coefficient A
    antb : float
        antoine coefficient B
    antc : float
        antoine coefficient C
    perm : float
        permeability [kg-mol/m**2-min-MPa]
    cbeta : float
        constant values (exp(beta)) in absorber
    aabs : float
        absorption factors
    eps1 : float
        small number to avoid division by zero
    heatrxn : float
        heat of reaction [kJ/kg-mol]
    f1comp : float
        feedstock compositions (h2 feed) [mol/mol]
    f66comp : float
        feedstock compositions (tol feed) [mol/mol]
    f67comp : float
        feedstock compositions (tol feed) [mol/mol]

    Sets
    ----
    str : int
        process streams
    compon : str
        chemical components
    abs : int
        absorber
    comp : int
        compressor
    dist : int
        distillation column
    flsh : int
        flash drums
    furn : int
        furnace
    hec : int
        coolers
    heh : int
        heaters
    exch : int
        heat exchangers
    memb : int
        membrane separators
    mxr1 : int
        single inlet stream mixers
    mxr : int
        mixers
    pump : int
        pumps
    rct : int
        reactors
    spl1 : int
        single outlet stream splitters
    spl : int
        splitter
    valve : int
        expansion valve
    str2 : int
        process streams
    compon2 : str
        chemical components


    Returns
    -------
    m : Pyomo ConcreteModel
        Pyomo model of the Hydrodealkylation of Toluene process

    """
    dir_path = os.path.dirname(os.path.abspath(__file__))

    m = ConcreteModel()

    # ## scalars

    m.alpha = Param(initialize=0.3665, doc="compressor coefficient")
    m.compeff = Param(initialize=0.750, doc="compressor efficiency")
    m.cp_cv_ratio = Param(initialize=1.300, doc="ratio of cp to cv")
    m.abseff = Param(initialize=0.333, doc="absorber tray efficiency")
    m.disteff = Param(initialize=0.5000, doc="column tray efficiency")
    m.uflow = Param(initialize=50, doc="upper bound - flow logicals")
    m.upress = Param(initialize=4.0, doc="upper bound - pressure logicals")
    m.utemp = Param(initialize=7.0, doc="upper bound - temperature logicals")
    m.costelec = Param(initialize=0.340, doc="electricity cost")
    m.costqc = Param(initialize=0.7000, doc="cooling cost")
    m.costqh = Param(initialize=8.0000, doc="heating cost")
    m.costfuel = Param(initialize=4.0, doc="fuel cost furnace")
    m.furnpdrop = Param(initialize=0.4826, doc="pressure drop of furnace")

    # ## sets

    def strset(i):
        """
        Process streams

        Returns
        -------
        s : list
            integer list from 1 to 74
        """
        s = []
        i = 1
        for i in range(1, 36):
            s.append(i)
            i += i
        i = 37
        for i in range(37, 74):
            s.append(i)
            i += i
        return s

    m.str = Set(initialize=strset, doc="process streams")
    m.compon = Set(
        initialize=["h2", "ch4", "ben", "tol", "dip"], doc="chemical components"
    )
    m.abs = RangeSet(1)
    m.comp = RangeSet(4)
    m.dist = RangeSet(3)
    m.flsh = RangeSet(3)
    m.furn = RangeSet(1)
    m.hec = RangeSet(2)
    m.heh = RangeSet(4)
    m.exch = RangeSet(1)
    m.memb = RangeSet(2)
    m.mxr1 = RangeSet(5)
    m.mxr = RangeSet(5)
    m.pump = RangeSet(2)
    m.rct = RangeSet(2)
    m.spl1 = RangeSet(6)
    m.spl = RangeSet(3)
    m.valve = RangeSet(6)
    m.str2 = Set(initialize=strset, doc="process streams")
    m.compon2 = Set(
        initialize=["h2", "ch4", "ben", "tol", "dip"], doc="chemical components"
    )

    # parameters
    Heatvap = {}
    Heatvap["tol"] = 30890.00
    m.heatvap = Param(
        m.compon, initialize=Heatvap, default=0, doc="heat of vaporization [kJ/kg-mol]"
    )
    Cppure = {}
    # h2 'hydrogen', ch4 'methane', ben 'benzene', tol 'toluene', dip 'diphenyl'
    Cppure["h2"] = 30
    Cppure["ch4"] = 40
    Cppure["ben"] = 225
    Cppure["tol"] = 225
    Cppure["dip"] = 450
    m.cppure = Param(
        m.compon,
        initialize=Cppure,
        default=0,
        doc="pure component heat capacities [kJ/kg-mol-K]",
    )
    Gcomp = {}
    Gcomp[7, "h2"] = 0.95
    Gcomp[7, "ch4"] = 0.05
    Gcomp[8, "h2"] = 0.5
    Gcomp[8, "ch4"] = 0.40
    Gcomp[8, "tol"] = 0.1
    Gcomp[9, "h2"] = 0.5
    Gcomp[9, "ch4"] = 0.40
    Gcomp[9, "tol"] = 0.1
    Gcomp[10, "h2"] = 0.5
    Gcomp[10, "ch4"] = 0.40
    Gcomp[10, "tol"] = 0.1
    Gcomp[11, "h2"] = 0.45
    Gcomp[11, "ben"] = 0.05
    Gcomp[11, "ch4"] = 0.45
    Gcomp[11, "tol"] = 0.05
    Gcomp[12, "h2"] = 0.50
    Gcomp[12, "ch4"] = 0.40
    Gcomp[12, "tol"] = 0.10
    Gcomp[13, "h2"] = 0.45
    Gcomp[13, "ch4"] = 0.45
    Gcomp[13, "ben"] = 0.05
    Gcomp[13, "tol"] = 0.05
    Gcomp[14, "h2"] = 0.45
    Gcomp[14, "ch4"] = 0.45
    Gcomp[14, "ben"] = 0.05
    Gcomp[14, "tol"] = 0.05
    Gcomp[15, "h2"] = 0.45
    Gcomp[15, "ch4"] = 0.45
    Gcomp[15, "ben"] = 0.05
    Gcomp[15, "tol"] = 0.05
    Gcomp[16, "h2"] = 0.4
    Gcomp[16, "ch4"] = 0.4
    Gcomp[16, "ben"] = 0.1
    Gcomp[16, "tol"] = 0.1
    Gcomp[17, "h2"] = 0.40
    Gcomp[17, "ch4"] = 0.40
    Gcomp[17, "ben"] = 0.1
    Gcomp[17, "tol"] = 0.1
    Gcomp[20, "h2"] = 0.03
    Gcomp[20, "ch4"] = 0.07
    Gcomp[20, "ben"] = 0.55
    Gcomp[20, "tol"] = 0.35
    Gcomp[21, "h2"] = 0.03
    Gcomp[21, "ch4"] = 0.07
    Gcomp[21, "ben"] = 0.55
    Gcomp[21, "tol"] = 0.35
    Gcomp[22, "h2"] = 0.03
    Gcomp[22, "ch4"] = 0.07
    Gcomp[22, "ben"] = 0.55
    Gcomp[22, "tol"] = 0.35
    Gcomp[24, "h2"] = 0.03
    Gcomp[24, "ch4"] = 0.07
    Gcomp[24, "ben"] = 0.55
    Gcomp[24, "tol"] = 0.35
    Gcomp[25, "h2"] = 0.03
    Gcomp[25, "ch4"] = 0.07
    Gcomp[25, "ben"] = 0.55
    Gcomp[25, "tol"] = 0.35
    Gcomp[37, "tol"] = 1.00
    Gcomp[38, "tol"] = 1.00
    Gcomp[43, "ben"] = 0.05
    Gcomp[43, "tol"] = 0.95
    Gcomp[44, "h2"] = 0.03
    Gcomp[44, "ch4"] = 0.07
    Gcomp[44, "ben"] = 0.55
    Gcomp[44, "tol"] = 0.35
    Gcomp[45, "h2"] = 0.03
    Gcomp[45, "ch4"] = 0.07
    Gcomp[45, "ben"] = 0.55
    Gcomp[45, "tol"] = 0.35
    Gcomp[46, "h2"] = 0.03
    Gcomp[46, "ch4"] = 0.07
    Gcomp[46, "ben"] = 0.55
    Gcomp[46, "tol"] = 0.35
    Gcomp[51, "h2"] = 0.30
    Gcomp[51, "ch4"] = 0.70
    Gcomp[57, "h2"] = 0.80
    Gcomp[57, "ch4"] = 0.20
    Gcomp[60, "h2"] = 0.50
    Gcomp[60, "ch4"] = 0.50
    Gcomp[62, "h2"] = 0.50
    Gcomp[62, "ch4"] = 0.50
    Gcomp[63, "h2"] = 0.47
    Gcomp[63, "ch4"] = 0.40
    Gcomp[63, "ben"] = 0.01
    Gcomp[63, "tol"] = 0.12
    Gcomp[65, "h2"] = 0.50
    Gcomp[65, "ch4"] = 0.50
    Gcomp[66, "tol"] = 1.0
    Gcomp[69, "tol"] = 1.0
    Gcomp[70, "h2"] = 0.5
    Gcomp[70, "ch4"] = 0.4
    Gcomp[70, "tol"] = 0.10
    Gcomp[71, "h2"] = 0.40
    Gcomp[71, "ch4"] = 0.40
    Gcomp[71, "ben"] = 0.10
    Gcomp[71, "tol"] = 0.10
    Gcomp[72, "h2"] = 0.50
    Gcomp[72, "ch4"] = 0.50
    m.gcomp = Param(
        m.str,
        m.compon,
        initialize=Gcomp,
        default=0,
        doc="guess composition values [mol/mol]",
    )

    def cppara(compon, stream):
        """
        heat capacities [kJ/kg-mol-K]
        sum of heat capacities of all components in a stream, weighted by their composition
        """
        return sum(m.cppure[compon] * m.gcomp[stream, compon] for compon in m.compon)

    m.cp = Param(
        m.str, initialize=cppara, default=0, doc="heat capacities [kJ/kg-mol-K]"
    )

    Anta = {}
    Anta["h2"] = 13.6333
    Anta["ch4"] = 15.2243
    Anta["ben"] = 15.9008
    Anta["tol"] = 16.0137
    Anta["dip"] = 16.6832
    m.anta = Param(m.compon, initialize=Anta, default=0, doc="antoine coefficient A")

    Antb = {}
    Antb["h2"] = 164.9
    Antb["ch4"] = 897.84
    Antb["ben"] = 2788.51
    Antb["tol"] = 3096.52
    Antb["dip"] = 4602.23
    m.antb = Param(m.compon, initialize=Antb, default=0, doc="antoine coefficient B")

    Antc = {}
    Antc["h2"] = 3.19
    Antc["ch4"] = -7.16
    Antc["ben"] = -52.36
    Antc["tol"] = -53.67
    Antc["dip"] = -70.42
    m.antc = Param(m.compon, initialize=Antc, default=0, doc="antoine coefficient C")

    Perm = {}
    for i in m.compon:
        Perm[i] = 0
    Perm["h2"] = 55.0e-06
    Perm["ch4"] = 2.3e-06

    def Permset(m, compon):
        """
        permeability [kg-mol/m**2-min-MPa]
        converting unit for permeability from [cc/cm**2-sec-cmHg] to [kg-mol/m**2-min-MPa]
        """
        return Perm[compon] * (1.0 / 22400.0) * 1.0e4 * 750.062 * 60.0 / 1000.0

    m.perm = Param(
        m.compon,
        initialize=Permset,
        default=0,
        doc="permeability [kg-mol/m**2-min-MPa]",
    )

    Cbeta = {}
    Cbeta["h2"] = 1.0003
    Cbeta["ch4"] = 1.0008
    Cbeta["dip"] = 1.0e04
    m.cbeta = Param(
        m.compon,
        initialize=Cbeta,
        default=0,
        doc="constant values (exp(beta)) in absorber",
    )

    Aabs = {}
    Aabs["ben"] = 1.4
    Aabs["tol"] = 4.0
    m.aabs = Param(m.compon, initialize=Aabs, default=0, doc="absorption factors")
    m.eps1 = Param(initialize=1e-4, doc="small number to avoid division by zero")

    Heatrxn = {}
    Heatrxn[1] = 50100.0
    Heatrxn[2] = 50100.0
    m.heatrxn = Param(
        m.rct, initialize=Heatrxn, default=0, doc="heat of reaction [kJ/kg-mol]"
    )

    F1comp = {}
    F1comp["h2"] = 0.95
    F1comp["ch4"] = 0.05
    F1comp["dip"] = 0.00
    F1comp["ben"] = 0.00
    F1comp["tol"] = 0.00
    m.f1comp = Param(
        m.compon,
        initialize=F1comp,
        default=0,
        doc="feedstock compositions (h2 feed) [mol/mol]",
    )

    F66comp = {}
    F66comp["tol"] = 1.0
    F66comp["h2"] = 0.00
    F66comp["ch4"] = 0.00
    F66comp["dip"] = 0.00
    F66comp["ben"] = 0.00
    m.f66comp = Param(
        m.compon,
        initialize=F66comp,
        default=0,
        doc="feedstock compositions (tol feed) [mol/mol]",
    )

    F67comp = {}
    F67comp["tol"] = 1.0
    F67comp["h2"] = 0.00
    F67comp["ch4"] = 0.00
    F67comp["dip"] = 0.00
    F67comp["ben"] = 0.00
    m.f67comp = Param(
        m.compon,
        initialize=F67comp,
        default=0,
        doc="feedstock compositions (tol feed) [mol/mol]",
    )

    # # matching streams
    m.ilabs = Set(initialize=[(1, 67)], doc="abs-stream (inlet liquid) matches")
    m.olabs = Set(initialize=[(1, 68)], doc="abs-stream (outlet liquid) matches")
    m.ivabs = Set(initialize=[(1, 63)], doc="abs-stream (inlet vapor) matches")
    m.ovabs = Set(initialize=[(1, 64)], doc="abs-stream (outlet vapor) matches")
    m.asolv = Set(initialize=[(1, "tol")], doc="abs-solvent component matches")
    m.anorm = Set(initialize=[(1, "ben")], doc="abs-comp matches (normal model)")
    m.asimp = Set(
        initialize=[(1, "h2"), (1, "ch4"), (1, "dip")],
        doc="abs-heavy component matches",
    )

    m.icomp = Set(
        initialize=[(1, 5), (2, 59), (3, 64), (4, 56)],
        doc="compressor-stream (inlet) matches",
    )
    m.ocomp = Set(
        initialize=[(1, 6), (2, 60), (3, 65), (4, 57)],
        doc="compressor-stream (outlet) matches",
    )

    m.idist = Set(
        initialize=[(1, 25), (2, 30), (3, 33)], doc="dist-stream (inlet) matches"
    )
    m.vdist = Set(
        initialize=[(1, 26), (2, 31), (3, 34)], doc="dist-stream (vapor) matches"
    )
    m.ldist = Set(
        initialize=[(1, 27), (2, 32), (3, 35)], doc="dist-stream (liquid) matches"
    )
    m.dl = Set(
        initialize=[(1, "h2"), (2, "ch4"), (3, "ben")],
        doc="dist-light components matches",
    )
    m.dlkey = Set(
        initialize=[(1, "ch4"), (2, "ben"), (3, "tol")],
        doc="dist-heavy key component matches",
    )
    m.dhkey = Set(
        initialize=[(1, "ben"), (2, "tol"), (3, "dip")],
        doc="dist-heavy components matches",
    )
    m.dh = Set(
        initialize=[(1, "tol"), (1, "dip"), (2, "dip")],
        doc="dist-key component matches",
    )

    i = list(m.dlkey)
    q = list(m.dhkey)
    dkeyset = i + q
    m.dkey = Set(initialize=dkeyset, doc="dist-key component matches")

    m.iflsh = Set(
        initialize=[(1, 17), (2, 46), (3, 39)], doc="flsh-stream (inlet) matches"
    )
    m.vflsh = Set(
        initialize=[(1, 18), (2, 47), (3, 40)], doc="flsh-stream (vapor) matches"
    )
    m.lflsh = Set(
        initialize=[(1, 19), (2, 48), (3, 41)], doc="flsh-stream (liquid) matches"
    )
    m.fkey = Set(
        initialize=[(1, "ch4"), (2, "ch4"), (3, "tol")],
        doc="flash-key component matches",
    )

    m.ifurn = Set(initialize=[(1, 70)], doc="furn-stream (inlet) matches")
    m.ofurn = Set(initialize=[(1, 9)], doc="furn-stream (outlet) matches")

    m.ihec = Set(initialize=[(1, 71), (2, 45)], doc="hec-stream (inlet) matches")
    m.ohec = Set(initialize=[(1, 17), (2, 46)], doc="hec-stream (outlet) matches")

    m.iheh = Set(
        initialize=[(1, 24), (2, 23), (3, 37), (4, 61)],
        doc="heh-stream (inlet) matches",
    )
    m.oheh = Set(
        initialize=[(1, 25), (2, 44), (3, 38), (4, 73)],
        doc="heh-stream (outlet) matches",
    )

    m.icexch = Set(initialize=[(1, 8)], doc="exch-cold stream (inlet)  matches")
    m.ocexch = Set(initialize=[(1, 70)], doc="exch-cold stream (outlet) matches")
    m.ihexch = Set(initialize=[(1, 16)], doc="exch-hot stream (inlet)  matches")
    m.ohexch = Set(initialize=[(1, 71)], doc="exch-hot stream (outlet) matches")

    m.imemb = Set(initialize=[(1, 3), (2, 54)], doc="memb-stream (inlet) matches")
    m.nmemb = Set(
        initialize=[(1, 4), (2, 55)], doc="memb-stream (non-permeate) matches"
    )
    m.pmemb = Set(initialize=[(1, 5), (2, 56)], doc="memb-stream (permeate) matches")
    m.mnorm = Set(
        initialize=[(1, "h2"), (1, "ch4"), (2, "h2"), (2, "ch4")],
        doc="normal components",
    )
    m.msimp = Set(
        initialize=[
            (1, "ben"),
            (1, "tol"),
            (1, "dip"),
            (2, "ben"),
            (2, "tol"),
            (2, "dip"),
        ],
        doc="simplified flux components",
    )

    m.imxr1 = Set(
        initialize=[
            (1, 2),
            (1, 6),
            (2, 11),
            (2, 13),
            (3, 27),
            (3, 48),
            (4, 34),
            (4, 40),
            (5, 49),
            (5, 50),
        ],
        doc="mixer-stream (inlet) matches",
    )
    m.omxr1 = Set(
        initialize=[(1, 7), (2, 14), (3, 30), (4, 42), (5, 51)],
        doc="mixer-stream (outlet) matches",
    )
    m.mxr1spl1 = Set(
        initialize=[
            (1, 2, 2),
            (1, 6, 3),
            (2, 11, 10),
            (2, 13, 12),
            (3, 27, 24),
            (3, 48, 23),
            (4, 34, 33),
            (4, 40, 37),
            (5, 49, 23),
            (5, 50, 24),
        ],
        doc="1-mxr-inlet 1-spl-outlet matches",
    )

    m.imxr = Set(
        initialize=[
            (1, 7),
            (1, 43),
            (1, 66),
            (1, 72),
            (2, 15),
            (2, 20),
            (3, 21),
            (3, 69),
            (4, 51),
            (4, 62),
            (5, 57),
            (5, 60),
            (5, 65),
        ],
        doc="mixer-stream (inlet) matches",
    )
    m.omxr = Set(
        initialize=[(1, 8), (2, 16), (3, 22), (4, 63), (5, 72)],
        doc="mixer-stream (outlet) matches ",
    )

    m.ipump = Set(initialize=[(1, 42), (2, 68)], doc="pump-stream (inlet) matches")
    m.opump = Set(initialize=[(1, 43), (2, 69)], doc="pump-stream (outlet) matches")

    m.irct = Set(initialize=[(1, 10), (2, 12)], doc="reactor-stream (inlet) matches")
    m.orct = Set(initialize=[(1, 11), (2, 13)], doc="reactor-stream (outlet) matches")
    m.rkey = Set(
        initialize=[(1, "tol"), (2, "tol")], doc="reactor-key component matches"
    )

    m.ispl1 = Set(
        initialize=[(1, 1), (2, 9), (3, 22), (4, 32), (5, 52), (6, 58)],
        doc="splitter-stream (inlet) matches",
    )
    m.ospl1 = Set(
        initialize=[
            (1, 2),
            (1, 3),
            (2, 10),
            (2, 12),
            (3, 23),
            (3, 24),
            (4, 33),
            (4, 37),
            (5, 53),
            (5, 54),
            (6, 59),
            (6, 61),
        ],
        doc="splitter-stream (outlet) matches",
    )

    m.ispl = Set(
        initialize=[(1, 19), (2, 18), (3, 26)], doc="splitter-stream (inlet) matches"
    )
    m.ospl = Set(
        initialize=[(1, 20), (1, 21), (2, 52), (2, 58), (3, 28), (3, 29)],
        doc="splitter-stream (outlet) matches",
    )

    m.ival = Set(
        initialize=[(1, 44), (2, 38), (3, 14), (4, 47), (5, 29), (6, 73)],
        doc="exp.valve-stream (inlet) matches",
    )
    m.oval = Set(
        initialize=[(1, 45), (2, 39), (3, 15), (4, 49), (5, 50), (6, 62)],
        doc="exp.valve-stream (outlet) matches",
    )

    # variables

    # absorber
    m.nabs = Var(
        m.abs,
        within=NonNegativeReals,
        bounds=(0, 40),
        initialize=1,
        doc="number of absorber trays",
    )
    m.gamma = Var(m.abs, m.compon, within=Reals, initialize=1, doc="gamma")
    m.beta = Var(m.abs, m.compon, within=Reals, initialize=1, doc="beta")

    # compressor
    m.elec = Var(
        m.comp,
        within=NonNegativeReals,
        bounds=(0, 100),
        initialize=1,
        doc="electricity requirement [kW]",
    )
    m.presrat = Var(
        m.comp,
        within=NonNegativeReals,
        bounds=(1, 8 / 3),
        initialize=1,
        doc="ratio of outlet to inlet pressure",
    )

    # distillation
    m.nmin = Var(
        m.dist,
        within=NonNegativeReals,
        initialize=1,
        doc="minimum number of trays in column",
    )
    m.ndist = Var(
        m.dist, within=NonNegativeReals, initialize=1, doc="number of trays in column"
    )
    m.rmin = Var(
        m.dist, within=NonNegativeReals, initialize=1, doc="minimum reflux ratio"
    )
    m.reflux = Var(m.dist, within=NonNegativeReals, initialize=1, doc="reflux ratio")
    m.distp = Var(
        m.dist,
        within=NonNegativeReals,
        initialize=1,
        bounds=(0.1, 4.0),
        doc="column pressure [MPa]",
    )
    m.avevlt = Var(
        m.dist, within=NonNegativeReals, initialize=1, doc="average volatility"
    )

    # flash
    m.flsht = Var(
        m.flsh, within=NonNegativeReals, initialize=1, doc="flash temperature [100 K]"
    )
    m.flshp = Var(
        m.flsh, within=NonNegativeReals, initialize=1, doc="flash pressure [MPa]"
    )
    m.eflsh = Var(
        m.flsh,
        m.compon,
        within=NonNegativeReals,
        bounds=(0, 1),
        initialize=0.5,
        doc="vapor phase recovery in flash",
    )

    # furnace
    m.qfuel = Var(
        m.furn,
        within=NonNegativeReals,
        bounds=(None, 10),
        initialize=1,
        doc="heating required [1.e+12 kJ/yr]",
    )
    # cooler
    m.qc = Var(
        m.hec,
        within=NonNegativeReals,
        bounds=(None, 10),
        initialize=1,
        doc="utility requirement [1.e+12 kJ/yr]",
    )
    # heater
    m.qh = Var(
        m.heh,
        within=NonNegativeReals,
        bounds=(None, 10),
        initialize=1,
        doc="utility requirement [1.e+12 kJ/yr]",
    )
    # exchanger
    m.qexch = Var(
        m.exch,
        within=NonNegativeReals,
        bounds=(None, 10),
        initialize=1,
        doc="heat exchanged [1.e+12 kJ/yr]",
    )
    # membrane
    m.a = Var(
        m.memb,
        within=NonNegativeReals,
        bounds=(100, 10000),
        initialize=1,
        doc="surface area for mass transfer [m**2]",
    )
    # mixer(1 input)
    m.mxr1p = Var(
        m.mxr1,
        within=NonNegativeReals,
        bounds=(0.1, 4),
        initialize=0,
        doc="mixer temperature [100 K]",
    )
    m.mxr1t = Var(
        m.mxr1,
        within=NonNegativeReals,
        bounds=(3, 10),
        initialize=0,
        doc="mixer pressure [MPa]",
    )
    # mixer
    m.mxrt = Var(
        m.mxr,
        within=NonNegativeReals,
        bounds=(3.0, 10),
        initialize=3,
        doc="mixer temperature [100 K]",
    )
    m.mxrp = Var(
        m.mxr,
        within=NonNegativeReals,
        bounds=(0.1, 4.0),
        initialize=3,
        doc="mixer pressure [MPa]",
    )
    # reactor
    m.rctt = Var(
        m.rct,
        within=NonNegativeReals,
        bounds=(8.9427, 9.7760),
        doc="reactor temperature [100 K]",
    )
    m.rctp = Var(
        m.rct,
        within=NonNegativeReals,
        bounds=(3.4474, 3.4474),
        doc="reactor pressure [MPa]",
    )
    m.rctvol = Var(
        m.rct, within=NonNegativeReals, bounds=(None, 200), doc="reactor volume [m**3]"
    )
    m.krct = Var(
        m.rct,
        within=NonNegativeReals,
        initialize=1,
        bounds=(0.0123471, 0.149543),
        doc="rate constant",
    )
    m.conv = Var(
        m.rct,
        m.compon,
        within=NonNegativeReals,
        bounds=(None, 0.973),
        doc="conversion of key component",
    )
    m.sel = Var(
        m.rct,
        within=NonNegativeReals,
        bounds=(None, 0.9964),
        doc="selectivity to benzene",
    )
    m.consum = Var(
        m.rct,
        m.compon,
        within=NonNegativeReals,
        bounds=(0, 10000000000),
        initialize=0,
        doc="consumption rate of key",
    )
    m.q = Var(
        m.rct,
        within=NonNegativeReals,
        bounds=(0, 10000000000),
        doc="heat removed [1.e+9 kJ/yr]",
    )
    # splitter (1 output)
    m.spl1t = Var(
        m.spl1,
        within=PositiveReals,
        bounds=(3.00, 10.00),
        doc="splitter temperature [100 K]",
    )
    m.spl1p = Var(
        m.spl1, within=PositiveReals, bounds=(0.1, 4.0), doc="splitter pressure [MPa]"
    )
    # splitter
    m.splp = Var(m.spl, within=Reals, bounds=(0.1, 4.0), doc="splitter pressure [MPa]")
    m.splt = Var(
        m.spl, within=Reals, bounds=(3.0, 10.0), doc="splitter temperature [100 K]"
    )

    # stream
    def bound_f(m, stream):
        """
        stream flowrates [kg-mol/min]
        setting appropriate bounds for stream flowrates
        """
        if stream in range(8, 19):
            return (0, 50)
        elif stream in [52, 54, 56, 57, 58, 59, 60, 70, 71, 72]:
            return (0, 50)
        else:
            return (0, 10)

    m.f = Var(
        m.str,
        within=NonNegativeReals,
        bounds=bound_f,
        initialize=1,
        doc="stream flowrates [kg-mol/min]",
    )

    def bound_fc(m, stream, compon):
        """
        setting appropriate bounds for component flowrates
        """
        if stream in range(8, 19) or stream in [52, 54, 56, 57, 58, 59, 60, 70, 71, 72]:
            return (0, 30)
        else:
            return (0, 10)

    m.fc = Var(
        m.str,
        m.compon,
        within=Reals,
        bounds=bound_fc,
        initialize=1,
        doc="component flowrates [kg-mol/min]",
    )
    m.p = Var(
        m.str,
        within=NonNegativeReals,
        bounds=(0.1, 4.0),
        initialize=3.0,
        doc="stream pressure [MPa]",
    )
    m.t = Var(
        m.str,
        within=NonNegativeReals,
        bounds=(3.0, 10.0),
        initialize=3.0,
        doc="stream temperature [100 K]",
    )
    m.vp = Var(
        m.str,
        m.compon,
        within=NonNegativeReals,
        initialize=1,
        bounds=(0, 10),
        doc="vapor pressure [MPa]",
    )

    def boundsofe(m):
        """
        setting appropriate bounds for split fraction
        """
        if i == 20:
            return (None, 0.5)
        elif i == 21:
            return (0.5, 1.0)
        else:
            return (None, 1.0)

    m.e = Var(m.str, within=NonNegativeReals, bounds=boundsofe, doc="split fraction")

    # obj function constant term
    m.const = Param(initialize=22.5, doc="constant term in obj fcn")

    # ## setting variable bounds

    m.q[2].setub(100)
    for rct in m.rct:
        m.conv[rct, "tol"].setub(0.973)
    m.sel.setub(1.0 - 0.0036)
    m.reflux[1].setlb(0.02 * 1.2)
    m.reflux[1].setub(0.10 * 1.2)
    m.reflux[2].setlb(0.50 * 1.2)
    m.reflux[2].setub(2.00 * 1.2)
    m.reflux[3].setlb(0.02 * 1.2)
    m.reflux[3].setub(0.1 * 1.2)
    m.nmin[1].setlb(0)
    m.nmin[1].setub(4)
    m.nmin[2].setlb(8)
    m.nmin[2].setub(14)
    m.nmin[3].setlb(0)
    m.nmin[3].setub(4)
    m.ndist[1].setlb(0)
    m.ndist[1].setub(4 * 2 / m.disteff)
    m.ndist[3].setlb(0)
    m.ndist[3].setub(4 * 2 / m.disteff)
    m.ndist[2].setlb(8 * 2 / m.disteff)
    m.ndist[2].setub(14 * 2 / m.disteff)
    m.rmin[1].setlb(0.02)
    m.rmin[1].setub(0.10)
    m.rmin[2].setlb(0.50)
    m.rmin[2].setub(2.00)
    m.rmin[3].setlb(0.02)
    m.rmin[3].setub(0.1)
    m.distp[1].setub(1.0200000000000002)
    m.distp[1].setlb(1.0200000000000002)
    m.distp[2].setub(0.4)
    m.distp[3].setub(0.250)
    m.t[26].setlb(3.2)
    m.t[26].setub(3.2)
    for i in range(49, 52):
        m.t[i].setlb(2.0)
    m.t[27].setlb(
        (
            m.antb["ben"] / (m.anta["ben"] - log(m.distp[1].lb * 7500.6168))
            - m.antc["ben"]
        )
        / 100.0
    )
    m.t[27].setub(
        (
            m.antb["ben"] / (m.anta["ben"] - log(m.distp[1].ub * 7500.6168))
            - m.antc["ben"]
        )
        / 100.0
    )
    m.t[31].setlb(
        (
            m.antb["ben"] / (m.anta["ben"] - log(m.distp[2].lb * 7500.6168))
            - m.antc["ben"]
        )
        / 100.0
    )
    m.t[31].setub(
        (
            m.antb["ben"] / (m.anta["ben"] - log(m.distp[2].ub * 7500.6168))
            - m.antc["ben"]
        )
        / 100.0
    )
    m.t[32].setlb(
        (
            m.antb["tol"] / (m.anta["tol"] - log(m.distp[2].lb * 7500.6168))
            - m.antc["tol"]
        )
        / 100.0
    )
    m.t[32].setub(
        (
            m.antb["tol"] / (m.anta["tol"] - log(m.distp[2].ub * 7500.6168))
            - m.antc["tol"]
        )
        / 100.0
    )
    m.t[34].setlb(
        (
            m.antb["tol"] / (m.anta["tol"] - log(m.distp[3].lb * 7500.6168))
            - m.antc["tol"]
        )
        / 100.0
    )
    m.t[34].setub(
        (
            m.antb["tol"] / (m.anta["tol"] - log(m.distp[3].ub * 7500.6168))
            - m.antc["tol"]
        )
        / 100.0
    )
    m.t[35].setlb(
        (
            m.antb["dip"] / (m.anta["dip"] - log(m.distp[3].lb * 7500.6168))
            - m.antc["dip"]
        )
        / 100.0
    )
    m.t[35].setub(
        (
            m.antb["dip"] / (m.anta["dip"] - log(m.distp[3].ub * 7500.6168))
            - m.antc["dip"]
        )
        / 100.0
    )

    # absorber
    m.beta[1, "ben"].setlb(0.00011776)
    m.beta[1, "ben"].setub(5.72649)
    m.beta[1, "tol"].setlb(0.00018483515)
    m.beta[1, "tol"].setub(15)
    m.gamma[1, "tol"].setlb(
        log(
            (1 - m.aabs["tol"] ** (m.nabs[1].lb * m.abseff + m.eps1))
            / (1 - m.aabs["tol"])
        )
    )
    m.gamma[1, "tol"].setub(
        min(
            15,
            log(
                (1 - m.aabs["tol"] ** (m.nabs[1].ub * m.abseff + m.eps1))
                / (1 - m.aabs["tol"])
            ),
        )
    )
    for abso in m.abs:
        for compon in m.compon:
            m.beta[abso, compon].setlb(
                log(
                    (1 - m.aabs[compon] ** (m.nabs[1].lb * m.abseff + m.eps1 + 1))
                    / (1 - m.aabs[compon])
                )
            )
            m.beta[abso, compon].setub(
                min(
                    15,
                    log(
                        (1 - m.aabs[compon] ** (m.nabs[1].ub * m.abseff + m.eps1 + 1))
                        / (1 - m.aabs[compon])
                    ),
                )
            )
    m.t[67].setlb(3.0)
    m.t[67].setub(3.0)
    for compon in m.compon:
        m.vp[67, compon].setlb(
            (1.0 / 7500.6168)
            * exp(
                m.anta[compon]
                - m.antb[compon] / (value(m.t[67]) * 100.0 + m.antc[compon])
            )
        )
        m.vp[67, compon].setub(
            (1.0 / 7500.6168)
            * exp(
                m.anta[compon]
                - m.antb[compon] / (value(m.t[67]) * 100.0 + m.antc[compon])
            )
        )

    flashdata_file = os.path.join(dir_path, "flashdata.csv")
    flash = pd.read_csv(flashdata_file, header=0)
    number = flash.iloc[:, [4]].dropna().values
    two_digit_number = flash.iloc[:, [0]].dropna().values
    two_digit_compon = flash.iloc[:, [1]].dropna().values
    for i in range(len(two_digit_number)):
        m.eflsh[two_digit_number[i, 0], two_digit_compon[i, 0]].setlb(
            flash.iloc[:, [2]].dropna().values[i, 0]
        )
        m.eflsh[two_digit_number[i, 0], two_digit_compon[i, 0]].setub(
            flash.iloc[:, [3]].dropna().values[i, 0]
        )
    for i in range(len(number)):
        m.flshp[number[i, 0]].setlb(flash.iloc[:, [5]].dropna().values[i, 0])
        m.flshp[number[i, 0]].setub(flash.iloc[:, [6]].dropna().values[i, 0])
        m.flsht[number[i, 0]].setlb(flash.iloc[:, [7]].dropna().values[i, 0])
        m.flsht[number[i, 0]].setub(flash.iloc[:, [8]].dropna().values[i, 0])
    m.t[19].setlb(m.flsht[1].lb)
    m.t[19].setub(m.flsht[1].ub)
    m.t[48].setlb(m.flsht[2].lb)
    m.t[48].setub(m.flsht[2].ub)
    m.t[41].setlb(m.t[32].lb)
    m.t[41].setub(m.flsht[3].ub)
    m.t[1].setlb(3.0)
    m.t[1].setub(3.0)
    m.t[16].setub(8.943)
    m.t[66].setlb(3.0)

    for stream in m.str:
        for compon in m.compon:
            m.vp[stream, compon].setlb(
                (1.0 / 7500.6168)
                * exp(
                    m.anta[compon]
                    - m.antb[compon] / (m.t[stream].lb * 100.0 + m.antc[compon])
                )
            )
            m.vp[stream, compon].setub(
                (1.0 / 7500.6168)
                * exp(
                    m.anta[compon]
                    - m.antb[compon] / (m.t[stream].ub * 100.0 + m.antc[compon])
                )
            )

    m.p[1].setub(3.93)
    m.p[1].setlb(3.93)
    m.f[31].setlb(2.08)
    m.f[31].setub(2.08)
    m.p[66].setub(3.93)
    m.p[66].setub(3.93)

    # distillation bounds
    for dist in m.dist:
        for stream in m.str:
            for compon in m.compon:
                if (dist, stream) in m.ldist and (dist, compon) in m.dlkey:
                    m.avevlt[dist].setlb(m.vp[stream, compon].ub)
                if (dist, stream) in m.ldist and (dist, compon) in m.dhkey:
                    m.avevlt[dist].setlb(m.avevlt[dist].lb / m.vp[stream, compon].ub)
    for dist in m.dist:
        for stream in m.str:
            for compon in m.compon:
                if (dist, stream) in m.vdist and (dist, compon) in m.dlkey:
                    m.avevlt[dist].setub(m.vp[stream, compon].lb)
                if (dist, stream) in m.vdist and (dist, compon) in m.dhkey:
                    m.avevlt[dist].setub(m.avevlt[dist].ub / m.vp[stream, compon].lb)

    # ## initialization procedure

    # flash1
    m.eflsh[1, "h2"] = 0.995
    m.eflsh[1, "ch4"] = 0.99
    m.eflsh[1, "ben"] = 0.04
    m.eflsh[1, "tol"] = 0.01
    m.eflsh[1, "dip"] = 0.0001

    # compressor
    m.distp[1] = 1.02
    m.distp[2] = 0.1
    m.distp[3] = 0.1
    m.qexch[1] = 0.497842
    m.elec[1] = 0
    m.elec[2] = 12.384
    m.elec[3] = 0
    m.elec[4] = 28.7602
    m.presrat[1] = 1
    m.presrat[2] = 1.04552
    m.presrat[3] = 1.36516
    m.presrat[4] = 1.95418
    m.qfuel[1] = 0.0475341
    m.q[2] = 54.3002

    file_1 = os.path.join(dir_path, "GAMS_init_stream_data.csv")
    stream = pd.read_csv(file_1, usecols=[0])
    data = pd.read_csv(file_1, usecols=[1])
    temp = pd.read_csv(file_1, usecols=[3])
    flow = pd.read_csv(file_1, usecols=[4])
    e = pd.read_csv(file_1, usecols=[5])

    for i in range(len(stream)):
        m.p[stream.to_numpy()[i, 0]] = data.to_numpy()[i, 0]
    for i in range(72):
        m.t[stream.to_numpy()[i, 0]] = temp.to_numpy()[i, 0]
        m.f[stream.to_numpy()[i, 0]] = flow.to_numpy()[i, 0]
        m.e[stream.to_numpy()[i, 0]] = e.to_numpy()[i, 0]

    file_2 = os.path.join(dir_path, "GAMS_init_stream_compon_data.csv")
    streamfc = pd.read_csv(file_2, usecols=[0])
    comp = pd.read_csv(file_2, usecols=[1])
    fc = pd.read_csv(file_2, usecols=[2])
    streamvp = pd.read_csv(file_2, usecols=[3])
    compvp = pd.read_csv(file_2, usecols=[4])
    vp = pd.read_csv(file_2, usecols=[5])

    for i in range(len(streamfc)):
        m.fc[streamfc.to_numpy()[i, 0], comp.to_numpy()[i, 0]] = fc.to_numpy()[i, 0]
        m.vp[streamvp.to_numpy()[i, 0], compvp.to_numpy()[i, 0]] = vp.to_numpy()[i, 0]

    file_3 = os.path.join(dir_path, "GAMS_init_data.csv")
    stream3 = pd.read_csv(file_3, usecols=[0])
    a = pd.read_csv(file_3, usecols=[1])
    avevlt = pd.read_csv(file_3, usecols=[3])
    comp1 = pd.read_csv(file_3, usecols=[5])
    beta = pd.read_csv(file_3, usecols=[6])
    consum = pd.read_csv(file_3, usecols=[9])
    conv = pd.read_csv(file_3, usecols=[12])
    disp = pd.read_csv(file_3, usecols=[14])
    stream4 = pd.read_csv(file_3, usecols=[15])
    comp2 = pd.read_csv(file_3, usecols=[16])
    eflsh = pd.read_csv(file_3, usecols=[17])
    flshp = pd.read_csv(file_3, usecols=[19])
    flsht = pd.read_csv(file_3, usecols=[21])
    krct = pd.read_csv(file_3, usecols=[23])
    mxrp = pd.read_csv(file_3, usecols=[25])
    ndist = pd.read_csv(file_3, usecols=[27])
    nmin = pd.read_csv(file_3, usecols=[29])
    qc = pd.read_csv(file_3, usecols=[31])
    qh = pd.read_csv(file_3, usecols=[33])
    rctp = pd.read_csv(file_3, usecols=[35])
    rctt = pd.read_csv(file_3, usecols=[37])
    rctvol = pd.read_csv(file_3, usecols=[39])
    reflux = pd.read_csv(file_3, usecols=[41])
    rmin = pd.read_csv(file_3, usecols=[43])
    sel = pd.read_csv(file_3, usecols=[45])
    spl1p = pd.read_csv(file_3, usecols=[47])
    spl1t = pd.read_csv(file_3, usecols=[49])
    splp = pd.read_csv(file_3, usecols=[51])
    splt = pd.read_csv(file_3, usecols=[53])

    for i in range(2):
        m.rctp[i + 1] = rctp.to_numpy()[i, 0]
        m.rctt[i + 1] = rctt.to_numpy()[i, 0]
        m.rctvol[i + 1] = rctvol.to_numpy()[i, 0]
        m.sel[i + 1] = sel.to_numpy()[i, 0]
        m.krct[i + 1] = krct.to_numpy()[i, 0]
        m.consum[i + 1, "tol"] = consum.to_numpy()[i, 0]
        m.conv[i + 1, "tol"] = conv.to_numpy()[i, 0]
        m.a[stream3.to_numpy()[i, 0]] = a.to_numpy()[i, 0]
        m.qc[i + 1] = qc.to_numpy()[i, 0]
    for i in range(3):
        m.avevlt[i + 1] = avevlt.to_numpy()[i, 0]
        m.distp[i + 1] = disp.to_numpy()[i, 0]
        m.flshp[i + 1] = flshp.to_numpy()[i, 0]
        m.flsht[i + 1] = flsht.to_numpy()[i, 0]
        m.ndist[i + 1] = ndist.to_numpy()[i, 0]
        m.nmin[i + 1] = nmin.to_numpy()[i, 0]
        m.reflux[i + 1] = reflux.to_numpy()[i, 0]
        m.rmin[i + 1] = rmin.to_numpy()[i, 0]
        m.splp[i + 1] = splp.to_numpy()[i, 0]
        m.splt[i + 1] = splt.to_numpy()[i, 0]
    for i in range(5):
        m.beta[1, comp1.to_numpy()[i, 0]] = beta.to_numpy()[i, 0]
        m.mxrp[i + 1] = mxrp.to_numpy()[i, 0]
    for i in range(4):
        m.qh[i + 1] = qh.to_numpy()[i, 0]
    for i in range(len(stream4)):
        m.eflsh[stream4.to_numpy()[i, 0], comp2.to_numpy()[i, 0]] = eflsh.to_numpy()[
            i, 0
        ]
    for i in range(6):
        m.spl1p[i + 1] = spl1p.to_numpy()[i, 0]
        m.spl1t[i + 1] = spl1t.to_numpy()[i, 0]

    # ## constraints
    m.specrec = Constraint(
        expr=m.fc[72, "h2"] >= 0.5 * m.f[72], doc="specification on h2 recycle"
    )
    m.specprod = Constraint(
        expr=m.fc[31, "ben"] >= 0.9997 * m.f[31],
        doc="specification on benzene production",
    )

    def Fbal(_m, stream):
        return m.f[stream] == sum(m.fc[stream, compon] for compon in m.compon)

    m.fbal = Constraint(m.str, rule=Fbal, doc="flow balance")

    def H2feed(m, compon):
        return m.fc[1, compon] == m.f[1] * m.f1comp[compon]

    m.h2feed = Constraint(m.compon, rule=H2feed, doc="h2 feed composition")

    def Tolfeed(_m, compon):
        return m.fc[66, compon] == m.f[66] * m.f66comp[compon]

    m.tolfeed = Constraint(m.compon, rule=Tolfeed, doc="toluene feed composition")

    def Tolabs(_m, compon):
        return m.fc[67, compon] == m.f[67] * m.f67comp[compon]

    m.tolabs = Constraint(m.compon, rule=Tolabs, doc="toluene absorber composition")

    def build_absorber(b, absorber):
        """
        Functions relevant to the absorber block

        Parameters
        ----------
        b : Pyomo Block
            absorber block
        absorber : int
            Index of the absorber
        """

        def Absfact(_m, i, compon):
            """
            Absorption factor equation
            sum of flowrates of feed components = sum of flowrates of vapor components * absorption factor * sum of vapor pressures

            """
            if (i, compon) in m.anorm:
                return sum(
                    m.f[stream] * m.p[stream] for (absb, stream) in m.ilabs if absb == i
                ) == sum(
                    m.f[stream] for (absc, stream) in m.ivabs if absc == i
                ) * m.aabs[
                    compon
                ] * sum(
                    m.vp[stream, compon] for (absd, stream) in m.ilabs if absd == i
                )
            return Constraint.Skip

        b.absfact = Constraint(
            [absorber], m.compon, rule=Absfact, doc="absorption factor equation"
        )

        def Gameqn(_m, i, compon):
            # definition of gamma
            if (i, compon) in m.asolv:
                return m.gamma[i, compon] == log(
                    (1 - m.aabs[compon] ** (m.nabs[i] * m.abseff + m.eps1))
                    / (1 - m.aabs[compon])
                )
            return Constraint.Skip

        b.gameqn = Constraint(
            [absorber], m.compon, rule=Gameqn, doc="definition of gamma"
        )

        def Betaeqn(_m, i, compon):
            # definition of beta
            if (i, compon) not in m.asimp:
                return m.beta[i, compon] == log(
                    (1 - m.aabs[compon] ** (m.nabs[i] * m.abseff + 1))
                    / (1 - m.aabs[compon])
                )
            return Constraint.Skip

        b.betaeqn = Constraint(
            [absorber], m.compon, rule=Betaeqn, doc="definition of beta"
        )

        def Abssvrec(_m, i, compon):
            # recovery of solvent
            if (i, compon) in m.asolv:
                return sum(m.fc[stream, compon] for (i, stream) in m.ovabs) * exp(
                    m.beta[i, compon]
                ) == sum(m.fc[stream, compon] for (i_, stream) in m.ivabs) + exp(
                    m.gamma[i, compon]
                ) * sum(
                    m.fc[stream, compon] for (i_, stream) in m.ilabs
                )
            return Constraint.Skip

        b.abssvrec = Constraint(
            [absorber], m.compon, rule=Abssvrec, doc="recovery of solvent"
        )

        def Absrec(_m, i, compon):
            # recovery of non-solvent
            if (i, compon) in m.anorm:
                return sum(m.fc[i, compon] for (abs, i) in m.ovabs) * exp(
                    m.beta[i, compon]
                ) == sum(m.fc[i, compon] for (abs, i) in m.ivabs)
            return Constraint.Skip

        b.absrec = Constraint(
            [absorber], m.compon, rule=Absrec, doc="recovery of non-solvent"
        )

        def abssimp(_m, absorb, compon):
            # recovery of simplified components
            if (absorb, compon) in m.asimp:
                return (
                    sum(m.fc[i, compon] for (absorb, i) in m.ovabs)
                    == sum(m.fc[i, compon] for (absorb, i) in m.ivabs) / m.cbeta[compon]
                )
            return Constraint.Skip

        b.abssimp = Constraint(
            [absorber], m.compon, rule=abssimp, doc="recovery of simplified components"
        )

        def Abscmb(_m, i, compon):
            return sum(m.fc[stream, compon] for (i, stream) in m.ilabs) + sum(
                m.fc[stream, compon] for (i, stream) in m.ivabs
            ) == sum(m.fc[stream, compon] for (i, stream) in m.olabs) + sum(
                m.fc[stream, compon] for (i, stream) in m.ovabs
            )

        b.abscmb = Constraint(
            [absorber],
            m.compon,
            rule=Abscmb,
            doc="overall component mass balance in absorber",
        )

        def Abspl(_m, i):
            return sum(m.p[stream] for (_, stream) in m.ilabs) == sum(
                m.p[stream] for (_, stream) in m.olabs
            )

        b.abspl = Constraint(
            [absorber], rule=Abspl, doc="pressure relation for liquid in absorber"
        )

        def Abstl(_m, i):
            return sum(m.t[stream] for (_, stream) in m.ilabs) == sum(
                m.t[stream] for (_, stream) in m.olabs
            )

        b.abstl = Constraint(
            [absorber], rule=Abstl, doc="temperature relation for liquid in absorber"
        )

        def Abspv(_m, i):
            return sum(m.p[stream] for (_, stream) in m.ivabs) == sum(
                m.p[stream] for (_, stream) in m.ovabs
            )

        b.abspv = Constraint(
            [absorber], rule=Abspv, doc="pressure relation for vapor in absorber"
        )

        def Abspin(_m, i):
            return sum(m.p[stream] for (_, stream) in m.ilabs) == sum(
                m.p[stream] for (_, stream) in m.ivabs
            )

        b.absp = Constraint(
            [absorber], rule=Abspin, doc="pressure relation at inlet of absorber"
        )

        def Absttop(_m, i):
            return sum(m.t[stream] for (_, stream) in m.ilabs) == sum(
                m.t[stream] for (_, stream) in m.ovabs
            )

        b.abst = Constraint(
            [absorber], rule=Absttop, doc="temperature relation at top of absorber"
        )

    def build_compressor(b, comp):
        """
        Functions relevant to the compressor block

        Parameters
        ----------
        b : Pyomo Block
            compressor block
        comp : int
            Index of the compressor
        """

        def Compcmb(_m, comp1, compon):
            if comp1 == comp:
                return sum(
                    m.fc[stream, compon]
                    for (comp_, stream) in m.ocomp
                    if comp_ == comp1
                ) == sum(
                    m.fc[stream, compon]
                    for (comp_, stream) in m.icomp
                    if comp_ == comp1
                )
            return Constraint.Skip

        b.compcmb = Constraint(
            [comp], m.compon, rule=Compcmb, doc="component balance in compressor"
        )

        def Comphb(_m, comp1):
            if comp1 == comp:
                return sum(
                    m.t[stream] for (_, stream) in m.ocomp if _ == comp
                ) == m.presrat[comp] * sum(
                    m.t[stream] for (_, stream) in m.icomp if _ == comp
                )
            return Constraint.Skip

        b.comphb = Constraint([comp], rule=Comphb, doc="heat balance in compressor")

        def Compelec(_m, comp_):
            if comp_ == comp:
                return m.elec[comp_] == m.alpha * (m.presrat[comp_] - 1) * sum(
                    100.0
                    * m.t[stream]
                    * m.f[stream]
                    / 60.0
                    * (1.0 / m.compeff)
                    * (m.cp_cv_ratio / (m.cp_cv_ratio - 1.0))
                    for (comp1, stream) in m.icomp
                    if comp_ == comp1
                )
            return Constraint.Skip

        b.compelec = Constraint(
            [comp], rule=Compelec, doc="energy balance in compressor"
        )

        def Ratio(_m, comp_):
            if comp == comp_:
                return m.presrat[comp_] ** (
                    m.cp_cv_ratio / (m.cp_cv_ratio - 1.0)
                ) == sum(
                    m.p[stream] for (comp1, stream) in m.ocomp if comp_ == comp1
                ) / sum(
                    m.p[stream] for (comp1, stream) in m.icomp if comp1 == comp_
                )
            return Constraint.Skip

        b.ratio = Constraint(
            [comp], rule=Ratio, doc="pressure ratio (out to in) in compressor"
        )

    m.vapor_pressure_unit_match = Param(
        initialize=7500.6168,
        doc="unit match coefficient for vapor pressure calculation",
    )
    m.actual_reflux_ratio = Param(initialize=1.2, doc="actual reflux ratio coefficient")
    m.recovery_specification_coefficient = Param(
        initialize=0.05, doc="recovery specification coefficient"
    )

    def build_distillation(b, dist):
        """
        Functions relevant to the distillation block

        Parameters
        ----------
        b : Pyomo Block
            distillation block
        dist : int
            Index of the distillation column
        """

        def Antdistb(_m, dist_, stream, compon):
            if (
                (dist_, stream) in m.ldist
                and (dist_, compon) in m.dkey
                and dist_ == dist
            ):
                return log(
                    m.vp[stream, compon] * m.vapor_pressure_unit_match
                ) == m.anta[compon] - m.antb[compon] / (
                    m.t[stream] * 100.0 + m.antc[compon]
                )
            return Constraint.Skip

        b.antdistb = Constraint(
            [dist],
            m.str,
            m.compon,
            rule=Antdistb,
            doc="vapor pressure correlation (bottom)",
        )

        def Antdistt(_m, dist_, stream, compon):
            if (
                (dist_, stream) in m.vdist
                and (dist_, compon) in m.dkey
                and dist == dist_
            ):
                return log(
                    m.vp[stream, compon] * m.vapor_pressure_unit_match
                ) == m.anta[compon] - m.antb[compon] / (
                    m.t[stream] * 100.0 + m.antc[compon]
                )
            return Constraint.Skip

        b.antdistt = Constraint(
            [dist],
            m.str,
            m.compon,
            rule=Antdistt,
            doc="vapor pressure correlation (top)",
        )

        def Relvol(_m, dist_):
            if dist == dist_:
                divided1 = sum(
                    sum(
                        m.vp[stream, compon]
                        for (dist_, compon) in m.dlkey
                        if dist_ == dist
                    )
                    / sum(
                        m.vp[stream, compon]
                        for (dist_, compon) in m.dhkey
                        if dist_ == dist
                    )
                    for (dist_, stream) in m.vdist
                    if dist_ == dist
                )
                divided2 = sum(
                    sum(
                        m.vp[stream, compon]
                        for (dist_, compon) in m.dlkey
                        if dist_ == dist
                    )
                    / sum(
                        m.vp[stream, compon]
                        for (dist_, compon) in m.dhkey
                        if dist_ == dist
                    )
                    for (dist_, stream) in m.ldist
                    if dist_ == dist
                )
                return m.avevlt[dist] == sqrt(divided1 * divided2)
            return Constraint.Skip

        b.relvol = Constraint([dist], rule=Relvol, doc="average relative volatility")

        def Undwood(_m, dist_):
            # minimum reflux ratio from Underwood equation
            if dist_ == dist:
                return sum(
                    m.fc[stream, compon]
                    for (dist1, compon) in m.dlkey
                    if dist1 == dist_
                    for (dist1, stream) in m.idist
                    if dist1 == dist_
                ) * m.rmin[dist_] * (m.avevlt[dist_] - 1) == sum(
                    m.f[stream] for (dist1, stream) in m.idist if dist1 == dist_
                )
            return Constraint.Skip

        b.undwood = Constraint(
            [dist], rule=Undwood, doc="minimum reflux ratio equation"
        )

        def Actreflux(_m, dist_):
            # actual reflux ratio (heuristic)
            if dist_ == dist:
                return m.reflux[dist_] == m.actual_reflux_ratio * m.rmin[dist_]
            return Constraint.Skip

        b.actreflux = Constraint([dist], rule=Actreflux, doc="actual reflux ratio")

        def Fenske(_m, dist_):
            # minimum number of trays from Fenske equation
            if dist == dist_:
                sum1 = sum(
                    (m.f[stream] + m.eps1) / (m.fc[stream, compon] + m.eps1)
                    for (dist1, compon) in m.dhkey
                    if dist1 == dist_
                    for (dist1, stream) in m.vdist
                    if dist1 == dist_
                )
                sum2 = sum(
                    (m.f[stream] + m.eps1) / (m.fc[stream, compon] + m.eps1)
                    for (dist1, compon) in m.dlkey
                    if dist1 == dist_
                    for (dist1, stream) in m.ldist
                    if dist1 == dist_
                )
                return m.nmin[dist_] * log(m.avevlt[dist_]) == log(sum1 * sum2)
            return Constraint.Skip

        b.fenske = Constraint([dist], rule=Fenske, doc="minimum number of trays")

        def Acttray(_m, dist_):
            # actual number of trays (Gilliland approximation)
            if dist == dist_:
                return m.ndist[dist_] == m.nmin[dist_] * 2.0 / m.disteff
            return Constraint.Skip

        b.acttray = Constraint([dist], rule=Acttray, doc="actual number of trays")

        def Distspec(_m, dist_, stream, compon):
            if (
                (dist_, stream) in m.vdist
                and (dist_, compon) in m.dhkey
                and dist_ == dist
            ):
                return m.fc[
                    stream, compon
                ] <= m.recovery_specification_coefficient * sum(
                    m.fc[str2, compon] for (dist_, str2) in m.idist if dist == dist_
                )
            return Constraint.Skip

        b.distspec = Constraint(
            [dist], m.str, m.compon, rule=Distspec, doc="recovery specification"
        )

        def Distheav(_m, dist_, compon):
            if (dist_, compon) in m.dh and dist == dist_:
                return sum(
                    m.fc[str2, compon] for (dist_, str2) in m.idist if dist_ == dist
                ) == sum(
                    m.fc[str2, compon] for (dist_, str2) in m.ldist if dist_ == dist
                )
            return Constraint.Skip

        b.distheav = Constraint([dist], m.compon, rule=Distheav, doc="heavy components")

        def Distlite(_m, dist_, compon):
            if (dist_, compon) in m.dl and dist_ == dist:
                return sum(
                    m.fc[str2, compon] for (dist_, str2) in m.idist if dist == dist_
                ) == sum(
                    m.fc[str2, compon] for (dist_, str2) in m.vdist if dist == dist_
                )
            return Constraint.Skip

        b.distlite = Constraint([dist], m.compon, rule=Distlite, doc="light components")

        def Distpi(_m, dist_, stream):
            if (dist_, stream) in m.idist and dist_ == dist:
                return m.distp[dist_] <= m.p[stream]
            return Constraint.Skip

        b.distpi = Constraint([dist], m.str, rule=Distpi, doc="inlet pressure relation")

        def Distvpl(_m, dist_, stream):
            if (dist_, stream) in m.ldist and dist == dist_:
                return m.distp[dist_] == sum(
                    m.vp[stream, compon] for (dist_, compon) in m.dhkey if dist_ == dist
                )
            return Constraint.Skip

        b.distvpl = Constraint(
            [dist], m.str, rule=Distvpl, doc="bottom vapor pressure relation"
        )

        def Distvpv(_m, dist_, stream):
            if dist > 1 and (dist, stream) in m.vdist and dist_ == dist:
                return m.distp[dist_] == sum(
                    m.vp[stream, compon] for (dist_, compon) in m.dlkey if dist_ == dist
                )
            return Constraint.Skip

        b.distvpv = Constraint(
            [dist], m.str, rule=Distvpv, doc="top vapor pressure relation"
        )

        def Distpl(_m, dist_, stream):
            if (dist_, stream) in m.ldist and dist_ == dist:
                return m.distp[dist_] == m.p[stream]
            return Constraint.Skip

        b.distpl = Constraint(
            [dist], m.str, rule=Distpl, doc="outlet pressure relation (liquid)"
        )

        def Distpv(_m, dist_, stream):
            if (dist_, stream) in m.vdist and dist == dist_:
                return m.distp[dist_] == m.p[stream]
            return Constraint.Skip

        b.distpv = Constraint(
            [dist], m.str, rule=Distpv, doc="outlet pressure relation (vapor)"
        )

        def Distcmb(_m, dist_, compon):
            if dist_ == dist:
                return sum(
                    m.fc[stream, compon]
                    for (dist1, stream) in m.idist
                    if dist1 == dist_
                ) == sum(
                    m.fc[stream, compon]
                    for (dist1, stream) in m.vdist
                    if dist1 == dist_
                ) + sum(
                    m.fc[stream, compon]
                    for (dist1, stream) in m.ldist
                    if dist1 == dist_
                )
            return Constraint.Skip

        b.distcmb = Constraint(
            [dist],
            m.compon,
            rule=Distcmb,
            doc="component mass balance in distillation column",
        )

    def build_flash(b, flsh):
        """
        Functions relevant to the flash block

        Parameters
        ----------
        b : Pyomo Block
            flash block
        flsh : int
            Index of the flash
        """

        def Flshcmb(_m, flsh_, compon):
            if flsh_ in m.flsh and compon in m.compon and flsh_ == flsh:
                return sum(
                    m.fc[stream, compon]
                    for (flsh1, stream) in m.iflsh
                    if flsh1 == flsh_
                ) == sum(
                    m.fc[stream, compon]
                    for (flsh1, stream) in m.vflsh
                    if flsh1 == flsh_
                ) + sum(
                    m.fc[stream, compon]
                    for (flsh1, stream) in m.lflsh
                    if flsh1 == flsh_
                )
            return Constraint.Skip

        b.flshcmb = Constraint(
            [flsh], m.compon, rule=Flshcmb, doc="component mass balance in flash"
        )

        def Antflsh(_m, flsh_, stream, compon):
            if (flsh_, stream) in m.lflsh and flsh_ == flsh:
                return log(
                    m.vp[stream, compon] * m.vapor_pressure_unit_match
                ) == m.anta[compon] - m.antb[compon] / (
                    m.t[stream] * 100.0 + m.antc[compon]
                )
            return Constraint.Skip

        b.antflsh = Constraint(
            [flsh], m.str, m.compon, rule=Antflsh, doc="flash pressure relation"
        )

        def Flshrec(_m, flsh_, stream, compon):
            if (flsh_, stream) in m.lflsh and flsh_ == flsh:
                return (
                    sum(
                        m.eflsh[flsh1, compon2]
                        for (flsh1, compon2) in m.fkey
                        if flsh1 == flsh_
                    )
                    * (
                        m.eflsh[flsh_, compon]
                        * sum(
                            m.vp[stream, compon2]
                            for (flsh1, compon2) in m.fkey
                            if flsh_ == flsh1
                        )
                        + (1.0 - m.eflsh[flsh_, compon]) * m.vp[stream, compon]
                    )
                    == sum(
                        m.vp[stream, compon2]
                        for (flsh1, compon2) in m.fkey
                        if flsh_ == flsh1
                    )
                    * m.eflsh[flsh_, compon]
                )
            return Constraint.Skip

        b.flshrec = Constraint(
            [flsh], m.str, m.compon, rule=Flshrec, doc="vapor recovery relation"
        )

        def Flsheql(_m, flsh_, compon):
            if flsh in m.flsh and compon in m.compon and flsh_ == flsh:
                return (
                    sum(
                        m.fc[stream, compon]
                        for (flsh1, stream) in m.vflsh
                        if flsh1 == flsh_
                    )
                    == sum(
                        m.fc[stream, compon]
                        for (flsh1, stream) in m.iflsh
                        if flsh1 == flsh_
                    )
                    * m.eflsh[flsh, compon]
                )
            return Constraint.Skip

        b.flsheql = Constraint(
            [flsh], m.compon, rule=Flsheql, doc="equilibrium relation"
        )

        def Flshpr(_m, flsh_, stream):
            if (flsh_, stream) in m.lflsh and flsh_ == flsh:
                return m.flshp[flsh_] * m.f[stream] == sum(
                    m.vp[stream, compon] * m.fc[stream, compon] for compon in m.compon
                )
            return Constraint.Skip

        b.flshpr = Constraint([flsh], m.str, rule=Flshpr, doc="flash pressure relation")

        def Flshpi(_m, flsh_, stream):
            if (flsh_, stream) in m.iflsh and flsh_ == flsh:
                return m.flshp[flsh_] == m.p[stream]
            return Constraint.Skip

        b.flshpi = Constraint([flsh], m.str, rule=Flshpi, doc="inlet pressure relation")

        def Flshpl(_m, flsh_, stream):
            if (flsh_, stream) in m.lflsh and flsh_ == flsh:
                return m.flshp[flsh_] == m.p[stream]
            return Constraint.Skip

        b.flshpl = Constraint(
            [flsh], m.str, rule=Flshpl, doc="outlet pressure relation (liquid)"
        )

        def Flshpv(_m, flsh_, stream):
            if (flsh_, stream) in m.vflsh and flsh_ == flsh:
                return m.flshp[flsh_] == m.p[stream]
            return Constraint.Skip

        b.flshpv = Constraint(
            [flsh], m.str, rule=Flshpv, doc="outlet pressure relation (vapor)"
        )

        def Flshti(_m, flsh_, stream):
            if (flsh_, stream) in m.iflsh and flsh_ == flsh:
                return m.flsht[flsh_] == m.t[stream]
            return Constraint.Skip

        b.flshti = Constraint(
            [flsh], m.str, rule=Flshti, doc="inlet temperature relation"
        )

        def Flshtl(_m, flsh_, stream):
            if (flsh_, stream) in m.lflsh and flsh_ == flsh:
                return m.flsht[flsh_] == m.t[stream]
            return Constraint.Skip

        b.flshtl = Constraint(
            [flsh], m.str, rule=Flshtl, doc="outlet temperature relation (liquid)"
        )

        def Flshtv(_m, flsh_, stream):
            if (flsh_, stream) in m.vflsh and flsh_ == flsh:
                return m.flsht[flsh_] == m.t[stream]
            return Constraint.Skip

        b.flshtv = Constraint(
            [flsh], m.str, rule=Flshtv, doc="outlet temperature relation (vapor)"
        )

    m.heat_unit_match = Param(
        initialize=3600.0 * 8500.0 * 1.0e-12 / 60.0,
        doc="unit change on heat balance from [kJ/min] to [1e12kJ/yr]",
    )

    def build_furnace(b, furnace):
        """
        Functions relevant to the furnace block

        Parameters
        ----------
        b : Pyomo Block
            furnace block
        furnace : int
            Index of the furnace
        """

        def Furnhb(_m, furn):
            if furn == furnace:
                return (
                    m.qfuel[furn]
                    == (
                        sum(
                            m.cp[stream] * m.f[stream] * 100.0 * m.t[stream]
                            for (furn, stream) in m.ofurn
                        )
                        - sum(
                            m.cp[stream] * m.f[stream] * 100.0 * m.t[stream]
                            for (furn, stream) in m.ifurn
                        )
                    )
                    * m.heat_unit_match
                )
            return Constraint.Skip

        b.furnhb = Constraint([furnace], rule=Furnhb, doc="heat balance in furnace")

        def Furncmb(_m, furn, compon):
            if furn == furnace:
                return sum(m.fc[stream, compon] for (furn, stream) in m.ofurn) == sum(
                    m.fc[stream, compon] for (furn, stream) in m.ifurn
                )
            return Constraint.Skip

        b.furncmb = Constraint(
            [furnace], m.compon, rule=Furncmb, doc="component mass balance in furnace"
        )

        def Furnp(_m, furn):
            if furn == furnace:
                return (
                    sum(m.p[stream] for (furn, stream) in m.ofurn)
                    == sum(m.p[stream] for (furn, stream) in m.ifurn) - m.furnpdrop
                )
            return Constraint.Skip

        b.furnp = Constraint([furnace], rule=Furnp, doc="pressure relation in furnace")

    def build_cooler(b, cooler):
        """
        Functions relevant to the cooler block

        Parameters
        ----------
        b : Pyomo Block
            cooler block
        cooler : int
            Index of the cooler
        """

        def Heccmb(_m, hec, compon):
            return sum(
                m.fc[stream, compon] for (hec_, stream) in m.ohec if hec_ == hec
            ) == sum(m.fc[stream, compon] for (hec_, stream) in m.ihec if hec_ == hec)

        b.heccmb = Constraint(
            [cooler], m.compon, rule=Heccmb, doc="heat balance in cooler"
        )

        def Hechb(_m, hec):
            return (
                m.qc[hec]
                == (
                    sum(
                        m.cp[stream] * m.f[stream] * 100.0 * m.t[stream]
                        for (hec_, stream) in m.ihec
                        if hec_ == hec
                    )
                    - sum(
                        m.cp[stream] * m.f[stream] * 100.0 * m.t[stream]
                        for (hec_, stream) in m.ohec
                        if hec_ == hec
                    )
                )
                * m.heat_unit_match
            )

        b.hechb = Constraint(
            [cooler], rule=Hechb, doc="component mass balance in cooler"
        )

        def Hecp(_m, hec):
            return sum(m.p[stream] for (hec_, stream) in m.ihec if hec_ == hec) == sum(
                m.p[stream] for (hec_, stream) in m.ohec if hec_ == hec
            )

        b.hecp = Constraint([cooler], rule=Hecp, doc="pressure relation in cooler")

    def build_heater(b, heater):
        """
        Functions relevant to the heater block

        Parameters
        ----------
        b : Pyomo Block
            heater block
        heater : int
            Index of the heater
        """

        def Hehcmb(_m, heh, compon):
            if heh == heater and compon in m.compon:
                return sum(
                    m.fc[stream, compon] for (heh_, stream) in m.oheh if heh_ == heh
                ) == sum(
                    m.fc[stream, compon] for (heh_, stream) in m.iheh if heh == heh_
                )
            return Constraint.Skip

        b.hehcmb = Constraint(
            Set(initialize=[heater]),
            m.compon,
            rule=Hehcmb,
            doc="component balance in heater",
        )

        def Hehhb(_m, heh):
            if heh == heater:
                return (
                    m.qh[heh]
                    == (
                        sum(
                            m.cp[stream] * m.f[stream] * 100.0 * m.t[stream]
                            for (heh_, stream) in m.oheh
                            if heh_ == heh
                        )
                        - sum(
                            m.cp[stream] * m.f[stream] * 100.0 * m.t[stream]
                            for (heh_, stream) in m.iheh
                            if heh_ == heh
                        )
                    )
                    * m.heat_unit_match
                )
            return Constraint.Skip

        b.hehhb = Constraint(
            Set(initialize=[heater]), rule=Hehhb, doc="heat balance in heater"
        )

        def hehp(_m, heh):
            if heh == heater:
                return sum(
                    m.p[stream] for (heh_, stream) in m.iheh if heh_ == heh
                ) == sum(m.p[stream] for (heh_, stream) in m.oheh if heh == heh_)
            return Constraint.Skip

        b.Hehp = Constraint(
            Set(initialize=[heater]), rule=hehp, doc="no pressure drop thru heater"
        )

    m.exchanger_temp_drop = Param(initialize=0.25)

    def build_exchanger(b, exchanger):
        """
        Functions relevant to the exchanger block

        Parameters
        ----------
        b : Pyomo Block
            exchanger block
        exchanger : int
            Index of the exchanger
        """

        def Exchcmbc(_m, exch, compon):
            if exch in m.exch and compon in m.compon:
                return sum(
                    m.fc[stream, compon]
                    for (exch_, stream) in m.ocexch
                    if exch == exch_
                ) == sum(
                    m.fc[stream, compon]
                    for (exch_, stream) in m.icexch
                    if exch == exch_
                )
            return Constraint.Skip

        b.exchcmbc = Constraint(
            [exchanger],
            m.compon,
            rule=Exchcmbc,
            doc="component balance (cold) in exchanger",
        )

        def Exchcmbh(_m, exch, compon):
            if exch in m.exch and compon in m.compon:
                return sum(
                    m.fc[stream, compon]
                    for (exch_, stream) in m.ohexch
                    if exch == exch_
                ) == sum(
                    m.fc[stream, compon]
                    for (exch_, stream) in m.ihexch
                    if exch == exch_
                )
            return Constraint.Skip

        b.exchcmbh = Constraint(
            [exchanger],
            m.compon,
            rule=Exchcmbh,
            doc="component balance (hot) in exchanger",
        )

        def Exchhbc(_m, exch):
            if exch in m.exch:
                return (
                    sum(
                        m.cp[stream] * m.f[stream] * 100.0 * m.t[stream]
                        for (exch_, stream) in m.ocexch
                        if exch == exch_
                    )
                    - sum(
                        m.cp[stream] * m.f[stream] * 100.0 * m.t[stream]
                        for (exch_, stream) in m.icexch
                        if exch == exch_
                    )
                ) * m.heat_unit_match == m.qexch[exch]
            return Constraint.Skip

        b.exchhbc = Constraint(
            [exchanger], rule=Exchhbc, doc="heat balance for cold stream in exchanger"
        )

        def Exchhbh(_m, exch):
            return (
                sum(
                    m.cp[stream] * m.f[stream] * 100.0 * m.t[stream]
                    for (exch, stream) in m.ihexch
                )
                - sum(
                    m.cp[stream] * m.f[stream] * 100.0 * m.t[stream]
                    for (exch, stream) in m.ohexch
                )
            ) * m.heat_unit_match == m.qexch[exch]

        b.exchhbh = Constraint(
            [exchanger], rule=Exchhbh, doc="heat balance for hot stream in exchanger"
        )

        def Exchdtm1(_m, exch):
            return (
                sum(m.t[stream] for (exch, stream) in m.ohexch)
                >= sum(m.t[stream] for (exch, stream) in m.icexch)
                + m.exchanger_temp_drop
            )

        b.exchdtm1 = Constraint([exchanger], rule=Exchdtm1, doc="delta t min condition")

        def Exchdtm2(_m, exch):
            return (
                sum(m.t[stream] for (exch, stream) in m.ocexch)
                <= sum(m.t[stream] for (exch, stream) in m.ihexch)
                - m.exchanger_temp_drop
            )

        b.exchdtm2 = Constraint([exchanger], rule=Exchdtm2, doc="delta t min condition")

        def Exchpc(_m, exch):
            return sum(m.p[stream] for (exch, stream) in m.ocexch) == sum(
                m.p[stream] for (exch, stream) in m.icexch
            )

        b.exchpc = Constraint(
            [exchanger], rule=Exchpc, doc="pressure relation (cold) in exchanger"
        )

        def Exchph(_m, exch):
            return sum(m.p[stream] for (exch, stream) in m.ohexch) == sum(
                m.p[stream] for (exch, stream) in m.ihexch
            )

        b.exchph = Constraint(
            [exchanger], rule=Exchph, doc="pressure relation (hot) in exchanger"
        )

    m.membrane_recovery_sepc = Param(initialize=0.50)
    m.membrane_purity_sepc = Param(initialize=0.50)

    def build_membrane(b, membrane):
        """
        Functions relevant to the membrane block

        Parameters
        ----------
        b : Pyomo Block
            membrane block
        membrane : int
            Index of the membrane
        """

        def Memcmb(_m, memb, stream, compon):
            if (memb, stream) in m.imemb and memb == membrane:
                return m.fc[stream, compon] == sum(
                    m.fc[stream, compon] for (memb_, stream) in m.pmemb if memb == memb_
                ) + sum(
                    m.fc[stream, compon] for (memb_, stream) in m.nmemb if memb == memb_
                )
            return Constraint.Skip

        b.memcmb = Constraint(
            [membrane],
            m.str,
            m.compon,
            rule=Memcmb,
            doc="component mass balance in membrane separator",
        )

        def Flux(_m, memb, stream, compon):
            if (
                (memb, stream) in m.pmemb
                and (memb, compon) in m.mnorm
                and memb == membrane
            ):
                return m.fc[stream, compon] == m.a[memb] * m.perm[compon] / 2.0 * (
                    sum(m.p[stream2] for (memb_, stream2) in m.imemb if memb_ == memb)
                    * (
                        sum(
                            (m.fc[stream2, compon] + m.eps1) / (m.f[stream2] + m.eps1)
                            for (memb_, stream2) in m.imemb
                            if memb_ == memb
                        )
                        + sum(
                            (m.fc[stream2, compon] + m.eps1) / (m.f[stream2] + m.eps1)
                            for (memb_, stream2) in m.nmemb
                            if memb_ == memb
                        )
                    )
                    - 2.0
                    * m.p[stream]
                    * (m.fc[stream, compon] + m.eps1)
                    / (m.f[stream] + m.eps1)
                )
            return Constraint.Skip

        b.flux = Constraint(
            [membrane],
            m.str,
            m.compon,
            rule=Flux,
            doc="mass flux relation in membrane separator",
        )

        def Simp(_m, memb, stream, compon):
            if (
                (memb, stream) in m.pmemb
                and (memb, compon) in m.msimp
                and memb == membrane
            ):
                return m.fc[stream, compon] == 0.0
            return Constraint.Skip

        b.simp = Constraint(
            [membrane],
            m.str,
            m.compon,
            rule=Simp,
            doc="mass flux relation (simplified) in membrane separator",
        )

        def Memtp(_m, memb, stream):
            if (memb, stream) in m.pmemb and memb == membrane:
                return m.t[stream] == sum(
                    m.t[stream2] for (memb, stream2) in m.imemb if memb == membrane
                )
            return Constraint.Skip

        b.memtp = Constraint(
            [membrane], m.str, rule=Memtp, doc="temperature relation for permeate"
        )

        def Mempp(_m, memb, stream):
            if (memb, stream) in m.pmemb and memb == membrane:
                return m.p[stream] <= sum(
                    m.p[stream2] for (memb, stream2) in m.imemb if memb == membrane
                )
            return Constraint.Skip

        b.mempp = Constraint(
            [membrane], m.str, rule=Mempp, doc="pressure relation for permeate"
        )

        def Memtn(_m, memb, stream):
            if (memb, stream) in m.nmemb and memb == membrane:
                return m.t[stream] == sum(
                    m.t[stream2] for (memb, stream2) in m.imemb if memb == membrane
                )
            return Constraint.Skip

        b.Memtn = Constraint(
            [membrane], m.str, rule=Memtn, doc="temperature relation for non-permeate"
        )

        def Mempn(_m, memb, stream):
            if (memb, stream) in m.nmemb and memb == membrane:
                return m.p[stream] == sum(
                    m.p[stream] for (memb_, stream) in m.imemb if memb_ == memb
                )
            return Constraint.Skip

        b.Mempn = Constraint(
            [membrane], m.str, rule=Mempn, doc="pressure relation for non-permeate"
        )

        def Rec(_m, memb_, stream):
            if (memb_, stream) in m.pmemb and memb_ == membrane:
                return m.fc[stream, "h2"] >= m.membrane_recovery_sepc * sum(
                    m.fc[stream, "h2"] for (memb, stream) in m.imemb if memb == memb_
                )
            return Constraint.Skip

        b.rec = Constraint([membrane], m.str, rule=Rec, doc="recovery spec")

        def Pure(_m, memb, stream):
            if (memb, stream) in m.pmemb and memb == membrane:
                return m.fc[stream, "h2"] >= m.membrane_purity_sepc * m.f[stream]
            return Constraint.Skip

        b.pure = Constraint([membrane], m.str, rule=Pure, doc="purity spec")

    def build_multiple_mixer(b, multiple_mxr):
        """
        Functions relevant to the mixer block

        Parameters
        ----------
        b : Pyomo Block
            mixer block
        multiple_mxr : int
            Index of the mixer
        """

        def Mxrcmb(_b, mxr, compon):
            if mxr == multiple_mxr:
                return sum(
                    m.fc[stream, compon] for (mxr_, stream) in m.omxr if mxr == mxr_
                ) == sum(
                    m.fc[stream, compon] for (mxr_, stream) in m.imxr if mxr == mxr_
                )
            return Constraint.Skip

        b.mxrcmb = Constraint(
            [multiple_mxr], m.compon, rule=Mxrcmb, doc="component balance in mixer"
        )

        def Mxrhb(_b, mxr):
            if mxr == multiple_mxr and mxr != 2:
                return sum(
                    m.f[stream] * m.t[stream] * m.cp[stream]
                    for (mxr_, stream) in m.imxr
                    if mxr == mxr_
                ) == sum(
                    m.f[stream] * m.t[stream] * m.cp[stream]
                    for (mxr_, stream) in m.omxr
                    if mxr == mxr_
                )
            return Constraint.Skip

        b.mxrhb = Constraint([multiple_mxr], rule=Mxrhb, doc="heat balance in mixer")

        def Mxrhbq(_b, mxr):
            if mxr == 2 and mxr == multiple_mxr:
                return m.f[16] * m.t[16] == m.f[15] * m.t[15] - (
                    m.fc[20, "ben"] + m.fc[20, "tol"]
                ) * m.heatvap["tol"] / (100.0 * m.cp[15])
            return Constraint.Skip

        b.mxrhbq = Constraint([multiple_mxr], rule=Mxrhbq, doc="heat balance in quench")

        def Mxrpi(_b, mxr, stream):
            if (mxr, stream) in m.imxr and mxr == multiple_mxr:
                return m.mxrp[mxr] == m.p[stream]
            return Constraint.Skip

        b.mxrpi = Constraint(
            [multiple_mxr], m.str, rule=Mxrpi, doc="inlet pressure relation"
        )

        def Mxrpo(_b, mxr, stream):
            if (mxr, stream) in m.omxr and mxr == multiple_mxr:
                return m.mxrp[mxr] == m.p[stream]
            return Constraint.Skip

        b.mxrpo = Constraint(
            [multiple_mxr], m.str, rule=Mxrpo, doc="outlet pressure relation"
        )

    def build_pump(b, pump_):
        """
        Functions relevant to the pump block

        Parameters
        ----------
        b : Pyomo Block
            pump block
        pump_ : int
            Index of the pump
        """

        def Pumpcmb(_m, pump, compon):
            if pump == pump_ and compon in m.compon:
                return sum(
                    m.fc[stream, compon] for (pump_, stream) in m.opump if pump == pump_
                ) == sum(
                    m.fc[stream, compon] for (pump_, stream) in m.ipump if pump_ == pump
                )
            return Constraint.Skip

        b.pumpcmb = Constraint(
            [pump_], m.compon, rule=Pumpcmb, doc="component balance in pump"
        )

        def Pumphb(_m, pump):
            if pump == pump_:
                return sum(
                    m.t[stream] for (pump_, stream) in m.opump if pump == pump_
                ) == sum(m.t[stream] for (pump_, stream) in m.ipump if pump == pump_)
            return Constraint.Skip

        b.pumphb = Constraint([pump_], rule=Pumphb, doc="heat balance in pump")

        def Pumppr(_m, pump):
            if pump == pump_:
                return sum(
                    m.p[stream] for (pump_, stream) in m.opump if pump == pump_
                ) >= sum(m.p[stream] for (pump_, stream) in m.ipump if pump == pump_)
            return Constraint.Skip

        b.pumppr = Constraint([pump_], rule=Pumppr, doc="pressure relation in pump")

    def build_multiple_splitter(b, multi_splitter):
        """
        Functions relevant to the splitter block

        Parameters
        ----------
        b : Pyomo Block
            splitter block
        multi_splitter : int
            Index of the splitter
        """

        def Splcmb(_m, spl, stream, compon):
            if (spl, stream) in m.ospl and spl == multi_splitter:
                return m.fc[stream, compon] == sum(
                    m.e[stream] * m.fc[str2, compon]
                    for (spl_, str2) in m.ispl
                    if spl == spl_
                )
            return Constraint.Skip

        b.splcmb = Constraint(
            [multi_splitter],
            m.str,
            m.compon,
            rule=Splcmb,
            doc="component balance in splitter",
        )

        def Esum(_m, spl):
            if spl in m.spl and spl == multi_splitter:
                return (
                    sum(m.e[stream] for (spl_, stream) in m.ospl if spl_ == spl) == 1.0
                )
            return Constraint.Skip

        b.esum = Constraint(
            [multi_splitter], rule=Esum, doc="split fraction relation in splitter"
        )

        def Splpi(_m, spl, stream):
            if (spl, stream) in m.ispl and spl == multi_splitter:
                return m.splp[spl] == m.p[stream]
            return Constraint.Skip

        b.splpi = Constraint(
            [multi_splitter],
            m.str,
            rule=Splpi,
            doc="inlet pressure relation (splitter)",
        )

        def Splpo(_m, spl, stream):
            if (spl, stream) in m.ospl and spl == multi_splitter:
                return m.splp[spl] == m.p[stream]
            return Constraint.Skip

        b.splpo = Constraint(
            [multi_splitter],
            m.str,
            rule=Splpo,
            doc="outlet pressure relation (splitter)",
        )

        def Splti(_m, spl, stream):
            if (spl, stream) in m.ispl and spl == multi_splitter:
                return m.splt[spl] == m.t[stream]
            return Constraint.Skip

        b.splti = Constraint(
            [multi_splitter],
            m.str,
            rule=Splti,
            doc="inlet temperature relation (splitter)",
        )

        def Splto(_m, spl, stream):
            if (spl, stream) in m.ospl and spl == multi_splitter:
                return m.splt[spl] == m.t[stream]
            return Constraint.Skip

        b.splto = Constraint(
            [multi_splitter],
            m.str,
            rule=Splto,
            doc="outlet temperature relation (splitter)",
        )

    def build_valve(b, valve_):
        """
        Functions relevant to the valve block

        Parameters
        ----------
        b : Pyomo Block
            valve block
        valve_ : int
            Index of the valve
        """

        def Valcmb(_m, valve, compon):
            return sum(
                m.fc[stream, compon] for (valve_, stream) in m.oval if valve == valve_
            ) == sum(
                m.fc[stream, compon] for (valve_, stream) in m.ival if valve == valve_
            )

        b.valcmb = Constraint(
            [valve_], m.compon, rule=Valcmb, doc="component balance in valve"
        )

        def Valt(_m, valve):
            return sum(
                m.t[stream] / (m.p[stream] ** ((m.cp_cv_ratio - 1.0) / m.cp_cv_ratio))
                for (valv, stream) in m.oval
                if valv == valve
            ) == sum(
                m.t[stream] / (m.p[stream] ** ((m.cp_cv_ratio - 1.0) / m.cp_cv_ratio))
                for (valv, stream) in m.ival
                if valv == valve
            )

        b.valt = Constraint([valve_], rule=Valt, doc="temperature relation in valve")

        def Valp(_m, valve):
            return sum(
                m.p[stream] for (valv, stream) in m.oval if valv == valve
            ) <= sum(m.p[stream] for (valv, stream) in m.ival if valv == valve)

        b.valp = Constraint([valve_], rule=Valp, doc="pressure relation in valve")

    m.Prereference_factor = Param(
        initialize=6.3e10, doc="Pre-reference factor for reaction rate constant"
    )
    m.Ea_R = Param(
        initialize=-26167.0, doc="Activation energy for reaction rate constant"
    )
    m.pressure_drop = Param(initialize=0.20684, doc="Pressure drop")
    m.selectivity_1 = Param(initialize=0.0036, doc="Selectivity to benzene")
    m.selectivity_2 = Param(initialize=-1.544, doc="Selectivity to benzene")
    m.conversion_coefficient = Param(initialize=0.372, doc="Conversion coefficient")

    def build_reactor(b, rct):
        """
        Functions relevant to the reactor block

        Parameters
        ----------
        b : Pyomo Block
            reactor block
        rct : int
            Index of the reactor
        """

        def rctspec(_m, rct, stream):
            if (rct, stream) in m.irct:
                return m.fc[stream, "h2"] >= 5 * (
                    m.fc[stream, "ben"] + m.fc[stream, "tol"] + m.fc[stream, "dip"]
                )
            return Constraint.Skip

        b.Rctspec = Constraint(
            [rct], m.str, rule=rctspec, doc="specification on reactor feed stream"
        )

        def rxnrate(_m, rct):
            return m.krct[rct] == m.Prereference_factor * exp(
                m.Ea_R / (m.rctt[rct] * 100.0)
            )

        b.Rxnrate = Constraint([rct], rule=rxnrate, doc="reaction rate constant")

        def rctconv(_m, rct, stream, compon):
            if (rct, compon) in m.rkey and (rct, stream) in m.irct:
                return (
                    1.0 - m.conv[rct, compon]
                    == (
                        1.0
                        / (
                            1.0
                            + m.conversion_coefficient
                            * m.krct[rct]
                            * m.rctvol[rct]
                            * sqrt(m.fc[stream, compon] / 60 + m.eps1)
                            * (m.f[stream] / 60.0 + m.eps1) ** (-3.0 / 2.0)
                        )
                    )
                    ** 2.0
                )
            return Constraint.Skip

        b.Rctconv = Constraint(
            [rct], m.str, m.compon, rule=rctconv, doc="conversion of key component"
        )

        def rctsel(_m, rct):
            return (1.0 - m.sel[rct]) == m.selectivity_1 * (
                1.0 - m.conv[rct, "tol"]
            ) ** m.selectivity_2

        b.Rctsel = Constraint([rct], rule=rctsel, doc="selectivity to benzene")

        def rctcns(_m, rct, stream, compon):
            if (rct, compon) in m.rkey and (rct, stream) in m.irct:
                return (
                    m.consum[rct, compon] == m.conv[rct, compon] * m.fc[stream, compon]
                )
            return Constraint.Skip

        b.Rctcns = Constraint(
            [rct],
            m.str,
            m.compon,
            rule=rctcns,
            doc="consumption rate of key components",
        )

        def rctmbtol(_m, rct):
            return (
                sum(m.fc[stream, "tol"] for (rct_, stream) in m.orct if rct_ == rct)
                == sum(m.fc[stream, "tol"] for (rct_, stream) in m.irct if rct_ == rct)
                - m.consum[rct, "tol"]
            )

        b.Rctmbtol = Constraint(
            [rct], rule=rctmbtol, doc="mass balance in reactor (tol)"
        )

        def rctmbben(_m, rct):
            return (
                sum(m.fc[stream, "ben"] for (rct_, stream) in m.orct if rct_ == rct)
                == sum(m.fc[stream, "ben"] for (rct_, stream) in m.irct if rct_ == rct)
                + m.consum[rct, "tol"] * m.sel[rct]
            )

        b.Rctmbben = Constraint(
            [rct], rule=rctmbben, doc="mass balance in reactor (ben)"
        )

        def rctmbdip(_m, rct):
            return (
                sum(m.fc[stream, "dip"] for (rct1, stream) in m.orct if rct1 == rct)
                == sum(m.fc[stream, "dip"] for (rct1, stream) in m.irct if rct1 == rct)
                + m.consum[rct, "tol"] * 0.5
                + (
                    sum(m.fc[stream, "ben"] for (rct1, stream) in m.irct if rct1 == rct)
                    - sum(
                        m.fc[stream, "ben"] for (rct1, stream) in m.orct if rct1 == rct
                    )
                )
                * 0.5
            )

        b.Rctmbdip = Constraint(
            [rct], rule=rctmbdip, doc="mass balance in reactor (dip)"
        )

        def rctmbh2(_m, rct):
            return sum(
                m.fc[stream, "h2"] for (rct1, stream) in m.orct if rct1 == rct
            ) == sum(
                m.fc[stream, "h2"] for (rct1, stream) in m.irct if rct1 == rct
            ) - m.consum[
                rct, "tol"
            ] - sum(
                m.fc[stream, "dip"] for (rct1, stream) in m.irct if rct1 == rct
            ) + sum(
                m.fc[stream, "dip"] for (rct1, stream) in m.orct if rct1 == rct
            )

        b.Rctmbh2 = Constraint([rct], rule=rctmbh2, doc="mass balance in reactor (h2)")

        def rctpi(_m, rct, stream):
            if (rct, stream) in m.irct:
                return m.rctp[rct] == m.p[stream]
            return Constraint.Skip

        b.Rctpi = Constraint([rct], m.str, rule=rctpi, doc="inlet pressure relation")

        def rctpo(_m, rct, stream):
            if (rct, stream) in m.orct:
                return m.rctp[rct] - m.pressure_drop == m.p[stream]
            return Constraint.Skip

        b.Rctpo = Constraint([rct], m.str, rule=rctpo, doc="outlet pressure relation")

        def rcttave(_m, rct):
            return (
                m.rctt[rct]
                == (
                    sum(m.t[stream] for (rct1, stream) in m.irct if rct1 == rct)
                    + sum(m.t[stream] for (rct1, stream) in m.orct if rct1 == rct)
                )
                / 2
            )

        b.Rcttave = Constraint([rct], rule=rcttave, doc="average temperature relation")

        def Rctmbch4(_m, rct):
            return (
                sum(m.fc[stream, "ch4"] for (rct_, stream) in m.orct if rct_ == rct)
                == sum(m.fc[stream, "ch4"] for (rct_, stream) in m.irct if rct == rct_)
                + m.consum[rct, "tol"]
            )

        b.rctmbch4 = Constraint(
            [rct], rule=Rctmbch4, doc="mass balance in reactor (ch4)"
        )

        def Rcthbadb(_m, rct):
            if rct == 1:
                return m.heatrxn[rct] * m.consum[rct, "tol"] / 100.0 == sum(
                    m.cp[stream] * m.f[stream] * m.t[stream]
                    for (rct_, stream) in m.orct
                    if rct_ == rct
                ) - sum(
                    m.cp[stream] * m.f[stream] * m.t[stream]
                    for (rct_, stream) in m.irct
                    if rct_ == rct
                )
            return Constraint.Skip

        b.rcthbadb = Constraint([rct], rule=Rcthbadb, doc="heat balance (adiabatic)")

        def Rcthbiso(_m, rct):
            if rct == 2:
                return (
                    m.heatrxn[rct] * m.consum[rct, "tol"] * 60.0 * 8500 * 1.0e-09
                    == m.q[rct]
                )
            return Constraint.Skip

        b.rcthbiso = Constraint(
            [rct], rule=Rcthbiso, doc="temperature relation (isothermal)"
        )

        def Rctisot(_m, rct):
            if rct == 2:
                return sum(
                    m.t[stream] for (rct_, stream) in m.irct if rct_ == rct
                ) == sum(m.t[stream] for (rct_, stream) in m.orct if rct_ == rct)
            return Constraint.Skip

        b.rctisot = Constraint(
            [rct], rule=Rctisot, doc="temperature relation (isothermal)"
        )

    def build_single_mixer(b, mixer):
        """
        Functions relevant to the single mixer block

        Parameters
        ----------
        b : Pyomo Block
            single mixer block
        mixer : int
            Index of the mixer
        """

        def Mxr1cmb(m_, mxr1, str1, compon):
            if (mxr1, str1) in m.omxr1 and mxr1 == mixer:
                return m.fc[str1, compon] == sum(
                    m.fc[str2, compon] for (mxr1_, str2) in m.imxr1 if mxr1_ == mxr1
                )
            return Constraint.Skip

        b.mxr1cmb = Constraint(
            [mixer], m.str, m.compon, rule=Mxr1cmb, doc="component balance in mixer"
        )

    m.single_mixer = Block(m.mxr1, rule=build_single_mixer)

    # single output splitter
    def build_single_splitter(b, splitter):
        """
        Functions relevant to the single splitter block

        Parameters
        ----------
        b : Pyomo Block
            single splitter block
        splitter : int
            Index of the splitter
        """

        def Spl1cmb(m_, spl1, compon):
            return sum(
                m.fc[str1, compon] for (spl1_, str1) in m.ospl1 if spl1_ == spl1
            ) == sum(m.fc[str1, compon] for (spl1_, str1) in m.ispl1 if spl1_ == spl1)

        b.spl1cmb = Constraint(
            [splitter], m.compon, rule=Spl1cmb, doc="component balance in splitter"
        )

        def Spl1pi(m_, spl1, str1):
            if (spl1, str1) in m.ispl1:
                return m.spl1p[spl1] == m.p[str1]
            return Constraint.Skip

        b.spl1pi = Constraint(
            [splitter], m.str, rule=Spl1pi, doc="inlet pressure relation (splitter)"
        )

        def Spl1po(m_, spl1, str1):
            if (spl1, str1) in m.ospl1:
                return m.spl1p[spl1] == m.p[str1]
            return Constraint.Skip

        b.spl1po = Constraint(
            [splitter], m.str, rule=Spl1po, doc="outlet pressure relation (splitter)"
        )

        def Spl1ti(m_, spl1, str1):
            if (spl1, str1) in m.ispl1:
                return m.spl1t[spl1] == m.t[str1]
            return Constraint.Skip

        b.spl1ti = Constraint(
            [splitter], m.str, rule=Spl1ti, doc="inlet temperature relation (splitter)"
        )

        def Spl1to(m_, spl1, str1):
            if (spl1, str1) in m.ospl1:
                return m.spl1t[spl1] == m.t[str1]
            return Constraint.Skip

        b.spl1to = Constraint(
            [splitter], m.str, rule=Spl1to, doc="outlet temperature relation (splitter)"
        )

    m.single_splitter = Block(m.spl1, rule=build_single_splitter)

    # ## GDP formulation

    m.one = Set(initialize=[1])
    m.two = Set(initialize=[2])
    m.three = Set(initialize=[3])
    m.four = Set(initialize=[4])
    m.five = Set(initialize=[5])
    m.six = Set(initialize=[6])

    # first disjunction: Purify H2 inlet or not
    @m.Disjunct()
    def purify_H2(disj):
        disj.membrane_1 = Block(m.one, rule=build_membrane)
        disj.compressor_1 = Block(m.one, rule=build_compressor)
        disj.no_flow_2 = Constraint(expr=m.f[2] == 0)
        disj.pressure_match_out = Constraint(expr=m.p[6] == m.p[7])
        disj.tempressure_match_out = Constraint(expr=m.t[6] == m.t[7])

    @m.Disjunct()
    def no_purify_H2(disj):
        disj.no_flow_3 = Constraint(expr=m.f[3] == 0)
        disj.no_flow_4 = Constraint(expr=m.f[4] == 0)
        disj.no_flow_5 = Constraint(expr=m.f[5] == 0)
        disj.no_flow_6 = Constraint(expr=m.f[6] == 0)
        disj.pressure_match = Constraint(expr=m.p[2] == m.p[7])
        disj.tempressure_match = Constraint(expr=m.t[2] == m.t[7])

    @m.Disjunction()
    def inlet_treatment(m):
        return [m.purify_H2, m.no_purify_H2]

    m.multi_mixer_1 = Block(m.one, rule=build_multiple_mixer)
    m.furnace_1 = Block(m.one, rule=build_furnace)

    # second disjunction: adiabatic or isothermal reactor
    @m.Disjunct()
    def adiabatic_reactor(disj):
        disj.Adiabatic_reactor = Block(m.one, rule=build_reactor)
        disj.no_flow_12 = Constraint(expr=m.f[12] == 0)
        disj.no_flow_13 = Constraint(expr=m.f[13] == 0)
        disj.pressure_match = Constraint(expr=m.p[11] == m.p[14])
        disj.tempressure_match = Constraint(expr=m.t[11] == m.t[14])

    @m.Disjunct()
    def isothermal_reactor(disj):
        disj.Isothermal_reactor = Block(m.two, rule=build_reactor)
        disj.no_flow_10 = Constraint(expr=m.f[10] == 0)
        disj.no_flow_11 = Constraint(expr=m.f[11] == 0)
        disj.pressure_match = Constraint(expr=m.p[13] == m.p[14])
        disj.tempressure_match = Constraint(expr=m.t[13] == m.t[14])

    @m.Disjunction()
    def reactor_selection(m):
        return [m.adiabatic_reactor, m.isothermal_reactor]

    m.valve_3 = Block(m.three, rule=build_valve)
    m.multi_mixer_2 = Block(m.two, rule=build_multiple_mixer)
    m.exchanger_1 = Block(m.one, rule=build_exchanger)
    m.cooler_1 = Block(m.one, rule=build_cooler)
    m.flash_1 = Block(m.one, rule=build_flash)
    m.multi_splitter_2 = Block(m.two, rule=build_multiple_splitter)

    # third disjunction: recycle methane with membrane or purge it
    @m.Disjunct()
    def recycle_methane_purge(disj):
        disj.no_flow_54 = Constraint(expr=m.f[54] == 0)
        disj.no_flow_55 = Constraint(expr=m.f[55] == 0)
        disj.no_flow_56 = Constraint(expr=m.f[56] == 0)
        disj.no_flow_57 = Constraint(expr=m.f[57] == 0)

    @m.Disjunct()
    def recycle_methane_membrane(disj):
        disj.no_flow_53 = Constraint(expr=m.f[53] == 0)
        disj.membrane_2 = Block(m.two, rule=build_membrane)
        disj.compressor_4 = Block(m.four, rule=build_compressor)

    @m.Disjunction()
    def methane_treatment(m):
        return [m.recycle_methane_purge, m.recycle_methane_membrane]

    # fourth disjunction: recycle hydrogen with absorber or not
    @m.Disjunct()
    def recycle_hydrogen(disj):
        disj.no_flow_61 = Constraint(expr=m.f[61] == 0)
        disj.no_flow_73 = Constraint(expr=m.f[73] == 0)
        disj.no_flow_62 = Constraint(expr=m.f[62] == 0)
        disj.no_flow_64 = Constraint(expr=m.f[64] == 0)
        disj.no_flow_65 = Constraint(expr=m.f[65] == 0)
        disj.no_flow_68 = Constraint(expr=m.f[68] == 0)
        disj.no_flow_51 = Constraint(expr=m.f[51] == 0)
        disj.compressor_2 = Block(m.two, rule=build_compressor)
        disj.stream_1 = Constraint(expr=m.f[63] == 0)
        disj.stream_2 = Constraint(expr=m.f[67] == 0)
        disj.no_flow_69 = Constraint(expr=m.f[69] == 0)

    @m.Disjunct()
    def absorber_hydrogen(disj):
        disj.heater_4 = Block(m.four, rule=build_heater)
        disj.no_flow_59 = Constraint(expr=m.f[59] == 0)
        disj.no_flow_60 = Constraint(expr=m.f[60] == 0)
        disj.valve_6 = Block(m.six, rule=build_valve)
        disj.multi_mixer_4 = Block(m.four, rule=build_multiple_mixer)
        disj.absorber_1 = Block(m.one, rule=build_absorber)
        disj.compressor_3 = Block(m.three, rule=build_compressor)
        disj.absorber_stream = Constraint(expr=m.f[63] + m.f[67] <= 25)
        disj.pump_2 = Block(m.two, rule=build_pump)

    @m.Disjunction()
    def recycle_selection(m):
        return [m.recycle_hydrogen, m.absorber_hydrogen]

    m.multi_mixer_5 = Block(m.five, rule=build_multiple_mixer)
    m.multi_mixer_3 = Block(m.three, rule=build_multiple_mixer)
    m.multi_splitter_1 = Block(m.one, rule=build_multiple_splitter)

    # fifth disjunction: methane stabilizing selection
    @m.Disjunct()
    def methane_distillation_column(disj):
        disj.no_flow_23 = Constraint(expr=m.f[23] == 0)
        disj.no_flow_44 = Constraint(expr=m.f[44] == 0)
        disj.no_flow_45 = Constraint(expr=m.f[45] == 0)
        disj.no_flow_46 = Constraint(expr=m.f[46] == 0)
        disj.no_flow_47 = Constraint(expr=m.f[47] == 0)
        disj.no_flow_49 = Constraint(expr=m.f[49] == 0)
        disj.no_flow_48 = Constraint(expr=m.f[48] == 0)
        disj.heater_1 = Block(m.one, rule=build_heater)
        disj.stabilizing_Column_1 = Block(m.one, rule=build_distillation)
        disj.multi_splitter_3 = Block(m.three, rule=build_multiple_splitter)
        disj.valve_5 = Block(m.five, rule=build_valve)
        disj.pressure_match_1 = Constraint(expr=m.p[27] == m.p[30])
        disj.tempressure_match_1 = Constraint(expr=m.t[27] == m.t[30])
        disj.pressure_match_2 = Constraint(expr=m.p[50] == m.p[51])
        disj.tempressure_match_2 = Constraint(expr=m.t[50] == m.t[51])

    @m.Disjunct()
    def methane_flash_separation(disj):
        disj.heater_2 = Block(m.two, rule=build_heater)
        disj.no_flow_24 = Constraint(expr=m.f[24] == 0)
        disj.no_flow_25 = Constraint(expr=m.f[25] == 0)
        disj.no_flow_26 = Constraint(expr=m.f[26] == 0)
        disj.no_flow_27 = Constraint(expr=m.f[27] == 0)
        disj.no_flow_28 = Constraint(expr=m.f[28] == 0)
        disj.no_flow_29 = Constraint(expr=m.f[29] == 0)
        disj.no_flow_50 = Constraint(expr=m.f[50] == 0)
        disj.valve_1 = Block(m.one, rule=build_valve)
        disj.cooler_2 = Block(m.two, rule=build_cooler)
        disj.flash_2 = Block(m.two, rule=build_flash)
        disj.valve_4 = Block(m.four, rule=build_valve)
        disj.pressure_match_1 = Constraint(expr=m.p[48] == m.p[30])
        disj.tempressure_match_1 = Constraint(expr=m.t[48] == m.t[30])
        disj.pressure_match_2 = Constraint(expr=m.p[49] == m.p[51])
        disj.tempressure_match_2 = Constraint(expr=m.t[49] == m.t[51])

    @m.Disjunction()
    def H2_selection(m):
        return [m.methane_distillation_column, m.methane_flash_separation]

    m.benzene_column = Block(m.two, rule=build_distillation)

    # sixth disjunction: toluene stabilizing selection
    @m.Disjunct()
    def toluene_distillation_column(disj):
        disj.no_flow_37 = Constraint(expr=m.f[37] == 0)
        disj.no_flow_38 = Constraint(expr=m.f[38] == 0)
        disj.no_flow_39 = Constraint(expr=m.f[39] == 0)
        disj.no_flow_40 = Constraint(expr=m.f[40] == 0)
        disj.no_flow_41 = Constraint(expr=m.f[41] == 0)
        disj.stabilizing_Column_3 = Block(m.three, rule=build_distillation)
        disj.pressure_match = Constraint(expr=m.p[34] == m.p[42])
        disj.tempressure_match = Constraint(expr=m.t[34] == m.t[42])

    @m.Disjunct()
    def toluene_flash_separation(disj):
        disj.heater_3 = Block(m.three, rule=build_heater)
        disj.no_flow_33 = Constraint(expr=m.f[33] == 0)
        disj.no_flow_34 = Constraint(expr=m.f[34] == 0)
        disj.no_flow_35 = Constraint(expr=m.f[35] == 0)
        disj.valve_2 = Block(m.two, rule=build_valve)
        disj.flash_3 = Block(m.three, rule=build_flash)
        disj.pressure_match = Constraint(expr=m.p[40] == m.p[42])
        disj.tempressure_match = Constraint(expr=m.t[40] == m.t[42])

    @m.Disjunction()
    def toluene_selection(m):
        return [m.toluene_distillation_column, m.toluene_flash_separation]

    m.pump_1 = Block(m.one, rule=build_pump)
    m.abound = Constraint(expr=m.a[1] >= 0.0)

    # ## objective function

    m.hydrogen_purge_value = Param(
        initialize=1.08, doc="heating value of hydrogen purge"
    )
    m.electricity_cost = Param(
        initialize=0.04 * 24 * 365 / 1000,
        doc="electricity cost, value is 0.04 with the unit of [kW/h], now is [kW/yr/$1e3]",
    )
    m.methane_purge_value = Param(initialize=3.37, doc="heating value of methane purge")
    m.heating_cost = Param(
        initialize=8000.0, doc="heating cost (steam) with unit [1e6 kJ]"
    )
    m.cooling_cost = Param(
        initialize=700.0, doc="heating cost (water) with unit [1e6 kJ]"
    )
    m.fuel_cost = Param(initialize=4000.0, doc="fuel cost with unit [1e6 kJ]")
    m.abs_fixed_cost = Param(initialize=13, doc="fixed cost of absober [$1e3/yr]")
    m.abs_linear_coefficient = Param(
        initialize=1.2,
        doc="linear coefficient of absorber (times tray number) [$1e3/yr]",
    )
    m.compressor_fixed_cost = Param(
        initialize=7.155, doc="compressor fixed cost [$1e3/yr]"
    )
    m.compressor_fixed_cost_4 = Param(
        initialize=4.866, doc="compressor fixed cost for compressor 4 [$1e3/yr]"
    )
    m.compressor_linear_coefficient = Param(
        initialize=0.815,
        doc="compressor linear coefficient (vapor flow rate) [$1e3/yr]",
    )
    m.compressor_linear_coefficient_4 = Param(
        initialize=0.887,
        doc="compressor linear coefficient (vapor flow rate) [$1e3/yr]",
    )
    m.stabilizing_column_fixed_cost = Param(
        initialize=1.126, doc="stabilizing column fixed cost [$1e3/yr]"
    )
    m.stabilizing_column_linear_coefficient = Param(
        initialize=0.375,
        doc="stabilizing column linear coefficient (times number of trays) [$1e3/yr]",
    )
    m.benzene_column_fixed_cost = Param(
        initialize=16.3, doc="benzene column fixed cost [$1e3/yr]"
    )
    m.benzene_column_linear_coefficient = Param(
        initialize=1.55,
        doc="benzene column linear coefficient (times number of trays) [$1e3/yr]",
    )
    m.toluene_column_fixed_cost = Param(
        initialize=3.9, doc="toluene column fixed cost [$1e3/yr]"
    )
    m.toluene_column_linear_coefficient = Param(
        initialize=1.12,
        doc="toluene column linear coefficient (times number of trays) [$1e3/yr]",
    )
    m.furnace_fixed_cost = Param(
        initialize=6.20, doc="toluene column fixed cost [$1e3/yr]"
    )
    m.furnace_linear_coefficient = Param(
        initialize=1171.7, doc="furnace column linear coefficient [$1e3/(1e12 kJ/yr)]"
    )
    m.membrane_separator_fixed_cost = Param(
        initialize=43.24, doc="membrane separator fixed cost [$1e3/yr]"
    )
    m.membrane_separator_linear_coefficient = Param(
        initialize=49.0,
        doc="furnace column linear coefficient (times inlet flowrate) [$1e3/yr]",
    )
    m.adiabtic_reactor_fixed_cost = Param(
        initialize=74.3, doc="adiabatic reactor fixed cost [$1e3/yr]"
    )
    m.adiabtic_reactor_linear_coefficient = Param(
        initialize=1.257,
        doc="adiabatic reactor linear coefficient (times reactor volume) [$1e3/yr]",
    )
    m.isothermal_reactor_fixed_cost = Param(
        initialize=92.875, doc="isothermal reactor fixed cost [$1e3/yr]"
    )
    m.isothermal_reactor_linear_coefficient = Param(
        initialize=1.57125,
        doc="isothermal reactor linear coefficient (times reactor volume) [$1e3/yr]",
    )
    m.h2_feed_cost = Param(initialize=2.5, doc="h2 feed cost (95% h2,5% Ch4)")
    m.toluene_feed_cost = Param(initialize=14.0, doc="toluene feed cost (100% toluene)")
    m.benzene_product = Param(
        initialize=19.9, doc="benzene product profit (benzene >= 99.97%)"
    )
    m.diphenyl_product = Param(
        initialize=11.84, doc="diphenyl product profit (diphenyl = 100%)"
    )

    def profits_from_paper(m):
        return (
            510.0
            * (
                -m.h2_feed_cost * m.f[1]
                - m.toluene_feed_cost * (m.f[66] + m.f[67])
                + m.benzene_product * m.f[31]
                + m.diphenyl_product * m.f[35]
                + m.hydrogen_purge_value
                * (m.fc[4, "h2"] + m.fc[28, "h2"] + m.fc[53, "h2"] + m.fc[55, "h2"])
                + m.methane_purge_value
                * (m.fc[4, "ch4"] + m.fc[28, "ch4"] + m.fc[53, "ch4"] + m.fc[55, "ch4"])
            )
            - m.compressor_linear_coefficient * (m.elec[1] + m.elec[2] + m.elec[3])
            - m.compressor_linear_coefficient * m.elec[4]
            - m.compressor_fixed_cost
            * (
                m.purify_H2.binary_indicator_var
                + m.recycle_hydrogen.binary_indicator_var
                + m.absorber_hydrogen.binary_indicator_var
            )
            - m.compressor_fixed_cost * m.recycle_methane_membrane.binary_indicator_var
            - sum((m.electricity_cost * m.elec[comp]) for comp in m.comp)
            - (
                m.adiabtic_reactor_fixed_cost * m.adiabatic_reactor.binary_indicator_var
                + m.adiabtic_reactor_linear_coefficient * m.rctvol[1]
            )
            - (
                m.isothermal_reactor_fixed_cost
                * m.isothermal_reactor.binary_indicator_var
                + m.isothermal_reactor_linear_coefficient * m.rctvol[2]
            )
            - m.cooling_cost / 1000 * m.q[2]
            - (
                m.stabilizing_column_fixed_cost
                * m.methane_distillation_column.binary_indicator_var
                + m.stabilizing_column_linear_coefficient * m.ndist[1]
            )
            - (
                m.benzene_column_fixed_cost
                + m.benzene_column_linear_coefficient * m.ndist[2]
            )
            - (
                m.toluene_column_fixed_cost
                * m.toluene_distillation_column.binary_indicator_var
                + m.toluene_column_linear_coefficient * m.ndist[3]
            )
            - (
                m.membrane_separator_fixed_cost * m.purify_H2.binary_indicator_var
                + m.membrane_separator_linear_coefficient * m.f[3]
            )
            - (
                m.membrane_separator_fixed_cost
                * m.recycle_methane_membrane.binary_indicator_var
                + m.membrane_separator_linear_coefficient * m.f[54]
            )
            - (
                m.abs_fixed_cost * m.absorber_hydrogen.binary_indicator_var
                + m.abs_linear_coefficient * m.nabs[1]
            )
            - (m.fuel_cost * m.qfuel[1] + m.furnace_linear_coefficient * m.qfuel[1])
            - sum(m.cooling_cost * m.qc[hec] for hec in m.hec)
            - sum(m.heating_cost * m.qh[heh] for heh in m.heh)
            - m.furnace_fixed_cost
        )

    m.obj = Objective(rule=profits_from_paper, sense=maximize)
    # def profits_GAMS_file(m):

    #     "there are several differences between the data from GAMS file and the paper: 1. all the compressor share the same fixed and linear cost in paper but in GAMS they have different fixed and linear cost in GAMS file. 2. the fixed cost for absorber in GAMS file is 3.0 but in the paper is 13.0, but they are getting the same results 3. the electricity cost is not the same"

    #     return 510. * (- m.h2_feed_cost * m.f[1] - m.toluene_feed_cost * (m.f[66] + m.f[67]) + m.benzene_product * m.f[31] + m.diphenyl_product * m.f[35] + m.hydrogen_purge_value * (m.fc[4, 'h2'] + m.fc[28, 'h2'] + m.fc[53, 'h2'] + m.fc[55, 'h2']) + m.methane_purge_value * (m.fc[4, 'ch4'] + m.fc[28, 'ch4'] + m.fc[53, 'ch4'] + m.fc[55, 'ch4'])) - m.compressor_linear_coefficient * (m.elec[1] + m.elec[2] + m.elec[3]) - m.compressor_linear_coefficient_4  * m.elec[4] - m.compressor_fixed_cost * (m.purify_H2.binary_indicator_var + m.recycle_hydrogen.binary_indicator_var + m.absorber_hydrogen.binary_indicator_var) - m.compressor_fixed_cost_4 * m.recycle_methane_membrane.binary_indicator_var - sum((m.costelec * m.elec[comp]) for comp in m.comp) - (m.adiabtic_reactor_fixed_cost * m.adiabatic_reactor.binary_indicator_var + m.adiabtic_reactor_linear_coefficient * m.rctvol[1]) -  (m.isothermal_reactor_fixed_cost * m.isothermal_reactor.binary_indicator_var + m.isothermal_reactor_linear_coefficient * m.rctvol[2]) - m.cooling_cost/1000 * m.q[2] - (m.stabilizing_column_fixed_cost * m.methane_distillation_column.binary_indicator_var +m.stabilizing_column_linear_coefficient * m.ndist[1]) - (m.benzene_column_fixed_cost + m.benzene_column_linear_coefficient  * m.ndist[2]) - (m.toluene_column_fixed_cost * m.toluene_distillation_column.binary_indicator_var + m.toluene_column_linear_coefficient * m.ndist[3]) - (m.membrane_separator_fixed_cost * m.purify_H2.binary_indicator_var + m.membrane_separator_linear_coefficient * m.f[3]) - (m.membrane_separator_fixed_cost * m.recycle_methane_membrane.binary_indicator_var + m.membrane_separator_linear_coefficient * m.f[54]) - (3.0 * m.absorber_hydrogen.binary_indicator_var + m.abs_linear_coefficient * m.nabs[1]) - (m.fuel_cost * m.qfuel[1] + m.furnace_linear_coefficient* m.qfuel[1]) - sum(m.cooling_cost * m.qc[hec] for hec in m.hec) - sum(m.heating_cost * m.qh[heh] for heh in m.heh) - m.furnace_fixed_cost
    # m.obj = Objective(rule=profits_GAMS_file, sense=maximize)

    return m


# %%


def solve_with_gdpopt(m):
    """
    This function solves model m using GDPOpt

    Parameters
    ----------
    m : Pyomo Model
        The model to be solved

    Returns
    -------
    res : solver results
        The result of the optimization
    """
    opt = SolverFactory("gdpopt")
    res = opt.solve(
        m,
        tee=True,
        strategy="LOA",
        # strategy='GLOA',
        time_limit=3600,
        mip_solver="gams",
        mip_solver_args=dict(solver="cplex", warmstart=True),
        nlp_solver="gams",
        nlp_solver_args=dict(solver="ipopth", warmstart=True),
        minlp_solver="gams",
        minlp_solver_args=dict(solver="dicopt", warmstart=True),
        subproblem_presolve=False,
        # init_strategy='no_init',
        set_cover_iterlim=20,
        # calc_disjunctive_bounds=True
    )
    return res


def solve_with_minlp(m):
    """
    This function solves model m using minlp transformation by either Big-M or convex hull

    Parameters
    ----------
    m : Pyomo Model
        The model to be solved

    Returns
    -------
    result : solver results
        The result of the optimization
    """

    TransformationFactory("gdp.bigm").apply_to(m, bigM=60)
    # TransformationFactory('gdp.hull').apply_to(m)
    # result = SolverFactory('baron').solve(m, tee=True)
    result = SolverFactory("gams").solve(
        m, solver="baron", tee=True, add_options=["option reslim=120;"]
    )

    return result


# %%


def infeasible_constraints(m):
    """
    This function checks infeasible constraint in the model
    """
    log_infeasible_constraints(m)


# %%

# enumeration each possible route selection by fixing binary variable values in every disjunctions


def enumerate_solutions(m):
    """
    Enumerate all possible route selections by fixing binary variables in each disjunctions

    Parameters
    ----------
    m : Pyomo Model
        Pyomo model to be solved
    """

    H2_treatments = ["purify", "none_purify"]
    Reactor_selections = ["adiabatic_reactor", "isothermal_reactor"]
    Methane_recycle_selections = ["recycle_membrane", "recycle_purge"]
    Absorber_recycle_selections = ["no_absorber", "yes_absorber"]
    Methane_product_selections = ["methane_flash", "methane_column"]
    Toluene_product_selections = ["toluene_flash", "toluene_column"]

    for H2_treatment in H2_treatments:
        for Reactor_selection in Reactor_selections:
            for Methane_recycle_selection in Methane_recycle_selections:
                for Absorber_recycle_selection in Absorber_recycle_selections:
                    for Methane_product_selection in Methane_product_selections:
                        for Toluene_product_selection in Toluene_product_selections:
                            if H2_treatment == "purify":
                                m.purify_H2.indicator_var.fix(True)
                                m.no_purify_H2.indicator_var.fix(False)
                            else:
                                m.purify_H2.indicator_var.fix(False)
                                m.no_purify_H2.indicator_var.fix(True)
                            if Reactor_selection == "adiabatic_reactor":
                                m.adiabatic_reactor.indicator_var.fix(True)
                                m.isothermal_reactor.indicator_var.fix(False)
                            else:
                                m.adiabatic_reactor.indicator_var.fix(False)
                                m.isothermal_reactor.indicator_var.fix(True)
                            if Methane_recycle_selection == "recycle_membrane":
                                m.recycle_methane_purge.indicator_var.fix(False)
                                m.recycle_methane_membrane.indicator_var.fix(True)
                            else:
                                m.recycle_methane_purge.indicator_var.fix(True)
                                m.recycle_methane_membrane.indicator_var.fix(False)
                            if Absorber_recycle_selection == "yes_absorber":
                                m.absorber_hydrogen.indicator_var.fix(True)
                                m.recycle_hydrogen.indicator_var.fix(False)
                            else:
                                m.absorber_hydrogen.indicator_var.fix(False)
                                m.recycle_hydrogen.indicator_var.fix(True)
                            if Methane_product_selection == "methane_column":
                                m.methane_flash_separation.indicator_var.fix(False)
                                m.methane_distillation_column.indicator_var.fix(True)
                            else:
                                m.methane_flash_separation.indicator_var.fix(True)
                                m.methane_distillation_column.indicator_var.fix(False)
                            if Toluene_product_selection == "toluene_column":
                                m.toluene_flash_separation.indicator_var.fix(False)
                                m.toluene_distillation_column.indicator_var.fix(True)
                            else:
                                m.toluene_flash_separation.indicator_var.fix(True)
                                m.toluene_distillation_column.indicator_var.fix(False)
                            opt = SolverFactory("gdpopt")
                            res = opt.solve(
                                m,
                                tee=False,
                                strategy="LOA",
                                time_limit=3600,
                                mip_solver="gams",
                                mip_solver_args=dict(solver="gurobi", warmstart=True),
                                nlp_solver="gams",
                                nlp_solver_args=dict(
                                    solver="ipopth",
                                    add_options=["option optcr = 0"],
                                    warmstart=True,
                                ),
                                minlp_solver="gams",
                                minlp_solver_args=dict(solver="dicopt", warmstart=True),
                                subproblem_presolve=False,
                                init_strategy="no_init",
                                set_cover_iterlim=20,
                            )
                            print(
                                "{0:<30}{1:<30}{2:<30}{3:<30}{4:<30}{5:<30}{6:<30}{7:<30}".format(
                                    H2_treatment,
                                    Reactor_selection,
                                    Methane_recycle_selection,
                                    Absorber_recycle_selection,
                                    Methane_product_selection,
                                    Toluene_product_selection,
                                    str(res.solver.termination_condition),
                                    value(m.obj),
                                )
                            )


# %%
def show_decision(m):
    """
    print indicator variable value
    """
    if value(m.purify_H2.binary_indicator_var) == 1:
        print("purify inlet H2")
    else:
        print("no purify inlet H2")
    if value(m.adiabatic_reactor.binary_indicator_var) == 1:
        print("adiabatic reactor")
    else:
        print("isothermal reactor")
    if value(m.recycle_methane_membrane.binary_indicator_var) == 1:
        print("recycle_membrane")
    else:
        print("methane purge")
    if value(m.absorber_hydrogen.binary_indicator_var) == 1:
        print("yes_absorber")
    else:
        print("no_absorber")
    if value(m.methane_distillation_column.binary_indicator_var) == 1:
        print("methane_column")
    else:
        print("methane_flash")
    if value(m.toluene_distillation_column.binary_indicator_var) == 1:
        print("toluene_column")
    else:
        print("toluene_flash")


# %%


if __name__ == "__main__":
    # Create GDP model
    m = HDA_model()

    # Solve model
    res = solve_with_gdpopt(m)
    # res = solve_with_minlp(m)

    # Enumerate all solutions
    # res =  enumerate_solutions(m)

    # Check if constraints are violated
    infeasible_constraints(m)

    # show optimal flowsheet selection
    # show_decision(m)

    print(res)
