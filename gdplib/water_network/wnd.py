"""
Water Network Design
--------------------

Water networks involve water-using process and treatment units, offering several water integration alternatives that reduce freshwater consumption and wastewater generation while minimizing the total network cost subject to a specified discharge limit.

In the Water Treatment Network (WTN) design problem, given is a set of water streams with known concentrations of contaminants and flow rate.
The objective is to find the set of treatment units and interconnections that minimize the cost of the WTN while satisfying maximum concentrations of contaminants in the reclaimed outlet stream.
The WTN superstructure consists of a set of treatment units, contaminated feed streams carrying a set of contaminants, and a discharge unit.
The fouled feed waters can be allocated to one or more treatment units or disposed of in the sink unit. Upon treatment, the reclaimed streams can be recycled, forwarded to other treatment units, or discharged into the sink unit.

The mass balances are defined in terms of total flows and contaminants concentration.
Nonconvexities arise from bilinear terms “flows times concentration” in the mixers mass balances and concave investment cost functions of treatment units.

The instance incorporates two approximations of the concave cost term (piecewise linear and quadratic) to reformulate the GDP model (approximation = 'none') into a bilinear quadratic one.
The user can create each instance like this:

build_model(approximation='none')
build_model(approximation='quadratic')
build_model(approximation='piecewise')

The general model description can be summarized as follows:
Min Cost of Treatment Units
s.t.
Physical Constraints:
(a) Mass balance around each splitter
(b) Mass balance around each mixer
(c) Mass balance around each treatment unit
Performance Constraints:
(d) Contaminant composition of the purified stream less or equal than a given limit
for each contaminant.
Logic Constraints:
(e) Treatment units not chosen have their inlet flow set to zero
(f) Every treatment unit chosen must have a minimum flow

Assumptions:
(i) The performance of the treatment units only depends on the total flow entering the unit and its composition.
(ii) The flow of contaminants leaving the unit is a linear function of the inlet flow of contaminants. 

Case study
----------
The WTN comprises five inlet streams with four contaminants and four treatment units.
The contaminant concentration and flow rate of the feed streams, contaminant recovery rates, minimum flow rate and cost coefficients of the treatment units, and the upper limit on the molar flow of contaminant j in the purified stream, are reported in (Ruiz and Grossmann, 2009).

References
---------- 
Ruiz J., Grossmann IE. Water Treatment Network Design. 2009 Available from CyberInfrastructure for [MINLP](<www.minlp.org>), a collaboration of Carnegie Mellon University and IBM 
at: www.minlp.org/library/problem/index.php?i=24

Ruiz, J. P., & Grossmann, I. E. (2011). Using redundancy to strengthen the relaxation for the global optimization of MINLP problems. Computers & Chemical Engineering, 35(12), 2729–2740. https://doi.org/10.1016/J.COMPCHEMENG.2011.01.035

"""

# Importing the required libraries
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import os
import re
from scipy.optimize import curve_fit
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

from pyomo.core.expr.logical_expr import *

wnd_dir = os.path.dirname(os.path.realpath(__file__))

# Data for the problem
# The feed.csv file contains the contaminant concentration and flow rate of the feed streams.
# The TU.csv file contains the contaminant recovery rates, minimum flow rate, and cost coefficients of the treatment units.
# The T.csv file contains the upper limit on the molar flow of contaminant j in the purified stream.

feed = pd.read_csv(os.path.join(wnd_dir, "feed.csv"), index_col=0)
TU = pd.read_csv(os.path.join(wnd_dir, "TU.csv"), index_col=0)
T = pd.read_csv(os.path.join(wnd_dir, "T.csv"), index_col=0)

# GDP model

# TU mass balances in the disjunct.


def build_model(approximation='quadratic'):
    """
    Builds a Pyomo ConcreteModel for Water Network Design.
    Generates a Pyomo model for the water network design problem with the specified approximation for the capital cost term of the active treatment units.

    Returns
    -------
    pyo.ConcreteModel
        A Pyomo ConcreteModel object for the water network design.

    """
    print(f"Using {approximation} approximation for the capital cost term.")
    m = pyo.ConcreteModel('Water Network Design')

    # ============================================================================
    #     Sets
    # =============================================================================

    m.contaminant = pyo.Set(doc='Set of contaminants', initialize=T.index.tolist())

    m.FSU = pyo.Set(doc='Set of feed splitters', initialize=feed.index.tolist())

    m.feed = pyo.Set(
        doc='Set of feedstreams',
        initialize=['f' + str(nf) for nf in pyo.RangeSet(len(m.FSU))],
    )

    m.TU = pyo.Set(doc='Set of treatment units, TU', initialize=TU.index.tolist())

    m.inTU = pyo.Set(
        doc="Inlet TU Port",
        initialize=['mt' + str(ntu) for ntu in pyo.RangeSet(len(m.TU))],
    )

    m.outTU = pyo.Set(
        doc="Outlet TU Port",
        initialize=['st' + str(ntu) for ntu in pyo.RangeSet(len(m.TU))],
    )

    m.TUport = pyo.Set(doc="Inlet and Outlet TU Ports", initialize=m.inTU | m.outTU)

    m.units = pyo.Set(
        doc="Superstructure Units",
        initialize=m.feed | m.FSU | m.TU | m.TUport | ['dm'] | ['sink'],
    )

    m.splitters = pyo.Set(
        doc="Set of splitters", within=m.units, initialize=m.FSU | m.outTU
    )

    m.mixers = pyo.Set(doc="Set of mixers", within=m.units, initialize=['dm'] | m.inTU)

    m.MU_TU_streams = pyo.Set(
        doc="MU to TU 1-1 port pairing",
        initialize=m.inTU * m.TU,
        filter=lambda _, x, y: re.findall(r'\d+', x) == re.findall(r'\d+', y),
    )

    m.TU_SU_streams = pyo.Set(
        doc="TU to SU 1-1 port pairing",
        initialize=m.TU * m.outTU,
        filter=lambda _, x, y: re.findall(r'\d+', x) == re.findall(r'\d+', y),
    )

    m.TU_streams = pyo.Set(
        doc="Set of feasible TU streams",
        initialize=m.MU_TU_streams | m.outTU * m.inTU | m.TU_SU_streams,
    )

    m.feed_streams = pyo.Set(
        doc="Feed to FSU 1-1 port pairing",
        initialize=m.feed * m.FSU,
        filter=lambda _, x, y: re.findall(r'\d+', x) == re.findall(r'\d+', y),
    )

    m.streams = pyo.Set(
        doc="Set of feasible streams",
        initialize=m.feed_streams
        | m.FSU * ['dm']
        | m.FSU * m.inTU
        | m.TU_streams
        | m.outTU * ['dm']
        | [('dm', 'sink')],
    )

    m.from_splitters = pyo.Set(
        doc='Streams from splitters',
        within=m.streams,
        initialize=m.streams,
        filter=lambda _, x, y: x in m.splitters or (x, y) == ('dm', 'sink'),
    )
    m.to_splitters = pyo.Set(
        doc='Streams to splitters',
        within=m.streams,
        initialize=m.streams,
        filter=lambda _, x, y: y in m.splitters or (x, y) in m.feed_streams,
    )

    # =============================================================================
    #     Treatment Units Parameters
    # =============================================================================

    m.removal_ratio = pyo.Param(
        m.contaminant,
        m.TU,
        doc='Removal ratio for contaminant j in treatment unit t',
        within=pyo.NonNegativeReals,
        initialize=lambda _, j, k: TU.loc[k, j],
    )
    m.theta = pyo.Param(
        m.TU,
        within=pyo.NonNegativeReals,
        doc='Cost parameter theta for unit TU',
        initialize=lambda _, k: TU.loc[k, 'theta'],
    )
    m.beta = pyo.Param(
        m.TU,
        within=pyo.NonNegativeReals,
        doc='Cost parameter beta for unit TU',
        initialize=lambda _, k: TU.loc[k, 'beta'],
    )
    m.gamma = pyo.Param(
        m.TU,
        within=pyo.NonNegativeReals,
        doc='Cost parameter gamma for unit TU',
        default=0,
        initialize=0,
    )

    # =============================================================================
    #     Variables
    # =============================================================================

    m.conc = pyo.Var(
        m.contaminant,
        m.streams - m.MU_TU_streams - m.TU_SU_streams,
        doc="Concentration",
        domain=pyo.NonNegativeReals,
        bounds=(0, 100),
        initialize=lambda _, j, i, k: feed.loc[i, j] if i in m.FSU else None,
    )

    # Concentration of component j in feedstream i
    for j, i, k in m.contaminant * (m.FSU * ['dm'] | m.FSU * m.inTU | m.feed_streams):
        # print(j,i,k)
        if i in m.feed:
            m.conc[j, i, k].fix(feed.loc[k, j])
        else:
            m.conc[j, i, k].fix(feed.loc[i, j])

    m.flow = pyo.Var(
        m.streams - m.MU_TU_streams - m.TU_SU_streams,
        doc="Flowrate",
        domain=pyo.NonNegativeReals,
        bounds=lambda _, i, k: (
            (None, feed.loc[i, 'flow_rate'])
            if i in m.FSU
            # else (None,100), # Upper bound for the flow rate from Ruiz and Grossmann (2009) is 100
            else (0, feed['flow_rate'].sum())
        ),
    )

    # Fix the flow rates of the feed streams
    [m.flow[i, k].fix(feed.loc[k, 'flow_rate']) for i, k in m.feed_streams]
    # Fix the flow rate from the mixer to the sink
    m.flow['dm', 'sink'].fix(feed['flow_rate'].sum())

    m.costTU = pyo.Var(
        m.TU,
        domain=pyo.NonNegativeReals,
        doc='CTUk cost of TU',
        bounds=lambda _, tu: (
            0,
            m.beta[tu] * feed['flow_rate'].sum()
            + m.gamma[tu]
            + m.theta[tu] * feed['flow_rate'].sum() ** 0.7,
        ),
    )

    # =============================================================================
    #     Mass balances
    # =============================================================================

    # @m.Expression(m.units)
    @m.Expression(m.units - m.TU - m.outTU)
    def _flow_into(m, option):
        """
        Expression for the flow into a unit.

        Parameters
        ----------
        m : ConcreteModel
            The Pyomo model.
        option : str
            The unit except the treatment units and the outlet treatment units.

        Returns
        -------
        Expression
            The flow into the unit, which is the sum of the flow from the feed splitters and the flow from the mixers.
        """
        return sum(m.flow[src, sink] for src, sink in m.streams if sink == option)

    @m.Expression(m.units - m.TU - m.outTU, m.contaminant)
    def _conc_into(m, option, j):
        """
        Expression for the mass balance of the contaminant in the unit to compute the concentration into the unit.

        Parameters
        ----------
        m : ConcreteModel
            The Pyomo model.
        option : str
            The units except the treatment units and the treatment units' outlet.
        j : str
            The contaminant.

        Returns
        -------
        Expression
            The concentration into the unit, which is the sum of the flow from the feed splitters and the flow from the mixers multiplied by the concentration of the contaminant.
        """
        if option in m.mixers:
            return sum(
                m.flow[src, sink] * m.conc[j, src, sink]
                for src, sink in m.streams
                if sink == option
            )
        return pyo.Expression.Skip

    @m.Expression(m.units - m.TU - m.inTU)
    def _flow_out_from(m, option):
        """
        Expression for to compute the flow rate leaving the unit.

        Parameters
        ----------
        m : ConcreteModel
            The Pyomo model.
        option : str
            The units except the treatment units and the treatment units' inlet.

        Returns
        -------
        Expression
            The flow rate leaving the unit, which is the sum of the flow to the discharge mixers and the splitters.
        """
        return sum(m.flow[src, sink] for src, sink in m.streams if src == option)

    @m.Expression(m.units - m.inTU, m.contaminant)
    def _conc_out_from(m, option, j):
        """The mass balance of the contaminant in the unit to compute the concentration out of the unit.

        Parameters
        ----------
        m : ConcreteModel
            The Pyomo model.
        option : str
            The unit except the treatment units and the treatment units' inlet.
        j : str
            The contaminant.

        Returns
        -------
        Expression
            The concentration out of the unit, which is the sum of the flow from the feed splitters and the flow from the mixers multiplied by the concentration of the contaminant.
        """
        if option in m.mixers:
            return sum(
                m.flow[src, sink] * m.conc[j, src, sink]
                for src, sink in m.streams
                if src == option
            )
        return pyo.Expression.Skip

    # Global constraints

    # Adding the mass balances for the splitters and mixers
    m.mixer_balances = pyo.ConstraintList()
    # for mu in m.mixers:
    for mu in m.mixers - m.TU - m.inTU:
        # Flowrate balance
        [m.mixer_balances.add(m._flow_into[mu] == m._flow_out_from[mu])]
        # Concentration balance
        [
            m.mixer_balances.add(m._conc_into[mu, j] == m._conc_out_from[mu, j])
            for j in m.contaminant
        ]

    m.splitter_balances = pyo.ConstraintList()

    for su in m.splitters - m.TU - m.outTU:
        # Flowrate balance
        [m.splitter_balances.add(m._flow_into[su] == m._flow_out_from[su])]

    for src1, sink1 in m.from_splitters - m.TU_SU_streams:
        for src2, sink2 in m.to_splitters - m.TU_SU_streams:
            # Concentration balance
            [
                m.splitter_balances.add(
                    m.conc[j, src2, sink2] == m.conc[j, src1, sink1]
                )
                for j in m.contaminant
                if src1 == sink2
            ]

    @m.Constraint(m.contaminant)
    def outlet_contaminant_limit(m, j):
        """Constraint: Outlet contaminant limit.
        The constraint ensures that the concentration of the contaminant in the outlet stream is less than or equal to the specified limit.

        Parameters
        ----------
        m : ConcreteModel
            The Pyomo model.
        j : str
            The contaminant.

        Returns
        -------
        Constraint
            The constraint that the concentration of the contaminant in the outlet stream is less than or equal to the specified limit.
        """
        return m._conc_into['dm', j] <= T.loc[j].values[0] * 10

    @m.Disjunct(m.TU)
    def unit_exists(disj, unit):
        '''Disjunct: Unit exists.
        This disjunct specifies that the unit exists.
        '''
        pass

    @m.Disjunct(m.TU)
    def unit_absent(no_unit, unit):
        '''Disjunct: Unit absent.
        This disjunct specifies that the unit is absent.'''

        @no_unit.Constraint(m.inTU)
        def _no_flow_in(disj, mt):
            '''Constraint: No flow enters the unit when the unit is inactive.

            This constraint ensures that no flow enters the unit when the unit is inactive.
            Flow from the feed splitters and mixers to the unit is set to zero when the unit is inactive.

            Parameters
            ----------
            disj : Disjunct
                The disjunct for the inactive unit.
            mt : str
                The inlet TU port.

            Returns
            -------
            Constraint
                The constraint that no flow enters the unit when the unit is inactive.
            '''
            if (mt, unit) in m.MU_TU_streams:
                return m._flow_into[mt] == 0
            return pyo.Constraint.Skip

        @no_unit.Constraint(doc='Cost inactive TU')
        def _no_cost(disj):
            '''Constraint: Cost of inactive unit.
            This constraint ensures that the cost of the inactive unit is zero.

            Returns
            -------
            Constraint
                The constraint enforcing the cost of the inactive unit to be zero.
            '''
            return m.costTU[unit] == 0

        pass

    @m.Disjunction(m.TU)
    def unit_exists_or_not(m, unit):
        '''Disjunction: Unit exists or not.
        This disjunctiont specifies if the treatment unit exists or does not exist.

        Parameters
        ----------
        m : ConcreteModel
            The Pyomo model.
        unit : str
            The treatment unit.

        Returns
        -------
        Disjunction
            The disjunction specifying if the treatment unit exists or does not exist.
        '''
        return [m.unit_exists[unit], m.unit_absent[unit]]

    # Yunit is a Boolean variable that indicates if the unit is active.
    m.Yunit = pyo.BooleanVar(
        m.TU, doc="Boolean variable for existence of a treatment unit"
    )
    for unit in m.TU:
        m.Yunit[unit].associate_binary_var(
            m.unit_exists[unit].indicator_var.get_associated_binary()
        )

    # Logical constraints that ensure that at least one unit is active
    # m.atleast_oneRU = pyo.LogicalConstraint(expr=atleast(1, m.Yunit))

    for unit in m.TU:
        ue = m.unit_exists[unit]
        ue.streams = pyo.Set(
            doc="Streams in active TU",
            initialize=m.TU_streams,
            filter=lambda _, x, y: x == unit or y == unit,
        )
        ue.MU_TU_streams = pyo.Set(
            doc="MU to TU 1-1 port pairing",
            initialize=m.inTU * m.TU,
            filter=lambda _, x, y: re.findall(r'\d+', x) == re.findall(r'\d+', y)
            and y == unit,
        )

        ue.flow = pyo.Var(
            ue.streams,
            doc="TU streams flowrate",
            domain=pyo.NonNegativeReals,
            # bounds=lambda _,i,k:(TU.loc[unit,'L'],100) # Upper bound for the flow rate from Ruiz and Grossmann (2009) is 100
            bounds=lambda _, i, k: (TU.loc[unit, 'L'], feed['flow_rate'].sum()),
        )

        ue.conc = pyo.Var(
            m.contaminant,
            ue.streams,
            doc="TU streams concentration",
            domain=pyo.NonNegativeReals,
            bounds=lambda _, j, i, k: (0, 100) if i == unit else (0, 4),
            # initialize= lambda _, j, i, k: feed.loc[i,j] if i in m.FSU else None
        )

        # Adding the mass balances for the active treatment units
        ue.balances_con = pyo.ConstraintList(doc='TU Material balances')
        # The flowrate at the inlet of the treatment unit is equal to the flowrate at the outlet of the treatment unit.
        [
            ue.balances_con.add(ue.flow[mt, unit] == ue.flow[unit, st])
            for mt, t in ue.streams
            if t == unit
            for t, st in ue.streams
            if t == unit
        ]
        # The concentration of the contaminant at the inlet of the treatment unit is equal to the concentration of the contaminant at the outlet of the treatment unit times the removal ratio.
        [
            ue.balances_con.add(
                ue.conc[j, unit, st]
                == (1 - m.removal_ratio[j, t]) * ue.conc[j, mt, unit]
            )
            for mt, t in ue.streams
            if t == unit
            for t, st in ue.streams
            if t == unit
            for j in m.contaminant
        ]
        # Treatment unit's mixer mass balance on the flowrate.
        [
            ue.balances_con.add(m._flow_into[mt] == ue.flow[mt, unit])
            for mt, t in ue.streams
            if t == unit
        ]
        # Treatment unit's mixer mass balance on the concentration of contaminants.
        [
            ue.balances_con.add(
                m._conc_into[mt, j] == ue.conc[j, mt, unit] * ue.flow[mt, unit]
            )
            for mt, t in ue.streams
            if t == unit
            for j in m.contaminant
        ]
        # Treatment unit's splitter mass balance on the flowrate.
        [
            ue.balances_con.add(ue.flow[unit, st] == m._flow_out_from[st])
            for t, st in ue.streams
            if t == unit
        ]
        # Treatment unit's splitter mass balance on the concentration of contaminants.
        [
            ue.balances_con.add(ue.conc[j, src2, sink2] == m.conc[j, src1, sink1])
            for src1, sink1 in m.from_splitters
            for src2, sink2 in ue.streams
            if src2 == unit and src1 == sink2
            for j in m.contaminant
        ]
        # Setting inlet flowrate bounds for the active treatment units.
        ue.flow_bound = pyo.ConstraintList(doc='Flowrate bounds to/from active RU')
        [
            ue.flow_bound.add(
                (ue.flow[mt, unit].lb, m._flow_into[mt], ue.flow[mt, unit].ub)
            )
            for mt, t in m.MU_TU_streams
            if t == unit
        ]

        # Approximation of the concave capital cost term for the active treatment units in the objective function.

        if approximation == 'quadratic':
            # New variable for potential term in capital cost.  Z = sum(Q)**0.7, Z >= 0
            ue.cost_var = pyo.Var(
                m.TU,
                domain=pyo.NonNegativeReals,
                initialize=lambda _, unit: TU.loc[unit, 'L'] ** 0.7,
                bounds=(0, 100**0.7),
                doc="New var for potential term in capital cost",
            )

            def _quadratic_curve_fit(lb, ub, x):
                """This function fits a quadratic curve to the function Z^(1/0.7).

                Parameters
                ----------
                lb : float
                    The lower bound of the flow rate.
                ub : float
                    The upper bound of the flow rate.
                x : variable
                    The flow rate.

                Returns
                -------
                Expression
                    The quadratic curve fit to the function Z^(1/0.7).
                """
                # Equally spaced points between the lower and upper bounds of the flow rate
                z = np.linspace(lb, ub, 100)

                def _func(x, a, b):
                    """This function computes the quadratic curve fit.
                    The curve fit is a quadratic function of the form a*x + b*x^2.

                    Parameters
                    ----------
                    x : variable
                        The flow rate.
                    a : float
                        The coefficient a.
                    b : float
                        The coefficient b.

                    Returns
                    -------
                    Expression
                        The quadratic curve fit for the function Z^(1/0.7).
                    """
                    return a * x + b * x**2

                # popt is the optimal values for the coefficients a and b, pcov is the covariance matrix
                # curve_fit is used to fit the quadratic curve to the function Z^(1/0.7) given the flow rate and the quadratic expression _func.
                popt, pcov = curve_fit(_func, z, z ** (1 / 0.7))

                return _func(x, *popt)

            # Z^(1/0.7)=sum(Q)
            @ue.Constraint(doc='New var potential term in capital cost cstr.')
            def _cost_nv(ue):
                """Constraint: New variable potential term in capital cost.
                The constraint ensures that the new variable potential term in the capital cost is equal to the flow rate entering the treatment unit.

                Parameters
                ----------
                ue : Disjunct
                    The disjunct for the active treatment unit.

                Returns
                -------
                Constraint
                    The constraint that the new variable potential term in the capital cost is equal to the flow rate.
                """
                for mt, t in ue.streams:
                    if t == unit:
                        return (
                            _quadratic_curve_fit(
                                0, ue.cost_var[unit].ub, ue.cost_var[unit]
                            )
                            == ue.flow[mt, unit]
                        )

        elif approximation == "quadratic2":

            def _func2(x, a, b, c):
                """This function computes the quadratic curve fit.
                The curve fit is a quadratic function of the form a + b*x + c*x^2 given the flow rate and the coefficients a, b, and c obtained from the curve_fit function.

                Parameters
                ----------
                x : variable
                    The flow rate.
                a : float
                    The coefficient a.
                b : float
                    The coefficient b.
                c : float
                    The coefficient c.

                Returns
                -------
                Expression
                    The quadratic curve fit for the function q^0.7.
                """
                return a + b * x + c * x**2

            def _g(x):
                """This function provides the expression for the quadratic curve fit of the capital cost using the curve_fit function.

                Parameters
                ----------
                x : variable
                    The flow rate.

                Returns
                -------
                Expression
                    The quadratic curve fit of the capital cost
                """
                # Equally spaced points between 0 and 100
                q = np.linspace(0, 100, 100)
                # curve_fit is used to fit the quadratic curve to the function q^0.7 given the flow rate and the quadratic expression func2.
                # popt is the optimal values for the coefficients a, b, and c, pcov is the covariance matrix
                # func is a quadratic function of the form a + b*x + c*x^2.
                popt, pcov = curve_fit(_func2, q, q**0.7)
                return _func2(x, *popt)

        elif approximation == "piecewise":
            # New variable for potential term in capital cost.  Z = sum(Q)**0.7, Z >= 0
            ue.cost_var = pyo.Var(
                ue.MU_TU_streams,
                domain=pyo.NonNegativeReals,
                initialize=lambda _, mt, unit: TU.loc[unit, 'L'] ** 0.7,
                bounds=lambda _, mt, unit: (
                    TU.loc[unit, 'L'] ** 0.7,
                    feed['flow_rate'].sum() ** 0.7,
                ),
                # bounds= lambda _,mt,unit: (TU.loc[unit,'L']**0.7,100**0.7), # Upper bound for the flow rate from Ruiz and Grossmann (2009) is 100
                doc="New var for potential term in capital cost",
            )

            # to avoid warnings, we set breakpoints at or beyond the bounds
            PieceCnt = 100
            bpts = []
            for mt, t in ue.streams:
                if t == unit:
                    Topx = ue.flow[mt, unit].ub
            for i in range(PieceCnt + 2):
                bpts.append(float((i * Topx) / PieceCnt))

            def _func(model, i, j, xp):
                """This function provides the expression for the piecewise linear approximation of the capital cost using the Picewise class.

                Parameters
                ----------
                model : ConcreteModel
                    The Pyomo model.
                    i : str
                    j : str
                    xp : variable
                        The flow rate.

                Returns
                -------
                Expression
                    The expression for the piecewise linear approximation of the capital cost.
                """
                # we not need i, j, but it are passed as the index for the constraint
                return xp**0.7

            # Piecewise is a class that provides a piecewise linear approximation of a function using the INC representation.
            # The INC representation is a piecewise linear approximation of a function using the incremental form.
            ue.ComputeObj = pyo.Piecewise(
                ue.MU_TU_streams,
                ue.cost_var,
                ue.flow,
                pw_pts=bpts,
                pw_constr_type='EQ',
                f_rule=_func,
                pw_repn='INC',
            )

            # Setting bounds for the INC_delta variable which is a binary variable that indicates the piecewise linear segment.
            for i, j in ue.MU_TU_streams:
                ue.ComputeObj[i, j].INC_delta.setub(1)
                ue.ComputeObj[i, j].INC_delta.setlb(0)

        @ue.Constraint(doc='Cost active TU')
        def costTU(ue):
            """Constraint: Cost of active treatment unit.
            The constraint ensures that the cost of the active treatment unit is equal to the sum of an investment cost which is proportional to the total flow to 0.7 exponent and an operating cost which is proportional to the flow.
            If approximation is quadratic, the investment cost is approximated by a quadratic function of the flow rate.
            If approximation is piecewise, the investment cost is approximated by a piecewise linear function of the flow rate.
            If approximation is none, the investment cost is equal to the flow rate to the 0.7 exponent, the original concave function.

            Parameters
            ----------
            ue : Disjunct
                The disjunct for the active treatment unit.

            Returns
            -------
            Constraint
                The constraint that the cost of the active treatment unit is equal to the sum of an investment cost and an operating cost.
            """
            for mt, t in ue.streams:
                if approximation == 'quadratic':
                    new_var = ue.cost_var[unit]
                elif approximation == 'quadratic2':
                    new_var = _g(ue.flow[mt, unit])
                elif approximation == 'piecewise':
                    new_var = ue.cost_var[mt, unit]
                elif approximation == 'none':
                    new_var = ue.flow[mt, unit] ** 0.7
                if t == unit:
                    return (
                        m.costTU[unit]
                        == m.beta[unit] * ue.flow[mt, unit]
                        + m.gamma[unit]
                        + m.theta[unit] * new_var
                    )

    # Objective function: minimize the total cost of the treatment units
    m.obj = pyo.Objective(expr=sum(m.costTU[k] for k in m.TU), sense=pyo.minimize)

    # init_vars.InitMidpoint().apply_to(m)

    return m
