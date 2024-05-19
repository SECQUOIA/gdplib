import os
import sys
import pyomo.environ as pyo
from pyomo.core.base.misc import display
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.base.solvers import SolverFactory


def build_cstrs(NT: int = 5) -> pyo.ConcreteModel():
    """
    Build the CSTR superstructure model of size NT.
    NT is the number of reactors in series.
    The CSTRs have a single 1st order auto catalytic reaction A -> B and minimizes total reactors series volume.
    The optimal solution should yield NT reactors with a recycle before reactor NT.

    Parameters
    ----------
    NT : int
        Number of possible reactors in the reactor series superstructure

    Returns
    -------
    m : Pyomo.ConcreteModel
        Pyomo GDP model which represents the superstructure model of size of NT reactors.
    """

    # PYOMO MODEL
    m = pyo.ConcreteModel(name="gdp_reactors")

    # SETS
    m.I = pyo.Set(initialize=["A", "B"], doc="Set of components")
    m.N = pyo.RangeSet(1, NT, doc="Set of units in the superstructure")

    # PARAMETERS
    m.k = pyo.Param(
        initialize=2, doc="Kinetic constant [L/(mol*s)]"
    )  # Kinetic constant [L/(mol*s)]
    m.order1 = pyo.Param(
        initialize=1, doc="Partial order of reaction 1"
    )  # Partial order of reacton 1
    m.order2 = pyo.Param(
        initialize=1, doc="Partial order of reaction 2"
    )  # Partial order of reaction 2
    m.QF0 = pyo.Param(
        initialize=1, doc="Inlet volumetric flow [L/s]"
    )  # Inlet volumetric flow [L/s]
    C0_Def = {"A": 0.99, "B": 0.01}

    # Initial concentration of reagents [mol/L]
    m.C0 = pyo.Param(
        m.I, initialize=C0_Def, doc="Initial concentration of reagents [mol/L]"
    )

    # Inlet molar flow [mol/s]

    def F0_Def(m, i):
        """
        Inlet molar flow [mol/s] for component i.
        The function multiplies the initial concentration of component i by the inlet volumetric flow.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        i : float
            Index of the component in the reactor series.

        Returns
        -------
        Pyomo.Param
            Inlet molar flow [mol/s] for component i
        """
        return m.C0[i] * m.QF0

    m.F0 = pyo.Param(m.I, initialize=F0_Def)

    # BOOLEAN VARIABLES

    # Unreacted feed in reactor n
    m.YF = pyo.BooleanVar(m.N, doc="Unreacted feed in reactor n")

    # Existence of recycle flow in unit n
    m.YR = pyo.BooleanVar(m.N, doc="Existence of recycle flow in unit n")

    # Unit operation in n (True if unit n is a CSTR, False if unit n is a bypass)
    m.YP = pyo.BooleanVar(m.N, doc="Unit operation in n")

    # REAL VARIABLES

    # Network Variables
    # Outlet flow rate of the superstructure unit [L/s]
    m.Q = pyo.Var(
        m.N,
        initialize=0,
        within=pyo.NonNegativeReals,
        bounds=(0, 10),
        doc="Outlet flow rate of the superstructure unit [L/s]",
    )

    # Outlet flow rate recycle activation of the superstructure unit [L/s]
    m.QFR = pyo.Var(
        m.N,
        initialize=0,
        within=pyo.NonNegativeReals,
        bounds=(0, 10),
        doc="Outlet flow rate recycle activation of the superstructure unit [L/s]",
    )

    # Molar flow [mol/s]
    m.F = pyo.Var(
        m.I,
        m.N,
        initialize=0,
        within=pyo.NonNegativeReals,
        bounds=(0, 10),
        doc="Molar flow [mol/s]",
    )

    # Molar flow  recycle activation [mol/s]
    m.FR = pyo.Var(
        m.I,
        m.N,
        initialize=0,
        within=pyo.NonNegativeReals,
        bounds=(0, 10),
        doc="Molar flow  recycle activation [mol/s]",
    )

    # Reaction rate [mol/(L*s)]
    m.rate = pyo.Var(
        m.I,
        m.N,
        initialize=0,
        within=pyo.Reals,
        bounds=(-10, 10),
        doc="Reaction rate [mol/(L*s)]",
    )

    # Reactor volume [L]
    m.V = pyo.Var(
        m.N,
        initialize=0,
        within=pyo.NonNegativeReals,
        bounds=(0, 10),
        doc="Reactor volume [L]",
    )

    # Volume activation [L]
    m.c = pyo.Var(
        m.N,
        initialize=0,
        within=pyo.NonNegativeReals,
        bounds=(0, 10),
        doc="Volume activation [L]",
    )

    # Splitter Variables
    # Recycle flow rate [L/s]
    m.QR = pyo.Var(
        initialize=0,
        within=pyo.NonNegativeReals,
        bounds=(0, 10),
        doc="Recycle flow rate [L/s]",
    )

    # Product flow rate [L/s]
    m.QP = pyo.Var(
        initialize=0,
        within=pyo.NonNegativeReals,
        bounds=(0, 10),
        doc="Product flow rate [L/s]",
    )

    # Recycle molar flow [mol/s]
    m.R = pyo.Var(
        m.I,
        initialize=0,
        within=pyo.NonNegativeReals,
        bounds=(0, 10),
        doc="Recycle molar flow [mol/s]",
    )

    # Product molar flow [mol/s]
    m.P = pyo.Var(
        m.I,
        initialize=0,
        within=pyo.NonNegativeReals,
        bounds=(0, 10),
        doc="Product molar flow [mol/s]",
    )

    # CONSTRAINTS

    # Unreacted Feed Balances
    # Unreacted feed unit mole balance

    def unreact_mole_rule(m, i, n):
        """
        Unreacted feed unit mole balance.
        The mole balance is calculated using the inlet molar flow, the recycle molar flow, the outlet molar flow, and the reaction rate for each component.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        i : float
            Index of the component in the reactor series.
        n : int
            Index of the reactor in the reactor series. The reactor index starts at 1. The number increases on the left direction of the reactor series. The last reactor is indexed as NT which is the feed for reagent.

        Returns
        -------
        Pyomo.Constraint or Pyomo.Constraint.Skip
            The constraint for the unreacted feed unit mole balance if n is equal to NT. Otherwise, it returns Pyomo.Constraint.Skip.
        """
        if n == NT:
            return m.F0[i] + m.FR[i, n] - m.F[i, n] + m.rate[i, n] * m.V[n] == 0
        else:
            return pyo.Constraint.Skip

    m.unreact_mole = pyo.Constraint(
        m.I, m.N, rule=unreact_mole_rule, doc="Unreacted feed unit mole balance"
    )

    # Unreacted feed unit continuity

    def unreact_cont_rule(m, n):
        """
        Unreacted feed unit continuity.
        The continuity is calculated using the inlet volumetric flow, the recycle flow rate, and the outlet flow rate for the reactor NT.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        Pyomo.Constraint or Pyomo.Constraint.Skip
            The constraint for the unreacted feed unit continuity if n is equal to NT. Otherwise, it returns Pyomo.Constraint.Skip.
        """
        if n == NT:
            return m.QF0 + m.QFR[n] - m.Q[n] == 0
        else:
            return pyo.Constraint.Skip

    m.unreact_cont = pyo.Constraint(
        m.N, rule=unreact_cont_rule, doc="Unreacted feed unit continuity"
    )

    # Reactor Balances
    # Reactor mole balance

    def react_mole_rule(m, i, n):
        """
        Reactor mole balance.
        The mole balance is calculated using the inlet molar flow, the recycle molar flow, the outlet molar flow, and the reaction rate for each component.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        i : float
            Index of the component in the reactor series.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        Pyomo.Constraint or Pyomo.Constraint.Skip
            The constraint for the reactor mole balance if n is different from NT. Otherwise, it returns Pyomo.Constraint.Skip.
        """
        if n != NT:
            return m.F[i, n + 1] + m.FR[i, n] - m.F[i, n] + m.rate[i, n] * m.V[n] == 0
        else:
            return pyo.Constraint.Skip

    m.react_mole = pyo.Constraint(
        m.I, m.N, rule=react_mole_rule, doc="Reactor mole balance"
    )

    # Reactor continuity

    def react_cont_rule(m, n):
        """
        Reactor continuity.
        The continuity is calculated using the inlet volumetric flow, the recycle flow rate, and the outlet flow rate for the reactor n.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        Pyomo.Constraint or Pyomo.Constraint.Skip
            The constraint for the reactor continuity if n is different from NT. Otherwise, it returns Pyomo.Constraint.Skip.
        """
        if n != NT:
            return m.Q[n + 1] + m.QFR[n] - m.Q[n] == 0
        else:
            return pyo.Constraint.Skip

    m.react_cont = pyo.Constraint(m.N, rule=react_cont_rule, doc="Reactor continuity")

    # Splitting Point Balances
    # Splitting point mole balance

    def split_mole_rule(m, i):
        """
        Splitting point mole balance.
        The mole balance is calculated using the product molar flow, the recycle molar flow, and the outlet molar flow for each component.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        i : float
            Index of the component in the reactor series.

        Returns
        -------
        Pyomo.Constraint
            The constraint for the splitting point mole balance.
        """
        return m.F[i, 1] - m.P[i] - m.R[i] == 0

    m.split_mole = pyo.Constraint(
        m.I, rule=split_mole_rule, doc="Splitting point mole balance"
    )

    # Splitting point continuity

    def split_cont_rule(m):
        """
        Splitting point continuity.
        The continuity is calculated using the product flow rate, the recycle flow rate, and the outlet flow rate for the reactor 1.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.

        Returns
        -------
        Pyomo.Constraint
            The constraint for the splitting point continuity.
        """
        return m.Q[1] - m.QP - m.QR == 0

    m.split_cont = pyo.Constraint(
        rule=split_cont_rule, doc="Splitting point continuity"
    )

    # Splitting point additional constraints

    def split_add_rule(m, i):
        """
        Splitting point additional constraints.
        Molarity constraints over initial and final flows, read as an multiplication avoid the numerical complication.
        m.P[i]/m.QP =  m.F[i,1]/m.Q[1] (molarity balance)

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        i : float
            Index of the component in the reactor series.

        Returns
        -------
        Pyomo.Constraint
            The constraint for the splitting point additional constraints.
        """
        return m.P[i] * m.Q[1] - m.F[i, 1] * m.QP == 0

    m.split_add = pyo.Constraint(
        m.I, rule=split_add_rule, doc="Splitting point additional constraints"
    )

    # Product Specification

    def prod_spec_rule(m):
        """
        Product B Specification.
        The product B specification is calculated using the product flow rate and the product molar flow.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.

        Returns
        -------
        Pyomo.Constraint
            The constraint for the product B specification.
        """
        return m.QP * 0.95 - m.P["B"] == 0

    m.prod_spec = pyo.Constraint(rule=prod_spec_rule, doc="Product B Specification")

    # Volume Constraint

    def vol_cons_rule(m, n):
        """
        Volume Constraint.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        Pyomo.Constraint or Pyomo.Constraint.Skip
            The constraint for the volume constraint if n is different from 1. Otherwise, it returns Pyomo.Constraint.Skip.
        """
        if n != 1:
            return m.V[n] - m.V[n - 1] == 0
        else:
            return pyo.Constraint.Skip

    m.vol_cons = pyo.Constraint(m.N, rule=vol_cons_rule, doc="Volume Constraint")

    # YD Disjunction block equation definition

    def build_cstr_equations(disjunct, n):
        """
        Build the constraints for the activation of the CSTR reactor.

        Parameters
        ----------
        disjunct : Pyomo.Disjunct
            Pyomo Disjunct block to include the constraints for the activation of the CSTR reactor.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        None
            None, the function builds the constraints for the activation of the CSTR reactor.
        """
        m = disjunct.model()

        # Reaction rates calculation
        @disjunct.Constraint()
        def YPD_rate_calc(disjunct):
            """
            Calculate the reaction rate of A for each reactor.
            The reaction rate is calculated using the kinetic constant, the outlet flow rate, the molar flow of A, and the molar flow of B.
            The outlet flow is multiplied on the reaction rate to avoid numerical complications.

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Pyomo Disjunct block to include the constraints for the activation of the CSTR reactor.

            Returns
            -------
            Pyomo.Constraint
                The constraint for the calculation of the reaction rate of A for each reactor.
            """
            return (
                m.rate["A", n] * ((m.Q[n]) ** m.order1) * ((m.Q[n]) ** m.order2)
                + m.k * ((m.F["A", n]) ** m.order1) * ((m.F["B", n]) ** m.order2)
                == 0
            )

        # Reaction rate relation
        @disjunct.Constraint()
        def YPD_rate_rel(disjunct):
            """
            Reaction rate relation for defining pyomo model.
            Since the chemical reaction goes from A to B, the rate of A is equal to the negative rate of B.

            A -> B

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Pyomo Disjunct block to include the constraints for the activation of the CSTR reactor.

            Returns
            -------
            Pyomo.Constraint
                Reaction rate relation for defining pyomo model.
            """
            return m.rate["B", n] + m.rate["A", n] == 0

        # Volume activation
        @disjunct.Constraint()
        def YPD_vol_act(disjunct):
            """
            Volume Activation function for defining pyomo model.

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Pyomo Disjunct block to include the constraints for the activation of the CSTR reactor.

            Returns
            -------
            Pyomo.Constraint
                Activation function for the volume of the reactor.
            """
            return m.c[n] - m.V[n] == 0

    def build_bypass_equations(disjunct, n):
        """
        Build the constraints for the deactivation of the reactor (bypass the reactor).

        Parameters
        ----------
        disjunct : Pyomo.Disjunct
            Pyomo Disjunct block to include the constraints for the deactivation of the reactor (bypass the reactor).
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        None
            None, the function builds the constraints for the deactivation of the reactor (bypass the reactor).
        """
        m = disjunct.model()

        # FR deactivation
        @disjunct.Constraint(m.I)
        def neg_YPD_FR_deactivation(disjunct, i):
            """
            Deactivation of the recycle flow for each component in the reactor series.
            There are no recycle flows when the reactor is deactivated (bypassed).

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Pyomo Disjunct block to include the constraints for the deactivation of the reactor (bypass the reactor).
            i : float
                Index of the component in the reactor series.

            Returns
            -------
            Pyomo.Constraint
                Deactivation of the recycle flow for each component in the reactor series.
            """
            return m.FR[i, n] == 0

        # Rate deactivation
        @disjunct.Constraint(m.I)
        def neg_YPD_rate_deactivation(disjunct, i):
            """
            Deactivate the reaction rate for each component in the reactor series.
            There are no reaction rates when the reactor is deactivated (bypassed).

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Pyomo Disjunct block to include the constraints for the deactivation of the reactor (bypass the reactor).
            i : float
                Index of the component in the reactor series.

            Returns
            -------
            Pyomo.Constraint
                Deactivation of the reaction rate for each component in the reactor series.
            """
            return m.rate[i, n] == 0

        # QFR deactivation
        @disjunct.Constraint()
        def neg_YPD_QFR_deactivation(disjunct):
            """
            Deactivate the outlet flow rate recycle activation of the reactor.
            There is no outlet flow rate recycle activation when the reactor is deactivated (bypassed).

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Pyomo Disjunct block to include the constraints for the deactivation of the reactor (bypass the reactor).

            Returns
            -------
            Pyomo.Constraint
                Deactivation of the outlet flow rate recycle activation of the reactor.
            """
            return m.QFR[n] == 0

        @disjunct.Constraint()
        def neg_YPD_vol_deactivation(disjunct):
            """
            Volume deactivation function for defining pyomo model.
            There is no volume when the reactor is deactivated (bypassed).

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Pyomo Disjunct block to include the constraints for the deactivation of the reactor (bypass the reactor).

            Returns
            -------
            Pyomo.Constraint
                Volume deactivation function for defining pyomo model.
            """
            return m.c[n] == 0

    # YR Disjuction block equation definition

    def build_recycle_equations(disjunct, n):
        """
        Build the constraints for the activation of the recycle flow.

        Parameters
        ----------
        disjunct : Pyomo.Disjunct
            Pyomo Disjunct block to include the constraints for the activation of the reactor (recycle flow existence).
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        None
            None, the function builds the constraints for the activation of the recycle flow.
        """
        m = disjunct.model()

        # FR activation
        @disjunct.Constraint(m.I)
        def YRD_FR_act(disjunct, i):
            """
            Activation of the recycle flow for each component in the reactor series.

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Pyomo Disjunct block to include the constraints for the activation of the reactor (recycle flow existence).
            i : float
                Index of the component in the reactor series.

            Returns
            -------
            Pyomo.Constraint
                Activation of the recycle flow for each component in the reactor series.
            """
            return m.FR[i, n] - m.R[i] == 0

        # QFR activation
        @disjunct.Constraint()
        def YRD_QFR_act(disjunct):
            """
            Activation of the outlet flow rate recycle activation of the reactor.

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Pyomo Disjunct block to include the constraints for the activation of the reactor (recycle flow existence).

            Returns
            -------
            Pyomo.Constraint
                Activation of the outlet flow rate recycle activation of the reactor.
            """
            return m.QFR[n] - m.QR == 0

    def build_no_recycle_equations(disjunct, n):
        """
        Build the constraints for the deactivation of the recycle flow.

        Parameters
        ----------
        disjunct : Pyomo.Disjunct
            Pyomo Disjunct block to include the constraints for the deactivation of the reactor (recycle flow absence).
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        None
            None, the function builds the constraints for the deactivation of the recycle flow.
        """
        m = disjunct.model()

        # FR deactivation
        @disjunct.Constraint(m.I)
        def neg_YRD_FR_deactivation(disjunct, i):
            """
            Deactivation of the recycle flow for each component in the reactor series.

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Pyomo Disjunct block to include the constraints for the deactivation of the reactor (recycle flow absence).
            i : float
                Index of the component in the reactor series.

            Returns
            -------
            Pyomo.Constraint
                Deactivation of the recycle flow for each component in the reactor series.
            """
            return m.FR[i, n] == 0

        # QFR deactivation
        @disjunct.Constraint()
        def neg_YRD_QFR_deactivation(disjunct):
            """
            Deactivation of the outlet flow rate recycle activation of the reactor.

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                Pyomo Disjunct block to include the constraints for the deactivation of the reactor (recycle flow absence).

            Returns
            -------
            Pyomo.Constraint
                Deactivation of the outlet flow rate recycle activation of the reactor.
            """
            return m.QFR[n] == 0

    # Create disjunction blocks
    m.YR_is_recycle = Disjunct(
        m.N, rule=build_recycle_equations, doc="Recycle flow in reactor n"
    )
    m.YR_is_not_recycle = Disjunct(
        m.N, rule=build_no_recycle_equations, doc="No recycle flow in reactor n"
    )

    m.YP_is_cstr = Disjunct(m.N, rule=build_cstr_equations, doc="CSTR reactor n")
    m.YP_is_bypass = Disjunct(m.N, rule=build_bypass_equations, doc="Bypass reactor n")

    # Create disjunctions

    @m.Disjunction(m.N)
    def YP_is_cstr_or_bypass(m, n):
        """
        Build the disjunction for the activation of the CSTR reactor or bypass the reactor.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        list
            list of the disjunctions for the activation of the CSTR reactor or bypass the reactor
        """
        return [m.YP_is_cstr[n], m.YP_is_bypass[n]]

    @m.Disjunction(m.N)
    def YR_is_recycle_or_not(m, n):
        """
        Build the disjunction for the existence of a recycle flow in the reactor.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        list
            list of the disjunctions for the existence of a recycle flow in the reactor
        """
        return [m.YR_is_recycle[n], m.YR_is_not_recycle[n]]

    # Associate Boolean variables with with disjunctions
    for n in m.N:
        m.YP[n].associate_binary_var(m.YP_is_cstr[n].indicator_var)
        m.YR[n].associate_binary_var(m.YR_is_recycle[n].indicator_var)

    # Logic Constraints
    # Unit must be a CSTR to include a recycle

    def cstr_if_recycle_rule(m, n):
        """
        Build the logical constraint for the unit to be a CSTR to include a recycle.
        The existence of a recycle flow implies the existence of a CSTR reactor.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        Pyomo.LogicalConstraint
            Logical constraint for the unit to be a CSTR to include a recycle.
        """
        return m.YR[n].implies(m.YP[n])

    m.cstr_if_recycle = pyo.LogicalConstraint(
        m.N, rule=cstr_if_recycle_rule, doc="Unit must be a CSTR to include a recycle"
    )

    # There is only one unreacted feed

    def one_unreacted_feed_rule(m):
        """
        Build the logical constraint for the existence of only one unreacted feed.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.

        Returns
        -------
        Pyomo.LogicalConstraint
            Logical constraint for the existence of only one unreacted feed.
        """
        return pyo.exactly(1, m.YF)

    m.one_unreacted_feed = pyo.LogicalConstraint(
        rule=one_unreacted_feed_rule, doc="There is only one unreacted feed"
    )

    # There is only one recycle stream

    def one_recycle_rule(m):
        """
        Build the logical constraint for the existence of only one recycle stream.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.

        Returns
        -------
        Pyomo.LogicalConstraint
            Logical constraint for the existence of only one recycle stream.
        """
        return pyo.exactly(1, m.YR)

    m.one_recycle = pyo.LogicalConstraint(
        rule=one_recycle_rule, doc="There is only one recycle stream"
    )

    # Unit operation in n constraint

    def unit_in_n_rule(m, n):
        """
        Build the logical constraint for the unit operation in n.
        If n is equal to 1, the unit operation is a CSTR.
        Otherwise, the unit operation for reactor n except reactor 1 is equivalent to the logical OR of the negation of the unreacted feed of the previous reactors and the unreacted feed of reactor n.
        Reactor n is active if either the previous reactors (1 through n-1) have no unreacted feed or reactor n has unreacted feed.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        Pyomo.LogicalConstraint
            Logical constraint for the unit operation in n.
        """
        if n == 1:
            return m.YP[n].equivalent_to(True)
        else:
            return m.YP[n].equivalent_to(
                pyo.lor(pyo.land(~m.YF[n2] for n2 in range(1, n)), m.YF[n])
            )

    m.unit_in_n = pyo.LogicalConstraint(m.N, rule=unit_in_n_rule)

    # OBJECTIVE

    def obj_rule(m):
        """
        Objective function to minimize the total reactor volume.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.

        Returns
        -------
        Pyomo Objective
            Objective function to minimize the total reactor volume.
        """
        return sum(m.c[n] for n in m.N)

    m.obj = pyo.Objective(
        rule=obj_rule, sense=pyo.minimize, doc="minimum total reactor volume"
    )

    return m


if __name__ == "__main__":
    m = build_cstrs()
    pyo.TransformationFactory("core.logical_to_linear").apply_to(m)
    pyo.TransformationFactory("gdp.bigm").apply_to(m)
    pyo.SolverFactory("gams").solve(
        m, solver="baron", tee=True, add_options=["option optcr=1e-6;"]
    )
    display(m)
