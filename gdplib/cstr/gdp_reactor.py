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
    m = pyo.ConcreteModel(name='gdp_reactors')

    # SETS
    m.I = pyo.Set(initialize=['A', 'B'], doc='Set of components')
    m.N = pyo.RangeSet(1, NT, doc='Set of units in the superstructure')

    # PARAMETERS
    m.k = pyo.Param(initialize=2, doc="Kinetic constant [L/(mol*s)]")  # Kinetic constant [L/(mol*s)]
    m.order1 = pyo.Param(initialize=1, doc="Partial order of reaction 1")  # Partial order of reacton 1
    m.order2 = pyo.Param(initialize=1, doc="Partial order of reaction 2")  # Partial order of reaction 2
    m.QF0 = pyo.Param(initialize=1, doc="Inlet volumetric flow [L/s]")  # Inlet volumetric flow [L/s]
    C0_Def = {'A': 0.99, 'B': 0.01}

    # Initial concentration of reagents [mol/L]
    m.C0 = pyo.Param(m.I, initialize=C0_Def, doc="Initial concentration of reagents [mol/L]")

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
        return m.C0[i]*m.QF0
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
    m.Q = pyo.Var(m.N, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 10), doc="Outlet flow rate of the superstructure unit [L/s]")

    # Outlet flow rate recycle activation of the superstructure unit [L/s]
    m.QFR = pyo.Var(m.N, initialize=0,
                   within=pyo.NonNegativeReals, bounds=(0, 10), doc="Outlet flow rate recycle activation of the superstructure unit [L/s]")

    # Molar flow [mol/s]
    m.F = pyo.Var(m.I, m.N, initialize=0,
                 within=pyo.NonNegativeReals, bounds=(0, 10), doc="Molar flow [mol/s]")

    # Molar flow  recycle activation [mol/s]
    m.FR = pyo.Var(m.I, m.N, initialize=0,
                  within=pyo.NonNegativeReals, bounds=(0, 10), doc="Molar flow  recycle activation [mol/s]")

    # Reaction rate [mol/(L*s)]
    m.rate = pyo.Var(m.I, m.N, initialize=0, within=pyo.Reals, bounds=(-10, 10), doc="Reaction rate [mol/(L*s)]")

    # Reactor volume [L]
    m.V = pyo.Var(m.N, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 10), doc="Reactor volume [L]")

    # Volume activation [L]
    m.c = pyo.Var(m.N, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 10), doc="Volume activation [L]")

    # Splitter Variables
    # Recycle flow rate [L/s]
    m.QR = pyo.Var(initialize=0, within=pyo.NonNegativeReals, bounds=(0, 10), doc="Recycle flow rate [L/s]")

    # Product flow rate [L/s]
    m.QP = pyo.Var(initialize=0, within=pyo.NonNegativeReals, bounds=(0, 10), doc="Product flow rate [L/s]")

    # Recycle molar flow [mol/s]
    m.R = pyo.Var(m.I, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 10), doc="Recycle molar flow [mol/s]")

    # Product molar flow [mol/s]
    m.P = pyo.Var(m.I, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 10), doc="Product molar flow [mol/s]")

    # CONSTRAINTS

    # Unreacted Feed Balances
    # Unreacted feed unit mole balance

    def unreact_mole_rule(m, i, n):
        """
        Unreacted feed unit mole balance.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        i : float
            Index of the component in the reactor series.
        n : int
            Index of the reactor in the reactor series. The reactor index starts at 1. The number increases on the left direction

        Returns
        -------
        Pyomo.Constraint or Pyomo.Constraint.Skip
            _description_
        """
        if n == NT:
            return m.F0[i] + m.FR[i, n] - m.F[i, n] + m.rate[i, n]*m.V[n] == 0
        else:
            return pyo.Constraint.Skip

    m.unreact_mole = pyo.Constraint(m.I, m.N, rule=unreact_mole_rule)

    # Unreacted feed unit continuity

    def unreact_cont_rule(m, n):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        _type_
            _description_
        """
        if n == NT:
            return m.QF0 + m.QFR[n] - m.Q[n] == 0
        else:
            return pyo.Constraint.Skip

    m.unreact_cont = pyo.Constraint(m.N, rule=unreact_cont_rule)

    # Reactor Balances
    # Reactor mole balance

    def react_mole_rule(m, i, n):
        """_summary_

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
        _type_
            _description_
        """
        if n != NT:
            return m.F[i, n+1] + m.FR[i, n] - m.F[i, n] + m.rate[i, n]*m.V[n] == 0
        else:
            return pyo.Constraint.Skip

    m.react_mole = pyo.Constraint(m.I, m.N, rule=react_mole_rule)

    # Reactor continuity

    def react_cont_rule(m, n):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        _type_
            _description_
        """
        if n != NT:
            return m.Q[n+1] + m.QFR[n] - m.Q[n] == 0
        else:
            return pyo.Constraint.Skip

    m.react_cont = pyo.Constraint(m.N, rule=react_cont_rule)

    # Splitting Point Balances
    # Splitting point mole balance

    def split_mole_rule(m, i):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        i : float
            Index of the component in the reactor series.

        Returns
        -------
        _type_
            _description_
        """
        return m.F[i, 1] - m.P[i] - m.R[i] == 0

    m.split_mole = pyo.Constraint(m.I, rule=split_mole_rule)

    # Splitting point continuity

    def split_cont_rule(m):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.

        Returns
        -------
        _type_
            _description_
        """
        return m.Q[1] - m.QP - m.QR == 0

    m.split_cont = pyo.Constraint(rule=split_cont_rule)

    # Splitting point additional constraints

    def split_add_rule(m, i):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        i : float
            Index of the component in the reactor series.

        Returns
        -------
        _type_
            _description_
        """
        return m.P[i]*m.Q[1] - m.F[i, 1]*m.QP == 0

    m.split_add = pyo.Constraint(m.I, rule=split_add_rule)

    # Product Specification

    def prod_spec_rule(m):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.

        Returns
        -------
        _type_
            _description_
        """
        return m.QP*0.95 - m.P['B'] == 0

    m.prod_spec = pyo.Constraint(rule=prod_spec_rule)

    # Volume Constraint

    def vol_cons_rule(m, n):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        _type_
            _description_
        """
        if n != 1:
            return m.V[n] - m.V[n-1] == 0
        else:
            return pyo.Constraint.Skip

    m.vol_cons = pyo.Constraint(m.N, rule=vol_cons_rule)

    # YD Disjunction block equation definition

    def build_cstr_equations(disjunct, n):
        """_summary_

        Parameters
        ----------
        disjunct : _type_
            _description_
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        _type_
            _description_
        """
        m = disjunct.model()

        # Reaction rates calculation
        @disjunct.Constraint()
        def YPD_rate_calc(disjunct):
            """_summary_

            Parameters
            ----------
            disjunct : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            return m.rate['A', n]*((m.Q[n])**m.order1)*((m.Q[n])**m.order2)+m.k*((m.F['A', n])**m.order1)*((m.F['B', n])**m.order2) == 0

        # Reaction rate relation
        @disjunct.Constraint()
        def YPD_rate_rel(disjunct):
            """_summary_

            Parameters
            ----------
            disjunct : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            return m.rate['B', n] + m.rate['A', n] == 0

        # Volume activation
        @disjunct.Constraint()
        def YPD_vol_act(disjunct):
            """_summary_

            Parameters
            ----------
            disjunct : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            return m.c[n] - m.V[n] == 0

    def build_bypass_equations(disjunct, n):
        """_summary_

        Parameters
        ----------
        disjunct : _type_
            _description_
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        _type_
            _description_
        """
        m = disjunct.model()

        # FR desactivation
        @disjunct.Constraint(m.I)
        def neg_YPD_FR_desact(disjunct, i):
            """_summary_

            Parameters
            ----------
            disjunct : _type_
                _description_
            i : float
                Index of the component in the reactor series.

            Returns
            -------
            _type_
                _description_
            """
            return m.FR[i, n] == 0

        # Rate desactivation
        @disjunct.Constraint(m.I)
        def neg_YPD_rate_desact(disjunct, i):
            """_summary_

            Parameters
            ----------
            disjunct : _type_
                _description_
            i : float
                Index of the component in the reactor series.

            Returns
            -------
            _type_
                _description_
            """
            return m.rate[i, n] == 0

        # QFR desactivation
        @disjunct.Constraint()
        def neg_YPD_QFR_desact(disjunct):
            """_summary_

            Parameters
            ----------
            disjunct : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            return m.QFR[n] == 0

        @disjunct.Constraint()
        def neg_YPD_vol_desact(disjunct):
            '''
            Volume desactivation function for defining pyomo model
            args:
                disjunct: pyomo block with disjunct to include the constraint
                n: pyomo set with reactor index
            return: 
                return constraint
            '''
            return m.c[n] == 0

    # YR Disjuction block equation definition

    def build_recycle_equations(disjunct, n):
        """_summary_

        Parameters
        ----------
        disjunct : _type_
            _description_
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        _type_
            _description_
        """
        m = disjunct.model()

        # FR activation
        @disjunct.Constraint(m.I)
        def YRD_FR_act(disjunct, i):
            """_summary_

            Parameters
            ----------
            disjunct : _type_
                _description_
            i : float
                Index of the component in the reactor series.

            Returns
            -------
            _type_
                _description_
            """
            return m.FR[i, n] - m.R[i] == 0

        # QFR activation
        @disjunct.Constraint()
        def YRD_QFR_act(disjunct):
            """_summary_

            Parameters
            ----------
            disjunct : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            return m.QFR[n] - m.QR == 0

    def build_no_recycle_equations(disjunct, n):
        """_summary_

        Parameters
        ----------
        disjunct : _type_
            _description_
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        _type_
            _description_
        """
        m = disjunct.model()

        # FR desactivation
        @disjunct.Constraint(m.I)
        def neg_YRD_FR_desact(disjunct, i):
            """_summary_

            Parameters
            ----------
            disjunct : _type_
                _description_
            i : float
                Index of the component in the reactor series.
            
            Returns
            -------
            _type_
                _description_
            """
            return m.FR[i, n] == 0

        # QFR desactivation
        @disjunct.Constraint()
        def neg_YRD_QFR_desact(disjunct):
            """_summary_

            Parameters
            ----------
            disjunct : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            return m.QFR[n] == 0

    # Create disjunction blocks
    m.YR_is_recycle = Disjunct(m.N, rule=build_recycle_equations)
    m.YR_is_not_recycle = Disjunct(m.N, rule=build_no_recycle_equations)

    m.YP_is_cstr = Disjunct(m.N, rule=build_cstr_equations)
    m.YP_is_bypass = Disjunct(m.N, rule=build_bypass_equations)

    # Create disjunctions

    @m.Disjunction(m.N)
    def YP_is_cstr_or_bypass(m, n):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        list
            _description_
        """
        return [m.YP_is_cstr[n], m.YP_is_bypass[n]]

    @m.Disjunction(m.N)
    def YR_is_recycle_or_not(m, n):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        list
            _description_
        """
        return [m.YR_is_recycle[n], m.YR_is_not_recycle[n]]

    # Associate Boolean variables with with disjunctions
    for n in m.N:
        m.YP[n].associate_binary_var(m.YP_is_cstr[n].indicator_var)
        m.YR[n].associate_binary_var(m.YR_is_recycle[n].indicator_var)

    # Logic Constraints
    # Unit must be a CSTR to include a recycle

    def cstr_if_recycle_rule(m, n):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        _type_
            _description_
        """
        return m.YR[n].implies(m.YP[n])

    m.cstr_if_recycle = pyo.LogicalConstraint(m.N, rule=cstr_if_recycle_rule)

    # There is only one unreacted feed

    def one_unreacted_feed_rule(m):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.

        Returns
        -------
        Pyomo.LogicalConstraint
            _description_
        """
        return pyo.exactly(1, m.YF)

    m.one_unreacted_feed = pyo.LogicalConstraint(rule=one_unreacted_feed_rule)

    # There is only one recycle stream

    def one_recycle_rule(m):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.

        Returns
        -------
        Pyomo.LogicalConstraint
            _description_
        """
        return pyo.exactly(1, m.YR)

    m.one_recycle = pyo.LogicalConstraint(rule=one_recycle_rule)

    # Unit operation in n constraint

    def unit_in_n_rule(m, n):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo GDP model of the CSTR superstructure.
        n : int
            Index of the reactor in the reactor series.

        Returns
        -------
        Pyomo.LogicalConstraint
            _description_
        """
        if n == 1:
            return m.YP[n].equivalent_to(True)
        else:
            return m.YP[n].equivalent_to(pyo.lor(pyo.land(~m.YF[n2] for n2 in range(1, n)), m.YF[n]))

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

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize, doc="minimum total reactor volume")

    return m

if __name__ == "__main__":
    m = build_cstrs()
    pyo.TransformationFactory('core.logical_to_linear').apply_to(m)
    pyo.TransformationFactory('gdp.bigm').apply_to(m)
    pyo.SolverFactory('gams').solve(m, solver='baron', tee=True, add_options=['option optcr=1e-6;'])
    display(m)