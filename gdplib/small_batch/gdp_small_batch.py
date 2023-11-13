"""
gdp_small_batch.py
The gdp_small_batch.py module contains the GDP model for the small batch problem based on the Kocis and Grossmann (1988) paper.
The problem is based on the Example 4 of the paper.

References:
    - Kocis, G. R.; Grossmann, I. E. Global Optimization of Nonconvex Mixed-Integer Nonlinear Programming (MINLP) Problems in Process Synthesis. Ind. Eng. Chem. Res. 1988, 27 (8), 1407–1421. 
"""
import os

import pyomo.environ as pe
from pyomo.core.base.misc import display
from pyomo.core.plugins.transform.logical_to_linear import (
    update_boolean_vars_from_binary,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.base.solvers import SolverFactory


def build_small_batch():
    """
    The function build the GDP model for the small batch problem.

    References:
    - Kocis, G. R.; Grossmann, I. E. Global Optimization of Nonconvex Mixed-Integer Nonlinear Programming (MINLP) Problems in Process Synthesis. Ind. Eng. Chem. Res. 1988, 27 (8), 1407–1421.

    Args:
        None

    Returns:
        m (pyomo.ConcreteModel): The GDP model for the small batch problem is created.
    """
    NK = 3

    # Model
    m = pe.ConcreteModel()

    # Sets
    m.i = pe.Set(
        initialize=['a', 'b'], doc='Set of products'
    )  # Set of products, i = a, b
    m.j = pe.Set(
        initialize=['mixer', 'reactor', 'centrifuge']
    )  # Set of stages, j = mixer, reactor, centrifuge
    m.k = pe.RangeSet(NK)  # Set of potential number of parallel units, k = 1, 2, 3

    # Parameters and Scalars

    m.h = pe.Param(
        initialize=6000, doc='Horizon time [hr]'
    )  # Horizon time  (available time) [hr]
    m.vlow = pe.Param(
        initialize=250, doc='Lower bound for size of batch unit [L]'
    )  # Lower bound for size of batch unit [L]
    m.vupp = pe.Param(
        initialize=2500, doc='Upper bound for size of batch unit [L]'
    )  # Upper bound for size of batch unit [L]

    # Demand of product i
    m.q = pe.Param(
        m.i,
        initialize={'a': 200000, 'b': 150000},
        doc='Production rate of the product [kg]',
    )
    # Cost coefficient for batch units
    m.alpha = pe.Param(
        m.j,
        initialize={'mixer': 250, 'reactor': 500, 'centrifuge': 340},
        doc='Cost coefficient for batch units [$/L^beta*No. of units]]',
    )
    # Cost exponent for batch units
    m.beta = pe.Param(
        m.j,
        initialize={'mixer': 0.6, 'reactor': 0.6, 'centrifuge': 0.6},
        doc='Cost exponent for batch units',
    )

    def coeff_init(m, k):
        """
        Coefficient for number of parallel units.

        Args:
            m (pyomo.ConcreteModel): small batch GDP model
            k (int): number of parallel units

        Returns:
            Coefficient for number of parallel units.
        """
        return pe.log(k)

    # Represent number of parallel units
    m.coeff = pe.Param(
        m.k, initialize=coeff_init, doc='Coefficient for number of parallel units'
    )

    s_init = {
        ('a', 'mixer'): 2,
        ('a', 'reactor'): 3,
        ('a', 'centrifuge'): 4,
        ('b', 'mixer'): 4,
        ('b', 'reactor'): 6,
        ('b', 'centrifuge'): 3,
    }

    # Size factor for product i in stage j [kg/L]
    m.s = pe.Param(
        m.i, m.j, initialize=s_init, doc='Size factor for product i in stage j [kg/L]'
    )

    t_init = {
        ('a', 'mixer'): 8,
        ('a', 'reactor'): 20,
        ('a', 'centrifuge'): 4,
        ('b', 'mixer'): 10,
        ('b', 'reactor'): 12,
        ('b', 'centrifuge'): 3,
    }

    # Processing time of product i in batch j [hr]
    m.t = pe.Param(
        m.i, m.j, initialize=t_init, doc='Processing time of product i in batch j [hr]'
    )

    # Variables
    m.Y = pe.BooleanVar(m.k, m.j, doc='Stage existence')  # Stage existence
    m.coeffval = pe.Var(
        m.k,
        m.j,
        within=pe.NonNegativeReals,
        bounds=(0, pe.log(NK)),
        doc='Activation of Coefficient',
    )  # Activation of coeff
    m.v = pe.Var(
        m.j,
        within=pe.NonNegativeReals,
        bounds=(pe.log(m.vlow), pe.log(m.vupp)),
        doc='Colume of stage j [L]',
    )  # Volume of stage j [L]
    m.b = pe.Var(
        m.i, within=pe.NonNegativeReals, doc='Batch size of product i [L]'
    )  # Batch size of product i [L]
    m.tl = pe.Var(
        m.i, within=pe.NonNegativeReals, doc='Cycle time of product i [hr]'
    )  # Cycle time of product i [hr]
    # Number of units in parallel stage j
    m.n = pe.Var(
        m.j, within=pe.NonNegativeReals, doc='Number of units in parallel stage j'
    )

    # Constraints

    # Volume requirement in stage j
    @m.Constraint(m.i, m.j)
    def vol(m, i, j):
        """
        Volume Requirement for Stage j.
        Equation:
            v_j \geq log(s_ij) + b_i for i = a, b and j = mixer, reactor, centrifuge

        Args:
            m (pyomo.ConcreteModel): small batch GDP model
            i (str): product
            j (str): stage

        Returns:
            Algebraic Constraint
        """
        return m.v[j] >= pe.log(m.s[i, j]) + m.b[i]

    # Cycle time for each product i
    @m.Constraint(m.i, m.j)
    def cycle(m, i, j):
        """
        Cycle time for each product i.
        Equation:
            n_j + tl_i \geq log(t_ij) for i = a, b and j = mixer, reactor, centrifuge

        Args:
            m (pyomo.ConcreteModel): small batch GDP model
            i (str): product
            j (str): stage

        Returns:
            Algebraic Constraint
        """
        return m.n[j] + m.tl[i] >= pe.log(m.t[i, j])

    # Constraint for production time
    @m.Constraint()
    def time(m):
        """
        Production time constraint.
        Equation:
            \sum_{i \in I} q_i * \exp(tl_i - b_i) \leq h

        Args:
            m (pyomo.ConcreteModel): small batch GDP model

        Returns:
            Algebraic Constraint
        """
        return sum(m.q[i] * pe.exp(m.tl[i] - m.b[i]) for i in m.i) <= m.h

    # Relating number of units to 0-1 variables
    @m.Constraint(m.j)
    def units(m, j):
        """
        Relating number of units to 0-1 variables.
        Equation:
            n_j = \sum_{k \in K} coeffval_{k,j} for j = mixer, reactor, centrifuge

        Args:
            m (pyomo.ConcreteModel): small batch GDP model
            j (str): stage
            k (int): number of parallel units

        Returns:
            Algebraic Constraint
        """
        return m.n[j] == sum(m.coeffval[k, j] for k in m.k)

    # Only one choice for parallel units is feasible
    @m.LogicalConstraint(m.j)
    def lim(m, j):
        """
        Only one choice for parallel units is feasible.
        Equation:
            \sum_{k \in K} Y_{k,j} = 1 for j = mixer, reactor, centrifuge

        Args:
            m (pyomo.ConcreteModel): small batch GDP model
            j (str): stage

        Returns:
            Logical Constraint
        """
        return pe.exactly(1, m.Y[1, j], m.Y[2, j], m.Y[3, j])

    # _______ Disjunction_________

    def build_existence_equations(disjunct, k, j):
        """
        Build the Logic Proposition (euqations) for the existence of the stage.

        Args:
            disjunct (pyomo.gdp.Disjunct): Disjunct block
            k (int): number of parallel units
            j (str): stage

        Returns:
            None, the proposition is built inside the function
        """
        m = disjunct.model()

        # Coeffval activation
        @disjunct.Constraint()
        def coeffval_act(disjunct):
            """
            Coeffval activation.
            m.coeffval[k,j] = m.coeff[k] = log(k)

            Args:
                disjunct (pyomo.gdp.Disjunct): Disjunct block

            Returns:
               Logical Constraint
            """
            return m.coeffval[k, j] == m.coeff[k]

    def build_not_existence_equations(disjunct, k, j):
        """
        Build the Logic Proposition (euqations) for the unexistence of the stage.

        Args:
            disjunct (pyomo.gdp.Disjunct): Disjunct block
            k (int): number of parallel units
            j (str): stage

        Returns:
            None, the proposition is built inside the function.
        """
        m = disjunct.model()

        # Coeffval deactivation
        @disjunct.Constraint()
        def coeffval_deact(disjunct):
            """
            Coeffval deactivation.
            m.coeffval[k,j] = 0

            Args:
                disjunct (pyomo.gdp.Disjunct): Disjunct block

            Returns:
                Logical Constraint
            """
            return m.coeffval[k, j] == 0

    # Create disjunction block
    m.Y_exists = Disjunct(m.k, m.j, rule=build_existence_equations)
    m.Y_not_exists = Disjunct(m.k, m.j, rule=build_not_existence_equations)

    # Create disjunction

    @m.Disjunction(m.k, m.j)
    def Y_exists_or_not(m, k, j):
        """
        Build the Logical Disjunctions of the GDP model for the small batch problem.

        Args:
            m (pyomo.ConcreteModel): small batch GDP model
            k (int): number of parallel units
            j (str): stage

        Returns:
            Y_exists_or_not (list): List of disjuncts
        """
        return [m.Y_exists[k, j], m.Y_not_exists[k, j]]

    # Associate Boolean variables with with disjunction
    for k in m.k:
        for j in m.j:
            m.Y[k, j].associate_binary_var(m.Y_exists[k, j].indicator_var)

    # ____________________________

    # Objective
    def obj_rule(m):
        """
        Objective: mininimize the investment cost [$].
        Equation:
            min z = sum(alpha[j] * exp(n[j] + beta[j]*v[j])) for j = mixer, reactor, centrifuge

        Args:
            m (pyomo.ConcreteModel): small batch GDP model

        Returns:
            Objective function (pyomo.Objective): Objective function to minimize the investment cost [$].
        """
        return sum(m.alpha[j] * (pe.exp(m.n[j] + m.beta[j] * m.v[j])) for j in m.j)

    m.obj = pe.Objective(rule=obj_rule, sense=pe.minimize)

    return m


def external_ref(m, x, logic_expr=None):
    """
    Add the external variables to the GDP optimization problem.

    Args:
        m (pyomo.ConcreteModel): GDP optimization model
        x (list): External variables
        logic_expr (list, optional): Logic expressions to be used in the disjunctive constraints

    Returns:
        m (pyomo.ConcreteModel): GDP optimization model with the external variables
    """
    ext_var = {}
    p = 0
    for j in m.j:
        ext_var[j] = x[p]
        p = p + 1

    for k in m.k:
        for j in m.j:
            if k == ext_var[j]:
                m.Y[k, j].fix(True)
                # m.Y_exists[k, j].indicator_var.fix(
                #     True
                # )  # Is this necessary?: m.Y_exists[k, j].indicator_var.fix(True).
                # m.Y_not_exists[k, j].indicator_var.fix(
                #     False
                # )  # Is this necessary?: m.Y_not_exists[k, j].indicator_var.fix(True),
            else:
                m.Y[k, j].fix(False)
                # m.Y_exists[k, j].indicator_var.fix(
                #     False
                # )  # Is this necessary?: m.Y_exists[k, j].indicator_var.fix(True),
                # m.Y_not_exists[k, j].indicator_var.fix(
                #     True
                # )  # Is this necessary?: m.Y_not_exists[k, j].indicator_var.fix(True),

    pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    pe.TransformationFactory('gdp.fix_disjuncts').apply_to(m)
    pe.TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
        m, tmp=False, ignore_infeasible=True
    )

    return m


def solve_with_minlp(m, transformation='bigm', minlp='baron', timelimit=10):
    """
    Solve the GDP optimization problem with a MINLP solver.
    The function applies the big-M Reformulation on the GDP and solve the MINLP problem with BARON.

    Args:
        m (pyomo.ConcreteModel): GDP optimization model
        transformation (str, optional): Reformulation applied to the GDP.
        minlp (str, optional): MINLP solver.
        timelimit (float, optional): Time limit for the MINLP solver.

    Returns:
        m (pyomo.ConcreteModel): GDP optimization model with the solution.
    """
    # Transformation step
    pe.TransformationFactory('core.logical_to_linear').apply_to(m)
    transformation_string = 'gdp.' + transformation
    pe.TransformationFactory(transformation_string).apply_to(m)

    # Solution step
    dir_path = os.path.dirname(os.path.abspath(__file__))
    gams_path = os.path.join(dir_path, "gamsfiles/")
    if not (os.path.exists(gams_path)):
        print(
            'Directory for automatically generated files '
            + gams_path
            + ' does not exist. We will create it'
        )
        os.makedirs(gams_path)

    solvername = 'gams'
    opt = SolverFactory(solvername, solver=minlp)
    m.results = opt.solve(
        m,
        tee=True,
        # Uncomment the following lines if you want to save GAMS models
        # keepfiles=True,
        # tmpdir=gams_path,
        # symbolic_solver_labels=True,
        add_options=[
            'option reslim = ' + str(timelimit) + ';'
            'option optcr = 0.0;'
            # Uncomment the following lines to setup IIS computation of BARON through option file
            # 'GAMS_MODEL.optfile = 1;'
            # '\n'
            # '$onecho > baron.opt \n'
            # 'CompIIS 1 \n'
            # '$offecho'
            # 'display(execError);'
        ],
    )
    update_boolean_vars_from_binary(m)
    return m


if __name__ == "__main__":
    m = build_small_batch()
    m_solved = solve_with_minlp(m, transformation='bigm', minlp='baron', timelimit=120)

    # EXTERNAL REF TEST (this thest can be deleted)
    newmodel = external_ref(m, [1, 2, 3], logic_expr=None)
    # print('External Ref Test')
    # print('Y[1, mixer] = ', newmodel.Y[1, 'mixer'].value)
    # print('Y_exists[1, mixer] = ', newmodel.Y_exists[1, 'mixer'].indicator_var.value)
    # print('Y_not_exists[1, mixer] = ', newmodel.Y_not_exists[1, 'mixer'].indicator_var.value)
    # print('Y[2, mixer] = ', newmodel.Y[2, 'mixer'].value)
    # print('Y_exists[2, mixer] = ', newmodel.Y_exists[2, 'mixer'].indicator_var.value)
    # print('Y_not_exists[2, mixer] = ', newmodel.Y_not_exists[2, 'mixer'].indicator_var.value)
