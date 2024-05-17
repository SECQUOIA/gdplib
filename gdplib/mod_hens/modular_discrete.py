"""Heat integration case study.

This is example 1 of the Yee & Grossmann, 1990 paper "Simultaneous optimization
models for heat integration--II".
DOI: 10.1016/0098-1354(90)85010-8

This is a modification to support the incorporation of standardized exchanger
modules.

This is a further modification to support discretization of area simplifying
the nonlinear expressions.

"""

from __future__ import division

from pyomo.environ import Binary, Constraint, log, Set, Var
from pyomo.gdp import Disjunct, Disjunction

from . import common


def build_require_modular(cafaro_approx, num_stages):
    """
    Builds a heat integration model requiring all exchangers to use modular configurations.

    Parameters
    ----------
    cafaro_approx : bool
        Specifies whether to use the Cafaro approximation in the model.
    num_stages : int
        The number of stages in the heat integration model.

    Returns
    -------
    Pyomo.ConcreteModel
        A Pyomo model configured to use only modular heat exchanger configurations.
    """
    m = build_model(cafaro_approx, num_stages)
    # Require modular
    # Enforce modular configuration for all valid matches and stages
    for hot, cold, stg in m.valid_matches * m.stages:
        disj = m.exchanger_exists[hot, cold, stg]
        disj.modular.indicator_var.fix(True)
        disj.conventional.deactivate()

    # Optimize modular configurations based on cost
    for hot, cold in m.valid_matches:
        lowest_price = float('inf')
        # Determine the least costly configuration for each size
        for size in sorted(m.possible_sizes, reverse=True):
            current_size_cost = (
                m.modular_size_cost[hot, cold, size]
                + m.modular_fixed_cost[hot, cold, size]
            )
            if current_size_cost > lowest_price:
                # Deactivate configurations that are not the least costly
                for stg in m.stages:
                    m.module_size_active[hot, cold, stg, size].fix(0)
            else:
                lowest_price = current_size_cost

    return m


def build_modular_option(cafaro_approx, num_stages):
    """
    Constructs a heat integration model with the option for using modular configurations based on cost optimization.

    Parameters
    ----------
    cafaro_approx : bool
        Specifies whether to use the Cafaro approximation in the model.
    num_stages : int
        The number of stages in the heat integration model.

    Returns
    -------
    Pyomo.ConcreteModel
        A Pyomo model that considers modular exchanger options based on cost efficiencies.
    """
    m = build_model(cafaro_approx, num_stages)

    # Optimize for the least cost configuration across all stages and matche
    for hot, cold in m.valid_matches:
        lowest_price = float('inf')
        for size in sorted(m.possible_sizes, reverse=True):
            current_size_cost = (
                m.modular_size_cost[hot, cold, size]
                + m.modular_fixed_cost[hot, cold, size]
            )
            if current_size_cost > lowest_price:
                for stg in m.stages:
                    m.module_size_active[hot, cold, stg, size].fix(0)
            else:
                lowest_price = current_size_cost

    return m


def build_model(use_cafaro_approximation, num_stages):
    """
    Initializes a base Pyomo model for heat integration using the common building blocks, with additional configuration for modular sizing.

    Parameters
    ----------
    use_cafaro_approximation : bool
        Specifies whether to use the Cafaro approximation in the model.
    num_stages : int
        The number of stages in the heat integration model.

    Returns
    -------
    Pyomo.ConcreteModel
        A Pyomo model configured for heat integration with modular exchanger options.
    """
    m = common.build_model(use_cafaro_approximation, num_stages)

    m.possible_sizes = Set(
        initialize=[10 * (i + 1) for i in range(50)],
        doc="Set of possible module sizes, ranging from 10 to 500 in increments of 10.",
    )
    m.module_size_active = Var(
        m.valid_matches,
        m.stages,
        m.possible_sizes,
        doc="Total area of modular exchangers for each match.",
        domain=Binary,
        initialize=0,
    )

    num_modules_required = {}
    for size in m.possible_sizes:
        # For each possible size, calculate the number of exchangers of each
        # area required to satisfy that size.
        # Use progressively smaller exchangers to satisfy the size
        remaining_size = size
        module_sizes = sorted(m.module_sizes, reverse=True)
        for area in module_sizes:
            num_modules_required[size, area] = remaining_size // area
            remaining_size = remaining_size % area

    @m.Param(
        m.valid_matches,
        m.possible_sizes,
        m.module_sizes,
        doc="Number of exchangers of each area required to "
        "yield a certain total size.",
    )
    def modular_num_exchangers(m, hot, cold, size, area):
        """
        Returns the number of exchangers required for a given total module size and area.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo model configured for heat integration with modular exchanger options.
        hot : str
            The index for the hot stream involved in the heat exchanger.
        cold : str
            The index for the cold stream involved in the heat exchanger.
        size : int
            Module size under consideration.
        area : float
            The modular area size of the heat exchanger.

        Returns
        -------
        Pyomo.Parameter
            Number of modules of the specified area required to achieve the total size.
        """
        return num_modules_required[size, area]

    @m.Param(
        m.valid_matches,
        m.possible_sizes,
        doc="Area cost for each modular exchanger size.",
    )
    def modular_size_cost(m, hot, cold, size):
        """
        Returns the total area cost for a specified module size by summing costs of all required module areas.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo model configured for heat integration with modular exchanger options.
        hot : str
            The index for the hot stream involved in the heat exchanger.
        cold : str
            The index for the cold stream involved in the heat exchanger.
        size : int
            Module size under consideration.

        Returns
        -------
        Pyomo.Parameter
            Total area cost for the specified module size.
        """
        return sum(
            m.modular_num_exchangers[hot, cold, size, area]
            * m.module_area_cost[hot, cold, area]
            for area in m.module_sizes
        )

    @m.Param(
        m.valid_matches,
        m.possible_sizes,
        doc="Fixed cost for each modular exchanger size.",
    )
    def modular_fixed_cost(m, hot, cold, size):
        """
        Returns the total fixed cost associated with a specific modular exchanger size.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo model configured for heat integration with modular exchanger options.
        hot : str
            The index for the hot stream involved in the heat exchanger.
        cold : str
            The index for the cold stream involved in the heat exchanger.
        size : int
            Module size under consideration.

        Returns
        -------
        Pyomo.Parameter
            Total fixed cost for the specified module size.
        """
        return sum(
            m.modular_num_exchangers[hot, cold, size, area] * m.module_fixed_unit_cost
            for area in m.module_sizes
        )

    m.LMTD_discretize = Var(
        m.hot_streams,
        m.cold_streams,
        m.stages,
        m.possible_sizes,
        doc="Discretized log mean temperature difference",
        bounds=(0, 500),
        initialize=0,
    )

    for hot, cold, stg in m.valid_matches * m.stages:
        disj = m.exchanger_exists[hot, cold, stg]
        disj.conventional = Disjunct()
        if not use_cafaro_approximation:
            disj.conventional.exchanger_area_cost = Constraint(
                expr=m.exchanger_area_cost[stg, hot, cold] * 1e-3
                >= m.exchanger_area_cost_factor[hot, cold]
                * 1e-3
                * m.exchanger_area[stg, hot, cold] ** m.area_cost_exponent,
                doc="Ensures area cost meets the standard cost scaling.",
            )
        else:
            disj.conventional.exchanger_area_cost = Constraint(
                expr=m.exchanger_area_cost[stg, hot, cold] * 1e-3
                >= m.exchanger_area_cost_factor[hot, cold]
                * 1e-3
                * m.cafaro_k
                * log(m.cafaro_b * m.exchanger_area[stg, hot, cold] + 1),
                doc="Ensures area cost meets the Cafaro approximation.",
            )
        m.BigM[disj.conventional.exchanger_area_cost] = 100

        disj.conventional.exchanger_fixed_cost = Constraint(
            expr=m.exchanger_fixed_cost[stg, hot, cold]
            == m.exchanger_fixed_unit_cost[hot, cold],
            doc="Sets the fixed cost for conventional exchangers.",
        )

        @disj.conventional.Constraint(m.possible_sizes)
        def no_modules(_, size):
            """
            Ensures that no modules are active in the conventional configuration.

            Parameters
            ----------
            _ : Pyomo.ConcreteModel
                The Pyomo model instance, not used directly in the function.
            size : int
                Module size under consideration.

            Returns
            -------
            Pyomo.Constraint
                A constraint that forces the module size active variables to zero, ensuring no modular units are mistakenly considered in conventional configurations.
            """
            return m.module_size_active[hot, cold, stg, size] == 0

        # Area requirement
        disj.conventional.exchanger_required_area = Constraint(
            expr=m.exchanger_area[stg, hot, cold]
            * m.U[hot, cold]
            * m.LMTD[hot, cold, stg]
            >= m.heat_exchanged[hot, cold, stg],
            doc="Calculates the required area based on heat exchanged and LMTD.",
        )
        m.BigM[disj.conventional.exchanger_required_area] = 5000

        disj.modular = Disjunct()
        disj.modular.choose_one_config = Constraint(
            expr=sum(
                m.module_size_active[hot, cold, stg, size] for size in m.possible_sizes
            )
            == 1,
            doc="Only one module size can be active.",
        )
        disj.modular.exchanger_area_cost = Constraint(
            expr=m.exchanger_area_cost[stg, hot, cold] * 1e-3
            == sum(
                m.modular_size_cost[hot, cold, size]
                * 1e-3
                * m.module_size_active[hot, cold, stg, size]
                for size in m.possible_sizes
            ),
            doc="Area cost for modular exchangers.",
        )
        disj.modular.exchanger_fixed_cost = Constraint(
            expr=m.exchanger_fixed_cost[stg, hot, cold]
            == sum(
                m.modular_fixed_cost[hot, cold, size]
                * 1e-3
                * m.module_size_active[hot, cold, stg, size]
                for size in m.possible_sizes
            ),
            doc="Fixed cost for modular exchangers.",
        )

        disj.modular.discretize_area = Constraint(
            expr=m.exchanger_area[stg, hot, cold]
            == sum(
                area * m.module_size_active[hot, cold, stg, area]
                for area in m.possible_sizes
            ),
            doc="Total area of modular exchangers for each match.",
        )

        disj.modular.discretized_LMTD = Constraint(
            expr=m.LMTD[hot, cold, stg]
            == sum(
                m.LMTD_discretize[hot, cold, stg, size] for size in m.possible_sizes
            ),
            doc="Discretized LMTD for each match.",
        )

        @disj.modular.Constraint(m.possible_sizes)
        def discretized_LMTD_LB(disj, size):
            """
            Sets the lower bound on the discretized Log Mean Temperature Difference (LMTD) for each possible size.

            Parameters
            ----------
            disj : Pyomo.Disjunct
                The disjunct object representing a specific module size.
            size : int
                Module size under consideration.

            Returns
            -------
            Pyomo.Constraint
                A constraint that sets the lower limit for the discretized LMTD based on the module size active in the configuration.
            """
            return (
                m.LMTD[hot, cold, stg].lb * m.module_size_active[hot, cold, stg, size]
            ) <= m.LMTD_discretize[hot, cold, stg, size]

        @disj.modular.Constraint(m.possible_sizes)
        def discretized_LMTD_UB(disj, size):
            """
            Sets the upper bound on the discretized Log Mean Temperature Difference (LMTD) for each possible size.

            Parameters
            ----------
            disj : Pyomo.Disjunct
                The disjunct object representing a specific module size.
            size : int
                Module size under consideration.

            Returns
            -------
            Pyomo.Constraint
                A constraint that sets the upper limit for the discretized LMTD based on the module size active in the configuration.
            """
            return m.LMTD_discretize[hot, cold, stg, size] <= (
                m.LMTD[hot, cold, stg].ub * m.module_size_active[hot, cold, stg, size]
            )

        disj.modular.exchanger_required_area = Constraint(
            expr=m.U[hot, cold]
            * sum(
                area * m.LMTD_discretize[hot, cold, stg, area]
                for area in m.possible_sizes
            )
            >= m.heat_exchanged[hot, cold, stg],
            doc="Calculates the required area based on heat exchanged and LMTD.",
        )

        disj.modular_or_not = Disjunction(
            expr=[disj.modular, disj.conventional],
            doc="Disjunction between modular and conventional configurations.",
        )

    return m
