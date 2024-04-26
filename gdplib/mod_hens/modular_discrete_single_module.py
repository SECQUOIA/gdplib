"""Heat integration case study.

This is example 1 of the Yee & Grossmann, 1990 paper "Simultaneous optimization
models for heat integration--II".
DOI: 10.1016/0098-1354(90)85010-8

This is a modification to support the incorporation of standardized exchanger
modules.

This is a further modification to support discretization of area simplifying
the nonlinear expressions, specialized to the case of allowing only a single
exchanger module type (size).

"""
from __future__ import division

from pyomo.environ import Binary, Constraint, RangeSet, Var
from pyomo.gdp import Disjunction

from . import common


def build_single_module(cafaro_approx, num_stages):
    """
    Builds a heat integration model tailored to handle single module types, with the option to utilize Cafaro's approximation for cost and efficiency calculations.

    Parameters
    ----------
    cafaro_approx : bool
        Specifies whether to use the Cafaro approximation in the model.
    num_stages : int
        The number of stages in the heat integration model.

    Returns
    -------
    Pyomo.ConcreteModel
        A Pyomo model configured with constraints and parameters specific to the requirements of using single module types in a discretized format.
    """
    return build_model(cafaro_approx, num_stages)


def build_model(use_cafaro_approximation, num_stages):
    """
    Extends a base heat integration model by incorporating a module configuration approach. It allows only single exchanger module types, optimizing the model for specific operational constraints and simplifying the nonlinear terms through discretization.

    Parameters
    ----------
    use_cafaro_approximation : bool
        Specifies whether to use the Cafaro approximation in the model.
    num_stages : int
        The number of stages in the heat integration model.

    Returns
    -------
    Pyomo.ConcreteModel
        An enhanced heat integration model that supports module configurations with discretized area considerations to simplify calculations and improve optimization performance.
    """
    m = common.build_model(use_cafaro_approximation, num_stages)

    # list of tuples (num_modules, module_size)
    configurations_list = []
    for size in m.module_sizes:
        configs = [(i + 1, size) for i in range(m.max_num_modules[size])]
        configurations_list += configs

    # Map of config indx: (# modules, module size)
    m.configurations_map = {(k + 1): v for k, v in enumerate(configurations_list)}

    m.module_index_set = RangeSet(len(configurations_list))

    m.module_config_active = Var(
        m.valid_matches,
        m.stages,
        m.module_index_set,
        doc="Binary for if which module configuration is active for a match.",
        domain=Binary,
        initialize=0,
    )

    @m.Param(m.module_index_set, doc="Area of each configuration")
    def module_area(m, indx):
        """
        Calculates the total area of a module configuration based on the number of modules and the size of each module.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model modified to support discretization of area simplifying the nonlinear expressions, specialized to the case of allowing only a single exchanger module type (size).
        indx : int
            Index of the module configuration in the model.

        Returns
        -------
        Pyomo.Parameter
            The total area of the configuration corresponding to the given index.
        """
        num_modules, size = m.configurations_map[indx]
        return num_modules * size

    @m.Param(
        m.valid_matches,
        m.module_index_set,
        doc="Area cost for each modular configuration.",
    )
    def modular_size_cost(m, hot, cold, indx):
        """
        Determines the cost associated with a specific modular configuration, taking into account the number of modules and their individual sizes.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model modified for the case of allowing only a single exchanger module type (size).
        hot : str
            The index for the hot stream involved in the heat exchanger.
        cold : str
            The index for the cold stream involved in the heat exchanger.
        indx : int
            Index of the module configuration in the model.

        Returns
        -------
        Pyomo.Parameter
            Cost associated with the specified modular configuration.
        """
        num_modules, size = m.configurations_map[indx]
        return num_modules * m.module_area_cost[hot, cold, size]

    @m.Param(
        m.valid_matches,
        m.module_index_set,
        doc="Fixed cost for each modular exchanger size.",
    )
    def modular_fixed_cost(m, hot, cold, indx):
        """
        Computes the fixed cost for a modular exchanger configuration, factoring in the number of modules and the set fixed cost per unit.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
           A Pyomo concrete model representing the heat exchange model modified for the case of allowing only a single exchanger module type (size).
        cold : str
            The index for the cold stream involved in the heat exchanger.
        indx : int
            Index of the module configuration in the model.

        Returns
        -------
        Pyomo.Parameter
            Fixed cost for the given modular exchanger configuration.
        """
        num_modules, size = m.configurations_map[indx]
        return num_modules * m.module_fixed_unit_cost

    m.LMTD_discretize = Var(
        m.hot_streams,
        m.cold_streams,
        m.stages,
        m.module_index_set,
        doc="Discretized log mean temperature difference",
        bounds=(0, 500),
        initialize=0,
    )

    for hot, cold, stg in m.valid_matches * m.stages:
        disj = m.exchanger_exists[hot, cold, stg]
        disj.choose_one_config = Constraint(
            expr=sum(
                m.module_config_active[hot, cold, stg, indx]
                for indx in m.module_index_set
            )
            == 1,
            doc="Enforce a single active configuration per exchanger per stage.",
        )

        disj.exchanger_area_cost = Constraint(
            expr=m.exchanger_area_cost[stg, hot, cold] * 1e-3
            == sum(
                m.modular_size_cost[hot, cold, indx]
                * 1e-3
                * m.module_config_active[hot, cold, stg, indx]
                for indx in m.module_index_set
            ),
            doc="Compute total area cost from active configurations.",
        )
        disj.exchanger_fixed_cost = Constraint(
            expr=m.exchanger_fixed_cost[stg, hot, cold]
            == sum(
                m.modular_fixed_cost[hot, cold, indx]
                * 1e-3
                * m.module_config_active[hot, cold, stg, indx]
                for indx in m.module_index_set
            ),
            doc="Sum fixed costs of active configurations for total investment.",
        )

        disj.discretize_area = Constraint(
            expr=m.exchanger_area[stg, hot, cold]
            == sum(
                m.module_area[indx] * m.module_config_active[hot, cold, stg, indx]
                for indx in m.module_index_set
            ),
            doc="Match exchanger area with sum of active configuration areas.",
        )

        disj.discretized_LMTD = Constraint(
            expr=m.LMTD[hot, cold, stg]
            == sum(
                m.LMTD_discretize[hot, cold, stg, indx] for indx in m.module_index_set
            ),
            doc="Aggregate LMTD from active configurations for thermal modeling.",
        )

        @disj.Constraint(m.module_index_set)
        def discretized_LMTD_LB(disj, indx):
            """
            Sets the lower bound on the discretized Log Mean Temperature Difference (LMTD) for each module configuration.

            Parameters
            ----------
            disj : Pyomo.Disjunct
                The disjunct object representing a specific module configuration.
            indx : int
                Index of the module configuration in the model.

            Returns
            -------
            Pyomo.Constraint
                A constraint ensuring that the discretized LMTD respects the specified lower bound for active configurations.
            """
            return (
                m.LMTD[hot, cold, stg].lb * m.module_config_active[hot, cold, stg, indx]
            ) <= m.LMTD_discretize[hot, cold, stg, indx]

        @disj.Constraint(m.module_index_set)
        def discretized_LMTD_UB(disj, indx):
            """
            Sets the upper bound on the discretized Log Mean Temperature Difference (LMTD) for each module configuration.

            Parameters
            ----------
            disj : Pyomo.Disjunct
                The disjunct object representing a specific module configuration.
            indx : int
                Index of the module configuration in the model.

            Returns
            -------
            Pyomo.Constraint
                A constraint ensuring that the discretized LMTD does not exceed the specified upper bound for active configurations.
            """
            return m.LMTD_discretize[hot, cold, stg, indx] <= (
                m.LMTD[hot, cold, stg].ub * m.module_config_active[hot, cold, stg, indx]
            )

        disj.exchanger_required_area = Constraint(
            expr=m.U[hot, cold]
            * sum(
                m.module_area[indx] * m.LMTD_discretize[hot, cold, stg, indx]
                for indx in m.module_index_set
            )
            >= m.heat_exchanged[hot, cold, stg],
            doc="Ensures sufficient heat transfer capacity for required heat exchange.",
        )

    @m.Disjunct(m.module_sizes)
    def module_type(disj, size):
        """
        Disjunct for selecting a specific module size in the heat exchange model. This disjunct applies constraints to enforce that only the selected module size is active within any given configuration across all stages and matches.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct object associated with a specific module size.
        size : int
            The specific size of the module being considered in this disjunct.

        Returns
        -------
        Pyomo.Disjunct
            A Pyomo Disjunct object that contains constraints to limit the module configuration to a single size throughout the model.
        """

        @disj.Constraint(m.valid_matches, m.stages, m.module_index_set)
        def no_other_module_types(_, hot, cold, stg, indx):
            """
            Ensures only modules of the selected size are active, deactivating other sizes.

            Parameters
            ----------
            _ : Pyomo.ConcreteModel
                The Pyomo model instance, not used directly in the function.
            hot : str
                The index for the hot stream involved in the heat exchanger.
            cold : str
                The index for the cold stream involved in the heat exchanger.
            stg : int
                The index for the stage involved in the heat exchanger.
            indx : int
                Index of the module configuration in the model.

            Returns
            -------
            Pyomo.Constraint
                A constraint expression that ensures only modules of the specified size are active, effectively disabling other module sizes for the current configuration.
            """
            # num_modules, size = configurations_map[indx]
            if m.configurations_map[indx][1] != size:
                return m.module_config_active[hot, cold, stg, indx] == 0
            else:
                return Constraint.NoConstraint

        # disj.no_other_module_types = Constraint(
        #     expr=sum(
        #         m.module_config_active[hot, cold, stg, indx]
        #         for indx in m.module_index_set
        #         if m.configurations_map[indx][1] != size
        #     )
        #     == 0,
        #     doc="Deactivates non-selected module sizes.",
        # )

    m.select_one_module_type = Disjunction(
        expr=[m.module_type[area] for area in m.module_sizes],
        doc="Selects exactly one module size for use across all configurations.",
    )

    return m
