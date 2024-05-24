"""Heat integration case study.

This is example 1 of the Yee & Grossmann, 1990 paper "Simultaneous optimization
models for heat integration--II".
DOI: 10.1016/0098-1354(90)85010-8

This is a modification to support the incorporation of standardized exchanger
modules using integer variables for module selection.

"""

from __future__ import division

from pyomo.environ import Constraint, Integers, log, Var
from pyomo.gdp import Disjunct, Disjunction

from gdplib.mod_hens import common


def build_single_module(cafaro_approx, num_stages):
    """
    Constructs a Pyomo model configured to exclusively use modular heat exchangers, forcing selection of a single module type per stage and stream match. This configuration utilizes the Cafaro approximation if specified.

    Parameters
    ----------
    cafaro_approx : bool
        Specifies whether to use the Cafaro approximation in the model.
    num_stages : int
        The number of stages in the heat integration model.

    Returns
    -------
    Pyomo.ConcreteModel
        A Pyomo ConcreteModel optimized with constraints for modular heat exchanger configurations, fixed to a single module type per valid match.
    """
    m = build_model(cafaro_approx, num_stages)
    # Require modular
    for hot, cold, stg in m.valid_matches * m.stages:
        disj = m.exchanger_exists[hot, cold, stg]
        disj.modular.indicator_var.fix(True)
        disj.conventional.deactivate()

    # Must choose only one type of module
    @m.Disjunct(m.module_sizes)
    def module_type(disj, size):
        """
        Disjunct for selection of one module type.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            _description_
        size : int
            Module size under consideration.
        """
        disj.no_other_module_types = Constraint(
            expr=sum(
                m.num_modules[hot, cold, stage, area]
                for hot, cold in m.valid_matches
                for stage in m.stages
                for area in m.module_sizes
                if area != size
            )
            == 0,
            doc="Ensures no modules of other sizes are active when this size is selected.",
        )

    m.select_one_module_type = Disjunction(
        expr=[m.module_type[area] for area in m.module_sizes],
        doc="Select one module type",
    )

    return m


def build_require_modular(cafaro_approx, num_stages):
    """
    Builds a Pyomo model that requires the use of modular configurations for all heat exchangers within the model. This setup deactivates any conventional exchanger configurations.

    Parameters
    ----------
    cafaro_approx : bool
        Specifies whether to use the Cafaro approximation in the model.
    num_stages : int
        The number of stages in the heat integration model.

    Returns
    -------
    Pyomo.ConcreteModel
        A Pyomo model configured to require modular heat exchangers throughout the network.
    """
    m = build_model(cafaro_approx, num_stages)
    # Require modular
    for hot, cold, stg in m.valid_matches * m.stages:
        m.exchanger_exists[hot, cold, stg].conventional.deactivate()

    return m


def build_modular_option(cafaro_approx, num_stages):
    """
    Builds a Pyomo model that can optionally use modular heat exchangers based on configuration decisions within the model. This function initializes a model using the Cafaro approximation as specified.

    Parameters
    ----------
    cafaro_approx : bool
        Specifies whether to use the Cafaro approximation in the model.
    num_stages : int
        The number of stages in the heat integration model.

    Returns
    -------
    Pyomo.ConcreteModel
        Returns a Pyomo model with the flexibility to choose modular heat exchanger configurations based on optimization results.
    """
    return build_model(cafaro_approx, num_stages)


def build_model(use_cafaro_approximation, num_stages):
    """
    Base function for constructing a heat exchange network model with optional use of Cafaro approximation and integration of modular heat exchangers represented by integer variables.

    Parameters
    ----------
    use_cafaro_approximation : bool
        Specifies whether to use the Cafaro approximation in the model.
    num_stages : int
        The number of stages in the heat integration model.

    Returns
    -------
    Pyomo.ConcreteModel
        The initialized Pyomo model including both conventional and modular heat exchanger options.
    """
    m = common.build_model(use_cafaro_approximation, num_stages)

    m.num_modules = Var(
        m.valid_matches,
        m.stages,
        m.module_sizes,
        doc="The number of modules of each size at each exchanger.",
        domain=Integers,
        bounds=(0, 100),
        initialize=0,
    )
    # improve quality of bounds
    for size in m.module_sizes:
        for var in m.num_modules[:, :, :, size]:
            var.setub(m.max_num_modules[size])

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
                doc="Applies Cafaro's logarithmic cost scaling to area cost.",
            )
        m.BigM[disj.conventional.exchanger_area_cost] = 100

        disj.conventional.exchanger_fixed_cost = Constraint(
            expr=m.exchanger_fixed_cost[stg, hot, cold]
            == m.exchanger_fixed_unit_cost[hot, cold],
            doc="Sets fixed cost for the exchanger based on unit costs.",
        )

        @disj.conventional.Constraint(m.module_sizes)
        def no_modules(_, area):
            """
            Ensures that no modules are active in the conventional configuration.

            Parameters
            ----------
            _ : Pyomo.ConcreteModel
                The Pyomo model instance, not used directly in the function
            area : float
                The modular area size of the heat exchanger

            Returns
            -------
            Pyomo.Constraint
                A constraint that forces the module size active variables to zero, ensuring no modular units are mistakenly considered in conventional configurations.
            """
            return m.num_modules[hot, cold, stg, area] == 0

        disj.modular = Disjunct()
        disj.modular.exchanger_area_cost = Constraint(
            expr=m.exchanger_area_cost[stg, hot, cold] * 1e-3
            == sum(
                m.module_area_cost[hot, cold, area]
                * m.num_modules[hot, cold, stg, area]
                for area in m.module_sizes
            )
            * 1e-3,
            doc="Area cost for modular exchanger",
        )
        disj.modular.exchanger_fixed_cost = Constraint(
            expr=m.exchanger_fixed_cost[stg, hot, cold]
            == m.module_fixed_unit_cost
            * sum(m.num_modules[hot, cold, stg, area] for area in m.module_sizes),
            doc="Fixed cost for modular exchanger",
        )
        disj.modular.exchanger_area = Constraint(
            expr=m.exchanger_area[stg, hot, cold]
            == sum(
                area * m.num_modules[hot, cold, stg, area] for area in m.module_sizes
            ),
            doc="Area for modular exchanger",
        )
        disj.modular_or_not = Disjunction(
            expr=[disj.modular, disj.conventional],
            doc="Module or conventional exchanger",
        )

        # Area requirement
        disj.exchanger_required_area = Constraint(
            expr=m.exchanger_area[stg, hot, cold]
            * (m.U[hot, cold] * m.LMTD[hot, cold, stg])
            >= m.heat_exchanged[hot, cold, stg],
            doc="Area requirement for exchanger",
        )
        m.BigM[disj.exchanger_required_area] = 5000

    return m
