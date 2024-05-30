"""
Heat integration case study.

This is example 1 of the Yee & Grossmann, 1990 paper "Simultaneous optimization models for heat integration--II". DOI: 10.1016/0098-1354(90)85010-8

This file provides common modeling elements of the heat exchanger network.
The model utilizes sets to organize hot and cold process streams, utility streams, and stages of heat exchange, with parameters defining the essential properties like temperatures and flow capacities. This structure facilitates detailed modeling of the heat transfer process across different stages and stream types.
Disjunctions are employed to model the binary decision of either installing or not installing a heat exchanger between specific stream pairs at each stage, enhancing the model's flexibility and ability to find an optimal solution that balances cost and efficiency.
The objective function aims to minimize the total cost of the heat exchanger network, which includes the costs associated with utility usage and the capital and operational expenses of the heat exchangers, ensuring economic feasibility alongside energy optimization.

Given the common.py, the model can be shown as the conventional model, or can be modified into single module type, integer or discretized formulation, and other various formulations.

References:
    Yee, T. F., & Grossmann, I. E. (1990). Simultaneous optimization models for heat integration—II. Heat exchanger network synthesis. Computers & Chemical Engineering, 14(10), 1165–1184. https://doi.org/10.1016/0098-1354(90)85010-8
"""

from __future__ import division

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    minimize,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Set,
    Suffix,
    value,
    Var,
)
from pyomo.gdp import Disjunct, Disjunction

from .cafaro_approx import calculate_cafaro_coefficients


def build_model(use_cafaro_approximation, num_stages):
    """
    Constructs a Pyomo concrete model for heat integration optimization. This model incorporates various components including process and utility streams, heat exchangers, and stages of heat exchange, with optional application of the Cafaro approximation for certain calculations.

    Parameters
    ----------
    use_cafaro_approximation : bool
        A Boolean flag indicating whether the Cafaro approximation method should be used
        to calculate certain coefficients in the model
    num_stages : int
        The number of stages in the heat exchange model.

    Returns
    -------
    Pyomo.ConcreteModel
        A Pyomo concrete model representing the heat exchange model based on the specified number of stages and the use of Cafaro approximation, if applicable. The model is ready to be solved using an optimization solver to determine optimal heat integration strategies.
    """
    m = ConcreteModel()
    m.hot_process_streams = Set(initialize=['H1', 'H2'], doc="Hot process streams")
    m.cold_process_streams = Set(initialize=['C1', 'C2'], doc="Cold process streams")
    m.process_streams = (
        m.hot_process_streams | m.cold_process_streams
    )  # All process streams
    m.hot_utility_streams = Set(initialize=['steam'], doc="Hot utility streams")
    m.cold_utility_streams = Set(initialize=['water'], doc="Cold utility streams")
    m.hot_streams = Set(
        initialize=m.hot_process_streams | m.hot_utility_streams, doc="Hot streams"
    )
    m.cold_streams = Set(
        initialize=m.cold_process_streams | m.cold_utility_streams, doc="Cold streams"
    )
    m.utility_streams = Set(
        initialize=m.hot_utility_streams | m.cold_utility_streams, doc="Utility streams"
    )
    m.streams = Set(initialize=m.process_streams | m.utility_streams, doc="All streams")
    m.valid_matches = Set(
        initialize=(m.hot_process_streams * m.cold_streams)
        | (m.hot_utility_streams * m.cold_process_streams),
        doc="Match all hot streams to cold streams, but exclude "
        "matches between hot and cold utilities.",
    )
    # m.EMAT = Param(doc="Exchanger minimum approach temperature [K]",
    #                initialize=1)
    # Unused right now, but could be used for variable bound tightening
    # in the LMTD calculation.

    m.stages = RangeSet(num_stages, doc="Number of stages")

    m.T_in = Param(
        m.streams,
        doc="Inlet temperature of stream [K]",
        initialize={
            'H1': 443,
            'H2': 423,
            'C1': 293,
            'C2': 353,
            'steam': 450,
            'water': 293,
        },
    )
    m.T_out = Param(
        m.streams,
        doc="Outlet temperature of stream [K]",
        initialize={
            'H1': 333,
            'H2': 303,
            'C1': 408,
            'C2': 413,
            'steam': 450,
            'water': 313,
        },
    )

    m.heat_exchanged = Var(
        m.valid_matches,
        m.stages,
        domain=NonNegativeReals,
        doc="Heat exchanged from hot stream to cold stream in stage [kW]",
        initialize=1,
        bounds=(0, 5000),
    )

    m.overall_FCp = Param(
        m.process_streams,
        doc="Flow times heat capacity of stream [kW / K]",
        initialize={'H1': 30, 'H2': 15, 'C1': 20, 'C2': 40},
    )
    m.utility_usage = Var(
        m.utility_streams,
        doc="Hot or cold utility used [kW]",
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, 5000),
    )
    m.stage_entry_T = Var(
        m.streams,
        m.stages,
        doc="Temperature of stream at stage entry [K].",
        initialize=350,
        bounds=(293, 450),  # TODO set to be equal to min and max temps
    )
    m.stage_exit_T = Var(
        m.streams,
        m.stages,
        doc="Temperature of stream at stage exit [K].",
        initialize=350,
        bounds=(293, 450),  # TODO set to be equal to min and max temps
    )
    # Improve bounds on stage entry and exit temperatures
    for strm, stg in m.process_streams * m.stages:
        m.stage_entry_T[strm, stg].setlb(min(value(m.T_in[strm]), value(m.T_out[strm])))
        m.stage_exit_T[strm, stg].setlb(min(value(m.T_in[strm]), value(m.T_out[strm])))
        m.stage_entry_T[strm, stg].setub(max(value(m.T_in[strm]), value(m.T_out[strm])))
        m.stage_exit_T[strm, stg].setub(max(value(m.T_in[strm]), value(m.T_out[strm])))
    for strm, stg in m.utility_streams * m.stages:
        _fix_and_bound(m.stage_entry_T[strm, stg], m.T_in[strm])
        _fix_and_bound(m.stage_exit_T[strm, stg], m.T_out[strm])
    for strm in m.hot_process_streams:
        _fix_and_bound(m.stage_entry_T[strm, 1], m.T_in[strm])
        _fix_and_bound(m.stage_exit_T[strm, num_stages], m.T_out[strm])
    for strm in m.cold_process_streams:
        _fix_and_bound(m.stage_exit_T[strm, 1], m.T_out[strm])
        _fix_and_bound(m.stage_entry_T[strm, num_stages], m.T_in[strm])

    m.BigM = Suffix(direction=Suffix.LOCAL)

    m.utility_unit_cost = Param(
        m.utility_streams,
        doc="Annual unit cost of utilities [$/kW]",
        initialize={'steam': 80, 'water': 20},
    )

    m.module_sizes = Set(initialize=[10, 50, 100], doc="Available module sizes.")
    m.max_num_modules = Param(
        m.module_sizes,
        initialize={
            # 5: 100,
            10: 50,
            50: 10,
            100: 5,
            # 250: 2
        },
        doc="maximum number of each module size available.",
    )

    m.exchanger_fixed_unit_cost = Param(
        m.valid_matches, default=2000, doc="exchanger fixed cost [$/kW]"
    )
    m.exchanger_area_cost_factor = Param(
        m.valid_matches,
        default=1000,
        initialize={('steam', cold): 1200 for cold in m.cold_process_streams},
        doc="1200 for heaters. 1000 for all other exchangers.",
    )
    m.area_cost_exponent = Param(default=0.6, doc="Area cost exponent.")

    if use_cafaro_approximation:  # Use Cafaro approximation for coefficients if True
        k, b = calculate_cafaro_coefficients(10, 500, m.area_cost_exponent)
        m.cafaro_k = Param(default=k)
        m.cafaro_b = Param(default=b)

    @m.Param(
        m.valid_matches, m.module_sizes, doc="Area cost factor for modular exchangers."
    )
    def module_area_cost_factor(m, hot, cold, area):
        """
        Determines the area cost factor for modular exchangers within the heat integration model. The cost factor is based on the specified module size and stream pair, with different values for steam and other hot streams. The unit is [$/(m^2)^0.6].

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        hot : str
            The index for the hot stream involved in the heat exchanger.
        cold : str
            The index for the cold stream involved in the heat exchanger.
        area : float
            The modular area size of the heat exchanger.

        Returns
        -------
        Pyomo.Parameter
            The area cost factor for the specified module size and stream pair. It returns a higher value for steam (1300) compared to other hot streams (1100), reflecting specific cost adjustments based on utility type.
        """
        if hot == 'steam':
            return 1300
        else:
            return 1100

    m.module_fixed_unit_cost = Param(default=0, doc="Fixed cost for a module.")
    m.module_area_cost_exponent = Param(default=0.6, doc="Area cost exponent.")

    @m.Param(
        m.valid_matches, m.module_sizes, doc="Cost of a module with a particular area."
    )
    def module_area_cost(m, hot, cold, area):
        """
        Determines the cost of a module with a specified area size for a given hot and cold stream pair. The cost is calculated based on the area cost factor and exponent for the module size and stream pair. The unit is [$].

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        hot : str
            The index for the hot stream involved in the heat exchanger.
        cold : str
            The index for the cold stream involved in the heat exchanger.
        area : float
            The modular area size of the heat exchanger.

        Returns
        -------
        Pyomo.Parameter
            The cost of a module with the specified area size for the given hot and cold stream pair. The cost is calculated based on the area cost factor and exponent for the module size and stream pair.
        """
        return (
            m.module_area_cost_factor[hot, cold, area]
            * area**m.module_area_cost_exponent
        )

    m.U = Param(
        m.valid_matches,
        default=0.8,
        initialize={('steam', cold): 1.2 for cold in m.cold_process_streams},
        doc="Overall heat transfer coefficient."
        "1.2 for heaters. 0.8 for everything else. The unit is [kW/m^2/K].",
    )

    m.exchanger_hot_side_approach_T = Var(
        m.valid_matches,
        m.stages,
        doc="Temperature difference between the hot stream inlet and cold "
        "stream outlet of the exchanger. The unit is [K].",
        bounds=(0.1, 500),
        initialize=10,
    )
    m.exchanger_cold_side_approach_T = Var(
        m.valid_matches,
        m.stages,
        doc="Temperature difference between the hot stream outlet and cold "
        "stream inlet of the exchanger. The unit is [K].",
        bounds=(0.1, 500),
        initialize=10,
    )
    m.LMTD = Var(
        m.valid_matches,
        m.stages,
        doc="Log mean temperature difference across the exchanger.",
        bounds=(1, 500),
        initialize=10,
    )
    # Improve LMTD bounds based on T values
    for hot, cold, stg in m.valid_matches * m.stages:
        hot_side_dT_LB = max(
            0, value(m.stage_entry_T[hot, stg].lb - m.stage_exit_T[cold, stg].ub)
        )
        hot_side_dT_UB = max(
            0, value(m.stage_entry_T[hot, stg].ub - m.stage_exit_T[cold, stg].lb)
        )
        cold_side_dT_LB = max(
            0, value(m.stage_exit_T[hot, stg].lb - m.stage_entry_T[cold, stg].ub)
        )
        cold_side_dT_UB = max(
            0, value(m.stage_exit_T[hot, stg].ub - m.stage_entry_T[cold, stg].lb)
        )
        m.LMTD[hot, cold, stg].setlb(
            (hot_side_dT_LB * cold_side_dT_LB * (hot_side_dT_LB + cold_side_dT_LB) / 2)
            ** (1 / 3)
        )
        m.LMTD[hot, cold, stg].setub(
            (hot_side_dT_UB * cold_side_dT_UB * (hot_side_dT_UB + cold_side_dT_UB) / 2)
            ** (1 / 3)
        )

    m.exchanger_fixed_cost = Var(
        m.stages,
        m.valid_matches,
        doc="Fixed cost for an exchanger between a hot and cold stream.",
        domain=NonNegativeReals,
        bounds=(0, 1e5),
        initialize=0,
    )

    m.exchanger_area = Var(
        m.stages,
        m.valid_matches,
        doc="Area for an exchanger between a hot and cold stream.",
        domain=NonNegativeReals,
        bounds=(0, 500),
        initialize=5,
    )
    m.exchanger_area_cost = Var(
        m.stages,
        m.valid_matches,
        doc="Capital cost contribution from exchanger area.",
        domain=NonNegativeReals,
        bounds=(0, 1e5),
        initialize=1000,
    )

    @m.Constraint(m.hot_process_streams)
    def overall_hot_stream_heat_balance(m, strm):
        """
        Enforces the heat balance for a hot process stream within the model. This constraint ensures that the total heat loss from the hot stream equals the sum of heat transferred to all paired cold streams across all stages. The heat loss is calculated based on the temperature difference between the stream outlet and inlet, multiplied by the overall flow times heat capacity of the stream.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        strm : str
            The index for the hot stream involved in the heat exchanger.

        Returns
        -------
        Pyomo.Constraint
            A constraint object that ensures the heat balance across the specified hot stream over all stages and cold stream interactions.
        """
        return (m.T_in[strm] - m.T_out[strm]) * m.overall_FCp[strm] == (
            sum(
                m.heat_exchanged[strm, cold, stg]
                for cold in m.cold_streams
                for stg in m.stages
            )
        )

    @m.Constraint(m.cold_process_streams)
    def overall_cold_stream_heat_balance(m, strm):
        """
        Enforces the heat balance for a cold process stream within the model. This constraint ensures that the total heat gain for the cold stream equals the sum of heat received from all paired hot streams across all stages. The heat gain is calculated based on the temperature difference between the stream outlet and inlet, multiplied by the overall flow times heat capacity of the stream.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        strm : str
            The index for the cold stream involved in the heat exchanger.

        Returns
        -------
        Pyomo.Constraint
            A constraint object that ensures the heat balance across the specified cold stream over all stages and hot stream interactions.
        """
        return (m.T_out[strm] - m.T_in[strm]) * m.overall_FCp[strm] == (
            sum(
                m.heat_exchanged[hot, strm, stg]
                for hot in m.hot_streams
                for stg in m.stages
            )
        )

    @m.Constraint(m.utility_streams)
    def overall_utility_stream_usage(m, strm):
        """
        Ensures the total utility usage for each utility stream matches the sum of heat exchanged involving that utility across all stages. This constraint separates the calculations for hot and cold utility streams. For cold utility streams, it sums the heat exchanged from all hot process streams to the utility, and for hot utility streams, it sums the heat exchanged from the utility to all cold process streams.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        strm : str
            The index for the utility stream involved in the heat exchanger. This can be a hot or cold utility, and the constraint dynamically adjusts to sum the appropriate heat transfers based on this classification.

        Returns
        -------
        Pyomo.Constraint
            A constraint object that ensures the total calculated utility usage for the specified utility stream accurately reflects the sum of relevant heat exchanges in the system. This helps maintain energy balance specifically for utility streams within the overall heat exchange model.
        """
        return m.utility_usage[strm] == (
            sum(
                m.heat_exchanged[hot, strm, stg]
                for hot in m.hot_process_streams
                for stg in m.stages
            )
            if strm in m.cold_utility_streams
            else (
                0
                + sum(
                    m.heat_exchanged[strm, cold, stg]
                    for cold in m.cold_process_streams
                    for stg in m.stages
                )
                if strm in m.hot_utility_streams
                else 0
            )
        )

    @m.Constraint(
        m.stages,
        m.hot_process_streams,
        doc="Hot side overall heat balance for a stage.",
    )
    def hot_stage_overall_heat_balance(m, stg, strm):
        """
        Establishes an overall heat balance for a specific hot stream within a particular stage of the heat exchange process. This constraint ensures that the heat loss from the hot stream, calculated as the product of the temperature drop across the stage and the flow capacity of the stream, equals the total heat transferred to all corresponding cold streams within the same stage.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        stg : int
            The index for the stage involved in the heat exchanger.
        strm : str
            The index for the hot stream involved in the heat exchanger.

        Returns
        -------
        Pyomo.Constraint
            A constraint object that enforces the heat balance for the specified hot stream at the given stage. This ensures that the heat output from this stream is appropriately accounted for and matched by heat intake by the cold streams, promoting efficient energy use.
        """
        return (
            (m.stage_entry_T[strm, stg] - m.stage_exit_T[strm, stg])
            * m.overall_FCp[strm]
        ) == sum(m.heat_exchanged[strm, cold, stg] for cold in m.cold_streams)

    @m.Constraint(
        m.stages,
        m.cold_process_streams,
        doc="Cold side overall heat balance for a stage.",
    )
    def cold_stage_overall_heat_balance(m, stg, strm):
        """
        Establishes an overall heat balance for a specific cold stream within a particular stage of the heat exchange process. This constraint ensures that the heat gain for the cold stream, calculated as the product of the temperature increase across the stage and the flow capacity of the stream, equals the total heat received from all corresponding hot streams within the same stage.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        stg : int
            The index for the stage involved in the heat exchanger.
        strm : str
            The index for the cold stream involved in the heat exchanger.

        Returns
        -------
        Pyomo.Constraint
            A constraint object that enforces the heat balance for the specified cold stream at the given stage. This ensures that the heat intake by this stream is appropriately accounted for and matched by heat output from the hot streams, promoting efficient energy use.
        """
        return (
            (m.stage_exit_T[strm, stg] - m.stage_entry_T[strm, stg])
            * m.overall_FCp[strm]
        ) == sum(m.heat_exchanged[hot, strm, stg] for hot in m.hot_streams)

    @m.Constraint(m.stages, m.hot_process_streams)
    def hot_stream_monotonic_T_decrease(m, stg, strm):
        """
        Ensures that the temperature of a hot stream decreases monotonically across a given stage. This constraint is critical for modeling realistic heat exchange scenarios where hot streams naturally cool down as they transfer heat to colder streams. It enforces that the exit temperature of the hot stream from any stage is less than or equal to its entry temperature for that stage.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        stg : int
            The index for the stage involved in the heat exchanger.
        strm : str
            The index for the hot stream involved in the heat exchanger.

        Returns
        -------
        Pyomo.Constraint
            A constraint object that ensures the temperature of the hot stream does not increase as it passes through the stage, which is essential for maintaining the physical feasibility of the heat exchange process.
        """
        return m.stage_exit_T[strm, stg] <= m.stage_entry_T[strm, stg]

    @m.Constraint(m.stages, m.cold_process_streams)
    def cold_stream_monotonic_T_increase(m, stg, strm):
        """
        Ensures that the temperature of a cold stream increases monotonically across a given stage. This constraint is essential for modeling realistic heat exchange scenarios where cold streams naturally warm up as they absorb heat from hotter streams. It enforces that the exit temperature of the cold stream from any stage is greater than or equal to its entry temperature for that stage.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        stg : int
            The index for the stage involved in the heat exchanger.
        strm : str
            The index for the cold stream involved in the heat exchanger.

        Returns
        -------
        Pyomo.Constraint
            A constraint object that ensures the temperature of the cold stream increases as it passes through the stage, reflecting the natural heat absorption process and maintaining the physical feasibility of the heat exchange model.
        """
        return m.stage_exit_T[strm, stg] >= m.stage_entry_T[strm, stg]

    @m.Constraint(m.stages, m.hot_process_streams)
    def hot_stream_stage_T_link(m, stg, strm):
        """
        Links the exit temperature of a hot stream from one stage to the entry temperature of the same stream in the subsequent stage, ensuring continuity and consistency in temperature progression across stages. This constraint is vital for maintaining a coherent thermal profile within each hot stream as it progresses through the heat exchange stages. For the final stage, no constraint is applied since there is no subsequent stage.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        stg : int
            The index for the stage involved in the heat exchanger.
        strm : str
            The index for the hot stream involved in the heat exchanger.

        Returns
        -------
        Pyomo.Constraint
            A constraint object that ensures the exit temperature at the end of one stage matches the entry temperature at the beginning of the next stage for the hot streams. In the final stage, where there is no subsequent stage, no constraint is applied.
        """
        return (
            (m.stage_exit_T[strm, stg] == m.stage_entry_T[strm, stg + 1])
            if stg < num_stages
            else Constraint.NoConstraint
        )

    @m.Constraint(m.stages, m.cold_process_streams)
    def cold_stream_stage_T_link(m, stg, strm):
        """
        Ensures continuity in the temperature profiles of cold streams across stages in the heat exchange model by linking the exit temperature of a cold stream in one stage to its entry temperature in the following stage. This constraint is crucial for maintaining consistent and logical heat absorption sequences within the cold streams as they move through successive stages. For the final stage, no constraint is applied since there is no subsequent stage.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        stg : int
            The index for the stage involved in the heat exchanger.
        strm : str
            The index for the cold stream involved in the heat exchanger.

        Returns
        -------
        Pyomo.Constraint
            A constraint object that ensures the exit temperature at the end of one stage matches the entry temperature at the beginning of the next stage for cold streams. In the final stage, where there is no subsequent stage, no constraint is applied, reflecting the end of the process sequence.
        """
        return (
            (m.stage_entry_T[strm, stg] == m.stage_exit_T[strm, stg + 1])
            if stg < num_stages
            else Constraint.NoConstraint
        )

    @m.Expression(m.valid_matches, m.stages)
    def exchanger_capacity(m, hot, cold, stg):
        """
        Calculates the heat transfer capacity of an exchanger for a given hot stream, cold stream, and stage combination. This capacity is derived from the exchanger's area, the overall heat transfer coefficient, and the geometric mean of the approach temperatures at both sides of the exchanger. This expression is used to estimate the efficiency and effectiveness of heat transfer in each stage of the heat exchange process.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        hot : str
            The index for the hot stream involved in the heat exchanger.
        cold : str
            The index for the cold stream involved in the heat exchanger.
        stg : int
            The index for the stage involved in the heat exchanger.

        Returns
        -------
        Pyomo.Expression
            A Pyomo expression that quantifies the heat transfer capacity of the exchanger. This value is crucial for optimizing the heat exchange system, ensuring that each stage is designed to maximize heat recovery while adhering to operational constraints and physical laws.
        """
        return m.exchanger_area[stg, hot, cold] * (
            m.U[hot, cold]
            * (
                m.exchanger_hot_side_approach_T[hot, cold, stg]
                * m.exchanger_cold_side_approach_T[hot, cold, stg]
                * (
                    m.exchanger_hot_side_approach_T[hot, cold, stg]
                    + m.exchanger_cold_side_approach_T[hot, cold, stg]
                )
                / 2
            )
            ** (1 / 3)
        )

    def _exchanger_exists(disj, hot, cold, stg):
        """
        Defines the conditions and constraints for the existence of an exchanger between a specified hot and cold stream at a given stage. This function sets the disjunct's indicator variable to true and configures constraints that model the physical behavior of the heat exchanger, including the log mean temperature difference and approach temperatures.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct object representing a potential heat exchanger scenario between the specified hot and cold streams.
        hot : str
            The index for the hot stream involved in the heat exchanger.
        cold : str
            The index for the cold stream involved in the heat exchanger.
        stg : int
            The index for the stage involved in the heat exchanger.
        """
        disj.indicator_var.value = True

        # Log mean temperature difference calculation
        disj.LMTD_calc = Constraint(
            doc="Log mean temperature difference",
            expr=m.LMTD[hot, cold, stg]
            == (
                m.exchanger_hot_side_approach_T[hot, cold, stg]
                * m.exchanger_cold_side_approach_T[hot, cold, stg]
                * (
                    m.exchanger_hot_side_approach_T[hot, cold, stg]
                    + m.exchanger_cold_side_approach_T[hot, cold, stg]
                )
                / 2
            )
            ** (1 / 3),
        )
        m.BigM[disj.LMTD_calc] = 160

        # disj.MTD_calc = Constraint(
        #     doc="Mean temperature difference",
        #     expr=m.LMTD[hot, cold, stg] <= (
        #         m.exchanger_hot_side_approach_T[hot, cold, stg] +
        #         m.exchanger_cold_side_approach_T[hot, cold, stg]) / 2
        # )

        # Calculation of the approach temperatures
        if hot in m.hot_utility_streams:
            disj.stage_hot_approach_temperature = Constraint(
                expr=m.exchanger_hot_side_approach_T[hot, cold, stg]
                <= m.T_in[hot] - m.stage_exit_T[cold, stg],
                doc="Hot utility: hot side limit.",
            )
            disj.stage_cold_approach_temperature = Constraint(
                expr=m.exchanger_cold_side_approach_T[hot, cold, stg]
                <= m.T_out[hot] - m.stage_entry_T[cold, stg],
                doc="Hot utility: cold side limit.",
            )
        elif cold in m.cold_utility_streams:
            disj.stage_hot_approach_temperature = Constraint(
                expr=m.exchanger_hot_side_approach_T[hot, cold, stg]
                <= m.stage_entry_T[hot, stg] - m.T_out[cold],
                doc="Cold utility: hot side limit.",
            )
            disj.stage_cold_approach_temperature = Constraint(
                expr=m.exchanger_cold_side_approach_T[hot, cold, stg]
                <= m.stage_exit_T[hot, stg] - m.T_in[cold],
                doc="Cold utility: cold side limit.",
            )
        else:
            disj.stage_hot_approach_temperature = Constraint(
                expr=m.exchanger_hot_side_approach_T[hot, cold, stg]
                <= m.stage_entry_T[hot, stg] - m.stage_exit_T[cold, stg],
                doc="Process stream: hot side limit.",
            )
            disj.stage_cold_approach_temperature = Constraint(
                expr=m.exchanger_cold_side_approach_T[hot, cold, stg]
                <= m.stage_exit_T[hot, stg] - m.stage_entry_T[cold, stg],
                doc="Process stream: cold side limit.",
            )

    def _exchanger_absent(disj, hot, cold, stg):
        """
        Defines the conditions for the absence of a heat exchanger between a specified hot and cold stream at a given stage. This function sets the disjunct's indicator variable to false and ensures that all associated costs and heat exchanged values are set to zero, effectively removing the exchanger from the model for this configuration.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct object representing a scenario where no heat exchanger is present between the specified hot and cold streams at the given stage.
        hot : str
            The index for the hot stream involved in the heat exchanger.
        cold : str
            The index for the cold stream involved in the heat exchanger.
        stg : int
            The index for the stage involved in the heat exchanger.
        """
        disj.indicator_var.value = False
        disj.no_match_exchanger_cost = Constraint(
            expr=m.exchanger_area_cost[stg, hot, cold] == 0, doc="No exchanger cost."
        )
        disj.no_match_exchanger_area = Constraint(
            expr=m.exchanger_area[stg, hot, cold] == 0, doc="No exchanger area."
        )
        disj.no_match_exchanger_fixed_cost = Constraint(
            expr=m.exchanger_fixed_cost[stg, hot, cold] == 0,
            doc="No exchanger fixed cost.",
        )
        disj.no_heat_exchange = Constraint(
            expr=m.heat_exchanged[hot, cold, stg] == 0, doc="No heat exchange."
        )

    m.exchanger_exists = Disjunct(
        m.valid_matches,
        m.stages,
        doc="Disjunct for the presence of an exchanger between a "
        "hot stream and a cold stream at a stage.",
        rule=_exchanger_exists,
    )
    m.exchanger_absent = Disjunct(
        m.valid_matches,
        m.stages,
        doc="Disjunct for the absence of an exchanger between a "
        "hot stream and a cold stream at a stage.",
        rule=_exchanger_absent,
    )

    def _exchanger_exists_or_absent(m, hot, cold, stg):
        """
        Defines a disjunction to represent the decision between installing or not installing a heat exchanger between a specific hot and cold stream at a certain stage.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        hot : str
            The index for the hot stream involved in the heat exchanger.
        cold : str
            The index for the cold stream involved in the heat exchanger.
        stg : int
            The index for the stage involved in the heat exchanger.

        Returns
        -------
        list
            A list of Pyomo Disjunct objects, which includes the scenarios where the exchanger exists or is absent, allowing the model to explore different configurations for optimal energy use and cost efficiency.
        """
        return [m.exchanger_exists[hot, cold, stg], m.exchanger_absent[hot, cold, stg]]

    m.exchanger_exists_or_absent = Disjunction(
        m.valid_matches,
        m.stages,
        doc="Disjunction between presence or absence of an exchanger between "
        "a hot stream and a cold stream at a stage.",
        rule=_exchanger_exists_or_absent,
        xor=True,
    )
    # Only hot utility matches in first stage and cold utility matches in last
    # stage
    for hot, cold in m.valid_matches:
        if hot not in m.utility_streams:
            m.exchanger_exists[hot, cold, 1].deactivate()
            m.exchanger_absent[hot, cold, 1].indicator_var.fix(True)
        if cold not in m.utility_streams:
            m.exchanger_exists[hot, cold, num_stages].deactivate()
            m.exchanger_absent[hot, cold, num_stages].indicator_var.fix(True)
    # Exclude utility-stream matches in middle stages
    for hot, cold, stg in m.valid_matches * (m.stages - [1, num_stages]):
        if hot in m.utility_streams or cold in m.utility_streams:
            m.exchanger_exists[hot, cold, stg].deactivate()
            m.exchanger_absent[hot, cold, stg].indicator_var.fix(True)

    @m.Expression(m.utility_streams)
    def utility_cost(m, strm):
        """
        alculates the cost associated with the usage of a utility stream within the heat exchange model.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo concrete model representing the heat exchange model.
        strm : str
            The index for the utility stream involved in the heat exchanger. This can be a hot or cold utility.

        Returns
        -------
        Pyomo.Expression
            An expression representing the total cost of using the specified utility stream within the model, computed as the product of unit cost and usage. This helps in assessing the economic impact of utility choices in the heat exchange system.
        """
        return m.utility_unit_cost[strm] * m.utility_usage[strm]

    m.total_cost = Objective(
        expr=sum(m.utility_cost[strm] for strm in m.utility_streams)
        + sum(
            m.exchanger_fixed_cost[stg, hot, cold]
            for stg in m.stages
            for hot, cold in m.valid_matches
        )
        + sum(
            m.exchanger_area_cost[stg, hot, cold]
            for stg in m.stages
            for hot, cold in m.valid_matches
        ),
        sense=minimize,
        doc="Total cost of the heat exchanger network.",
    )

    return m


def _fix_and_bound(var, val):
    """
    Fix a Pyomo variable to a value and set bounds to that value.

    Parameters
    ----------
    var : Pyomo.Var
        The Pyomo variable to be fixed.
    val : float
        The value to fix the variable to. This value will also be used to set both the lower and upper bounds of the variable.
    """
    var.fix(val)
    var.setlb(val)
    var.setub(val)
