# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import pandas as pd
from pyomo.core.expr.logical_expr import *
from pyomo.core.plugins.transform.logical_to_linear import (
    update_boolean_vars_from_binary,
)
from pyomo.environ import *
from pyomo.environ import TerminationCondition as tc
import os

syngas_dir = os.path.dirname(os.path.realpath(__file__))


def build_model():
    """_summary_

    Returns
    -------
    Pyomo.ConcreteModel
        Build the ConcreteModel of combined synthesis gas reforming technologies
    Notes
    -----

    References
    ----------
    [1] Chen, Q. (2020). Pyosyn: Advanced Computational Tools for Process Synthesis (Doctoral dissertation, Carnegie Mellon University).
    [2] Medrano-García, J. D., Ruiz-Femenia, R., & Caballero, J. A. (2017). Multi-objective optimization of combined synthesis gas reforming technologies. Journal of CO2 Utilization, 22, 355-373. https://doi.org/10.1016/j.jcou.2017.09.019
    """
    m = ConcreteModel()

    m.syngas_techs = Set(
        doc="syngas process technologies",
        initialize=['SMR', 'POX', 'ATR', 'CR', 'DMR', 'BR', 'TR'],
    )
    m.species = Set(
        doc="chemical species", initialize=['CH4', 'H2O', 'O2', 'H2', 'CO', 'CO2']
    )
    m.syngas_tech_units = Set(
        doc="process units involved in syngas process technologies",
        initialize=['compressor', 'exchanger', 'reformer'],
    )
    m.utilities = Set(
        doc="process utilities", initialize=['power', 'coolingwater', 'naturalgas']
    )
    m.aux_equipment = Set(
        doc="auxiliary process equipment to condition syngas",
        initialize=[
            'absorber1',
            'bypass1',
            'WGS',
            'flash',
            'PSA',
            'absorber2',
            'bypass3',
            'compressor',
            'bypass4',
        ],
    )
    m.process_options = Set(
        doc="process superstructure options",
        initialize=m.syngas_techs | m.aux_equipment,
    )
    m.extra_process_nodes = Set(
        doc="extra process nodes for mixers and splitters",
        initialize=['in', 'ms1', 'm1', 's1', 'ms2', 'ms3', 'm2', 'ms4', 's2'],
    )
    m.all_unit_types = Set(
        doc="all process unit types",
        initialize=m.process_options | m.syngas_tech_units | m.extra_process_nodes,
    )
    m.superstructure_nodes = Set(
        doc="nodes in the superstructure",
        initialize=m.process_options | m.extra_process_nodes,
    )
    group1 = {'absorber1', 'bypass1', 'WGS'}
    group2 = {'compressor', 'bypass3'}
    group3 = {'absorber2', 'bypass4'}
    m.streams = Set(
        doc="superstructure potential streams",
        initialize=[(tech, 'ms1') for tech in m.syngas_techs]
        + [('in', tech) for tech in m.syngas_techs]
        + [('ms1', option) for option in group1]
        + [(option, 'm1') for option in group1]
        + [('m1', 'flash'), ('flash', 's1'), ('s1', 'PSA'), ('s1', 'ms2')]
        + [('ms2', option) for option in group2]
        + [(option, 'ms3') for option in group2]
        + [('ms3', option) for option in group3]
        + [(option, 'm2') for option in group3]
        + [('PSA', 'ms4'), ('ms4', 'ms3'), ('ms4', 's2'), ('s2', 'm1'), ('s2', 'ms1')],
    )

    """
    Parameters
    """

    m.flow_limit = Param(initialize=5, doc="upper bound of the flow")
    m.min_flow_division = Param(
        initialize=0.1, doc="minimum flow fraction into active unit"
    )

    m.raw_material_cost = Param(
        m.species,
        doc="raw material cost [dollar·kmol-1]",
        initialize={
            'CH4': 2.6826,
            'H2O': 0.18,
            'O2': 0.7432,
            'H2': 8,
            'CO': 0,
            'CO2': 1.8946,
        },
    )

    feed_ratios = {
        ('SMR', 'H2O'): 3,
        ('POX', 'O2'): 0.5,
        ('ATR', 'H2O'): 1.43,
        ('ATR', 'O2'): 0.6,
        ('CR', 'H2O'): 2.5,
        ('CR', 'O2'): 0.19,
        ('DMR', 'CO2'): 1,
        ('BR', 'H2O'): 1.6,
        ('BR', 'CO2'): 0.8,
        ('TR', 'H2O'): 2.46,
        ('TR', 'CO2'): 1.3,
        ('TR', 'O2'): 0.47,
    }

    data_dict = (
        pd.read_csv('%s/syngas_conversion_data.txt' % syngas_dir, delimiter=r'\s+')
        .fillna(0)
        .stack()
        .to_dict()
    )
    m.syngas_conversion_factor = Param(
        m.species,
        m.syngas_techs,
        initialize=data_dict,
        doc="conversion factor of the chemical species with different technology",
    )
    m.co2_ratio = Param(
        doc="CO2-CO ratio of total carbon in final syngas", initialize=0.05
    )

    # Capital costs -->  cap = (p1*x + p2)*(B1 + B2*Fmf*Fp)
    m.p1 = Param(
        m.all_unit_types,
        doc="capital cost variable parameter",
        initialize={
            'compressor': 172.4,
            'exchanger': 59.99,
            'reformer': 67.64,
            'absorber1': 314.1,
            'bypass1': 0,
            'WGS': 314.1,
            'flash': 314.1,
            'PSA': 3.1863e04,
            'absorber2': 314.1,
            'bypass3': 0,
        },
        default=0,
    )
    m.p2 = Param(
        m.all_unit_types,
        doc="capital cost fixed parameter",
        initialize={
            'compressor': 104300,
            'exchanger': 187100,
            'reformer': 480100,
            'absorber1': 1.531e04,
            'bypass1': 0,
            'WGS': 1.531e04,
            'flash': 1.531e04,
            'PSA': 666.34e03,
            'absorber2': 1.531e04,
            'bypass3': 0,
        },
        default=0,
    )
    m.material_factor = Param(
        m.all_unit_types,
        doc="capital cost material factor",
        initialize={
            'compressor': 3.5,
            'exchanger': 1.7,
            'reformer': 4,
            'absorber1': 1,
            'bypass1': 0,
            'WGS': 1,
            'flash': 1,
            'PSA': 3.5,
            'absorber2': 1,
            'bypass3': 0,
        },
        default=0,
    )
    m.B1 = Param(
        m.all_unit_types,
        doc="bare module parameter 1",
        initialize={
            'compressor': 0,
            'exchanger': 1.63,
            'reformer': 0,
            'absorber1': 2.25,
            'bypass1': 0,
            'WGS': 1.49,
            'flash': 2.25,
            'PSA': 0,
            'absorber2': 2.25,
            'bypass3': 0,
        },
        default=0,
    )
    m.B2 = Param(
        m.all_unit_types,
        doc="bare module parameter 2",
        initialize={
            'compressor': 1,
            'exchanger': 1.66,
            'reformer': 1,
            'absorber1': 1.82,
            'bypass1': 0,
            'WGS': 1.52,
            'flash': 1.82,
            'PSA': 1,
            'absorber2': 1.82,
            'bypass3': 0,
        },
        default=0,
    )
    data_dict = (
        pd.read_csv('%s/syngas_pressure_factor_data.txt' % syngas_dir, delimiter=r'\s+')
        .stack()
        .to_dict()
    )
    m.syngas_pressure_factor = Param(
        m.syngas_tech_units, m.syngas_techs, initialize=data_dict
    )

    m.utility_cost = Param(
        m.utilities,
        doc="dollars per kWh of utility u [dollar·KWh-1]",
        initialize={'power': 0.127, 'coolingwater': 0.01, 'naturalgas': 0.036},
    )
    m.utility_emission = Param(
        m.utilities,
        doc="CO2 emitted per kW [kgCO2·kWh-1]",
        initialize={'power': 0.44732, 'coolingwater': 0, 'naturalgas': 0.2674},
    )
    m.raw_material_emission = Param(
        m.species,
        doc="kg CO2 emitted per kmol of raw material [kgCO2·kmol-1]",
        initialize={
            'CH4': 11.7749,
            'H2O': 0,
            'O2': 10.9525,
            'H2': 0,
            'CO': 0,
            'CO2': 0,
        },
    )

    m.interest_rate = Param(initialize=0.1, doc="annualization index")
    m.project_years = Param(initialize=8, doc="annualization years")
    m.annualization_factor = Expression(
        expr=m.interest_rate
        * (1 + m.interest_rate) ** m.project_years
        / ((1 + m.interest_rate) ** m.project_years - 1),
        doc="Factor for annualized payments",
    )
    m.CEPCI2015 = Param(initialize=560, doc="equipment cost index 2015")
    m.CEPCI2001 = Param(initialize=397, doc="equipment cost index 2001")
    m.cost_index_ratio = Param(
        initialize=m.CEPCI2015 / m.CEPCI2001, doc="cost index ratio"
    )

    data_dict = (
        pd.read_csv('%s/syngas_utility_data.txt' % syngas_dir, delimiter=r'\s+')
        .stack()
        .to_dict()
    )
    m.syngas_tech_utility_rate = Param(
        m.utilities,
        m.syngas_techs,
        initialize=data_dict,
        doc="kWh of utility u per kmol of methane fed in syngas process i [kWh·kmol methane -1]",
    )

    data_dict = (
        pd.read_csv('%s/syngas_num_units_data.txt' % syngas_dir, delimiter=r'\s+')
        .stack()
        .to_dict()
    )
    m.syngas_tech_num_units = Param(
        m.syngas_tech_units,
        m.syngas_techs,
        initialize=data_dict,
        doc="number of units h in syngas process i",
    )

    m.syngas_tech_exchanger_area = Param(
        m.syngas_techs,
        doc="total exchanger area in process i per kmol·h-1 methane fed [m2·h·kmol methane -1]",
        initialize={
            'SMR': 0.885917,
            'POX': 0.153036,
            'ATR': 0.260322,
            'CR': 0.726294,
            'DMR': 0.116814,
            'BR': 0.825808,
            'TR': 0.10539,
        },
    )

    m.syngas_tech_reformer_duty = Param(
        m.syngas_techs,
        doc="reformer duties per kmol methane fed [kWh·kmol methane-1]",
        initialize={
            'SMR': 54.654,
            'POX': 39.0104,
            'ATR': 44.4600,
            'CR': 68.2382,
            'DMR': 68.412,
            'BR': 61.442,
            'TR': 6.592,
        },
    )

    m.process_tech_pressure = Param(
        m.syngas_techs,
        doc="process i operating pressure  [bar]",
        initialize={
            'SMR': 20,
            'POX': 30,
            'ATR': 25,
            'CR': 25,
            'DMR': 1,
            'BR': 7,
            'TR': 20,
        },
    )
    m.final_syngas_pressure = Param(doc="final syngas pressure [bar]", initialize=1)
    m.psa_hydrogen_recovery = Param(
        initialize=0.9,
        doc="percentage of hydrogen separated from the inlet syngas stream",
    )
    m.psa_separation_hydrogen_purity = Param(initialize=0.999)

    m.Keqw = Param(
        initialize=83.3429, doc="equilibrium constant for WGS reaction at 250ºC"
    )

    m.stoichiometric_number = Param(
        doc="stoichiometric number of product syngas", initialize=1
    )
    m.max_impurity = Param(initialize=0.1, doc="maximum allowed of impurities")
    m.max_syngas_techs = Param(
        initialize=1, doc="Number of syngas technologies that can be selected"
    )

    """
    Variables
    """

    m.flow = Var(
        m.streams,
        m.species,
        bounds=(0, m.flow_limit),
        doc="molar flow of each species in each stream [kmol·s-1]",
    )
    m.wgs_steam = Var(
        doc="steam molar flow provided in the WGS reactor [kmol·s-1]", bounds=(0, None)
    )
    m.oxygen_flow = Var(
        doc="O2 molar flow provided in the selective oxidation reactor [kmol·s-1]",
        bounds=(0, None),
    )
    m.Fabs1 = Var(
        bounds=(0, None), doc="molar flow of CO2 absorbed in absorber1 [kmol·s-1]"
    )
    m.Fabs2 = Var(
        bounds=(0, None), doc="molar flow of CO2 absorbed in absorber2 [kmol·s-1]"
    )
    m.flash_water = Var(bounds=(0, None), doc="water removed in flash [kmol·s-1]")
    m.co2_inject = Var(
        bounds=(0, None),
        doc="molar flow of CO2 used to adjust syngas composition [kmol·s-1]",
    )
    m.psa_recovered = Var(
        m.species,
        bounds=(0, None),
        doc="pure hydrogen stream retained in the PSA [kmol·s-1]",
    )
    m.purge_flow = Var(
        m.species, bounds=(0, None), doc="purged molar flow from the PSA [kmol·s-1]"
    )
    m.final_syngas_flow = Var(
        m.species,
        bounds=(0, m.flow_limit),
        doc="final adjusted syngas molar flow [kmol·s-1]",
    )

    @m.Expression(m.superstructure_nodes, m.species)
    def flow_into(m, option, species):
        """
        Calculate the total incoming flow of a specific chemical species to a designated process node (option) within the syngas production superstructure. This expression aggregates the flow rates for each species entering a particular node from all connected upstream nodes (sources).

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        option : str
            The specific node in the superstructure to which the flow is being calculated. This node acts as the destination for various streams in the model. 'option is the element of 'm.superstructure_nodes'.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Expression
            A Pyomo Expression that sums up all the incoming flow rates of the specified `species` to the `option` node. The sum is over all streams that terminate at this node. The unit is in [kmol·s-1].
        """
        return sum(
            m.flow[src, sink, species] for src, sink in m.streams if sink == option
        )

    @m.Expression(m.superstructure_nodes, m.species)
    def flow_out_from(m, option, species):
        """
        Calculate the total outgoing flow of a specific chemical species from a designated process node (option) within the syngas production superstructure. This expression sums the flow rates for each species leaving a particular node to all connected downstream nodes (destinations).

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        option : str
            The specific node in the superstructure to which the flow is being calculated. This node acts as the destination for various streams in the model. 'option is the element of 'm.superstructure_nodes'.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Expression
            A Pyomo Expression that sums up all the outgoing flow rates of the specified 'species' from the 'option' node. The sum is over all streams that originate from this node. The unit is in [kmol·s-1].
        """
        return sum(
            m.flow[src, sink, species] for src, sink in m.streams if src == option
        )

    @m.Expression(m.superstructure_nodes)
    def total_flow_into(m, option):
        """
        Calculate the total incoming flow to a specified node (option) in the syngas production superstructure, aggregating across all chemical species.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        option : str
            The specific node in the superstructure to which the flow is being calculated. This node acts as the destination for various streams in the model. 'option is the element of 'm.superstructure_nodes'.

        Returns
        -------
        Pyomo.Expression
            A Pyomo Expression summing all incoming flows (across all species) to the node 'option'. The unit is in [kmol·s-1].
        """
        return sum(m.flow_into[option, species] for species in m.species)

    @m.Expression(m.superstructure_nodes)
    def total_flow_from(m, option):
        """
        Calculate the total outgoing flow from a specified node (option) in the syngas production superstructure, aggregating across all chemical species.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        option : str
            The specific node in the superstructure to which the flow is being calculated. This node acts as the destination for various streams in the model. 'option is the element of 'm.superstructure_nodes'.

        Returns
        -------
        Pyomo.Expression
            A Pyomo Expression summing all outgoing flows (across all species) from the node 'option'. The unit is in [kmol·s-1].
        """
        return sum(m.flow_out_from[option, species] for species in m.species)

    m.base_tech_capital_cost = Var(
        m.syngas_techs,
        m.syngas_tech_units,
        bounds=(0, None),
        doc="capital cost for each syngas technology and process unit combination [$·h-1]",
    )
    m.base_tech_operating_cost = Var(
        m.syngas_techs,
        m.utilities,
        bounds=(0, None),
        doc="perating cost for each syngas technology and utility used [$·h-1]",
    )
    m.raw_material_total_cost = Var(
        bounds=(0, None), doc="total cost of raw materials [$·s-1]"
    )

    @m.Expression(m.species)
    def raw_material_flow(m, species):
        """
        Calculate the total input flow of a specific chemical species across all syngas technologies.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Expression
            A Pyomo Expression representing the total flow of the specified 'species' into the process from all entry points. The unit is in [kmol·s-1].
        """
        return sum(m.flow['in', tech, species] for tech in m.syngas_techs)

    m.syngas_tech_cost = Var(
        bounds=(0, None), doc="total cost of sygas process [$·y-1]"
    )
    m.syngas_tech_emissions = Var(
        bounds=(0, None), doc="CO2 emission of syngas processes [kmol·s-1]"
    )

    @m.Expression(m.syngas_techs, m.syngas_tech_units)
    def module_factors(m, tech, equip):
        """
        Compute the module factors for each combination of syngas technology and equipment.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        tech : str
            Index of the syngas process technology (e.g., SMR, POX, ATR).
        equip : str
            Index of the equipment type (e.g., compressor, exchanger, reformer) begin modeled.

        Returns
        -------
        Pyomo.Expression
            A Pyomo Expression that calculates the module factor for the specified technology and equipment combination. This is done by summing the baseline module parameter (B1) and the product of the second baseline module parameter (B2), the material factor, and the syngas pressure factor specific to the technology and equipment.
        """
        return (
            m.B1[equip]
            + m.B2[equip]
            * m.material_factor[equip]
            * m.syngas_pressure_factor[equip, tech]
        )

    @m.Expression(m.syngas_techs, m.syngas_tech_units)
    def variable_utilization(m, tech, equip):
        """
        Calculate the utilization rate for different equipment types within each syngas technology based on the methane input flow.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        tech : str
            Index of the syngas process technology (e.g., SMR, POX, ATR).
        equip : str
            Index of the equipment type (e.g., compressor, exchanger, reformer) begin modeled.

        Returns
        -------
        Pyomo.Expression
            A Pyomo Expression that calculates the equipment-specific utilization rate. It multiplies the relevant equipment parameter (power usage, exchanger area, or reformer duty) by the methane input flow and then scales this product to an hourly basis (multiplied by 3600 seconds/hour). The result is a key factor in determining how effectively each piece of equipment is used in the syngas process based on the input flow rates.
        """
        variable_rate_term = {
            'compressor': m.syngas_tech_utility_rate['power', tech],
            'exchanger': m.syngas_tech_exchanger_area[tech],
            'reformer': m.syngas_tech_reformer_duty[tech],
        }
        return variable_rate_term[equip] * m.flow['in', tech, 'CH4'] * 3600

    m.aux_unit_capital_cost = Var(
        m.aux_equipment, bounds=(0, None), doc="auxiliary unit capital cost [$·h-1]"
    )

    m.first_stage_outlet_pressure = Var(
        doc="final pressure in the mixer before WGS and absorber", bounds=(0, None)
    )
    m.syngas_tech_outlet_pressure = Var(
        m.syngas_techs,
        doc="final pressure after compression after syngas synthesis [bar]",
        bounds=(0, None),
    )
    m.syngas_tech_compressor_power = Var(
        m.syngas_techs,
        doc="utility of compressor i after syngas synthesis",
        bounds=(0, None),
    )
    m.syngas_tech_compressor_cost = Var(
        doc="capital cost of compressors after syngas synthesis", bounds=(0, None)
    )

    m.Xw = Var(bounds=(0, None), doc="Moles per second reacted in WGS reactor")
    m.wgs_inlet_temperature = Var(
        bounds=(0, None), doc="WGS reactor temperature before adjustment [ºC]"
    )
    m.wgs_heater = Var(
        bounds=(0, None),
        doc="duty required to preheat the syngas molar flow entering the WGS reactor [kW]",
    )
    m.psa_power = Var(bounds=(0, None), doc="power consumed by the PSA [kWh]")
    m.syngas_power = Var(
        bounds=(0, None), doc="power needed to compress syngas from Pinlet to 30.01 bar"
    )

    m.syngas_total_flow = Expression(
        expr=sum(
            m.final_syngas_flow[species] for species in {'H2', 'CO2', 'CO', 'CH4'}
        ),
        doc="total molar flow of syngas comonents at final stage of process [kmol·s-1]",
    )

    @m.Expression(m.aux_equipment)
    def aux_module_factors(m, equip):
        """
        Calculate the module factors for each type of auxiliary equipment used in the syngas production process.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        equip : str
            Index of the equipment type (e.g., compressor, exchanger, reformer) begin modeled.

        Returns
        -------
        Pyomo.Expression
            A Pyomo Expression that computes the module factor for the specified type of equipment. The calculation uses the base module parameter (B1), enhanced by the product of the second base module parameter (B2) and the material factor for the equipment.
        """
        return m.B1[equip] + m.B2[equip] * m.material_factor[equip]

    m.final_total_emissions = Expression(
        expr=m.syngas_tech_emissions * 3600
        + m.Fabs1 * 3600 * 44
        + m.Fabs2 * 3600 * 44
        - m.co2_inject * 3600 * 44
        + m.oxygen_flow * m.raw_material_emission['O2'] * 3600
        + m.wgs_steam * m.raw_material_emission['H2O'] * 3600
        + m.purge_flow['CO2'] * 3600 * 44
        + (
            m.psa_power
            + m.syngas_power
            + sum(m.syngas_tech_compressor_power[tech] for tech in m.syngas_techs)
        )
        * m.utility_emission['power']
        + m.wgs_heater * 0.094316389565193,
        doc="total estimated emissions from the syngas process, combining CO2 and power-related outputs, adjusted for sequestration and purification [kgCO2·h-1]",
    )

    m.final_total_cost = Expression(
        expr=m.syngas_tech_cost * 3600
        + m.syngas_tech_compressor_cost
        + sum(m.aux_unit_capital_cost[option] for option in m.aux_equipment)
        * m.annualization_factor
        + (
            m.psa_power
            + m.syngas_power
            + sum(m.syngas_tech_compressor_power[tech] for tech in m.syngas_techs)
        )
        * m.utility_cost['power']
        + m.wgs_heater * 0.064,
        doc="total cost of syngas production, including capital and operating expenses [$·h-1]",
    )

    """
    Constraints
    """

    @m.Constraint(m.syngas_techs, m.species)
    def syngas_process_feed_species_ratio(m, tech, species):
        """
        Enforce the stoichiometric feeding ratio of various chemical species relative to methane (CH4) for different syngas technologies.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        tech : str
            Index of the syngas process technology (e.g., SMR, POX, ATR).
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint or Constraint.Skip
            A constraint object that sets the flow of `species` in relation to methane flow according to specified ratios. The constraint is skipped if the species is methane.
        """
        if species == 'CH4':
            return Constraint.Skip
        return (
            m.flow['in', tech, species]
            == feed_ratios.get((tech, species), 0) * m.flow['in', tech, 'CH4']
        )

    @m.Constraint(m.syngas_techs, m.species)
    def syngas_conversion_calc(m, tech, species):
        """
        Defines the conversion rates for each species within a specified syngas technology at the mixer stage 'ms1'.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        tech : str
            Index of the syngas process technology (e.g., SMR, POX, ATR).
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Sets the flow of `species` in the mixer/splitter (ms1) to the product of incoming methane flow and the species-specific conversion factor.

        Notes
        -----
        This constraint ensures the output flow for each species is proportional to the methane input flow, scaled by the conversion factor specific to that species and technology.
        """
        return (
            m.flow[tech, 'ms1', species]
            == m.flow['in', tech, 'CH4'] * m.syngas_conversion_factor[species, tech]
        )

    m.raw_material_cost_calc = Constraint(
        expr=m.raw_material_total_cost
        == (
            sum(
                m.raw_material_flow[species] * m.raw_material_cost[species]
                for species in m.species
            )
            + m.wgs_steam * m.raw_material_cost['H2O']
            + m.oxygen_flow * m.raw_material_cost['O2']
        ),
        doc="total cost of raw materials [$·h-1]",
    )

    @m.Disjunct(m.process_options)
    def unit_exists(disj, option):
        """
        Represents the scenario where a specific process unit (identified by 'option') is operational within the syngas production system.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            _description_
        option : str
            The identifier for the process unit being considered. This could be any unit within the set 'm.process_options'.
        """
        pass

    @m.Disjunct(m.process_options)
    def unit_absent(no_unit, option):
        """
        Represents the scenario where a specific process unit (identified by 'option') is not operational.

        Parameters
        ----------
        no_unit : Pyomo.Disjunct
            The disjunct instance representing the absence of the unit.
        option : str
            The identifier for the process unit being considered. This could be any unit within the set 'm.process_options'.

        """

        @no_unit.Constraint(m.species)
        def no_flow_in(disj, species):
            """
            Ensures zero inflow of the specified species into the non-operational unit.

            Parameters
            ----------
            disj : Pyomo.Disjunct
                _description_
            species : str
                The index of chemical species which is the element of the set 'm.species'.

            Returns
            -------
            Pyomo.Constraint
                A constraint that forces the flow into the unit for this species to be zero.
            """
            return m.flow_into[option, species] == 0

        @no_unit.Constraint(m.species)
        def no_flow_out(disj, species):
            """
            Ensures zero inflow of the specified species into the non-operational unit.

            Parameters
            ----------
            disj : Pyomo.Disjunct
                _description_
            species : str
                The index of chemical species which is the element of the set 'm.species'.

            Returns
            -------
            Pyomo.Constraint
                A constraint that forces the flow into the unit for this species to be zero.
            """
            return m.flow_out_from[option, species] == 0

    @m.Disjunction(m.process_options)
    def unit_exists_or_not(disj, option):
        """
        Defines a disjunction that determines the operational status of a process unit within the syngas production system.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct instance that encapsulates this logical decision-making process.
        option : str
            The identifier for the process unit being considered. This could be any unit within the set 'm.process_options'.

        Returns
        -------
        Pyomo.Disjunction
             A disjunction object within Pyomo that forces the model to select either the active or inactive state for the specified unit.
        """
        return [m.unit_exists[option], m.unit_absent[option]]

    m.Yunit = BooleanVar(
        m.process_options, doc="Boolean variable for existence of a process unit"
    )
    for option in m.process_options:
        m.Yunit[option].associate_binary_var(m.unit_exists[option].binary_indicator_var)

    for tech in m.syngas_techs:
        # Capital costs -->  cap = (p1*x + p2)*(B1 + B2*Fmf*Fp)
        tech_selected = m.unit_exists[tech]

        @tech_selected.Constraint(m.syngas_tech_units)
        def base_tech_capital_cost_calc(disj, equip):
            """
            Calculates the capital cost for each syngas technology and its associated equipment.

            Parameters
            ----------
            disj : Pyomo.Disjunct
                Represents the scenario where the specified technology is active.
            equip : str
                Index of the equipment type (e.g., compressor, exchanger, reformer) begin modeled.

            Returns
            -------
            Pyomo.Constraint
                Sets the capital cost for the equipment within a technology, considering utilization rates, number of units, and adjustment factors to reflect accurate cost modeling in [$·h-1].
            """
            return m.base_tech_capital_cost[tech, equip] == (
                (
                    m.p1[equip] * m.variable_utilization[tech, equip]
                    + m.p2[equip]
                    * m.syngas_tech_num_units[equip, tech]
                    * m.module_factors[tech, equip]
                )
                * m.cost_index_ratio
                / 8000
            )

        @tech_selected.Constraint(m.utilities)
        def base_tech_operating_cost_calc(disj, util):
            """
            Calculates the operating cost for each utility used by a specific syngas technology based on the methane flow rate, utility usage rates, and cost per utility unit.

            Parameters
            ----------
            disj : Pyomo.Disjunct
                Represents the scenario where the specified technology is active.
            util : str
                Index of the utility (e.g., power, cooling water, natural gas) being considered.

            Returns
            -------
            Pyomo.Constraint
                Sets the operating cost for each utility in a technology and integrated into the overall syngas production cost model. The unit is in [$·h-1].
            """
            return m.base_tech_operating_cost[tech, util] == (
                m.syngas_tech_utility_rate[util, tech]
                * m.flow['in', tech, 'CH4']
                * 3600
                * m.utility_cost[util]
            )

    m.syngas_process_cost_calc = Constraint(
        expr=m.syngas_tech_cost
        == (
            sum(
                sum(
                    m.base_tech_capital_cost[tech, equip]
                    for equip in m.syngas_tech_units
                )
                * m.annualization_factor
                + sum(m.base_tech_operating_cost[tech, util] for util in m.utilities)
                for tech in m.syngas_techs
            )
            + m.raw_material_total_cost * 3600
        )
        / 3600,
        doc="total cost of the syngas production process [$·s-1].",
    )

    m.syngas_emissions_calc = Constraint(
        expr=m.syngas_tech_emissions
        == (
            # Emissions from utilities
            sum(
                m.base_tech_operating_cost[tech, util]
                / m.utility_cost[util]
                * m.utility_emission[util]
                for tech in m.syngas_techs
                for util in m.utilities
            )
            # CO2 consumed by syngas processes
            - sum(m.flow['in', tech, 'CO2'] for tech in m.syngas_techs) * 3600 * 44
            # Emissions from raw materials
            + sum(
                m.raw_material_flow[species] * m.raw_material_emission[species] * 3600
                for species in m.species
            )
        )
        / 3600,
        doc="total syngas process emissions with various factors [kgCO2·s-1].",
    )

    # Syngas process pressure adjustment

    @m.Disjunct(m.syngas_techs)
    def stage_one_compressor(disj, tech):
        """
        Defines the operational constraints for the compressor in stage one of the syngas process, calculating the required compressor power based on process pressures and species flow rates.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct instance for the compressor, indicating its active status in the model.
        tech : str
            Index of the syngas process technology (e.g., SMR, POX, ATR).
        """
        disj.compressor_power_calc = Constraint(
            expr=m.syngas_tech_compressor_power[tech]
            == (
                (1.5 / (1.5 - 1))
                / 0.8
                * (40 + 273)
                * 8.314
                * sum(m.flow[tech, 'ms1', species] for species in m.species)
                * (
                    (
                        m.syngas_tech_outlet_pressure[tech]
                        / m.process_tech_pressure[tech]
                    )
                    ** (1.5 - 1 / 1.5)
                    - 1
                )
            ),
            doc="compressor power requirement based on the ideal gas law and the pressure ratio needed for the syngas technology [kW]",
        )
        pass

    @m.Disjunct(m.syngas_techs)
    def stage_one_bypass(bypass, tech):
        """
        Represents a bypass scenario where no compressor is used, and the outlet pressure of the syngas is set directly to the process technology pressure without any increase.

        Parameters
        ----------
        bypass : Pyomo.Disjunct
            The disjunct instance for the bypass of the compressor, indicating its inactive status in the model.
        tech : str
            Index of the syngas process technology (e.g., SMR, POX, ATR).
        """
        bypass.no_pressure_increase = Constraint(
            expr=m.syngas_tech_outlet_pressure[tech] == m.process_tech_pressure[tech],
            doc="outlet pressure is the same as the process pressure",
        )
        pass

    @m.Disjunction(m.syngas_techs)
    def stage_one_compressor_or_bypass(m, tech):
        """
        Logical disjunction that decides whether a compressor is used or bypassed in the first stage of the syngas process based on the specified technology.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        tech : str
            Index of the syngas process technology (e.g., SMR, POX, ATR).

        Returns
        -------
        Pyomo.Disjunction
            A choice between using a compressor (stage_one_compressor) or bypassing it (stage_one_bypass).
        """
        return [m.stage_one_compressor[tech], m.stage_one_bypass[tech]]

    m.Ycomp = BooleanVar(
        m.syngas_techs,
        doc="indicates if a compressor is active for each syngas technology",
    )
    for tech in m.syngas_techs:
        m.Ycomp[tech].associate_binary_var(
            m.stage_one_compressor[tech].binary_indicator_var
        )

    @m.LogicalConstraint(m.syngas_techs)
    def compressor_implies_tech(m, tech):
        """
        Ensures that if a compressor is active for a given syngas technology 'tech', then the technology must also be active.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        tech : str
            Index of the syngas process technology (e.g., SMR, POX, ATR).

        Returns
        -------
        Pyomo.LogicalConstraint
            A logical constraint that enforces the operational dependency between a compressor and its syngas technology.
        """
        return m.Ycomp[tech].implies(m.Yunit[tech])

    m.syngas_tech_compressor_cost_calc = Constraint(
        expr=m.syngas_tech_compressor_cost
        == (
            (
                (3.553 * 10**5)
                * sum(m.Ycomp[tech].get_associated_binary() for tech in m.syngas_techs)
                + 586
                * sum(m.syngas_tech_compressor_power[tech] for tech in m.syngas_techs)
            )
            * m.annualization_factor
            / 8000
        ),
        doc="total cost of compressors across all syngas technologies based on their power usage and operational status [$·h-1]",
    )

    for tech in m.syngas_techs:
        tech_selected = m.unit_exists[tech]
        tech_selected.pressure_balance = Constraint(
            expr=m.first_stage_outlet_pressure <= m.syngas_tech_outlet_pressure[tech],
            doc="first stage outlet pressure does not exceed syngas technology 'tech' limits",
        )

    # ms1 balances
    @m.Constraint(m.species)
    def ms1_mass_balance(m, species):
        """
        Sets mass balance for each chemical species at the mixer/splitter (ms1) by equating the total inflow to the total outflow.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            A constraint that maintains mass balance at the mixer for each species
        """
        return m.flow_into['ms1', species] == m.flow_out_from['ms1', species]

    for this_unit in {'absorber1', 'WGS'}:
        unit_exists = m.unit_exists[this_unit]

        @unit_exists.Constraint(m.species)
        def unit_inlet_composition_balance(disj, species):
            """
            Ensures that the proportion of each species entering the unit matches its proportion at the mixer outlet, maintaining compositional consistency across the process.

            Parameters
            ----------
            disj : Pyomo.Disjunct
                The disjunct under which this constraint is active, indicating that the unit is operational.
            species : str
                The index of chemical species which is the element of the set 'm.species'.

            Returns
            -------
            Pyomo.Constraint
                Balances the composition of each species entering the unit relative to its total flow from the mixer.
            """
            total_flow = sum(
                m.flow_out_from['ms1', jj] for jj in m.species
            )  # Total flow of all species from the mixer (ms1)
            total_flow_to_this_unit = sum(
                m.flow_into[this_unit, jj] for jj in m.species
            )  # Total incoming flow to this unit
            total_flow_species = m.flow_out_from[
                'ms1', species
            ]  # Flow of the current species from the mixer
            return (
                total_flow * m.flow['ms1', this_unit, species]
                == total_flow_to_this_unit * total_flow_species
            )  # Ensure the flow ratios into the unit for each species matches its ratio in the total mixer output

    # WGS Reactor  CO + H2O <-> CO2 + H2
    wgs_exists = m.unit_exists['WGS']
    wgs_exists.CH4_balance = Constraint(
        expr=m.flow_out_from['WGS', 'CH4'] == m.flow_into['WGS', 'CH4'],
        doc="methane in equals methane out",
    )
    wgs_exists.CO_balance = Constraint(
        expr=m.flow_out_from['WGS', 'CO'] == m.flow_into['WGS', 'CO'] - m.Xw,
        doc="CO out equals CO in minus reacted CO",
    )
    wgs_exists.CO2_balance = Constraint(
        expr=m.flow_out_from['WGS', 'CO2'] == m.flow_into['WGS', 'CO2'] + m.Xw,
        doc="CO2 out equals CO2 in plus produced CO2",
    )
    wgs_exists.H2_balance = Constraint(
        expr=m.flow_out_from['WGS', 'H2'] == m.flow_into['WGS', 'H2'] + m.Xw,
        doc="H2 out equals H2 in plus produced H2",
    )
    wgs_exists.H2O_balance = Constraint(
        expr=m.flow_out_from['WGS', 'H2O']
        == m.flow_into['WGS', 'H2O'] + m.wgs_steam - m.Xw,
        doc="water balance including steam input and water reacted",
    )
    wgs_exists.max_molar_reaction = Constraint(
        expr=m.Xw <= m.flow_into['WGS', 'CO'],
        doc="reaction does not exceed available CO",
    )

    wgs_exists.rxn_equilibrium = Constraint(
        expr=m.Keqw * m.flow_out_from['WGS', 'CO'] * m.flow_out_from['WGS', 'H2O']
        == (m.flow_out_from['WGS', 'CO2'] * m.flow_out_from['WGS', 'H2']),
        doc="maintenance of chemical equilibrium for WGS reaction",
    )

    wgs_exists.capital_cost = Constraint(
        expr=m.aux_unit_capital_cost['WGS']
        == (
            (m.p1['WGS'] * 100 * m.flow_out_from['WGS', 'H2'] + m.p2['WGS'])
            * m.aux_module_factors['WGS']
            / 8000
            * m.cost_index_ratio
        ),
        doc="capital cost for the WGS unit based on output and configuration [$·h-1]",
    )

    wgs_exists.temperature_balance = Constraint(
        expr=m.wgs_inlet_temperature * m.total_flow_into['WGS']
        == (
            sum(
                m.flow[tech, 'ms1', species]
                for tech in m.syngas_techs
                for species in m.species
            )
            * 250
            + sum(m.flow['s2', 'ms1', species] for species in m.species) * 40
        )
    )
    wgs_exists.heater_duty = Constraint(
        expr=m.wgs_heater
        == m.total_flow_into['WGS'] * 46 * (250 - m.wgs_inlet_temperature),
        doc="heater duty required to reach the target reactor inlet temperature",
    )

    # Bypass 1
    bypass1_exists = m.unit_exists['bypass1']

    @bypass1_exists.Constraint(m.species)
    def bypass1_mass_balance(disj, species):
        """
        Sets that the flow of each chemical species into and out of the bypass1 is balanced, indicating no accumulation or loss.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct that absorber and WGS are both inactive, there are no units operational
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            A constraint that ensures mass conservation for each species through bypass1.
        """
        return m.flow_into['bypass1', species] == m.flow_out_from['bypass1', species]

    # Absorber
    absorber_exists = m.unit_exists['absorber1']

    @absorber_exists.Constraint(m.species)
    def absorber_mass_balance(disj, species):
        """
        Set mass balance in the absorber by accounting for CO2 absorption.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            Indicates that this constraint is relevant when the absorber is operational.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            A constraint ensuring mass conservation within the absorber.
        """
        return m.flow_out_from['absorber1', species] == m.flow_into[
            'absorber1', species
        ] - (m.Fabs1 if species == 'CO2' else 0)

    absorber_exists.co2_absorption = Constraint(
        expr=m.Fabs1 == 0.96 * m.flow_into['absorber1', 'CO2'],
        doc="the CO2 absorption rate as 96% of the incoming CO2 to the absorber",
    )
    absorber_exists.cost = Constraint(
        expr=m.aux_unit_capital_cost['absorber1']
        == (
            (m.p1['absorber1'] * 100 * m.Fabs1 + m.p2['absorber1'])
            * m.aux_module_factors['absorber1']
            / 8000
            * m.cost_index_ratio
        ),
        doc="the capital cost for the absorber based on CO2 absorbed [$·h-1]",
    )

    m.unit_absent['absorber1'].no_absorption = Constraint(expr=m.Fabs1 == 0)

    for this_unit in group1:
        unit_exists = m.unit_exists[this_unit]
        unit_exists.minimum_flow = Constraint(
            expr=m.total_flow_into[this_unit]
            >= m.total_flow_from['ms1'] * m.min_flow_division,
            doc="minimum flow into each unit for operational stability",
        )

    # Flash separator
    @m.Constraint(m.species)
    def m1_mass_balance(m, species):
        """
        Ensures total mass balance at the mixer 'm1' node for each chemical species, meaning all mass entering must equal all mass leaving the mixer.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            A Pyomo constraint ensures the mass conservation at mixer 'm1' for each chemical species
        """
        return m.flow_into['m1', species] == m.flow_out_from['m1', species]

    flash_exists = m.unit_exists['flash']

    @flash_exists.Constraint(m.species)
    def flash_mass_balance(disj, species):
        """
        Ensures mass balance for each species through the flash unit, excluding water which is separated out.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            A Pyomo disjunct which indicates the mass balance is relevant when the flash unit is operational.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            A constraint that ensures all species except water have balanced mass flows in and out of the flash unit.
        """
        return m.flow_out_from['flash', species] == (
            m.flow_into['flash', species] if not species == 'H2O' else 0
        )

    flash_exists.water_sep = Constraint(
        expr=m.flash_water == m.flow_into['flash', 'H2O'],
        doc="Captures the total water separated out in the flash process.",
    )

    @m.Constraint(m.species)
    def post_flash_split_outlet(m, species):
        """
        Balances the distribution of each species from the flash unit to the PSA and mixer(ms2), ensuring flow continuity.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Ensures that the sum of species flows to PSA and mixer2(ms2) equals the output from the flash unit.
        """
        return (
            m.flow_out_from['flash', species]
            == m.flow_into['PSA', species] + m.flow_into['ms2', species]
        )

    flash_exists.cost = Constraint(
        expr=m.aux_unit_capital_cost['flash']
        == (
            (m.p1['flash'] * 100 * m.flash_water + m.p2['flash'])
            * m.aux_module_factors['flash']
            / 8000
            * m.cost_index_ratio
        ),
        doc="the capital cost for the flash unit based on the amount of water separated [$·h-1]",
    )

    # PSA(pressure swing adsorption)
    psa_exists = m.unit_exists['PSA']

    @psa_exists.Constraint(m.species)
    def psa_inlet_composition_balance(disj, species):
        """
        Ensures the compositional ratios at the PSA's inlet mirror those at the output of splitter 's1', maintaining consistency in feed composition.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            A Pyomo disjunct which represent the active constraint when the PSA unit is operational.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Balances feed ratios entering the PSA to uphold system integrity and performance accuracy.
        """
        total_flow = sum(m.flow_out_from['s1', jj] for jj in m.species)
        total_flow_to_this_unit = sum(m.flow_into['PSA', jj] for jj in m.species)
        total_flow_species = m.flow_out_from['s1', species]
        return (
            total_flow * m.flow['s1', 'PSA', species]
            == total_flow_to_this_unit * total_flow_species
        )

    @m.Constraint(m.species)
    def ms2_inlet_composition_balance(disj, species):
        """
        Ensures the compositional ratios at the mixer 'ms2's inlet align with those from the splitter 's1', for consistent mixture preparation.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            A Pyomo disjunct which can be applied, when consider
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Maintains consistent inlet compositions to the mixer 'ms2' from the splitter 's1'.
        """
        total_flow = sum(m.flow_out_from['s1', jj] for jj in m.species)
        total_flow_to_this_unit = sum(m.flow_into['ms2', jj] for jj in m.species)
        total_flow_species = m.flow_out_from['s1', species]
        return (
            total_flow * m.flow['s1', 'ms2', species]
            == total_flow_to_this_unit * total_flow_species
        )

    @psa_exists.Constraint(m.species)
    def psa_mass_balance(disj, species):
        """
        Ensures that all species entering the PSA are either processed inside or exit as part of the recovered stream.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            A Pyomo disjunct indicates operational context for PSA
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Ensures total incoming flow of each species equals the outgoing plus recovered flow in the PSA.
        """
        return (
            m.flow_out_from['PSA', species] + m.psa_recovered[species]
            == m.flow_into['PSA', species]
        )

    @psa_exists.Constraint(m.species)
    def psa_recovery(disj, species):
        """
        Defines the recovery efficiency for hydrogen and purity constraints for other gases in the PSA.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            A Pyomo disjunct indicates operational context for PSA
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Enforces hydrogen recovery rate and purity levels for other gases.
        """
        return m.flow_out_from['PSA', species] == m.flow_into['PSA', species] * (
            (
                1 - m.psa_hydrogen_recovery
                if species == 'H2'
                else m.psa_separation_hydrogen_purity
            )
        )

    psa_exists.cost = Constraint(
        expr=m.aux_unit_capital_cost['PSA']
        == (
            (m.p1['PSA'] * m.flow_into['PSA', 'H2'] + m.p2['PSA'])
            * m.aux_module_factors['PSA']
            / 8000
        ),
        doc="PSA's capital cost based on hydrogen flow and cost parameters [$·h-1]",
    )
    psa_exists.psa_utility = Constraint(
        expr=m.psa_power * m.first_stage_outlet_pressure ** (1.5 - 1 / 1.5)
        == (
            (1.5 / (1.5 - 1))
            / 0.8
            * (40 + 273)
            * 8.314
            * m.total_flow_into['PSA']
            * (
                (30 + 1e-6) ** (1.5 - 1 / 1.5)
                - m.first_stage_outlet_pressure ** (1.5 - 1 / 1.5)
            )
        ),
        doc="power needs for the PSA based on flow and pressure requirements [kW]",
    )

    psa_absent = m.unit_absent['PSA']

    @psa_absent.Constraint(m.species)
    def no_psa_recovery(disj, species):
        """
        Ensures no species recovery when the PSA unit is not operational.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            A Pyomo disjunct where PSA is not in use.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Prevents any unintended recovery flows when PSA is inactive.
        """
        return m.psa_recovered[species] == 0

    @psa_absent.Constraint(m.species)
    def no_purge(disj, species):
        """
        Prevents purge flow of any species when PSA is not operational, avoiding errors in system dynamics.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            A Pyomo disjunct where PSA is not in use.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Blocks any purge flow for species when PSA is deactivated.
        """
        return m.purge_flow[species] == 0

    # ms4
    @m.Constraint(m.species)
    def ms4_inlet_mass_balance(m, species):
        """
        Ensures all species' flow from PSA is equal to their inflow to the mixer (ms4), maintaining continuity.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            A Pyomo Constraint which matches inflow to mixer(ms4) with outflow from PSA for each species.
        """
        return m.flow_out_from['PSA', species] == m.flow_into['ms4', species]

    @m.Constraint(m.species)
    def ms4_outlet_mass_balance(m, species):
        """
        Balances the total outflow from mixer(ms4), including any purge flows, with its inflow, ensuring mass conservation.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Ensures the sum of outflow and purge equals the inflow for the mixer(ms4) across all species.
        """
        return (
            m.flow_out_from['ms4', species] + m.purge_flow[species]
            == m.flow_into['ms4', species]
        )

    @m.Constraint(m.species)
    def purge_flow_limit(m, species):
        """
        Limits the purge flow of each species to no more than 1% of its inflow to the mixer(ms4).

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Constrains the purge flow to a maximum of 1% of the respective species' inflow to the MS4, aiding in maintaining environmental and process control standards.
        """
        return m.purge_flow[species] <= m.flow_into['ms4', species] * 0.01

    @m.Constraint(m.species)
    def s2_inlet_composition(m, species):
        """
        Ensures flow composition into splitter (s2) from mixer(ms4) is proportional to overall flow into splitter.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Proportional flow composition constraint for splitter(s2) inlet.
        """
        total_flow = sum(m.flow_into['ms4', jj] for jj in m.species)
        total_flow_to_this_unit = sum(m.flow_into['s2', jj] for jj in m.species)
        total_flow_species = m.flow_into['ms4', species]
        return (
            total_flow * m.flow['ms4', 's2', species]
            == total_flow_to_this_unit * total_flow_species
        )

    @m.Constraint(m.species)
    def ms4_to_ms3_composition(m, species):
        """
        Balances species flow from mixer(ms4) to splitter(s2) to match total flow ratios.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            A constraint that maintains proportional species distribution between mixer(ms4) and splitter(s2).
        """
        total_flow = sum(m.flow_into['ms4', jj] for jj in m.species)
        total_flow_to_this_unit = sum(m.flow['ms4', 's2', jj] for jj in m.species)
        total_flow_species = m.flow_into['ms4', species]
        return (
            total_flow * m.flow['ms4', 's2', species]
            == total_flow_to_this_unit * total_flow_species
        )

    # s2
    @m.Constraint(m.species)
    def s2_mass_balance(m, species):
        """
        Ensures mass conservation at splitter S2 by equating the total inflow of each species to its outflow.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Enforces that the inflow of each species into splitter(s2) equals the outflow from splitter(s2) to the mixer(ms1).
        """
        return m.flow_into['s2', species] == m.flow['s2', 'ms1', species]

    @m.Constraint(m.species)
    def no_flow_s2_to_m1(m, species):
        """
        Prevents any flow from the splitter(s2) to the mixer(m1).

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Ensures that no species inadvertently flows back from splitter(s2) to mixer(m1), maintaining intended flow paths.
        """
        return m.flow['s2', 'm1', species] == 0

    # ms2
    @m.Constraint(m.species)
    def ms2_mass_balance(m, species):
        """
        Balances the mass of each species at the MS2 unit by accounting for additional CO2 injection where applicable.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            Enforces that the output from MS2 equals the input plus any CO2 injected, maintaining mass conservation with external inputs considered.
        """
        return m.flow_out_from['ms2', species] == (
            m.flow_into['ms2', species] + (m.co2_inject if species == 'CO2' else 0)
        )

    # bypass3
    bypass3_exists = m.unit_exists['bypass3']

    @bypass3_exists.Constraint(m.species)
    def bypass3_mass_balance(disj, species):
        """
        Ensures mass balance for each species through the bypass3 unit, indicating no accumulation or loss.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct which indicates the bypass3 unit is operational.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            A constraint that ensures mass conservation for each species through bypass3.
        """
        return m.flow_into['bypass3', species] == m.flow_out_from['bypass3', species]

    # compressor
    compressor_exists = m.unit_exists['compressor']

    @compressor_exists.Constraint(m.species)
    def compressor_inlet_mass_balance(disj, species):
        """
        Ensures mass balance for each species entering the compressor, indicating the flow into the compressor is equal to the outlet of mixer(ms2).

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct which indicates the compressor is operational.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            A constraint that ensures mass conservation for each species entering the compressor.
        """
        return m.flow_into['compressor', species] == m.flow_out_from['ms2', species]

    @compressor_exists.Constraint(m.species)
    def compressor_mass_balance(disj, species):
        """
        Ensures that the mass of each species leaving the compressor is equal to the mass entering it.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct which indicates the compressor is operational.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            A constraint that ensures mass conservation for each species through the compressor.
        """
        return (
            m.flow_out_from['compressor', species] == m.flow_into['compressor', species]
        )

    compressor_exists.cost = Constraint(
        expr=m.aux_unit_capital_cost['compressor']
        == (((3.553 * 10**5) + 586 * m.syngas_power) / 8000 * m.cost_index_ratio),
        doc="capital cost for the compressor based on power usage and cost parameters [$·h-1]",
    )

    compressor_exists.work = Constraint(
        expr=m.syngas_power * m.first_stage_outlet_pressure ** (1.5 - 1 / 1.5)
        == (
            (1.5 / (1.5 - 1))
            / 0.8
            * (40 + 273)
            * 8.314
            * m.total_flow_into['compressor']
            * (
                m.final_syngas_pressure ** (1.5 - 1 / 1.5)
                - m.first_stage_outlet_pressure ** (1.5 - 1 / 1.5)
            )
        ),
        doc="compressor power requirement based on the ideal gas law and the pressure ratio needed for the syngas technology [kW]",
    )

    no_compressor = m.unit_absent['compressor']
    no_compressor.final_pressure = Constraint(
        expr=m.first_stage_outlet_pressure >= m.final_syngas_pressure,
        doc="final pressure is less than or equal to the process pressure",
    )

    compressor_exists.compressor_minimum_flow = Constraint(
        expr=m.total_flow_into['compressor']
        >= (m.total_flow_from['flash'] * m.min_flow_division),
        doc="minimum flow into the compressor for operational stability",
    )
    psa_exists.psa_minimum_flow = Constraint(
        expr=m.total_flow_into['PSA']
        >= (m.total_flow_from['flash'] * m.min_flow_division),
        doc="minimum flow into the PSA for operational stability",
    )

    m.compressor_or_bypass = LogicalConstraint(
        expr=exactly(1, m.Yunit['bypass3'], m.Yunit['compressor']),
        doc="only one of the compressor or bypass3 can be active",
    )

    # ms3
    @m.Constraint(m.species)
    def ms3_mass_balance(m, species):
        """
        Ensures that the total mass flow into and out of mixer/splitter(ms3) is conserved for each species.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            The pyomo constraint that maintains mass conservation at the mixer/splitter(ms3) for each species.
        """
        return m.flow_into['ms3', species] == m.flow_out_from['ms3', species]

    # bypass 4
    bypass4_exists = m.unit_exists['bypass4']

    @bypass4_exists.Constraint(m.species)
    def bypass4_mass_balance(disj, species):
        """
        Ensures mass balance for each species through the bypass4 unit, indicating no accumulation or loss.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct which indicates the bypass4 unit is operational.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            A constraint that ensures mass conservation for each species through bypass4.
        """
        return m.flow_into['bypass4', species] == m.flow_out_from['bypass4', species]

    # absorber 2
    absorber2_exists = m.unit_exists['absorber2']

    @absorber2_exists.Constraint(m.species)
    def absorber2_mass_balance(disj, species):
        """
        Ensures mass balance in the absorber2 by accounting for CO2 absorption.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct which indicates the absorber2 unit is operational.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            A constraint ensuring mass conservation within the absorber2.
        """
        return m.flow_out_from['absorber2', species] == (
            m.flow_into['absorber2', species] - (m.Fabs2 if species == 'CO2' else 0)
        )

    absorber2_exists.co2_absorption = Constraint(
        expr=m.Fabs2 == 0.96 * m.flow_into['absorber2', 'CO2'],
        doc="CO2 absorption rate as 96% of incoming CO2 to the absorber2",
    )
    absorber2_exists.cost = Constraint(
        expr=m.aux_unit_capital_cost['absorber2']
        == (
            (m.p1['absorber2'] * 100 * m.Fabs2 + m.p2['absorber2'])
            * m.aux_module_factors['absorber2']
            / 8000
            * m.cost_index_ratio
        ),
        doc="capital cost for the absorber2 based on CO2 absorbed [$·h-1]",
    )

    m.unit_absent['absorber2'].no_absorption = Constraint(
        expr=m.Fabs2 == 0, doc="no CO2 absorption when the absorber2 is not operational"
    )

    m.only_one_absorber = LogicalConstraint(
        expr=atmost(1, m.Yunit['absorber1'], m.Yunit['absorber2']),
        doc="only one absorber can be active",
    )

    @m.Constraint(m.species)
    def final_mass_balance(m, species):
        """
        Ensures that the total mass flow into mixer(m2) and out of the final syngas output is conserved for each species.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The concrete model instance which contains sets, parameters, and variables that define the syngas production process.
        species : str
            The index of chemical species which is the element of the set 'm.species'.

        Returns
        -------
        Pyomo.Constraint
            The pyomo constraint that maintains mass conservation at the final syngas output for each species.
        """
        return m.final_syngas_flow[species] == m.flow_into['m2', species]

    m.syngas_stoich_number = Constraint(
        expr=m.stoichiometric_number
        * (m.final_syngas_flow['CO'] + m.final_syngas_flow['CO2'])
        == (m.final_syngas_flow['H2'] - m.final_syngas_flow['CO2']),
        doc="stoichiometric number for syngas production",
    )
    m.impurity_limit = Constraint(
        expr=m.max_impurity
        * sum(
            m.final_syngas_flow[species]
            for species in {'CO', 'H2', 'CH4', 'CO2', 'H2O'}
        )
        >= sum(m.final_syngas_flow[species] for species in {'CH4', 'H2O'}),
        doc="maximum impurity limit for syngas production",
    )

    m.syngas_process_limits = LogicalConstraint(
        expr=atmost(m.max_syngas_techs, [m.Yunit[tech] for tech in m.syngas_techs]),
        doc="at most 1 syngas technologies can be active",
    )

    # Bounds
    m.wgs_heater.setub(10000)
    m.syngas_tech_compressor_power.setub(10000)
    m.syngas_tech_compressor_cost.setub(10000)
    m.psa_power.setub(10000)

    m.Fabs1.setub(20)
    m.Fabs2.setub(20)
    m.flash_water.setub(20)

    m.psa_recovered.setub(20)
    m.purge_flow.setub(20)
    m.co2_inject.setub(20)
    m.final_syngas_flow.setub(20)
    m.syngas_power.setub(10000)

    m.raw_material_total_cost.setub(1000)
    m.base_tech_capital_cost.setub(10000)
    m.base_tech_operating_cost.setub(10000)
    m.syngas_tech_cost.setub(1000)
    m.syngas_tech_emissions.setub(100)
    m.syngas_tech_emissions.setlb(-100)

    m.Xw.setub(30)
    m.wgs_inlet_temperature.setub(250)

    m.syngas_tech_outlet_pressure.setub(100)
    m.first_stage_outlet_pressure.setlb(1)
    m.first_stage_outlet_pressure.setub(m.final_syngas_pressure)

    m.wgs_steam.setub(20)
    m.oxygen_flow.setub(0)
    # Not done yet:
    # FPSA.up(j) = 20;
    # FPSAmain.up(j) = 20;
    m.flow['ms4', 's2', :].setub(20)
    m.flow['s2', 'ms1', :].setub(20)
    m.flow['ms4', 'ms3', :].setub(20)
    m.flow['psa', 'ms4', :].setub(20)

    m.no_downstream_oxygen = Constraint(
        expr=m.flow_out_from['ms1', 'O2'] == 0, doc="no oxygen in downstream processes"
    )

    # No oxygen in downstream processes: not enforced yet

    m.aux_unit_capital_cost.setub(10000)

    m.always_use_flash = LogicalConstraint(expr=m.Yunit['flash'])

    m.syngas_minimum_demand = Constraint(
        expr=m.syngas_total_flow >= 0.3, doc="minimum syngas flow for demand"
    )
    m.syngas_maximum_demand = Constraint(
        expr=m.syngas_total_flow <= 5, doc="maximum syngas flow for demand"
    )

    m.final_syngas_flow['CO'].fix(0.3)
    m.final_syngas_flow['CO2'].fix(0.3 * m.co2_ratio)

    m.obj = Objective(
        expr=m.final_total_cost, doc="total cost of the syngas production process"
    )

    # # Attempting to troubleshoot:
    # m.optimal_feed = Constraint(expr=m.flow_out_from['in', 'CH4'] == 0.1674)
    # m.wgs_flow = Constraint(
    #     expr=(0.00991008 - 1e-5, m.flow['ms1', 'WGS', 'CH4'], 0.00991008 + 1e-5),
    #     doc="WGS flow",
    # )
    # m.wgs_conv = Constraint(
    #     expr=(0.0201 - 1e-4, m.Xw, 0.0201 + 1e-4), doc="WGS conversion"
    # )
    # m.use_dmr = LogicalConstraint(
    #     expr=m.Yunit['DMR']
    #     & m.Yunit['WGS']
    #     & ~m.Yunit['bypass1']
    #     & ~m.Yunit['absorber1']
    #     & ~m.Yunit['PSA']
    #     & m.Yunit['absorber2']
    #     & m.Yunit['bypass4']
    #     & m.Yunit['bypass3'],
    #     doc="use DMR, WGS, absorber2, bypass4, and bypass3",
    # )
    # psa_exists.psa_minimum_flow.deactivate()
    # m.flow['s1', 'PSA', 'CO'].fix(0)
    # m.post_flash_split_outlet.deactivate()

    return m


def display_nonzeros(var):
    """
    Display non-zero values of a Pyomo variable. This function filters and prints only those variables that have a non-zero value, which is especially useful in large-scale optimization models to quickly identify variables of interest.

    Parameters
    ----------
    var : Pyomo.Var
        The Pyomo variable for which non-zero values are to be displayed.

    Yields
    ------
    tuple or None
        A tuple containing the variable index and its corresponding value, lower bound, upper bound, fixed status, stale status, and domain. If the variable has a non-zero value, the tuple is yielded; otherwise, None is returned.
    """
    if var.is_indexed():

        def nonzero_rows():
            """
            Yields the non-zero values of the variable.

            Yields
            ------
            tuple
                A tuple containing the variable index and its corresponding value, lower bound, upper bound, fixed status, stale status, and domain.
            """
            for k, v in var.items():
                if v.value == 0:
                    continue
                yield k, [value(v.lb), v.value, value(v.ub), v.fixed, v.stale, v.domain]

        _attr, _, _header, _ = var._pprint()
        var._pprint_base_impl(
            None,
            False,
            "",
            var.local_name,
            var.doc,
            var.is_constructed(),
            _attr,
            nonzero_rows(),
            _header,
            lambda k, v: v,
        )
    else:
        var.display()


if __name__ == "__main__":
    m = build_model()
    TransformationFactory('core.logical_to_linear').apply_to(m)
    # TransformationFactory('gdp.bigm').apply_to(m)
    # result = SolverFactory('gams').solve(
    #     m, tee=True, keepfiles=True, symbolic_solver_labels=True,
    #     solver='scip', add_options=['option optcr=0;'],
    #     # solver="cplex", add_options=['GAMS_MODEL.optfile=1;', '$onecho > cplex.opt', 'iis=1', '$offecho'],
    # )
    result = SolverFactory('gdpopt').solve(
        m,
        strategy='LOA',
        tee=True,
        mip_solver='gams',
        nlp_solver='gams',
        nlp_solver_args=dict(solver='scip', add_options=['option optcr=0;']),
        minlp_solver='gams',
        minlp_solver_args=dict(solver='baron', add_options=['option optcr=0;']),
    )
    if not result.solver.termination_condition == tc.optimal:
        print("Termination Condition: ", result.solver.termination_condition)
    else:
        update_boolean_vars_from_binary(m)
        display_nonzeros(m.flow)
        # m.flow.display()
        m.Yunit.display()
        m.final_total_cost.display()
        m.final_total_emissions.display()
