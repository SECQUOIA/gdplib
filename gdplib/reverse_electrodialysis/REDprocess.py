"""Reverse Electrodialysis (RED) Process Model
This module contains the Generalized Disjunctive Programming (GDP) model of the RED process.

The decision variables are:
Continuous:
- Volumetric flow rate [m3 h-1] of the high and low salinity streams
- Molar concentration [mol L-1] of the high and low salinity streams
- Electric current [A] of the RED stack
Discrete:
- Active RED stacks
- Distribution of the high and low salinity streams.

The objective function is:
- Maximize the net present value (NPV) of the RED process.

The constraints are:
- Mass balance constraints
- Energy balance constraints
- Power balance constraints

Attributes
----------
financial_param : DataFrame
    Financial parameters
stack_param : DataFrame
    Stack parameters
flow_conc_data : DataFrame
    Feed flow and concentration data
T : DataFrame
    Feed temperature

Methods
-------
build_model()
    Builds the RED GDP model

Reference
---------
Tristán, C., Fallanza, M., Ibáñez, R., Ortiz, I., & Grossmann, I. E. (2023). A generalized disjunctive programming model for the optimal design of reverse electrodialysis process for salinity gradient-based power generation. Computers & Chemical Engineering, 174, 108196. https://doi.org/https://doi.org/10.1016/j.compchemeng.2023.108196
Tristán, C., Fallanza, M., Ibáñez, R., & Ortiz, I. (2020). Recovery of salinity gradient energy in desalination plants by reverse electrodialysis. Desalination, 496, 114699. https://doi.org/10.1016/j.desal.2020.114699

"""

# Importing libraries
import pyomo.environ as pyo

from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

import scipy.constants as cst

import pyomo.gdp as gdp

import os

import re

import pandas as pd
import numpy as np

from pyomo.contrib.preprocessing.plugins import init_vars

from pyomo.core.expr.logical_expr import *

from .REDstack import build_REDstack

wnd_dir = os.path.dirname(os.path.realpath(__file__))


# The data.xlsx file contains the financial and stack parameters, and properties of the high and low salinity feed streams.
# The financial_param dataframe contains the financial parameters
# The stack_param dataframe contains the stack parameters
# The flow_conc_data dataframe contains the feed flow and concentration data
# The T dataframe contains the feed temperature
with pd.ExcelFile(os.path.join(wnd_dir, "data.xlsx")) as xls:
    financial_param = pd.read_excel(xls, sheet_name="financial_param", dtype=object)
    stack_param = pd.read_excel(xls, sheet_name="stack_param", header=0, dtype=object)
    flow_conc_data = pd.read_excel(
        xls,
        sheet_name="feed_data",
        index_col=0,
        header=[0],
        usecols="A:C",
        dtype=object,
    )
    T = pd.read_excel(xls, sheet_name="feed_data", nrows=1, usecols="D", dtype=object)


def build_model():
    """Builds the RED GDP model
    This function builds the RED GDP model using Pyomo's ConcreteModel class.

    Returns
    -------
    Pyomo.ConcreteModel
        RED GDP model
    """
    # For the data from the data.xlsx file, m_stack is the solution of the RED stack model that maximizes the power output of the RED stack.
    # The resulting gross power output is used to initialize and set the upper bound of the NP variable in the RED GDP model.
    m_stack = build_REDstack()

    m = pyo.ConcreteModel('RED GDP model')

    # ============================================================================
    #     Sets
    # =============================================================================

    m.SOL = pyo.Set(
        doc="High- and Low-concentration streams", initialize=['HC', 'LC'], ordered=True
    )

    m.iem = pyo.Set(doc='Ion-excahnge membrane type', initialize=['AEM', 'CEM'])

    m.nr = pyo.Param(
        doc="Max. # of RED stacks",
        within=pyo.NonNegativeReals,
        default=10,
        initialize=stack_param.nr.values[0],
    )

    m.RU = pyo.Set(
        doc="Set of potential RED units (RU)",
        initialize=['r' + str(nr) for nr in pyo.RangeSet(m.nr)],
    )

    m.in_RU = pyo.Set(
        doc="Inlet RU Port", initialize=['ri' + str(nr) for nr in pyo.RangeSet(m.nr)]
    )

    m.out_RU = pyo.Set(
        doc="Outlet RU Port", initialize=['ro' + str(nr) for nr in pyo.RangeSet(m.nr)]
    )
    m.RU_port = pyo.Set(doc="Inlet and Outlet RU Ports", initialize=m.in_RU | m.out_RU)

    m.units = pyo.Set(
        doc="Superstructure Units",
        initialize=['in', 'fs', 'rsu', 'rmu', 'dm', 'disch'] | m.RU | m.RU_port,
    )

    m.splitters = pyo.Set(
        doc="Set of splitters", within=m.units, initialize=['fs', 'rsu'] | m.out_RU
    )

    m.mixers = pyo.Set(
        doc="Set of mixers", within=m.units, initialize=['dm', 'rmu'] | m.in_RU
    )

    def _streams_filter(m, val):
        """
        This function filters the 1-1 port pairing

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        val : tuple
            Tuple of RMU and RU ports

        Returns
        -------
        Boolean
            True if the ports are the same, False otherwise
        """
        x, y = val
        return re.findall(r'\d+', x) == re.findall(r'\d+', y)

    m.RMU_RU_streams = pyo.Set(
        doc="RMU to RU 1-1 port pairing",
        initialize=m.in_RU * m.RU,
        filter=_streams_filter,
        # filter=lambda _, x, y: re.findall(r'\d+', x) == re.findall(r'\d+', y),
    )  # Filter function _streams_filter as suggested in Pyomo PR #3338 (Support validate / filter for IndexedSet components using the index) that fixes the issue #2655

    m.RU_RSU_streams = pyo.Set(
        doc="RU to RSU 1-1 port pairing",
        initialize=m.RU * m.out_RU,
        filter=_streams_filter,
        # filter=lambda _, x, y: re.findall(r'\d+', x) == re.findall(r'\d+', y),
    )

    m.RU_streams = pyo.Set(
        doc="Set of feasible RU units'streams in RPU",
        initialize=m.RMU_RU_streams | m.out_RU * m.in_RU | m.RU_RSU_streams,
    )

    m.streams = pyo.Set(
        doc="Set of feasible streams",
        initialize=[('in', 'fs'), ('fs', 'rsu'), ('fs', 'dm')]
        | ['rsu'] * m.in_RU
        | m.RU_streams
        | m.out_RU * ['rmu']
        | [('rmu', 'dm')]
        | [('dm', 'disch')],
    )

    def _from_splitters_filter(m, val):
        """
        This function filters the streams from splitters and discharge port

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        val : tuple
            Tuple of streams

        Returns
        -------
        Boolean
            True if the stream is from splitters and discharge port, False otherwise
        """
        x, y = val
        return x in m.splitters or (x, y) == ('dm', 'disch')

    m.from_splitters = pyo.Set(
        doc='Set of streams from splitters and discharge port',
        within=m.streams,
        initialize=m.streams,
        filter=_from_splitters_filter,
        # filter=lambda _, x, y: x in m.splitters or (x, y) == ('dm', 'disch'),
    )  # Filter function _from_splitters_filter as suggested in Pyomo PR #3338 (Support validate / filter for IndexedSet components using the index) that fixes the issue #2655

    def _to_splitters_filter(m, val):
        """
        This function filters the streams to splitters and feed port

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        val : tuple
            Tuple of streams

        Returns
        -------
        Boolean
            True if the stream is to splitters and feed port, False otherwise
        """
        x, y = val
        return y in m.splitters or (x, y) == ('in', 'fs')

    m.to_splitters = pyo.Set(
        doc='Set of streams to splitters and feed port',
        within=m.streams,
        initialize=m.streams,
        filter=_to_splitters_filter,
        # filter=lambda _, x, y: y in m.splitters or (x, y) == ('in', 'fs'),
    )  # Filter function _to_splitters_filter as suggested in Pyomo PR #3338 (Support validate / filter for IndexedSet components using the index) that fixes the issue #2655

    # ============================================================================
    #     Constant parameters
    # =============================================================================

    m.gas_constant = cst.physical_constants['molar gas constant'][
        0
    ]  # Ideal gas constant [J mol-1 K-1]
    m.faraday_constant = cst.physical_constants['Faraday constant'][
        0
    ]  # Faraday’s Constant [C mol-1] [A s mol-1]
    m.Tref = 298.15  # Reference temperature [K]

    m.T = pyo.Param(
        doc='Feed streams temperature [K]', initialize=T.loc[0].values[0], mutable=True
    )

    m.pump_eff = pyo.Param(doc='Pump efficiency [-]', default=0.75, initialize=0.75)

    # =============================================================================
    #     Financial Parameters
    # =============================================================================

    m.electricity_price = pyo.Param(
        doc="Elctricity price [USD kWh-1]",
        initialize=financial_param.electricity_price.values[0],
        default=0.12,
        # initialize = 0.07, # 0.07–0.12 (EIA) US 2019 annual average prices industrial costumers
        mutable=True,
    )

    m.load_factor = pyo.Param(
        doc='Working hours per year of the RU [h year-1]',
        default=0.9,
        initialize=0.9,
        mutable=True,
    )

    m.interest_rate = pyo.Param(
        doc="annualization index",
        default=0.075,
        initialize=financial_param.interest_rate.values[0],
        mutable=True,
    )

    m.project_years = pyo.Param(
        doc="annualization years",
        default=30,
        initialize=financial_param.project_years.values[0],
        mutable=True,
    )

    m.iem_lifetime = pyo.Param(
        doc="Membrane's life time [years]",
        default=10,
        initialize=financial_param.iem_lifetime.values[0],
        mutable=True,
    )

    m.iems_price = pyo.Param(
        doc="Specific membrane's price per effective area [EUR m-2]",
        default=10,
        initialize=financial_param.iems_price.values[0],
        mutable=True,
    )

    m.CRF = pyo.Expression(
        doc="Capital Recovery Factor [years-1]",  # Given P (principal) get A (Annuity)
        expr=m.interest_rate / (1 - (1 + m.interest_rate) ** (-m.project_years)),
    )

    @m.Expression(doc="IEMs replacement cost equivalent annuity")
    def CRFm(m):
        """IEMs replacement cost equivalent annuity

        A/Fi,N = i / ((i+1)**N - 1) --> Disbursment at LTm year = LTm yearly disbursments at the end of LTm year period

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model

        Returns
        -------
        Pyomo.Expression
            IEMs replacement cost equivalent annuity
        """
        # A/Fi,N = i / ((i+1)**N - 1) --> Disbursment at LTm year = LTm yearly disbursments at the end of LTm year period
        return m.interest_rate / ((m.interest_rate + 1) ** m.iem_lifetime - 1)

    m.CEPCI2019 = pyo.Param(
        initialize=607.5,
        default=607.5,
        doc="Chemical Engineering Plant Cost Index [USD] 2019",
    )
    m.CEPCI2007 = pyo.Param(
        initialize=509.7,
        default=509.7,
        doc="Chemical Engineering Plant Cost Index [USD] 2007",
    )
    m.cost_index_ratio = pyo.Param(
        initialize=m.CEPCI2019 / m.CEPCI2007, doc="cost index ratio"
    )

    m.eur2usd = pyo.Param(
        initialize=1.12, default=1.12, doc='Market Exchange Rate [USD EUR-1]'
    )  # Base year 2019 ECB Data

    # =============================================================================
    #     RED Stack Parameters
    # =============================================================================

    m.b = pyo.Param(
        doc="Channel's width = IEMs [m]",
        within=pyo.NonNegativeReals,
        initialize=stack_param.width.values[0],
        default=0.456,
    )
    m.L = pyo.Param(
        doc="Channel's lenght = IEMs [m]",
        initialize=stack_param.length.values[0],
        default=0.383,
    )
    m.spacer_porosity = pyo.Param(
        m.SOL,
        doc="Spacer's porosity [-]",
        default=0.825,
        initialize=stack_param.spacer_porosity.values[0],
    )
    m.spacer_thickness = pyo.Param(
        m.SOL,
        doc="Channel's thickness = Spacer's thickness [m]",
        default=270e-6,
        initialize=stack_param.spacer_thickness.values[0],
        mutable=True,
    )
    m.cell_pairs = pyo.Param(
        within=pyo.NonNegativeIntegers,
        doc="Number of Cell Pairs [-]",
        default=1e3,
        initialize=stack_param.cell_pairs.values[0],
        mutable=True,
    )

    @m.Param(m.SOL, doc="Channel's hydraulic diameter [m]", mutable=True)
    def dh(m, s):
        """
        This function calculates the hydraulic diameter of the channel

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        s : str
            High or low salinity stream (HC or LC).

        Returns
        -------
        float
            Hydraulic diameter of the channel
        """
        return (
            4
            * m.spacer_porosity[s]
            / (
                2 / m.spacer_thickness[s]
                + (1 - m.spacer_porosity[s]) * 8 / m.spacer_thickness[s]
            )
        )

    @m.Param(doc='Membrane area [m2]')
    def Aiem(m):
        """
        This function calculates the membrane area

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model

        Returns
        -------
        float
            Membrane area
        """
        return m.b * m.L

    @m.Param(doc='Total membrane area per cell pair of the RU [m2]')
    def _total_iem_area(m):
        """
        This function calculates the total membrane area per cell pair of the RU

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model

        Returns
        -------
        float
            Total membrane area per cell pair of the RU
        """
        return m.cell_pairs * m.Aiem

    @m.Param(m.SOL, doc='Cross-sectional area [m2]')
    def _cross_area(m, sol):
        """
        This function calculates the cross-sectional area of the channel

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        sol : str
            High or low salinity stream (HC or LC).

        Returns
        -------
        float
            Cross-sectional area of the channel
        """
        return m.b * m.spacer_thickness[sol] * m.spacer_porosity[sol]

    m.iems_resistance = pyo.Param(
        m.iem,
        doc="Membranes' resistance [ohm m2]",
        within=pyo.NonNegativeReals,
        default={'CEM': 1.8e-4, 'AEM': 0.6e-4},
        initialize={
            'CEM': stack_param.cem_resistance.values[0],
            'AEM': stack_param.aem_resistance.values[0],
        },
    )
    m.iems_permsel = pyo.Param(
        m.iem,
        doc="Membranes' permselectivity [-]",
        within=pyo.NonNegativeReals,
        default={'CEM': 0.97, 'AEM': 0.92},
        initialize={
            'CEM': stack_param.cem_permsel.values[0],
            'AEM': stack_param.aem_permsel.values[0],
        },
    )

    @m.Param(doc="Avg. membranes' permselectivity [-]")
    def iems_permsel_avg(m):
        return sum(m.iems_permsel[iem] for iem in m.iem) / 2

    m.iems_thickness = pyo.Param(
        m.iem,
        doc="Membranes' thickness [m]",
        within=pyo.NonNegativeReals,
        default=50e-6,
        initialize={
            'CEM': stack_param.cem_thickness.values[0],
            'AEM': stack_param.aem_thickness.values[0],
        },
    )

    m.diff_nacl = pyo.Param(
        doc="NaCl Membranes' diffusivity [m2 s-1]",
        within=pyo.NonNegativeReals,
        default=4.52e-12,
        initialize=4.52e-12,
    )

    m.vel_ub = pyo.Param(
        m.SOL,
        doc='Max.linear crossflow velocity [cm s-1]',
        default=3.0,
        #                          initialize=5.0,
        initialize=stack_param.vel_ub.values[0],
    )
    # initialize=3.0)
    m.vel_lb = pyo.Param(
        m.SOL, doc='Min.linear crossflow velocity [cm s-1]', initialize=1.0e-3
    )
    m.vel_init = pyo.Param(
        m.SOL,
        doc='Initial linear crossflow velocity [cm s-1]',
        default=1,
        initialize=stack_param.vel_init.values[0],
    )
    # initialize=1.)

    # =============================================================================
    #     Variables
    # =============================================================================

    def _flow_vol(m, i, k, sol):
        """
        This function initializes the flow rate of the high and low salinity streams

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        i : str
            Stream origin
        k : str
            Stream destination
        sol : str
            High or low salinity stream (HC or LC).

        Returns
        -------
        float
            Flow rate of the high and low salinity streams

        Notes
        -----
        The flow rate of the high and low salinity streams is initialized to the feed flow rate flow_init.
        flow_init is calculated as the product of the initial velocity, the cross-sectional area, and the number of cell pairs.
        If the initial value is greater than the available feed flow rate, the feed flow rate is evenly split between the RED units' high and low salinity streams.
        """
        if (i, k) in (['rsu'] * m.in_RU | m.out_RU * m.in_RU | m.out_RU * ['rmu']):
            flow_init = pyo.value(
                36 * m.vel_init[sol] * m._cross_area[sol] * m.cell_pairs
            )
            #             HC and LC inlet flow rate to RU evenly split
            #             if initial value > available feed flow rate
            if sol == 'HC':
                if flow_init * m.nr > flow_conc_data['feed_flow_vol']['fh1']:
                    return flow_conc_data['feed_flow_vol']['fh1'] / m.nr
                return flow_init
            if sol == 'LC':
                if flow_init * m.nr > flow_conc_data['feed_flow_vol']['fl1']:
                    return flow_conc_data['feed_flow_vol']['fl1'] / m.nr
                #                 if flow_init * m.nr > feed_flow_vol['fl1']:
                #                     return feed_flow_vol['fl1'] / m.nr
                return flow_init

    def _flow_vol_b(m, i, k, sol):
        """
        This function bounds the flow rate of the high and low salinity streams to the available feed flow rate

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        i : str
            Stream origin
        k : str
            Stream destination
        sol : str
            High or low salinity stream (HC or LC).

        Returns
        -------
        tuple
            Lower and upper bounds of the flow rate of the high and low salinity streams

        Notes
        -----
        The flow rate of the high and low salinity streams is bounded to the available feed flow rate.
        If the flow rate is greater than the available feed flow rate, the flow rate is bounded to the available feed flow rate.
        The RED units upper bound flow rate is equal to the maximum flow rate within the RED unit channels, ub.
        ub is calculated as the product of the maximum velocity, the cross-sectional area, and the number of cell pairs.
        """
        # lb = pyo.value(36 * m.vel_lb[sol] * m._cross_area[sol] * m.cell_pairs)
        ub = pyo.value(36 * m.vel_ub[sol] * m._cross_area[sol] * m.cell_pairs)
        if (i, k) in (m.out_RU * m.in_RU):
            return (None, ub)
        elif (i, k) in (['rsu'] * m.in_RU | m.out_RU * ['rmu']):
            if sol == 'HC':
                if ub > flow_conc_data['feed_flow_vol']['fh1']:
                    return (None, flow_conc_data['feed_flow_vol']['fh1'])
                #                 if ub > feed_flow_vol['fh1']:
                #                     return (None, feed_flow_vol['fh1'])
                return (None, ub)
            else:
                if ub > flow_conc_data['feed_flow_vol']['fl1']:
                    return (None, flow_conc_data['feed_flow_vol']['fl1'])
                #                 if ub > feed_flow_vol['fl1']:
                #                     return (None, feed_flow_vol['fl1'])
                return (None, ub)
        else:
            if sol == 'HC':
                return (None, flow_conc_data['feed_flow_vol']['fh1'])
        return (None, flow_conc_data['feed_flow_vol']['fl1'])

    m.flow_vol = pyo.Var(
        m.streams - m.RMU_RU_streams - m.RU_RSU_streams,
        m.SOL,
        doc="Volumetric flow rate [m3 h-1]",
        domain=pyo.NonNegativeReals,
        bounds=_flow_vol_b,
        initialize=_flow_vol,
    )

    #    Fixing the feed flow rate of the high and low salinity feed streams.
    m.flow_vol['in', 'fs', 'HC'].fix(flow_conc_data['feed_flow_vol']['fh1'])
    m.flow_vol['in', 'fs', 'LC'].fix(flow_conc_data['feed_flow_vol']['fl1'])

    # Set the flow rate of the high and low salinity feed streams of the discharge port.
    m.flow_vol['dm', 'disch', 'HC'].set_value(flow_conc_data['feed_flow_vol']['fh1'])
    m.flow_vol['dm', 'disch', 'LC'].set_value(flow_conc_data['feed_flow_vol']['fl1'])

    def _flowrate_ratio_b(m):
        """
        This function bounds the flow rate ratio of the high and low salinity streams.
        The flowrate ratio is equal to the flow rate of the low salinity stream divided by the sum of the flow rates of the high and low salinity streams.

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model

        Returns
        -------
        tuple
            Lower and upper bounds of the flow rate ratio of the high and low salinity streams
        """
        lb = m.vel_lb['LC'] / (m.vel_lb['LC'] + m.vel_ub['HC'])
        ub = m.vel_ub['LC'] / (m.vel_ub['LC'] + m.vel_lb['HC'])
        return (lb, ub)

    def _flowrate_ratio(m):
        """
        This function initializes the flow rate ratio of the high and low salinity streams.
        The flowrate ratio is equal to the flow rate of the low salinity stream divided by the sum of the flow rates of the high and low salinity streams.

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model

        Returns
        -------
        float
            Flow rate ratio of the high and low salinity streams.
        """
        return m.vel_init['LC'] / sum(m.vel_init[sol] for sol in m.SOL)

    m.phi = pyo.Var(
        doc='Vol. flow rate ratio = In LC to total In (LC+HC) RU [-]',
        initialize=_flowrate_ratio,
        bounds=_flowrate_ratio_b,
        domain=pyo.NonNegativeReals,
    )

    def _conc_mol_eq_b(m):
        """
        This function bounds the concentration of the HC and LC mixed stream reaching equilibrium.

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model

        Returns
        -------
        tuple
            Lower and upper bounds of the concentration of the HC and LC mixed stream reaching equilibrium
        """
        lb = (
            m.phi.ub * flow_conc_data['feed_conc_mol']['fl1']
            + (1 - m.phi.ub) * flow_conc_data['feed_conc_mol']['fh1']
        )
        ub = (
            m.phi.lb * flow_conc_data['feed_conc_mol']['fl1']
            + (1 - m.phi.lb) * flow_conc_data['feed_conc_mol']['fh1']
        )
        return (lb, ub)

    def _conc_mol_eq(m):
        """
        This function initializes the concentration of the HC and LC mixed stream reaching equilibrium.

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model

        Returns
        -------
        float
            Concentration of the HC and LC mixed stream reaching equilibrium.
        """
        return (
            m.phi * flow_conc_data['feed_conc_mol']['fl1']
            + (1 - m.phi) * flow_conc_data['feed_conc_mol']['fh1']
        )

    m.conc_mol_eq = pyo.Var(
        doc='Concentration of the HC and LC mixed stream reaching equilibrium [mol L-1]',
        initialize=_conc_mol_eq,
        bounds=_conc_mol_eq_b,
        domain=pyo.NonNegativeReals,
    )

    def _conc_mol_b(m, i, k, sol):
        """
        This function bounds the molar concentration of the high and low salinity streams.

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        i : str
            Stream origin
        k : str
            Stream destination
        sol : str
            High or low salinity stream (HC or LC).

        Returns
        -------
        tuple
            Lower and upper bounds of the molar concentration of the high and low salinity streams
        """
        if (i, k) in (m.out_RU * ['rmu'] | [('rmu', 'dm'), ('dm', 'disch')]):
            if sol == 'HC':
                return (m.conc_mol_eq.lb, flow_conc_data['feed_conc_mol']['fh1'])
            return (flow_conc_data['feed_conc_mol']['fl1'], m.conc_mol_eq.ub)
        if sol == 'HC':
            return (m.conc_mol_eq.lb, flow_conc_data['feed_conc_mol']['fh1'])
        return (flow_conc_data['feed_conc_mol']['fl1'], m.conc_mol_eq.ub)

    m.conc_mol = pyo.Var(
        m.streams - m.RMU_RU_streams - m.RU_RSU_streams,
        m.SOL,
        doc="Molar concentration [mol L-1]",
        domain=pyo.NonNegativeReals,
        bounds=_conc_mol_b,
        initialize=lambda _, i, k, sol: (
            flow_conc_data['feed_conc_mol']['fh1']
            if sol == 'HC'
            else flow_conc_data['feed_conc_mol']['fl1']
        ),
    )

    m.conc_mol['in', 'fs', 'HC'].fix(flow_conc_data['feed_conc_mol']['fh1'])
    m.conc_mol['in', 'fs', 'LC'].fix(flow_conc_data['feed_conc_mol']['fl1'])

    m.NP = pyo.Var(
        m.RU,
        domain=pyo.NonNegativeReals,
        initialize=lambda m: m_stack.GP.value * 1e-2,
        bounds=lambda m: (None, m_stack.GP.value * 1e-2),  # 2e3)
        doc="Net Power output RED stack * 1e-2 [W]",
    )

    @m.Expression(doc='Membranes Capital Cost [USD]')
    def iems_cap_cost(m):
        """
        Expression to calculate the membranes capital cost in USD

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model

        Returns
        -------
        float
            Membranes capital cost in USD
        """
        return 2 * m._total_iem_area * m.iems_price * m.eur2usd  # 2 iems per cp

    @m.Expression(doc='RED Stack Capital Cost [USD]')
    def stack_cap_cost(m):
        """
        Expression to calculate the RED stack capital cost in USD

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model

        Returns
        -------
        float
            RED stack capital cost in USD
        """
        return m.iems_cap_cost * (1 + 0.517)  # 51.7% IEMs cap. cost

    m.stack_cost = pyo.Var(
        m.RU,
        doc='Stack capital cost [USD]',
        initialize=m.stack_cap_cost.expr(),
        domain=pyo.NonNegativeReals,
        bounds=(None, m.stack_cap_cost.expr()),
    )

    def _op_cost(m):
        """
        This function calculates the operational cost of the RED units.

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model

        Returns
        -------
        float
            Operational cost of the RED units in USD

        Notes
        -----
        The operational cost is the sum of the membrane replacement cost and the electricity cost.
        The electricity cost is calculated as the product of the electricity price, the load factor, and the pump power.
        """
        PP = sum(
            ureg.convert(48e-7 * m.vel_init[sol] / m.dh[sol] ** 2 * m.L, 'mbar', 'Pa')
            * ureg.convert(m.vel_init[sol], 'cm', 'm')
            * m._cross_area[sol]
            * m.cell_pairs
            / m.pump_eff
            for sol in m.SOL
        )
        return pyo.value(
            m.iems_cap_cost * m.CRFm
            + m.electricity_price * 8760 * m.load_factor * ureg.convert(PP, 'W', 'kW')
        )

    def _op_cost_b(m):
        """
        This function bounds the operational cost of the RED units.

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model

        Returns
        -------
        tuple
            Lower and upper bounds of the operational cost of the RED units in USD
        """
        PP = sum(
            ureg.convert(48e-7 * m.vel_ub[sol] / m.dh[sol] ** 2 * m.L, 'mbar', 'Pa')
            * ureg.convert(m.vel_ub[sol], 'cm', 'm')
            * m._cross_area[sol]
            * m.cell_pairs
            / m.pump_eff
            for sol in m.SOL
        )
        return (
            None,
            pyo.value(
                m.iems_cap_cost * m.CRFm
                + m.electricity_price
                * 8760
                * m.load_factor
                * ureg.convert(PP, 'W', 'kW')
            ),
        )

    m.operating_cost = pyo.Var(
        m.RU,
        initialize=_op_cost,
        bounds=_op_cost_b,
        doc='RU operational cost [USD]',
        domain=pyo.NonNegativeReals,
    )

    @m.Expression(m.units - m.RU - m.out_RU, m.SOL)
    def _flow_into(m, option, sol):
        """
        The expression calculates the flow rate into the mixers and splitters.

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        option : str
            Unit (mixer or splitter)
        sol : str
            High or low salinity stream (HC or LC).

        Returns
        -------
        Pyomo.Expression
            Flow rate into the mixers and splitters
        """
        return sum(
            m.flow_vol[src, sink, sol] for src, sink in m.streams if sink == option
        )  # src can be any unit except RU and their outlets

    @m.Expression(m.units - m.RU - m.out_RU, m.SOL)
    def _conc_into(m, option, sol):
        """
        The expression calculates the molar concentration into the mixers.

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        option : str
            Unit (mixer)
        sol : str
            High or low salinity stream (HC or LC).

        Returns
        -------
        Pyomo.Expression
            Molar concentration into the mixers
        """
        if option in m.mixers:
            return sum(
                m.flow_vol[src, sink, sol] * m.conc_mol[src, sink, sol]
                for src, sink in m.streams
                if sink == option
            )
        return pyo.Expression.Skip

    @m.Expression(m.units - m.RU - m.in_RU, m.SOL)
    def _flow_out_from(m, option, sol):
        """
        The expression calculates the flow rate out of the mixers and splitters.

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        option : str
            Unit (mixer or splitter)
        sol : str
            High or low salinity stream (HC or LC).

        Returns
        -------
        Pyomo.Expression
            Flow rate out of the mixers and splitters.
        """
        return sum(
            m.flow_vol[src, sink, sol] for src, sink in m.streams if src == option
        )

    @m.Expression(m.units - m.in_RU, m.SOL)
    def _conc_out_from(m, option, sol):
        """
        The expression calculates the molar concentration out of the mixers.

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        option : str
            Unit (mixer)
        sol : str
            High or low salinity stream (HC or LC).

        Returns
        -------
        Pyomo.Expression
            Molar concentration out of the mixers
        """
        if option in m.mixers:
            return sum(
                m.flow_vol[src, sink, sol] * m.conc_mol[src, sink, sol]
                for src, sink in m.streams
                if src == option
            )
        return pyo.Expression.Skip

    # Mixers mass balance
    m.mixer_balances = pyo.ConstraintList()
    for mu in m.mixers - m.RU - m.in_RU:
        [
            m.mixer_balances.add(m._flow_into[mu, sol] == m._flow_out_from[mu, sol])
            for sol in m.SOL
        ]  # Mass balance for mixers, flow rate
        [
            m.mixer_balances.add(m._conc_into[mu, sol] == m._conc_out_from[mu, sol])
            for sol in m.SOL
        ]  # Mass balance for mixers, molar concentration

    # Splitter mass balance
    m.splitter_balances = pyo.ConstraintList()
    for su in m.splitters - m.RU - m.out_RU:
        [
            m.splitter_balances.add(m._flow_into[su, sol] == m._flow_out_from[su, sol])
            for sol in m.SOL
        ]  # Mass balance for splitters, flow rate
    for src1, sink1 in m.from_splitters - m.RU_RSU_streams:
        for src2, sink2 in m.to_splitters - m.RU_RSU_streams:
            [
                m.splitter_balances.add(
                    m.conc_mol[src2, sink2, sol] == m.conc_mol[src1, sink1, sol]
                )
                for sol in m.SOL
                if src1 == sink2
            ]  # Mass balance for splitters, molar concentration

    # Disjunct for the existence of RED units
    @m.Disjunct(m.RU)
    def unit_exists(disj, unit):
        pass

    # Disjunct for the absence of RED units
    @m.Disjunct(m.RU)
    def unit_absent(no_unit, unit):
        @no_unit.Constraint(m.SOL, m.in_RU)
        def _no_flow_in(disj, sol, ri):
            """
            This function constrains the flow rate into the RED units to zero.

            Parameters
            ----------
            disj : Disjunct
                Disjunct for the absence of RED units
            sol : str
                High or low salinity stream (HC or LC).
            ri : RED unit inlets
                The index of the RED unit inlets

            Returns
            -------
            Pyomo.Constraint
                Flow rate into the RED units is zero
            """
            # No flow into RED units if the RED unit does not exist and the RED unit inlet is in the set of RED unit inlets RMU_RU_streams
            if (ri, unit) in m.RMU_RU_streams:
                return m._flow_into[ri, sol] == 0
            return pyo.Constraint.Skip

        pass

    @m.Disjunction(m.RU)
    def unit_exists_or_not(m, unit):
        """
        Disjunction for the existence or absence of RED units

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        unit : RED unit
            The index of the RED unit

        Returns
        -------
        Disjunction
            Disjunction for the existence or absence of RED units
        """
        return [m.unit_exists[unit], m.unit_absent[unit]]

    # Boolean variable for existence of RED units
    m.Yunit = pyo.BooleanVar(m.RU, doc="Boolean variable for existence of a RED unit")
    for unit in m.RU:
        m.Yunit[unit].associate_binary_var(
            m.unit_exists[unit].indicator_var.get_associated_binary()
        )  # Associate the binary variable with the disjunct for the existence of RED units

    # Logical Constraint: At least one RED unit must exist
    m.atleast_oneRU = pyo.LogicalConstraint(
        doc='At least one RED unit must exist', expr=atleast(1, m.Yunit)
    )

    # Logical Constraint: Yi+1 implies Yi
    @m.LogicalConstraint(m.RU, doc='Existence of RUn+1 implies RUn')
    def Yii_implies_Yi(m, unit):
        """
        Logical constraint for the existence of RED units that implies the existence of the previous RED unit.
        To avoid redundant solutions, the existence of RED unit n+1 implies the existence of RED unit n.

        Parameters
        ----------
        m : Pyomo concrete model
            RED GDP model
        unit : RED unit
            The index of the RED unit

        Returns
        -------
        LogicalConstraint
            Existence of RED unit n+1 implies the existence of RED unit n
        """
        ru = list(m.RU)
        idx = m.RU.ord(unit) - 1
        return (
            m.Yunit[ru[idx + 1]].implies(m.Yunit[ru[idx]])
            if idx < len(ru) - 1
            else pyo.Constraint.Skip
        )

    # =============================================================================
    #     RED Stack Model
    # =============================================================================

    def REDunit_model(ru, unit):
        """
        RED unit model equations and constraints when the RED unit exists.

        Parameters
        ----------
        ru : set
            Set of RED units
        unit : RED unit
            The index of the active RED unit

        Returns
        -------
        Block
            RED unit model equations and constraints
        """
        nfe = 5  # Number of finite elements
        ru.length_domain = pyo.Set(
            bounds=(0.0, 1.0),
            initialize=sorted(np.linspace(0.0, 1.0, nfe + 1, dtype=np.float32)),
            doc="Normalized length domain",
        )

        # General expressions to calculate the integral of the trapezoidal rule
        def _int_trap_rule(x, v):
            """
            This function calculates the integral of the trapezoidal rule.

            Parameters
            ----------
            x : set
                Length domain
            v : str
                Variable

            Returns
            -------
            Pyomo.Expression
                Integral of the trapezoidal rule
            """
            ds = sorted(x)  # Sort the x values
            a = list(v.values())  # Get the values of v
            return sum(
                0.5 * (ds[i + 1] - ds[i]) * (a[i + 1] + a[i])
                for i in range(len(ds) - 1)
            )  # Calculate the integral of the trapezoidal rule

        def _int_trap_rule_sol(m, x, sol, v):
            """
            This function calculates the integral of the trapezoidal rule for the high and low salinity streams.

            Parameters
            ----------
            m : Pyomo concrete model
                RED GDP model
            x : set
                Length domain
            sol : str
                High or low salinity stream (HC or LC) index
            v : str
                Variable

            Returns
            -------
            Pyomo.Expression
                Integral of the trapezoidal rule for the high and low salinity streams
            """
            ds = sorted(x)  # Sort the x values
            return sum(
                0.5 * (ds[i + 1] - ds[i]) * (v(m, ds[i + 1], sol) + v(m, ds[i], sol))
                for i in range(len(ds) - 1)
            )  # Calculate the integral of the trapezoidal rule for the high and low salinity streams

        def _bwd_fun(ru, x, sol, v, dv):
            """
            This function calculates the backward finite difference.

            Parameters
            ----------
            ru : str
                Set of RED units
            x : set
                Length domain
            sol : str
                High or low salinity stream (HC or LC) index
            v : str
                Variable
            dv : str
                Derivative of the variable

            Returns
            -------
            Pyomo.Expression
                Backward finite difference
            """
            tmp = list(ru.length_domain)  # Get the length domain
            idx = ru.length_domain.ord(x) - 1  # Get the index of the length domain
            if (
                idx != 0
            ):  # idx == 0 Needed since '-1' is considered a valid index in Python
                return (
                    dv(ru, tmp[idx], sol)
                    - 1
                    / (tmp[idx] - tmp[idx - 1])
                    * (v(ru, tmp[idx], sol) - v(ru, tmp[idx - 1], sol))
                    == 0
                )  # Calculate the backward finite difference

        def _flow_vol(ru, x, sol):
            """
            This function initializes the flow rate of the high and low salinity streams.

            Parameters
            ----------
            ru : str
                Set of RED units
            x : float
                Length domain
            sol : str
                High or low salinity stream (HC or LC) index

            Returns
            -------
            float
                Flow rate of the high and low salinity streams
            """
            flow_init = pyo.value(36 * m.vel_init[sol] * m._cross_area[sol])
            if sol == 'HC':
                if (
                    pyo.value(flow_init * m.nr * m.cell_pairs)
                    > flow_conc_data['feed_flow_vol']['fh1']
                ):
                    return ureg.convert(
                        flow_conc_data['feed_flow_vol']['fh1'] / m.nr / m.cell_pairs,
                        'm**3',
                        'liter',
                    )
                return ureg.convert(flow_init, 'm**3', 'liter')
            if sol == 'LC':
                if (
                    pyo.value(flow_init * m.nr * m.cell_pairs)
                    > flow_conc_data['feed_flow_vol']['fl1']
                ):
                    return ureg.convert(
                        flow_conc_data['feed_flow_vol']['fl1'] / m.nr / m.cell_pairs,
                        'm**3',
                        'liter',
                    )
                return ureg.convert(flow_init, 'm**3', 'liter')

        def _flow_vol_b(ru, x, sol):
            """
            This function bounds the flow rate of the high and low salinity streams to the available feed flow rate.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            tuple
                Lower and upper bounds of the flow rate of the high and low salinity streams
            """
            lb = pyo.value(36 * m.vel_lb[sol] * m._cross_area[sol])
            ub = pyo.value(36 * m.vel_ub[sol] * m._cross_area[sol])
            return (
                ureg.convert(lb, 'm**3', 'liter'),
                ureg.convert(ub, 'm**3', 'liter'),
            )

        ru.flow_vol_x = pyo.Var(
            ru.length_domain,
            m.SOL,
            initialize=_flow_vol,
            bounds=_flow_vol_b,
            domain=pyo.NonNegativeReals,
            doc="Discretized Volumetric Flow Rate [L h-1]",
        )

        def _conc_molx_b(ru, x, sol):
            """
            This function bounds the molar concentration of the high and low salinity streams.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            tuple
                Lower and upper bounds of the molar concentration of the high and low salinity streams
            """
            if sol == 'HC':
                return (m.conc_mol_eq.lb, flow_conc_data['feed_conc_mol']['fh1'])
            #                 return (m.conc_mol_eq.lb, feed_conc_mol['fh1'])
            return (flow_conc_data['feed_conc_mol']['fl1'], m.conc_mol_eq.ub)

        #             return (feed_conc_mol['fl1'], m.conc_mol_eq.ub)

        def _conc_molx(ru, x, sol):
            """
            This function initializes the molar concentration of the high and low salinity streams.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            float
                Molar concentration of the high and low salinity streams
            """
            if sol == 'HC':
                return flow_conc_data['feed_conc_mol']['fh1']
            return flow_conc_data['feed_conc_mol']['fl1']

        ru.conc_mol_x = pyo.Var(
            ru.length_domain,
            m.SOL,
            initialize=_conc_molx,
            bounds=_conc_molx_b,
            doc="Discretized Molar NaCl concentration [mol L-1]",
            domain=pyo.NonNegativeReals,
        )

        def _pressure_x(ru, x, sol):
            """
            This function initializes the discretized pressure.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            float
                Discretized pressure
            """
            # Pressure drop calculation with the Darcy-Weisbach equation with spacer correction
            delta_p = pyo.value(
                48
                * ureg.convert(1, 'cP', 'Pa*s')
                * ureg.convert(m.vel_init[sol], 'cm', 'm')
                / m.dh[sol] ** 2
                * m.L
            )
            ub = ureg.convert(1, 'atm', 'mbar')
            lb = ub - ureg.convert(delta_p, 'Pa', 'mbar')
            if x == ru.length_domain.first():
                return ub
            return lb

        def _pressure_x_b(ru, x, sol):
            """
            This function bounds the discretized pressure.

            Parameters
            ----------
            ru : Set
                The set of RED units
            x : Set
                Length domain index
            sol : Set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            tuple
                Lower and upper bounds of the discretized pressure
            """
            delta_p = pyo.value(
                48
                * ureg.convert(1, 'cP', 'Pa*s')
                * ureg.convert(m.vel_ub[sol], 'cm', 'm')
                / m.dh[sol] ** 2
                * m.L
            )
            ub = ureg.convert(1, 'atm', 'mbar')
            # lb = ub - ureg.convert(delta_p, 'Pa', 'mbar')
            return (None, ub)

        ru.pressure_x = pyo.Var(
            ru.length_domain,
            m.SOL,
            domain=pyo.NonNegativeReals,
            #                                initialize=101325.0,
            initialize=_pressure_x,  # ureg.convert(1., 'atm','mbar'),
            bounds=_pressure_x_b,
            doc='Discretized pressure [mbar]',
        )

        # =============================================================================
        #     Electric variables
        # =============================================================================
        ru.Ecpx = pyo.Var(
            ru.length_domain,
            domain=pyo.NonNegativeReals,
            initialize=lambda _, x: 2e3
            * m.gas_constant
            * m.T
            / m.faraday_constant
            * m.iems_permsel_avg
            * (pyo.log(ru.conc_mol_x[x, 'HC']) - pyo.log(ru.conc_mol_x[x, 'LC'])),
            bounds=lambda ru, x: (
                None,
                2e3
                * m.gas_constant
                * m.T
                / m.faraday_constant
                * m.iems_permsel_avg
                * (
                    pyo.log(flow_conc_data['feed_conc_mol']['fh1'])
                    - pyo.log(flow_conc_data['feed_conc_mol']['fl1'])
                ),
            ),
            doc="Nernst ELectric Potential per cell pair [mV per cell pair]",
        )

        ru.EMF = pyo.Var(
            domain=pyo.NonNegativeReals,
            initialize=m.cell_pairs * _int_trap_rule(ru.length_domain, ru.Ecpx) * 1e-3,
            bounds=(None, m.cell_pairs * ureg.convert(ru.Ecpx[0].ub, 'mV', 'V')),
            doc="Nernst Potential RED Stack [V]",
        )

        def _ksol_b(ru, x, sol):
            """
            This function bounds the sol. conductivity per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            tuple
                Lower and upper bounds of the sol. conductivity per unit length
            """
            # Conductivity bounds based on correlation from experimental data
            if sol == 'HC':
                ub = 7.7228559 * flow_conc_data['feed_conc_mol']['fh1'] + 0.5670209
                lb = 7.7228559 * m.conc_mol_eq.lb + 0.5670209
                return (lb, ub)
            ub = 10.5763914 * m.conc_mol_eq.ub + 0.0087379
            lb = 10.5763914 * flow_conc_data['feed_conc_mol']['fl1'] + 0.0087379
            return (lb, ub)

        def _ksol(ru, x, sol):
            """
            This function initializes the sol. conductivity per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            float
                Solutions conductivity per unit length
            """
            if sol == 'HC':
                return pyo.value(7.7228559 * ru.conc_mol_x[0, 'HC'] + 0.5670209)
            return pyo.value(10.5763914 * ru.conc_mol_x[0, 'LC'] + 0.0087379)

        ru.ksol = pyo.Var(
            ru.length_domain,
            m.SOL,
            domain=pyo.NonNegativeReals,
            bounds=_ksol_b,
            initialize=_ksol,
            doc="Sol. conductivity per unit length [S m-1]",
        )

        ru.ksol_T = pyo.Var(
            ru.length_domain,
            m.SOL,
            domain=pyo.NonNegativeReals,
            bounds=lambda _, x, sol: (
                ru.ksol[x, sol].lb * (1 + 0.02 * (m.T - m.Tref)),
                ru.ksol[x, sol].ub * (1 + 0.02 * (m.T - m.Tref)),
            ),  # 0.02 is the temperature coefficient of the conductivity from the literature
            initialize=lambda _, x, sol: ru.ksol[x, sol] * (1 + 0.02 * (m.T - m.Tref)),
            doc="Temperature corrected sol. conductivity per unit length [S m-1]",
        )

        def _Rsol_b(ru, x, sol):
            """
            This function bounds the solution resistance per cell pair per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            tuple
                Lower and upper bounds of the solution resistance per cell pair per unit length
            """
            if sol == 'HC':
                lb = (
                    m.spacer_thickness[sol]
                    / m.spacer_porosity[sol] ** 2
                    / ru.ksol_T[0, 'HC'].ub
                    * 1e4
                )
                ub = (
                    m.spacer_thickness[sol]
                    / m.spacer_porosity[sol] ** 2
                    / ru.ksol_T[0, 'HC'].lb
                    * 1e4
                )
                return (lb, ub)
            lb = (
                m.spacer_thickness[sol]
                / m.spacer_porosity[sol] ** 2
                / ru.ksol_T[0, 'LC'].ub
                * 1e4
            )
            ub = (
                m.spacer_thickness[sol]
                / m.spacer_porosity[sol] ** 2
                / ru.ksol_T[0, 'LC'].lb
                * 1e4
            )
            return (lb, ub)

        def _Rsol(ru, x, sol):
            """
            This function initializes the solution resistance per cell pair per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            float
                Solution resistance per cell pair per unit length
            """
            if sol == 'HC':
                return pyo.value(
                    m.spacer_thickness[sol]
                    / m.spacer_porosity[sol] ** 2
                    / ru.ksol_T[0, 'HC']
                    * 1e4
                )
            return pyo.value(
                m.spacer_thickness[sol]
                / m.spacer_porosity[sol] ** 2
                / ru.ksol_T[0, 'LC']
                * 1e4
            )

        ru.Rsol = pyo.Var(
            ru.length_domain,
            m.SOL,
            domain=pyo.NonNegativeReals,
            bounds=_Rsol_b,
            initialize=_Rsol,
            doc="Solution resistance per cell pair per unit length [ohm cm2 per cp]",
        )

        def _Rcpx_b(ru, x):
            """
            This function bounds the internal resistance per cell pair per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index

            Returns
            -------
            tuple
                Lower and upper bounds of the internal resistance per cell pair per unit length
            """
            lb = (
                sum(ru.Rsol[0, sol].lb for sol in m.SOL)
                + sum(m.iems_resistance[iem] for iem in m.iem) * 1e4
            )
            ub = (
                sum(ru.Rsol[0, sol].ub for sol in m.SOL)
                + sum(m.iems_resistance[iem] for iem in m.iem) * 1e4
            )
            return (lb, ub)

        def _Rcpx(ru, x):
            """
            This function initializes the internal resistance per cell pair per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index

            Returns
            -------
            float
                Internal resistance per cell pair per unit length
            """
            r_iem = sum(m.iems_resistance[iem] for iem in m.iem) * 1e4
            return sum(ru.Rsol[0, sol] for sol in m.SOL) + r_iem

        ru.Rcpx = pyo.Var(
            ru.length_domain,
            domain=pyo.NonNegativeReals,
            initialize=_Rcpx,
            bounds=_Rcpx_b,
            doc="Internal resistance per cell pair per unit length [ohm cm2 per cp]",
        )

        ru.Rstack = pyo.Var(
            domain=pyo.NonNegativeReals,
            initialize=lambda _: m.cell_pairs
            * _int_trap_rule(ru.length_domain, ru.Rcpx)
            / m.Aiem
            * 1e-4,
            bounds=(
                m.cell_pairs * ru.Rcpx[0].lb / m.Aiem * 1e-4,
                m.cell_pairs * ru.Rcpx[0].ub / m.Aiem * 1e-4,
            ),
            doc="RED stack Internal resistance [ohm]",
        )

        ru.Rload = pyo.Var(
            domain=pyo.NonNegativeReals,
            initialize=_int_trap_rule(ru.length_domain, ru.Rcpx),
            bounds=(0.2, 100.0),  # (0.02,100.),
            doc="Load resistance [ohm cm2 per cp]",
        )
        #         ru.Rload.fix(8.75)

        ru.Idx = pyo.Var(
            ru.length_domain,
            domain=pyo.NonNegativeReals,
            initialize=lambda _, x: ru.Ecpx[x] / (ru.Rcpx[x] + ru.Rload),
            bounds=lambda _, x: (None, ru.Ecpx[x].ub / (ru.Rcpx[x].lb + ru.Rload.lb)),
            doc="Electric Current Density [mA cm-2]",
        )

        ru.Istack = pyo.Var(
            domain=pyo.NonNegativeReals,
            initialize=lambda _: _int_trap_rule(ru.length_domain, ru.Idx) * m.Aiem * 10,
            bounds=(None, ru.Idx[0].ub * m.Aiem * 10),
            doc="Electric Current Stack [A]",
        )

        # =============================================================================
        #     Material transfer terms
        # =============================================================================

        def _Jcond_b(ru, x):
            """
            This function bounds the conductive molar flux per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index

            Returns
            -------
            tuple
                Lower and upper bounds of the conductive molar flux per unit length
            """
            lb = ru.Idx[0].lb * 3.6e4 / m.faraday_constant
            ub = ru.Idx[0].ub * 3.6e4 / m.faraday_constant
            return (lb, ub)

        ru.Jcond = pyo.Var(
            ru.length_domain,
            domain=pyo.NonNegativeReals,
            initialize=lambda _, x: ru.Idx[x] * 3.6e4 / m.faraday_constant,
            bounds=_Jcond_b,
            doc="Conductive Molar Flux (electromigration) NaCl per unit length [mol m-2 h-1]",
        )

        def _Jdiff_b(ru, x):
            """
            This function bounds the diffusive molar flux per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index

            Returns
            -------
            tuple
                Lower and upper bounds of the diffusive molar flux per unit length
            """
            lb = (
                3.6e6
                * 2
                * m.diff_nacl
                / m.iems_thickness['CEM']
                * (ru.conc_mol_x[0, 'HC'].lb - ru.conc_mol_x[0, 'LC'].ub)
            )
            ub = (
                3.6e6
                * 2
                * m.diff_nacl
                / m.iems_thickness['CEM']
                * (ru.conc_mol_x[0, 'HC'].ub - ru.conc_mol_x[0, 'LC'].lb)
            )
            return (lb, ub)

        ru.Jdiff = pyo.Var(
            ru.length_domain,
            domain=pyo.NonNegativeReals,
            initialize=lambda _, x: 3.6e6
            * 2
            * m.diff_nacl
            / m.iems_thickness['CEM']
            * (ru.conc_mol_x[0, 'HC'] - ru.conc_mol_x[0, 'LC']),
            bounds=_Jdiff_b,
            doc="Diffusive Molar Flux NaCl per unit length [mol m-2 h-1]",
        )

        ru.Ji = pyo.Var(
            ru.length_domain,
            domain=pyo.NonNegativeReals,
            initialize=lambda _, x: ru.Jcond[x] + ru.Jdiff[x],
            bounds=(None, ru.Jcond[0].ub + ru.Jdiff[0].ub),
            doc="Molar Flux NaCl per unit length [mol m-2 h-1]",
        )

        # =============================================================================
        #     Derivative terms
        # =============================================================================

        ru.flow_vol_dx = pyo.Var(
            ru.flow_vol_x.index_set(),
            doc="Partial derivative of volumetric flow wrt to normalized length",
            bounds=(-1.0, 1.0),
            initialize=0,
        )

        ru.conc_mol_dx = pyo.Var(
            ru.conc_mol_x.index_set(),
            doc="Partial derivative of molar concentration wrt to normalized length",
            bounds=(-1.0, 1.0),
            initialize=0,
        )

        ru.pressure_dx = pyo.Var(
            ru.pressure_x.index_set(),
            doc="Partial derivative of pressure wrt to normalized length",
            domain=pyo.NonPositiveReals,
            # [mbar]
            bounds=lambda _, x, sol: (
                ureg.convert(
                    -48
                    * ureg.convert(1, 'cP', 'Pa*s')
                    * ureg.convert(m.vel_ub[sol], 'cm', 'm')
                    / m.dh[sol] ** 2
                    * m.L,
                    'Pa',
                    'mbar',
                ),
                None,
            ),
            initialize=lambda _, x, sol: (
                0
                if x == ru.length_domain.first()
                else ureg.convert(
                    -48
                    * ureg.convert(1, 'cP', 'Pa*s')
                    * ureg.convert(m.vel_init[sol], 'cm', 'm')
                    / m.dh[sol] ** 2
                    * m.L,
                    'Pa',
                    'mbar',
                )
            ),
        )

        ru.GP = pyo.Var(
            domain=pyo.NonNegativeReals,
            initialize=ru.Istack * (ru.EMF - ru.Rstack * ru.Istack),
            bounds=(None, m_stack.GP.value),  # ru.EMF.ub**2/4/ru.Rstack.lb),
            doc="Gross Power output RED stack [W]",
        )  # The upper bound is the optimal value from the stand-alone RED stack optimization

        ru.PP = pyo.Var(
            domain=pyo.NonNegativeReals,
            initialize=sum(
                ureg.convert(
                    48e-7 * m.vel_init[sol] / m.dh[sol] ** 2 * m.L, 'mbar', 'Pa'
                )
                * m.cell_pairs
                * ureg.convert(ru.flow_vol_x[0, sol], 'dm**3/hour', 'm**3/s')
                / m.pump_eff
                for sol in m.SOL
            ),
            bounds=(
                None,
                sum(
                    48
                    * ureg.convert(1, 'cP', 'Pa*s')
                    * ureg.convert(m.vel_ub[sol], 'cm', 'm')
                    / m.dh[sol] ** 2
                    * m.L
                    * m.cell_pairs
                    * ureg.convert(ru.flow_vol_x[0, sol].ub, 'dm**3/hour', 'm**3/s')
                    / m.pump_eff
                    for sol in m.SOL
                ),
            ),  # Q [m3 h-1]; Ap [Pa]
            doc="Pumping Power loss RED stack [W]",
        )

        ru.NP = pyo.Var(
            domain=pyo.NonNegativeReals,
            initialize=ru.GP - ru.PP,
            bounds=(None, ru.GP.ub - ru.PP.ub),
            doc="Net Power output RED stack [W]",
        )

        @ru.Constraint(
            ru.length_domain,
            doc='Nernst Potential per unit length per cell pair [mV per cp]',
        )
        def _nernst_potential_cp(ru, x):  # Rg[J mol-1 K-1] , F[A s mol-1], T[K]
            """
            This function calculates the Nernst potential per unit length per cell pair.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index

            Returns
            -------
            Pyomo.Constraint
                Nernst potential per unit length per cell pair
            """
            cst = 2 * m.gas_constant * m.T / m.faraday_constant * m.iems_permsel_avg
            return ru.conc_mol_x[x, 'HC'] == ru.conc_mol_x[x, 'LC'] * pyo.exp(
                ru.Ecpx[x] / ureg.convert(cst, 'V', 'mV')
            )

        @ru.Constraint(
            ru.length_domain,
            m.SOL,
            doc="Solution's Conductivity per unit length [S m-1]",
        )
        def _sol_cond(ru, x, sol):
            """
            Constraint for the solution's conductivity per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            Pyomo.Constraint
                Solution's conductivity per unit length
            """
            if sol == 'HC':
                return ru.ksol[x, sol] == 7.7228559 * ru.conc_mol_x[x, sol] + 0.5670209
            return ru.ksol[x, sol] == 10.5763914 * ru.conc_mol_x[x, sol] + 0.0087379

        @ru.Constraint(
            ru.length_domain,
            m.SOL,
            doc="Temperature corrected Solution's Conductivity per unit length [S m-1]",
        )
        def _sol_cond_T(ru, x, sol):
            """
            Constraint for the temperature corrected solution's conductivity per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            Pyomo.Constraint
                Temperature corrected solution's conductivity per unit length
            """
            return ru.ksol_T[x, sol] == ru.ksol[x, sol] * (1 + 0.02 * (m.T - m.Tref))

        @ru.Constraint(
            ru.length_domain,
            m.SOL,
            doc="Channel's resistance per cell pair per unit length [ohm cm2 per cp]",
        )
        def _channel_res(ru, x, sol):
            """
            Constraint for the channel's resistance per cell pair per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            Pyomo.Constraint
                Channel's resistance per cell pair per unit length
            """
            return (
                ru.Rsol[x, sol] * ru.ksol_T[x, sol]
                == m.spacer_thickness[sol] / m.spacer_porosity[sol] ** 2 * 1e4
            )

        @ru.Constraint(
            ru.length_domain,
            doc="Internal resistance per cell pair per unit length [ohm cm2 per cp]",
        )
        def _int_res(ru, x):
            """
            Constraint for the internal resistance per cell pair per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index

            Returns
            -------
            Pyomo.Constraint
                Internal resistance per cell pair per unit length
            """
            return (
                ru.Rcpx[x]
                == sum(ru.Rsol[x, sol] for sol in m.SOL)
                + sum(m.iems_resistance[iem] for iem in m.iem) * 1e4
            )

        @ru.Constraint(
            ru.length_domain, doc='Electric current density per unit length [mA cm-2]'
        )
        def _current_dens_calc(ru, x):
            """
            Constraint for the electric current density per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index

            Returns
            -------
            Pyomo.Constraint
                Electric current density per unit length
            """
            return ru.Idx[x] * (ru.Rcpx[x] + ru.Rload) == ru.Ecpx[x]

        @ru.Expression(
            ru.length_domain, m.SOL, doc='Crossflow velocity in channel eq. [cm s-1]'
        )
        def vel(ru, x, sol):
            """
            This function calculates the crossflow velocity in the channel.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            float
                Crossflow velocity in the channel
            """
            return ureg.convert(
                ru.flow_vol_x[x, sol], 'liter/hour', 'cm**3/s'
            ) / ureg.convert(m._cross_area[sol], 'm**2', 'cm**2')

        def _vel_x(ru, x, sol):
            """
            This function returns the crossflow velocity in the channel.
            It is used to calculate the average crossflow velocity in the channel with the integral trapezoidal rule.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            var
               Variable for the crossflow velocity in the channel
            """
            return ru.vel[x, sol]

        @ru.Expression(m.SOL, doc='Average cross-flow velocity [cm s-1]')
        def vel_avg(ru, sol):
            """
            This function calculates the average cross-flow velocity in the channel.

            Parameters
            ----------
            ru : set
                The set of RED units
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            Pyomo.Constraint
                Average cross-flow velocity
            """
            v = _vel_x
            return _int_trap_rule_sol(ru, ru.length_domain, sol, v)

        @ru.Constraint(ru.length_domain, doc='Conductive molar flux (electromigration)')
        def _cond_molar_flux(ru, x):
            """
            Constraint for the conductive molar flux (electromigration).

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index

            Returns
            -------
            Pyomo.Constraint
                Conductive molar flux (electromigration)
            """
            return ru.Jcond[x] * m.faraday_constant / 3.6e4 == ru.Idx[x]

        #             return ru.Jcond[x] * m.faraday_constant == ru.Idx[x] * 3.6e4 # J [mol m-2 h-1], F [A s mol-1], Id[mA cm-2]

        @ru.Constraint(ru.length_domain, doc='Diffusive molar flux [mol m-2 h-1]')
        def _diff_molar_flux(ru, x):
            """
            Constraint for the diffusive molar flux.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index

            Returns
            -------
            Pyomo.Constraint
                Diffusive molar flux
            """
            return ru.Jdiff[x] == 3.6e6 * 2 * m.diff_nacl / m.iems_thickness['CEM'] * (
                ru.conc_mol_x[x, 'HC'] - ru.conc_mol_x[x, 'LC']
            )
            # J[mol m-2 h-1], D[m2 s-1], dm=50e-6 m, C[mol L-1]

        @ru.Constraint(
            ru.length_domain, doc='Total molar flux from HC to LC side [mol m-2 h-1]'
        )
        def _total_molar_flux(ru, x):
            """
            Constraint for the total molar flux from the high to low salinity side.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index

            Returns
            -------
            Pyomo.Constraint
                Total molar flux from the high to low salinity side
            """
            return ru.Ji[x] == ru.Jcond[x] + ru.Jdiff[x]

        def _flow_vol_x(ru, x, sol):
            """
            This function initializes the discretized volumetric flow rate.
            It is required to calculate the discretized volumetric flow rate balance constraint with the backward finite difference method.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            var
                Discretized volumetric flow rate variable
            """
            return ru.flow_vol_x[x, sol]

        def _flow_vol_dx(ru, x, sol):
            """
            This function initializes the discretized volumetric flow rate derivative.
            It is required to calculate the discretized volumetric flow rate balance constraint with the backward finite difference method.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            var
                Discretized volumetric flow rate derivative variable
            """
            return ru.flow_vol_dx[x, sol]

        @ru.Constraint(ru.length_domain, m.SOL)
        def flow_vol_dx_disc_eq(ru, x, sol):
            """
            This function calculates the discretized volumetric flow rate derivative with the backward finite difference method.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            Pyomo.Constraint
                Discretized volumetric flow rate derivative with the backward finite difference method.
            """
            if x == 0:
                return pyo.Constraint.Skip
            v = _flow_vol_x
            dv = _flow_vol_dx
            return _bwd_fun(ru, x, sol, v, dv)

        @ru.Constraint(
            ru.length_domain,
            m.SOL,
            doc='Volumetric flow rate balance w/o water transfer (i.e. no osmotic flux)',
        )
        def _flow_balance(ru, x, sol):
            """
            Constraint for the volumetric flow rate balance without water transfer (i.e. no osmotic flux).
            The flowrte along the length of the RED unit is constant.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            Pyomo.Constraint
                Volumetric flow rate balance without water transfer (i.e. no osmotic flux)
            """
            if x == ru.length_domain.first():  # or x == ru.length_domain.last():
                return pyo.Constraint.Skip
            return ru.flow_vol_dx[x, sol] == 0

        def _conc_mol_x(ru, x, sol):
            """
            This function initializes the molar concentration.
            It is required to calculate the molar concentration balance constraint with the backward finite difference method.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            var
                Molar concentration variable
            """
            return ru.conc_mol_x[x, sol]

        def _conc_mol_dx(ru, x, sol):
            """
            This function initializes the molar concentration derivative.
            It is required to calculate the molar concentration balance constraint with the backward finite difference method.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            var
                Molar concentration derivative variable
            """
            return ru.conc_mol_dx[x, sol]

        @ru.Constraint(ru.length_domain, m.SOL)
        def conc_mol_dx_disc_eq(ru, x, sol):
            """
            This constraint calculates the molar concentration derivative with the backward finite difference method.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            Pyomo.Constraint
                Molar concentration derivative with the backward finite difference method.
            """
            if x == 0:
                return pyo.Constraint.Skip
            v = _conc_mol_x
            dv = _conc_mol_dx
            return _bwd_fun(ru, x, sol, v, dv)

        @ru.Constraint(ru.length_domain, m.SOL, doc='Molar concentration balance')
        def _conc_balance(ru, x, sol):
            """
            This constraint calculates the molar concentration balance.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            Pyomo.Constraint
                Molar concentration balance
            """
            if x == ru.length_domain.first():  # or x == ru.length_domain.last():
                return pyo.Constraint.Skip
            if sol == 'LC':
                return (
                    ru.conc_mol_dx[x, sol] * ru.flow_vol_x[x, sol] == ru.Ji[x] * m.Aiem
                )
            return ru.conc_mol_dx[x, sol] * ru.flow_vol_x[x, sol] == -ru.Ji[x] * m.Aiem
            # dC/dx [mol L-1 m-1] * [m3 h-1] * 1e3 [L m-3] = [m] [mol m-2 h-1]
            # dC/dx [mol L-1 m-1] * [L h-1] = [m] [mol m-2 h-1]

        @ru.Constraint(ru.length_domain, doc='Concentration HC >= LC')
        def _conc_hc_gt_lc(ru, x):
            """
            This constraint ensures that the concentration of the high salinity stream is greater than the concentration of the low salinity stream.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index

            Returns
            -------
            Pyomo.Constraint
                Concentration of the high salinity stream is greater than the concentration of the low salinity stream
            """
            return ru.conc_mol_x[x, 'HC'] >= ru.conc_mol_x[x, 'LC']

        @ru.Expression(
            ru.length_domain, m.SOL, doc='Pressure drop per unit length [mbar m-1]'
        )
        def _deltaP(ru, x, sol):
            """
            This expression calculates the pressure drop per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            Pyomo.Expression
                Pressure drop per unit length computed with the Darcy-Weisbach equation for a spacer-filled channel.
            """
            return ureg.convert(
                48
                * ureg.convert(1, 'cP', 'Pa*s')
                * ureg.convert(ru.vel[x, sol], 'cm', 'm')
                / m.dh[sol] ** 2,
                'Pa',
                'mbar',
            )  # [mbar m-1]

        def _pressure_x(ru, x, sol):
            """
            This function initializes the pressure.
            It is required to calculate the pressure drop along the channel with the backward finite difference method.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            var
                Pressure variable
            """
            return ru.pressure_x[x, sol]

        def _pressure_dx(ru, x, sol):
            """
            This function initializes the pressure derivative.
            It is required to calculate the pressure drop along the channel with the backward finite difference method.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            var
                Pressure derivative variable
            """
            return ru.pressure_dx[x, sol]

        @ru.Constraint(ru.length_domain, m.SOL)
        def pressure_dx_disc_eq(ru, x, sol):
            """
            This constraint calculates the pressure derivative with the backward finite difference method.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            Pyomo.Constraint
                Pressure derivative with the backward finite difference method.
            """
            if x == 0:
                return pyo.Constraint.Skip
            v = _pressure_x
            dv = _pressure_dx
            return _bwd_fun(ru, x, sol, v, dv)

        @ru.Constraint(
            ru.length_domain, m.SOL, doc='Friction pressure drop per unit length [mbar]'
        )
        def _pressure_drop(ru, x, sol):
            """
            This constraint calculates the friction pressure drop per unit length.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            Pyomo.Constraint
                Friction pressure drop per unit length
            """
            if x == ru.length_domain.first():  # or x == ru.length_domain.last():
                return pyo.Constraint.Skip
            return ru.pressure_dx[x, sol] == -ru._deltaP[x, sol] * m.L
            # [mbar m-1]; 1e-2 [mbar Pa-1]; mu = 1 cP = 1e-3 Pa s; v [cm s-1] * m/100cm;

        def _Rsol_x(ru, x, sol):
            """
            This function passes the solution resistance per unit length to the integral trapezoidal rule function.

            Parameters
            ----------
            ru : set
                The set of RED units
            x : set
                Length domain index

            Returns
            -------
            var
                Solution resistance per unit length variable
            """
            return ru.Rsol[x, sol]

        @ru.Expression(
            m.SOL, doc='Average cross-flow velocity per unit length [ohm cm2 per cp]'
        )
        def Rsol_avg(ru, sol):
            """
            This function calculates the average solution resistance per unit length using the integral trapezoidal rule.

            Parameters
            ----------
            ru : set
                The set of RED units
            sol : set
                High or low salinity stream (HC or LC) index

            Returns
            -------
            Pyomo.Expression
                Average solution resistance per unit length
            """
            v = _Rsol_x
            return _int_trap_rule_sol(ru, ru.length_domain, sol, v)

        @ru.Expression(
            doc='Average cell pair resistance per unit length [ohm cm2 per cp]'
        )
        def Rcp_avg(ru):
            """
            This function calculates the average cell pair resistance per unit length using the integral trapezoidal rule.

            Parameters
            ----------
            ru : set
                The set of RED units

            Returns
            -------
            Pyomo.Expression
                Average cell pair resistance per unit length
            """
            return _int_trap_rule(ru.length_domain, ru.Rcpx)

        @ru.Expression(doc='Average current density [mA cm-2]')
        def Id_avg(ru):
            """
            This function calculates the average current density using the integral trapezoidal rule.

            Parameters
            ----------
            ru : set
                The set of RED units

            Returns
            -------
            Pyomo.Expression
                Average current density
            """
            return _int_trap_rule(ru.length_domain, ru.Idx)

        @ru.Expression(doc='Average cell pair potential per unit length [mV per cp]')
        def Ecp_avg(ru):
            """
            This function calculates the average cell pair potential per unit length using the integral trapezoidal rule.

            Parameters
            ----------
            ru : set
                The set of RED units

            Returns
            -------
            Pyomo.Expression
                Average cell pair potential per unit length
            """
            return _int_trap_rule(ru.length_domain, ru.Ecpx)

        @ru.Constraint(doc='Electromotive force RED unit [mV]')
        def _electric_potential_stack(ru):
            """
            This constraint calculates the electromotive force of the RED unit.

            Parameters
            ----------
            ru : set
                The set of RED units

            Returns
            -------
            Pyomo.Constraint
                Electromotive force of the RED unit
            """
            return ru.EMF == m.cell_pairs * ureg.convert(ru.Ecp_avg, 'mV', 'V')

        @ru.Constraint(doc='RED stack internal resistance [ohm]')
        def _int_resistance_stack(ru):
            """
            This constraint calculates the internal resistance of the RED stack.

            Parameters
            ----------
            ru : set
                The set of RED units

            Returns
            -------
            Pyomo.Constraint
                Internal resistance of the RED stack
            """
            return ru.Rstack * m.Aiem * 1e4 == m.cell_pairs * ru.Rcp_avg

        @ru.Constraint(
            doc='Electric current RED unit [A]'
        )  # [A] [mA cm-2] 1e-3 [A mA-1] 1e4 [cm2 m-2]
        def _electric_current_stack(ru):
            """
            This constraint calculates the electric current of the RED unit.

            Parameters
            ----------
            ru : set
                The set of RED units

            Returns
            -------
            Pyomo.Constraint
                Electric current of the RED unit
            """
            return ru.Istack == ru.Id_avg * m.Aiem * 10

        @ru.Constraint(doc='Gross Power Output RED unit [W]')
        def _gross_power(ru):
            """
            This constraint calculates the gross power output of the RED unit.

            Parameters
            ----------
            ru : set
                The set of RED units

            Returns
            -------
            Pyomo.Constraint
                Gross power output of the RED unit
            """
            return ru.GP == ru.Istack * (ru.EMF - ru.Rstack * ru.Istack)

        @ru.Constraint(doc='Pumping Power Consumption RED unit [W]')
        def _pump_power(ru):
            """
            This constraint calculates the pumping power consumption of the RED unit.

            Parameters
            ----------
            ru : set
                The set of RED units

            Returns
            -------
            Pyomo.Constraint
                Pumping power consumption of the RED unit
            """
            return ru.PP * m.pump_eff == sum(
                ureg.convert(
                    ru.pressure_x[0, sol] - ru.pressure_x[ru.length_domain.last(), sol],
                    'mbar',
                    'Pa',
                )
                * m.cell_pairs
                * ureg.convert(ru.flow_vol_x[0, sol], 'dm**3/hour', 'm**3/s')
                for sol in m.SOL
            )

        @ru.Constraint(doc='Net Power Output RED unit [W]')
        def _net_power(ru):
            """
            This constraint calculates the net power output of the RED unit.

            Parameters
            ----------
            ru : set
                The set of RED units

            Returns
            -------
            Pyomo.Constraint
                Net power output of the RED unit
            """
            return ru.NP == ru.GP - ru.PP

        # Fixing the midpoint of the uninitialized variables
        init_vars.InitMidpoint().apply_to(ru)

        return ru

    for unit in m.RU:

        ue = m.unit_exists[unit]

        # Create a block for each existing RED unit.
        # The rule calls the REDunit_model function to create the RED unit model.
        ue.ru = pyo.Block(rule=REDunit_model)

        def _stream_filter_unit_exists(ue, val):
            """
            This function filters the streams that are connected to the existing RED unit.

            Parameters
            ----------
            ue : block
                The block of the existing RED unit
            val : tuple
                The stream tuple

            Returns
            -------
            bool
                True if the stream is connected to the existing RED unit, False otherwise
            """
            x, y = val
            return x == unit or y == unit

        ue.streams = pyo.Set(
            initialize=m.RU_streams,
            filter=_stream_filter_unit_exists,
            # filter=lambda _, x, y: x == unit or y == unit
        )

        def _flow_vol(ue, i, k, sol):
            flow_init = pyo.value(
                36 * m.vel_init[sol] * m._cross_area[sol] * m.cell_pairs
            )
            if sol == 'HC':
                if flow_init * m.nr > flow_conc_data['feed_flow_vol']['fh1']:
                    return flow_conc_data['feed_flow_vol']['fh1'] / m.nr
                return flow_init
            if sol == 'LC':
                if flow_init * m.nr > flow_conc_data['feed_flow_vol']['fl1']:
                    return flow_conc_data['feed_flow_vol']['fl1'] / m.nr
                return flow_init

        def _flow_vol_b(ue, i, k, sol):
            lb = pyo.value(36 * m.vel_lb[sol] * m._cross_area[sol] * m.cell_pairs)
            ub = pyo.value(36 * m.vel_ub[sol] * m._cross_area[sol] * m.cell_pairs)
            return (lb, ub)

        ue.flow_vol = pyo.Var(
            ue.streams,
            m.SOL,
            doc="Volumetric flow rate [m3 h-1]",
            domain=pyo.NonNegativeReals,
            bounds=_flow_vol_b,
            initialize=_flow_vol,
        )

        def _conc_mol_b(ue, i, k, sol):
            if sol == 'HC':
                return (m.conc_mol_eq.lb, flow_conc_data['feed_conc_mol']['fh1'])
            return (flow_conc_data['feed_conc_mol']['fl1'], m.conc_mol_eq.ub)

        ue.conc_mol = pyo.Var(
            ue.streams,
            m.SOL,
            doc="Molar concentration [mol L-1]",
            domain=pyo.NonNegativeReals,
            bounds=_conc_mol_b,
            initialize=lambda _, i, k, sol: (
                flow_conc_data['feed_conc_mol']['fh1']
                if sol == 'HC'
                else flow_conc_data['feed_conc_mol']['fl1']
            ),
        )

        @ue.Constraint(doc='Net power output RU')
        def _net_power(ue):
            return ue.ru.NP * 1e-2 == m.NP[unit]

        @ue.Constraint(doc='RU Stack Capital Cost [USD]')
        def _stack_cap_cost(ue):
            # [USD] Stack + electrodes = 51.7% IEMs
            return m.stack_cost[unit] == m.iems_cap_cost * (1 + 0.517)

        @ue.Constraint(doc='Operating Cost RU [USD y-1]')
        def _operating_cost(ue):
            return m.operating_cost[
                unit
            ] == m.iems_cap_cost * m.CRFm + m.electricity_price * 8760 * m.load_factor * ureg.convert(
                ue.ru.PP, 'W', 'kW'
            )

        # =============================================================================
        #     Boundary Conditions and Material Balances
        # =============================================================================

        ue.bound_con = pyo.ConstraintList(doc='Boundary conditions')
        ue.balances_con = pyo.ConstraintList(doc='Material balances')

        ue.flow_vol_bound = pyo.ConstraintList(
            doc='Volumetric flow rate bounds to/from active RU'
        )

        for sol in m.SOL:
            [
                ue.flow_vol_bound.add(
                    (
                        ue.flow_vol[rm, unit, sol].lb,
                        m._flow_into[rm, sol],
                        ue.flow_vol[rm, unit, sol].ub,
                    )
                )
                for rm, r in m.RMU_RU_streams
                if r == unit
            ]  # Upper and lower bounds for flow into the RED unit

            [
                ue.bound_con.add(
                    ureg.convert(ue.ru.flow_vol_x[0, sol], 'liter', 'm**3')
                    * m.cell_pairs
                    == ue.flow_vol[rm, unit, sol]
                )
                for (rm, r) in m.RMU_RU_streams
                if r == unit
            ]  # The inlet flow rate to the RED unit is equal to the feed's flow rate equally distributed among the cell pairs
            [
                ue.bound_con.add(
                    ureg.convert(
                        ue.ru.flow_vol_x[ue.ru.length_domain.last(), sol],
                        'liter',
                        'm**3',
                    )
                    * m.cell_pairs
                    == ue.flow_vol[unit, rs, sol]
                )
                for (r, rs) in m.RU_RSU_streams
                if r == unit
            ]  # The outlet flow rate from the RED unit is equal to the flow rate of the stream leaving the set of cell pairs.
            [
                ue.bound_con.add(ue.ru.conc_mol_x[0, sol] == ue.conc_mol[rm, unit, sol])
                for (rm, r) in m.RMU_RU_streams
                if r == unit
            ]  # The inlet concentration to the cell pairs is equal to the inlet concentration of the RED unit.
            [
                ue.bound_con.add(
                    ue.ru.conc_mol_x[ue.ru.length_domain.last(), sol]
                    == ue.conc_mol[unit, rs, sol]
                )
                for (r, rs) in m.RU_RSU_streams
                if r == unit
            ]  # The outlet concentration from the cell pairs is equal to the outlet concentration of the RED unit.
            ue.bound_con.add(ue.ru.pressure_x[0, sol] == ureg.convert(1, 'atm', 'mbar'))

            [
                ue.balances_con.add(m._flow_into[rm, sol] == ue.flow_vol[rm, unit, sol])
                for rm, r in m.RMU_RU_streams
                if r == unit
            ]  # Flowrate balance for the RED unit mixers
            [
                ue.balances_con.add(
                    m._conc_into[rm, sol]
                    == ue.conc_mol[rm, unit, sol] * ue.flow_vol[rm, unit, sol]
                )
                for rm, r in m.RMU_RU_streams
                if r == unit
            ]  # Concentration balance for the RED unit mixers

            [
                ue.balances_con.add(
                    ue.flow_vol[unit, rs, sol] == m._flow_out_from[rs, sol]
                )
                for r, rs in m.RU_RSU_streams
                if r == unit
            ]  # Flowrate balance for the RED unit splitters
            [
                ue.balances_con.add(
                    ue.conc_mol[src2, sink2, sol] == m.conc_mol[src1, sink1, sol]
                )
                for src1, sink1 in m.from_splitters
                for src2, sink2 in m.RU_RSU_streams
                if src2 == unit and src1 == sink2
            ]  # Concentration balance for the RED unit splitters

        ua = m.unit_absent[unit]

        ua._no_flow = pyo.ConstraintList(doc='No-flow constraint')
        ua._no_conc = pyo.ConstraintList(doc="Concentration into RU")
        for sol in m.SOL:
            [
                ua._no_flow.add(m.flow_vol[src, ri, sol] == 0)
                for src, ri in ['rsu'] * m.in_RU
                if (ri, unit) in m.RMU_RU_streams
            ]  # No flow into the RED unit
            [
                ua._no_conc.add(
                    m.conc_mol[ro, sink, sol] == m.conc_mol[ro, sink, sol].lb
                )
                for ro, sink in m.out_RU * m.in_RU | m.out_RU * ['rmu']
                if (unit, ro) in m.RU_RSU_streams
            ]  # Outlet stream concentration set to lower bound.

        ua._no_net_power = pyo.Constraint(
            doc='No Net Power Output', expr=m.NP[unit] == 0
        )

        ua._no_costs = pyo.ConstraintList(doc='No capital and operating costs')
        ua._no_costs.add(m.stack_cost[unit] == 0)  # Stack capital cost set to 0
        ua._no_costs.add(m.operating_cost[unit] == 0)  # Operating cost set to 0

    # Sets the midpoint between the lower and upper bounds of the uninitialized variables
    init_vars.InitMidpoint().apply_to(m)

    @m.Expression(doc='Total Net Power Output active RU [kW]')
    def TNP(m):
        """
        This expression calculates the total net power output of the active RED unit.

        Parameters
        ----------
        m : Pyomo model
            The Pyomo GDP model of the RED system.

        Returns
        -------
        Pyomo.Expression
            Total Net Power Output active RU [kW] as the sum of the net power output of the RED units
        """
        return ureg.convert(sum(m.NP[ru] * 1e2 for ru in m.RU), 'W', 'kW')

    @m.Expression(
        doc='Total Net Specific Energy per m3 of HC and LC inlet streams to active RU [kWh m-3]'
    )
    def TNSE(m):
        """
        This expression calculates the total net specific energy per m3 of the high and low salinity inlet streams to the active RED unit.

        Parameters
        ----------
        m : Pyomo model
            The Pyomo GDP model of the RED system.

        Returns
        -------
        Pyomo.Expression
            Total Net Specific Energy per m3 of HC and LC inlet streams to active RU [kWh m-3] as the ratio between the total net power output and the total volumetric flow rate of the high and low salinity streams
        """
        return m.TNP / sum(m._flow_into['rsu', sol] for sol in m.SOL)

    # New var. for pump capital cost: Z = sum(Q)**0.9, Z >= 0
    # The new variable is introduced to avoid numerical issues with the concave term in the pump capital cost correlation.
    m.pump_cap_cost_var = pyo.Var(
        m.SOL,
        domain=pyo.NonNegativeReals,
        initialize=pyo.value(
            ureg.convert(m._flow_out_from['rsu', sol], 'm**3/hour', 'liter/sec') ** 0.9
        ),
        bounds=lambda _, sol: (
            None,
            ureg.convert(
                sum(m.flow_vol['rsu', ri, sol].ub for ri in m.in_RU),
                'm**3/hour',
                'liter/sec',
            )
            ** 0.9,
        ),
        doc="New var for potential term in pump capital cost correlation",
    )

    # Z^(1/0.9)=sum(Q)
    @m.Constraint(m.SOL, doc='New var potential term in pump capital cost cstr.')
    def _pump_cap_cost_nv(m, sol):
        """
        This constraint defines the potential term in the pump capital cost equation.

        Parameters
        ----------
        m : Pyomo model
            The Pyomo GDP model of the RED system.
        sol : set
            High or low salinity stream (HC or LC) index

        Returns
        -------
        Pyomo.Constraint
            Potential term in the pump capital cost correlation
        """
        return m.pump_cap_cost_var[sol] ** (1 / 0.9) == ureg.convert(
            m._flow_out_from['rsu', sol], 'm**3/hour', 'liter/sec'
        )

    @m.Expression(doc='Pumps Capital Cost [USD]')
    def pump_cap_cost(m):
        """
        This expression calculates the pumps capital cost based on the correlation from Sinnot and Towler (2012).

        Parameters
        ----------
        m : Pyomo model
            The Pyomo GDP model of the RED system.

        Returns
        -------
        Pyomo.Expression
            Pumps Capital Cost [USD] as the sum of the pump capital costs for the high and low salinity streams
        """
        #         [0.72–453.6] [m3 h-1]; CE = a + b(S)**n; S [0.2–126] [L s-1] Sinnot and Towler Single Stage Centrifugal
        return (
            sum(6900 + 206 * m.pump_cap_cost_var[sol] for sol in m.SOL)
            * m.cost_index_ratio
        )

    @m.Expression(doc='Total capital expenses [USD]')
    def CAPEX(m):
        """
        This expression calculates the total capital expenses of the RED system as the sum of the stack and pump capital costs and civil and infrastructure costs.
        """
        return (
            sum(m.stack_cost[ru] for ru in m.RU)  # Cost of the stack and electrodes
            + m.pump_cap_cost  # Cost of the pumps
            + m.TNP * 250 * m.eur2usd  # Civil and infrastructure costs [USD kW-1]
        )

    @m.Expression(doc='Total operational expenses [USD y-1]')
    def OPEX(m):
        """
        This expression calculates the total operational expenses of the RED system as the sum of the operating costs and the 2% of the capital expenses for the operation and maintenance.

        Parameters
        ----------
        m : Pyomo model
            The Pyomo GDP model of the RED system.

        Returns
        -------
        Pyomo.Expression
            Total operational expenses [USD y-1].
        """
        return (
            sum(m.operating_cost[ru] for ru in m.RU) + 0.02 * m.CAPEX
        )  # O&M = 2%–4% CAPEX [USD y-1]

    @m.Expression(doc='Total Annualized Cost [USD y-1]')
    def TAC(m):
        """
        This expression calculates the total annualized cost of the RED system as the sum of the annualized capital cost and operational expenses.

        Parameters
        ----------
        m : Pyomo model
            The Pyomo GDP model of the RED system.

        Returns
        -------
        Pyomo.Expression
            Total Annualized Cost [USD y-1].
        """
        return m.CRF * m.CAPEX + m.OPEX

    @m.Expression(doc='Net Energy Yield [kWh y-1]')
    def net_energy_yield(m):
        """
        This expression defines the annual net energy yield of the RED system.

        Parameters
        ----------
        m : Pyomo model
            The Pyomo GDP model of the RED system.

        Returns
        -------
        Pyomo.Expression
            Net Energy Yield [kWh y-1].
        """
        return 8760 * m.load_factor * m.TNP

    @m.Expression(doc='Net Present Value [kUSD]')
    def NPV(m):
        """
        This function calculates the Net Present Value (NPV) of the RED unit.
        The benefits are the revenue from the electricity sold.
        The TAC is the total annualized cost, which includes the capital and operating costs of the RED system.

        Parameters
        ----------
        m : Pyomo model
            The Pyomo GDP model of the RED system.

        Returns
        -------
        Pyomo.Expression
            Net Present Value [kUSD] as the difference between the benefits and the costs.
        """
        # Assuming energy produced w selling price eq ep [USD]
        return (m.electricity_price * m.net_energy_yield - m.TAC) * 1e-3 / m.CRF

    @m.Expression(doc='Levelized Cost of Energy [USD kWh-1]')
    def LCOE(m):
        """
        This function calculates the Levelized Cost of Energy (LCOE) of the RED unit.

        Parameters
        ----------
        m : Pyomo model
            The Pyomo GDP model of the RED system.

        Returns
        -------
        Pyomo.Expression
            Levelized Cost of Energy [USD kWh-1] as the ratio between the total annualized cost and the net energy yield.
        """
        return m.TAC / m.net_energy_yield

    # Objective function is to maximize the NPV
    m.obj = pyo.Objective(expr=m.NPV, sense=pyo.maximize)

    print("GDP model built successfully.")

    return m


if __name__ == "__main__":
    build_model()
