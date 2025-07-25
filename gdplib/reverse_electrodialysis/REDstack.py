"""
This module contains the implementation of the RED stack model using Pyomo.

The backward finite difference method is used to discretize the differential-algebraic model.

The decision variables are the flow rate, concentration, and electric current.
The objective function is to maximize the net power output.
We use the optimal solution to initialize the optimization model of the RED process.

References
----------
Tristán, C., Fallanza, M., Ibáñez, R., Ortiz, I., & Grossmann, I. E. (2023). A generalized disjunctive programming model for the optimal design of reverse electrodialysis process for salinity gradient-based power generation. Computers & Chemical Engineering, 174, 108196. https://doi.org/https://doi.org/10.1016/j.compchemeng.2023.108196
Tristán, C., Fallanza, M., Ibáñez, R., & Ortiz, I. (2020). Recovery of salinity gradient energy in desalination plants by reverse electrodialysis. Desalination, 496, 114699. https://doi.org/10.1016/j.desal.2020.114699
"""

import pyomo.environ as pyo
from pint import UnitRegistry

ureg = UnitRegistry()

import os

import numpy as np
import pandas as pd
from scipy.constants import physical_constants

wnd_dir = os.path.dirname(os.path.realpath(__file__))

# Data files containing the stack parameters, feed concentration, and temperature.
stack_param = pd.read_csv(os.path.join(wnd_dir, "stack_param.csv"))
flow_conc_data = pd.read_csv(os.path.join(wnd_dir, "flow_conc_data.csv"), index_col=0)
T = pd.read_csv(os.path.join(wnd_dir, "T.csv"))


def build_REDstack():
    '''
    This function builds the RED stack model using the data from the stack_param.csv, flow_conc_data.csv, and T.csv files.

    Returns
    -------
    m : pyomo.ConcreteModel
        RED stack model

    '''
    m = pyo.ConcreteModel('RED model')

    # ============================================================================
    #     Sets
    # =============================================================================

    m.SOL = pyo.Set(
        doc="Set of High- and Low-concentration streams",
        initialize=['HC', 'LC'],
        ordered=True,
    )

    m.iem = pyo.Set(doc='Ion-exchange membrane type', initialize=['AEM', 'CEM'])

    m.port = pyo.Set(
        doc='Inlet and Outlet RU Ports', initialize=['rm', 'rs'], ordered=True
    )

    # ============================================================================
    #     Constant parameters
    # =============================================================================

    m.gas_constant = physical_constants['molar gas constant'][
        0
    ]  # Ideal gas constant [J/mol/K]
    m.faraday_constant = physical_constants['Faraday constant'][
        0
    ]  # Faraday’s Constant [C/mol] [A*s/mol]
    m.Tref = 298.15  # Reference temperature [K]

    m.T = pyo.Param(
        doc='Feed streams temperature [K]',
        within=pyo.NonNegativeReals,
        initialize=T.loc[0].values[0],
        mutable=True,
    )

    m.temperature_coeff = pyo.Param(
        doc='Temperature correction factor [-] of the solution conductivity',
        default=0.02,
        initialize=0.02,
    )  # Mehdizadeh, et al. (2019) Membranes, 9(6), 73. https://doi.org/10.3390/membranes9060073
    # Linear temperature dependence of the solution conductivity. The temperature coefficient of the solution conductivity is 0.02/K.

    m.dynamic_viscosity = pyo.Param(
        doc='Dynamic viscosity of the solution [Pa s]', default=1e-3, initialize=1e-3
    )

    m.pump_eff = pyo.Param(doc='Pump efficiency [-]', default=0.75, initialize=0.75)

    # =============================================================================
    #     RED Stack Parameters
    # =============================================================================

    m.b = pyo.Param(
        doc="Channel's width = IEMs [m]",
        within=pyo.NonNegativeReals,
        default=0.456,
        initialize=stack_param.width.values[0],
    )
    m.L = pyo.Param(
        doc="Channel's length = IEMs [m]",
        within=pyo.NonNegativeReals,
        default=0.383,
        initialize=stack_param.length.values[0],
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
    )
    m.cell_pairs = pyo.Param(
        within=pyo.NonNegativeIntegers,
        doc="Number of Cell Pairs [-]",
        default=1e3,
        initialize=stack_param.cell_pairs.values[0],
    )

    @m.Param(m.SOL, doc="Channel's hydraulic diameter [m]", mutable=True)
    def dh(m, s):
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
        return m.b * m.L

    @m.Param(m.SOL, doc='Cross-sectional area [m2]')
    def _cross_area(m, sol):
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

    @m.Param(doc="Avg. membranes' permselectivity")
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
        doc="NaCl Membranes' diffusivity [m2/s]",
        within=pyo.NonNegativeReals,
        default=4.52e-12,
        initialize=4.52e-12,
    )

    m.vel_ub = pyo.Param(
        m.SOL,
        doc='Max. linear crossflow velocity [cm/s]',
        default=3.0,
        initialize=stack_param.vel_ub.values[0],
    )

    m.vel_lb = pyo.Param(
        m.SOL, doc='Min. linear crossflow velocity [cm/s]', initialize=0.01
    )

    m.vel_init = pyo.Param(
        m.SOL,
        doc='Min. linear crossflow velocity [cm/s]',
        default=1.0,
        initialize=stack_param.vel_init.values[0],
    )

    # =============================================================================
    #     RED Stack Model
    # =============================================================================

    # Number of finite elements
    nfe = 5

    m.length_domain = pyo.Set(
        bounds=(0.0, 1.0),
        initialize=sorted(np.linspace(0.0, 1.0, nfe + 1)),
        doc="Normalized length domain",
    )

    # The following functions are used to discretize the differential-algebraic model
    # Based on the backward finite difference method and inspired by Pyomo.DAE
    def _int_trap_rule(x, v):
        """
        This function computes the integral of a function using the trapezoidal rule.

        Parameters
        ----------
        x : int
            Index of the length domain, i.e., the position in the length domain
        v : var
            Variable to be integrated

        Returns
        -------
        Expression
            Integral of the variable v
        """
        ds = sorted(x)
        a = list(v.values())
        return sum(
            0.5 * (ds[i + 1] - ds[i]) * (a[i + 1] + a[i]) for i in range(len(ds) - 1)
        )

    def _int_trap_rule_sol(m, x, sol, v):
        """
        This function computes the integral of a function using the trapezoidal rule for the high and low concentration channels.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Index of the length domain, i.e., the position in the length domain
        sol : str
            Index of the high and low concentration compartments, i.e., 'HC' and 'LC'
        v : str
            Variable to be integrated

        Returns
        -------
        Expression
            Integral of the variable v in the high and low concentration compartments
        """
        ds = sorted(x)  # Sorted list of the length domain
        return sum(
            0.5 * (ds[i + 1] - ds[i]) * (v(m, ds[i + 1], sol) + v(m, ds[i], sol))
            for i in range(len(ds) - 1)
        )  # Trapezoidal rule

    def _bwd_fun(m, x, sol, v, dv):
        """
        This function computes the backward finite difference of the differential equation.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments
        v : str
            Variable to be discretized
        dv : str
            The derivative of the variable v

        Returns
        -------
        Expression
            The discretized differential equation
        """
        tmp = list(m.length_domain)  # List of the length domain
        idx = m.length_domain.ord(x) - 1  # Position in the length domain
        if idx != 0:
            return (
                dv(m, tmp[idx], sol)
                - 1
                / (tmp[idx] - tmp[idx - 1])
                * (v(m, tmp[idx], sol) - v(m, tmp[idx - 1], sol))
                == 0
            )  # Calculate the backward finite difference

    def _flow_vol(m, x, sol):
        """
        This function initializes the volumetric flow rate based on the initial velocity according to the continuity equation.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        int
            The initial volumetric flow rate
        """
        init = ureg.convert(m.vel_init[sol], 'cm/second', 'm/hour') * m._cross_area[sol]
        return ureg.convert(init, 'm**3', 'liter')

    def _flow_vol_b(m, x, sol):
        """
        This function sets the bounds of the volumetric flow rate based on the initial velocity according to the continuity equation.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments
        Returns
        -------
        tuple
            The bounds of the volumetric flow rate
        """
        ub = pyo.value(ureg.convert(m.vel_ub[sol], 'cm/s', 'm/h') * m._cross_area[sol])
        return (None, ureg.convert(ub, 'm**3', 'liter'))

    m.flow_vol_x = pyo.Var(
        m.length_domain,
        m.SOL,
        initialize=_flow_vol,
        bounds=_flow_vol_b,
        domain=pyo.NonNegativeReals,
        doc="Discretized Volumetric Flow Rate [L/h]",
    )

    def _flowrate_ratio_b(m):
        """
        This function sets the bounds of the flow rate ratio based on the bounds of the velocity in the high and low concentration compartments.
        The flowrate ratio is the ratio of the flow rate in the low concentration compartment to the total flow rate in the high and low concentration compartments.
        It is used to calculate the concentration of the mixed stream reaching equilibrium.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        tuple
            The bounds of the flow rate ratio
        """
        lb = m.vel_lb['LC'] / (m.vel_lb['LC'] + m.vel_ub['HC'])
        ub = m.vel_ub['LC'] / (m.vel_ub['LC'] + m.vel_lb['HC'])
        return (lb, ub)

    def _flowrate_ratio(m):
        """
        This function initializes the flow rate ratio based on the initial velocity in the low concentration compartment.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        int
            The initial flow rate ratio
        """
        return m.vel_init['LC'] / sum(m.vel_init[sol] for sol in m.SOL)

    m.phi = pyo.Var(
        doc='Vol. flow rate ratio = In LC to total In (LC+HC) RU [-]',
        initialize=_flowrate_ratio,
        bounds=_flowrate_ratio_b,
        domain=pyo.NonNegativeReals,
    )

    def _conc_mol_eq_b(m):
        """ "
        This function sets the bounds of the concentration of the mixed stream reaching equilibrium based on the bounds of the flow rate ratio.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        tuple
            The bounds of the concentration of the mixed stream reaching equilibrium
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
        This function initializes the concentration of the mixed stream reaching equilibrium based on the flow rate ratio.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        int
            The initial concentration of the mixed stream reaching equilibrium
        """
        return (
            m.phi * flow_conc_data['feed_conc_mol']['fl1']
            + (1 - m.phi) * flow_conc_data['feed_conc_mol']['fh1']
        )

    m.conc_mol_eq = pyo.Var(
        doc='Concentration of the HC and LC mixed stream reaching equilibrium [mol/L]',
        initialize=_conc_mol_eq,
        bounds=_conc_mol_eq_b,
        domain=pyo.NonNegativeReals,
    )

    def _conc_molx_b(m, x, sol):
        """
        This function sets the bounds of the molar concentration based on the feed concentration in the high and low concentration compartments.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        tuple
            The bounds of the molar concentration
        """
        if sol == 'HC':
            return (m.conc_mol_eq.lb, flow_conc_data['feed_conc_mol']['fh1'])
        return (flow_conc_data['feed_conc_mol']['fl1'], m.conc_mol_eq.ub)

    def _conc_molx(m, x, sol):
        """
        This function initializes the molar concentration based on the feed concentration in the high and low concentration compartments.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        int
            The initial molar concentration
        """
        if sol == 'HC':
            return flow_conc_data['feed_conc_mol']['fh1']
        return flow_conc_data['feed_conc_mol']['fl1']

    m.conc_mol_x = pyo.Var(
        m.length_domain,
        m.SOL,
        initialize=_conc_molx,
        bounds=_conc_molx_b,
        doc="Discretized Molar NaCl concentration [mol/L]",
        domain=pyo.NonNegativeReals,
    )

    def _pressure_x(m, x, sol):
        """
        This function initializes the pressure in the high and low concentration compartments
        If the position is the first in the length domain, the pressure is set to the upper bound.
        If not, the pressrure is equal to the upper bound minus the pressure drop.
        The ppressure drop is computed using the Darcy-Weisbach equation considering the presence of the spacer.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        int
            The initial pressure
        """
        delta_p = pyo.value(
            48
            * m.dynamic_viscosity
            * ureg.convert(m.vel_init[sol], 'cm', 'm')
            / m.dh[sol] ** 2
            * m.L
        )
        ub = ureg.convert(1, 'atm', 'mbar')
        lb = ub - ureg.convert(delta_p, 'Pa', 'mbar')
        if x == m.length_domain.first():
            return ub
        return lb

    def _pressure_x_b(m, x, sol):
        """
        This function sets the bounds of the pressure in the high and low concentration compartments.
        The lower bound is computed using the Darcy-Weisbach equation considering the presence of the spacer.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain

        Returns
        -------
        tuple
            The bounds of the pressure
        """
        delta_p = pyo.value(
            48
            * m.dynamic_viscosity
            * ureg.convert(m.vel_ub[sol], 'cm', 'm')
            / m.dh[sol] ** 2
            * m.L
        )
        ub = ureg.convert(1, 'atm', 'mbar')
        lb = ub - ureg.convert(delta_p, 'Pa', 'mbar')
        return (lb, ub)

    m.pressure_x = pyo.Var(
        m.length_domain,
        m.SOL,
        domain=pyo.NonNegativeReals,
        initialize=_pressure_x,
        bounds=_pressure_x_b,
        doc='Discretized pressure [mbar]',
    )

    m.flow_vol_dx = pyo.Var(
        m.flow_vol_x.index_set(),
        doc="Derivative of volumetric flow wrt to normalized length",
        bounds=(-1.0, 1.0),
        initialize=0,
    )

    m.conc_mol_dx = pyo.Var(
        m.conc_mol_x.index_set(),
        doc="Derivative of molar concentration wrt to normalized length",
        bounds=(-1.0, 1.0),
        initialize=0,
    )

    m.pressure_dx = pyo.Var(
        m.pressure_x.index_set(),
        doc="Derivative of pressure wrt to normalized length",
        domain=pyo.NonPositiveReals,
        bounds=lambda _, x, sol: (
            ureg.convert(
                -48
                * m.dynamic_viscosity
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
            if x == m.length_domain.first()
            else ureg.convert(
                -48
                * m.dynamic_viscosity
                * ureg.convert(m.vel_init[sol], 'cm', 'm')
                / m.dh[sol] ** 2
                * m.L,
                'Pa',
                'mbar',
            )
        ),
    )

    m.flow_vol = pyo.Var(
        m.port,
        m.SOL,
        doc="Volumetric flow rate [m3/h]",
        initialize=lambda _, p, sol: m.cell_pairs
        * ureg.convert(m.vel_init[sol], 'cm/s', 'm/h')
        * m._cross_area[sol],
        domain=pyo.NonNegativeReals,
        bounds=lambda _, p, sol: (
            m.cell_pairs
            * ureg.convert(m.vel_lb[sol], 'cm/s', 'm/h')
            * m._cross_area[sol],
            m.cell_pairs
            * ureg.convert(m.vel_ub[sol], 'cm/s', 'm/h')
            * m._cross_area[sol],
        ),
    )

    def _conc_mol_b(m, p, sol):
        """
        This function sets the bounds of the molar concentration based on the feed concentration in the high and low concentration compartments.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        p : str
            The inlet and outlet RU ports
        sol : str
            The high and low concentration compartments

        Returns
        -------
        tuple
            The bounds of the molar concentration
        """
        if sol == 'HC':
            return (m.conc_mol_eq.lb, flow_conc_data['feed_conc_mol']['fh1'])
        return (flow_conc_data['feed_conc_mol']['fl1'], m.conc_mol_eq.ub)

    def _conc_mol(m, p, sol):
        """
        This function initializes the molar concentration based on the feed concentration in the high and low concentration compartments.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        p : str
            The inlet and outlet RU ports
        sol : str
            The high and low concentration compartments

        Returns
        -------
        int
            The initial molar concentration
        """
        if sol == 'HC':
            return flow_conc_data['feed_conc_mol']['fh1']
        return flow_conc_data['feed_conc_mol']['fl1']

    m.conc_mol = pyo.Var(
        m.port,
        m.SOL,
        doc="Molar concentration [mol/L]",
        domain=pyo.NonNegativeReals,
        initialize=_conc_mol,
        bounds=_conc_mol_b,
    )

    # =============================================================================
    #     Electric variables
    # =============================================================================

    m.Ecpx = pyo.Var(
        m.length_domain,
        domain=pyo.NonNegativeReals,
        initialize=lambda _, x: ureg.convert(
            2
            * m.gas_constant
            * m.T
            / m.faraday_constant
            * m.iems_permsel_avg
            * (pyo.log(m.conc_mol_x[x, 'HC']) - pyo.log(m.conc_mol_x[x, 'LC'])),
            'V',
            'mV',
        ),
        bounds=lambda _, x: (
            None,
            ureg.convert(
                2
                * m.gas_constant
                * m.T
                / m.faraday_constant
                * m.iems_permsel_avg
                * (
                    pyo.log(flow_conc_data['feed_conc_mol']['fh1'])
                    - pyo.log(flow_conc_data['feed_conc_mol']['fl1'])
                ),
                'V',
                'mV',
            ),
        ),
        doc="Nernst ELectric Potential per cell pair [mV per cell pair]",
    )

    m.EMF = pyo.Var(
        domain=pyo.NonNegativeReals,
        initialize=m.cell_pairs
        * ureg.convert(_int_trap_rule(m.length_domain, m.Ecpx), 'mV', 'V'),
        bounds=(None, m.cell_pairs * ureg.convert(m.Ecpx[0].ub, 'mV', 'V')),
        doc="Nernst Potential RED Stack [V]",
    )

    def _ksol_b(m, x, sol):
        """
        This function sets the bounds of the solution conductivity based on the feed concentration in the high and low concentration compartments.
        The expression is derived from the linear regression of the experimental data.
        Tristán et al. (2020) Desalination, 496, 114699. https://doi.org/10.1016/j.desal.2020.114699

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        tuple
            The bounds of the solution conductivity
        """
        # Conductivity bounds based on the feed concentration in the high and low concentration compartments
        # The expression is derived from the linear regression of the experimental data (Tristán et al., 2020)
        if sol == 'HC':
            ub = 7.7228559 * flow_conc_data['feed_conc_mol']['fh1'] + 0.5670209
            lb = 7.7228559 * m.conc_mol_eq.lb + 0.5670209
            return (lb, ub)
        ub = 10.5763914 * m.conc_mol_eq.ub + 0.0087379
        lb = 10.5763914 * flow_conc_data['feed_conc_mol']['fl1'] + 0.0087379
        return (lb, ub)

    def _ksol(m, x, sol):
        """
        This function initializes the solution conductivity based on the feed concentration in the high and low concentration compartments.
        The expression is derived from the linear regression of the experimental data.
        Tristán et al. (2020) Desalination, 496, 114699. https://doi.org/10.1016/j.desal.2020.114699

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        int
            The initial solution conductivity
        """
        if sol == 'HC':
            return pyo.value(7.7228559 * m.conc_mol_x[0, 'HC'] + 0.5670209)
        return pyo.value(10.5763914 * m.conc_mol_x[0, 'LC'] + 0.0087379)

    m.ksol = pyo.Var(
        m.length_domain,
        m.SOL,
        domain=pyo.NonNegativeReals,
        bounds=_ksol_b,
        initialize=_ksol,
        doc="Sol. conductivity per unit length [S/m]",
    )

    m.ksol_T = pyo.Var(
        m.length_domain,
        m.SOL,
        domain=pyo.NonNegativeReals,
        bounds=lambda _, x, sol: (
            m.ksol[x, sol].lb * (1 + m.temperature_coeff * (m.T - m.Tref)),
            m.ksol[x, sol].ub * (1 + m.temperature_coeff * (m.T - m.Tref)),
        ),
        initialize=lambda _, x, sol: m.ksol[x, sol]
        * (1 + m.temperature_coeff * (m.T - m.Tref)),
        doc="Temperature corrected sol. conductivity per unit length [S/m]",
    )

    def _Rsol_b(m, x, sol):
        """
        This function sets the bounds of the solution resistance based on the conductivity and the thickness and porosity of the spacer in the high and low concentration compartments.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        tuple
            The bounds of the solution resistance
        """
        if sol == 'HC':
            lb = ureg.convert(
                m.spacer_thickness[sol]
                / m.spacer_porosity[sol] ** 2
                / m.ksol_T[0, 'HC'].ub,
                'ohm*m**2',
                'ohm*cm**2',
            )
            ub = ureg.convert(
                m.spacer_thickness[sol]
                / m.spacer_porosity[sol] ** 2
                / m.ksol_T[0, 'HC'].lb,
                'ohm*m**2',
                'ohm*cm**2',
            )
            return (lb, ub)
        lb = ureg.convert(
            m.spacer_thickness[sol]
            / m.spacer_porosity[sol] ** 2
            / m.ksol_T[0, 'LC'].ub,
            'ohm*m**2',
            'ohm*cm**2',
        )
        ub = ureg.convert(
            m.spacer_thickness[sol]
            / m.spacer_porosity[sol] ** 2
            / m.ksol_T[0, 'LC'].lb,
            'ohm*m**2',
            'ohm*cm**2',
        )
        return (lb, ub)

    def _Rsol(m, x, sol):
        """
        This function initializes the solution resistance based on the conductivity and the thickness and porosity of the spacer in the high and low concentration compartments.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        int
            The initial solution resistance
        """
        if sol == 'HC':
            return pyo.value(
                ureg.convert(
                    m.spacer_thickness[sol]
                    / m.spacer_porosity[sol] ** 2
                    / m.ksol_T[0, 'HC'],
                    'ohm*m**2',
                    'ohm*cm**2',
                )
            )
        return pyo.value(
            ureg.convert(
                m.spacer_thickness[sol]
                / m.spacer_porosity[sol] ** 2
                / m.ksol_T[0, 'LC'],
                'ohm*m**2',
                'ohm*cm**2',
            )
        )

    m.Rsol = pyo.Var(
        m.length_domain,
        m.SOL,
        domain=pyo.NonNegativeReals,
        bounds=_Rsol_b,
        initialize=_Rsol,
        doc="Solution resistance per cell pair per unit length [ohm cm2 per cp]",
    )

    def _Rcpx_b(m, x):
        """
        This function sets the bounds of the internal resistance per cell pair based on the resistance of the solution and the resistance of the membranes.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain

        Returns
        -------
        tuple
            The bounds of the internal resistance per cell pair
        """
        lb = sum(m.Rsol[0, sol].lb for sol in m.SOL) + ureg.convert(
            sum(m.iems_resistance[iem] for iem in m.iem), 'ohm*m**2', 'ohm*cm**2'
        )
        ub = sum(m.Rsol[0, sol].ub for sol in m.SOL) + ureg.convert(
            sum(m.iems_resistance[iem] for iem in m.iem), 'ohm*m**2', 'ohm*cm**2'
        )
        return (lb, ub)

    def _Rcpx(m, x):
        """
        This function initializes the internal resistance per cell pair based on the resistance of the solution and the resistance of the membranes.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain

        Returns
        -------
        int
            The initial internal resistance per cell pair
        """
        return sum(m.Rsol[0, sol] for sol in m.SOL) + ureg.convert(
            sum(m.iems_resistance[iem] for iem in m.iem), 'ohm*m**2', 'ohm*cm**2'
        )

    m.Rcpx = pyo.Var(
        m.length_domain,
        domain=pyo.NonNegativeReals,
        initialize=_Rcpx,
        bounds=_Rcpx_b,
        doc="Internal resistance per cell pair per unit length [ohm cm2 per cp]",
    )

    m.Rstack = pyo.Var(
        domain=pyo.NonNegativeReals,
        initialize=lambda _: m.cell_pairs
        * ureg.convert(_int_trap_rule(m.length_domain, m.Rcpx), 'ohm*cm**2', 'ohm*m**2')
        / m.Aiem,  # Resistance of the RED stack computed from the internal resistance using the trapezoidal rule.
        bounds=(
            m.cell_pairs * ureg.convert(m.Rcpx[0].lb, 'ohm*cm**2', 'ohm*m**2') / m.Aiem,
            m.cell_pairs * ureg.convert(m.Rcpx[0].ub, 'ohm*cm**2', 'ohm*m**2') / m.Aiem,
        ),
        doc="RED stack Internal resistance [ohm]",
    )

    m.Rload = pyo.Var(
        domain=pyo.NonNegativeReals,
        initialize=_int_trap_rule(m.length_domain, m.Rcpx),
        bounds=(0.02, 100.0),
        doc="Load resistance [ohm cm2 per cp]",
    )

    m.Idx = pyo.Var(
        m.length_domain,
        domain=pyo.NonNegativeReals,
        initialize=lambda _, x: m.Ecpx[x] / (m.Rcpx[x] + m.Rload),
        bounds=lambda _, x: (None, m.Ecpx[x].ub / (m.Rcpx[x].lb + m.Rload.lb)),
        doc="Electric Current Density [mA/cm2]",
    )

    m.Istack = pyo.Var(
        domain=pyo.NonNegativeReals,
        initialize=lambda _: ureg.convert(
            _int_trap_rule(m.length_domain, m.Idx), 'mA/cm**2', 'A/m**2'
        )
        * m.Aiem,
        bounds=(None, ureg.convert(m.Idx[0].ub, 'mA/cm**2', 'A/m**2') * m.Aiem),
        doc="Electric Current Stack [A]",
    )

    m.GP = pyo.Var(
        domain=pyo.NonNegativeReals,
        initialize=m.Istack * (m.EMF - m.Rstack * m.Istack),
        bounds=(None, 40.0e3),
        doc="Gross Power output RED stack [W]",
    )

    m.PP = pyo.Var(
        domain=pyo.NonNegativeReals,
        initialize=sum(
            48
            * m.dynamic_viscosity
            * ureg.convert(m.vel_init[sol], 'cm/s', 'm/s')
            / m.dh[sol] ** 2
            * ureg.convert(m.flow_vol['rm', sol], 'm**3/h', 'm**3/s')
            / m.pump_eff
            for sol in m.SOL
        ),
        bounds=(
            None,
            sum(
                48
                * m.dynamic_viscosity
                * ureg.convert(m.vel_ub[sol], 'cm', 'm')
                / m.dh[sol] ** 2
                * ureg.convert(m.flow_vol['rm', sol].ub, 'm**3/h', 'm**3/s')
                / m.pump_eff
                for sol in m.SOL
            ),
        ),
        doc="Pumping Power loss RED stack [W]",
    )

    m.NP = pyo.Var(
        initialize=m.GP - m.PP,
        bounds=(None, m.GP.ub - m.PP.ub),
        doc="Net Power output RED stack [W]",
    )

    # =============================================================================
    #     Material transfer terms
    # =============================================================================

    def _Jcond_b(m, x):
        """
        This function sets the bounds of the conductive molar flux which depends on the discretized electric current density.
        Units:
        Jcond [mol/m2/h], Idx [mA/cm2], faraday_constant [A*s/mol]

        Parameters
        ----------
        ru : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain

        Returns
        -------
        tuple
            The bounds of the conductive molar flux
        """
        lb = ureg.convert(m.Idx[0].lb, 'mA/cm**2', 'A/m**2') / ureg.convert(
            m.faraday_constant, 'A*s/mol', 'A*h/mol'
        )
        ub = ureg.convert(m.Idx[0].ub, 'mA/cm**2', 'A/m**2') / ureg.convert(
            m.faraday_constant, 'A*s/mol', 'A*h/mol'
        )
        return (lb, ub)

    m.Jcond = pyo.Var(
        m.length_domain,
        domain=pyo.NonNegativeReals,
        initialize=lambda _, x: ureg.convert(m.Idx[x], 'mA/cm**2', 'A/m**2')
        / ureg.convert(m.faraday_constant, 'A*s/mol', 'A*h/mol'),
        bounds=_Jcond_b,
        doc="Conductive Molar Flux (electromigration) NaCl per unit length [mol/m2/h]",
    )

    def _Jdiff_b(ru, x):
        """
        This function sets the bounds of the diffusive molar flux.
        Units:
        Jdiff [mol/m2/h], diff_nacl [m2/s], iems_thickness [m], conc_mol_x [mol/L]

        Parameters
        ----------
        ru : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain

        Returns
        -------
        tuple
            The bounds of the diffusive molar flux
        """
        lb = (
            2
            * ureg.convert(m.diff_nacl, 'm**2/s', 'm**2/h')
            / m.iems_thickness['CEM']
            * ureg.convert(
                (m.conc_mol_x[0, 'HC'].lb - m.conc_mol_x[0, 'LC'].ub),
                'mol/L',
                'mol/m**3',
            )
        )
        ub = (
            2
            * ureg.convert(m.diff_nacl, 'm**2/s', 'm**2/h')
            / m.iems_thickness['CEM']
            * ureg.convert(
                (m.conc_mol_x[0, 'HC'].ub - m.conc_mol_x[0, 'LC'].lb),
                'mol/L',
                'mol/m**3',
            )
        )
        return (lb, ub)

    m.Jdiff = pyo.Var(
        m.length_domain,
        domain=pyo.NonNegativeReals,
        initialize=lambda _, x: 2
        * ureg.convert(m.diff_nacl, 'm**2/s', 'm**2/h')
        / m.iems_thickness['CEM']
        * ureg.convert(
            (m.conc_mol_x[0, 'HC'] - m.conc_mol_x[0, 'LC']), 'mol/L', 'mol/m**3'
        ),
        bounds=_Jdiff_b,
        doc="Diffusive Molar Flux NaCl per unit length [mol/m2/h]",
    )

    m.Ji = pyo.Var(
        m.length_domain,
        domain=pyo.NonNegativeReals,
        initialize=lambda _, x: m.Jcond[x] + m.Jdiff[x],
        bounds=(None, m.Jcond[0].ub + m.Jdiff[0].ub),
        doc="Molar Flux NaCl per unit length [mol/m2/h]",
    )

    # =============================================================================
    #     Boundary Conditions
    # =============================================================================

    # The boundary conditions stablish relations between the variables at the inlet and outlet of the RED stack.
    m.bound_con = pyo.ConstraintList(doc='Boundary conditions')
    for sol in m.SOL:
        [
            m.bound_con.add(
                ureg.convert(m.flow_vol_x[0, sol], 'dm**3', 'm**3') * m.cell_pairs
                == m.flow_vol['rm', sol]
            )
        ]  # The inlet flow rate is equally distributed among the cell pairs at the inlet.
        [
            m.bound_con.add(
                ureg.convert(m.flow_vol_x[m.length_domain.last(), sol], 'dm**3', 'm**3')
                * m.cell_pairs
                == m.flow_vol['rs', sol]
            )
        ]  # The outlet flow rate is equally distributed among the cell pairs at the outlet.
        [m.bound_con.add(m.conc_mol_x[0, sol] == m.conc_mol['rm', sol])]
        # The concentration at the inlet position is equal to the concentration at the inlet port.
        [
            m.bound_con.add(
                m.conc_mol_x[m.length_domain.last(), sol] == m.conc_mol['rs', sol]
            )
        ]  # The concetration at the outlet position is equal to the concentration at the outlet port.
        [m.bound_con.add(m.pressure_x[0, sol] == ureg.convert(1, 'atm', 'mbar'))]
        # The pressure at the inlet position is set to 1 atm.

    @m.Constraint(doc='Flow rate ratio diluate to total flow rate [-]')
    def _flowrate_ratio(m):
        """
        This function sets the flow rate ratio based on the total flow rate in the high and low concentration compartments.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        pyomo.Constraint
            The flow rate ratio constraint
        """
        total_flow = sum(m.flow_vol['rm', sol] for sol in m.SOL)
        return m.phi * total_flow == m.flow_vol['rm', 'LC']

    @m.Constraint(doc="Molar concentration reaching equilibrium [mol/L]")
    def _conc_mol_eq(m):
        """
        This function sets the concentration of the mixed stream reaching equilibrium based on the flow rate ratio.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        pyomo.Constraint
            The concentration of the mixed stream reaching equilibrium constraint
        """
        return (
            m.conc_mol_eq
            == m.phi * m.conc_mol['rm', 'LC'] + (1 - m.phi) * m.conc_mol['rm', 'HC']
        )

    @m.Constraint(
        m.length_domain,
        doc='Nernst Potential per unit length per cell pair [mV per cp]',
    )
    def _nernst_potential_cp(m, x):  # Rg[J/mol/K] , F [A*s/mol], T[K]
        """
        This function computes the Nernst potential per unit length per cell pair based on the molar concentration in the high and low concentration compartments.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain

        Returns
        -------
        pyomo.Constraint
            The Nernst potential per unit length per cell pair constraint
        """
        nernst_constant = (
            2 * m.gas_constant * m.T / m.faraday_constant * m.iems_permsel_avg
        )
        return m.conc_mol_x[x, 'HC'] == m.conc_mol_x[x, 'LC'] * pyo.exp(
            m.Ecpx[x] / ureg.convert(nernst_constant, 'V', 'mV')
        )

    @m.Constraint(
        m.length_domain, m.SOL, doc="Solution's Conductivity per unit length [S/m]"
    )
    def _sol_cond(m, x, sol):
        """
        This function computes the solution conductivity per unit length based on the molar concentration in the high and low concentration compartments.
        The expression is derived from the linear regression of the experimental data.
        Tristán et al. (2020) Desalination, 496, 114699. https://doi.org/10.1016/j.desal.2020.114699

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        pyomo.Constraint
            The solution conductivity per unit length constraint
        """
        if sol == 'HC':
            return m.ksol[x, sol] == 7.7228559 * m.conc_mol_x[x, sol] + 0.5670209
        return m.ksol[x, sol] == 10.5763914 * m.conc_mol_x[x, sol] + 0.0087379

    @m.Constraint(
        m.length_domain,
        m.SOL,
        doc="Temperature corrected Solution's Conductivity per unit length [S/m]",
    )
    def _sol_cond_T(m, x, sol):
        """
        This function computes the temperature corrected solution conductivity per unit length based on the solution conductivity.
        Linear temperature dependence of the solution conductivity from Mehdizadeh, et al. (2019).

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        pyomo.Constraint
            The temperature corrected solution conductivity per unit length constraint
        """
        return m.ksol_T[x, sol] == m.ksol[x, sol] * (
            1 + m.temperature_coeff * (m.T - m.Tref)
        )

    @m.Constraint(
        m.length_domain,
        m.SOL,
        doc="Channel's resistance per cell pair per unit length [ohm cm2 per cp]",
    )
    def _channel_res(m, x, sol):
        """
        This function computes the channel resistance per cell pair per unit length based on the solution resistance and the spacer characteristics (thickness and porosity).
        Units:
        Rsol [ohm cm2 per cp], ksol_T [S/m], spacer_thickness [m], spacer_porosity [-]

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        pyomo.Constraint
            The channel resistance per cell pair per unit length constraint
        """
        return (
            m.Rsol[x, sol] * ureg.convert(m.ksol_T[x, sol], 'S/m', 'S/cm')
            == ureg.convert(m.spacer_thickness[sol], 'm', 'cm')
            / m.spacer_porosity[sol] ** 2
        )

    @m.Constraint(
        m.length_domain,
        doc="Internal resistance per cell pair per unit length [ohm cm2 per cp]",
    )
    def _int_res(m, x):
        """
        This function computes the internal resistance per cell pair per unit length based on the solution resistance and the membrane resistance.
        It only considers the resistance of the solution and the membrane.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain

        Returns
        -------
        pyomo.Constraint
            The internal resistance per cell pair per unit length constraint
        """
        return m.Rcpx[x] == sum(m.Rsol[x, sol] for sol in m.SOL) + ureg.convert(
            sum(m.iems_resistance[iem] for iem in m.iem), 'ohm*m**2', 'ohm*cm**2'
        )

    @m.Constraint(
        m.length_domain, doc='Electric current density per unit length [mA/cm2]'
    )
    def _current_dens_calc(m, x):
        """
        This function computes the electric current density per unit length based on the electric potential, the internal resistance, and the load resistance.
        Based on Kirchhoff's law.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain

        Returns
        -------
        pyomo.Constraint
            The electric current density per unit length constraint
        TODO: Revise the equation to complpy the Kirchhoff's law
        """
        return m.Idx[x] * (m.Rcpx[x] + m.Rload) == m.Ecpx[x]

    @m.Expression(
        m.length_domain, m.SOL, doc='Crossflow velocity in channel eq. [cm/s]'
    )
    def vel(m, x, sol):
        """
        This function computes the crossflow velocity in the channel based on the volumetric flow rate and the cross-sectional area of the channel.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        Pyomo.Expression
            The crossflow velocity in the channel
        """
        return ureg.convert(
            m.flow_vol_x[x, sol], 'liter/hour', 'cm**3/s'
        ) / ureg.convert(m._cross_area[sol], 'm**2', 'cm**2')

    def _vel_x(m, x, sol):
        """
        This function returns the crossflow velocity in the channel.
        It is used to compute the average crossflow velocity with the trapezoidal rule.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        Pyomo.Var
            The crossflow velocity in the channel
        """
        return m.vel[x, sol]

    @m.Expression(m.SOL, doc='Average cross-flow velocity [cm/s]')
    def vel_avg(m, sol):
        """
        This function computes the average cross-flow velocity based on the crossflow velocity in the channel.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        sol : str
            The high and low concentration compartments

        Returns
        -------
        Pyomo.Expression
            The average cross-flow velocity
        """
        v = _vel_x
        return _int_trap_rule_sol(m, m.length_domain, sol, v)

    @m.Constraint(m.length_domain, doc='Conductive molar flux (electromigration)')
    def _cond_molar_flux(m, x):
        """
        This function computes the conductive molar flux.
        Units:
        Jcond [mol/m2/h], faraday_constant [A*s/mol], Idx [mA/cm2]

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain

        Returns
        -------
        pyomo.Constraint
            The conductive molar flux constraint
        """
        return m.Jcond[x] * ureg.convert(
            m.faraday_constant, 'A*s/mol', 'A*h/mol'
        ) == ureg.convert(m.Idx[x], 'mA/cm**2', 'A/m**2')

    @m.Constraint(m.length_domain, doc='Diffusive molar flux [mol/m2/h]')
    def _diff_molar_flux(m, x):
        """
        This function computes the diffusive molar flux.
        Units:
        Jdiff [mol/m2/h], diff_nacl [m2/s], iems_thickness [m], conc_mol_x [mol/L]

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain

        Returns
        -------
        pyomo.Constraint
            The diffusive molar flux constraint
        """
        return m.Jdiff[x] == 2 * ureg.convert(
            m.diff_nacl, 'm**2/s', 'm**2/h'
        ) / m.iems_thickness['CEM'] * ureg.convert(
            (m.conc_mol_x[x, 'HC'] - m.conc_mol_x[x, 'LC']), 'mol/L', 'mol/m**3'
        )

    @m.Constraint(m.length_domain, doc='Total molar flux from HC to LC side [mol/m2/h]')
    def _total_molar_flux(m, x):
        """
        This function computes the total molar flux from the high concentration compartment to the low concentration compartment.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain

        Returns
        -------
        pyomo.Constraint
            The total molar flux from the high concentration compartment to the low concentration compartment constraint
        """
        return m.Ji[x] == m.Jcond[x] + m.Jdiff[x]

    def _flow_vol_x(m, x, sol):
        """
        This function returns the volumetric flow rate in the channel.
        It is used to transform the differential equations into algebraic equations with the backward finite difference method.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        Pyomo.Var
            The volumetric flow rate in the channel
        """
        return m.flow_vol_x[x, sol]

    def _flow_vol_dx(m, x, sol):
        """
        This function returns the derivative of the volumetric flow rate in the channel.
        It is used to transform the differential equations into algebraic equations with the backward finite difference method.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        Pyomo.Var
            The derivative of the volumetric flow rate in the channel
        """
        return m.flow_vol_dx[x, sol]

    @m.Constraint(m.length_domain, m.SOL)
    def flow_vol_dx_disc_eq(m, x, sol):
        """
        This function computes the derivative of the volumetric flow rate in the channel based on the volumetric flow rate in the channel with the backward finite difference method.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        pyomo.Constraint
            The derivative of the volumetric flow rate in the channel constraint
        """
        if x == 0:
            return pyo.Constraint.Skip
        v = _flow_vol_x
        dv = _flow_vol_dx
        return _bwd_fun(m, x, sol, v, dv)

    @m.Constraint(
        m.length_domain,
        m.SOL,
        doc='Volumetric flow rate balance w/o water transfer (i.e. no osmotic flux)',
    )
    def _flow_balance(m, x, sol):
        """
        This function computes the volumetric flow rate balance without water transfer (i.e. no osmotic flux).
        The flowrate is constant along the length of the RED stack.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        pyomo.Constraint
            The volumetric flow rate balance without water transfer constraint
        """
        if x == m.length_domain.first():
            return pyo.Constraint.Skip
        return m.flow_vol_dx[x, sol] == 0

    def _conc_mol_x(m, x, sol):
        """
        This function returns the molar concentration in the channel.
        It is used to transform the differential equations into algebraic equations with the backward finite difference method.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        Pyomo.Expression
            The molar concentration in the channel
        """
        return m.conc_mol_x[x, sol]

    def _conc_mol_dx(m, x, sol):
        """
        This function returns the derivative of the molar concentration in the channel.
        It is used to transform the differential equations into algebraic equations with the backward finite difference method.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        Pyomo.Expression
            The derivative of the molar concentration in the channel
        """
        return m.conc_mol_dx[x, sol]

    @m.Constraint(m.length_domain, m.SOL)
    def conc_mol_dx_disc_eq(m, x, sol):
        """
        This function computes the derivative of the molar concentration in the channel based on the molar concentration in the channel with the backward finite difference method.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        pyomo.Constraint
            The derivative of the molar concentration in the channel constraint
        """
        if x == 0:
            return pyo.Constraint.Skip
        v = _conc_mol_x
        dv = _conc_mol_dx
        return _bwd_fun(m, x, sol, v, dv)

    @m.Constraint(m.length_domain, m.SOL, doc='Molar concentration balance')
    def _conc_balance(m, x, sol):
        """
        This function computes the molar concentration balance.
        The concentration increases in the low concetration compartment due to the ionic transfer from the high concentration compartment.
        The concentration decreases in the high concentration compartment due to the ionic transfer to the low concentration compartment.
        Units:
        dC/dx [mol/L/m], Ji [mol/m2/h], L [m], b [m], flow_vol_x [L/h]

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        pyomo.Constraint
            The molar concentration balance constraint
        """
        if x == m.length_domain.first():
            return pyo.Constraint.Skip
        if sol == 'LC':
            return m.conc_mol_dx[x, sol] * m.flow_vol_x[x, sol] == m.b * m.Ji[x] * m.L
        return m.conc_mol_dx[x, sol] * m.flow_vol_x[x, sol] == -m.b * m.Ji[x] * m.L

    @m.Constraint(m.length_domain, doc='Concentration HC >= LC')
    def _conc_hc_gt_lc(m, x):
        """
        This constraint ensures that the molar concentration in the high concentration compartment is greater than or equal to the molar concentration in the low concentration compartment.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain

        Returns
        -------
        pyomo.Constraint
            The molar concentration in the high concentration compartment is greater than or equal to the molar concentration in the low concentration compartment constraint
        """
        return m.conc_mol_x[x, 'HC'] >= m.conc_mol_x[x, 'LC']

    @m.Expression(m.length_domain, m.SOL, doc='Pressure drop per unit length [mbar/m]')
    def _deltaP(m, x, sol):
        """
        This function computes the pressure drop per unit length based on the crossflow velocity and the hydraulic diameter of the channel.
        The Darcy-Weisbach equation is used to compute the pressure drop corrected for the presence of the spacer.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        Pyomo.Expression
            The pressure drop per unit length
        """
        return ureg.convert(
            48
            * m.dynamic_viscosity
            * ureg.convert(m.vel[x, sol], 'cm', 'm')
            / m.dh[sol] ** 2,
            'Pa',
            'mbar',
        )  # [mbar/m]

    def _pressure_x(m, x, sol):
        """
        This function returns the pressure in the channel.
        It is used to transform the differential equations into algebraic equations with the backward finite difference method.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        Pyomo.Var
            The pressure in the channel
        """
        return m.pressure_x[x, sol]

    def _pressure_dx(m, x, sol):
        """
        This function returns the derivative of the pressure in the channel.
        It is used to transform the differential equations into algebraic equations with the backward finite difference method.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        Pyomo.Var
            The derivative of the pressure in the channel
        """
        return m.pressure_dx[x, sol]

    @m.Constraint(m.length_domain, m.SOL)
    def pressure_dx_disc_eq(m, x, sol):
        """
        This function computes the discrete derivative of the pressure in the channel with the backward finite difference method.
        Calls the _bwd_fun function to compute the discrete derivative of the pressure in the channel.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        pyomo.Constraint
            The discrete derivative of the pressure in the channel constraint
        """
        if x == m.length_domain.first():
            return pyo.Constraint.Skip
        v = _pressure_x
        dv = _pressure_dx
        return _bwd_fun(m, x, sol, v, dv)

    @m.Constraint(m.length_domain, m.SOL, doc='Friction pressure drop [mbar/m]')
    def _pressure_drop(m, x, sol):
        """
        This function computes the friction pressure drop based on the crossflow velocity and the hydraulic diameter of the channel.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model
        x : int
            Position in the length domain
        sol : str
            The high and low concentration compartments

        Returns
        -------
        pyomo.Constraint
            The friction pressure drop constraint
        """
        if x == m.length_domain.first():
            return pyo.Constraint.Skip
        return m.pressure_dx[x, sol] == -m._deltaP[x, sol] * m.L

    @m.Expression(doc='Average cell pair resistance per unit length [ohm cm2 per cp]')
    def Rcp_avg(m):
        """
        This function computes the average cell pair resistance per unit length with the trapezoidal rule.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        Pyomo.Expression
            The average cell pair resistance per unit length
        """
        return _int_trap_rule(m.length_domain, m.Rcpx)

    @m.Expression(doc='Average current density [mA/cm2]')
    def Id_avg(m):
        """
        This function computes the average current density based on the electric current density with the trapezoidal rule.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        Pyomo.Expression
            The average current density
        """
        return _int_trap_rule(m.length_domain, m.Idx)

    @m.Expression(doc='Average cell pair potential per unit length [mV per cp]')
    def Ecp_avg(m):
        """
        This function computes the average cell pair potential per unit length with the trapezoidal rule.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        Pyomo.Expression
            The average cell pair potential per unit length
        """
        return _int_trap_rule(m.length_domain, m.Ecpx)

    @m.Constraint(doc='Electromotive force RED unit [V]')
    def _electric_potential_stack(m):
        """
        This function computes the electromotive force of the RED stack based on the average cell pair potential and the number of cell pairs.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        pyomo.Constraint
            The electromotive force of the RED stack constraint
        """
        return m.EMF == m.cell_pairs * ureg.convert(m.Ecp_avg, 'mV', 'V')

    @m.Constraint(doc='RED stack internal resistance [ohm]')
    def _int_resistance_stack(m):
        """
        This function computes the internal resistance of the RED stack based on the average cell pair resistance and the number of cell pairs.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        pyomo.Constraint
            The internal resistance of the RED stack constraint
        """
        return (
            m.Rstack * ureg.convert(m.b * m.L, 'm**2', 'cm**2')
            == m.cell_pairs * m.Rcp_avg
        )

    @m.Constraint(doc='Electric current RED unit [A]')
    def _electric_current_stack(m):
        """
        This function computes the electric current of the RED stack based on the average current density, the area of the ion exchange membrane, and the number of cell pairs.
        Units:
        I [A], Id_avg [mA/cm2], Aiem [m2]

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        pyomo.Constraint
            The electric current of the RED stack constraint
        """
        return m.Istack == ureg.convert(m.Id_avg, 'mA/cm**2', 'A/m**2') * m.Aiem

    @m.Constraint(doc='Gross Power Output RED unit [W]')
    def _gross_power(m):
        """
        This function computes the gross power output of the RED stack.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        pyomo.Constraint
            The gross power output of the RED stack constraint
        """
        return m.GP == m.Istack * (m.EMF - m.Rstack * m.Istack)

    @m.Constraint(doc='Pumping Power Consumption RED unit [W]')
    def _pump_power(m):
        """
        This function computes the pumping power consumption of the RED stack based on the pressure difference at the inlet and outlet of the RED stack, the volumetric flow rate, and the pump efficiency.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        pyomo.Constraint
            The pumping power consumption of the RED stack constraint
        """
        return m.PP * m.pump_eff == sum(
            ureg.convert(
                (m.pressure_x[0, sol] - m.pressure_x[m.length_domain.last(), sol]),
                'mbar',
                'Pa',
            )
            * ureg.convert(m.flow_vol['rm', sol], '1/hour', '1/s')
            for sol in m.SOL
        )

    @m.Constraint(doc='Net Power Output RED unit [W]')
    def _net_power(m):
        """
        This function computes the net power output of the RED stack.

        Parameters
        ----------
        m : pyomo.ConcreteModel
            The RED stack model

        Returns
        -------
        pyomo.Constraint
            The net power output of the RED stack constraint
        """
        return m.NP == m.GP - m.PP

    m.OBJ = pyo.Objective(doc="Maximize Net Power Output", expr=-m.NP * 1e-2)

    #     Create a 'dual' suffix component on the instance
    #     so the solver plugin will know which suffixes to collect
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Solve the model
    pyo.SolverFactory('gams').solve(
        m,
        tee=False,
        # load_solutions=False,
        io_options=dict(
            # solver='baron',
            solver='ipopth',  #'conopt4',#'msnlp',#'baron',
            mtype='nlp',
            # add_options=['option optcr=1e-2','optca = 1.e-6','reslim = 30000']
        ),
    )
    print("RED stack model solved")
    # print(f"Gross Power Output: {m.GP.value:.2f} W")
    return m


if __name__ == "__main__":
    # Create GDP model
    build_REDstack()
