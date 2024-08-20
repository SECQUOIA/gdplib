""" Side feed flash """

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    exp,
    minimize,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    SolverFactory,
    value,
    Var,
)


def calc_side_feed_flash(m):
    """Calculate the side feed flash.

    This function solves a flash calculation for a side feed in a distillation process.
    It calculates the vapor-liquid equilibrium, vapor pressure, and liquid and vapor compositions
    for the given side feed.

    Parameters
    ----------
    m : Pyomo ConcreteModel
        The Pyomo model object containing the necessary parameters and variables.

    Returns
    -------
    Pyomo ConcreteModel
        The updated Pyomo model with the calculated values, which include the vapor-liquid equilibrium, vapor pressure, and liquid and vapor compositions for the side feed.

    """
    msf = ConcreteModel('SIDE FEED FLASH')  # Main side feed flash model

    msf.nc = RangeSet(1, m.c, doc='Number of components')

    m.xfi = (
        {}
    )  # Liquid composition in the side feed of the main model for each component.
    for nc in msf.nc:
        m.xfi[nc] = 1 / m.c

    msf.Tf = Param(doc='Side feed temperature in K', initialize=m.Tf0)
    msf.xf = Var(
        msf.nc,
        doc='Side feed liquid composition',
        domain=NonNegativeReals,
        bounds=(0, 1),
        initialize=m.xfi,
    )
    msf.yf = Var(
        msf.nc,
        doc='Side feed vapor composition',
        domain=NonNegativeReals,
        bounds=(0, 1),
        initialize=m.xfi,
    )
    msf.Keqf = Var(
        msf.nc,
        doc='Vapor-liquid equilibrium constant',
        domain=NonNegativeReals,
        bounds=(0, 10),
        initialize=0,
    )
    msf.Pvf = Var(
        msf.nc,
        doc='Side feed vapor pressure in bar',
        domain=NonNegativeReals,
        bounds=(0, 10),
        initialize=0,
    )
    msf.q = Var(
        doc='Vapor fraction', bounds=(0, 1), domain=NonNegativeReals, initialize=0
    )

    @msf.Constraint(doc="Vapor fraction")
    def _algq(msf):
        """This function calculates the vapor fraction (q) in the side feed using the Peng-Robinson equation of state.

        Parameters
        ----------
        msf : Pyomo ConcreteModel
            The main side feed flash model.

        Returns
        -------
        q : float
            The vapor fraction in the side feed.
        """
        return (
            sum(
                m.xfi[nc] * (1 - msf.Keqf[nc]) / (1 + msf.q * (msf.Keqf[nc] - 1))
                for nc in msf.nc
            )
            == 0
        )

    @msf.Constraint(msf.nc, doc="Side feed liquid composition")
    def _algx(msf, nc):
        """Side feed liquid composition

        This function calculates the liquid composition (xf) for a given component (nc) in the side feed.

        Parameters
        ----------
        msf : Pyomo ConcreteModel
            The main side feed flash model.
        nc : int
            The component index.

        Returns
        -------
        xf : float
            The liquid composition for the given component with Keqf, q, and xfi, which are the equilibrium constant, vapor fraction, and liquid composition in the side feed, respectively.
        """
        return msf.xf[nc] * (1 + msf.q * (msf.Keqf[nc] - 1)) == m.xfi[nc]

    @msf.Constraint(msf.nc, doc="Side feed vapor composition")
    def _algy(msf, nc):
        """Side feed vapor composition

        This function calculates the vapor composition (ysf) for a given component (nc) in the side.

        Parameters
        ----------
        msf : Pyomo ConcreteModel
            The main side feed flash model.
        nc : int
            The component index.

        Returns
        -------
        yf : float
            The vapor composition for the given component given the liquid composition (xf) and the equilibrium constant (Keqf).
        """
        return msf.yf[nc] == msf.xf[nc] * msf.Keqf[nc]

    # TODO: Is it computed using the Peng-Robinson equation of state?

    @msf.Constraint(msf.nc, doc="Vapor-liquid equilibrium constant")
    def _algKeq(msf, nc):
        """Calculate the vapor-liquid equilibrium constant for a given component using the Peng-Robinson equation of state.

        Parameters
        ----------
        msf : Pyomo ConcreteModel
            The MultiStageFlash model.
        nc : int
            The component index.

        Returns
        -------
        Keqf : float
            The equilibrium constant for the component taking into account the vapor pressure (Pvf) and the liquid pressure (Pf).
        """
        return msf.Keqf[nc] * m.Pf == msf.Pvf[nc]

    @msf.Constraint(msf.nc, doc="Side feed vapor pressure")
    def _algPvf(msf, nc):
        """Calculate the vapor fraction for a given component.

        This function calculates the vapor fraction (Pvf) for a given component (nc) using the Peng-Robinson equation of state.

        Parameters
        ----------
        msf : Pyomo ConcreteModel
            The main side flash object.
        nc : int
            The component index.

        Returns
        -------
        Pvf : float
            The vapor fraction for the given component considering the temperature (Tf) and the properties of the component set in the main model.
        """
        return msf.Pvf[nc] == m.prop[nc, 'PC'] * exp(
            m.prop[nc, 'TC']
            / msf.Tf
            * (
                m.prop[nc, 'vpA'] * (1 - msf.Tf / m.prop[nc, 'TC'])
                + m.prop[nc, 'vpB'] * (1 - msf.Tf / m.prop[nc, 'TC']) ** 1.5
                + m.prop[nc, 'vpC'] * (1 - msf.Tf / m.prop[nc, 'TC']) ** 3
                + m.prop[nc, 'vpD'] * (1 - msf.Tf / m.prop[nc, 'TC']) ** 6
            )
        )

    # TODO: Is it computed using the Peng-Robinson equation of state?

    msf.OBJ = Objective(expr=1, sense=minimize)

    ####
    SolverFactory('ipopt').solve(msf, tee=False)

    # Update the main model with the calculated values
    m.yfi = {}  # Vapor composition
    for nc in msf.nc:
        m.yfi[nc] = value(msf.yf[nc])

    m.q_init = value(msf.q)  # Vapor fraction

    return m
