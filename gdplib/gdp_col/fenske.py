from __future__ import division

from pyomo.environ import (
    ConcreteModel,
    NonNegativeReals,
    Set,
    SolverFactory,
    Var,
    log,
    sqrt,
)


def calculate_Fenske(xD, xB):
    """
    Calculate the minimum number of plates required for a given separation using the Fenske equation.

    Parameters
    ----------
    xD : float
        Distillate(benzene) purity
    xB : float
        Bottoms(toluene) purity

    Returns
    -------
    None
        None, but prints the Fenske equation calculating the minimum number of plates required for a given separation.
    """
    m = ConcreteModel()
    min_T, max_T = 300, 400
    m.comps = Set(initialize=["benzene", "toluene"])
    m.trays = Set(initialize=["condenser", "reboiler"])
    m.Kc = Var(
        m.comps,
        m.trays,
        doc="Phase equilibrium constant",
        domain=NonNegativeReals,
        initialize=1,
        bounds=(0, 1000),
    )
    m.T = Var(
        m.trays, doc="Temperature [K]", domain=NonNegativeReals, bounds=(min_T, max_T)
    )
    m.T["condenser"].fix(82 + 273.15)
    m.T["reboiler"].fix(108 + 273.15)
    m.P = Var(doc="Pressure [bar]", bounds=(0, 5))
    m.P.fix(1.01)
    m.T_ref = 298.15
    m.gamma = Var(
        m.comps,
        m.trays,
        doc="liquid activity coefficient of component on tray",
        domain=NonNegativeReals,
        bounds=(0, 10),
        initialize=1,
    )
    m.Pvap = Var(
        m.comps,
        m.trays,
        doc="pure component vapor pressure of component on tray in bar",
        domain=NonNegativeReals,
        bounds=(1e-3, 5),
        initialize=0.4,
    )
    m.Pvap_X = Var(
        m.comps,
        m.trays,
        doc="Related to fraction of critical temperature (1 - T/Tc)",
        bounds=(0.25, 0.5),
        initialize=0.4,
    )

    m.pvap_const = {
        "benzene": {
            "A": -6.98273,
            "B": 1.33213,
            "C": -2.62863,
            "D": -3.33399,
            "Tc": 562.2,
            "Pc": 48.9,
        },
        "toluene": {
            "A": -7.28607,
            "B": 1.38091,
            "C": -2.83433,
            "D": -2.79168,
            "Tc": 591.8,
            "Pc": 41.0,
        },
    }

    @m.Constraint(m.comps, m.trays)
    def phase_equil_const(_, c, t):
        """
        Phase equilibrium constraint for each component in a tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the phase equilibrium constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.
        t : int
            Index of tray in the distillation column model.

        Returns
        -------
        Pyomo.Constraint
            The phase equilibrium constant for each component in a tray multiplied with the pressure is equal to the product of the activity coefficient and the pure component vapor pressure.
        """
        return m.Kc[c, t] * m.P == (m.gamma[c, t] * m.Pvap[c, t])

    @m.Constraint(m.comps, m.trays)
    def Pvap_relation(_, c, t):
        """
        Antoine's equation for the vapor pressure of each component in a tray.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the phase equilibrium constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.
        t : int
            Index of tray in the distillation column model.

        Returns
        -------
        Pyomo.Constraint
            Antoine's equation for the vapor pressure of each component in a tray is calculated as the logarithm of the vapor pressure minus the logarithm of the critical pressure times one minus the fraction of critical temperature. The equation is equal to the sum of the Antoine coefficients times the fraction of critical temperature raised to different powers.
        """
        k = m.pvap_const[c]
        x = m.Pvap_X[c, t]
        return (log(m.Pvap[c, t]) - log(k["Pc"])) * (1 - x) == (
            k["A"] * x + k["B"] * x**1.5 + k["C"] * x**3 + k["D"] * x**6
        )

    @m.Constraint(m.comps, m.trays)
    def Pvap_X_defn(_, c, t):
        """
        Defines the relationship between the one minus the reduced temperature variable (Pvap_X) for each component in a tray, and the actual temperature of the tray, normalized by the critical temperature of the component (Tc).

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the phase equilibrium constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.
        t : int
            Index of tray in the distillation column model.

        Returns
        -------
        Pyomo.Constraint
            The relationship between the one minus the reduced temperature variable (Pvap_X) for each component in a tray, and the actual temperature of the tray, normalized by the critical temperature of the component (Tc).
        """
        k = m.pvap_const[c]
        return m.Pvap_X[c, t] == 1 - m.T[t] / k["Tc"]

    @m.Constraint(m.comps, m.trays)
    def gamma_calc(_, c, t):
        """
        Activity coefficient calculation.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the phase equilibrium constraints to the trays in the distillation column.
        c : str
            Index of component in the distillation column model. 'benzene' or 'toluene'.
        t : int
            Index of tray in the distillation column model.

        Returns
        -------
        Pyomo.Constraint
            Set the activity coefficient of the component on the tray as 1.
        """
        return m.gamma[c, t] == 1

    m.relative_volatility = Var(m.trays, domain=NonNegativeReals)

    @m.Constraint(m.trays)
    def relative_volatility_calc(_, t):
        """
        Relative volatility calculation.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the phase equilibrium constraints to the trays in the distillation column.
        t : int
            Index of tray in the distillation column model.

        Returns
        -------
        Pyomo.Constraint
            The relative volatility of benzene to toluene is the ratio of the phase equilibrium constants of benzene to toluene on the tray.
        """
        return m.Kc["benzene", t] == (m.Kc["toluene", t] * m.relative_volatility[t])

    @m.Expression()
    def fenske(_):
        """
        Fenske equation for minimum number of plates.

        Parameters
        ----------
        _ : Pyomo.ConcreteModel
            A placeholder representing the Pyomo model instance. It specifies the context for applying the phase equilibrium constraints to the trays in the distillation column.

        Returns
        -------
        Pyomo.Expression
            The Fenske equation calculating the minimum number of plates required for a given separation.
        """
        return log((xD / (1 - xD)) * (xB / (1 - xB))) / (
            log(
                sqrt(
                    m.relative_volatility["condenser"]
                    * m.relative_volatility["reboiler"]
                )
            )
        )

    SolverFactory("ipopt").solve(m, tee=True)
    from pyomo.util.infeasible import log_infeasible_constraints

    log_infeasible_constraints(m, tol=1e-3)
    m.fenske.display()


if __name__ == "__main__":
    m = calculate_Fenske(0.95, 0.95)
