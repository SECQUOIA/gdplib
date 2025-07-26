"""
       Calculation of the theoretical minimum number of trays and initial
                            temperature values.
            (written by E. Soraya Rawlings, esoraya@rwlngs.net)

The separation of four components require a sequence of at least three distillation
columns. Here, we calculate the minimum number of theoretical trays for the three
columns. The sequence is shown in Figure 2.

         COLUMN 1              COLUMN 2              COLUMN 3
             -----                 ----                 -----
            |     |               |    |               |     |
          -----   |   A        -----   |             -----   |
         |     |<---> B --    |     |<----> A --    |     |<---> A
         |     |      C   |   |     |       B   |   |     |
   A     |     |          |   |     |           |   |     |
   B     |     |          |   |     |           |   |     |
   C --->|     |           -->|     |            -->|     |
   D     |     |              |     |               |     |
         |     |              |     |               |     |
         |     |<-            |     |<-             |     |<-
          -----   |            -----   |             -----   |
            |     |              |     |               |     |
             -------> D           -------> C            -------> B
  Figure 2. Sequence of columns for the separation of a quaternary mixture
"""

from __future__ import division

from pyomo.environ import (
    exp,
    log10,
    minimize,
    NonNegativeReals,
    Objective,
    RangeSet,
    SolverFactory,
    value,
    Var,
)

from gdplib.kaibel.kaibel_prop import get_model_with_properties

# from .kaibel_prop import get_model_with_properties


def initialize_kaibel():
    """Initialize the Kaibel optimization model.

    This function initializes the Kaibel optimization model by setting up the operating conditions,
    initial liquid compositions, and creating the necessary variables and constraints.

    Returns
    -------
    None
    """

    ## Get the model with properties from kaibel_prop.py
    m = get_model_with_properties()

    ## Operating conditions
    m.Preb = 1.2  # Reboiler pressure in bar
    m.Pcon = 1.05  # Condenser pressure in bar
    m.Pf = 1.02

    Pnmin = {}  # Pressure in bars
    Pnmin[1] = m.Preb  # Reboiler pressure in bars
    Pnmin[3] = m.Pcon  # Distillate pressure in bars
    Pnmin[2] = m.Pf  # Side feed pressure in bars

    xi_nmin = {}  # Initial liquid composition: first number = column and
    # second number = 1 reboiler, 2 side feed, and
    # 3 for condenser

    ## Column 1
    c_c1 = 4  # Components in Column 1
    lc_c1 = 3  # Light component in Column 1
    hc_c1 = 4  # Heavy component in Column 1
    inter1_c1 = 1  # Intermediate component in Column 1
    inter2_c1 = 2  # Intermediate component in Column 1

    xi_nmin[1, 1, hc_c1] = 0.999
    xi_nmin[1, 1, lc_c1] = (1 - xi_nmin[1, 1, hc_c1]) / (c_c1 - 1)
    xi_nmin[1, 1, inter1_c1] = (1 - xi_nmin[1, 1, hc_c1]) / (c_c1 - 1)
    xi_nmin[1, 1, inter2_c1] = (1 - xi_nmin[1, 1, hc_c1]) / (c_c1 - 1)
    xi_nmin[1, 3, lc_c1] = 0.33
    xi_nmin[1, 3, inter1_c1] = 0.33
    xi_nmin[1, 3, inter2_c1] = 0.33
    xi_nmin[1, 3, hc_c1] = 1 - (
        xi_nmin[1, 3, lc_c1] + xi_nmin[1, 3, inter1_c1] + xi_nmin[1, 3, inter2_c1]
    )
    xi_nmin[1, 2, lc_c1] = 1 / c_c1
    xi_nmin[1, 2, inter1_c1] = 1 / c_c1
    xi_nmin[1, 2, inter2_c1] = 1 / c_c1
    xi_nmin[1, 2, hc_c1] = 1 / c_c1

    ## Column 2
    c_c2 = 3  # Light components in Column 2
    lc_c2 = 2  # Light component in Column 2
    hc_c2 = 3  # Heavy component in Column 2
    inter_c2 = 1  # Intermediate component in Column 2

    xi_nmin[2, 1, hc_c2] = 0.999
    xi_nmin[2, 1, lc_c2] = (1 - xi_nmin[2, 1, hc_c2]) / (c_c2 - 1)
    xi_nmin[2, 1, inter_c2] = (1 - xi_nmin[2, 1, hc_c2]) / (c_c2 - 1)
    xi_nmin[2, 3, lc_c2] = 0.499
    xi_nmin[2, 3, inter_c2] = 0.499
    xi_nmin[2, 3, hc_c2] = 1 - (xi_nmin[2, 3, lc_c2] + xi_nmin[2, 3, inter_c2])
    xi_nmin[2, 2, lc_c2] = 1 / c_c2
    xi_nmin[2, 2, inter_c2] = 1 / c_c2
    xi_nmin[2, 2, hc_c2] = 1 / c_c2

    ## Column 3
    c_c3 = 2  # Components in Column 3
    lc_c3 = 1  # Light component in Column 3
    hc_c3 = 2  # Heavy component in Column 3

    xi_nmin[3, 1, hc_c3] = 0.999
    xi_nmin[3, 1, lc_c3] = 1 - xi_nmin[3, 1, hc_c3]
    xi_nmin[3, 3, lc_c3] = 0.999
    xi_nmin[3, 3, hc_c3] = 1 - xi_nmin[3, 3, lc_c3]
    xi_nmin[3, 2, lc_c3] = 0.50
    xi_nmin[3, 2, hc_c3] = 0.50

    ####

    mn = m.clone()  # Clone the model to add the initialization code

    mn.name = "Initialization Code"

    mn.cols = RangeSet(3, doc="Number of columns ")
    mn.sec = RangeSet(3, doc="Sections in column: 1 reb, 2 side feed, 3 cond")
    mn.nc1 = RangeSet(c_c1, doc="Number of components in Column 1")
    mn.nc2 = RangeSet(c_c2, doc="Number of components in Column 2")
    mn.nc3 = RangeSet(c_c3, doc="Number of components in Column 3")

    mn.Tnmin = Var(
        mn.cols,
        mn.sec,
        doc="Temperature in K",
        bounds=(0, 500),
        domain=NonNegativeReals,
    )
    mn.Tr1nmin = Var(
        mn.cols,
        mn.sec,
        mn.nc1,
        doc="Temperature term for vapor pressure",
        domain=NonNegativeReals,
        bounds=(0, None),
    )
    mn.Tr2nmin = Var(
        mn.cols,
        mn.sec,
        mn.nc2,
        doc="Temperature term for vapor pressure",
        domain=NonNegativeReals,
        bounds=(0, None),
    )
    mn.Tr3nmin = Var(
        mn.cols,
        mn.sec,
        mn.nc3,
        doc="Temperature term for vapor pressure",
        domain=NonNegativeReals,
        bounds=(0, None),
    )

    @mn.Constraint(mn.cols, mn.sec, mn.nc1, doc="Temperature term for vapor pressure")
    def _column1_reduced_temperature(mn, col, sec, nc):
        """Calculate the reduced temperature for column 1.

        This function calculates the reduced temperature for column 1 based on the given parameters using the Peng-Robinson equation of state.

        Parameters
        ----------
        mn : Pyomo ConcreteModel
            The optimization model
        col : int
            The column index
        sec : int
            The section index
        nc : int
            The component index in column 1

        Returns
        -------
        Constraint
            The constraint statement to calculate the reduced temperature.
        """
        return mn.Tr1nmin[col, sec, nc] * mn.Tnmin[col, sec] == mn.prop[nc, "TC"]

    @mn.Constraint(mn.cols, mn.sec, mn.nc2, doc="Temperature term for vapor pressure")
    def _column2_reduced_temperature(mn, col, sec, nc):
        """Calculate the reduced temperature for column 2.

        This function calculates the reduced temperature for column 2 based on the given parameters using the Peng-Robinson equation of state.

        Parameters
        ----------
        mn : Pyomo ConcreteModel
            The optimization model
        col : int
            The column index
        sec : int
            The section index
        nc : int
            The component index in column 2

        Returns
        -------
        Constraint
            The constraint equation to calculate the reduced temperature
        """
        return mn.Tr2nmin[col, sec, nc] * mn.Tnmin[col, sec] == mn.prop[nc, "TC"]

    @mn.Constraint(mn.cols, mn.sec, mn.nc3, doc="Temperature term for vapor pressure")
    def _column3_reduced_temperature(mn, col, sec, nc):
        """Calculate the reduced temperature for column 3.

        This function calculates the reduced temperature for column 3 based on the given parameters.

        Parameters
        ----------
        mn : Pyomo ConcreteModel
            The optimization model
        col : int
            The column index
        sec : int
            The section index
        nc : int
            The component index in column 3

        Returns
        -------
        Constraint
            The constraint equation to calculate the reduced temperature in column 3
        """
        return mn.Tr3nmin[col, sec, nc] * mn.Tnmin[col, sec] == mn.prop[nc, "TC"]

    @mn.Constraint(mn.cols, mn.sec, doc="Boiling point temperature")
    def _equilibrium_equation(mn, col, sec):
        """Equilibrium equations for a given column and section.

        Parameters
        ----------
        mn : Pyomo ConcreteModel
            The optimization model object with properties
        col : int
            The column index
        sec : int
            The section index

        Returns
        -------
        Constraint
            The constraint equation to calculate the boiling point temperature using the Peng-Robinson equation of state
        """
        if col == 1:
            a = mn.Tr1nmin
            b = mn.nc1
        elif col == 2:
            a = mn.Tr2nmin
            b = mn.nc2
        elif col == 3:
            a = mn.Tr3nmin
            b = mn.nc3
        return (
            sum(
                xi_nmin[col, sec, nc]
                * mn.prop[nc, "PC"]
                * exp(
                    a[col, sec, nc]
                    * (
                        mn.prop[nc, "vpA"]
                        * (1 - mn.Tnmin[col, sec] / mn.prop[nc, "TC"])
                        + mn.prop[nc, "vpB"]
                        * (1 - mn.Tnmin[col, sec] / mn.prop[nc, "TC"]) ** 1.5
                        + mn.prop[nc, "vpC"]
                        * (1 - mn.Tnmin[col, sec] / mn.prop[nc, "TC"]) ** 3
                        + mn.prop[nc, "vpD"]
                        * (1 - mn.Tnmin[col, sec] / mn.prop[nc, "TC"]) ** 6
                    )
                )
                / Pnmin[sec]
                for nc in b
            )
            == 1
        )

    mn.OBJ = Objective(expr=1, sense=minimize)

    ####

    SolverFactory("ipopt").solve(mn)

    yc = {}  # Vapor composition
    kl = {}  # Light key component
    kh = {}  # Heavy key component
    alpha = {}  # Relative volatility of kl
    ter = {}  # Term to calculate the minimum number of trays
    Nmin = {}  # Minimum number of stages
    Nminopt = {}  # Total optimal minimum number of trays
    Nfeed = {}  # Side feed optimal location using Kirkbride's method:
    # 1 = number of trays in rectifying section and
    # 2 = number of trays in stripping section
    side_feed = {}  # Side feed location
    av_alpha = {}  # Average relative volatilities
    xi_lhc = {}  # Liquid composition in columns
    rel = mn.Bdes / mn.Ddes  # Ratio between products flowrates
    ln = {}  # Light component for the different columns
    hn = {}  # Heavy component for the different columns
    ln[1] = lc_c1
    ln[2] = lc_c2
    ln[3] = lc_c3
    hn[1] = hc_c1
    hn[2] = hc_c2
    hn[3] = hc_c3

    for col in mn.cols:
        if col == 1:
            b = mn.nc1
        elif col == 2:
            b = mn.nc2
        else:
            b = mn.nc3
        # For each component in the column and section calculate the vapor composition with the Peng-Robinson equation of state
        for sec in mn.sec:
            for nc in b:
                yc[col, sec, nc] = (
                    xi_nmin[col, sec, nc]
                    * mn.prop[nc, "PC"]
                    * exp(
                        mn.prop[nc, "TC"]
                        / value(mn.Tnmin[col, sec])
                        * (
                            mn.prop[nc, "vpA"]
                            * (1 - value(mn.Tnmin[col, sec]) / mn.prop[nc, "TC"])
                            + mn.prop[nc, "vpB"]
                            * (1 - value(mn.Tnmin[col, sec]) / mn.prop[nc, "TC"]) ** 1.5
                            + mn.prop[nc, "vpC"]
                            * (1 - value(mn.Tnmin[col, sec]) / mn.prop[nc, "TC"]) ** 3
                            + mn.prop[nc, "vpD"]
                            * (1 - value(mn.Tnmin[col, sec]) / mn.prop[nc, "TC"]) ** 6
                        )
                    )
                ) / Pnmin[
                    sec
                ]  # Vapor composition in the different sections for the different components in the columns

    for col in mn.cols:
        # Calculate the relative volatility of the light and heavy components in the different sections for the different columns
        xi_lhc[col, 4] = (
            xi_nmin[col, 1, ln[col]] / xi_nmin[col, 3, hn[col]]
        )  # Liquid composition in the different sections with the initial liquid composition of the components in the different sections and columns and ln and hn which are the light and heavy components in the different columns
        for sec in mn.sec:
            kl[col, sec] = (
                yc[col, sec, ln[col]] / xi_nmin[col, sec, ln[col]]
            )  # Light component in the different sections
            kh[col, sec] = (
                yc[col, sec, hn[col]] / xi_nmin[col, sec, hn[col]]
            )  # Heavy component in the different sections
            xi_lhc[col, sec] = (
                xi_nmin[col, sec, hn[col]] / xi_nmin[col, sec, ln[col]]
            )  # Liquid composition in the different sections
            alpha[col, sec] = (
                kl[col, sec] / kh[col, sec]
            )  # Relative volatility in the different sections

    for col in mn.cols:
        # Calculate the average relative volatilities and the minimum number of trays with Fenske's and Kirkbride's method
        av_alpha[col] = (alpha[col, 1] * alpha[col, 2] * alpha[col, 3]) ** (
            1 / 3
        )  # Average relative volatilities calculated with the relative volatilities of the components in the three sections
        Nmin[col] = log10((1 / xi_lhc[col, 3]) * xi_lhc[col, 1]) / log10(
            av_alpha[col]
        )  # Minimum number of trays calculated with Fenske's method
        ter[col] = (
            rel * xi_lhc[col, 2] * (xi_lhc[col, 4] ** 2)
        ) ** 0.206  # Term to calculate the minimum number of trays with Kirkbride's method
        # Side feed optimal location using Kirkbride's method
        Nfeed[1, col] = (
            ter[col] * Nmin[col] / (1 + ter[col])
        )  # Number of trays in rectifying section
        Nfeed[2, col] = Nfeed[1, col] / ter[col]  # Number of trays in stripping section
        side_feed[col] = Nfeed[2, col]  # Side feed location

    m.Nmintot = sum(Nmin[col] for col in mn.cols)  # Total minimum number of trays
    m.Knmin = int(m.Nmintot) + 1  # Total optimal minimum number of trays

    m.TB0 = value(mn.Tnmin[1, 1])  # Reboiler temperature in K in column 1
    m.Tf0 = value(mn.Tnmin[1, 2])  # Side feed temperature in K in column 1
    m.TD0 = value(mn.Tnmin[2, 3])  # Distillate temperature in K in column 2

    return m


if __name__ == "__main__":
    initialize_kaibel()
