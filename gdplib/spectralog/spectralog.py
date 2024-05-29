"""
spectrolog.py
IR Spectroscopy Parameter Estimation

This is a reproduction of the IR spectroscopy parameter estimation problem found in:

[1] Vecchietti, A., & Grossmann, I. E. (1997). LOGMIP: a disjunctive 0-1 nonlinear optimizer for process systems models. Computers & chemical engineering, 21, S427-S432. https://doi.org/10.1016/S0098-1354(97)87539-4
[2] Brink, A., & Westerlund, T. (1995). The joint problem of model structure determination and parameter estimation in quantitative IR spectroscopy. Chemometrics and intelligent laboratory systems, 29(1), 29-36. https://doi.org/10.1016/0169-7439(95)00033-3
  
Optimal value: 12.0893
"""

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.expr.logical_expr import *
from pyomo.core.plugins.transform.logical_to_linear import (
    update_boolean_vars_from_binary,
)
from six import StringIO
import pandas as pd


def build_model():
    """
    Constructs and returns a Pyomo Concrete Model for IR spectroscopy parameter estimation.

    Returns
    -------
    Pyomo.ConcreteModel
        A Pyomo model representing the IR spectroscopy parameter estimation problem.

    Notes
    -----
    - The model uses a disjunctive programming approach where decision variables can trigger different sets of constraints,
    representing different physcochemical conditions.

    References
    ----------
    [1] Vecchietti, A., & Grossmann, I. E. (1997). LOGMIP: a disjunctive 0-1 nonlinear optimizer for process systems models. Computers & chemical engineering, 21, S427-S432. https://doi.org/10.1016/S0098-1354(97)87539-4
    [2] Brink, A., & Westerlund, T. (1995). The joint problem of model structure determination and parameter estimation in quantitative IR spectroscopy. Chemometrics and intelligent laboratory systems, 29(1), 29-36. https://doi.org/10.1016/0169-7439(95)00033-3
    """
    # Matrix of absorbance values across different wave numbers (rows) and spectra numbers (columns)
    spectroscopic_data = StringIO(
        """
             1      2       3       4       5       6       7       8
    1     0.0003  0.0764  0.0318  0.0007  0.0534  0.0773  0.0536  0.0320
    2     0.0007  0.0003  0.0004  0.0009  0.0005  0.0009  0.0005  0.0003
    3     0.0066  0.0789  0.0275  0.0043  0.0704  0.0683  0.0842  0.0309
    4     0.0044  0.0186  0.0180  0.0179  0.0351  0.0024  0.0108  0.0052
    5     0.0208  0.0605  0.0601  0.0604  0.0981  0.0025  0.0394  0.0221
    6     0.0518  0.1656  0.1491  0.1385  0.2389  0.0248  0.1122  0.0633
    7     0.0036  0.0035  0.0032  0.0051  0.0015  0.0094  0.0015  0.0024
    8     0.0507  0.0361  0.0433  0.0635  0.0048  0.0891  0.0213  0.0310
    9     0.0905  0.0600  0.0754  0.1098  0.0038  0.1443  0.0420  0.0574
    10    0.0016  0.0209  0.0063  0.0010  0.0132  0.0203  0.0139  0.0057
    """
    )
    # Note: this could come from an external data file
    spectroscopic_data_table = pd.read_csv(spectroscopic_data, delimiter=r'\s+')
    flat_spectro_data = spectroscopic_data_table.stack()
    spectro_data_dict = {
        (k[0], int(k[1])): v for k, v in flat_spectro_data.to_dict().items()
    }  # column labels to integer

    # Measured concentration data for each compound(row) and spectra number(column)
    # Units for concentration for each component 1, 2, and 3 are ppm, ppm, and % for CO, NO, and CO2, respectively.
    c_data = StringIO(
        """
            1       2       3       4       5       6       7       8
    1       502     204     353     702     0       1016    104     204
    2        97     351     351     351     700     0       201      97
    3        0      22      8       0       14      22      14       8 
    """
    )
    c_data_table = pd.read_csv(c_data, delimiter=r'\s+')
    c_data_dict = {
        (k[0], int(k[1])): v for k, v in c_data_table.stack().to_dict().items()
    }

    # Covariance matrix; It is assumed to be known that it is equal to the identity matrix at first problem iteration
    r_data = StringIO(
        """
            1       2       3
    1       1       0       0
    2       0       1       0
    3       0       0       1
    """
    )
    r_data_table = pd.read_csv(r_data, delimiter=r'\s+')
    r_data_dict = {
        (k[0], int(k[1])): v for k, v in r_data_table.stack().to_dict().items()
    }

    m = ConcreteModel(name="IR spectroscopy parameter estimation")
    m.wave_number = RangeSet(10)  # 10 wave numbers
    m.spectra_data = RangeSet(8)  # 8 spectra data points
    m.compounds = RangeSet(
        3
    )  # 3 compounds; 1, 2, 3 refer to CO, NO, and CO2, respectively

    m.A = Param(
        m.wave_number,
        m.spectra_data,
        initialize=spectro_data_dict,
        doc='Absorbance data',
    )
    m.C = Param(
        m.compounds, m.spectra_data, initialize=c_data_dict, doc='Concentration data'
    )
    m.R = Param(
        m.compounds, m.compounds, initialize=r_data_dict, doc='Covariance matrix'
    )

    m.val = Var(
        m.spectra_data, doc='Calculated objective values for each spectra data point'
    )
    m.ent = Var(
        m.compounds,
        m.wave_number,
        bounds=(0, 1),
        doc='Binary variables affecting the objective function and constraints.',
    )
    m.Y = BooleanVar(
        m.compounds,
        m.wave_number,
        doc='Boolean decisions for compound presence at each wave number.',
    )
    m.P = Var(
        m.compounds,
        m.wave_number,
        bounds=(0, 1000),
        doc='Continuous variables estimating the concentration level of each compound at each wave number.',
    )

    @m.Disjunction(m.compounds, m.wave_number)
    def d(m, k, i):
        """_summary_

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo model representing the IR spectroscopy parameter estimation problem.
        k : int
            Index of compounds.
        i : int
            Index of wave numbers.

        Returns
        -------
        Pyomo.Disjunction
            A disjunctive constraint that specifies the operational conditions for each compound-wave number pair based on the model's parameters.
        """
        return [
            [
                m.P[k, i] <= 1000,
                m.P[k, i] >= 0,
                m.ent[k, i] == 1,
            ],  # Conditions for the compound being active
            [
                m.P[k, i] == 0,
                m.ent[k, i] == 0,
            ],  # Conditions for the compound being inactive
        ]

    # Associate each Boolean variable with a corresponding binary variable to handle logical conditions.
    for k, i in m.compounds * m.wave_number:
        m.Y[k, i].associate_binary_var(m.d[k, i].disjuncts[0].binary_indicator_var)

    @m.Constraint(m.spectra_data)
    def eq1(m, j):
        """
        Defines a disjunction for each compound and wave number that determines whether a compound is active at a particular wave number based on the parameter estimates.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo model representing the IR spectroscopy parameter estimation problem.
        j : int
            Index for the spectra data points, representing different experimental conditions.

        Returns
        -------
        Pyomo.Constraint
            An expression that equates the calculated value for each spectra data point with a mathematically derived expression from the model.
        """
        return m.val[j] == sum(
            sum(
                (m.C[kk, j] / 100 - sum(m.P[kk, i] * m.A[i, j] for i in m.wave_number))
                * m.R[kk, k]
                for kk in m.compounds
            )
            * (m.C[k, j] / 100 - sum(m.P[k, i] * m.A[i, j] for i in m.wave_number))
            for k in m.compounds
        )

    m.profit = Objective(
        expr=sum(m.val[j] for j in m.spectra_data)
        + 2 * sum(m.ent[k, i] for k in m.compounds for i in m.wave_number),
        doc='Maximizes the total spectroscopic agreement across data points and promotes the activation of compound-wave number pairs.',
    )
    #  The first sum represents total spectroscopic value across data points, and the second weighted sum promotes activation of compound-wave number pairs.

    return m


if __name__ == "__main__":
    m = build_model()
    TransformationFactory('core.logical_to_linear').apply_to(m)
    # res = SolverFactory('gdpopt').solve(m, tee=False, nlp_solver='gams')
    TransformationFactory('gdp.bigm').apply_to(m)
    SolverFactory('gams').solve(m, tee=True, solver='baron')
    update_boolean_vars_from_binary(m)
    m.profit.display()
    m.Y.display()
    m.P.display()
