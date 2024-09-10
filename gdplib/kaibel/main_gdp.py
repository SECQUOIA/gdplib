"""
                     KAIBEL COLUMN: Modeling, Optimization, and
               Conceptual Design of Multi-product Dividing Wall Columns
                 (written by E. Soraya Rawlings, esoraya@rwlngs.net,
                          in collaboration with Qi Chen)

This is a dividing wall distillation column design problem to determine the optimal minimum number of trays and the optimal location of side streams for the separation of a quaternary mixture:
 1 = methanol
 2 = ethanol
 3 = propanol
 4 = butanol
while minimizing its capital and operating costs.

 The scheme of the Kaibel Column is shown in Figure 1:
                              ____
                           --|Cond|---
                          |   ----    |
                   --------------     |
                  |    sect 4    |<------> D        mostly 1
                  |  ----------  |
                  |              |
                  | ----- |----- |
                  |       |      |-------> S2
        Fj ------>| sect  | sect |
                  |   2   |  3   |
                  |       |      |-------> S1
                  | ----- |----- |
                  |              |
                  |  ----------  |
                  |     sect 1   |<------> B        mostly 4
                   --------------      |
                          |    ____    |
                           ---|Reb |---
                               ----
               Figure 1. Kaibel Column scheme

Permanent trays:
- Reboiler and vapor distributor in the bottom section (sect 1)
- Liquid distributor and condenser in the top section (sect 4)
- Side feed tray for the feed side and dividing wall starting and and ening tray in the feed section (sect 2).
- Side product trays and dividing wall starting and ending tray in the product section (sect 3).

The trays in each section are counted from bottom to top, being tray 1 the bottom tray in each section and tray np the top tray in each section, where np is a specified upper bound for the number of possible trays for each section.
Each section has the same number of possible trays.

Six degrees of freedom: the reflux ratio, the product outlets (bottom, intermediate, and distillate product flowrates), and the liquid and vapor flowrates between the two sections of the dividing wall, controlled by a liquid and vapor distributor on the top and bottom section of the column, respectively.
including also the vapor and liquid flowrate and the energy consumption in the reboiler and condenser.
The vapor distributor is fixed and remain constant during the column operation.

Source paper:
Rawlings, E. S., Chen, Q., Grossmann, I. E., & Caballero, J. A. (2019). Kaibel Column: Modeling, optimization, and conceptual design of multi-product dividing wall columns. *Computers and Chemical Engineering*, 125, 31–39. https://doi.org/10.1016/j.compchemeng.2019.03.006

"""

from math import fabs

import matplotlib.pyplot as plt
from pyomo.environ import *

# from kaibel_solve_gdp import build_model
from gdplib.kaibel.kaibel_solve_gdp import build_model


def main():
    """
    This is the main function that executes the optimization process.

    It builds the model, fixes certain variables, sets initial values for tray existence or absence,
    solves the model using the 'gdpopt' solver, and displays the results.

    Returns:
        None
    """
    m = build_model()

    # Fixing variables
    m.F[1].fix(50)  # feed flowrate in mol/s
    m.F[2].fix(50)
    m.F[3].fix(50)
    m.F[4].fix(50)
    m.q.fix(
        m.q_init
    )  # vapor fraction q_init from the feed set in the build_model function
    m.dv[2].fix(0.394299)  # vapor distributor in the feed section

    for sec in m.section:
        for n_tray in m.tray:
            m.P[sec, n_tray].fix(m.Preb)

    ## Initial values for the tray existence or absence
    for n_tray in m.candidate_trays_main:
        for sec in m.section_main:
            m.tray_exists[sec, n_tray].indicator_var.set_value(1)
            m.tray_absent[sec, n_tray].indicator_var.set_value(0)
    for n_tray in m.candidate_trays_feed:
        m.tray_exists[2, n_tray].indicator_var.set_value(1)
        m.tray_absent[2, n_tray].indicator_var.set_value(0)
    for n_tray in m.candidate_trays_product:
        m.tray_exists[3, n_tray].indicator_var.set_value(1)
        m.tray_absent[3, n_tray].indicator_var.set_value(0)

    intro_message(m)

    results = SolverFactory('gdpopt').solve(
        m,
        strategy='LOA',
        tee=True,
        time_limit=3600,
        mip_solver='gams',
        mip_solver_args=dict(solver='cplex'),
    )

    m.calc_nt = sum(
        sum(m.tray_exists[sec, n_tray].indicator_var.value for n_tray in m.tray)
        for sec in m.section
    ) - sum(m.tray_exists[3, n_tray].indicator_var.value for n_tray in m.tray)
    m.dw_start = (
        sum(m.tray_exists[1, n_tray].indicator_var.value for n_tray in m.tray) + 1
    )
    m.dw_end = sum(
        m.tray_exists[1, n_tray].indicator_var.value for n_tray in m.tray
    ) + sum(m.tray_exists[2, n_tray].indicator_var.value for n_tray in m.tray)

    display_results(m)

    print('  ', results)
    print('  Solver Status: ', results.solver.status)
    print('  Termination condition: ', results.solver.termination_condition)


def intro_message(m):
    """
    Display the introduction message.



    """
    print(
        """

If you use this model and/or initialization strategy, you may cite the following:
Rawlings, ES; Chen, Q; Grossmann, IE; Caballero, JA. Kaibel Column: Modeling, 
optimization, and conceptual design of multi-product dividing wall columns. 
Comp. and Chem. Eng., 2019, 125, 31-39. 
DOI: https://doi.org/10.1016/j.compchemeng.2019.03.006


    """
    )


def display_results(m):
    """
    Display the results of the optimization process.

    Parameters
    ----------
    m : Pyomo ConcreteModel
        The Pyomo model object containing the results of the optimization process.
    """
    print('')
    print('Components:')
    print('1 methanol')
    print('2 ethanol')
    print('3 butanol')
    print('4 propanol')
    print(' ')
    print('Objective: %s' % value(m.obj))
    print('Trays: %s' % value(m.calc_nt))
    print('Dividing_wall_start: %s' % value(m.dw_start))
    print('Dividing_wall_end: %s' % value(m.dw_end))
    print(' ')
    print(
        'Qreb: {: >3.0f}kW  B_1: {: > 2.0f}  B_2: {: >2.0f}  B_3: {: >2.0f}  B_4: {: >2.0f}  Btotal: {: >2.0f}'.format(
            value(m.Qreb / m.Qscale),
            value(m.B[1]),
            value(m.B[2]),
            value(m.B[3]),
            value(m.B[4]),
            value(m.Btotal),
        )
    )
    print(
        'Qcon: {: >2.0f}kW  D_1: {: >2.0f}  D_2: {: >2.0f}  D_3: {: >2.0f}  D_4: {: >2.0f}  Dtotal: {: >2.0f}'.format(
            value(m.Qcon / m.Qscale),
            value(m.D[1]),
            value(m.D[2]),
            value(m.D[3]),
            value(m.D[4]),
            value(m.Dtotal),
        )
    )
    print(' ')
    print('Reflux: {: >3.4f}'.format(value(m.rr)))
    print('Reboil: {: >3.4f} '.format(value(m.bu)))
    print(' ')
    print('Flowrates[mol/s]')
    print(
        'F_1: {: > 3.0f}  F_2: {: >2.0f}  F_3: {: >2.0f}  F_4: {: >2.0f}  Ftotal: {: >2.0f}'.format(
            value(m.F[1]),
            value(m.F[2]),
            value(m.F[3]),
            value(m.F[4]),
            sum(value(m.F[comp]) for comp in m.comp),
        )
    )
    print(
        'S1_1:  {: > 1.0f}  S1_2: {: >2.0f}  S1_3: {: >2.0f}  S1_4: {: >2.0f}  S1total: {: >2.0f}'.format(
            value(m.S[1, 1]),
            value(m.S[1, 2]),
            value(m.S[1, 3]),
            value(m.S[1, 4]),
            sum(value(m.S[1, comp]) for comp in m.comp),
        )
    )
    print(
        'S2_1:  {: > 1.0f}  S2_2: {: >2.0f}  S2_3: {: >2.0f}  S2_4: {: >2.0f}  S2total: {: >2.0f}'.format(
            value(m.S[2, 1]),
            value(m.S[2, 2]),
            value(m.S[2, 3]),
            value(m.S[2, 4]),
            sum(value(m.S[2, comp]) for comp in m.comp),
        )
    )
    print(' ')
    print('Distributors:')
    print('dl[2]: {: >3.4f} dl[3]: {: >3.4f}'.format(value(m.dl[2]), value(m.dl[3])))
    print('dv[2]: {: >3.4f} dv[3]: {: >3.4f}'.format(value(m.dv[2]), value(m.dv[3])))
    print(' ')
    print(' ')
    print(' ')
    print('               Kaibel Column Sections     ')
    print('__________________________________________')
    print(' ')
    print(' Tray       Bottom                Feed    ')
    print('__________________________________________')
    for t in reversed(list(m.tray)):
        print(
            '[{: >2.0f}] {: >9.0g} {: >18.0g}   F:{: >3.0f} '.format(
                t,
                (
                    fabs(value(m.tray_exists[1, t].indicator_var))
                    if t in m.candidate_trays_main
                    else 1
                ),
                (
                    fabs(value(m.tray_exists[2, t].indicator_var))
                    if t in m.candidate_trays_feed
                    else 1
                ),
                sum(value(m.F[comp]) for comp in m.comp) if t == m.feed_tray else 0,
            )
        )
    print(' ')
    print('__________________________________________')
    print(' ')
    print('            Product                 Top   ')
    print('__________________________________________')
    for t in reversed(list(m.tray)):
        print(
            '[{: >2.0f}] {: >9.0g}   S1:{: >2.0f}   S2:{: >2.0f} {: >8.0g}'.format(
                t,
                (
                    fabs(value(m.tray_exists[3, t].indicator_var))
                    if t in m.candidate_trays_product
                    else 1
                ),
                (
                    sum(value(m.S[1, comp]) for comp in m.comp)
                    if t == m.sideout1_tray
                    else 0
                ),
                (
                    sum(value(m.S[2, comp]) for comp in m.comp)
                    if t == m.sideout2_tray
                    else 0
                ),
                (
                    fabs(value(m.tray_exists[4, t].indicator_var))
                    if t in m.candidate_trays_main
                    else 1
                ),
            )
        )
    print(' 1 = trays exists, 0 = absent tray')


if __name__ == "__main__":
    main()
