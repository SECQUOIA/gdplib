"""Initialization routine for distillation column"""
from __future__ import division

import pandas
from math import fabs, floor
from pyomo.environ import value, exp


def initialize(m):
    m.reflux_frac.set_value(value(
        m.reflux_ratio / (1 + m.reflux_ratio)))
    m.boilup_frac.set_value(value(
        m.reboil_ratio / (1 + m.reboil_ratio)))

    _excel_sheets = pandas.read_excel('init.xlsx', sheet_name=None)

    def set_value_if_not_fixed(var, val):
        """Set variable to the value if it is not fixed."""
        if not var.fixed:
            var.set_value(val)

    active_trays = [
        t for t in m.trays
        if t not in m.conditional_trays or
        fabs(value(m.tray[t].indicator_var - 1)) <= 1E-3]
    num_active_trays = len(active_trays)

    feed_tray = m.feed_tray

    tray_indexed_data = _excel_sheets['trays']
    tray_indexed_data.sort_values(by=['tray'], inplace=True)
    tray_indexed_data.set_index('tray', inplace=True)

    comp_and_tray_indexed_data = _excel_sheets['comps_and_trays']
    comp_and_tray_indexed_data.sort_values(by=['comp', 'tray'],
                                           inplace=True)
    comp_and_tray_indexed_data.set_index(['comp', 'tray'], inplace=True)
    comp_slices = {c: comp_and_tray_indexed_data.loc[c, :]
                   for c in m.comps}

    num_data_trays = tray_indexed_data.index.size
    if num_active_trays < num_data_trays:
        # Number of model trays is less than number of trays in data. Need to
        # do averaging
        new_indices = [1] + [
            1 + (num_data_trays - 1) / (num_active_trays - 1) * i
            for i in range(1, num_active_trays)]
        for tray in range(2, num_active_trays):
            indx = new_indices[tray - 1]
            lower = floor(indx)
            frac_above = indx - lower
            # Take linear combination of values
            tray_indexed_data.loc[tray] = (
                tray_indexed_data.loc[lower] * (1 - frac_above) +
                tray_indexed_data.loc[lower + 1] * frac_above)
            for c in m.comps:
                comp_slices[c].loc[tray] = (
                    comp_slices[c].loc[lower] * (1 - frac_above) +
                    comp_slices[c].loc[lower + 1] * frac_above)
        tray_indexed_data.loc[num_active_trays] = \
            tray_indexed_data.loc[num_data_trays]
        tray_indexed_data = tray_indexed_data.head(num_active_trays)
        for c in m.comps:
            comp_slices[c].loc[num_active_trays] = \
                comp_slices[c].loc[num_data_trays]
            comp_slices[c] = comp_slices[c].head(num_active_trays)
    else:
        # Stretch the data out and do interpolation
        tray_indexed_data.index = pandas.Index(
            [1] + [int(round(num_active_trays / num_data_trays * i))
                   for i in range(2, num_data_trays + 1)], name='tray')
        tray_indexed_data = tray_indexed_data.reindex(
            [i for i in range(1, num_active_trays + 1)]).interpolate()
        for c in m.comps:
            comp_slices[c].index = pandas.Index(
                [1] + [int(round(num_active_trays / num_data_trays * i))
                       for i in range(2, num_data_trays + 1)], name='tray')
            # special handling necessary for V near top of column and L
            # near column bottom. Do not want to interpolate with one end
            # being potentially 0. (ie. V from total condenser). Instead,
            # use back fill and forward fill.
            comp_slices[c] = comp_slices[c].reindex(
                [i for i in range(1, num_active_trays + 1)])
            tray_below_condenser = sorted(active_trays, reverse=True)[1]
            if pandas.isna(comp_slices[c]['V'][tray_below_condenser]):
                # V of the tray below the condenser is N/A. Find a valid
                # value lower down to use.
                val = next(
                    comp_slices[c]['V'][t]
                    for t in reversed(list(m.trays))
                    if pandas.notna(comp_slices[c]['V'][t])
                    and not t == m.condens_tray)
                comp_slices[c]['V'][tray_below_condenser] = val
            if pandas.isna(comp_slices[c]['L'][m.reboil_tray + 1]):
                # L of the tray above the reboiler is N/A. Find a valid
                # value higher up to use.
                val = next(
                    comp_slices[c]['L'][t]
                    for t in m.trays
                    if pandas.notna(comp_slices[c]['L'][t])
                    and not t == m.reboil_tray)
                comp_slices[c]['L'][m.reboil_tray + 1] = val
            comp_slices[c] = comp_slices[c].interpolate()

    tray_indexed_data.index = pandas.Index(sorted(active_trays),
                                           name='tray')
    tray_indexed_data = tray_indexed_data.reindex(sorted(m.trays),
                                                  method='bfill')

    for t in m.trays:
        set_value_if_not_fixed(m.T[t], tray_indexed_data['T [K]'][t])

    for c in m.comps:
        comp_slices[c].index = pandas.Index(sorted(active_trays),
                                            name='tray')
        comp_slices[c] = comp_slices[c].reindex(sorted(m.trays))
        comp_slices[c][['L', 'x']] = comp_slices[c][['L', 'x']].bfill()
        comp_slices[c][['V', 'y']] = comp_slices[c][['V', 'y']].ffill()

    comp_and_tray_indexed_data = pandas.concat(comp_slices)

    for c, t in m.comps * m.trays:
        set_value_if_not_fixed(m.L[c, t],
                               comp_and_tray_indexed_data['L'][c, t])
        set_value_if_not_fixed(m.V[c, t],
                               comp_and_tray_indexed_data['V'][c, t])
        set_value_if_not_fixed(m.x[c, t],
                               comp_and_tray_indexed_data['x'][c, t])
        set_value_if_not_fixed(m.y[c, t],
                               comp_and_tray_indexed_data['y'][c, t])

    for c in m.comps:
        m.H_L_spec_feed[c].set_value(value(m.feed_liq_enthalpy_expr[c]))
        m.H_V_spec_feed[c].set_value(value(m.feed_vap_enthalpy_expr[c]))

    for t in m.trays:
        for c in m.comps:
            k = m.pvap_const[c]
            x = m.Pvap_X[c, t]

            x.set_value(value(1 - m.T[t] / k['Tc']))

            m.Pvap[c, t].set_value(value(exp((
                k['A'] * x +
                k['B'] * x ** 1.5 +
                k['C'] * x ** 3 +
                k['D'] * x ** 6) / (1 - x)) * k['Pc']))

            m.Kc[c, t].set_value(value(
                m.gamma[c, t] * m.Pvap[c, t] / m.P))

            m.H_L[c, t].set_value(value(m.liq_enthalpy_expr[t, c]))
            m.H_V[c, t].set_value(value(m.vap_enthalpy_expr[t, c]))

    m.D['benzene'].set_value(42.3152714)
    m.D['toluene'].set_value(5.4446286)
    m.B['benzene'].set_value(7.67928)
    m.B['toluene'].set_value(44.56072)
    m.L['benzene', m.reboil_tray].set_value(7.67928)
    m.L['toluene', m.reboil_tray].set_value(44.56072)
    m.V['benzene', m.reboil_tray].set_value(value(
        m.L['benzene', m.reboil_tray + 1] -
        m.L['benzene', m.reboil_tray]))
    m.V['toluene', m.reboil_tray].set_value(value(
        m.L['toluene', m.reboil_tray + 1] -
        m.L['toluene', m.reboil_tray]))
    m.L['benzene', m.condens_tray].set_value(value(
        m.V['benzene', m.condens_tray - 1] -
        m.D['benzene']))
    m.L['toluene', m.condens_tray].set_value(value(
        m.V['toluene', m.condens_tray - 1] -
        m.D['toluene']))

    for t in m.trays:
        m.liq[t].set_value(value(sum(m.L[c, t] for c in m.comps)))
        m.vap[t].set_value(value(sum(m.V[c, t] for c in m.comps)))
    m.bot.set_value(52.24)
    m.dis.set_value(47.7599)
    for c in m.comps:
        m.x[c, m.reboil_tray].set_value(value(
            m.L[c, m.reboil_tray] / m.liq[m.reboil_tray]))
        m.y[c, m.reboil_tray].set_value(value(
            m.V[c, m.reboil_tray] / m.vap[m.reboil_tray]))
        m.x[c, m.condens_tray].set_value(value(
            m.L[c, m.condens_tray] / m.liq[m.condens_tray]))
        m.y[c, m.condens_tray].set_value(value(
            m.x[c, m.condens_tray] * m.Kc[c, m.condens_tray]))
    m.Qb.set_value(2.307873115)
    m.Qc.set_value(3.62641882)
